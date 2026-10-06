"""
Google AI SDK middleware for Revenium.

This module provides middleware for the Google AI SDK (google-genai package),
supporting both Gemini Developer API and Vertex AI endpoints through the
unified google-genai interface.
"""

import datetime
import functools
import logging
from types import SimpleNamespace
from typing import Dict, Any, Optional, Tuple

import wrapt
from revenium_middleware import run_async_in_thread
from revenium_middleware._core.log_sanitize import sanitize_for_logging
from revenium_middleware._core.config import is_selective_metering_enabled
from revenium_middleware._core.context import is_inside_decorated_function
from revenium_middleware._core.patch_registry import register_patch

# Import common utilities and types
from ..common import (
    OperationType,
    UsageData,
    TokenCounts,
    normalize_stop_reason,
    Provider,
    create_metering_call,
    create_image_metering_call,
    extract_model_name,
    extract_token_counts,
    handle_metering_error,
    safe_getattr,
)
from ..common.trace_fields import detect_vision_content

# Google AI specific imports
from .provider import get_provider_metadata
from ..prompt_extractor import extract_prompt_data_if_enabled

logger = logging.getLogger("revenium_middleware.extension")


def extract_google_ai_usage_data(
    response: Any,
    operation_type: OperationType,
    request_time: datetime.datetime,
    response_time: datetime.datetime,
    client_instance: Optional[Any] = None,
    model_name_fallback: Optional[str] = None,
) -> UsageData:
    """
    Extract usage data from Google AI API responses.

    This function handles the specific quirks of the Google AI SDK,
    particularly the missing token counts for embeddings.
    """
    # Get provider metadata for Google AI SDK
    provider_metadata = get_provider_metadata()

    # Extract model name
    model_name = extract_model_name(response, model_name_fallback)

    # Extract token counts with Google AI specific handling
    if operation_type == OperationType.EMBED:
        # CRITICAL: Google AI SDK limitation - embeddings responses don't include token usage
        # The Vertex AI REST API has statistics.token_count, but Google AI SDK doesn't expose it
        token_counts = TokenCounts(
            input_tokens=0, output_tokens=0, total_tokens=0, cached_tokens=0
        )
        stop_reason = "END"  # Embeddings always complete successfully
        logger.debug(
            "Google AI SDK limitation: embeddings responses don't include token usage data"
        )
    else:  # CHAT
        # Extract usage metadata from Google AI response
        token_counts = TokenCounts(
            input_tokens=0, output_tokens=0, total_tokens=0, cached_tokens=0
        )

        if hasattr(response, "usage_metadata") and response.usage_metadata:
            usage_metadata = response.usage_metadata
            token_counts.input_tokens = getattr(usage_metadata, "prompt_token_count", 0)
            # Google AI uses 'candidates_token_count' not 'response_token_count'
            token_counts.output_tokens = getattr(
                usage_metadata, "candidates_token_count", 0
            )
            token_counts.total_tokens = getattr(
                usage_metadata,
                "total_token_count",
                token_counts.input_tokens + token_counts.output_tokens,
            )
            token_counts.cached_tokens = getattr(
                usage_metadata, "cached_content_token_count", 0
            )

            logger.debug(
                f"Chat token usage: prompt={token_counts.input_tokens}, "
                f"candidates={token_counts.output_tokens}, total={token_counts.total_tokens}"
            )
        else:
            logger.warning("No usage metadata found in Google AI chat response")

        # Determine finish reason from candidates
        google_finish_reason = None
        if hasattr(response, "candidates") and response.candidates:
            candidate = response.candidates[0]
            if hasattr(candidate, "finish_reason"):
                google_finish_reason = candidate.finish_reason

        stop_reason = normalize_stop_reason(
            google_finish_reason, Provider.GOOGLE_AI_SDK
        )

    # Create standardized UsageData
    return UsageData.create(
        operation_type=operation_type,
        input_tokens=token_counts.input_tokens,
        output_tokens=token_counts.output_tokens,
        total_tokens=token_counts.total_tokens,
        model=model_name,
        provider_metadata=provider_metadata,
        stop_reason=stop_reason,
        request_time=request_time,
        response_time=response_time,
        cache_read_token_count=token_counts.cached_tokens,
    )


def create_google_ai_metering_call(
    response: Any,
    operation_type: OperationType,
    request_time_dt: datetime.datetime,
    usage_metadata: Dict[str, Any],
    client_instance: Optional[Any] = None,
    time_to_first_token: int = 0,
    is_streamed: bool = False,
    model_name_fallback: Optional[str] = None,
    # Prompt capture fields
    system_prompt: Optional[str] = None,
    input_messages: Optional[str] = None,
    output_response: Optional[str] = None,
    prompts_truncated: Optional[bool] = None,
) -> None:
    """
    Create and execute a metering call for Google AI SDK responses.

    This is the main function used by the wrapper functions.
    """
    logger.debug("create_google_ai_metering_call started")

    # Record response timing
    response_time_dt = datetime.datetime.now(datetime.timezone.utc)

    # Extract usage data using Google AI specific logic
    logger.debug("Extracting usage data...")
    usage_data = extract_google_ai_usage_data(
        response=response,
        operation_type=operation_type,
        request_time=request_time_dt,
        response_time=response_time_dt,
        client_instance=client_instance,
        model_name_fallback=model_name_fallback,
    )
    logger.debug(f"Usage data extracted: {usage_data}")

    # Create metering call using common utilities
    logger.debug("About to call create_metering_call from common utilities")
    try:
        create_metering_call(
            usage_data=usage_data,
            usage_metadata=usage_metadata,
            time_to_first_token=time_to_first_token,
            is_streamed=is_streamed,
            # Prompt capture fields
            system_prompt=system_prompt,
            input_messages=input_messages,
            output_response=output_response,
            prompts_truncated=prompts_truncated,
        )
        logger.debug("create_metering_call completed successfully")
    except Exception as e:
        logger.error(f"Error in create_metering_call: {e}")
        import traceback

        logger.error(f"Traceback: {traceback.format_exc()}")


DEFAULT_IMAGEN_MODEL = "imagen-3.0-generate-001"


def _metering_skipped():
    return is_selective_metering_enabled() and not is_inside_decorated_function()


def _now():
    return datetime.datetime.now(datetime.timezone.utc)


def _request_contents(args, kwargs):
    return kwargs.get("contents") or (args[1] if args and len(args) > 1 else None)


def _request_model(args, kwargs):
    if args:
        return args[0]
    return kwargs.get("model")


def _flag_vision_content(args, kwargs, usage_metadata):
    if detect_vision_content(_request_contents(args, kwargs)):
        usage_metadata["has_vision_content"] = True
        logger.debug("Vision content detected in the generate_content request")


def _take_usage_metadata(kwargs):
    # A Revenium-only kwarg the SDK's signature rejects, so it has to go even when the call is not metered.
    return kwargs.pop("usage_metadata", {})


_reported_failures = set()


def _log_metering_failure(operation, error):
    """Metering failures never reach the caller; each kind is logged once at WARNING, then at DEBUG."""
    kind = (operation, type(error))
    level = logging.DEBUG if kind in _reported_failures else logging.WARNING
    _reported_failures.add(kind)
    logger.log(level, "Google AI %s metering failed: %s", operation, error, exc_info=True)


def _metering_requested(operation, args, kwargs):
    """False when the call must go through unmetered: selective metering, or our own setup failing."""
    try:
        if _metering_skipped():
            return False
        logger.debug(
            "Calling Google AI %s with args: %s, kwargs: %s",
            operation, sanitize_for_logging(args), sanitize_for_logging(kwargs)
        )
        return True
    except Exception as e:
        _log_metering_failure(operation, e)
        return False


def _meter_generate_content(response, instance, args, kwargs, usage_metadata, request_time_dt):
    try:
        _flag_vision_content(args, kwargs, usage_metadata)
        system_prompt, input_messages, output_response, prompts_truncated = (
            extract_prompt_data_if_enabled(kwargs, args=args, config=kwargs.get("config"), response=response)
        )
        create_google_ai_metering_call(
            response=response,
            operation_type=OperationType.CHAT,
            request_time_dt=request_time_dt,
            usage_metadata=usage_metadata,
            client_instance=getattr(instance, "_api_client", None),
            system_prompt=system_prompt,
            input_messages=input_messages,
            output_response=output_response,
            prompts_truncated=prompts_truncated,
        )
    except Exception as e:
        _log_metering_failure("generate_content", e)


def _meter_embed_content(response, instance, args, kwargs, usage_metadata, request_time_dt):
    try:
        create_google_ai_metering_call(
            response=response,
            operation_type=OperationType.EMBED,
            request_time_dt=request_time_dt,
            usage_metadata=usage_metadata,
            client_instance=getattr(instance, "_api_client", None),
            model_name_fallback=_request_model(args, kwargs),
        )
    except Exception as e:
        _log_metering_failure("embed_content", e)


def _config_value(config, name, default):
    if isinstance(config, dict):
        return config.get(name, default)
    return getattr(config, name, default)


def _generated_image_count(response):
    images = safe_getattr(response, "generated_images") or safe_getattr(response, "images")
    return len(images) if images else 0


def _meter_generate_images(response, instance, args, kwargs, usage_metadata, request_time_dt):
    try:
        config = kwargs.get("config") or {}
        create_image_metering_call(
            model=_request_model(args, kwargs) or DEFAULT_IMAGEN_MODEL,
            requested_image_count=_config_value(config, "number_of_images", 1) or 1,
            actual_image_count=_generated_image_count(response),
            request_time_dt=request_time_dt,
            response_time_dt=_now(),
            usage_metadata=usage_metadata,
            operation_subtype="generation",
            aspect_ratio=_config_value(config, "aspect_ratio", None),
        )
    except Exception as e:
        _log_metering_failure("generate_images", e)


def _sync_metered(operation, meter):
    """A wrapt wrapper that runs the call, then hands its response to ``meter``."""
    @handle_metering_error
    def wrapper(wrapped, instance, args, kwargs):
        usage_metadata = _take_usage_metadata(kwargs)
        if not _metering_requested(operation, args, kwargs):
            return wrapped(*args, **kwargs)
        request_time_dt = _now()
        response = wrapped(*args, **kwargs)
        meter(response, instance, args, kwargs, usage_metadata, request_time_dt)
        return response
    return wrapper


def _async_metered(operation, meter):
    """The coroutine twin of ``_sync_metered``: meters once the awaited call returns."""
    def wrapper(wrapped, instance, args, kwargs):
        usage_metadata = _take_usage_metadata(kwargs)

        async def call():
            # Decided when awaited: a coroutine created outside a @revenium_meter scope may be awaited inside it.
            if not _metering_requested(operation, args, kwargs):
                return await wrapped(*args, **kwargs)
            request_time_dt = _now()
            response = await wrapped(*args, **kwargs)
            meter(response, instance, args, kwargs, usage_metadata, request_time_dt)
            return response
        return call()
    return wrapper


generate_content_wrapper = _sync_metered("generate_content", _meter_generate_content)
embed_content_wrapper = _sync_metered("embed_content", _meter_embed_content)
generate_images_wrapper = _sync_metered("generate_images", _meter_generate_images)
async_generate_content_wrapper = _async_metered("generate_content", _meter_generate_content)
async_embed_content_wrapper = _async_metered("embed_content", _meter_embed_content)
async_generate_images_wrapper = _async_metered("generate_images", _meter_generate_images)


class _StreamMetering:
    """Collects a Gemini stream's chunks and sends its one metering record when the stream ends."""

    _max_chunks = 1000

    def __init__(self, stream, request_time_dt, usage_metadata, client_instance=None,
                 request_kwargs=None, request_args=None, config=None):
        self.stream = stream
        self.request_time_dt = request_time_dt
        self.request_usage_metadata = usage_metadata
        self.client_instance = client_instance
        self.request_kwargs = request_kwargs
        self.request_args = request_args
        self.config = config
        self.chunks = []
        self.accumulated_text = []
        self.model = None
        self.finish_reason = None
        self.usage_metadata = None
        self.first_chunk_time = None
        self.streaming_truncated = False
        self._closed = False
        self._usage_logged = False

    def _finalize(self):
        if not self._usage_logged:
            self._usage_logged = True
            self._log_usage()

    def _handle_error(self, error):
        logger.error("Error in streaming response: %s", error)
        try:
            self._finalize()
        except Exception as log_error:
            logger.error("Failed to log usage after stream error: %s", log_error)

    def _mark_closed(self):
        """Meter a stream abandoned before its end; False when it was already closed."""
        if self._closed:
            return False
        self._closed = True
        try:
            self._finalize()
        except Exception as e:
            logger.error("Error logging usage during stream cleanup: %s", e)
        self.chunks.clear()
        return True

    def __del__(self):
        # Last-resort cleanup: a broken-out-of loop leaves the wrapper to
        # the GC with no StopIteration/__exit__ ever firing.
        try:
            self._release_on_collect()
        except Exception:
            pass

    def _release_on_collect(self):
        self._mark_closed()

    def _process_chunk(self, chunk):
        if len(self.chunks) < self._max_chunks:
            self.chunks.append(chunk)
        elif len(self.chunks) == self._max_chunks:
            logger.warning(
                "Reached maximum chunk limit (%d), not storing additional chunks",
                self._max_chunks,
            )

        if self.first_chunk_time is None:
            self.first_chunk_time = _now()

        if self.model is None:
            self.model = safe_getattr(chunk, "model_version")

        self._accumulate_text(chunk)

        candidates = safe_getattr(chunk, "candidates")
        if candidates and len(candidates) > 0:
            finish_reason = safe_getattr(candidates[0], "finish_reason")
            if finish_reason:
                self.finish_reason = finish_reason

        usage_metadata = safe_getattr(chunk, "usage_metadata")
        if usage_metadata:
            self.usage_metadata = usage_metadata

    def _accumulate_text(self, chunk):
        from ..config import Config
        current_len = sum(len(t) for t in self.accumulated_text)

        if hasattr(chunk, 'text') and chunk.text:
            chunk_len = len(chunk.text)
            if current_len + chunk_len <= Config.MAX_PROMPT_LENGTH:
                self.accumulated_text.append(chunk.text)
            elif current_len < Config.MAX_PROMPT_LENGTH:
                remaining = Config.MAX_PROMPT_LENGTH - current_len
                self.accumulated_text.append(chunk.text[:remaining])
                self.streaming_truncated = True
            else:
                self.streaming_truncated = True
        elif hasattr(chunk, 'candidates') and chunk.candidates:
            for candidate in chunk.candidates:
                if hasattr(candidate, 'content') and candidate.content:
                    if hasattr(candidate.content, 'parts'):
                        for part in candidate.content.parts:
                            if hasattr(part, 'text') and part.text:
                                part_len = len(part.text)
                                if current_len + part_len <= Config.MAX_PROMPT_LENGTH:
                                    self.accumulated_text.append(part.text)
                                    current_len += part_len
                                elif current_len < Config.MAX_PROMPT_LENGTH:
                                    remaining = Config.MAX_PROMPT_LENGTH - current_len
                                    self.accumulated_text.append(part.text[:remaining])
                                    current_len = Config.MAX_PROMPT_LENGTH
                                    self.streaming_truncated = True
                                    break
                                else:
                                    self.streaming_truncated = True
                                    break

    def _log_usage(self):
        try:
            if not self.chunks:
                logger.warning("No chunks received in streaming response")
                return

            time_to_first_token = 0
            if self.first_chunk_time:
                time_to_first_token = int(
                    (self.first_chunk_time - self.request_time_dt).total_seconds() * 1000
                )

            accumulated_content = ''.join(self.accumulated_text) if self.accumulated_text else None
            if self.streaming_truncated and accumulated_content:
                accumulated_content += "...[TRUNCATED]"

            system_prompt, input_messages, output_response, prompts_truncated = (
                extract_prompt_data_if_enabled(
                    self.request_kwargs or {},
                    args=self.request_args,
                    config=self.config,
                    accumulated_content=accumulated_content
                )
            )

            if self.streaming_truncated:
                prompts_truncated = True

            create_google_ai_metering_call(
                response=self._synthetic_response(),
                operation_type=OperationType.CHAT,
                request_time_dt=self.request_time_dt,
                usage_metadata=self.request_usage_metadata,
                client_instance=self.client_instance,
                time_to_first_token=time_to_first_token,
                is_streamed=True,
                system_prompt=system_prompt,
                input_messages=input_messages,
                output_response=output_response,
                prompts_truncated=prompts_truncated,
            )

            logger.debug(
                "Streaming usage logged: model=%s, chunks=%d, time_to_first_token=%dms",
                self.model,
                len(self.chunks),
                time_to_first_token,
            )

        except Exception as e:
            logger.error("Metering error: %s", e)

    def _synthetic_response(self):
        candidates = [SimpleNamespace(finish_reason=self.finish_reason)] if self.finish_reason else []
        return SimpleNamespace(
            model_version=self.model, usage_metadata=self.usage_metadata, candidates=candidates
        )


class StreamWrapper(_StreamMetering):
    def __iter__(self):
        return self

    def __next__(self):
        if self._closed:
            raise StopIteration("Stream has been closed")
        try:
            chunk = next(self.stream)
            self._process_chunk(chunk)
            return chunk
        except StopIteration:
            self._finalize()
            raise
        except Exception as e:
            self._handle_error(e)
            raise

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def _release_on_collect(self):
        self.close()

    def close(self):
        if self._mark_closed() and hasattr(self.stream, "close"):
            try:
                self.stream.close()
            except Exception as e:
                logger.debug("Error closing underlying stream: %s", e)


class AsyncStreamWrapper(_StreamMetering):
    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._closed:
            raise StopAsyncIteration
        try:
            chunk = await self.stream.__anext__()
            self._process_chunk(chunk)
            return chunk
        except StopAsyncIteration:
            self._finalize()
            raise
        except Exception as e:
            self._handle_error(e)
            raise

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.aclose()
        return False

    async def aclose(self):
        if self._mark_closed() and hasattr(self.stream, "aclose"):
            try:
                await self.stream.aclose()
            except Exception as e:
                logger.debug("Error closing underlying async stream: %s", e)


def handle_streaming_response(
    stream, request_time_dt, usage_metadata, client_instance=None, request_kwargs=None, request_args=None, config=None
):
    return StreamWrapper(stream, request_time_dt, usage_metadata, client_instance,
                         request_kwargs, request_args, config)


def _stream_wrapper_for(wrapper_class, instance, args, kwargs, usage_metadata):
    """A callable that wraps the provider's stream for metering, or None when the stream goes unmetered."""
    if not _metering_requested("generate_content_stream", args, kwargs):
        return None
    try:
        _flag_vision_content(args, kwargs, usage_metadata)
        return functools.partial(
            wrapper_class,
            request_time_dt=_now(),
            usage_metadata=usage_metadata,
            client_instance=getattr(instance, "_api_client", None),
            request_kwargs=kwargs.copy(),
            request_args=args,
            config=kwargs.get("config"),
        )
    except Exception as e:
        _log_metering_failure("generate_content_stream", e)
        return None


@handle_metering_error
def generate_content_stream_wrapper(wrapped, instance, args, kwargs):
    usage_metadata = _take_usage_metadata(kwargs)
    wrap_stream = _stream_wrapper_for(StreamWrapper, instance, args, kwargs, usage_metadata)
    if wrap_stream is None:
        return wrapped(*args, **kwargs)
    return wrap_stream(wrapped(*args, **kwargs))


def async_generate_content_stream_wrapper(wrapped, instance, args, kwargs):
    usage_metadata = _take_usage_metadata(kwargs)

    async def call():
        # Decided when awaited, for the same reason as in _async_metered.
        wrap_stream = _stream_wrapper_for(AsyncStreamWrapper, instance, args, kwargs, usage_metadata)
        if wrap_stream is None:
            return await wrapped(*args, **kwargs)
        return wrap_stream(await wrapped(*args, **kwargs))
    return call()


WRAPPERS = {
    "Models.generate_content": generate_content_wrapper,
    "Models.embed_content": embed_content_wrapper,
    "Models.generate_content_stream": generate_content_stream_wrapper,
    "Models.generate_images": generate_images_wrapper,
    "AsyncModels.generate_content": async_generate_content_wrapper,
    "AsyncModels.embed_content": async_embed_content_wrapper,
    "AsyncModels.generate_content_stream": async_generate_content_stream_wrapper,
    "AsyncModels.generate_images": async_generate_images_wrapper,
}

for _target, _wrapper in WRAPPERS.items():
    if register_patch(f"google.genai.models.{_target}"):
        wrapt.wrap_function_wrapper("google.genai.models", _target, _wrapper)
