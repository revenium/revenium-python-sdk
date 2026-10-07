"""
Vertex AI SDK middleware for Revenium.

This module provides middleware for the native Vertex AI SDK (vertexai package),
offering enhanced features like comprehensive token counting and local tokenization.

Key advantages over Google AI SDK:
- Full token counting support including embeddings
- Local tokenization capabilities
- Enhanced metadata and usage tracking
- Better integration with Google Cloud services
"""

import contextlib
import contextvars
import datetime
import importlib
import logging
import re
from typing import Dict, Any, Optional, List, Tuple

import wrapt
from revenium_middleware import run_async_in_thread
from revenium_middleware._core.config import is_selective_metering_enabled
from revenium_middleware._core.context import is_inside_decorated_function, overlay_metadata
from revenium_middleware._core.log_sanitize import sanitize_for_logging
from revenium_middleware._core.patch_registry import register_patch

# Import common utilities and types
from ..common import (
    OperationType,
    ProviderMetadata,
    UsageData,
    TokenCounts,
    normalize_stop_reason,
    Provider,
    create_metering_call,
    create_image_metering_call,
    create_usage_data,
    extract_model_name,
    extract_token_counts,
    StreamingError,
    handle_metering_error,
    safe_getattr,
)
from ..common.trace_fields import detect_vision_content

# Vertex AI specific imports
from .provider import detect_provider, get_provider_metadata
from ..prompt_extractor import extract_prompt_data_if_enabled

logger = logging.getLogger("revenium_middleware.extension")

_GOOGLE_MODEL_PATH_PREFIXES = (
    "publishers/google/models/",
    "models/",
    "google/models/",
    "projects/",
)
_MODEL_NAME_ATTRIBUTES = ("_model_name", "model_name", "_model_id", "model_id", "_model", "model")
_MODEL_NAME_IN_REPR = (re.compile(r"model_name='([^']+)'"), re.compile(r"models/([^'\s)]+)"))


def _strip_model_path(model_name: str) -> str:
    for prefix in _GOOGLE_MODEL_PATH_PREFIXES:
        if model_name.startswith(prefix):
            return model_name[len(prefix):]
    return model_name


def _model_name_of(instance: Any) -> Optional[str]:
    """The model a Vertex AI model object calls, without Google's resource-path prefix."""
    for attribute in _MODEL_NAME_ATTRIBUTES:
        value = getattr(instance, attribute, None)
        if isinstance(value, str) and value:
            return _strip_model_path(value)
    for pattern in _MODEL_NAME_IN_REPR:
        match = pattern.search(str(instance))
        if match:
            return match.group(1)
    return None


def extract_vertex_ai_usage_data(
    response: Any,
    operation_type: OperationType,
    request_time: datetime.datetime,
    response_time: datetime.datetime,
    model_name_fallback: Optional[str] = None,
) -> UsageData:
    """
    Extract usage data from Vertex AI API responses.

    This function handles the enhanced features of the Vertex AI SDK,
    particularly the comprehensive token counting for all operations.
    """
    # Get provider metadata for Vertex AI
    provider_metadata = ProviderMetadata.for_vertex_ai_sdk()

    # Extract model name - Vertex AI specific logic
    model_name = None

    # First try Vertex AI specific fields
    if hasattr(response, "_raw_response") and response._raw_response:
        raw_response = response._raw_response
        if hasattr(raw_response, "model_version") and raw_response.model_version:
            model_name = raw_response.model_version
            logger.debug(
                f"Extracted model name from Vertex AI _raw_response.model_version: {model_name}"
            )

    # Fallback to common extraction if not found
    if not model_name:
        model_name = extract_model_name(response, model_name_fallback)

    # Use fallback if still not found
    if not model_name:
        model_name = model_name_fallback or "unknown-model"

    if isinstance(model_name, str):
        model_name = _strip_model_path(model_name)

    # Extract token counts with Vertex AI specific handling
    if operation_type == OperationType.EMBED:
        # Vertex AI SDK provides token counts for embeddings!
        token_counts = extract_vertex_ai_embedding_tokens(response)
        stop_reason = "END"  # Embeddings always complete successfully
        logger.debug(
            f"Vertex AI embeddings token usage: {token_counts.total_tokens} tokens"
        )
    else:  # CHAT
        # Extract usage metadata from Vertex AI response
        token_counts = extract_vertex_ai_generation_tokens(response)

        # Determine finish reason from candidates
        vertex_finish_reason = None
        if hasattr(response, "candidates") and response.candidates:
            candidate = response.candidates[0]
            if hasattr(candidate, "finish_reason"):
                vertex_finish_reason = candidate.finish_reason
                logger.debug(
                    f" Raw vertex_finish_reason: {vertex_finish_reason} (type: {type(vertex_finish_reason)})"
                )

                # Convert enum to string if needed
                if hasattr(vertex_finish_reason, "name"):
                    vertex_finish_reason = vertex_finish_reason.name
                    logger.debug(f" Converted enum to string: {vertex_finish_reason}")
                elif not isinstance(vertex_finish_reason, str):
                    vertex_finish_reason = str(vertex_finish_reason)
                    logger.debug(f" Converted to string: {vertex_finish_reason}")

        stop_reason = normalize_stop_reason(
            vertex_finish_reason, Provider.VERTEX_AI_SDK
        )
        logger.debug(f" Final stop_reason after normalization: {stop_reason}")
        logger.debug(
            f"Vertex AI chat token usage: prompt={token_counts.input_tokens}, "
            f"candidates={token_counts.output_tokens}, total={token_counts.total_tokens}"
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


def extract_vertex_ai_generation_tokens(response: Any) -> TokenCounts:
    """
    Extract token counts from Vertex AI generation responses.

    Vertex AI provides comprehensive token counting in the usage_metadata.
    """
    token_counts = TokenCounts(
        input_tokens=0, output_tokens=0, total_tokens=0, cached_tokens=0
    )

    if hasattr(response, "usage_metadata") and response.usage_metadata:
        usage_metadata = response.usage_metadata

        # Vertex AI uses different attribute names than Google AI SDK
        token_counts.input_tokens = getattr(usage_metadata, "prompt_token_count", 0)
        token_counts.output_tokens = getattr(
            usage_metadata, "candidates_token_count", 0
        )
        token_counts.total_tokens = getattr(
            usage_metadata,
            "total_token_count",
            token_counts.input_tokens + token_counts.output_tokens,
        )

        # Vertex AI may provide cached token counts
        token_counts.cached_tokens = getattr(
            usage_metadata, "cached_content_token_count", 0
        )
    else:
        logger.warning("No usage metadata found in Vertex AI generation response")

    return token_counts


def extract_vertex_ai_embedding_tokens(response: Any) -> TokenCounts:
    """
    Extract token counts from Vertex AI embedding responses.

    This is a key advantage of Vertex AI SDK - embeddings include token counts!
    """
    token_counts = TokenCounts(
        input_tokens=0, output_tokens=0, total_tokens=0, cached_tokens=0
    )

    # Vertex AI embeddings response is a list of TextEmbedding objects
    if isinstance(response, list) and len(response) > 0:
        # Get the first embedding object
        first_embedding = response[0]

        # Check if it has statistics with token_count
        if hasattr(first_embedding, "statistics") and first_embedding.statistics:
            stats = first_embedding.statistics
            if hasattr(stats, "token_count"):
                # Convert to int if it's a float
                token_count = (
                    int(stats.token_count)
                    if hasattr(stats.token_count, "__int__")
                    else stats.token_count
                )
                token_counts.input_tokens = token_count
                token_counts.total_tokens = token_count
                # Embeddings don't generate output tokens
                token_counts.output_tokens = 0
                logger.debug(
                    f"Extracted token count from Vertex AI embedding statistics: {token_count}"
                )
                return token_counts

        # Check if the embedding has _prediction_response with metadata
        if (
            hasattr(first_embedding, "_prediction_response")
            and first_embedding._prediction_response
        ):
            pred_response = first_embedding._prediction_response
            if hasattr(pred_response, "metadata") and pred_response.metadata:
                # Check for billableCharacterCount or other token-related fields
                metadata = pred_response.metadata
                if hasattr(metadata, "billableCharacterCount"):
                    # Use billable character count as a proxy for tokens
                    char_count = metadata.billableCharacterCount
                    # Rough approximation: 4 characters per token (common for many tokenizers)
                    estimated_tokens = max(1, int(char_count / 4))
                    token_counts.input_tokens = estimated_tokens
                    token_counts.total_tokens = estimated_tokens
                    token_counts.output_tokens = 0
                    logger.debug(
                        f"Estimated token count from billable characters: {char_count} chars -> {estimated_tokens} tokens"
                    )
                    return token_counts

    # Fallback: check if response itself has statistics or usage_metadata
    elif hasattr(response, "statistics") and response.statistics:
        # Some Vertex AI embedding responses have statistics
        stats = response.statistics
        if hasattr(stats, "token_count"):
            token_counts.input_tokens = stats.token_count
            token_counts.total_tokens = stats.token_count
            # Embeddings don't generate output tokens
            token_counts.output_tokens = 0
    elif hasattr(response, "usage_metadata") and response.usage_metadata:
        # Alternative location for token counts
        usage_metadata = response.usage_metadata
        token_counts.input_tokens = getattr(usage_metadata, "prompt_token_count", 0)
        token_counts.total_tokens = getattr(
            usage_metadata, "total_token_count", token_counts.input_tokens
        )
        token_counts.output_tokens = 0  # Embeddings don't generate output
    else:
        # If no token counts available, log warning
        logger.debug("No token counts found in Vertex AI embedding response")

    return token_counts


def create_vertex_ai_metering_call(
    response: Any,
    operation_type: OperationType,
    request_time_dt: datetime.datetime,
    usage_metadata: Dict[str, Any],
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
    Create and execute a metering call for Vertex AI SDK responses.

    This is the main function used by the wrapper functions.
    """
    # Record response timing
    response_time_dt = datetime.datetime.now(datetime.timezone.utc)

    # Extract usage data using Vertex AI specific logic
    usage_data = extract_vertex_ai_usage_data(
        response=response,
        operation_type=operation_type,
        request_time=request_time_dt,
        response_time=response_time_dt,
        model_name_fallback=model_name_fallback,
    )

    # Create metering call using common utilities
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


# ChatSession sends through the model's private _generate_content today. Should a
# vertexai release route it through a wrapped public method instead, the inner
# wrap sees this flag and lets the call through, so one request stays one record.
_inside_metered_call: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "revenium_vertex_inside_metered_call", default=False
)


@contextlib.contextmanager
def _metering_this_call():
    token = _inside_metered_call.set(True)
    try:
        yield
    finally:
        _inside_metered_call.reset(token)


def _passes_through() -> bool:
    if _inside_metered_call.get():
        return True
    return is_selective_metering_enabled() and not is_inside_decorated_function()


def _pop_usage_metadata(kwargs: Dict[str, Any], *owners: Any) -> Dict[str, Any]:
    """The owners' ``_revenium_usage_metadata``, least specific first, overlaid by the call's own."""
    merged: Dict[str, Any] = {}
    for owner in owners:
        from_owner = getattr(owner, "_revenium_usage_metadata", None)
        merged = overlay_metadata(merged, from_owner if isinstance(from_owner, dict) else None)
    return overlay_metadata(merged, kwargs.pop("usage_metadata", None))


def _as_turns(contents: Any) -> List[Any]:
    if contents is None:
        return []
    return list(contents) if isinstance(contents, (list, tuple)) else [contents]


def _first_argument(args: Tuple, kwargs: Dict[str, Any], name: str) -> Any:
    return kwargs.get(name) or (args[0] if args else None)


class _GenerationCall:
    """A generation request as metering needs it, captured before the SDK sees it."""

    def __init__(self, model: Any, contents: Any, args: Tuple, kwargs: Dict[str, Any],
                 usage_metadata: Dict[str, Any], history: Tuple[Any, ...] = ()):
        logger.debug(
            "Vertex AI generation request: args=%s kwargs=%s",
            sanitize_for_logging(args),
            sanitize_for_logging(kwargs),
        )
        self.usage_metadata = usage_metadata
        self.model_name = _model_name_of(model)
        if detect_vision_content(contents):
            self.usage_metadata["has_vision_content"] = True
        self.is_streaming = bool(kwargs.get("stream", False))
        prompt = [*history, *_as_turns(contents)] if history else contents
        self.request_kwargs = {**kwargs, "contents": prompt}
        self.request_args = args
        self.request_time = datetime.datetime.now(datetime.timezone.utc)

    def metered(self, response: Any) -> Any:
        if self.is_streaming:
            return VertexAIStreamWrapper(response, **self._stream_context())
        self._meter(response)
        return response

    def metered_async(self, response: Any) -> Any:
        if self.is_streaming:
            return VertexAIAsyncStreamWrapper(response, **self._stream_context())
        self._meter(response)
        return response

    def _stream_context(self) -> Dict[str, Any]:
        return {
            "request_time_dt": self.request_time,
            "usage_metadata": self.usage_metadata,
            "model_name_fallback": self.model_name,
            "request_kwargs": self.request_kwargs,
            "request_args": self.request_args,
        }

    def _meter(self, response: Any) -> None:
        system_prompt, input_messages, output_response, prompts_truncated = (
            extract_prompt_data_if_enabled(
                self.request_kwargs, args=self.request_args, response=response
            )
        )
        create_vertex_ai_metering_call(
            response=response,
            operation_type=OperationType.CHAT,
            request_time_dt=self.request_time,
            usage_metadata=self.usage_metadata,
            model_name_fallback=self.model_name,
            system_prompt=system_prompt,
            input_messages=input_messages,
            output_response=output_response,
            prompts_truncated=prompts_truncated,
        )


def _generate_content_call(instance, args, kwargs) -> _GenerationCall:
    usage_metadata = _pop_usage_metadata(kwargs, instance)
    contents = _first_argument(args, kwargs, "contents")
    return _GenerationCall(instance, contents, args, kwargs, usage_metadata)


def _send_message_call(instance, args, kwargs) -> _GenerationCall:
    model = getattr(instance, "_model", None)
    usage_metadata = _pop_usage_metadata(kwargs, model, instance)
    content = _first_argument(args, kwargs, "content")
    # Read before the call: the SDK appends this turn to the history only once the response is in.
    history = tuple(getattr(instance, "history", None) or ())
    return _GenerationCall(model, content, args, kwargs, usage_metadata, history)


def _call_and_meter(call, wrapped, args, kwargs):
    with _metering_this_call():
        response = wrapped(*args, **kwargs)
    return call.metered(response)


async def _await_and_meter(call, wrapped, args, kwargs):
    with _metering_this_call():
        response = await wrapped(*args, **kwargs)
    return call.metered_async(response)


def _sync_metered(build_call):
    """A wrapt wrapper that meters a sync Vertex AI method through the call object ``build_call`` returns."""
    def wrapper(wrapped, instance, args, kwargs):
        if _passes_through():
            return wrapped(*args, **kwargs)
        return _call_and_meter(build_call(instance, args, kwargs), wrapped, args, kwargs)
    return wrapper


def _async_metered(build_call):
    """The same for a method whose result is awaited."""
    def wrapper(wrapped, instance, args, kwargs):
        if _passes_through():
            return wrapped(*args, **kwargs)
        return _await_and_meter(build_call(instance, args, kwargs), wrapped, args, kwargs)
    return wrapper


generate_content_wrapper_impl = _sync_metered(_generate_content_call)
generate_content_async_wrapper_impl = _async_metered(_generate_content_call)
send_message_wrapper_impl = _sync_metered(_send_message_call)
send_message_async_wrapper_impl = _async_metered(_send_message_call)


_GENERATIVE_MODULE_PATHS = (
    "vertexai.generative_models",
    "vertexai.preview.generative_models",
    "vertexai.v1.generative_models",
    "vertexai.v1beta1.generative_models",
    "vertexai.v2.generative_models",
    "vertexai.beta.generative_models",
    "vertexai.alpha.generative_models",
)

_WRAPPERS_BY_CLASS = {
    "GenerativeModel": {
        "generate_content": generate_content_wrapper_impl,
        "generate_content_async": generate_content_async_wrapper_impl,
    },
    "ChatSession": {
        "send_message": send_message_wrapper_impl,
        "send_message_async": send_message_async_wrapper_impl,
    },
}


def _is_our_wrapper(attribute: Any) -> bool:
    return isinstance(attribute, wrapt.FunctionWrapper) and attribute._self_wrapper.__module__ == __name__


def _inherits_our_wrapper(cls: type, method_name: str) -> bool:
    defining_class = next(base for base in cls.__mro__ if method_name in vars(base))
    return defining_class is not cls and _is_our_wrapper(vars(defining_class)[method_name])


def _wrap_methods(module, class_name: str, wrappers: Dict[str, Any]) -> List[str]:
    cls = getattr(module, class_name, None)
    if cls is None:
        return []
    applied = []
    for method_name, wrapper in wrappers.items():
        # vertexai.preview.generative_models.ChatSession inherits send_message from
        # the ChatSession wrapped first; a second wrap would stack on the same method.
        if not hasattr(cls, method_name) or _inherits_our_wrapper(cls, method_name):
            continue
        patch_key = f"{module.__name__}.{class_name}.{method_name}"
        if register_patch(patch_key):
            wrapt.wrap_function_wrapper(module, f"{class_name}.{method_name}", wrapper)
            applied.append(patch_key)
    return applied


def _apply_generative_model_wrappers() -> List[str]:
    """Wrap GenerativeModel and ChatSession in every vertexai module path that defines them."""
    applied = []
    for module_path in _GENERATIVE_MODULE_PATHS:
        try:
            module = importlib.import_module(module_path)
        except ImportError:
            logger.debug("Module %s not available", module_path)
            continue
        for class_name, wrappers in _WRAPPERS_BY_CLASS.items():
            applied += _wrap_methods(module, class_name, wrappers)

    if applied:
        logger.info(" Vertex AI generation wrappers applied to: %s", ", ".join(applied))
    else:
        logger.warning("  No Vertex AI GenerativeModel modules found to wrap")
    return applied


class _EmbeddingCall:
    """An embedding request as metering needs it, captured before the SDK sees it."""

    def __init__(self, instance: Any, args: Tuple, kwargs: Dict[str, Any]):
        self.usage_metadata = _pop_usage_metadata(kwargs, instance)
        self.model_name = _model_name_of(instance)
        self.request_time = datetime.datetime.now(datetime.timezone.utc)

    def metered(self, response: Any) -> Any:
        create_vertex_ai_metering_call(
            response=response,
            operation_type=OperationType.EMBED,
            request_time_dt=self.request_time,
            usage_metadata=self.usage_metadata,
            model_name_fallback=self.model_name,
        )
        return response

    metered_async = metered


if register_patch("vertexai.language_models.TextEmbeddingModel.get_embeddings"):
    wrapt.wrap_function_wrapper(
        "vertexai.language_models", "TextEmbeddingModel.get_embeddings", _sync_metered(_EmbeddingCall)
    )

if register_patch("vertexai.language_models.TextEmbeddingModel.get_embeddings_async"):
    wrapt.wrap_function_wrapper(
        "vertexai.language_models", "TextEmbeddingModel.get_embeddings_async", _async_metered(_EmbeddingCall)
    )


class VertexAIStreamWrapper:
    """Passes a Vertex AI stream through and meters it once, when it ends or is closed."""

    def __init__(self, stream, request_time_dt, usage_metadata, model_name_fallback=None,
                 request_kwargs=None, request_args=None):
        self.stream = stream
        self._request_time_dt = request_time_dt
        self._metering_metadata = usage_metadata
        self._request_kwargs = request_kwargs
        self._request_args = request_args
        self.chunks = []
        self.accumulated_text = []  # For prompt capture
        self.model = model_name_fallback
        self.finish_reason = None
        self.usage_metadata = None
        self.first_chunk_time = None
        self._closed = False
        self._usage_logged = False
        self.streaming_truncated = False  # Track if streaming response was truncated

        # Limit chunk storage to prevent memory issues
        self._max_chunks = 1000

    def __iter__(self):
        return self

    def __next__(self):
        if self._closed:
            raise StopIteration("Stream has been closed")

        try:
            with _metering_this_call():
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
        return False  # Don't suppress exceptions

    def close(self):
        """Properly close the stream and clean up resources."""
        if not self._closed:
            self._closed = True
            if not self._usage_logged:
                try:
                    self._log_usage()
                except Exception as e:
                    logger.error(
                        "Error logging usage during Vertex AI stream cleanup: %s", e
                    )

            # Clear chunks to free memory
            self.chunks.clear()

            # Close underlying stream if it has a close method
            if hasattr(self.stream, "close"):
                try:
                    self.stream.close()
                except Exception as e:
                    logger.debug("Error closing underlying Vertex AI stream: %s", e)

    def _finalize(self):
        """Finalize the stream and log usage."""
        if not self._usage_logged:
            self._log_usage()
            self._usage_logged = True

    def __del__(self):
        # Last-resort cleanup: a broken-out-of loop leaves the wrapper to
        # the GC with no StopIteration/__exit__ ever firing.
        try:
            self.close()
        except Exception:
            pass

    def _handle_error(self, error: Exception):
        """Handle errors during streaming."""
        logger.error("Error in Vertex AI streaming response: %s", error)
        if not self._usage_logged:
            # Try to log partial usage data
            try:
                self._log_usage()
                self._usage_logged = True
            except Exception as log_error:
                logger.error(
                    "Failed to log Vertex AI usage after stream error: %s",
                    log_error,
                )

    def _process_chunk(self, chunk):
        """Process each chunk to extract metadata"""
        # Limit chunk storage to prevent memory issues
        if len(self.chunks) < self._max_chunks:
            self.chunks.append(chunk)
        elif len(self.chunks) == self._max_chunks:
            logger.warning(
                "Reached maximum chunk limit (%d) for Vertex AI stream, not storing additional chunks",
                self._max_chunks,
            )

        # Record time of first chunk
        if self.first_chunk_time is None:
            self.first_chunk_time = datetime.datetime.now(datetime.timezone.utc)

        # Extract model name from chunk if available using safe access
        if self.model is None:
            self.model = extract_model_name(chunk, self.model)

        # Accumulate text for prompt capture (with early truncation to prevent unbounded memory growth)
        from ..config import Config
        current_len = sum(len(t) for t in self.accumulated_text)

        if hasattr(chunk, 'text') and chunk.text:
            # Check if adding this chunk would exceed the limit
            chunk_len = len(chunk.text)
            if current_len + chunk_len <= Config.MAX_PROMPT_LENGTH:
                self.accumulated_text.append(chunk.text)
            elif current_len < Config.MAX_PROMPT_LENGTH:
                # Partial append: only add what fits
                remaining = Config.MAX_PROMPT_LENGTH - current_len
                self.accumulated_text.append(chunk.text[:remaining])
                self.streaming_truncated = True
            else:
                # Already at limit, mark as truncated
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
                                    # Partial append: only add what fits
                                    remaining = Config.MAX_PROMPT_LENGTH - current_len
                                    self.accumulated_text.append(part.text[:remaining])
                                    current_len = Config.MAX_PROMPT_LENGTH
                                    self.streaming_truncated = True
                                    break
                                else:
                                    # Already at limit
                                    self.streaming_truncated = True
                                    break

        # Check for finish reason and usage metadata in the chunk using safe access
        candidates = safe_getattr(chunk, "candidates")
        if candidates and len(candidates) > 0:
            candidate = candidates[0]
            finish_reason = safe_getattr(candidate, "finish_reason")
            if finish_reason:
                self.finish_reason = finish_reason

        # Check for usage metadata in the chunk (final chunk typically has this)
        usage_metadata = safe_getattr(chunk, "usage_metadata")
        if usage_metadata:
            self.usage_metadata = usage_metadata

    def _log_usage(self):
        """Log usage after stream completion"""
        try:
            if not self.chunks:
                logger.warning("No chunks received in Vertex AI streaming response")
                return

            # Calculate time to first token
            time_to_first_token = 0
            if self.first_chunk_time:
                time_to_first_token = int(
                    (self.first_chunk_time - self._request_time_dt).total_seconds() * 1000
                )

            # Extract prompt data if capture is enabled
            accumulated_content = ''.join(self.accumulated_text) if self.accumulated_text else None
            # Append truncation marker if streaming was truncated
            if self.streaming_truncated and accumulated_content:
                accumulated_content += "...[TRUNCATED]"

            system_prompt, input_messages, output_response, prompts_truncated = (
                extract_prompt_data_if_enabled(
                    self._request_kwargs or {},
                    args=self._request_args,
                    accumulated_content=accumulated_content
                )
            )

            # Update truncation flag if streaming was truncated
            if self.streaming_truncated:
                prompts_truncated = True

            # Create a synthetic response object for usage extraction
            class SyntheticResponse:
                def __init__(self, model_name, usage_metadata, candidates):
                    self.model_name = model_name
                    self.usage_metadata = usage_metadata
                    self.candidates = candidates

            # Create synthetic response from collected data
            synthetic_response = SyntheticResponse(
                model_name=self.model,
                usage_metadata=self.usage_metadata,
                candidates=(
                    [
                        type(
                            "obj", (object,), {"finish_reason": self.finish_reason}
                        )()
                    ]
                    if self.finish_reason
                    else []
                ),
            )

            # Create metering call for streaming response
            create_vertex_ai_metering_call(
                response=synthetic_response,
                operation_type=OperationType.CHAT,
                request_time_dt=self._request_time_dt,
                usage_metadata=self._metering_metadata,
                time_to_first_token=time_to_first_token,
                is_streamed=True,
                model_name_fallback=self.model,
                # Prompt capture fields
                system_prompt=system_prompt,
                input_messages=input_messages,
                output_response=output_response,
                prompts_truncated=prompts_truncated,
            )

            logger.debug(
                "Vertex AI streaming usage logged: model=%s, chunks=%d, time_to_first_token=%dms",
                self.model,
                len(self.chunks),
                time_to_first_token,
            )

        except Exception as e:
            # Don't let logging errors break the stream
            logger.error("Error logging Vertex AI streaming usage: %s", e)
            raise StreamingError(
                f"Failed to log Vertex AI streaming usage: {str(e)}",
                chunk_count=len(self.chunks) if self.chunks else 0,
                stream_state="completed",
            ) from e


class VertexAIAsyncStreamWrapper(VertexAIStreamWrapper):
    """The async-iterator form, for generate_content_async and send_message_async with stream=True."""

    def __iter__(self):
        raise TypeError("a Vertex AI async stream is consumed with async for")

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._closed:
            raise StopAsyncIteration
        try:
            with _metering_this_call():
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
        underlying_aclose = getattr(self.stream, "aclose", None)
        self.close()
        if underlying_aclose is not None:
            try:
                await underlying_aclose()
            except Exception as e:
                logger.debug("Error closing underlying Vertex AI async stream: %s", e)


def handle_vertex_ai_streaming_response(
    stream, request_time_dt, usage_metadata, model_name_fallback=None, request_kwargs=None, request_args=None
):
    """Wrap a sync Vertex AI stream so its usage is metered once it completes."""
    return VertexAIStreamWrapper(
        stream, request_time_dt, usage_metadata, model_name_fallback, request_kwargs, request_args
    )


# --- Vertex AI ImageGenerationModel wrapper (Imagen) ---

def _apply_imagen_wrappers():
    """
    Dynamically discover and wrap Vertex AI ImageGenerationModel.generate_images.
    Handles multiple module paths for forward compatibility.
    """
    module_patterns = [
        "vertexai.preview.vision_models",
        "vertexai.vision_models",
    ]

    wrapped_modules = []

    for module_path in module_patterns:
        try:
            module = importlib.import_module(module_path)

            if hasattr(module, "ImageGenerationModel"):
                img_model_class = getattr(module, "ImageGenerationModel")

                if hasattr(img_model_class, "generate_images"):
                    patch_key = f"{module_path}.ImageGenerationModel.generate_images"
                    if register_patch(patch_key):
                        @wrapt.patch_function_wrapper(
                            module_path, "ImageGenerationModel.generate_images"
                        )
                        def generate_images_wrapper_dynamic(wrapped, instance, args, kwargs):
                            return _imagen_generate_images_impl(wrapped, instance, args, kwargs)

                        wrapped_modules.append(patch_key)
                        logger.debug(f" Applied Imagen wrapper to {patch_key}")

                if hasattr(img_model_class, "edit_image"):
                    patch_key = f"{module_path}.ImageGenerationModel.edit_image"
                    if register_patch(patch_key):
                        @wrapt.patch_function_wrapper(
                            module_path, "ImageGenerationModel.edit_image"
                        )
                        def edit_image_wrapper_dynamic(wrapped, instance, args, kwargs):
                            return _imagen_edit_image_impl(wrapped, instance, args, kwargs)

                        wrapped_modules.append(patch_key)
                        logger.debug(f" Applied Imagen wrapper to {patch_key}")

        except ImportError:
            logger.debug(f"  Module {module_path} not available for Imagen")
        except Exception as e:
            logger.debug(f"  Error applying Imagen wrapper to {module_path}: {e}")

    if wrapped_modules:
        logger.info(f" Vertex AI Imagen wrappers applied to: {', '.join(wrapped_modules)}")

    return wrapped_modules


def _imagen_generate_images_impl(wrapped, instance, args, kwargs):
    if is_selective_metering_enabled() and not is_inside_decorated_function():
        return wrapped(*args, **kwargs)

    logger.debug("Vertex AI ImageGenerationModel.generate_images wrapper called")

    usage_metadata = getattr(instance, "_revenium_usage_metadata", {}) or kwargs.pop(
        "usage_metadata", {}
    )

    model_name = _model_name_of(instance) or "imagen-3.0-generate-001"

    # Extract image count from kwargs
    number_of_images = kwargs.get("number_of_images", 1)
    aspect_ratio = kwargs.get("aspect_ratio")

    request_time_dt = datetime.datetime.now(datetime.timezone.utc)

    # Call original
    response = wrapped(*args, **kwargs)

    response_time_dt = datetime.datetime.now(datetime.timezone.utc)

    # Count generated images
    actual_image_count = 0
    if hasattr(response, "images") and response.images:
        actual_image_count = len(response.images)
    elif isinstance(response, list):
        actual_image_count = len(response)

    logger.debug(
        f"Vertex AI Imagen generate_images: model={model_name}, "
        f"requested={number_of_images}, actual={actual_image_count}"
    )

    try:
        create_image_metering_call(
            model=model_name,
            requested_image_count=number_of_images,
            actual_image_count=actual_image_count,
            request_time_dt=request_time_dt,
            response_time_dt=response_time_dt,
            usage_metadata=usage_metadata,
            operation_subtype="generation",
            aspect_ratio=aspect_ratio,
        )
    except Exception as e:
        logger.error(f"Error in Vertex AI Imagen metering: {e}")

    return response


def _imagen_edit_image_impl(wrapped, instance, args, kwargs):
    if is_selective_metering_enabled() and not is_inside_decorated_function():
        return wrapped(*args, **kwargs)

    logger.debug("Vertex AI ImageGenerationModel.edit_image wrapper called")

    usage_metadata = getattr(instance, "_revenium_usage_metadata", {}) or kwargs.pop(
        "usage_metadata", {}
    )

    model_name = _model_name_of(instance) or "imagen-3.0-generate-001"

    number_of_images = kwargs.get("number_of_images", 1)

    request_time_dt = datetime.datetime.now(datetime.timezone.utc)

    response = wrapped(*args, **kwargs)

    response_time_dt = datetime.datetime.now(datetime.timezone.utc)

    actual_image_count = 0
    if hasattr(response, "images") and response.images:
        actual_image_count = len(response.images)

    try:
        create_image_metering_call(
            model=model_name,
            requested_image_count=number_of_images,
            actual_image_count=actual_image_count,
            request_time_dt=request_time_dt,
            response_time_dt=response_time_dt,
            usage_metadata=usage_metadata,
            operation_subtype="edit",
        )
    except Exception as e:
        logger.error(f"Error in Vertex AI Imagen edit metering: {e}")

    return response


# Apply the dynamic wrappers when this module is imported
try:
    _apply_generative_model_wrappers()
except Exception as e:
    logger.error(f"Failed to apply dynamic Vertex AI wrappers: {e}")

try:
    _apply_imagen_wrappers()
except Exception as e:
    logger.error(f"Failed to apply Vertex AI Imagen wrappers: {e}")
