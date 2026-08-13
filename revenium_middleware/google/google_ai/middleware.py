"""
Google AI SDK middleware for Revenium.

This module provides middleware for the Google AI SDK (google-genai package),
supporting both Gemini Developer API and Vertex AI endpoints through the
unified google-genai interface.
"""

import datetime
import logging
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


if register_patch("google.genai.models.Models.generate_content"):
    @wrapt.patch_function_wrapper("google.genai.models", "Models.generate_content")
    @handle_metering_error
    def generate_content_wrapper(wrapped, instance, args, kwargs):
        if is_selective_metering_enabled() and not is_inside_decorated_function():
            return wrapped(*args, **kwargs)

        logger.debug("Google AI generate_content wrapper called")

        usage_metadata = kwargs.pop("usage_metadata", {})
        config = kwargs.get("config")

        request_time_dt = datetime.datetime.now(datetime.timezone.utc)
        logger.debug(
            "Calling wrapped generate_content function with args: %s, kwargs: %s",
            sanitize_for_logging(args), sanitize_for_logging(kwargs)
        )

        response = wrapped(*args, **kwargs)

        logger.debug("Handling generate_content response: %s", response)

        contents = kwargs.get("contents") or (args[1] if args and len(args) > 1 else None)
        has_vision = detect_vision_content(contents)
        if has_vision:
            usage_metadata["has_vision_content"] = True
            logger.debug("Vision content detected in generate_content request")

        system_prompt, input_messages, output_response, prompts_truncated = (
            extract_prompt_data_if_enabled(kwargs, args=args, config=config, response=response)
        )

        logger.debug("About to call create_google_ai_metering_call")
        logger.debug(
            f"create_google_ai_metering_call function exists: {create_google_ai_metering_call}"
        )
        try:
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
            logger.debug("create_google_ai_metering_call completed")
        except Exception as e:
            logger.error(f"Error in create_google_ai_metering_call: {e}")
            import traceback

            logger.error(f"Traceback: {traceback.format_exc()}")

        return response


if register_patch("google.genai.models.Models.embed_content"):
    @wrapt.patch_function_wrapper("google.genai.models", "Models.embed_content")
    @handle_metering_error
    def embed_content_wrapper(wrapped, instance, args, kwargs):
        if is_selective_metering_enabled() and not is_inside_decorated_function():
            return wrapped(*args, **kwargs)

        logger.debug("Google AI embed_content wrapper called")

        usage_metadata = kwargs.pop("usage_metadata", {})

        model_name_from_call = None
        if args and len(args) > 0:
            model_name_from_call = args[0]
        elif "model" in kwargs:
            model_name_from_call = kwargs["model"]

        logger.debug(
            f"Captured model name from embeddings API call: {model_name_from_call}"
        )

        request_time_dt = datetime.datetime.now(datetime.timezone.utc)
        logger.debug(
            "Calling wrapped embed_content function with args: %s, kwargs: %s",
            sanitize_for_logging(args), sanitize_for_logging(kwargs)
        )

        response = wrapped(*args, **kwargs)

        logger.debug("Handling embed_content response: %s", response)

        create_google_ai_metering_call(
            response=response,
            operation_type=OperationType.EMBED,
            request_time_dt=request_time_dt,
            usage_metadata=usage_metadata,
            client_instance=getattr(instance, "_api_client", None),
            model_name_fallback=model_name_from_call,
        )

        return response


if register_patch("google.genai.models.Models.generate_content_stream"):
    @wrapt.patch_function_wrapper("google.genai.models", "Models.generate_content_stream")
    @handle_metering_error
    def generate_content_stream_wrapper(wrapped, instance, args, kwargs):
        if is_selective_metering_enabled() and not is_inside_decorated_function():
            return wrapped(*args, **kwargs)

        logger.debug("Google AI generate_content_stream wrapper called")

        usage_metadata = kwargs.pop("usage_metadata", {})
        config = kwargs.get("config")

        contents = kwargs.get("contents") or (args[1] if args and len(args) > 1 else None)
        has_vision = detect_vision_content(contents)
        if has_vision:
            usage_metadata["has_vision_content"] = True
            logger.debug("Vision content detected in generate_content_stream request")

        request_kwargs = kwargs.copy()
        request_args = args

        request_time_dt = datetime.datetime.now(datetime.timezone.utc)
        logger.debug(
            "Calling wrapped generate_content_stream function with args: %s, kwargs: %s",
            sanitize_for_logging(args), sanitize_for_logging(kwargs)
        )

        stream = wrapped(*args, **kwargs)

        logger.debug("Handling generate_content_stream response")

        return handle_streaming_response(
            stream=stream,
            request_time_dt=request_time_dt,
            usage_metadata=usage_metadata,
            client_instance=getattr(instance, "_api_client", None),
            request_kwargs=request_kwargs,
            request_args=request_args,
            config=config,
        )


def handle_streaming_response(
    stream, request_time_dt, usage_metadata, client_instance=None, request_kwargs=None, request_args=None, config=None
):
    """
    Handle streaming responses from Google AI.
    Wraps the stream to collect metrics and log them after completion.
    """

    class StreamWrapper:
        def __init__(self, stream):
            self.stream = stream
            self.chunks = []
            self.accumulated_text = []  # For prompt capture
            self.model = None
            self.finish_reason = None
            self.usage_metadata = None
            self.client_instance = client_instance
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
                        logger.error("Error logging usage during stream cleanup: %s", e)

                # Clear chunks to free memory
                self.chunks.clear()

                # Close underlying stream if it has a close method
                if hasattr(self.stream, "close"):
                    try:
                        self.stream.close()
                    except Exception as e:
                        logger.debug("Error closing underlying stream: %s", e)

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
            logger.error("Error in streaming response: %s", error)
            if not self._usage_logged:
                # Try to log partial usage data
                try:
                    self._log_usage()
                    self._usage_logged = True
                except Exception as log_error:
                    logger.error(
                        "Failed to log usage after stream error: %s", log_error
                    )

        def _process_chunk(self, chunk):
            """Process each chunk to extract metadata"""
            # Limit chunk storage to prevent memory issues
            if len(self.chunks) < self._max_chunks:
                self.chunks.append(chunk)
            elif len(self.chunks) == self._max_chunks:
                logger.warning(
                    "Reached maximum chunk limit (%d), not storing additional chunks",
                    self._max_chunks,
                )

            # Record time of first chunk
            if self.first_chunk_time is None:
                self.first_chunk_time = datetime.datetime.now(datetime.timezone.utc)

            # Extract model name from chunk using safe access
            if self.model is None:
                self.model = safe_getattr(chunk, "model_version")

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

            # Check for finish reason and usage metadata in the chunk
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
                    logger.warning("No chunks received in streaming response")
                    return

                # Calculate time to first token
                time_to_first_token = 0
                if self.first_chunk_time:
                    time_to_first_token = int(
                        (self.first_chunk_time - request_time_dt).total_seconds() * 1000
                    )

                # Extract prompt data if capture is enabled
                accumulated_content = ''.join(self.accumulated_text) if self.accumulated_text else None
                # Append truncation marker if streaming was truncated
                if self.streaming_truncated and accumulated_content:
                    accumulated_content += "...[TRUNCATED]"

                system_prompt, input_messages, output_response, prompts_truncated = (
                    extract_prompt_data_if_enabled(
                        request_kwargs or {},
                        args=request_args,
                        config=config,
                        accumulated_content=accumulated_content
                    )
                )

                # Update truncation flag if streaming was truncated
                if self.streaming_truncated:
                    prompts_truncated = True

                # Create a synthetic response object for usage extraction
                class SyntheticResponse:
                    def __init__(self, model_version, usage_metadata, candidates):
                        self.model_version = model_version
                        self.usage_metadata = usage_metadata
                        self.candidates = candidates

                # Create synthetic response from collected data
                synthetic_response = SyntheticResponse(
                    model_version=self.model,
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
                create_google_ai_metering_call(
                    response=synthetic_response,
                    operation_type=OperationType.CHAT,
                    request_time_dt=request_time_dt,
                    usage_metadata=usage_metadata,
                    client_instance=self.client_instance,
                    time_to_first_token=time_to_first_token,
                    is_streamed=True,
                    # Prompt capture fields
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

    return StreamWrapper(stream)


if register_patch("google.genai.models.Models.generate_images"):
    @wrapt.patch_function_wrapper("google.genai.models", "Models.generate_images")
    @handle_metering_error
    def generate_images_wrapper(wrapped, instance, args, kwargs):
        if is_selective_metering_enabled() and not is_inside_decorated_function():
            return wrapped(*args, **kwargs)

        logger.debug("Google AI generate_images wrapper called (Imagen)")

        usage_metadata = kwargs.pop("usage_metadata", {})

        model_name = None
        if args and len(args) > 0:
            model_name = args[0]
        elif "model" in kwargs:
            model_name = kwargs["model"]

        config = kwargs.get("config", {})
        number_of_images = 1
        aspect_ratio = None

        if config:
            if hasattr(config, "number_of_images"):
                number_of_images = config.number_of_images or 1
            elif isinstance(config, dict):
                number_of_images = config.get("number_of_images", 1)

            if hasattr(config, "aspect_ratio"):
                aspect_ratio = config.aspect_ratio
            elif isinstance(config, dict):
                aspect_ratio = config.get("aspect_ratio")

        logger.debug(
            f"Imagen request: model={model_name}, count={number_of_images}, aspect_ratio={aspect_ratio}"
        )

        request_time_dt = datetime.datetime.now(datetime.timezone.utc)

        response = wrapped(*args, **kwargs)

        response_time_dt = datetime.datetime.now(datetime.timezone.utc)

        actual_image_count = 0
        if hasattr(response, "generated_images") and response.generated_images:
            actual_image_count = len(response.generated_images)
        elif hasattr(response, "images") and response.images:
            actual_image_count = len(response.images)

        logger.debug(
            f"Imagen response: requested={number_of_images}, actual={actual_image_count}"
        )

        try:
            create_image_metering_call(
                model=model_name or "imagen-3.0-generate-001",
                requested_image_count=number_of_images,
                actual_image_count=actual_image_count,
                request_time_dt=request_time_dt,
                response_time_dt=response_time_dt,
                usage_metadata=usage_metadata,
                operation_subtype="generation",
                aspect_ratio=aspect_ratio,
            )
            logger.debug("Image metering call completed for Imagen")
        except Exception as e:
            logger.error(f"Error in image metering call: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")

        return response
