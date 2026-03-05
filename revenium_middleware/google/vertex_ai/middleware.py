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

import datetime
import logging
from typing import Dict, Any, Optional, List, Tuple

import wrapt
from revenium_middleware import run_async_in_thread

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
    create_video_metering_call,
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

    # Clean up model name - remove Google's path prefixes
    if model_name and isinstance(model_name, str):
        # Remove common Google path prefixes
        prefixes_to_remove = [
            "publishers/google/models/",
            "models/",
            "google/models/",
            "projects/",
        ]
        for prefix in prefixes_to_remove:
            if model_name.startswith(prefix):
                model_name = model_name[len(prefix) :]
                logger.debug(
                    f"Cleaned model name, removed prefix '{prefix}': {model_name}"
                )
                break

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
        cache_creation_token_count=token_counts.cached_tokens,
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


# Dynamic wrapper discovery and application for Vertex AI GenerativeModel.generate_content
def _apply_generate_content_wrappers():
    """
    Dynamically discover and wrap all Vertex AI GenerativeModel.generate_content methods.
    This handles current and future module path variations like:
    - vertexai.generative_models.GenerativeModel
    - vertexai.preview.generative_models.GenerativeModel
    - vertexai.v1.generative_models.GenerativeModel
    - etc.
    """
    import sys
    import importlib

    # Known module patterns to try
    module_patterns = [
        "vertexai.generative_models",
        "vertexai.preview.generative_models",
        "vertexai.v1.generative_models",
        "vertexai.v1beta1.generative_models",
        "vertexai.v2.generative_models",
        "vertexai.beta.generative_models",
        "vertexai.alpha.generative_models",
    ]

    wrapped_modules = []

    for module_path in module_patterns:
        try:
            # Try to import the module
            module = importlib.import_module(module_path)

            # Check if GenerativeModel class exists
            if hasattr(module, "GenerativeModel"):
                generative_model_class = getattr(module, "GenerativeModel")

                # Check if generate_content method exists
                if hasattr(generative_model_class, "generate_content"):
                    logger.debug(
                        f"Found GenerativeModel.generate_content in {module_path}"
                    )

                    # Apply wrapper using wrapt
                    @wrapt.patch_function_wrapper(
                        module_path, "GenerativeModel.generate_content"
                    )
                    def generate_content_wrapper_dynamic(
                        wrapped, instance, args, kwargs
                    ):
                        return generate_content_wrapper_impl(
                            wrapped, instance, args, kwargs
                        )

                    wrapped_modules.append(module_path)
                    logger.debug(
                        f" Applied wrapper to {module_path}.GenerativeModel.generate_content"
                    )
                else:
                    logger.debug(
                        f"  {module_path}.GenerativeModel exists but no generate_content method"
                    )
            else:
                logger.debug(f"  {module_path} exists but no GenerativeModel class")

        except ImportError:
            logger.debug(f"  Module {module_path} not available")
        except Exception as e:
            logger.debug(f"  Error checking {module_path}: {e}")

    if wrapped_modules:
        logger.info(
            f" Vertex AI GenerativeModel wrappers applied to: {', '.join(wrapped_modules)}"
        )
    else:
        logger.warning("  No Vertex AI GenerativeModel modules found to wrap")

    return wrapped_modules


def generate_content_wrapper_impl(wrapped, instance, args, kwargs):
    """Enhanced wrapper that handles both streaming and non-streaming Vertex AI calls."""
    logger.debug("Enhanced Vertex AI generate_content wrapper called!")
    logger.debug(f"Wrapper args: {args}")
    logger.debug(f"Wrapper kwargs: {kwargs}")
    logger.debug(f"Instance type: {type(instance)}")

    # Extract usage metadata from instance or kwargs
    usage_metadata = getattr(instance, "_revenium_usage_metadata", {}) or kwargs.pop(
        "usage_metadata", {}
    )
    logger.debug(f"Captured usage metadata for generate_content: {usage_metadata}")
    logger.debug(
        f"Instance has _revenium_usage_metadata: {hasattr(instance, '_revenium_usage_metadata')}"
    )
    if hasattr(instance, "_revenium_usage_metadata"):
        logger.debug(
            f"Instance._revenium_usage_metadata value: {getattr(instance, '_revenium_usage_metadata')}"
        )

    # Try to extract model name from the instance
    model_name_from_instance = None
    for attr in [
        "_model_name",
        "model_name",
        "_model_id",
        "model_id",
        "_model",
        "model",
    ]:
        if hasattr(instance, attr):
            model_name_from_instance = getattr(instance, attr)
            logger.debug(
                f"Found model name in instance.{attr}: {model_name_from_instance}"
            )
            break

    # Clean up the instance model name too
    if model_name_from_instance and isinstance(model_name_from_instance, str):
        # Remove common Google path prefixes
        prefixes_to_remove = [
            "publishers/google/models/",
            "models/",
            "google/models/",
            "projects/",
        ]
        for prefix in prefixes_to_remove:
            if model_name_from_instance.startswith(prefix):
                model_name_from_instance = model_name_from_instance[len(prefix) :]
                logger.debug(
                    f"Cleaned instance model name, removed prefix '{prefix}': {model_name_from_instance}"
                )
                break

    if not model_name_from_instance:
        logger.debug(
            f"Could not find model name in instance. Available attributes: {dir(instance)}"
        )
        # Try to get it from the instance string representation
        instance_str = str(instance)
        if "model_name=" in instance_str:
            # Extract from string like "GenerativeModel(model_name='gemini-2.0-flash-lite-001')"
            import re

            match = re.search(r"model_name='([^']+)'", instance_str)
            if match:
                model_name_from_instance = match.group(1)
                logger.debug(
                    f"Extracted model name from instance string: {model_name_from_instance}"
                )
        elif "models/" in instance_str:
            # Extract from string like "models/gemini-2.0-flash-lite-001"
            import re

            match = re.search(r"models/([^'\s)]+)", instance_str)
            if match:
                model_name_from_instance = match.group(1)
                logger.debug(
                    f"Extracted model name from instance string (models/): {model_name_from_instance}"
                )

    # Detect vision content in the request
    # Vertex AI generate_content takes contents as first positional arg or 'contents' kwarg
    contents = kwargs.get("contents") or (args[0] if args else None)
    has_vision = detect_vision_content(contents)
    if has_vision:
        usage_metadata["has_vision_content"] = True
        logger.debug("Vision content detected in Vertex AI generate_content request")

    # Check if this is a streaming call
    is_streaming = kwargs.get("stream", False)

    # Store kwargs and args for prompt extraction
    request_kwargs = kwargs.copy()
    request_args = args

    # Record request time
    request_time_dt = datetime.datetime.now(datetime.timezone.utc)
    logger.debug(
        f"Calling wrapped Vertex AI generate_content function (streaming={is_streaming}) with args: {args}, kwargs: {kwargs}"
    )

    # Call the original Vertex AI function
    response = wrapped(*args, **kwargs)

    if is_streaming:
        logger.debug("Handling Vertex AI streaming response")
        # Return wrapped stream that will meter usage when complete
        return handle_vertex_ai_streaming_response(
            stream=response,
            request_time_dt=request_time_dt,
            usage_metadata=usage_metadata,
            model_name_fallback=model_name_from_instance,
            request_kwargs=request_kwargs,
            request_args=request_args,
        )
    else:
        logger.debug("Handling Vertex AI non-streaming response: %s", response)

        # Extract prompt data if capture is enabled
        system_prompt, input_messages, output_response, prompts_truncated = (
            extract_prompt_data_if_enabled(request_kwargs, args=request_args, response=response)
        )

        # Handle non-streaming response immediately
        create_vertex_ai_metering_call(
            response=response,
            operation_type=OperationType.CHAT,
            request_time_dt=request_time_dt,
            usage_metadata=usage_metadata,
            model_name_fallback=model_name_from_instance,
            # Prompt capture fields
            system_prompt=system_prompt,
            input_messages=input_messages,
            output_response=output_response,
            prompts_truncated=prompts_truncated,
        )
        return response


# Wrapper for Vertex AI TextEmbeddingModel.get_embeddings method
@wrapt.patch_function_wrapper(
    "vertexai.language_models", "TextEmbeddingModel.get_embeddings"
)
def get_embeddings_wrapper(wrapped, instance, args, kwargs):
    """Wraps the vertexai.language_models.TextEmbeddingModel.get_embeddings method to log token usage."""
    logger.debug("Vertex AI get_embeddings wrapper called")

    # Extract usage metadata from instance or kwargs
    usage_metadata = getattr(instance, "_revenium_usage_metadata", {}) or kwargs.pop(
        "usage_metadata", {}
    )

    # Try to extract model name from the instance using the same logic as generate_content
    model_name_from_instance = None
    for attr in [
        "_model_name",
        "model_name",
        "_model_id",
        "model_id",
        "_model",
        "model",
    ]:
        if hasattr(instance, attr):
            model_name_from_instance = getattr(instance, attr)
            logger.debug(
                f"Found model name in embeddings instance.{attr}: {model_name_from_instance}"
            )
            break

    if not model_name_from_instance:
        logger.debug(
            f"Could not find model name in embeddings instance. Available attributes: {dir(instance)}"
        )
        # Try to get it from the instance string representation
        instance_str = str(instance)
        if "model_name=" in instance_str:
            # Extract from string like "TextEmbeddingModel(model_name='text-embedding-004')"
            import re

            match = re.search(r"model_name='([^']+)'", instance_str)
            if match:
                model_name_from_instance = match.group(1)
                logger.debug(
                    f"Extracted model name from embeddings instance string: {model_name_from_instance}"
                )
        elif "models/" in instance_str:
            # Extract from string like "models/text-embedding-004"
            import re

            match = re.search(r"models/([^'\s)]+)", instance_str)
            if match:
                model_name_from_instance = match.group(1)
                logger.debug(
                    f"Extracted model name from embeddings instance string (models/): {model_name_from_instance}"
                )

    logger.debug(
        f"Final captured model name from Vertex AI embeddings instance: {model_name_from_instance}"
    )

    # Record request time
    request_time_dt = datetime.datetime.now(datetime.timezone.utc)
    logger.debug(
        f"Calling wrapped Vertex AI get_embeddings function with args: {args}, kwargs: {kwargs}"
    )

    # Call the original Vertex AI function
    response = wrapped(*args, **kwargs)

    logger.debug("Handling Vertex AI get_embeddings response: %s", response)

    # Create metering call for embeddings
    create_vertex_ai_metering_call(
        response=response,
        operation_type=OperationType.EMBED,
        request_time_dt=request_time_dt,
        usage_metadata=usage_metadata,
        model_name_fallback=model_name_from_instance,
    )

    return response


def handle_vertex_ai_streaming_response(
    stream, request_time_dt, usage_metadata, model_name_fallback=None, request_kwargs=None, request_args=None
):
    """
    Handle streaming responses from Vertex AI.
    Wraps the stream to collect metrics and log them after completion.
    """

    class VertexAIStreamWrapper:
        def __init__(self, stream):
            self.stream = stream
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
                    request_time_dt=request_time_dt,
                    usage_metadata=usage_metadata,
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

    return VertexAIStreamWrapper(stream)


# --- Vertex AI ImageGenerationModel wrapper (Imagen) ---

def _apply_imagen_wrappers():
    """
    Dynamically discover and wrap Vertex AI ImageGenerationModel.generate_images.
    Handles multiple module paths for forward compatibility.
    """
    import importlib

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
                    @wrapt.patch_function_wrapper(
                        module_path, "ImageGenerationModel.generate_images"
                    )
                    def generate_images_wrapper_dynamic(wrapped, instance, args, kwargs):
                        return _imagen_generate_images_impl(wrapped, instance, args, kwargs)

                    wrapped_modules.append(module_path + ".ImageGenerationModel.generate_images")
                    logger.debug(f" Applied Imagen wrapper to {module_path}.ImageGenerationModel.generate_images")

                if hasattr(img_model_class, "edit_image"):
                    @wrapt.patch_function_wrapper(
                        module_path, "ImageGenerationModel.edit_image"
                    )
                    def edit_image_wrapper_dynamic(wrapped, instance, args, kwargs):
                        return _imagen_edit_image_impl(wrapped, instance, args, kwargs)

                    wrapped_modules.append(module_path + ".ImageGenerationModel.edit_image")
                    logger.debug(f" Applied Imagen wrapper to {module_path}.ImageGenerationModel.edit_image")

        except ImportError:
            logger.debug(f"  Module {module_path} not available for Imagen")
        except Exception as e:
            logger.debug(f"  Error applying Imagen wrapper to {module_path}: {e}")

    if wrapped_modules:
        logger.info(f" Vertex AI Imagen wrappers applied to: {', '.join(wrapped_modules)}")

    return wrapped_modules


def _extract_model_name_from_instance(instance) -> Optional[str]:
    """Extract and clean model name from a Vertex AI model instance."""
    model_name = None
    for attr in ["_model_id", "model_id", "_model_name", "model_name", "_model", "model"]:
        if hasattr(instance, attr):
            model_name = getattr(instance, attr)
            if model_name:
                break

    if model_name and isinstance(model_name, str):
        prefixes = ["publishers/google/models/", "models/", "google/models/", "projects/"]
        for prefix in prefixes:
            if model_name.startswith(prefix):
                model_name = model_name[len(prefix):]
                break

    return model_name


def _imagen_generate_images_impl(wrapped, instance, args, kwargs):
    """Wrapper implementation for Vertex AI ImageGenerationModel.generate_images."""
    logger.debug("Vertex AI ImageGenerationModel.generate_images wrapper called")

    usage_metadata = getattr(instance, "_revenium_usage_metadata", {}) or kwargs.pop(
        "usage_metadata", {}
    )

    model_name = _extract_model_name_from_instance(instance) or "imagen-3.0-generate-001"

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
    """Wrapper implementation for Vertex AI ImageGenerationModel.edit_image."""
    logger.debug("Vertex AI ImageGenerationModel.edit_image wrapper called")

    usage_metadata = getattr(instance, "_revenium_usage_metadata", {}) or kwargs.pop(
        "usage_metadata", {}
    )

    model_name = _extract_model_name_from_instance(instance) or "imagen-3.0-generate-001"

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


# --- Vertex AI Veo video generation wrapper ---

def _apply_veo_wrappers():
    """
    Dynamically discover and wrap Vertex AI video generation models.
    Handles multiple module paths for forward compatibility.
    """
    import importlib

    module_patterns = [
        "vertexai.preview.vision_models",
        "vertexai.vision_models",
    ]

    wrapped_modules = []

    for module_path in module_patterns:
        try:
            module = importlib.import_module(module_path)

            # Check for VideoGenerationModel
            if hasattr(module, "VideoGenerationModel"):
                video_model_class = getattr(module, "VideoGenerationModel")

                if hasattr(video_model_class, "generate_content"):
                    @wrapt.patch_function_wrapper(
                        module_path, "VideoGenerationModel.generate_content"
                    )
                    def veo_generate_wrapper(wrapped, instance, args, kwargs):
                        return _veo_generate_impl(wrapped, instance, args, kwargs)

                    wrapped_modules.append(module_path + ".VideoGenerationModel.generate_content")

                if hasattr(video_model_class, "generate"):
                    @wrapt.patch_function_wrapper(
                        module_path, "VideoGenerationModel.generate"
                    )
                    def veo_generate_alt_wrapper(wrapped, instance, args, kwargs):
                        return _veo_generate_impl(wrapped, instance, args, kwargs)

                    wrapped_modules.append(module_path + ".VideoGenerationModel.generate")

        except ImportError:
            logger.debug(f"  Module {module_path} not available for Veo")
        except Exception as e:
            logger.debug(f"  Error applying Veo wrapper to {module_path}: {e}")

    if wrapped_modules:
        logger.info(f" Vertex AI Veo wrappers applied to: {', '.join(wrapped_modules)}")

    return wrapped_modules


def _veo_generate_impl(wrapped, instance, args, kwargs):
    """Wrapper implementation for Vertex AI VideoGenerationModel.generate_content."""
    logger.debug("Vertex AI VideoGenerationModel wrapper called (Veo)")

    usage_metadata = getattr(instance, "_revenium_usage_metadata", {}) or kwargs.pop(
        "usage_metadata", {}
    )

    model_name = _extract_model_name_from_instance(instance) or "veo-2.0-generate-001"

    # Extract video generation params
    duration = kwargs.get("duration", 5)  # Default 5 seconds for Veo
    aspect_ratio = kwargs.get("aspect_ratio")

    request_time_dt = datetime.datetime.now(datetime.timezone.utc)

    # Call original
    response = wrapped(*args, **kwargs)

    response_time_dt = datetime.datetime.now(datetime.timezone.utc)

    # Extract video duration from response if available
    video_duration = float(duration)
    if hasattr(response, "duration_seconds"):
        video_duration = float(response.duration_seconds)
    elif hasattr(response, "duration"):
        video_duration = float(response.duration)

    # Extract resolution if available
    resolution = None
    if hasattr(response, "resolution"):
        resolution = str(response.resolution)

    # Extract video job ID for async operations
    video_job_id = None
    if hasattr(response, "operation_name"):
        video_job_id = str(response.operation_name)
    elif hasattr(response, "name"):
        video_job_id = str(response.name)

    logger.debug(
        f"Vertex AI Veo generation: model={model_name}, "
        f"duration={video_duration}s, aspect_ratio={aspect_ratio}"
    )

    # Auto-detect async operation when a job ID is present
    async_operation = video_job_id is not None

    try:
        create_video_metering_call(
            model=model_name,
            duration_seconds=video_duration,
            request_time_dt=request_time_dt,
            response_time_dt=response_time_dt,
            usage_metadata=usage_metadata,
            operation_subtype="generation",
            resolution=resolution,
            aspect_ratio=aspect_ratio,
            video_job_id=video_job_id,
            async_operation=async_operation,
        )
    except Exception as e:
        logger.error(f"Error in Vertex AI Veo metering: {e}")

    return response


# Apply the dynamic wrappers when this module is imported
try:
    _apply_generate_content_wrappers()
except Exception as e:
    logger.error(f"Failed to apply dynamic Vertex AI wrappers: {e}")

try:
    _apply_imagen_wrappers()
except Exception as e:
    logger.error(f"Failed to apply Vertex AI Imagen wrappers: {e}")

try:
    _apply_veo_wrappers()
except Exception as e:
    logger.error(f"Failed to apply Vertex AI Veo wrappers: {e}")
