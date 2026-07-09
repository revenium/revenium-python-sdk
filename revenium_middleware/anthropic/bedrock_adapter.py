# Copyright (c) 2025 Revenium, Inc.
# Licensed under the MIT License. See LICENSE file in the project root.

"""
AWS Bedrock adapter for Anthropic Claude models.

This module provides a clean interface for invoking Anthropic Claude models
through AWS Bedrock while maintaining compatibility with the direct Anthropic API.
"""

import json
import os
import logging
import time
import datetime
import uuid
import hashlib
import threading

from typing import Dict, Any, Optional, Tuple, Generator, Iterator, Union, List

logger = logging.getLogger("revenium_middleware.extension")


# Custom exceptions for better error handling
class BedrockError(Exception):
    """Base exception for Bedrock-related errors."""
    pass


class BedrockValidationError(BedrockError):
    """Exception raised for input validation errors."""
    pass


class BedrockInvokeError(BedrockError):
    """Exception raised for Bedrock invocation errors."""
    pass


class BedrockStreamError(BedrockError):
    """Exception raised for Bedrock streaming errors."""
    pass


# Input validation utilities
def _validate_messages(messages: Any) -> List[Dict[str, Any]]:
    """Validate messages parameter."""
    if not isinstance(messages, list):
        raise BedrockValidationError(f"messages must be a list, got {type(messages).__name__}")

    if not messages:
        raise BedrockValidationError("messages cannot be empty")

    for i, message in enumerate(messages):
        if not isinstance(message, dict):
            raise BedrockValidationError(f"message at index {i} must be a dict, got {type(message).__name__}")

        if "role" not in message:
            raise BedrockValidationError(f"message at index {i} missing required 'role' field")

        if "content" not in message:
            raise BedrockValidationError(f"message at index {i} missing required 'content' field")

    return messages


def _validate_max_tokens(max_tokens: Any) -> int:
    """Validate max_tokens parameter."""
    if not isinstance(max_tokens, int):
        try:
            max_tokens = int(max_tokens)
        except (ValueError, TypeError):
            raise BedrockValidationError(f"max_tokens must be an integer, got {type(max_tokens).__name__}")

    if max_tokens <= 0:
        raise BedrockValidationError(f"max_tokens must be positive, got {max_tokens}")

    if max_tokens > 200000:  # Reasonable upper limit
        raise BedrockValidationError(f"max_tokens too large, got {max_tokens} (max: 200000)")

    return max_tokens


def _validate_model_name(model: Any) -> str:
    """Validate model name parameter."""
    if not isinstance(model, str):
        raise BedrockValidationError(f"model must be a string, got {type(model).__name__}")

    if not model.strip():
        raise BedrockValidationError("model cannot be empty")

    return model.strip()


def _generate_safe_id(prefix: str = "msg_bedrock", content: str = "") -> str:
    """Generate a safe, deterministic ID using UUID and content hash."""
    # Create a deterministic hash of the content for reproducibility in tests
    if content:
        content_hash = hashlib.sha256(content.encode('utf-8')).hexdigest()[:8]
        return f"{prefix}_{content_hash}_{uuid.uuid4().hex[:8]}"
    else:
        return f"{prefix}_{uuid.uuid4().hex[:12]}"


# Thread-safe cache configuration
_CACHE_SIZE = int(os.getenv("REVENIUM_BEDROCK_CACHE_SIZE", "32"))
_cache_lock = threading.RLock()

# Simple model mapping from Anthropic to Bedrock model IDs
_MODEL_MAP = {
    "claude-opus-4-7": "anthropic.claude-opus-4-7",
    "us.claude-opus-4-7": "us.anthropic.claude-opus-4-7",
    "eu.claude-opus-4-7": "eu.anthropic.claude-opus-4-7",
    "au.claude-opus-4-7": "au.anthropic.claude-opus-4-7",
    "global.claude-opus-4-7": "global.anthropic.claude-opus-4-7",
    "claude-3-opus-20240229": "anthropic.claude-3-opus-20240229-v1:0",
    "claude-3-sonnet-20240229": "anthropic.claude-3-sonnet-20240229-v1:0",
    "claude-3-haiku-20240307": "us.anthropic.claude-3-5-haiku-20241022-v1:0",
    "claude-3-5-sonnet-20240620": "anthropic.claude-3-5-sonnet-20240620-v1:0",
    "claude-3-5-sonnet-20241022": "anthropic.claude-3-5-sonnet-20241022-v2:0",
    "claude-3-5-haiku-20241022": "anthropic.claude-3-5-haiku-20241022-v1:0",
}


def _import_boto3():
    """Import boto3 with helpful error message if not installed."""
    try:
        import boto3
        return boto3
    except ImportError:
        raise ImportError(
            "boto3 is required for Bedrock support. "
            "Install with: pip install revenium-python-sdk[anthropic]"
        )


# Thread-safe client cache
_client_cache: Dict[str, Any] = {}


def _resolve_region(region: Optional[str]) -> str:
    """Resolve the AWS region with the standard boto3/CLI precedence.

    Matches the trace-metadata resolver (_core/trace_fields.py), so the region
    a Bedrock call actually runs in agrees with the region recorded in the
    metering payload.
    """
    return region or os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION") or "us-east-1"


def get_bedrock_client(region: str):
    """Get a cached boto3 bedrock-runtime client for the specified region."""
    if not isinstance(region, str) or not region.strip():
        raise BedrockValidationError(f"region must be a non-empty string, got {type(region).__name__}")

    region = region.strip()

    with _cache_lock:
        if region in _client_cache:
            return _client_cache[region]

        # Limit cache size
        if len(_client_cache) >= _CACHE_SIZE:
            # Remove oldest entry (simple FIFO)
            oldest_key = next(iter(_client_cache))
            del _client_cache[oldest_key]
            logger.debug(f"Removed cached client for region {oldest_key} (cache full)")

        boto3 = _import_boto3()
        client = boto3.client("bedrock-runtime", region_name=region)
        _client_cache[region] = client
        logger.debug(f"Created new Bedrock client for region {region}")
        return client


def _model_id(model_name: str) -> str:
    """Map Anthropic model name to Bedrock model ID."""
    return _MODEL_MAP.get(model_name, f"anthropic.{model_name}")


def _as_dict(value: Any) -> dict:
    """Return value when it is a dict, otherwise an empty dict."""
    return value if isinstance(value, dict) else {}


def _token_count(value: Any) -> Optional[int]:
    """Return a non-negative integer token count, or None for invalid values."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value >= 0 else None
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return None


def _first_token_count(source: Any, keys: Tuple[str, ...]) -> Optional[int]:
    """Return the first valid token count for any of the supplied keys."""
    source = _as_dict(source)
    for key in keys:
        if key in source:
            value = _token_count(source.get(key))
            if value is not None:
                return value
    return None


def _extract_cache_tokens(usage: Any) -> Tuple[Optional[int], Optional[int]]:
    """Return (cache_creation_tokens, cache_read_tokens) from a Bedrock usage dict."""
    cache_creation = _first_token_count(
        usage,
        (
            "cacheWriteInputTokensCount",
            "cacheWriteInputTokens",
            "cache_write_input_tokens_count",
            "cache_write_input_tokens",
            "cache_creation_input_tokens_count",
            "cache_creation_input_tokens",
        )
    )
    cache_read = _first_token_count(
        usage,
        (
            "cacheReadInputTokensCount",
            "cacheReadInputTokens",
            "cache_read_input_tokens_count",
            "cache_read_input_tokens",
        )
    )
    return cache_creation, cache_read


def _extract_token_counts(usage: Any, metrics: Optional[Any] = None) -> Tuple[Optional[int], Optional[int]]:
    """Return (input_tokens, output_tokens) from Bedrock usage and metrics blocks."""
    usage = _as_dict(usage)
    metrics = _as_dict(metrics)
    input_tokens = _first_token_count(usage, ("inputTokens", "input_tokens", "prompt_tokens"))
    if input_tokens is None:
        input_tokens = _first_token_count(metrics, ("inputTokenCount",))

    output_tokens = _first_token_count(usage, ("outputTokens", "output_tokens", "completion_tokens"))
    if output_tokens is None:
        output_tokens = _first_token_count(metrics, ("outputTokenCount",))

    return input_tokens, output_tokens


def bedrock_invoke(model: str, payload: dict, region: Optional[str] = None) -> Tuple[str, int, int, int, int]:
    """
    Invoke Bedrock model with Anthropic-compatible parameters.

    Args:
        model: Anthropic model name (e.g., "claude-3-sonnet-20240229")
        payload: Request payload in Anthropic format
        region: AWS region (defaults to AWS_REGION, then AWS_DEFAULT_REGION, then us-east-1)

    Returns:
        Tuple of (text_content, input_tokens, output_tokens, cache_creation_tokens, cache_read_tokens)

    Raises:
        BedrockValidationError: For invalid input parameters
        BedrockInvokeError: For AWS/Bedrock API errors
        ImportError: If boto3 is not installed
    """
    # Validate inputs
    model = _validate_model_name(model)

    if not isinstance(payload, dict):
        raise BedrockValidationError(f"payload must be a dict, got {type(payload).__name__}")

    region = _resolve_region(region)
    if not isinstance(region, str) or not region.strip():
        raise BedrockValidationError(f"region must be a non-empty string, got {type(region).__name__}")

    try:
        client = get_bedrock_client(region.strip())
        model_id = _model_id(model)

        logger.debug(f"Invoking Bedrock model {model_id} in region {region}")

        # Make the API call
        resp = client.invoke_model(
            modelId=model_id,
            body=json.dumps(payload),
            accept="application/json"
        )

        # Parse response
        try:
            body = json.loads(resp["body"].read())
        except (json.JSONDecodeError, KeyError) as e:
            raise BedrockInvokeError(f"Failed to parse Bedrock response: {e}")

        usage = body.get("usage", {})

        # Extract text content from content array
        content_blocks = body.get("content", [])
        text = "".join(
            c.get("text", "")
            for c in content_blocks
            if c.get("type") == "text"
        )

        input_tokens, output_tokens = _extract_token_counts(usage)
        cache_creation_tokens, cache_read_tokens = _extract_cache_tokens(usage)
        input_tokens = input_tokens if input_tokens is not None else 0
        output_tokens = output_tokens if output_tokens is not None else 0
        cache_creation_tokens = cache_creation_tokens if cache_creation_tokens is not None else 0
        cache_read_tokens = cache_read_tokens if cache_read_tokens is not None else 0

        logger.debug(f"Bedrock invoke successful: {input_tokens} input tokens, {output_tokens} output tokens")
        return text, input_tokens, output_tokens, cache_creation_tokens, cache_read_tokens

    except BedrockValidationError:
        # Re-raise validation errors as-is
        raise
    except ImportError:
        # Re-raise import errors as-is
        raise
    except Exception as e:
        # Wrap other exceptions in BedrockInvokeError
        error_msg = f"Bedrock invoke failed for model {model} in region {region}: {e}"
        logger.error(error_msg)
        raise BedrockInvokeError(error_msg) from e


class BedrockStreamIterator:
    """
    Iterator for Bedrock streaming responses that tracks token usage.
    """

    def __init__(self, model: str, payload: dict, region: Optional[str] = None):
        self.model = model
        self.payload = payload
        self.region = _resolve_region(region)
        self.accumulated_text = ""
        self.input_tokens = 0
        self.output_tokens = 0
        self.cache_creation_tokens = 0
        self.cache_read_tokens = 0
        self._stream = None
        self._started = False

    def __iter__(self):
        return self

    def __next__(self):
        if not self._started:
            self._start_stream()
            self._started = True

        try:
            return next(self._stream)
        except StopIteration:
            raise

    def _start_stream(self):
        """Initialize the Bedrock streaming connection."""
        try:
            client = get_bedrock_client(self.region)
            model_id = _model_id(self.model)

            logger.debug(f"Starting Bedrock streaming for model {model_id} in region {self.region}")

            # Make the streaming API call
            resp = client.invoke_model_with_response_stream(
                modelId=model_id,
                body=json.dumps(self.payload),
                accept="application/json"
            )

            # Process the streaming response
            stream = resp.get("body")
            if not stream:
                logger.error("No response body in Bedrock streaming response")
                self._stream = iter([])
                return

            self._stream = self._process_stream(stream)

        except Exception as e:
            logger.error(f"Bedrock streaming invoke failed for model {self.model} in region {self.region}: {e}")
            raise

    def _process_stream(self, stream):
        """Process the Bedrock stream and yield text chunks."""
        for event in stream:
            if not hasattr(event, "get"):
                continue

            metadata = _as_dict(event.get("metadata"))
            if metadata:
                self._update_usage_from_blocks(
                    metadata.get("usage") or {},
                    metadata.get("metrics") or {}
                )

            chunk = _as_dict(event.get("chunk"))
            if not chunk:
                continue

            chunk_bytes = chunk.get("bytes")
            if not chunk_bytes:
                continue

            try:
                chunk_data = json.loads(chunk_bytes.decode("utf-8"))
                if not isinstance(chunk_data, dict):
                    continue

                logger.debug(f"Bedrock stream chunk: {chunk_data}")

                chunk_type = chunk_data.get("type")
                usage = _as_dict(chunk_data.get("usage"))
                metrics = _as_dict(chunk_data.get("amazon-bedrock-invocationMetrics"))
                if usage or metrics:
                    self._update_usage_from_blocks(usage, metrics)

                if chunk_type == "message_start":
                    message_usage = _as_dict(_as_dict(chunk_data.get("message")).get("usage"))
                    if message_usage:
                        self._update_usage_from_blocks(message_usage)

                # Handle content block delta (text chunk)
                if chunk_type == "content_block_delta":
                    delta = _as_dict(chunk_data.get("delta"))
                    text = delta.get("text", "")
                    if text:
                        self.accumulated_text += text
                        yield text

                # Handle message stop (final message with usage)
                elif chunk_type == "message_stop":
                    logger.debug(f"Bedrock streaming completed. Input tokens: {self.input_tokens}, Output tokens: {self.output_tokens}")

                # Handle message delta usage-only updates.
                elif chunk_type == "message_delta":
                    logger.debug("Bedrock stream: message_delta")

                # Handle other chunk types (content_block_start, etc.)
                elif chunk_type == "content_block_start":
                    logger.debug(f"Bedrock stream: {chunk_type}")

            except json.JSONDecodeError as e:
                logger.warning(f"Failed to decode Bedrock stream chunk: {e}")
                continue

        logger.debug(f"Bedrock streaming finished. Total text length: {len(self.accumulated_text)}")

    def _update_usage_from_blocks(self, usage: Any, metrics: Optional[Any] = None):
        """Update token counters from a stream usage block without resetting missing fields."""
        usage = _as_dict(usage)
        metrics = _as_dict(metrics)
        input_tokens, output_tokens = _extract_token_counts(usage, metrics)
        if input_tokens is not None:
            self.input_tokens = input_tokens
        if output_tokens is not None:
            self.output_tokens = output_tokens

        cache_creation_tokens, cache_read_tokens = _extract_cache_tokens(usage)
        if cache_creation_tokens is not None:
            self.cache_creation_tokens = cache_creation_tokens
        if cache_read_tokens is not None:
            self.cache_read_tokens = cache_read_tokens


def bedrock_invoke_stream(model: str, payload: dict, region: Optional[str] = None) -> BedrockStreamIterator:
    """
    Invoke Bedrock model with streaming response.

    Args:
        model: Anthropic model name (e.g., "claude-3-sonnet-20240229")
        payload: Request payload in Anthropic format
        region: AWS region (defaults to AWS_REGION, then AWS_DEFAULT_REGION, then us-east-1)

    Returns:
        BedrockStreamIterator: Iterator that yields text chunks and tracks token counts

    Raises:
        ImportError: If boto3 is not installed
        Exception: For any AWS/Bedrock API errors (re-raised with context)
    """
    return BedrockStreamIterator(model, payload, region)


def create_bedrock_payload(messages: Union[list, List[Dict[str, Any]]], **kwargs) -> dict:
    """
    Create a Bedrock-compatible payload from Anthropic parameters.

    Args:
        messages: List of message objects
        **kwargs: Additional parameters (max_tokens, temperature, etc.)

    Returns:
        Dictionary formatted for Bedrock API

    Raises:
        BedrockValidationError: For invalid input parameters
    """
    # Validate inputs
    validated_messages = _validate_messages(messages)
    max_tokens = _validate_max_tokens(kwargs.get("max_tokens", 1000))

    # Validate optional numeric parameters
    temperature = kwargs.get("temperature")
    if temperature is not None:
        if not isinstance(temperature, (int, float)) or not (0.0 <= temperature <= 1.0):
            raise BedrockValidationError(f"temperature must be a number between 0.0 and 1.0, got {temperature}")

    top_p = kwargs.get("top_p")
    if top_p is not None:
        if not isinstance(top_p, (int, float)) or not (0.0 <= top_p <= 1.0):
            raise BedrockValidationError(f"top_p must be a number between 0.0 and 1.0, got {top_p}")

    top_k = kwargs.get("top_k")
    if top_k is not None:
        if not isinstance(top_k, int) or top_k <= 0:
            raise BedrockValidationError(f"top_k must be a positive integer, got {top_k}")

    payload = {
        "anthropic_version": kwargs.get("anthropic_version", "bedrock-2023-05-31"),
        "messages": validated_messages,
        "max_tokens": max_tokens,
    }

    # Add optional parameters if provided and valid
    system = kwargs.get("system")
    if system:
        if not isinstance(system, str):
            raise BedrockValidationError(f"system must be a string, got {type(system).__name__}")
        payload["system"] = system

    if temperature is not None:
        payload["temperature"] = temperature
    if top_p is not None:
        payload["top_p"] = top_p
    if top_k is not None:
        payload["top_k"] = top_k

    stop_sequences = kwargs.get("stop_sequences")
    if stop_sequences:
        if not isinstance(stop_sequences, list):
            raise BedrockValidationError(f"stop_sequences must be a list, got {type(stop_sequences).__name__}")
        payload["stop_sequences"] = stop_sequences

    return payload


def create_anthropic_response(text: str, input_tokens: int, output_tokens: int,
                            model: str, request_id: Optional[str] = None,
                            cache_creation_tokens: int = 0, cache_read_tokens: int = 0):
    """
    Create an Anthropic-compatible response object.

    Args:
        text: Generated text content
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        model: Model name
        request_id: Optional request ID
        cache_creation_tokens: Number of cache creation input tokens (default 0)
        cache_read_tokens: Number of cache read input tokens (default 0)

    Returns:
        Object that mimics Anthropic's Message response structure with both
        attribute and dictionary access support

    Raises:
        BedrockValidationError: For invalid input parameters
    """
    # Validate inputs
    if not isinstance(text, str):
        raise BedrockValidationError(f"text must be a string, got {type(text).__name__}")

    if not isinstance(input_tokens, int) or input_tokens < 0:
        raise BedrockValidationError(f"input_tokens must be a non-negative integer, got {input_tokens}")

    if not isinstance(output_tokens, int) or output_tokens < 0:
        raise BedrockValidationError(f"output_tokens must be a non-negative integer, got {output_tokens}")

    model = _validate_model_name(model)

    # Create objects that mimic Anthropic's response structure with hybrid access
    class HybridAccessMixin:
        """Mixin to provide both attribute and dictionary access."""

        def __getitem__(self, key):
            """Support dictionary-style access."""
            try:
                return getattr(self, key)
            except AttributeError:
                raise KeyError(key)

        def __setitem__(self, key, value):
            """Support dictionary-style assignment."""
            setattr(self, key, value)

        def __contains__(self, key):
            """Support 'in' operator."""
            return hasattr(self, key)

        def get(self, key, default=None):
            """Support dict.get() method."""
            return getattr(self, key, default)

    class TextBlock(HybridAccessMixin):
        def __init__(self, text):
            self.type = "text"
            self.text = text

    class Usage(HybridAccessMixin):
        def __init__(self, input_tokens, output_tokens, cache_creation_tokens=0, cache_read_tokens=0):
            self.input_tokens = input_tokens
            self.output_tokens = output_tokens
            self.total_tokens = input_tokens + output_tokens
            self.cache_creation_input_tokens = cache_creation_tokens
            self.cache_read_input_tokens = cache_read_tokens

    class Message(HybridAccessMixin):
        def __init__(self, text, input_tokens, output_tokens, model, request_id, cache_creation_tokens=0, cache_read_tokens=0):
            self.id = request_id or _generate_safe_id("msg_bedrock", text)
            self.type = "message"
            self.role = "assistant"
            self.model = model
            self.content = [TextBlock(text)]
            self.usage = Usage(input_tokens, output_tokens, cache_creation_tokens, cache_read_tokens)
            self.stop_reason = "end_turn"
            self.stop_sequence = None

    return Message(text, input_tokens, output_tokens, model, request_id, cache_creation_tokens, cache_read_tokens)


class BedrockStreamWrapper:
    """
    Stream wrapper for Bedrock streaming responses that provides the same interface
    as Anthropic's stream wrapper for compatibility.
    """

    def __init__(self, model: str, payload: dict, region: Optional[str] = None,
                 messages: Optional[list] = None,
                 usage_metadata: Optional[dict] = None, request_time_dt: Optional[datetime.datetime] = None,
                 request_time: Optional[str] = None):
        self.model = model
        self.payload = payload
        self.messages = messages
        self.region = region
        self.usage_metadata = usage_metadata or {}
        self.request_time_dt = request_time_dt or datetime.datetime.now(datetime.timezone.utc)
        self.request_time = request_time or self.request_time_dt.strftime("%Y-%m-%dT%H:%M:%SZ")

        # Stream state
        self.stream_iterator = None
        self.response_time_dt = None
        self.response_id = None
        self.final_message = None
        self.first_token_time = None
        self.request_start_time = time.time() * 1000  # Convert to milliseconds
        self.accumulated_text = ""
        self.cache_creation_tokens = 0
        self.cache_read_tokens = 0

    def __enter__(self):
        """Enter the context manager and initialize the stream."""
        self.stream_iterator = bedrock_invoke_stream(self.model, self.payload, self.region)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):  # pylint: disable=unused-argument
        """Exit the context manager and handle metering."""
        # Get the final message with usage information
        try:
            self.response_time_dt = datetime.datetime.now(datetime.timezone.utc)
            self.response_time = self.response_time_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
            request_duration = (self.response_time_dt - self.request_time_dt).total_seconds() * 1000

            # Create final message if not already created
            if not self.final_message:
                self._create_final_message()

            # Send metering data
            self._send_metering_data(request_duration)

        except Exception as e:
            logger.warning(f"Error processing final message from Bedrock stream: {str(e)}")
            import traceback
            logger.warning(f"Traceback: {traceback.format_exc()}")

        return None

    @property
    def text_stream(self):
        """
        Property that returns an iterator for text chunks.
        Compatible with Anthropic's text_stream interface.
        """
        wrapper_self = self

        class TextStreamWrapper:
            def __iter__(self):
                return self

            def __next__(self):
                try:
                    chunk = next(wrapper_self.stream_iterator)
                    # Record the time of the first token
                    if wrapper_self.first_token_time is None and chunk:
                        wrapper_self.first_token_time = time.time() * 1000  # Convert to milliseconds

                    # Accumulate text for final message
                    wrapper_self.accumulated_text += chunk
                    return chunk
                except StopIteration:
                    # Stream is complete, create final message
                    wrapper_self._create_final_message()
                    raise

        return TextStreamWrapper()

    def get_final_message(self):
        """
        Get the final message with usage information.
        Compatible with Anthropic's get_final_message interface.
        """
        if self.final_message:
            return self.final_message

        # If final message not created yet, create it now
        self._create_final_message()
        return self.final_message

    def _create_final_message(self):
        """Create the final message object with usage information."""
        if self.final_message:
            return

        # Get token counts from the stream iterator
        input_tokens = getattr(self.stream_iterator, 'input_tokens', 0)
        output_tokens = getattr(self.stream_iterator, 'output_tokens', 0)
        self.cache_creation_tokens = getattr(self.stream_iterator, 'cache_creation_tokens', 0)
        self.cache_read_tokens = getattr(self.stream_iterator, 'cache_read_tokens', 0)

        # Generate a response ID
        self.response_id = _generate_safe_id("msg_bedrock_stream", self.accumulated_text)

        # Create an Anthropic-compatible message object
        self.final_message = create_anthropic_response(
            text=self.accumulated_text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model=self.model,
            request_id=self.response_id,
            cache_creation_tokens=self.cache_creation_tokens,
            cache_read_tokens=self.cache_read_tokens
        )

    def _send_metering_data(self, request_duration: float):
        """Send thread-safe metering data to Revenium."""
        try:
            # Import here to avoid circular imports
            from revenium_middleware import shutdown_event
            from revenium_middleware._core import submit_ai_event
            from .provider import Provider, get_provider_metadata
            from .middleware import _get_thread_safe_client, _safe_run_async_in_thread
            from .trace_fields import detect_vision_content
            from revenium_middleware._core.subscriber import extract_subscriber_from_metadata
            from revenium_middleware._core.fields import extract_org_and_product, extract_common_metadata, extract_agentic_job_fields, merge_extra_body

            if shutdown_event.is_set():
                logger.warning("Skipping metering call during shutdown")
                return

            if not self.final_message:
                logger.warning("No final message available for metering")
                return

            prompt_tokens = self.final_message.usage.input_tokens
            completion_tokens = self.final_message.usage.output_tokens

            logger.debug(
                "Bedrock streaming token usage - prompt: %d, completion: %d",
                prompt_tokens, completion_tokens
            )

            # Use Bedrock provider metadata for streaming
            provider_metadata = get_provider_metadata(Provider.BEDROCK)

            # Detect vision content
            has_vision_content = detect_vision_content(self.messages)

            # Build extra_body for vision detection and agentic job fields
            extra_body = {}
            if has_vision_content:
                extra_body['hasVisionContent'] = True
            extra_body = merge_extra_body(extra_body, extract_agentic_job_fields(self.usage_metadata or {}))

            async def metering_call():
                try:
                    if shutdown_event.is_set():
                        logger.warning("Skipping metering call during shutdown")
                        return
                    logger.debug("Metering call to Revenium for Bedrock stream completion %s", self.response_id)

                    # Get thread-safe client
                    client = _get_thread_safe_client()
                    if not client:
                        logger.warning("No thread-safe client available for Bedrock stream metering")
                        return

                    # Build subscriber object from usage metadata
                    subscriber = extract_subscriber_from_metadata(self.usage_metadata)
                    organization_name, product_name = extract_org_and_product(self.usage_metadata)
                    meta = extract_common_metadata(self.usage_metadata)

                    result = submit_ai_event("completion", {
                        "cache_creation_token_count": self.cache_creation_tokens,
                        "cache_read_token_count": self.cache_read_tokens,
                        "input_token_cost": None,
                        "output_token_cost": None,
                        "total_cost": None,
                        "output_token_count": completion_tokens,
                        "cost_type": "AI",
                        "model": self.final_message.model,
                        "input_token_count": prompt_tokens,
                        "provider": provider_metadata["provider"],
                        "model_source": provider_metadata["model_source"],
                        "reasoning_token_count": 0,
                        "request_time": self.request_time,
                        "response_time": self.response_time,
                        "completion_start_time": self.response_time,
                        "request_duration": int(request_duration),
                        "time_to_first_token": int(
                            self.first_token_time - self.request_start_time) if self.first_token_time else 0,
                        "stop_reason": "END",
                        "total_token_count": prompt_tokens + completion_tokens,
                        "transaction_id": self.response_id,
                        "trace_id": meta["trace_id"],
                        "task_type": meta["task_type"],
                        "subscriber": subscriber if subscriber else None,
                        "organization_name": organization_name,
                        "subscription_id": meta["subscription_id"],
                        "product_name": product_name,
                        "agent": meta["agent"],
                        "is_streamed": True,
                        "operation_type": "CHAT",
                        "response_quality_score": meta["response_quality_score"],
                        "middleware_source": "PYTHON",
                        "extra_body": extra_body if extra_body else None,
                    })
                    logger.debug("Metering call result for Bedrock stream: %s", result)
                except Exception as e:
                    if not shutdown_event.is_set():
                        logger.warning(f"Error in metering call for Bedrock stream: {str(e)}")
                        import traceback
                        logger.warning(f"Traceback: {traceback.format_exc()}")

            thread = _safe_run_async_in_thread(metering_call)
            logger.debug("Metering thread started for Bedrock stream: %s", thread)

        except Exception as e:
            logger.warning(f"Error setting up metering for Bedrock stream: {str(e)}")
            import traceback
            logger.warning(f"Traceback: {traceback.format_exc()}")

    def __getattr__(self, name):
        """Delegate unknown attributes to the stream iterator for compatibility."""
        if self.stream_iterator and hasattr(self.stream_iterator, name):
            return getattr(self.stream_iterator, name)
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
