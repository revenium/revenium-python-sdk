"""
Middleware for native Perplexity SDK.

This module provides metering support for the native perplexity-py SDK.
"""
import datetime
import logging
from typing import Dict, Any

import wrapt
from revenium_middleware import (
    client,
    get_client,
    run_async_in_thread,
    shutdown_event,
    merge_metadata,
)
from revenium_middleware._core import submit_ai_event
from revenium_middleware._core.config import is_selective_metering_enabled
from revenium_middleware._core.context import is_inside_decorated_function
from revenium_middleware._core.patch_registry import register_patch
from revenium_middleware._core.fields import (
    extract_org_and_product,
    extract_common_metadata,
    extract_agentic_job_fields,
    merge_extra_body,
)

from .provider import get_provider_metadata, Provider
from .trace_fields import (
    get_environment,
    get_region,
    get_credential_alias,
    get_trace_type,
    get_trace_name,
    get_parent_transaction_id,
    get_transaction_name,
    get_retry_number
)
from .middleware import (
    OperationType,
    get_stop_reason,
    extract_token_usage,
    build_trace_fields
)

logger = logging.getLogger("revenium_middleware.perplexity.sdk")


def perplexity_create_wrapper(wrapped, instance, args, kwargs):
    if is_selective_metering_enabled() and not is_inside_decorated_function():
        return wrapped(*args, **kwargs)

    logger.debug("Native Perplexity SDK chat completion wrapper called")

    # Get usage_metadata from extra_body if present
    extra_body = kwargs.get('extra_body', {})
    api_metadata = extra_body.pop('usage_metadata', {}) if isinstance(extra_body, dict) else {}

    # Merge with decorator metadata (API metadata takes precedence)
    usage_metadata = merge_metadata(api_metadata)

    # Get model from kwargs
    model = kwargs.get('model', 'sonar')

    # Check if streaming
    is_streaming = kwargs.get('stream', False)

    # Record request time
    request_time_dt = datetime.datetime.now(datetime.timezone.utc)

    # Generate transaction ID
    transaction_id = f"perplexity-sdk-{request_time_dt.timestamp()}"

    # Call original method
    logger.debug(
        f"Calling original Perplexity SDK create with model: {model}, "
        f"streaming: {is_streaming}"
    )
    response = wrapped(*args, **kwargs)

    # Handle response based on streaming
    if is_streaming:
        # For streaming, wrap the iterator
        logger.debug("Wrapping streaming response")
        return PerplexityStreamWrapper(
            response,
            model,
            request_time_dt,
            transaction_id,
            usage_metadata
        )
    else:
        # For non-streaming, send metering data
        logger.debug("Sending metering data for non-streaming response")
        run_async_in_thread(
            send_perplexity_metering_data(
                response=response,
                model=model,
                request_time_dt=request_time_dt,
                transaction_id=transaction_id,
                usage_metadata=usage_metadata,
                is_streaming=False
            )
        )

        return response


class PerplexityStreamWrapper:
    """Wrapper for Perplexity streaming responses to track usage."""

    def __init__(self, stream, model, request_time_dt, transaction_id, usage_metadata):
        self.stream = stream
        self.model = model
        self.request_time_dt = request_time_dt
        self.transaction_id = transaction_id
        self.usage_metadata = usage_metadata
        self.chunks = []
        self.last_chunk = None

    def __iter__(self):
        """Iterate over stream chunks and collect usage data."""
        for chunk in self.stream:
            self.chunks.append(chunk)
            self.last_chunk = chunk
            yield chunk

        # After stream completes, send metering data
        logger.debug("Stream completed, sending metering data")
        run_async_in_thread(
            send_perplexity_metering_data(
                response=self.last_chunk,
                model=self.model,
                request_time_dt=self.request_time_dt,
                transaction_id=self.transaction_id,
                usage_metadata=self.usage_metadata,
                is_streaming=True,
                chunks=self.chunks
            )
        )


async def send_perplexity_metering_data(
    response,
    model: str,
    request_time_dt: datetime.datetime,
    transaction_id: str,
    usage_metadata: Dict[str, Any],
    is_streaming: bool,
    chunks=None
):
    """
    Send metering data to Revenium for native Perplexity SDK.

    This function extracts usage information from the Perplexity response
    and sends it to Revenium's metering API.
    """
    if get_client() is None:
        return  # metering disabled (no API key configured)
    try:
        # Extract usage data from response
        if hasattr(response, 'usage') and response.usage:
            usage = response.usage
            input_tokens = getattr(usage, 'prompt_tokens', 0)
            output_tokens = getattr(usage, 'completion_tokens', 0)
            total_tokens = getattr(usage, 'total_tokens', input_tokens + output_tokens)
        else:
            logger.warning("No usage data found in response")
            input_tokens = 0
            output_tokens = 0
            total_tokens = 0

        # Get finish reason
        finish_reason = None
        if hasattr(response, 'choices') and response.choices:
            finish_reason = getattr(response.choices[0], 'finish_reason', None)

        # Map to Revenium stop reason
        stop_reason = get_stop_reason(finish_reason)

        # Calculate duration
        response_time_dt = datetime.datetime.now(datetime.timezone.utc)
        duration_ms = int((response_time_dt - request_time_dt).total_seconds() * 1000)

        # Get provider metadata
        provider_metadata = get_provider_metadata(Provider.PERPLEXITY)

        # Build trace fields
        trace_fields = build_trace_fields()

        # Detect operation type (native Perplexity SDK only supports chat)
        operation_type = OperationType.CHAT

        # Build completion args matching middleware.py schema
        completion_args = {
            "model": model,
            "provider": provider_metadata["provider"],
            "operation_type": operation_type.value,
            "input_token_count": input_tokens,
            "output_token_count": output_tokens,
            "total_token_count": total_tokens,
            "stop_reason": stop_reason,
            "request_time": request_time_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "response_time": response_time_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "completion_start_time": response_time_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "request_duration": duration_ms,
            "transaction_id": transaction_id,
            "is_streamed": is_streaming,
            "cost_type": "AI",
            "cache_creation_token_count": 0,
            "cache_read_token_count": 0,
            "reasoning_token_count": 0,
        }

        organization_name, product_name = extract_org_and_product(usage_metadata)
        if organization_name:
            completion_args["organization_name"] = organization_name
        if product_name:
            completion_args["product_name"] = product_name

        meta = extract_common_metadata(usage_metadata)
        for key, value in meta.items():
            if value:
                completion_args[key] = value

        if usage_metadata.get("subscriber"):
            completion_args["subscriber"] = usage_metadata.get("subscriber")

        custom_fields = ["service", "step", "service_name"]
        for field in custom_fields:
            if usage_metadata.get(field):
                completion_args[field] = usage_metadata.get(field)

        for key, value in trace_fields.items():
            if value is not None:
                completion_args[key] = value

        agentic_fields = extract_agentic_job_fields(usage_metadata)
        extra_body = merge_extra_body(None, agentic_fields)

        if extra_body:
            completion_args["extra_body"] = extra_body

        logger.debug(f"Sending metering data to Revenium: {completion_args}")
        result = submit_ai_event("completion", completion_args)
        logger.debug(f"Metering call result: {result}")

    except Exception as e:
        if not shutdown_event.is_set():
            logger.warning(f"Error in metering call: {str(e)}")


try:
    import perplexity.resources.chat.completions  # noqa: F401

    if register_patch("perplexity.resources.chat.completions.Completions.create"):
        wrapt.wrap_function_wrapper(
            'perplexity.resources.chat.completions',
            'Completions.create',
            perplexity_create_wrapper
        )
        logger.debug("Successfully patched native Perplexity SDK")
except (ImportError, AttributeError) as e:
    logger.debug(f"Native Perplexity SDK not available or incompatible: {e}")
