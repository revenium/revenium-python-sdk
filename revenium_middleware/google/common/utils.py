"""
Common utilities for Google AI middleware.

This module contains shared functionality used by both Google AI SDK
and Vertex AI SDK middleware implementations.
"""

import datetime
import logging
import os
import uuid
from typing import Dict, Any, Optional

from revenium_middleware import client, run_async_in_thread, shutdown_event
from revenium_middleware._core.fields import (
    extract_field_with_fallback as _core_extract_field_with_fallback,
    extract_org_and_product,
    extract_common_metadata,
    extract_agentic_job_fields,
    merge_extra_body,
)

from .types import UsageData, OperationType, ProviderMetadata, TokenCounts
from .exceptions import MeteringError, APIResponseError, safe_extract
from .protocols import has_token_counts, safe_getattr, get_token_count
from . import trace_fields

logger = logging.getLogger("revenium_middleware.extension")


def is_debug_logging_enabled() -> bool:
    """
    Check if debug logging is currently enabled for Revenium middleware.

    Returns:
        bool: True if debug logging is enabled, False otherwise
    """
    # Check environment variable first
    log_level_str = os.getenv("REVENIUM_LOG_LEVEL", "INFO").upper()
    if log_level_str == "DEBUG":
        return True

    # Check actual logger level as fallback
    revenium_logger = logging.getLogger("revenium_middleware")
    return revenium_logger.getEffectiveLevel() <= logging.DEBUG


def generate_transaction_id() -> str:
    """Generate a unique transaction ID."""
    return str(uuid.uuid4())


def format_timestamp(dt: datetime.datetime) -> str:
    """Format datetime as ISO string for API calls."""
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


extract_field_with_fallback = _core_extract_field_with_fallback


def calculate_duration_ms(
    start_time: datetime.datetime, end_time: datetime.datetime
) -> int:
    """Calculate duration in milliseconds between two timestamps."""
    return int((end_time - start_time).total_seconds() * 1000)


async def log_token_usage(
    transaction_id: str,
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    total_tokens: int,
    cached_tokens: int,
    stop_reason: str,
    request_time: str,
    response_time: str,
    request_duration: int,
    usage_metadata: Dict[str, Any],
    provider: str = "Google",
    model_source: str = "GOOGLE",
    is_streamed: bool = False,
    time_to_first_token: int = 0,
    operation_type: OperationType = OperationType.CHAT,
    # Prompt capture fields
    system_prompt: Optional[str] = None,
    input_messages: Optional[str] = None,
    output_response: Optional[str] = None,
    prompts_truncated: Optional[bool] = None,
) -> None:
    """
    Log token usage to Revenium.

    Args:
        transaction_id: Unique identifier for this API call
        model: Model name used for the request
        prompt_tokens: Number of input tokens
        completion_tokens: Number of output tokens
        total_tokens: Total token count
        cached_tokens: Number of cached tokens used
        stop_reason: Reason the generation stopped
        request_time: ISO timestamp of request start
        response_time: ISO timestamp of response completion
        request_duration: Duration in milliseconds
        usage_metadata: Additional metadata for the request
        provider: Provider name (always "Google")
        model_source: Model source (always "GOOGLE")
        is_streamed: Whether this was a streaming response
        time_to_first_token: Time to first token in milliseconds
        operation_type: Type of operation (CHAT or EMBED)
        system_prompt: System instruction/prompt text
        input_messages: Input messages as JSON string
        output_response: Output response text
        prompts_truncated: Whether any prompts were truncated
    """
    if client is None:
        return  # metering disabled (no API key configured)
    if shutdown_event.is_set():
        logger.warning("Skipping metering call during shutdown")
        return

    logger.debug(
        "Metering call to Revenium for %s operation %s",
        operation_type.lower(),
        transaction_id,
    )

    # Prepare arguments for create_completion (using snake_case for Python client library)
    completion_args = {
        "cache_creation_token_count": cached_tokens,
        "cache_read_token_count": 0,
        "input_token_cost": None,  # Let backend calculate from model pricing
        "output_token_cost": None,  # Let backend calculate from model pricing
        "total_cost": None,  # Let backend calculate from model pricing
        "output_token_count": completion_tokens,
        "cost_type": "AI",
        "model": model,
        "input_token_count": prompt_tokens,
        "provider": provider,
        "model_source": model_source,
        "reasoning_token_count": 0,
        "request_time": request_time,
        "response_time": response_time,
        "completion_start_time": response_time,
        "request_duration": int(request_duration),
        "stop_reason": stop_reason,
        "total_token_count": total_tokens,
        "transaction_id": transaction_id,
        "is_streamed": is_streamed,
        "operation_type": operation_type.value,  # Convert enum to string
        "time_to_first_token": time_to_first_token,
        "middleware_source": "python",  # Required parameter for Google Python middleware
    }

    organization_name, product_name = extract_org_and_product(usage_metadata)
    if organization_name:
        completion_args["organization_name"] = organization_name
    if product_name:
        completion_args["product_name"] = product_name

    meta = extract_common_metadata(usage_metadata)
    if meta["trace_id"]:
        completion_args["trace_id"] = meta["trace_id"]
    if meta["task_type"]:
        completion_args["task_type"] = meta["task_type"]
    if meta["subscription_id"]:
        completion_args["subscription_id"] = meta["subscription_id"]
    if meta["agent"]:
        completion_args["agent"] = meta["agent"]
    if meta["response_quality_score"]:
        completion_args["response_quality_score"] = meta["response_quality_score"]

    # Add vision content flag if detected
    if usage_metadata.get("has_vision_content"):
        completion_args["has_vision_content"] = True

    # Add trace visualization fields (v0.2.0+)
    # These fields support both environment variables and usage_metadata parameters
    # Priority: usage_metadata > environment variable

    # Environment field
    environment = (
        usage_metadata.get("environment") or
        trace_fields.get_environment()
    )
    if environment:
        completion_args["environment"] = environment

    # Region field
    region = (
        usage_metadata.get("region") or
        trace_fields.get_region()
    )
    if region:
        completion_args["region"] = region

    # Credential alias field
    credential_alias = (
        usage_metadata.get("credential_alias") or
        usage_metadata.get("credentialAlias") or
        trace_fields.get_credential_alias()
    )
    if credential_alias:
        completion_args["credential_alias"] = credential_alias

    # Trace type field
    trace_type = (
        usage_metadata.get("trace_type") or
        usage_metadata.get("traceType") or
        trace_fields.get_trace_type()
    )
    if trace_type:
        # Validate if coming from usage_metadata
        if usage_metadata.get("trace_type") or usage_metadata.get("traceType"):
            trace_type = trace_fields.validate_trace_type(trace_type)
        if trace_type:
            completion_args["trace_type"] = trace_type

    # Trace name field
    trace_name = (
        usage_metadata.get("trace_name") or
        usage_metadata.get("traceName") or
        trace_fields.get_trace_name()
    )
    if trace_name:
        # Validate if coming from usage_metadata
        if usage_metadata.get("trace_name") or usage_metadata.get("traceName"):
            trace_name = trace_fields.validate_trace_name(trace_name)
        if trace_name:
            completion_args["trace_name"] = trace_name

    # Parent transaction ID field
    parent_transaction_id = (
        usage_metadata.get("parent_transaction_id") or
        usage_metadata.get("parentTransactionId") or
        trace_fields.get_parent_transaction_id()
    )
    if parent_transaction_id:
        completion_args["parent_transaction_id"] = parent_transaction_id

    # Transaction name field (with fallback to task_type)
    transaction_name = trace_fields.get_transaction_name(usage_metadata)
    if transaction_name:
        completion_args["transaction_name"] = transaction_name

    # Retry number field
    retry_number = trace_fields.get_retry_number()
    if retry_number > 0:
        completion_args["retry_number"] = retry_number

    # Build subscriber object - support both nested and flat formats
    subscriber_data = {}
    flat_keys_used = []

    # Prefer nested format if present (recommended structure)
    if "subscriber" in usage_metadata:
        nested_subscriber = usage_metadata["subscriber"]
        if isinstance(nested_subscriber, dict):
            # Use nested structure directly
            subscriber_data = nested_subscriber.copy()
            logger.debug("Using nested subscriber format (recommended)")
    else:
        # Fall back to flat keys for backward compatibility
        subscriber_id = usage_metadata.get("subscriber_id")
        subscriber_email = usage_metadata.get("subscriber_email")
        credential_name = usage_metadata.get("subscriber_credential_name")
        credential_value = usage_metadata.get("subscriber_credential")

        if subscriber_id:
            subscriber_data["id"] = subscriber_id
            flat_keys_used.append("subscriber_id")
        if subscriber_email:
            subscriber_data["email"] = subscriber_email
            flat_keys_used.append("subscriber_email")

        # Add credential sub-object if credential data is provided
        credential_data = {}
        if credential_name:
            credential_data["name"] = credential_name
            flat_keys_used.append("subscriber_credential_name")
        if credential_value:
            credential_data["value"] = credential_value
            flat_keys_used.append("subscriber_credential")

        if credential_data:
            subscriber_data["credential"] = credential_data

        # Log deprecation warning if flat keys were used
        if flat_keys_used:
            logger.warning(
                f"Flat subscriber keys are deprecated: {flat_keys_used}. "
                "Please use nested 'subscriber' object format: "
                "{'subscriber': {'id': '...', 'email': '...', 'credential': {'name': '...', 'value': '...'}}}"
            )

    # Only add subscriber to completion_args if we have subscriber data
    if subscriber_data:
        completion_args["subscriber"] = subscriber_data

    # Add prompt capture fields only if they have values
    if system_prompt is not None:
        completion_args["system_prompt"] = system_prompt
    if input_messages is not None:
        completion_args["input_messages"] = input_messages
    if output_response is not None:
        completion_args["output_response"] = output_response
    if prompts_truncated is not None:
        completion_args["prompts_truncated"] = prompts_truncated

    # Log the arguments at debug level (redact sensitive prompt data)
    safe_args = {k: v for k, v in completion_args.items()
                 if k not in ('system_prompt', 'input_messages', 'output_response')}
    if 'system_prompt' in completion_args:
        safe_args['system_prompt'] = '[REDACTED]'
    if 'input_messages' in completion_args:
        safe_args['input_messages'] = '[REDACTED]'
    if 'output_response' in completion_args:
        safe_args['output_response'] = '[REDACTED]'
    logger.debug("Calling client.ai.create_completion with args: %s", safe_args)

    agentic_fields = extract_agentic_job_fields(usage_metadata)
    extra_body = merge_extra_body(completion_args.get("extra_body"), agentic_fields)
    if extra_body:
        completion_args["extra_body"] = extra_body

    logger.debug(
        f"Metering call for {operation_type.value}: {transaction_id}, tokens: {prompt_tokens}+{completion_tokens}={total_tokens}"
    )

    try:
        result = client.ai.create_completion(**completion_args)
        logger.debug("Metering call result: %s", result)
        logger.info(" REVENIUM SUCCESS: Metering call successful: %s", result.id)
    except Exception as e:
        if not shutdown_event.is_set():
            # Create a structured error for better handling
            error_details = {
                "transaction_id": transaction_id,
                "model": model,
                "error_type": type(e).__name__,
                "completion_args_keys": list(completion_args.keys()),
            }

            # Log error with structured information
            logger.error(" REVENIUM FAILURE: Error in metering call: %s", str(e))
            logger.error(" REVENIUM FAILURE: Error details: %s", error_details)

            # Log traceback at debug level to avoid spam
            logger.debug("Metering call traceback:", exc_info=True)

            # Raise a specific MeteringError for better error handling upstream
            raise MeteringError(
                f"Failed to send metering data: {str(e)}",
                transaction_id=transaction_id,
                api_response=None,
                error_details=error_details,
            ) from e
        else:
            logger.debug("Metering call failed during shutdown - this is expected")


def create_metering_call(
    usage_data: UsageData,
    usage_metadata: Dict[str, Any],
    time_to_first_token: int = 0,
    is_streamed: bool = False,
    # Prompt capture fields
    system_prompt: Optional[str] = None,
    input_messages: Optional[str] = None,
    output_response: Optional[str] = None,
    prompts_truncated: Optional[bool] = None,
) -> None:
    """
    Create and execute a metering call using UsageData.

    This is a higher-level function that uses the standardized UsageData structure.
    """
    # Override streaming and timing info
    usage_data.is_streamed = is_streamed
    usage_data.time_to_first_token = time_to_first_token

    # Create async metering call
    async def metering_call():
        await log_token_usage(
            transaction_id=usage_data.transaction_id,
            model=usage_data.model,
            prompt_tokens=usage_data.input_token_count,  # These are positional parameters
            completion_tokens=usage_data.output_token_count,  # These are positional parameters
            total_tokens=usage_data.total_token_count,  # These are positional parameters
            cached_tokens=usage_data.cache_creation_token_count,
            stop_reason=usage_data.stop_reason,
            request_time=usage_data.request_time,
            response_time=usage_data.response_time,
            request_duration=usage_data.request_duration,
            usage_metadata=usage_metadata,
            provider=usage_data.provider,
            model_source=usage_data.model_source,
            is_streamed=usage_data.is_streamed,
            time_to_first_token=usage_data.time_to_first_token,
            operation_type=OperationType(usage_data.operation_type),
            # Prompt capture fields
            system_prompt=system_prompt,
            input_messages=input_messages,
            output_response=output_response,
            prompts_truncated=prompts_truncated,
        )

    # Execute in background thread
    run_async_in_thread(metering_call())


def _build_common_metadata_args(usage_metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build common metadata arguments shared across all metering endpoints.

    Extracts trace fields, subscriber data, organization/product names, etc.
    from usage_metadata and environment variables.

    Returns:
        Dictionary of common keyword arguments for metering API calls.
    """
    args = {}

    organization_name, product_name = extract_org_and_product(usage_metadata)
    if organization_name:
        args["organization_name"] = organization_name
    if product_name:
        args["product_name"] = product_name

    meta = extract_common_metadata(usage_metadata)
    if meta["trace_id"]:
        args["trace_id"] = meta["trace_id"]
    if meta["task_type"]:
        args["task_type"] = meta["task_type"]
    if meta["subscription_id"]:
        args["subscription_id"] = meta["subscription_id"]
    if meta["agent"]:
        args["agent"] = meta["agent"]

    # Trace visualization fields
    environment = usage_metadata.get("environment") or trace_fields.get_environment()
    if environment:
        args["environment"] = environment

    region = usage_metadata.get("region") or trace_fields.get_region()
    if region:
        args["region"] = region

    credential_alias = (
        usage_metadata.get("credential_alias") or
        usage_metadata.get("credentialAlias") or
        trace_fields.get_credential_alias()
    )
    if credential_alias:
        args["credential_alias"] = credential_alias

    trace_type = (
        usage_metadata.get("trace_type") or
        usage_metadata.get("traceType") or
        trace_fields.get_trace_type()
    )
    if trace_type:
        if usage_metadata.get("trace_type") or usage_metadata.get("traceType"):
            trace_type = trace_fields.validate_trace_type(trace_type)
        if trace_type:
            args["trace_type"] = trace_type

    trace_name = (
        usage_metadata.get("trace_name") or
        usage_metadata.get("traceName") or
        trace_fields.get_trace_name()
    )
    if trace_name:
        if usage_metadata.get("trace_name") or usage_metadata.get("traceName"):
            trace_name = trace_fields.validate_trace_name(trace_name)
        if trace_name:
            args["trace_name"] = trace_name

    parent_transaction_id = (
        usage_metadata.get("parent_transaction_id") or
        usage_metadata.get("parentTransactionId") or
        trace_fields.get_parent_transaction_id()
    )
    if parent_transaction_id:
        args["parent_transaction_id"] = parent_transaction_id

    transaction_name = trace_fields.get_transaction_name(usage_metadata)
    if transaction_name:
        args["transaction_name"] = transaction_name

    retry_number = trace_fields.get_retry_number()
    if retry_number > 0:
        args["retry_number"] = retry_number

    # Subscriber data
    subscriber_data = {}
    flat_keys_used = []
    if "subscriber" in usage_metadata:
        nested_subscriber = usage_metadata["subscriber"]
        if isinstance(nested_subscriber, dict):
            subscriber_data = nested_subscriber.copy()
    else:
        subscriber_id = usage_metadata.get("subscriber_id")
        subscriber_email = usage_metadata.get("subscriber_email")
        credential_name = usage_metadata.get("subscriber_credential_name")
        credential_value = usage_metadata.get("subscriber_credential")
        if subscriber_id:
            subscriber_data["id"] = subscriber_id
            flat_keys_used.append("subscriber_id")
        if subscriber_email:
            subscriber_data["email"] = subscriber_email
            flat_keys_used.append("subscriber_email")
        credential_data = {}
        if credential_name:
            credential_data["name"] = credential_name
            flat_keys_used.append("subscriber_credential_name")
        if credential_value:
            credential_data["value"] = credential_value
            flat_keys_used.append("subscriber_credential")
        if credential_data:
            subscriber_data["credential"] = credential_data
        if flat_keys_used:
            logger.warning(
                f"Flat subscriber keys are deprecated: {flat_keys_used}. "
                "Please use nested 'subscriber' object format."
            )
    if subscriber_data:
        args["subscriber"] = subscriber_data

    return args


async def log_image_usage(
    transaction_id: str,
    model: str,
    requested_image_count: int,
    actual_image_count: int,
    request_time: str,
    response_time: str,
    request_duration: int,
    usage_metadata: Dict[str, Any],
    provider: str = "Google",
    model_source: str = "GOOGLE",
    operation_subtype: str = "generation",
    resolution: Optional[str] = None,
    quality: Optional[str] = None,
    style: Optional[str] = None,
    aspect_ratio: Optional[str] = None,
) -> None:
    """
    Log image generation usage to Revenium via /meter/v2/ai/images.

    Args:
        transaction_id: Unique identifier for this API call
        model: Model name (e.g., "imagen-3.0-generate-001")
        requested_image_count: Number of images requested
        actual_image_count: Number of images actually generated
        request_time: ISO timestamp of request start
        response_time: ISO timestamp of response completion
        request_duration: Duration in milliseconds
        usage_metadata: Additional metadata for the request
        provider: Provider name (always "Google")
        model_source: Model source (always "GOOGLE")
        operation_subtype: "generation", "edit", or "upscale"
        resolution: Image resolution (e.g., "1024x1024")
        quality: Image quality setting
        style: Image style setting
        aspect_ratio: Aspect ratio (e.g., "16:9")
    """
    if client is None:
        return  # metering disabled (no API key configured)
    if shutdown_event.is_set():
        logger.warning("Skipping image metering call during shutdown")
        return

    logger.debug(
        "Image metering call to Revenium for %s operation %s",
        operation_subtype,
        transaction_id,
    )

    image_args = {
        "model": model,
        "provider": provider,
        "model_source": model_source,
        "request_time": request_time,
        "response_time": response_time,
        "request_duration": int(request_duration),
        "transaction_id": transaction_id,
        "requested_image_count": requested_image_count,
        "actual_image_count": actual_image_count,
        "operation_subtype": operation_subtype,
        "middleware_source": "python",
    }

    if resolution:
        image_args["resolution"] = resolution
    if quality:
        image_args["quality"] = quality
    if style:
        image_args["style"] = style
    if aspect_ratio:
        image_args["aspect_ratio"] = aspect_ratio

    # Add common metadata
    image_args.update(_build_common_metadata_args(usage_metadata))

    agentic_fields = extract_agentic_job_fields(usage_metadata)
    extra_body = merge_extra_body(image_args.get("extra_body"), agentic_fields)
    if extra_body:
        image_args["extra_body"] = extra_body

    logger.debug("Calling client.ai.create_image with args: %s", image_args)

    try:
        result = client.ai.create_image(**image_args)
        logger.debug("Image metering call result: %s", result)
        logger.info(" REVENIUM SUCCESS: Image metering call successful: %s", result.id)
    except Exception as e:
        if not shutdown_event.is_set():
            error_details = {
                "transaction_id": transaction_id,
                "model": model,
                "error_type": type(e).__name__,
            }
            logger.error(" REVENIUM FAILURE: Error in image metering call: %s", str(e))
            logger.error(" REVENIUM FAILURE: Error details: %s", error_details)
            logger.debug("Image metering call traceback:", exc_info=True)
            raise MeteringError(
                f"Failed to send image metering data: {str(e)}",
                transaction_id=transaction_id,
                api_response=None,
                error_details=error_details,
            ) from e


async def log_video_usage(
    transaction_id: str,
    model: str,
    duration_seconds: float,
    request_time: str,
    response_time: str,
    request_duration: int,
    usage_metadata: Dict[str, Any],
    provider: str = "Google",
    model_source: str = "GOOGLE",
    operation_subtype: str = "generation",
    resolution: Optional[str] = None,
    fps: Optional[int] = None,
    aspect_ratio: Optional[str] = None,
    video_job_id: Optional[str] = None,
    async_operation: bool = False,
) -> None:
    """
    Log video generation usage to Revenium via /meter/v2/ai/video.

    Args:
        transaction_id: Unique identifier for this API call
        model: Model name (e.g., "veo-2.0-generate-001")
        duration_seconds: Duration of generated video in seconds
        request_time: ISO timestamp of request start
        response_time: ISO timestamp of response completion
        request_duration: Duration in milliseconds
        usage_metadata: Additional metadata for the request
        provider: Provider name (always "Google")
        model_source: Model source (always "GOOGLE")
        operation_subtype: "generation", "extend", or "upscale"
        resolution: Video resolution (e.g., "1080p")
        fps: Frames per second
        aspect_ratio: Aspect ratio (e.g., "16:9")
        video_job_id: Job ID for async operations
        async_operation: Whether this was an async operation
    """
    if client is None:
        return  # metering disabled (no API key configured)
    if shutdown_event.is_set():
        logger.warning("Skipping video metering call during shutdown")
        return

    logger.debug(
        "Video metering call to Revenium for %s operation %s",
        operation_subtype,
        transaction_id,
    )

    video_args = {
        "model": model,
        "provider": provider,
        "model_source": model_source,
        "request_time": request_time,
        "response_time": response_time,
        "request_duration": int(request_duration),
        "transaction_id": transaction_id,
        "duration_seconds": duration_seconds,
        "operation_subtype": operation_subtype,
        "middleware_source": "python",
    }

    if resolution:
        video_args["resolution"] = resolution
    if fps is not None:
        video_args["fps"] = fps
    if aspect_ratio:
        video_args["aspect_ratio"] = aspect_ratio
    if video_job_id:
        video_args["video_job_id"] = video_job_id
    if async_operation:
        video_args["async_operation"] = async_operation

    # Add common metadata
    video_args.update(_build_common_metadata_args(usage_metadata))

    agentic_fields = extract_agentic_job_fields(usage_metadata)
    extra_body = merge_extra_body(video_args.get("extra_body"), agentic_fields)
    if extra_body:
        video_args["extra_body"] = extra_body

    logger.debug("Calling client.ai.create_video with args: %s", video_args)

    try:
        result = client.ai.create_video(**video_args)
        logger.debug("Video metering call result: %s", result)
        logger.info(" REVENIUM SUCCESS: Video metering call successful: %s", result.id)
    except Exception as e:
        if not shutdown_event.is_set():
            error_details = {
                "transaction_id": transaction_id,
                "model": model,
                "error_type": type(e).__name__,
            }
            logger.error(" REVENIUM FAILURE: Error in video metering call: %s", str(e))
            logger.error(" REVENIUM FAILURE: Error details: %s", error_details)
            logger.debug("Video metering call traceback:", exc_info=True)
            raise MeteringError(
                f"Failed to send video metering data: {str(e)}",
                transaction_id=transaction_id,
                api_response=None,
                error_details=error_details,
            ) from e


def create_image_metering_call(
    model: str,
    requested_image_count: int,
    actual_image_count: int,
    request_time_dt: datetime.datetime,
    response_time_dt: datetime.datetime,
    usage_metadata: Dict[str, Any],
    provider: str = "Google",
    model_source: str = "GOOGLE",
    operation_subtype: str = "generation",
    resolution: Optional[str] = None,
    quality: Optional[str] = None,
    style: Optional[str] = None,
    aspect_ratio: Optional[str] = None,
) -> None:
    """Create and execute an image metering call in a background thread."""
    transaction_id = generate_transaction_id()
    request_time = format_timestamp(request_time_dt)
    response_time = format_timestamp(response_time_dt)
    request_duration = calculate_duration_ms(request_time_dt, response_time_dt)

    async def metering_call():
        await log_image_usage(
            transaction_id=transaction_id,
            model=model,
            requested_image_count=requested_image_count,
            actual_image_count=actual_image_count,
            request_time=request_time,
            response_time=response_time,
            request_duration=request_duration,
            usage_metadata=usage_metadata,
            provider=provider,
            model_source=model_source,
            operation_subtype=operation_subtype,
            resolution=resolution,
            quality=quality,
            style=style,
            aspect_ratio=aspect_ratio,
        )

    run_async_in_thread(metering_call())


def create_video_metering_call(
    model: str,
    duration_seconds: float,
    request_time_dt: datetime.datetime,
    response_time_dt: datetime.datetime,
    usage_metadata: Dict[str, Any],
    provider: str = "Google",
    model_source: str = "GOOGLE",
    operation_subtype: str = "generation",
    resolution: Optional[str] = None,
    fps: Optional[int] = None,
    aspect_ratio: Optional[str] = None,
    video_job_id: Optional[str] = None,
    async_operation: bool = False,
) -> None:
    """Create and execute a video metering call in a background thread."""
    transaction_id = generate_transaction_id()
    request_time = format_timestamp(request_time_dt)
    response_time = format_timestamp(response_time_dt)
    request_duration = calculate_duration_ms(request_time_dt, response_time_dt)

    async def metering_call():
        await log_video_usage(
            transaction_id=transaction_id,
            model=model,
            duration_seconds=duration_seconds,
            request_time=request_time,
            response_time=response_time,
            request_duration=request_duration,
            usage_metadata=usage_metadata,
            provider=provider,
            model_source=model_source,
            operation_subtype=operation_subtype,
            resolution=resolution,
            fps=fps,
            aspect_ratio=aspect_ratio,
            video_job_id=video_job_id,
            async_operation=async_operation,
        )

    run_async_in_thread(metering_call())


@safe_extract
def extract_model_name(response: Any, fallback: Optional[str] = None) -> str:
    """
    Extract model name from API response with fallback.

    Args:
        response: API response object
        fallback: Fallback model name if extraction fails

    Returns:
        Model name string

    Raises:
        TokenExtractionError: If extraction fails and no fallback is provided
    """
    if response is None:
        if fallback:
            logger.debug("Response is None, using fallback model name: %s", fallback)
            return fallback
        raise APIResponseError("Response is None and no fallback model name provided")

    # Try common model name attributes using safe access
    model_attrs = ["model", "model_name", "_model_name", "model_version"]
    for attr in model_attrs:
        model_name = safe_getattr(response, attr)
        if model_name:
            return str(model_name)

    # Try nested model attributes
    usage = safe_getattr(response, "usage")
    if usage:
        model_name = safe_getattr(usage, "model")
        if model_name:
            return str(model_name)

    # Use fallback if provided
    if fallback:
        logger.debug(
            "Could not extract model name from response, using fallback: %s", fallback
        )
        return fallback

    # Log available attributes for debugging
    available_attrs = [attr for attr in dir(response) if not attr.startswith("_")]
    logger.warning(
        "Could not extract model name from response. Available attributes: %s",
        available_attrs[:10],
    )

    return "unknown-model"


@safe_extract
def extract_token_counts(response: Any, operation_type: OperationType) -> TokenCounts:
    """
    Extract token counts from API response.

    This function handles the differences between SDKs and operation types.

    Args:
        response: API response object
        operation_type: Type of operation (CHAT or EMBED)

    Returns:
        TokenCounts object with extracted counts

    Raises:
        TokenExtractionError: If response is invalid
    """
    if response is None:
        logger.warning("Response is None, returning zero token counts")
        return TokenCounts(
            input_tokens=0, output_tokens=0, total_tokens=0, cached_tokens=0
        )

    # Initialize with zeros
    input_tokens = 0
    output_tokens = 0
    total_tokens = 0
    cached_tokens = 0

    # Try to extract from usage_metadata first (Google AI SDK pattern)
    usage = safe_getattr(response, "usage_metadata") or safe_getattr(response, "usage")

    if usage and has_token_counts(usage):
        # Use the utility function for safe token extraction
        input_tokens = get_token_count(
            usage, ["prompt_token_count", "prompt_tokens", "input_tokens"]
        )
        output_tokens = get_token_count(
            usage, ["candidates_token_count", "completion_tokens", "output_tokens"]
        )
        total_tokens = get_token_count(usage, ["total_token_count", "total_tokens"])
        cached_tokens = get_token_count(
            usage, ["cached_content_token_count", "cached_tokens"]
        )

        logger.debug(
            "Extracted token counts from usage: input=%d, output=%d, total=%d, cached=%d",
            input_tokens,
            output_tokens,
            total_tokens,
            cached_tokens,
        )
    else:
        logger.debug("No usage metadata found or no token counts available in response")

    # For embeddings, output tokens are always 0
    if operation_type == OperationType.EMBED:
        output_tokens = 0
        logger.debug("Operation type is EMBED, setting output_tokens to 0")

    # Calculate total if not provided but we have input/output
    if total_tokens == 0 and (input_tokens > 0 or output_tokens > 0):
        total_tokens = input_tokens + output_tokens
        logger.debug(
            "Calculated total_tokens as %d (input=%d + output=%d)",
            total_tokens,
            input_tokens,
            output_tokens,
        )

    return TokenCounts(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        cached_tokens=cached_tokens,
    )


def create_usage_data(
    response: Any,
    operation_type: OperationType,
    provider_metadata: ProviderMetadata,
    request_time: datetime.datetime,
    response_time: datetime.datetime,
    model_name_fallback: Optional[str] = None,
    stop_reason_fallback: str = "END",
) -> UsageData:
    """
    Create standardized UsageData from API response.

    This is the main function for converting SDK-specific responses
    to our common UsageData format.
    """
    # Extract token counts
    token_counts = extract_token_counts(response, operation_type)

    # Extract model name
    model_name = extract_model_name(response, model_name_fallback)

    # Extract stop reason (SDK-specific logic should be handled by caller)
    stop_reason = stop_reason_fallback
    if hasattr(response, "finish_reason"):
        stop_reason = response.finish_reason or stop_reason_fallback
    elif hasattr(response, "candidates") and response.candidates:
        candidate = response.candidates[0]
        if hasattr(candidate, "finish_reason"):
            stop_reason = candidate.finish_reason or stop_reason_fallback

    # Create UsageData
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
