"""
Middleware for native Perplexity SDK.

This module provides metering support for the native perplexity-py SDK.
"""
import datetime
import logging
from typing import Dict, Any

from revenium_middleware import (
    client,
    get_client,
    run_async_in_thread,
    shutdown_event,
    merge_metadata,
)
from revenium_middleware._core import submit_ai_event
from revenium_middleware._core.fields import (
    extract_org_and_product,
    extract_common_metadata,
    extract_effort_field,
    extract_prompt_context_fields,
    extract_agentic_job_fields,
    merge_extra_body,
)

from ._metering_scope import metering_skipped
from .patching import wrap_registered
from .provider import get_provider_metadata, Provider
from .streaming import AsyncMeteredStream, MeteredStream, StreamMeter
from .trace_fields import (
    get_environment,
    get_region,
    get_credential_alias,
    get_trace_type,
    get_trace_name,
    get_parent_transaction_id,
    get_transaction_name,
    get_retry_number,
    get_ticket_id,
    get_agent_version
)
from .middleware import (
    OperationType,
    get_stop_reason,
    extract_token_usage,
    build_trace_fields
)

logger = logging.getLogger("revenium_middleware.perplexity.sdk")


class _NativeCall:
    """The request-side facts of one native chat completion call."""

    def __init__(self, kwargs):
        extra_body = kwargs.get('extra_body', {})
        api_metadata = extra_body.pop('usage_metadata', {}) if isinstance(extra_body, dict) else {}
        self.usage_metadata = merge_metadata(api_metadata)
        self.model = kwargs.get('model', 'sonar')
        self.is_streaming = bool(kwargs.get('stream', False))
        self.request_time_dt = datetime.datetime.now(datetime.timezone.utc)
        self.transaction_id = f"perplexity-sdk-{self.request_time_dt.timestamp()}"

    def meter(self, response) -> None:
        run_async_in_thread(
            send_perplexity_metering_data(
                response=response,
                model=self.model,
                request_time_dt=self.request_time_dt,
                transaction_id=self.transaction_id,
                usage_metadata=self.usage_metadata,
                is_streaming=self.is_streaming,
            )
        )


def perplexity_create_wrapper(wrapped, instance, args, kwargs):
    if metering_skipped():
        return wrapped(*args, **kwargs)

    call = _NativeCall(kwargs)
    response = wrapped(*args, **kwargs)
    if call.is_streaming:
        return MeteredStream(response, StreamMeter(call.meter))
    call.meter(response)
    return response


def async_perplexity_create_wrapper(wrapped, instance, args, kwargs):
    if metering_skipped():
        return wrapped(*args, **kwargs)

    call = _NativeCall(kwargs)

    async def invoke():
        response = await wrapped(*args, **kwargs)
        if call.is_streaming:
            return AsyncMeteredStream(response, StreamMeter(call.meter))
        call.meter(response)
        return response

    return invoke()


async def send_perplexity_metering_data(
    response,
    model: str,
    request_time_dt: datetime.datetime,
    transaction_id: str,
    usage_metadata: Dict[str, Any],
    is_streaming: bool,
):
    """
    Send metering data to Revenium for native Perplexity SDK.

    This function extracts usage information from the Perplexity response
    and sends it to Revenium's metering API.
    """
    if get_client() is None:
        return  # metering disabled (no API key configured)
    try:
        if not getattr(response, 'usage', None):
            logger.warning("No usage data found in response")
        token_usage = extract_token_usage(response)

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
        if ticket_id := get_ticket_id(usage_metadata):
            trace_fields["ticket_id"] = ticket_id
        if agent_version := get_agent_version(usage_metadata):
            trace_fields["agent_version"] = agent_version

        # Detect operation type (native Perplexity SDK only supports chat)
        operation_type = OperationType.CHAT

        # Build completion args matching middleware.py schema
        completion_args = {
            "model": model,
            "provider": provider_metadata["provider"],
            "operation_type": operation_type.value,
            "input_token_count": token_usage["prompt_tokens"],
            "output_token_count": token_usage["completion_tokens"],
            "total_token_count": token_usage["total_tokens"],
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

        # Reasoning effort level (caller-supplied, forwarded verbatim)
        completion_args.update(extract_effort_field(usage_metadata))
        completion_args.update(extract_prompt_context_fields(usage_metadata))

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


NATIVE_COMPLETIONS_MODULE = "perplexity.resources.chat.completions"
NATIVE_COMPLETIONS_WRAPPERS = (
    ("CompletionsResource", perplexity_create_wrapper),
    ("AsyncCompletionsResource", async_perplexity_create_wrapper),
)


def patch_native_client() -> None:
    for class_name, wrapper in NATIVE_COMPLETIONS_WRAPPERS:
        wrap_registered(
            f"{NATIVE_COMPLETIONS_MODULE}.{class_name}.create",
            NATIVE_COMPLETIONS_MODULE,
            f"{class_name}.create",
            wrapper,
        )


patch_native_client()
