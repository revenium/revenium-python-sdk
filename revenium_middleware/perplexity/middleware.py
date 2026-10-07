"""
Revenium Middleware for Perplexity AI

This module provides automatic metering and tracking for Perplexity AI API
calls. It uses wrapt to patch the OpenAI client methods and send usage data
to Revenium.
"""
import datetime
import logging
import uuid
from typing import Dict, Any, Optional
from enum import Enum

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
from .provider import Provider, detect_provider, get_provider_metadata
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

logger = logging.getLogger("revenium_middleware.perplexity")


class OperationType(str, Enum):
    """Operation types for AI API calls."""
    CHAT = "CHAT"
    GENERATE = "GENERATE"
    EMBED = "EMBED"
    OTHER = "OTHER"


def get_stop_reason(finish_reason: Optional[str]) -> str:
    """
    Map Perplexity/OpenAI finish reasons to Revenium stop reasons.

    Valid Revenium stop reasons: END, END_SEQUENCE, TIMEOUT, TOKEN_LIMIT,
    COST_LIMIT, COMPLETION_LIMIT, ERROR, CANCELLED

    Args:
        finish_reason: Finish reason from API response

    Returns:
        Mapped stop reason string
    """
    if not finish_reason:
        return "END"

    reason_map = {
        "stop": "END",
        "length": "TOKEN_LIMIT",
        "content_filter": "ERROR",
        "tool_calls": "END_SEQUENCE",
        "function_call": "END_SEQUENCE",
    }

    return reason_map.get(finish_reason.lower(), "END")


def detect_operation_type(response: Any) -> OperationType:
    """
    Detect the operation type from the response.

    Args:
        response: API response object

    Returns:
        OperationType enum value
    """
    # For Perplexity, it's primarily chat completions
    if hasattr(response, 'choices') and response.choices:
        choice = response.choices[0]
        if hasattr(choice, 'message'):
            # Check for tool calls
            if hasattr(choice.message, 'tool_calls') and choice.message.tool_calls:
                return OperationType.OTHER  # Could be TOOL_CALL if we add it
            return OperationType.CHAT

    return OperationType.CHAT


def extract_token_usage(response: Any) -> Dict[str, int]:
    """
    Extract token usage from API response.

    Args:
        response: API response object

    Returns:
        Dictionary with token counts
    """
    usage = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }

    if hasattr(response, 'usage') and response.usage:
        usage["prompt_tokens"] = getattr(response.usage, 'prompt_tokens', 0)
        usage["completion_tokens"] = getattr(response.usage, 'completion_tokens', 0)
        usage["total_tokens"] = getattr(response.usage, 'total_tokens', 0)

    return usage


def build_trace_fields() -> Dict[str, Any]:
    """
    Build trace visualization fields from environment variables.

    Returns:
        Dictionary with trace fields
    """
    fields = {}

    # Add optional trace fields if available
    if env := get_environment():
        fields["environment"] = env
    if region := get_region():
        fields["region"] = region
    if alias := get_credential_alias():
        fields["credential_alias"] = alias
    if trace_type := get_trace_type():
        fields["trace_type"] = trace_type
    if trace_name := get_trace_name():
        fields["trace_name"] = trace_name
    if parent_id := get_parent_transaction_id():
        fields["parent_transaction_id"] = parent_id
    if txn_name := get_transaction_name():
        fields["transaction_name"] = txn_name

    # Always include retry number (defaults to 0)
    fields["retry_number"] = get_retry_number()

    return fields


def send_metering_data(
    response: Any,
    request_time_dt: datetime.datetime,
    usage_metadata: Dict[str, Any],
    model: str,
    provider: Provider,
    is_streaming: bool = False,
    transaction_id: Optional[str] = None,
) -> None:
    """
    Send metering data to Revenium asynchronously.

    Args:
        response: API response object
        request_time_dt: Request start time
        usage_metadata: User-provided metadata
        model: Model name
        provider: Provider enum
        is_streaming: Whether this is a streaming response
        transaction_id: Transaction ID for tracking
    """
    if get_client() is None:
        return  # metering disabled (no API key configured)
    response_time_dt = datetime.datetime.now(datetime.timezone.utc)
    request_duration = (response_time_dt - request_time_dt).total_seconds() * 1000

    async def metering_call():
        try:
            # Generate transaction ID if not provided
            if transaction_id is None:
                txn_id = getattr(response, 'id', str(uuid.uuid4()))
            else:
                txn_id = transaction_id

            # Extract token usage
            token_usage = extract_token_usage(response)

            # Detect operation type
            operation_type = detect_operation_type(response)

            # Get stop reason
            stop_reason = "END"
            if hasattr(response, 'choices') and response.choices:
                finish_reason = getattr(response.choices[0], 'finish_reason', None)
                stop_reason = get_stop_reason(finish_reason)

            # Get provider metadata
            provider_metadata = get_provider_metadata(provider)

            # Build trace fields
            trace_fields = build_trace_fields()
            if ticket_id := get_ticket_id(usage_metadata):
                trace_fields["ticket_id"] = ticket_id
            if agent_version := get_agent_version(usage_metadata):
                trace_fields["agent_version"] = agent_version

            # Build completion args
            completion_args = {
                # Required fields
                "model": model,
                "provider": provider_metadata["provider"],
                "input_token_count": token_usage["prompt_tokens"],
                "output_token_count": token_usage["completion_tokens"],
                "total_token_count": token_usage["total_tokens"],
                "request_duration": int(request_duration),
                "request_time": request_time_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "response_time": response_time_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "completion_start_time": response_time_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "transaction_id": txn_id,
                "stop_reason": stop_reason,
                "is_streamed": is_streaming,
                "cost_type": "AI",
                "operation_type": operation_type.value,
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

            logger.debug(f"Sending metering data: {completion_args}")
            result = submit_ai_event("completion", completion_args)
            logger.debug(f"Metering call result: {result}")

        except Exception as e:
            if not shutdown_event.is_set():
                logger.warning(f"Error in metering call: {str(e)}")

    # Run async in background thread
    thread = run_async_in_thread(metering_call())
    logger.debug(f"Metering thread started: {thread}")


def _perplexity_client(instance) -> Optional[Any]:
    """The OpenAI client behind ``instance`` when its base URL is Perplexity's, else None."""
    client_instance = getattr(instance, '_client', None)
    base_url = getattr(client_instance, 'base_url', None) if client_instance else None
    if base_url and "perplexity" in str(base_url).lower():
        return client_instance
    return None


class _CompatibleCall:
    """The request-side facts of one Perplexity call made through the OpenAI client."""

    def __init__(self, client_instance, kwargs):
        api_metadata = kwargs.pop("usage_metadata", {})

        extra_body = kwargs.get('extra_body', {})
        if isinstance(extra_body, dict) and 'usage_metadata' in extra_body:
            extra_metadata = extra_body.pop('usage_metadata', {})
            api_metadata = {**extra_metadata, **api_metadata}

        self.usage_metadata = merge_metadata(api_metadata)
        self.provider = detect_provider(client=client_instance, base_url=client_instance.base_url)
        self.model = kwargs.get('model', 'unknown')
        self.is_streaming = kwargs.get('stream', False)
        self.request_time_dt = datetime.datetime.now(datetime.timezone.utc)
        self.transaction_id = f"perplexity-{self.request_time_dt.timestamp()}"

    def meter(self, response) -> None:
        send_metering_data(
            response,
            self.request_time_dt,
            self.usage_metadata,
            self.model,
            self.provider,
            is_streaming=bool(self.is_streaming),
            transaction_id=self.transaction_id,
        )


def create_wrapper(wrapped, instance, args, kwargs):
    if metering_skipped():
        return wrapped(*args, **kwargs)

    # The Perplexity API is OpenAI-compatible, so this wrapper patches the same
    # openai.resources.chat.completions.Completions.create slot as the OpenAI
    # middleware. Without this guard, calls made by a plain OpenAI client would
    # be metered and tagged as PERPLEXITY. Defer non-Perplexity calls to the
    # next wrapper (or original).
    client_instance = _perplexity_client(instance)
    if client_instance is None:
        return wrapped(*args, **kwargs)

    call = _CompatibleCall(client_instance, kwargs)
    response = wrapped(*args, **kwargs)
    if call.is_streaming:
        return MeteredStream(response, StreamMeter(call.meter))
    call.meter(response)
    return response


def async_create_wrapper(wrapped, instance, args, kwargs):
    if metering_skipped():
        return wrapped(*args, **kwargs)

    # Same guard as create_wrapper: the OpenAI middleware wraps this
    # AsyncCompletions.create slot too and defers Perplexity-bound calls here.
    client_instance = _perplexity_client(instance)
    if client_instance is None:
        return wrapped(*args, **kwargs)

    call = _CompatibleCall(client_instance, kwargs)

    async def invoke():
        response = await wrapped(*args, **kwargs)
        if call.is_streaming:
            return AsyncMeteredStream(response, StreamMeter(call.meter))
        call.meter(response)
        return response

    return invoke()


OPENAI_COMPLETIONS_MODULE = "openai.resources.chat.completions"

wrap_registered(f"perplexity:{OPENAI_COMPLETIONS_MODULE}.Completions.create",
                OPENAI_COMPLETIONS_MODULE, "Completions.create", create_wrapper)
wrap_registered(f"perplexity:{OPENAI_COMPLETIONS_MODULE}.AsyncCompletions.create",
                OPENAI_COMPLETIONS_MODULE, "AsyncCompletions.create", async_create_wrapper)
