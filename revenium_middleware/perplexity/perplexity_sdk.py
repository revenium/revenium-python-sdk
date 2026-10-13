"""
Middleware for native Perplexity SDK.

This module provides metering support for the native perplexity-py SDK.
"""
import contextlib
import datetime
import logging
import uuid
from types import SimpleNamespace
from typing import Dict, Any

from revenium_middleware import (
    client,
    get_client,
    run_async_in_thread,
    shutdown_event,
    merge_metadata,
)
from revenium_middleware._core import submit_ai_event
from revenium_middleware._core.log_sanitize import sanitize_for_logging
from revenium_middleware._core.call_ownership import PERPLEXITY, with_callback_metadata
from revenium_middleware._core.enforcement import check_enforcement, check_unmetered_call
from revenium_middleware._core.failed_calls import ERROR_STOP_REASON, exception_in_flight, failure_of
from revenium_middleware._core.metering import current_client
from revenium_middleware._core.fields import (
    extract_org_and_product,
    extract_common_metadata,
    extract_effort_field,
    extract_prompt_context_fields,
    extract_agentic_job_fields,
    extract_skill_fields,
    merge_extra_body,
)
from revenium_middleware._core.perplexity_cost import extract_reported_cost

from ._metering_scope import metering_skipped, take_usage_metadata
from .patching import wrap_registered, wrap_strip_only
from .provider import get_provider_metadata, Provider
from .streaming import AsyncMeteredStream, meter_call
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

PERPLEXITY_PROVIDER = get_provider_metadata(Provider.PERPLEXITY)["provider"]
_REFUSED_RESPONSE = SimpleNamespace(usage=SimpleNamespace(prompt_tokens=0, completion_tokens=0, total_tokens=0),
                                    choices=[])


class _NativeCall:
    """The request-side facts of one native chat completion call."""

    def __init__(self, kwargs):
        self._api_metadata = take_usage_metadata(kwargs)
        self.usage_metadata = merge_metadata(self._api_metadata)
        self.model = kwargs.get('model', 'sonar')
        self.is_streaming = bool(kwargs.get('stream', False))
        self.request_time_dt = datetime.datetime.now(datetime.timezone.utc)
        self.metering_client = current_client()
        self.transaction_id = f"perplexity-sdk-{uuid.uuid4().hex}"

    def adopt_callback_metadata(self) -> None:
        """Lay the call's metadata over an attached LangChain callback's, in
        the context the call is made in."""
        self.usage_metadata = merge_metadata(with_callback_metadata(PERPLEXITY, self._api_metadata))

    def check_caps(self) -> None:
        check_enforcement(self.usage_metadata, model=self.model, provider=PERPLEXITY_PROVIDER)

    def meter(self, response, failure=None) -> None:
        run_async_in_thread(
            send_perplexity_metering_data(
                response=response,
                model=self.model,
                request_time_dt=self.request_time_dt,
                response_time_dt=datetime.datetime.now(datetime.timezone.utc),
                transaction_id=getattr(response, "id", None) or self.transaction_id,
                usage_metadata=self.usage_metadata,
                is_streaming=self.is_streaming,
                failure=failure,
            )
        )


@contextlib.contextmanager
def _recording_failure(call: _NativeCall):
    """Send one ERROR record for a call Perplexity refused, then let the
    caller's exception through unchanged."""
    outer = exception_in_flight()
    try:
        yield
    except Exception as error:
        failure = failure_of(error, outer)
        if failure is not None:
            try:
                call.meter(_REFUSED_RESPONSE, failure)
            except Exception as metering_error:
                logger.warning("Could not record a failed Perplexity call: %s", metering_error)
        raise


def _check_unmetered_caps(kwargs, api_metadata) -> None:
    check_unmetered_call(api_metadata, model=kwargs.get('model', 'sonar'), provider=PERPLEXITY_PROVIDER,
                         callback_transport=PERPLEXITY)


def perplexity_create_wrapper(wrapped, instance, args, kwargs):
    if metering_skipped():
        api_metadata = take_usage_metadata(kwargs)
        _check_unmetered_caps(kwargs, api_metadata)
        return wrapped(*args, **kwargs)

    call = _NativeCall(kwargs)
    call.adopt_callback_metadata()
    call.check_caps()
    with _recording_failure(call):
        response = wrapped(*args, **kwargs)
    return meter_call(call, response, PERPLEXITY)


def async_perplexity_create_wrapper(wrapped, instance, args, kwargs):
    if metering_skipped():
        api_metadata = take_usage_metadata(kwargs)

        async def invoke_unmetered():
            _check_unmetered_caps(kwargs, api_metadata)
            return await wrapped(*args, **kwargs)

        return invoke_unmetered()

    call = _NativeCall(kwargs)

    async def invoke():
        call.adopt_callback_metadata()
        call.check_caps()
        with _recording_failure(call):
            response = await wrapped(*args, **kwargs)
        return meter_call(call, response, PERPLEXITY, AsyncMeteredStream)

    return invoke()


def perplexity_responses_create_wrapper(wrapped, instance, args, kwargs):
    """Checks the cap on a Responses API call and sends no record: a LangChain
    call keeps the callback's one record, which this peek leaves in place."""
    api_metadata = take_usage_metadata(kwargs)
    _check_unmetered_caps(kwargs, api_metadata)
    return wrapped(*args, **kwargs)


async def async_perplexity_responses_create_wrapper(wrapped, instance, args, kwargs):
    api_metadata = take_usage_metadata(kwargs)
    _check_unmetered_caps(kwargs, api_metadata)
    return await wrapped(*args, **kwargs)


def strip_usage_metadata_wrapper(wrapped, instance, args, kwargs):
    """A call the SDK does not meter still sends Perplexity no Revenium metadata."""
    take_usage_metadata(kwargs)
    return wrapped(*args, **kwargs)


async def send_perplexity_metering_data(
    response,
    model: str,
    request_time_dt: datetime.datetime,
    response_time_dt: datetime.datetime,
    transaction_id: str,
    usage_metadata: Dict[str, Any],
    is_streaming: bool,
    failure=None,
):
    """
    Send metering data to Revenium for native Perplexity SDK.

    This function extracts usage information from the Perplexity response
    and sends it to Revenium's metering API. ``failure`` marks the record of
    a call Perplexity refused.
    """
    if current_client() is None:
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
        stop_reason = ERROR_STOP_REASON if failure else get_stop_reason(finish_reason)

        duration_ms = int((response_time_dt - request_time_dt).total_seconds() * 1000)

        # Get provider metadata
        provider_metadata = get_provider_metadata(Provider.PERPLEXITY)

        # Build trace fields
        trace_fields = build_trace_fields(usage_metadata)
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
        completion_args.update(extract_reported_cost(response))
        if failure is not None:
            completion_args.update(failure.record_fields())

        organization_name, product_name = extract_org_and_product(usage_metadata)
        if organization_name:
            completion_args["organization_name"] = organization_name
        if product_name:
            completion_args["product_name"] = product_name

        meta = extract_common_metadata(usage_metadata)
        quality_score = meta.pop("response_quality_score")
        for key, value in meta.items():
            if value:
                completion_args[key] = value
        if quality_score is not None:
            completion_args["response_quality_score"] = quality_score

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
        completion_args.update(extract_skill_fields(usage_metadata))

        agentic_fields = extract_agentic_job_fields(usage_metadata)
        extra_body = merge_extra_body(None, agentic_fields)

        if extra_body:
            completion_args["extra_body"] = extra_body

        logger.debug("Sending metering data to Revenium: %s", sanitize_for_logging(completion_args))
        result = submit_ai_event("completion", completion_args,
                                 idempotency_key=failure.idempotency_key if failure else None)
        logger.debug(f"Metering call result: {result}")

    except Exception as e:
        if not shutdown_event.is_set():
            logger.warning(f"Error in metering call: {str(e)}")


NATIVE_COMPLETIONS_MODULE = "perplexity.resources.chat.completions"
NATIVE_COMPLETIONS_WRAPPERS = (
    ("CompletionsResource", perplexity_create_wrapper),
    ("AsyncCompletionsResource", async_perplexity_create_wrapper),
)

NATIVE_RESPONSES_MODULE = "perplexity.resources.responses.responses"
NATIVE_RESPONSES_WRAPPERS = (
    ("ResponsesResource", perplexity_responses_create_wrapper),
    ("AsyncResponsesResource", async_perplexity_responses_create_wrapper),
)


NATIVE_UNMETERED_RESOURCES = (
    ("perplexity.resources.search", "SearchResource"),
    ("perplexity.resources.embeddings", "EmbeddingsResource"),
    ("perplexity.resources.contextualized_embeddings", "ContextualizedEmbeddingsResource"),
    ("perplexity.resources.async_.chat.completions", "CompletionsResource"),
    ("perplexity.resources.browser.sessions", "SessionsResource"),
)


def _wrap_resources(module: str, wrappers, wrap=wrap_registered) -> None:
    for class_name, wrapper in wrappers:
        wrap(f"{module}.{class_name}.create", module, f"{class_name}.create", wrapper)


def patch_native_client() -> None:
    _wrap_resources(NATIVE_COMPLETIONS_MODULE, NATIVE_COMPLETIONS_WRAPPERS)
    _wrap_resources(NATIVE_RESPONSES_MODULE, NATIVE_RESPONSES_WRAPPERS)
    for module, class_name in NATIVE_UNMETERED_RESOURCES:
        _wrap_resources(module, ((class_name, strip_usage_metadata_wrapper),
                                 (f"Async{class_name}", strip_usage_metadata_wrapper)), wrap_strip_only)


patch_native_client()
