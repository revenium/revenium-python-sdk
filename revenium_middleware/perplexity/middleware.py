"""
Revenium Middleware for Perplexity AI

This module provides automatic metering and tracking for Perplexity AI API
calls. It uses wrapt to patch the OpenAI client methods and send usage data
to Revenium.
"""
import contextlib
import datetime
import importlib
import logging
import uuid
from contextvars import ContextVar
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
from revenium_middleware._core.log_sanitize import sanitize_for_logging
from revenium_middleware._core.call_ownership import OPENAI, call_owned_by_outer_wrapper, with_callback_metadata
from revenium_middleware._core.enforcement import check_enforcement, check_unmetered_call
from revenium_middleware._core.metering import current_client
from revenium_middleware._core.parse_failures import metering_failed_parse
from revenium_middleware._core.patch_registry import is_patched
from revenium_middleware._core.responses_status import responses_stop_reason
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
from .patching import wrap_registered
from .provider import Provider, detect_provider, get_provider_metadata
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
        usage["prompt_tokens"] = _token_count(response.usage, 'prompt_tokens', 'input_tokens')
        usage["completion_tokens"] = _token_count(response.usage, 'completion_tokens', 'output_tokens')
        usage["total_tokens"] = getattr(response.usage, 'total_tokens', 0)

    return usage


def _token_count(usage: Any, chat_name: str, responses_name: str) -> int:
    count = getattr(usage, chat_name, None)
    return getattr(usage, responses_name, 0) if count is None else count


def build_trace_fields(usage_metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Build trace visualization fields from the call's metadata, then environment variables.

    Returns:
        Dictionary with trace fields
    """
    fields = {}

    # Add optional trace fields if available
    if env := get_environment(usage_metadata):
        fields["environment"] = env
    if region := get_region(usage_metadata):
        fields["region"] = region
    if alias := get_credential_alias(usage_metadata):
        fields["credential_alias"] = alias
    if trace_type := get_trace_type(usage_metadata):
        fields["trace_type"] = trace_type
    if trace_name := get_trace_name(usage_metadata):
        fields["trace_name"] = trace_name
    if parent_id := get_parent_transaction_id(usage_metadata):
        fields["parent_transaction_id"] = parent_id
    if txn_name := get_transaction_name(usage_metadata):
        fields["transaction_name"] = txn_name

    # Always include retry number (defaults to 0)
    fields["retry_number"] = get_retry_number(usage_metadata)

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
    if current_client() is None:
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

            if not getattr(response, 'usage', None):
                logger.warning("No usage data found in response")
            token_usage = extract_token_usage(response)

            # Detect operation type
            operation_type = detect_operation_type(response)

            # Get stop reason
            stop_reason = "END"
            if hasattr(response, 'choices') and response.choices:
                finish_reason = getattr(response.choices[0], 'finish_reason', None)
                stop_reason = get_stop_reason(finish_reason)
            else:
                stop_reason = responses_stop_reason(response)

            # Get provider metadata
            provider_metadata = get_provider_metadata(provider)

            # Build trace fields
            trace_fields = build_trace_fields(usage_metadata)
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
            completion_args.update(extract_reported_cost(response))

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

            logger.debug("Sending metering data: %s", sanitize_for_logging(completion_args))
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
        api_metadata = take_usage_metadata(kwargs)
        self._api_metadata = api_metadata
        self.usage_metadata = merge_metadata(api_metadata)
        self.provider = detect_provider(client=client_instance, base_url=client_instance.base_url)
        self.model = kwargs.get('model', 'unknown')
        self.is_streaming = kwargs.get('stream', False)
        self.request_time_dt = datetime.datetime.now(datetime.timezone.utc)
        self.metering_client = current_client()
        self.transaction_id = f"perplexity-{uuid.uuid4().hex}"

    def adopt_callback_metadata(self) -> None:
        """Lay the call's metadata over an attached LangChain callback's, in
        the context the call is made in. The call sits in the OpenAI client's
        slot, so a LangChain model reaches it as an OpenAI transport call."""
        self.usage_metadata = merge_metadata(with_callback_metadata(OPENAI, self._api_metadata))

    def check_caps(self) -> None:
        check_enforcement(self.usage_metadata, model=self.model,
                          provider=get_provider_metadata(self.provider)["provider"])

    def meter(self, response) -> None:
        send_metering_data(
            response,
            self.request_time_dt,
            self.usage_metadata,
            self.model,
            self.provider,
            is_streaming=bool(self.is_streaming),
            transaction_id=getattr(response, "id", None) or self.transaction_id,
        )


def _check_unmetered_caps(client_instance, kwargs, api_metadata) -> None:
    """A Perplexity call this wrapper does not meter is still refused by a breached cap."""
    provider = detect_provider(client=client_instance, base_url=client_instance.base_url)
    check_unmetered_call(api_metadata, model=kwargs.get('model'), provider=get_provider_metadata(provider)["provider"],
                         callback_transport=OPENAI)


def _unmetered_metadata(kwargs) -> Dict[str, Any]:
    """The metadata of a call this wrapper leaves to another: an outer wrapper that meters it
    gets the request with its metadata, and a selectively unmetered one without."""
    if call_owned_by_outer_wrapper():
        api_metadata = kwargs.get("usage_metadata")
        return dict(api_metadata) if isinstance(api_metadata, dict) else {}
    return take_usage_metadata(kwargs)


def _pass_on_unmetered(client_instance, kwargs) -> None:
    _check_unmetered_caps(client_instance, kwargs, _unmetered_metadata(kwargs))


def _left_unmetered() -> bool:
    return call_owned_by_outer_wrapper() or metering_skipped()


_parse_call_in_progress: ContextVar[bool] = ContextVar("revenium_perplexity_parse_call_in_progress", default=False)


@contextlib.contextmanager
def _metered_by_parse():
    """Mark the enclosed provider call as metered by a ``parse()`` wrap, so a
    ``create()`` that an OpenAI release routes ``parse()`` through stays
    unmetered instead of sending a second record."""
    token = _parse_call_in_progress.set(True)
    try:
        yield
    finally:
        _parse_call_in_progress.reset(token)


def create_wrapper(wrapped, instance, args, kwargs):
    # The Perplexity API is OpenAI-compatible, so this wrapper patches the same
    # openai.resources.chat.completions.Completions.create slot as the OpenAI
    # middleware. Without this guard, calls made by a plain OpenAI client would
    # be metered and tagged as PERPLEXITY. Defer non-Perplexity calls, metadata
    # and all, to the next wrapper (or original).
    client_instance = _perplexity_client(instance)
    if client_instance is None or _parse_call_in_progress.get():
        return wrapped(*args, **kwargs)

    if _left_unmetered():
        _pass_on_unmetered(client_instance, kwargs)
        return wrapped(*args, **kwargs)

    call = _CompatibleCall(client_instance, kwargs)
    call.adopt_callback_metadata()
    call.check_caps()
    return meter_call(call, wrapped(*args, **kwargs), OPENAI)


def async_create_wrapper(wrapped, instance, args, kwargs):
    # Same guard as create_wrapper: the OpenAI middleware wraps this
    # AsyncCompletions.create slot too and defers Perplexity-bound calls here.
    client_instance = _perplexity_client(instance)
    if client_instance is None or _parse_call_in_progress.get():
        return wrapped(*args, **kwargs)

    if _left_unmetered():
        api_metadata = _unmetered_metadata(kwargs)

        async def invoke_unmetered():
            _check_unmetered_caps(client_instance, kwargs, api_metadata)
            return await wrapped(*args, **kwargs)

        return invoke_unmetered()

    call = _CompatibleCall(client_instance, kwargs)

    async def invoke():
        call.adopt_callback_metadata()
        call.check_caps()
        return meter_call(call, await wrapped(*args, **kwargs), OPENAI, AsyncMeteredStream)

    return invoke()


def _meter_parsed_response(response, call) -> None:
    meter_call(call, response, OPENAI)


def parse_wrapper(wrapped, instance, args, kwargs):
    """``parse()`` posts its request itself rather than calling ``create()``,
    so a Perplexity-bound structured-output call is checked and metered here."""
    client_instance = _perplexity_client(instance)
    if client_instance is None or _parse_call_in_progress.get():
        return wrapped(*args, **kwargs)

    if _left_unmetered():
        _pass_on_unmetered(client_instance, kwargs)
        with _metered_by_parse():
            return wrapped(*args, **kwargs)

    call = _CompatibleCall(client_instance, kwargs)
    call.adopt_callback_metadata()
    call.check_caps()
    with _metered_by_parse(), metering_failed_parse(
        call, _meter_parsed_response, provider="Perplexity", log=logger
    ):
        response = wrapped(*args, **kwargs)
    return meter_call(call, response, OPENAI)


def async_parse_wrapper(wrapped, instance, args, kwargs):
    client_instance = _perplexity_client(instance)
    if client_instance is None or _parse_call_in_progress.get():
        return wrapped(*args, **kwargs)

    if _left_unmetered():
        api_metadata = _unmetered_metadata(kwargs)

        async def invoke_unmetered():
            _check_unmetered_caps(client_instance, kwargs, api_metadata)
            with _metered_by_parse():
                return await wrapped(*args, **kwargs)

        return invoke_unmetered()

    call = _CompatibleCall(client_instance, kwargs)

    async def invoke():
        call.adopt_callback_metadata()
        call.check_caps()
        with _metered_by_parse(), metering_failed_parse(
            call, _meter_parsed_response, provider="Perplexity", log=logger
        ):
            response = await wrapped(*args, **kwargs)
        return meter_call(call, response, OPENAI, AsyncMeteredStream)

    return invoke()


OPENAI_COMPLETIONS_MODULE = "openai.resources.chat.completions"
OPENAI_RESPONSES_MODULE = "openai.resources.responses"
OPENAI_RESPONSES_CREATE = f"{OPENAI_RESPONSES_MODULE}.Responses.create"
OPENAI_ASYNC_RESPONSES_CREATE = f"{OPENAI_RESPONSES_MODULE}.AsyncResponses.create"
OPENAI_RESPONSES_PARSE = f"{OPENAI_RESPONSES_MODULE}.Responses.parse"
OPENAI_ASYNC_RESPONSES_PARSE = f"{OPENAI_RESPONSES_MODULE}.AsyncResponses.parse"
OPENAI_EMBEDDINGS_MODULE = "openai.resources.embeddings"
OPENAI_EMBEDDINGS_CREATE = f"{OPENAI_EMBEDDINGS_MODULE}.Embeddings.create"
OPENAI_ASYNC_EMBEDDINGS_CREATE = f"{OPENAI_EMBEDDINGS_MODULE}.AsyncEmbeddings.create"


def embeddings_strip_wrapper(wrapped, instance, args, kwargs):
    """An embeddings call to Perplexity is not metered here, but its Revenium metadata
    must still not reach Perplexity."""
    if _perplexity_client(instance) is not None:
        take_usage_metadata(kwargs)
    return wrapped(*args, **kwargs)


def _unless_openai_integration_wraps(openai_wrap: str, perplexity_wrapper):
    # The OpenAI integration takes a Perplexity-bound Responses or embeddings call's metadata itself
    # (a Responses call is metered labelled PERPLEXITY), so this wrap only acts when that one is not installed.
    def wrapper(wrapped, instance, args, kwargs):
        if is_patched(openai_wrap):
            return wrapped(*args, **kwargs)
        return perplexity_wrapper(wrapped, instance, args, kwargs)
    return wrapper


def _openai_defines(module: str, attribute_path: str) -> bool:
    try:
        target = importlib.import_module(module)
        for name in attribute_path.split('.'):
            target = getattr(target, name)
    except (ImportError, AttributeError):
        return False
    return True


wrap_registered(f"perplexity:{OPENAI_COMPLETIONS_MODULE}.Completions.create",
                OPENAI_COMPLETIONS_MODULE, "Completions.create", create_wrapper)
wrap_registered(f"perplexity:{OPENAI_COMPLETIONS_MODULE}.AsyncCompletions.create",
                OPENAI_COMPLETIONS_MODULE, "AsyncCompletions.create", async_create_wrapper)
for _attribute, _wrapper in (("Completions.parse", parse_wrapper), ("AsyncCompletions.parse", async_parse_wrapper)):
    if _openai_defines(OPENAI_COMPLETIONS_MODULE, _attribute):
        wrap_registered(f"perplexity:{OPENAI_COMPLETIONS_MODULE}.{_attribute}",
                        OPENAI_COMPLETIONS_MODULE, _attribute, _wrapper)
for _openai_wrap, _wrapper in ((OPENAI_RESPONSES_CREATE, create_wrapper),
                               (OPENAI_ASYNC_RESPONSES_CREATE, async_create_wrapper),
                               (OPENAI_RESPONSES_PARSE, parse_wrapper),
                               (OPENAI_ASYNC_RESPONSES_PARSE, async_parse_wrapper)):
    _attribute = _openai_wrap[len(OPENAI_RESPONSES_MODULE) + 1:]
    if _openai_defines(OPENAI_RESPONSES_MODULE, _attribute):
        wrap_registered(f"perplexity:{_openai_wrap}", OPENAI_RESPONSES_MODULE, _attribute,
                        _unless_openai_integration_wraps(_openai_wrap, _wrapper))
for _openai_wrap in (OPENAI_EMBEDDINGS_CREATE, OPENAI_ASYNC_EMBEDDINGS_CREATE):
    _attribute = _openai_wrap[len(OPENAI_EMBEDDINGS_MODULE) + 1:]
    wrap_registered(f"perplexity:{_openai_wrap}", OPENAI_EMBEDDINGS_MODULE, _attribute,
                    _unless_openai_integration_wraps(_openai_wrap, embeddings_strip_wrapper))
