import logging
import datetime
import wrapt
import time
import functools
import importlib
import contextvars
import threading
import queue
import uuid
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any, Optional, Callable, Dict

from anthropic.types import Message

# Import our provider detection and Bedrock adapter
from .provider import Provider, detect_provider, get_provider_metadata
from .bedrock_adapter import (
    bedrock_invoke, create_bedrock_payload, create_anthropic_response, BedrockStreamWrapper,
    BedrockError, BedrockValidationError, BedrockInvokeError, BedrockStreamError, _model_id, anthropic_model_name,
    bedrock_reported_model_id,
)
from .bedrock_model_names import recorded_bedrock_model_id
from .bedrock_transport import cached_application_profile, resolve_application_profile
from .inference_profiles import NOT_CAPPED_BY_MODEL, AnthropicBedrockCredentials, is_application_profile_arn

try:
    from anthropic import AnthropicBedrock
except ImportError:  # anthropic releases that predate the Bedrock client
    AnthropicBedrock = None

# Import decorator support and metering client from core package
from revenium_middleware import client, run_async_in_thread, shutdown_event, merge_metadata
from revenium_middleware._core import submit_ai_event
from revenium_middleware._core.log_sanitize import sanitize_for_logging
from revenium_middleware._core.call_context import CallContext
from revenium_middleware._core.metering import current_client, recorded_under
from revenium_middleware._core.cache_tokens import extract_cache_creation_ttl_counts
from revenium_middleware._core.subscriber import extract_subscriber_from_metadata
from revenium_middleware._core.fields import extract_org_and_product, extract_common_metadata, extract_agentic_job_fields, extract_effort_field, extract_prompt_context_fields, extract_skill_fields, merge_extra_body
from revenium_middleware._core.config import is_selective_metering_enabled
from revenium_middleware._core.context import is_inside_decorated_function, next_idempotency_key
from revenium_middleware._core.selective_metering import reported_if_unmetered
from revenium_middleware._core.enforcement import (
    ModelKnownByAliases, check_enforcement, check_unmetered_call, is_circuit_breaker_enabled)
from revenium_middleware._core.patch_registry import wrap_once
from revenium_middleware._core.call_ownership import ANTHROPIC, claim_call_for_transport, with_callback_metadata

# Import trace visualization functions
from .trace_fields import (
    get_environment, get_region, get_credential_alias,
    get_trace_type, get_trace_name, get_parent_transaction_id,
    get_transaction_name, get_retry_number, get_ticket_id,
    get_agent_version,
    detect_operation_type, detect_vision_content
)

# Import configuration and prompt capture utilities
from .config import Config
from .stream_create import (
    StreamUsageState, RawStreamMeteringWrapper, AsyncRawStreamMeteringWrapper,
    meter_stream_helper_events, meter_async_stream_helper_events,
)
from revenium_middleware._core.usage_iterations import submit_iteration_records
from revenium_middleware._core.failed_calls import (
    ERROR_STOP_REASON, CallFailure, exception_in_flight, failed_transaction_id, failure_of,
)
from .raw_response import (
    is_raw_response, is_body_read, parsed_message, async_parsed_message,
    tap_sse_body, tap_async_sse_body, tap_message_body, tap_async_message_body,
)
from .prompt_extractor import captured_prompt_fields, extract_prompt_data_if_enabled

logger = logging.getLogger("revenium_middleware.extension")

_STOP_REASONS = {
    "end_turn": "END",
    "tool_use": "END_SEQUENCE",
    "stop_sequence": "END_SEQUENCE",
    "max_tokens": "TOKEN_LIMIT",
    "content_filter": "ERROR",
}

# Define usage context for thread-safe metadata storage
usage_context = contextvars.ContextVar('usage_metadata', default={})


# Thread-safe metering infrastructure
_metering_lock = threading.RLock()
_metering_queue = queue.Queue(maxsize=1000)  # Prevent memory issues


def _get_thread_safe_client():
    return current_client()


def _safe_run_async_in_thread(coro_func: Callable, *args, **kwargs):
    try:
        from revenium_middleware import run_async_in_thread, shutdown_event

        if shutdown_event.is_set():
            logger.warning("Skipping async operation during shutdown")
            return None

        with _metering_lock:
            coroutine = coro_func(*args, **kwargs)

        thread = run_async_in_thread(coroutine)
        logger.debug(f"Started thread-safe async operation: {thread}")
        return thread
    except Exception as e:
        logger.warning(f"Error in thread-safe async operation: {e}")
        return None


def _signs_its_own_bedrock_requests(client_instance):
    """An AnthropicBedrock client sends and signs Bedrock calls itself with its own credentials.

    Any other client that only points at a Bedrock URL cannot sign, so the
    middleware's boto3 shortcut is the only way its call reaches Bedrock.
    """
    return AnthropicBedrock is not None and isinstance(client_instance, AnthropicBedrock)


def _use_bedrock_model_id(kwargs):
    model = kwargs.get("model")
    if isinstance(model, str):
        kwargs["model"] = _model_id(model)


def _recorded_model(reported_model, request_kwargs, provider):
    """The model a call is recorded under.

    Bedrock answers with the bare model id, which drops the regional profile
    AWS prices the call by, so a Bedrock Claude call is recorded under the id
    it was sent with, as the boto3 transport records it, a foundation-model or
    inference-profile ARN as the model id it names.
    """
    requested = (request_kwargs or {}).get("model")
    if provider != Provider.BEDROCK or not isinstance(requested, str) or "anthropic." not in requested:
        return reported_model
    return recorded_bedrock_model_id(requested)


def _handle_bedrock_request(args, kwargs, usage_metadata, request_time_dt, request_time, region=None):  # pylint: disable=unused-argument
    """
    Handle a Bedrock request by converting parameters and invoking the Bedrock adapter.

    Returns:
        Anthropic-compatible response object
    """
    logger.debug("Handling Bedrock request")

    # Extract parameters from kwargs
    model = kwargs.get("model", "claude-3-sonnet-20240229")
    messages = kwargs.get("messages", [])

    # Create Bedrock payload - exclude 'messages' from kwargs to avoid conflict
    bedrock_kwargs = {k: v for k, v in kwargs.items() if k != "messages"}
    payload = create_bedrock_payload(messages, **bedrock_kwargs)

    # Invoke Bedrock
    text, input_tokens, output_tokens, cache_creation_tokens, cache_read_tokens, cache_creation = bedrock_invoke(
        model, payload, region=region
    )

    # Create Anthropic-compatible response
    response = create_anthropic_response(
        text=text,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        model=model,
        cache_creation_tokens=cache_creation_tokens,
        cache_read_tokens=cache_read_tokens,
        cache_creation=cache_creation,
    )

    # Calculate timing
    response_time_dt = datetime.datetime.now(datetime.timezone.utc)
    response_time = response_time_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    request_duration = (response_time_dt - request_time_dt).total_seconds() * 1000

    # Create metering call for Bedrock (pass kwargs for vision detection)
    _create_bedrock_metering_call(
        response, usage_metadata, request_time, response_time, request_duration, kwargs,
        cache_creation_tokens=cache_creation_tokens, cache_read_tokens=cache_read_tokens
    )

    return response


def _handle_bedrock_stream_request(args, kwargs, usage_metadata, request_time_dt, request_time, region=None):  # pylint: disable=unused-argument
    """
    Handle a Bedrock streaming request by creating a BedrockStreamWrapper.

    Returns:
        BedrockStreamWrapper: Stream wrapper compatible with Anthropic's interface
    """
    logger.debug("Handling Bedrock streaming request")

    # Extract parameters from kwargs
    model = kwargs.get("model", "claude-3-sonnet-20240229")
    messages = kwargs.get("messages", [])

    # Create Bedrock payload - exclude 'messages' from kwargs to avoid conflict
    bedrock_kwargs = {k: v for k, v in kwargs.items() if k != "messages"}
    payload = create_bedrock_payload(messages, **bedrock_kwargs)

    # Create and return BedrockStreamWrapper
    return BedrockStreamWrapper(
        model=model,
        payload=payload,
        messages=messages,
        region=region if region is not None else kwargs.get("region"),
        usage_metadata=usage_metadata,
        request_time_dt=request_time_dt,
        request_time=request_time
    )


def _effort_payload(trace_fields):
    """Spread into a completion payload: an unset level contributes no key.

    ``create_completion`` only drops ``NotGiven`` values when it serializes the
    request, so handing it ``effort=None`` would put ``"effort": null`` on the
    wire. Omitting the keyword entirely keeps the payload byte-identical to
    what integrations sent before BACK-2710.
    """
    effort = trace_fields.get('effort')
    return {} if effort is None else {'effort': effort}


def _submit_main_record(completion_args, idempotency_key):
    if idempotency_key is None:
        return submit_ai_event("completion", completion_args)
    return submit_ai_event("completion", completion_args, idempotency_key=idempotency_key)


def _submit_iteration_records(completion_args, usage_or_iterations, request_kwargs, main_idempotency_key):
    submit_iteration_records(submit_ai_event, completion_args, usage_or_iterations, request_kwargs,
                             main_idempotency_key)


def _extract_trace_fields(usage_metadata, request_body=None):
    """
    Extract trace visualization fields from usage_metadata and environment variables.

    Args:
        usage_metadata: Dictionary containing usage metadata
        request_body: Optional request body for operation type detection

    Returns:
        Dictionary with trace visualization fields
    """
    # Get trace fields (usage_metadata takes precedence over environment variables)
    environment = usage_metadata.get('environment') or get_environment()
    region = usage_metadata.get('region') or get_region()
    credential_alias = (
        usage_metadata.get('credentialAlias') or
        usage_metadata.get('credential_alias') or
        get_credential_alias()
    )
    trace_type = get_trace_type(usage_metadata)
    trace_name = get_trace_name(usage_metadata)
    ticket_id = get_ticket_id(usage_metadata)
    agent_version = get_agent_version(usage_metadata)
    # Reasoning effort is caller-supplied only and forwarded verbatim; absent
    # metadata resolves to None, and _effort_payload() drops the key entirely
    # at the payload sites so nothing is sent.
    effort = extract_effort_field(usage_metadata).get('effort')
    parent_transaction_id = (
        usage_metadata.get('parentTransactionId') or
        usage_metadata.get('parent_transaction_id') or
        get_parent_transaction_id()
    )
    transaction_name = (
        usage_metadata.get('transactionName') or
        usage_metadata.get('transaction_name') or
        get_transaction_name(usage_metadata)
    )
    retry_number = get_retry_number(usage_metadata)

    # Detect operation type and subtype
    request_body = request_body or {}
    operation_info = detect_operation_type(
        'anthropic', '/messages', request_body
    )
    operation_type = operation_info.get('operationType')
    operation_subtype = operation_info.get('operationSubtype')

    # Detect vision content in messages
    messages = request_body.get('messages', [])
    has_vision_content = detect_vision_content(messages)

    return {
        'environment': environment,
        'region': region,
        'credential_alias': credential_alias,
        'trace_type': trace_type,
        'trace_name': trace_name,
        'ticket_id': ticket_id,
        'agent_version': agent_version,
        'effort': effort,
        'prompt_context': extract_prompt_context_fields(usage_metadata),
        'skill': extract_skill_fields(usage_metadata),
        'parent_transaction_id': parent_transaction_id,
        'transaction_name': transaction_name,
        'retry_number': retry_number,
        'operation_type': operation_type,
        'operation_subtype': operation_subtype,
        'has_vision_content': has_vision_content,
    }


def _create_bedrock_metering_call(response, usage_metadata, request_time, response_time, request_duration, request_kwargs=None, cache_creation_tokens=0, cache_read_tokens=0):
    """Create a metering call for Bedrock usage."""
    if _get_thread_safe_client() is None:
        return

    # Get provider metadata
    provider_metadata = get_provider_metadata(Provider.BEDROCK)
    prompt_fields = captured_prompt_fields(request_kwargs, response=response)

    async def metering_call():
        try:
            from revenium_middleware import shutdown_event

            if shutdown_event.is_set():
                logger.warning("Skipping metering call during shutdown")
                return

            logger.debug("Metering call to Revenium for Bedrock completion %s", response.id)

            subscriber = extract_subscriber_from_metadata(usage_metadata)

            # Extract trace visualization fields (pass request kwargs for vision detection)
            trace_fields = _extract_trace_fields(usage_metadata, request_kwargs)

            # Build extra_body for additional fields not in SDK
            extra_body = {}
            if trace_fields.get('has_vision_content'):
                extra_body['hasVisionContent'] = True
            extra_body = merge_extra_body(extra_body, extract_agentic_job_fields(usage_metadata))

            # Extract organization, product, and common metadata with alias support
            organization_name, product_name = _extract_organization_and_product_names(usage_metadata)
            meta = extract_common_metadata(usage_metadata)

            result = submit_ai_event("completion", {
                "cache_creation_token_count": cache_creation_tokens,
                **extract_cache_creation_ttl_counts(response.usage),
                "cache_read_token_count": cache_read_tokens,
                "input_token_cost": None,
                "output_token_cost": None,
                "total_cost": None,
                "output_token_count": response.usage.output_tokens,
                "cost_type": "AI",
                "model": response.model,
                "input_token_count": response.usage.input_tokens,
                "provider": provider_metadata["provider"],
                "model_source": provider_metadata["model_source"],
                "reasoning_token_count": 0,
                "request_time": request_time,
                "response_time": response_time,
                "completion_start_time": response_time,
                "request_duration": int(request_duration),
                "time_to_first_token": int(request_duration),
                "stop_reason": "END",
                "total_token_count": response.usage.total_tokens,
                "transaction_id": response.id,
                "trace_id": meta["trace_id"],
                "task_type": meta["task_type"],
                "subscriber": subscriber if subscriber else None,
                "organization_name": organization_name,
                "subscription_id": meta["subscription_id"],
                "product_name": product_name,
                "agent": meta["agent"],
                "response_quality_score": meta["response_quality_score"],
                "is_streamed": False,
                "operation_type": trace_fields.get('operation_type', 'CHAT'),
                "environment": trace_fields.get('environment'),
                "region": trace_fields.get('region'),
                "credential_alias": trace_fields.get('credential_alias'),
                "trace_type": trace_fields.get('trace_type'),
                "trace_name": trace_fields.get('trace_name'),
                "ticket_id": trace_fields.get('ticket_id'),
                "agent_version": trace_fields.get('agent_version'),
                **_effort_payload(trace_fields),
                **trace_fields['prompt_context'],
                **trace_fields['skill'],
                "parent_transaction_id": trace_fields.get('parent_transaction_id'),
                "transaction_name": trace_fields.get('transaction_name'),
                "retry_number": trace_fields.get('retry_number'),
                "operation_subtype": trace_fields.get('operation_subtype'),
                **prompt_fields,
                "extra_body": extra_body if extra_body else None,
            })
            logger.debug("Bedrock metering call result: %s", result)
        except Exception as e:
            from revenium_middleware import shutdown_event
            if not shutdown_event.is_set():
                logger.warning(f"Error in Bedrock metering call: {str(e)}")
                import traceback
                logger.warning(f"Traceback: {traceback.format_exc()}")

    thread = _safe_run_async_in_thread(metering_call)
    logger.debug("Bedrock metering thread started: %s", thread)


def extract_usage_metadata_and_timing(kwargs: dict, operation_name: str = "operation"):
    """
    Extract usage metadata from kwargs.
    Provides robust error handling for malformed metadata structures.

    Args:
        kwargs: The kwargs dict to extract from (will be modified)
        operation_name: Name of operation for logging (e.g., "create", "stream")

    Returns:
        tuple: (usage_metadata, request_time, request_time_dt)
    """
    # Extract API-level metadata from kwargs
    api_metadata = kwargs.pop("usage_metadata", {})

    # Validate and sanitize API-level metadata
    if not isinstance(api_metadata, dict):
        logger.warning(f"usage_metadata for {operation_name} should be a dict, got {type(api_metadata)}. Using empty dict.")
        api_metadata = {}

    # Merge with decorator metadata (API-level takes precedence)
    usage_metadata = merge_metadata(with_callback_metadata(ANTHROPIC, api_metadata))
    logger.debug("Merged decorator metadata for %s: %s", operation_name, sanitize_for_logging(usage_metadata))

    # Sanitize metadata structure (defensive programming)
    usage_metadata = _sanitize_metadata(usage_metadata, operation_name)

    # Create request timestamp
    request_time_dt = datetime.datetime.now(datetime.timezone.utc)
    request_time = request_time_dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    # Debug logging
    logger.debug("Usage metadata for %s: %s", operation_name, sanitize_for_logging(usage_metadata))

    return usage_metadata, request_time, request_time_dt


def _sanitize_metadata(metadata: dict, operation_name: str, max_depth: int = 5, current_depth: int = 0) -> dict:
    """
    Sanitize metadata structure to prevent issues with deeply nested objects
    or problematic data types that could break metering calls.

    Args:
        metadata: The metadata dict to sanitize
        operation_name: Operation name for logging
        max_depth: Maximum allowed nesting depth
        current_depth: Current recursion depth

    Returns:
        dict: Sanitized metadata
    """
    if current_depth > max_depth:
        logger.warning(f"Metadata for {operation_name} exceeds maximum depth {max_depth}. Truncating.")
        return {}

    if not isinstance(metadata, dict):
        return {}

    sanitized = {}
    for key, value in metadata.items():
        # Ensure key is a string
        if not isinstance(key, str):
            key = str(key)

        # Sanitize value based on type
        if isinstance(value, dict):
            sanitized[key] = _sanitize_metadata(value, operation_name, max_depth, current_depth + 1)
        elif isinstance(value, (list, tuple)):
            sanitized[key] = type(value)(
                _sanitize_metadata(item, operation_name, max_depth, current_depth + 1) if isinstance(item, dict)
                else item if isinstance(item, (str, int, float, bool, type(None)))
                else str(item)
                for item in value
            )
        elif isinstance(value, (str, int, float, bool)):
            sanitized[key] = value
        elif value is None:
            sanitized[key] = None
        else:
            # Convert other types to string
            sanitized[key] = str(value)

    return sanitized


_extract_organization_and_product_names = extract_org_and_product


def _meter_raw_stream(state, usage_metadata, request_kwargs, request_time, request_time_dt, provider):
    """Fire the metering event for a client.messages.create(stream=True) call.

    Called exactly once by the raw-stream wrappers when the stream terminates
    (exhaustion, context-manager exit, close, early break, or GC). Partial
    usage from an interrupted stream is still metered.
    """
    if _get_thread_safe_client() is None:
        return
    failure = state.failure
    if not state.saw_message_start and failure is None:
        logger.debug("Raw stream ended before message_start; skipping metering (no usage data)")
        return

    response_time_dt = datetime.datetime.now(datetime.timezone.utc)
    response_time = response_time_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    request_duration = (response_time_dt - request_time_dt).total_seconds() * 1000
    time_to_first_token = 0
    if state.first_event_time_dt is not None:
        time_to_first_token = (state.first_event_time_dt - request_time_dt).total_seconds() * 1000
    completion_start_time = response_time
    if state.first_event_time_dt is not None:
        completion_start_time = state.first_event_time_dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    response_id = state.message_id or f"anthropic-{uuid.uuid4().hex[:16]}"
    prompt_tokens = state.input_tokens
    completion_tokens = state.output_tokens
    cache_creation_input_tokens = state.cache_creation_input_tokens
    cache_read_input_tokens = state.cache_read_input_tokens
    cache_creation_ttl_counts = state.cache_creation_ttl_counts
    model = _recorded_model(state.model, request_kwargs, provider) or request_kwargs.get("model")

    stop_reason = ERROR_STOP_REASON if failure else _STOP_REASONS.get(state.stop_reason, "END")

    accumulated_content = ''.join(state.accumulated_text)
    (system_prompt, input_messages, output_response, prompts_truncated) = (
        extract_prompt_data_if_enabled(None if failure else request_kwargs, accumulated_content=accumulated_content)
    )

    provider_metadata = get_provider_metadata(provider)

    async def metering_call():
        completion_args = None
        main_key = None
        try:
            from revenium_middleware import shutdown_event

            if shutdown_event.is_set():
                logger.warning("Skipping metering call during shutdown")
                return
            logger.debug("Metering call to Revenium for raw stream completion %s", response_id)

            subscriber = extract_subscriber_from_metadata(usage_metadata)

            trace_fields = _extract_trace_fields(usage_metadata, request_kwargs)

            extra_body = {}
            if trace_fields.get('has_vision_content'):
                extra_body['hasVisionContent'] = True
            extra_body = merge_extra_body(extra_body, extract_agentic_job_fields(usage_metadata))

            organization_name, product_name = _extract_organization_and_product_names(usage_metadata)
            meta = extract_common_metadata(usage_metadata)

            completion_args = {
                "cache_creation_token_count": cache_creation_input_tokens,
                **cache_creation_ttl_counts,
                "cache_read_token_count": cache_read_input_tokens,
                "input_token_cost": None,
                "output_token_cost": None,
                "total_cost": None,
                "output_token_count": completion_tokens,
                "cost_type": "AI",
                "model": model,
                "input_token_count": prompt_tokens,
                "provider": provider_metadata["provider"],
                "model_source": provider_metadata["model_source"],
                "reasoning_token_count": 0,
                "request_time": request_time,
                "response_time": response_time,
                "completion_start_time": completion_start_time,
                "request_duration": int(request_duration),
                "time_to_first_token": int(time_to_first_token),
                "stop_reason": stop_reason,
                "total_token_count": prompt_tokens + completion_tokens,
                "transaction_id": response_id,
                "trace_id": meta["trace_id"],
                "task_type": meta["task_type"],
                "subscriber": subscriber if subscriber else None,
                "organization_name": organization_name,
                "subscription_id": meta["subscription_id"],
                "product_name": product_name,
                "agent": meta["agent"],
                "is_streamed": True,
                "operation_type": trace_fields.get('operation_type', 'CHAT'),
                "response_quality_score": meta["response_quality_score"],
                "middleware_source": "PYTHON",
                "environment": trace_fields.get('environment'),
                "region": trace_fields.get('region'),
                "credential_alias": trace_fields.get('credential_alias'),
                "trace_type": trace_fields.get('trace_type'),
                "trace_name": trace_fields.get('trace_name'),
                "ticket_id": trace_fields.get('ticket_id'),
                "agent_version": trace_fields.get('agent_version'),
                **_effort_payload(trace_fields),
                **trace_fields['prompt_context'],
                **trace_fields['skill'],
                "parent_transaction_id": trace_fields.get('parent_transaction_id'),
                "transaction_name": trace_fields.get('transaction_name'),
                "retry_number": trace_fields.get('retry_number'),
                "operation_subtype": trace_fields.get('operation_subtype'),
                "system_prompt": system_prompt,
                "input_messages": input_messages,
                "output_response": output_response,
                "prompts_truncated": prompts_truncated,
                "extra_body": extra_body if extra_body else None,
                **_failure_fields(failure),
            }
            main_key = failure.idempotency_key if failure else next_idempotency_key()
            result = _submit_main_record(completion_args, main_key)
            logger.debug("Metering call result for raw stream: %s", result)
            success = False
            try:
                if result is None:
                    success = False
                elif hasattr(result, 'status_code'):
                    status_code = int(getattr(result, 'status_code', 0) or 0)
                    success = 200 <= status_code < 300
                elif hasattr(result, 'resource_type') or hasattr(result, 'resourceType') or hasattr(result, 'id'):
                    success = True
                else:
                    success = True
            except Exception:
                success = False

            if success:
                logger.debug("[REVENIUM SUCCESS] Raw stream metering call successful for transaction %s", response_id)
            else:
                logger.warning(
                    "[REVENIUM ERROR] Raw stream metering call did not return success for transaction %s: %s",
                    response_id, result
                )
        except Exception as e:
            from revenium_middleware import shutdown_event
            if not shutdown_event.is_set():
                logger.warning(f"Error in metering call for raw stream: {str(e)}")
                import traceback
                logger.warning(f"Traceback: {traceback.format_exc()}")
        finally:
            _submit_iteration_records(completion_args, state.iterations, request_kwargs, main_key)

    thread = _safe_run_async_in_thread(metering_call)
    logger.debug("Metering thread started for raw stream: %s", thread)


def _text_delta_of(event):
    """Return the text a messages.stream() event carries, if any.

    Keyed on the raw ``content_block_delta`` / ``text_delta`` pair rather than
    the SDK's synthesized ``text`` event: the stream fires both for the same
    delta, so reading only one of the two keeps the accumulation exact.
    """
    if getattr(event, "type", None) != "content_block_delta":
        return None
    delta = getattr(event, "delta", None)
    if getattr(delta, "type", None) != "text_delta":
        return None
    return getattr(delta, "text", None)


def _final_message_text(final_message):
    """Concatenate the text blocks of a (possibly partial) message.

    The authoritative output text, whatever the caller did with the stream.
    The SDK accumulates every text delta into the message before handing the
    event on, so this covers all the consumption styles at once -- including
    the mixed one that defeats a chunk-by-chunk accumulator: iterate a few
    chunks, then let ``get_final_message()`` drain the rest, which happens on
    the SDK stream directly and never reaches the wrappers' iterators.
    """
    parts = []
    for block in getattr(final_message, "content", None) or []:
        if getattr(block, "type", None) == "text":
            text = getattr(block, "text", None)
            if text:
                parts.append(text)
    return ''.join(parts)


def _accumulated_message(stream_context):
    """The message the SDK has assembled so far, or None if nothing arrived.

    A stream the caller abandoned part-way cannot be finalised -- the response
    is closed by then, and draining it here would do I/O the caller declined --
    but the events already read are billable, and the SDK kept them. Its
    ``current_message_snapshot`` asserts instead of returning None when no
    ``message_start`` was ever seen, hence the except.
    """
    try:
        return stream_context.current_message_snapshot
    except (AssertionError, AttributeError):
        return None


def _streamed_message(stream_context, failure, request_kwargs):
    """What a client.messages.stream() call delivered, or, for a stream that
    broke off before any message arrived, a usage-free stand-in for its ERROR
    record."""
    message = _accumulated_message(stream_context)
    if message is None and failure is not None:
        return _failed_message(request_kwargs.get("model"))
    return message


def _meter_message_stream(final_message, usage_metadata, request_kwargs, request_time,
                          request_time_dt, response_time_dt, provider,
                          first_token_time=None, request_start_time=None, stream_iterations=None,
                          failure: Optional[CallFailure] = None):
    """Fire the metering event for a client.messages.stream() finalisation.

    Shared by the sync and the async context-manager wrappers so both emit an
    identical payload; only the timing inputs and the accumulated text differ.
    ``final_message`` may be a partial snapshot from an interrupted stream --
    what was read is what gets billed. Callers must not call this when
    ``final_message.usage`` is None: there is nothing billable to report.

    ``first_token_time`` and ``request_start_time`` are milliseconds since the
    epoch, as recorded by the wrappers off ``time.time()``.

    ``stream_iterations`` is the usage iterations list read off the stream's own
    events, or None when the relay over the helper's events was not installed or
    the stream ended before its final usage event; then the helper's accumulated
    message is used. When given it is the only source of the per-pass records:
    the non-beta helper's accumulated message never holds the list, and the beta
    helper's holds the same one, so reading both would bill each pass twice.

    ``failure`` marks a stream that broke off with an error: it is recorded as
    ERROR, under the failure's own Idempotency-Key and without its prompts.
    """
    if _get_thread_safe_client() is None:
        return
    accumulated_content = _final_message_text(final_message)
    response_time = response_time_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    request_duration = (response_time_dt - request_time_dt).total_seconds() * 1000
    response_id = final_message.id

    # Latency to the first token, not to the end of the stream: reporting the
    # finish time as the completion start understates it for every stream.
    time_to_first_token = 0
    completion_start_time = response_time
    if first_token_time:
        completion_start_time = datetime.datetime.fromtimestamp(
            first_token_time / 1000, datetime.timezone.utc
        ).strftime("%Y-%m-%dT%H:%M:%SZ")
        if request_start_time:
            time_to_first_token = first_token_time - request_start_time

    prompt_tokens = final_message.usage.input_tokens
    completion_tokens = final_message.usage.output_tokens
    cache_creation_input_tokens = final_message.usage.cache_creation_input_tokens
    cache_read_input_tokens = final_message.usage.cache_read_input_tokens
    cache_creation_ttl_counts = extract_cache_creation_ttl_counts(final_message.usage)

    logger.debug(
        "Anthropic client.messages.stream token usage - prompt: %d, completion: %d, "
        "cache_creation_input_tokens: %s, cache_read_input_tokens: %s",
        prompt_tokens, completion_tokens, cache_creation_input_tokens, cache_read_input_tokens
    )

    anthropic_finish_reason = None
    if final_message.stop_reason:
        anthropic_finish_reason = final_message.stop_reason

    stop_reason = ERROR_STOP_REASON if failure else _STOP_REASONS.get(anthropic_finish_reason, "END")

    (system_prompt, input_messages, output_response, prompts_truncated) = (
        extract_prompt_data_if_enabled(None if failure else request_kwargs, accumulated_content=accumulated_content)
    )

    # The detected provider, not a hardcoded Anthropic label: Foundry clients
    # are served by this SDK-native streaming path, so hardcoding dropped their
    # spend into direct Anthropic totals.
    provider_metadata = get_provider_metadata(provider)

    async def metering_call():
        completion_args = None
        main_key = None
        try:
            from revenium_middleware import shutdown_event

            if shutdown_event.is_set():
                logger.warning("Skipping metering call during shutdown")
                return
            logger.debug("Metering call to Revenium for stream completion %s", response_id)

            subscriber = extract_subscriber_from_metadata(usage_metadata)

            trace_fields = _extract_trace_fields(usage_metadata, request_kwargs)

            extra_body = {}
            if trace_fields.get('has_vision_content'):
                extra_body['hasVisionContent'] = True
            extra_body = merge_extra_body(extra_body, extract_agentic_job_fields(usage_metadata))

            organization_name, product_name = _extract_organization_and_product_names(usage_metadata)
            meta = extract_common_metadata(usage_metadata)

            completion_args = {
                "cache_creation_token_count": cache_creation_input_tokens,
                **cache_creation_ttl_counts,
                "cache_read_token_count": cache_read_input_tokens,
                "input_token_cost": None,
                "output_token_cost": None,
                "total_cost": None,
                "output_token_count": completion_tokens,
                "cost_type": "AI",
                "model": _recorded_model(final_message.model, request_kwargs, provider),
                "input_token_count": prompt_tokens,
                "provider": provider_metadata["provider"],
                "model_source": provider_metadata["model_source"],
                "reasoning_token_count": 0,
                "request_time": request_time,
                "response_time": response_time,
                "completion_start_time": completion_start_time,
                "request_duration": int(request_duration),
                "time_to_first_token": int(time_to_first_token),
                "stop_reason": stop_reason,
                "total_token_count": prompt_tokens + completion_tokens,
                "transaction_id": response_id,
                "trace_id": meta["trace_id"],
                "task_type": meta["task_type"],
                "subscriber": subscriber if subscriber else None,
                "organization_name": organization_name,
                "subscription_id": meta["subscription_id"],
                "product_name": product_name,
                "agent": meta["agent"],
                "is_streamed": True,
                "operation_type": trace_fields.get('operation_type', 'CHAT'),
                "response_quality_score": meta["response_quality_score"],
                "middleware_source": "PYTHON",
                "environment": trace_fields.get('environment'),
                "region": trace_fields.get('region'),
                "credential_alias": trace_fields.get('credential_alias'),
                "trace_type": trace_fields.get('trace_type'),
                "trace_name": trace_fields.get('trace_name'),
                "ticket_id": trace_fields.get('ticket_id'),
                "agent_version": trace_fields.get('agent_version'),
                **_effort_payload(trace_fields),
                **trace_fields['prompt_context'],
                **trace_fields['skill'],
                "parent_transaction_id": trace_fields.get('parent_transaction_id'),
                "transaction_name": trace_fields.get('transaction_name'),
                "retry_number": trace_fields.get('retry_number'),
                "operation_subtype": trace_fields.get('operation_subtype'),
                "system_prompt": system_prompt,
                "input_messages": input_messages,
                "output_response": output_response,
                "prompts_truncated": prompts_truncated,
                "extra_body": extra_body if extra_body else None,
                **_failure_fields(failure),
            }
            main_key = failure.idempotency_key if failure else next_idempotency_key()
            result = _submit_main_record(completion_args, main_key)
            logger.debug("Metering call result for stream: %s", result)
            success = False
            try:
                if result is None:
                    success = False
                elif hasattr(result, 'status_code'):
                    status_code = int(getattr(result, 'status_code', 0) or 0)
                    success = 200 <= status_code < 300
                elif hasattr(result, 'resource_type') or hasattr(result, 'resourceType') or hasattr(result, 'id'):
                    success = True
                else:
                    success = True
            except Exception:
                success = False

            if success:
                logger.debug("[REVENIUM SUCCESS] Streaming metering call successful for transaction %s", response_id)
            else:
                logger.warning(
                    "[REVENIUM ERROR] Streaming metering call did not return success for transaction %s: %s",
                    response_id, result
                )
        except Exception as e:
            from revenium_middleware import shutdown_event
            if not shutdown_event.is_set():
                logger.warning(f"Error in metering call for stream: {str(e)}")
                import traceback
                logger.warning(f"Traceback: {traceback.format_exc()}")
        finally:
            _submit_iteration_records(
                completion_args, final_message.usage if stream_iterations is None else stream_iterations,
                request_kwargs, main_key)

    thread = _safe_run_async_in_thread(metering_call)
    logger.debug("Metering thread started for stream: %s", thread)


class _AsyncMessageStreamMetering:
    """Metering wrapper around anthropic's AsyncMessageStreamManager.

    ``AsyncMessages.stream`` hands back a manager whose ``__aenter__`` yields an
    ``AsyncMessageStream``, and none of that flow passes through
    ``AsyncMessages.create``. Until this wrapper existed, an
    ``async with client.messages.stream(...)`` block therefore produced no
    metering record at all -- silent, unbilled spend.

    It mirrors the sync ``stream_wrapper``: callers keep the stream's own
    surface (``async for``, ``text_stream``, ``await get_final_message()``,
    plus anything else forwarded by ``__getattr__``), and the accumulated usage
    is metered fire-and-forget when the ``async with`` block exits.
    """

    def __init__(self, stream_manager, usage_metadata, request_kwargs,
                 request_time, request_time_dt, provider):
        self._stream_manager = stream_manager
        self._metering_client = current_client()
        self._call_context = CallContext()
        self._usage_metadata = usage_metadata
        self._request_kwargs = request_kwargs
        self._request_time = request_time
        self._request_time_dt = request_time_dt
        self._provider = provider
        self._stream_context = None
        self._usage_state = StreamUsageState()
        self.final_message = None
        self.first_token_time = None
        self.request_start_time = time.time() * 1000

    async def __aenter__(self):
        with _recording_failure(self._usage_metadata, self._request_kwargs, self._request_time,
                                self._request_time_dt, self._provider, True, self._metering_client):
            self._stream_context = await self._stream_manager.__aenter__()
        meter_async_stream_helper_events(self._stream_context, self._usage_state)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        result = await self._stream_manager.__aexit__(exc_type, exc_val, exc_tb)

        if self._stream_context is None:
            return result

        try:
            # Whatever the SDK accumulated, without draining anything the
            # caller left unread. Tokens already delivered are billable, so a
            # stream broken off mid-iteration, or one that failed part-way, is
            # metered from its snapshot; only a stream nothing was ever read
            # from has nothing to report.
            failure = self._usage_state.failure
            if self.final_message is None:
                self.final_message = _streamed_message(self._stream_context, failure, self._request_kwargs)

            response_time_dt = datetime.datetime.now(datetime.timezone.utc)

            if self.final_message is None or self.final_message.usage is None:
                return result

            self._call_context.run_recorded_under(
                self._metering_client, _meter_message_stream,
                self.final_message, self._usage_metadata, self._request_kwargs,
                self._request_time, self._request_time_dt, response_time_dt,
                self._provider,
                first_token_time=self.first_token_time,
                request_start_time=self.request_start_time,
                stream_iterations=self._usage_state.iterations,
                failure=failure,
            )
        except Exception as e:
            logger.warning(f"Error processing final message from async stream: {str(e)}")
            import traceback
            logger.warning(f"Traceback: {traceback.format_exc()}")

        return result

    def _record_first_token(self, text):
        if text and self.first_token_time is None:
            self.first_token_time = time.time() * 1000

    def __aiter__(self):
        return self._iterate_events()

    async def _iterate_events(self):
        async for event in self._stream_context:
            self._record_first_token(_text_delta_of(event))
            yield event

    @property
    def text_stream(self):
        return self._iterate_text(self._stream_context.text_stream)

    async def _iterate_text(self, source):
        async for chunk in source:
            self._record_first_token(chunk)
            yield chunk

    async def get_final_message(self):
        # Cached so the finalisation reuses it instead of awaiting a second
        # consumption of the stream.
        if self.final_message is None:
            self.final_message = await self._stream_context.get_final_message()
        return self.final_message

    def __getattr__(self, name):
        # Read through __dict__: a plain attribute access would re-enter
        # __getattr__ for _stream_context itself and recurse forever.
        stream_context = self.__dict__.get('_stream_context')
        if stream_context is None:
            raise AttributeError(
                f"{type(self).__name__} has no attribute {name!r} before its "
                "'async with' block is entered"
            )
        return getattr(stream_context, name)


def _failure_fields(failure: Optional[CallFailure]) -> Dict[str, Any]:
    return failure.record_fields() if failure else {}


def _failed_message(model):
    return SimpleNamespace(
        id=failed_transaction_id(), model=model, stop_reason=None, content=[],
        usage=SimpleNamespace(input_tokens=0, output_tokens=0, cache_creation_input_tokens=0,
                              cache_read_input_tokens=0),
    )


_delivered_parse_messages: contextvars.ContextVar[Optional[list]] = contextvars.ContextVar(
    "revenium_anthropic_delivered_parse_messages", default=None)


def _capturing_delivered_message(wrapped, instance, args, kwargs):
    """Keep the message Anthropic delivered to ``parse()``, which the SDK's
    output parser may reject after the call was billed."""
    delivered = _delivered_parse_messages.get()
    message = kwargs.get("response", args[0] if args else None)
    if delivered is not None and message is not None:
        delivered.append(message)
    return wrapped(*args, **kwargs)


@contextmanager
def _recording_failure(usage_metadata, request_kwargs, request_time, request_time_dt, provider, is_streamed,
                       metering_client):
    """Send one record for a call that raised, under the metering client
    current when the call started, then let the caller's exception through
    unchanged: the delivered message when Anthropic answered (a ``parse()``
    answer its output parser rejected), else ERROR for a refused call."""
    outer = exception_in_flight()
    delivered = []
    token = _delivered_parse_messages.set(delivered)
    meter = functools.partial(_meter_message, usage_metadata=usage_metadata, request_kwargs=request_kwargs,
                              request_time=request_time, request_time_dt=request_time_dt, provider=provider,
                              is_streamed=is_streamed)
    try:
        yield
    except Exception as error:
        if delivered:
            _record_under(metering_client, meter, delivered[-1])
        else:
            failure = failure_of(error, outer)
            if failure is not None:
                _record_under(metering_client, meter, _failed_message(request_kwargs.get("model")), failure=failure)
        raise
    finally:
        _delivered_parse_messages.reset(token)


def _record_under(metering_client, meter, message, failure: Optional[CallFailure] = None):
    try:
        with recorded_under(metering_client):
            meter(message, failure=failure)
    except Exception as metering_error:
        logger.warning("Could not record a failed Anthropic call: %s", metering_error)


def _meter_message(response, usage_metadata, request_kwargs, request_time, request_time_dt, provider,
                   failure: Optional[CallFailure] = None, is_streamed=False):
    """Fire the metering event for a non-streamed Messages.create response,
    or, with ``failure``, the ERROR record of a call the provider refused."""
    if _get_thread_safe_client() is None:
        return
    response_id = getattr(response, 'id', None) or f"anthropic-{uuid.uuid4().hex[:16]}"
    logger.debug("REVENIUM MIDDLEWARE: Received response from client.messages.create: %s", response_id)
    response_time_dt = datetime.datetime.now(datetime.timezone.utc)
    response_time = response_time_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    request_duration = (response_time_dt - request_time_dt).total_seconds() * 1000

    if getattr(response, 'usage', None) is None:
        return

    prompt_tokens = response.usage.input_tokens
    completion_tokens = response.usage.output_tokens
    cache_creation_input_tokens = response.usage.cache_creation_input_tokens
    cache_read_input_tokens = response.usage.cache_read_input_tokens
    cache_creation_ttl_counts = extract_cache_creation_ttl_counts(response.usage)

    logger.debug(
        "Anthropic client.ai.create_completion token usage - prompt: %d, completion: %d, "
        "cache_creation_input_tokens: %s,cache_read_input_tokens: %s",
        prompt_tokens, completion_tokens, cache_creation_input_tokens, cache_read_input_tokens
    )

    anthropic_finish_reason = None
    if response.stop_reason:
        anthropic_finish_reason = response.stop_reason

    stop_reason = ERROR_STOP_REASON if failure else _STOP_REASONS.get(anthropic_finish_reason, "END")

    (system_prompt, input_messages, output_response, prompts_truncated) = (
        extract_prompt_data_if_enabled(None if failure else request_kwargs, response=response)
    )

    provider_metadata = get_provider_metadata(provider)

    async def metering_call():
        completion_args = None
        main_key = None
        try:
            from revenium_middleware import shutdown_event

            if shutdown_event.is_set():
                logger.warning("Skipping metering call during shutdown")
                return
            logger.debug("Metering call to Revenium for completion %s with usage_metadata: %s", response_id,
                         sanitize_for_logging(usage_metadata))

            subscriber = extract_subscriber_from_metadata(usage_metadata)

            trace_fields = _extract_trace_fields(usage_metadata, request_kwargs)

            extra_body = {}
            if trace_fields.get('has_vision_content'):
                extra_body['hasVisionContent'] = True
            extra_body = merge_extra_body(extra_body, extract_agentic_job_fields(usage_metadata))

            organization_name, product_name = _extract_organization_and_product_names(usage_metadata)
            meta = extract_common_metadata(usage_metadata)

            completion_args = {
                "cache_creation_token_count": cache_creation_input_tokens,
                **cache_creation_ttl_counts,
                "cache_read_token_count": cache_read_input_tokens,
                "input_token_cost": None,
                "output_token_cost": None,
                "total_cost": None,
                "output_token_count": completion_tokens,
                "cost_type": "AI",
                "model": _recorded_model(response.model, request_kwargs, provider),
                "input_token_count": prompt_tokens,
                "provider": provider_metadata["provider"],
                "model_source": provider_metadata["model_source"],
                "reasoning_token_count": 0,
                "request_time": request_time,
                "response_time": response_time,
                "completion_start_time": response_time,
                "request_duration": int(request_duration),
                "time_to_first_token": int(request_duration),
                "stop_reason": stop_reason,
                "total_token_count": prompt_tokens + completion_tokens,
                "transaction_id": response_id,
                "trace_id": meta["trace_id"],
                "task_type": meta["task_type"],
                "subscriber": subscriber if subscriber else None,
                "organization_name": organization_name,
                "subscription_id": meta["subscription_id"],
                "product_name": product_name,
                "agent": meta["agent"],
                "response_quality_score": meta["response_quality_score"],
                "is_streamed": is_streamed,
                "operation_type": trace_fields.get('operation_type', 'CHAT'),
                "middleware_source": "PYTHON",
                "environment": trace_fields.get('environment'),
                "region": trace_fields.get('region'),
                "credential_alias": trace_fields.get('credential_alias'),
                "trace_type": trace_fields.get('trace_type'),
                "trace_name": trace_fields.get('trace_name'),
                "ticket_id": trace_fields.get('ticket_id'),
                "agent_version": trace_fields.get('agent_version'),
                **_effort_payload(trace_fields),
                **trace_fields['prompt_context'],
                **trace_fields['skill'],
                "parent_transaction_id": trace_fields.get('parent_transaction_id'),
                "transaction_name": trace_fields.get('transaction_name'),
                "retry_number": trace_fields.get('retry_number'),
                "operation_subtype": trace_fields.get('operation_subtype'),
                "system_prompt": system_prompt,
                "input_messages": input_messages,
                "output_response": output_response,
                "prompts_truncated": prompts_truncated,
                "extra_body": extra_body if extra_body else None,
                **_failure_fields(failure),
            }
            main_key = failure.idempotency_key if failure else next_idempotency_key()
            result = _submit_main_record(completion_args, main_key)
            logger.debug("Metering call result: %s", result)
            success = False
            try:
                if result is None:
                    success = False
                elif hasattr(result, 'status_code'):
                    status_code = int(getattr(result, 'status_code', 0) or 0)
                    success = 200 <= status_code < 300
                elif hasattr(result, 'resource_type') or hasattr(result, 'resourceType') or hasattr(result, 'id'):
                    success = True
                else:
                    success = True
            except Exception:
                success = False

            if success:
                logger.debug("[REVENIUM SUCCESS] Metering call successful for transaction %s", response_id)
            else:
                logger.warning("[REVENIUM ERROR] Metering call did not return success for transaction %s: %s", response_id, result)
        except Exception as e:
            from revenium_middleware import shutdown_event
            if not shutdown_event.is_set():
                logger.warning(f"Error in metering call: {str(e)}")
                import traceback
                logger.warning(f"Traceback: {traceback.format_exc()}")
        finally:
            _submit_iteration_records(completion_args, response.usage, request_kwargs, main_key)

    thread = _safe_run_async_in_thread(metering_call)
    logger.debug("Metering thread started: %s", thread)
    if thread is not None and failure is None:
        claim_call_for_transport(ANTHROPIC, getattr(response, 'id', None))


def _selectively_unmetered():
    return reported_if_unmetered(is_selective_metering_enabled() and not is_inside_decorated_function())


def _call_unmetered(wrapped, instance, args, kwargs):
    """Call the SDK without metering, minus the Revenium-only
    ``usage_metadata`` keyword that the SDK's own signature rejects. A
    breached cap still refuses it."""
    api_metadata = kwargs.pop("usage_metadata", None)
    provider = _cap_provider(instance, kwargs)
    client_instance = getattr(instance, '_client', None) if instance else None
    check_unmetered_call(api_metadata if isinstance(api_metadata, dict) else {},
                         model=_capped_model_names(kwargs.get("model"), provider, client_instance),
                         provider=None if provider is None else get_provider_metadata(provider)["provider"],
                         callback_transport=ANTHROPIC)
    return wrapped(*args, **kwargs)


async def _await_unmetered(wrapped, instance, args, kwargs):
    return await _call_unmetered(wrapped, instance, args, kwargs)


def _cap_provider(instance, kwargs):
    """The provider the call's record would name, worked out only when caps are checked."""
    if not is_circuit_breaker_enabled():
        return None
    client_instance = getattr(instance, '_client', None) if instance else None
    return detect_provider(client=client_instance, base_url=kwargs.get('base_url'))


def _application_profile_route(profile_arn, client_instance):
    """The Claude model id an application inference profile routes to, or None.

    A sync AnthropicBedrock client looks the profile up with its own
    credentials, sharing the raw boto3 transport's once-per-process lookup.
    Any other caller, an async client among them, reads only what that
    lookup already cached: a blocking AWS call must never hold an event loop.
    """
    if _signs_its_own_bedrock_requests(client_instance):
        return resolve_application_profile(AnthropicBedrockCredentials(client_instance), profile_arn,
                                           unresolved=NOT_CAPPED_BY_MODEL)
    return cached_application_profile(profile_arn)


def _capped_model_names(model, provider, client_instance):
    """The names a model-scoped cap may use for ``model`` on ``client_instance``.

    The string is the name the call's record carries, which the platform
    counts the spend under. An application inference profile ARN keeps the
    ARN and is also known by the Claude model it routes to. For a Bedrock
    Claude id the id Bedrock reports back and the Claude model name are
    aliases. Any other model is judged as given.
    """
    if is_application_profile_arn(model) and is_circuit_breaker_enabled():
        routed = _application_profile_route(model, client_instance)
        if routed is not None:
            return ModelKnownByAliases(model, bedrock_reported_model_id(routed), anthropic_model_name(routed))
    reported, claude_name = bedrock_reported_model_id(model), anthropic_model_name(model)
    recorded = _recorded_model(reported, {"model": model}, provider)
    if recorded == reported == claude_name:
        return recorded
    return ModelKnownByAliases(recorded, reported, claude_name)


def _cap_model(kwargs, *, provider, bedrock_fast_path, client_instance):
    """The model a model-scoped cap judges the call by.

    The boto3 fast path records the model as requested. Every other Bedrock
    call records the Bedrock id it was sent with, and any other call the
    model the response reports.
    """
    model = kwargs.get("model")
    if bedrock_fast_path and provider == Provider.BEDROCK:
        if not _signs_its_own_bedrock_requests(client_instance):
            return model
        if isinstance(model, str):
            model = _model_id(model)
    return _capped_model_names(model, provider, client_instance)


def _check_caps(usage_metadata, kwargs, provider, *, bedrock_fast_path=False, client_instance=None):
    model = _cap_model(kwargs, provider=provider, bedrock_fast_path=bedrock_fast_path, client_instance=client_instance)
    check_enforcement(usage_metadata, model=model, provider=get_provider_metadata(provider)["provider"])


def _metered_under(metering_client, meter):
    """``meter`` for a body the caller reads later, recorded under the client
    current when its call started."""
    def meter_under_the_call_key(message):
        with recorded_under(metering_client):
            meter(message)
    return meter_under_the_call_key


def create_wrapper(wrapped, instance, args, kwargs, *, bedrock_fast_path=True, message_type=Message):
    if _selectively_unmetered():
        return _call_unmetered(wrapped, instance, args, kwargs)

    logger.debug("Anthropic client.messages.create wrapper called: %s: %s", wrapped, sanitize_for_logging(args))

    usage_metadata, request_time, request_time_dt = extract_usage_metadata_and_timing(kwargs, "create")

    # REVENIUM_BEDROCK_DISABLE is honoured inside detect_provider (it
    # suppresses only the Bedrock signals), so Foundry attribution survives
    # the switch instead of collapsing to direct Anthropic.
    client_instance = getattr(instance, '_client', None) if instance else None
    base_url = kwargs.get('base_url', None)
    provider = detect_provider(client=client_instance, base_url=base_url)

    logger.debug(f"Detected provider: {provider}")
    _check_caps(usage_metadata, kwargs, provider, bedrock_fast_path=bedrock_fast_path, client_instance=client_instance)

    if bedrock_fast_path and provider == Provider.BEDROCK and _signs_its_own_bedrock_requests(client_instance):
        _use_bedrock_model_id(kwargs)
    elif bedrock_fast_path and provider == Provider.BEDROCK:
        try:
            logger.debug("Routing to Bedrock handler")
            return _handle_bedrock_request(args, kwargs, usage_metadata, request_time_dt, request_time,
                                           region=getattr(client_instance, 'aws_region', None))
        except (BedrockValidationError, ImportError) as e:
            # The request was never sent (payload validation or missing
            # boto3), so the SDK-native path can still serve it.
            logger.error(f"Bedrock fast-path unavailable: {e}. Falling back to the SDK-native call.")
            provider = Provider.ANTHROPIC
        # Any other failure -- including BedrockInvokeError -- means the
        # AWS request was (or may have been) attempted. Propagate it:
        # silently re-invoking through a second path masks real provider
        # errors from the caller and risks double invocation/billing.

    logger.debug("REVENIUM MIDDLEWARE: Calling client.messages.create with model=%s, max_tokens=%s",
                 kwargs.get("model"), kwargs.get("max_tokens"))

    request_kwargs = dict(kwargs)

    metering_client = current_client()
    call_context = CallContext()
    failure_recorded = _recording_failure(usage_metadata, request_kwargs, request_time, request_time_dt,
                                          provider, bool(kwargs.get("stream")), metering_client)
    if kwargs.get("stream"):
        # Raw-stream form: wrapped() returns anthropic.Stream (no .usage/.id).
        # Relay events untouched and meter once when the stream terminates.
        with failure_recorded:
            stream = wrapped(*args, **kwargs)
        stream_provider = provider

        def _finalize_stream(final_state):
            call_context.run_recorded_under(metering_client, _meter_raw_stream, final_state, usage_metadata,
                                            request_kwargs, request_time, request_time_dt, stream_provider)

        claim_call_for_transport(ANTHROPIC)
        if is_raw_response(stream):
            tap_sse_body(stream, StreamUsageState(), _finalize_stream)
            return stream
        return RawStreamMeteringWrapper(stream, StreamUsageState(), _finalize_stream)

    with failure_recorded:
        returned = wrapped(*args, **kwargs)

    def _meter(message):
        _meter_message(message, usage_metadata, request_kwargs, request_time, request_time_dt, provider)

    if not is_raw_response(returned):
        _meter(returned)
    elif is_body_read(returned):
        _meter(parsed_message(returned))
    else:
        tap_message_body(returned, _metered_under(metering_client, _meter), message_type)
    return returned


wrap_once("anthropic.resources.messages.messages.Messages.create",
          "anthropic.resources.messages.messages", "Messages.create", create_wrapper)


def async_create_wrapper(wrapped, instance, args, kwargs, *, message_type=Message):
    import asyncio

    if _selectively_unmetered():
        return _await_unmetered(wrapped, instance, args, kwargs)

    logger.debug("Anthropic async client.messages.create wrapper called")

    usage_metadata, request_time, request_time_dt = extract_usage_metadata_and_timing(kwargs, "async_create")

    # Bedrock-routed async clients (e.g. AsyncAnthropicBedrock) go through
    # this same wrapper; detect the provider like the sync path does so
    # AWS usage is not attributed to direct Anthropic. Async Bedrock calls
    # are served natively by the anthropic SDK, so only the metering
    # attribution changes -- there is no async re-routing.
    # REVENIUM_BEDROCK_DISABLE is honoured inside detect_provider.
    client_instance = getattr(instance, '_client', None) if instance else None
    detected_provider = detect_provider(client=client_instance,
                                        base_url=kwargs.get('base_url', None))

    request_kwargs = dict(kwargs)

    async def _async_create():
        _check_caps(usage_metadata, kwargs, detected_provider)
        metering_client = current_client()
        call_context = CallContext()
        with _recording_failure(usage_metadata, request_kwargs, request_time, request_time_dt,
                                detected_provider, bool(request_kwargs.get("stream")), metering_client):
            returned = await wrapped(*args, **kwargs)

        if request_kwargs.get("stream"):
            # Raw-stream form: response is anthropic.AsyncStream (no .usage/.id).
            def _finalize_stream(final_state):
                call_context.run_recorded_under(metering_client, _meter_raw_stream, final_state, usage_metadata,
                                                request_kwargs, request_time, request_time_dt, detected_provider)

            claim_call_for_transport(ANTHROPIC)
            if is_raw_response(returned):
                tap_async_sse_body(returned, StreamUsageState(), _finalize_stream)
                return returned
            return AsyncRawStreamMeteringWrapper(returned, StreamUsageState(), _finalize_stream)

        def _meter(message):
            _meter_message(message, usage_metadata, request_kwargs, request_time, request_time_dt, detected_provider)

        if not is_raw_response(returned):
            _meter(returned)
        elif is_body_read(returned):
            _meter(await async_parsed_message(returned))
        else:
            tap_async_message_body(returned, _metered_under(metering_client, _meter), message_type)
        return returned

    return _async_create()


wrap_once("anthropic.resources.messages.messages.AsyncMessages.create",
          "anthropic.resources.messages.messages", "AsyncMessages.create", async_create_wrapper)


def stream_wrapper(wrapped, instance, args, kwargs, *, bedrock_fast_path=True):
    if _selectively_unmetered():
        return _call_unmetered(wrapped, instance, args, kwargs)

    logger.debug("REVENIUM MIDDLEWARE: Intercepted client.messages.stream call - wrapper active")

    usage_metadata, request_time, request_time_dt = extract_usage_metadata_and_timing(kwargs, "stream")

    # REVENIUM_BEDROCK_DISABLE is honoured inside detect_provider: it
    # suppresses only the Bedrock signals, so a Foundry stream keeps its
    # Foundry label when Bedrock routing is switched off.
    client_instance = getattr(instance, '_client', None) if instance else None
    base_url = kwargs.get('base_url', None)
    detected_provider = detect_provider(client=client_instance, base_url=base_url)
    _check_caps(usage_metadata, kwargs, detected_provider, bedrock_fast_path=bedrock_fast_path,
                client_instance=client_instance)

    if bedrock_fast_path and detected_provider == Provider.BEDROCK and _signs_its_own_bedrock_requests(client_instance):
        _use_bedrock_model_id(kwargs)
    elif bedrock_fast_path and detected_provider == Provider.BEDROCK:
        try:
            logger.debug("Routing streaming request to Bedrock handler")
            return _handle_bedrock_stream_request(args, kwargs, usage_metadata, request_time_dt, request_time,
                                                  region=getattr(client_instance, 'aws_region', None))
        except (BedrockValidationError, ImportError) as e:
            # The request was never sent (payload validation or missing
            # boto3), so the SDK-native path can still serve it.
            logger.error(f"Bedrock streaming fast-path unavailable: {e}. Falling back to the SDK-native call.")
            # Attributed to Anthropic once the SDK-native path serves
            # it, matching the non-streaming create wrapper's fallback.
            detected_provider = Provider.ANTHROPIC
        # Any other failure -- including BedrockStreamError -- means
        # the AWS request was (or may have been) attempted. Propagate
        # it: silently re-invoking through a second path masks real
        # provider errors and risks double invocation/billing.

    logger.debug("REVENIUM MIDDLEWARE: Calling client.messages.stream with model=%s, max_tokens=%s",
                 kwargs.get("model"), kwargs.get("max_tokens"))

    request_kwargs = dict(kwargs)

    stream = wrapped(*args, **kwargs)
    logger.debug("REVENIUM MIDDLEWARE: Received stream from client.messages.stream")

    class StreamWrapper:
        def __init__(self, stream):
            self.stream = stream
            self.metering_client = current_client()
            self.call_context = CallContext()
            self.response_time_dt = None
            self.response_id = None
            self.final_message = None
            self.first_token_time = None
            self.request_start_time = time.time() * 1000
            self.usage_state = StreamUsageState()

        def __enter__(self):
            # The request is sent here, so a refusal surfaces here too.
            with _recording_failure(usage_metadata, request_kwargs, request_time, request_time_dt,
                                    detected_provider, True, self.metering_client):
                self.stream_context = self.stream.__enter__()
            meter_stream_helper_events(self.stream_context, self.usage_state)
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            result = self.stream.__exit__(exc_type, exc_val, exc_tb)

            try:
                # Whatever the SDK accumulated, without draining anything
                # the caller left unread: a stream broken off part-way
                # still spent the tokens it delivered.
                failure = self.usage_state.failure
                if self.final_message is None:
                    self.final_message = _streamed_message(self.stream_context, failure, request_kwargs)

                self.response_time_dt = datetime.datetime.now(datetime.timezone.utc)
                self.response_time = self.response_time_dt.strftime("%Y-%m-%dT%H:%M:%SZ")

                if self.final_message is None or self.final_message.usage is None:
                    return result

                self.response_id = self.final_message.id

                self.call_context.run_recorded_under(
                    self.metering_client, _meter_message_stream,
                    self.final_message, usage_metadata, request_kwargs, request_time,
                    request_time_dt, self.response_time_dt, detected_provider,
                    first_token_time=self.first_token_time,
                    request_start_time=self.request_start_time,
                    stream_iterations=self.usage_state.iterations,
                    failure=failure,
                )

            except Exception as e:
                logger.warning(f"Error processing final message from stream: {str(e)}")
                import traceback
                logger.warning(f"Traceback: {traceback.format_exc()}")

            return result

        @property
        def text_stream(self):
            original_text_stream = self.stream_context.text_stream
            wrapper_self = self

            class TextStreamWrapper:
                def __iter__(self):
                    return self

                def __next__(self):
                    try:
                        chunk = next(original_text_stream)
                        wrapper_self._record_first_token(chunk)
                        return chunk
                    except StopIteration:
                        raise

            return TextStreamWrapper()

        def _record_first_token(self, text):
            if text and self.first_token_time is None:
                self.first_token_time = time.time() * 1000

        def get_final_message(self):
            # Cached so the finalisation reuses it instead of asking the
            # stream to consume itself a second time.
            if self.final_message is None:
                self.final_message = self.stream_context.get_final_message()
            return self.final_message

        def __iter__(self):
            # Yields through rather than delegating so that iterating the
            # events (instead of text_stream) still records the time to
            # first token.
            for event in self.stream_context:
                self._record_first_token(_text_delta_of(event))
                yield event

        def __getattr__(self, name):
            # Read through __dict__: a plain attribute access would re-enter
            # __getattr__ for stream_context itself and recurse forever.
            stream_context = self.__dict__.get('stream_context')
            if stream_context is None:
                raise AttributeError(
                    f"{type(self).__name__} has no attribute {name!r} before its "
                    "'with' block is entered"
                )
            return getattr(stream_context, name)

    claim_call_for_transport(ANTHROPIC)
    return StreamWrapper(stream)


wrap_once("anthropic.resources.messages.messages.Messages.stream",
          "anthropic.resources.messages.messages", "Messages.stream", stream_wrapper)


def async_stream_wrapper(wrapped, instance, args, kwargs):
    if _selectively_unmetered():
        return _call_unmetered(wrapped, instance, args, kwargs)

    logger.debug("REVENIUM MIDDLEWARE: Intercepted async client.messages.stream call - wrapper active")

    usage_metadata, request_time, request_time_dt = extract_usage_metadata_and_timing(kwargs, "async_stream")

    # Attribution only, no re-routing -- the same choice AsyncMessages.create
    # makes. The sync wrapper's Bedrock fast path is a blocking boto3 call
    # and cannot serve an async context manager, so an AsyncAnthropicBedrock
    # stream stays on the SDK transport and is metered here under the AWS
    # label rather than being handed to the Bedrock adapter.
    # REVENIUM_BEDROCK_DISABLE is honoured inside detect_provider, so a
    # Foundry stream keeps its Foundry label when Bedrock routing is off.
    client_instance = getattr(instance, '_client', None) if instance else None
    detected_provider = detect_provider(client=client_instance,
                                        base_url=kwargs.get('base_url', None))
    _check_caps(usage_metadata, kwargs, detected_provider)

    logger.debug("REVENIUM MIDDLEWARE: Calling async client.messages.stream with model=%s, max_tokens=%s",
                 kwargs.get("model"), kwargs.get("max_tokens"))

    request_kwargs = dict(kwargs)

    # AsyncMessages.stream is a plain def returning an
    # AsyncMessageStreamManager, so there is nothing to await here.
    stream_manager = wrapped(*args, **kwargs)

    claim_call_for_transport(ANTHROPIC)
    return _AsyncMessageStreamMetering(
        stream_manager, usage_metadata, request_kwargs, request_time,
        request_time_dt, detected_provider
    )


wrap_once("anthropic.resources.messages.messages.AsyncMessages.stream",
          "anthropic.resources.messages.messages", "AsyncMessages.stream", async_stream_wrapper)


# The Bedrock fast path rebuilds the request from a handful of standard fields
# and returns a plain Message, so it would drop the beta flags and the parse()
# output format. These surfaces stay on the SDK transport and keep the AWS label.
def _parse_wrapper(wrapped, instance, args, kwargs):
    return create_wrapper(wrapped, instance, args, kwargs, bedrock_fast_path=False)


def _beta_create_wrapper(wrapped, instance, args, kwargs):
    return create_wrapper(wrapped, instance, args, kwargs, bedrock_fast_path=False, message_type=BetaMessage)


def _beta_async_create_wrapper(wrapped, instance, args, kwargs):
    return async_create_wrapper(wrapped, instance, args, kwargs, message_type=BetaMessage)


def _beta_stream_wrapper(wrapped, instance, args, kwargs):
    return stream_wrapper(wrapped, instance, args, kwargs, bedrock_fast_path=False)


def _sdk_defines(module_name, attribute):
    try:
        target = importlib.import_module(module_name)
        for name in attribute.split("."):
            target = getattr(target, name)
    except (ImportError, AttributeError):
        return False
    return True


def _wrap_messages_method(module_name, attribute, wrapper):
    # The supported anthropic floor predates parse() and parts of beta.messages.
    if not _sdk_defines(module_name, attribute):
        logger.debug("%s.%s is not in the installed anthropic SDK; not wrapped", module_name, attribute)
        return
    wrap_once(f"{module_name}.{attribute}", module_name, attribute, wrapper)


_MESSAGES_MODULE = "anthropic.resources.messages.messages"
_BETA_MESSAGES_MODULE = "anthropic.resources.beta.messages.messages"

_wrap_messages_method(_MESSAGES_MODULE, "Messages.parse", _parse_wrapper)
_wrap_messages_method(_MESSAGES_MODULE, "AsyncMessages.parse", async_create_wrapper)
_wrap_messages_method(_MESSAGES_MODULE, "parse_response", _capturing_delivered_message)
_wrap_messages_method(_BETA_MESSAGES_MODULE, "parse_beta_response", _capturing_delivered_message)

try:
    from anthropic.types.beta import BetaMessage
except ImportError:
    BetaMessage = None

_BETA_WRAPPERS = (
    ("Messages.create", _beta_create_wrapper),
    ("Messages.parse", _beta_create_wrapper),
    ("Messages.stream", _beta_stream_wrapper),
    ("AsyncMessages.create", _beta_async_create_wrapper),
    ("AsyncMessages.parse", _beta_async_create_wrapper),
    ("AsyncMessages.stream", async_stream_wrapper),
)

# These platform classes copy the first-party beta functions into their own
# class body at import time, so the first-party wrap never reaches them.
_PLATFORM_BETA_MESSAGES_MODULES = (
    "anthropic.lib.bedrock._beta_messages",
    "anthropic.lib.vertex._beta_messages",
)


def _holds_own_unwrapped_copy(module_name, attribute):
    class_name, _, method = attribute.rpartition(".")
    try:
        cls = getattr(importlib.import_module(module_name), class_name)
    except (ImportError, AttributeError):
        return False
    value = vars(cls).get(method)
    return value is not None and not isinstance(value, wrapt.FunctionWrapper)


def _wrap_platform_copy(module_name, attribute, wrapper):
    if _holds_own_unwrapped_copy(module_name, attribute):
        _wrap_messages_method(module_name, attribute, wrapper)


if BetaMessage is not None:
    for _attribute, _wrapper in _BETA_WRAPPERS:
        _wrap_messages_method(_BETA_MESSAGES_MODULE, _attribute, _wrapper)
    for _module_name in _PLATFORM_BETA_MESSAGES_MODULES:
        for _attribute, _wrapper in _BETA_WRAPPERS:
            _wrap_platform_copy(_module_name, _attribute, _wrapper)

logger.debug("REVENIUM MIDDLEWARE: Anthropic middleware loaded and wrappers registered")
