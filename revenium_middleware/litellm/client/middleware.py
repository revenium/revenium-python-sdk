import wrapt
import logging
import datetime
from collections.abc import Mapping
from contextvars import ContextVar
from typing import Any, Dict, NamedTuple
from revenium_middleware import client, get_client, run_async_in_thread, shutdown_event
from revenium_middleware._core.subscriber import extract_subscriber_from_metadata
from revenium_middleware._core.fields import extract_org_and_product, extract_common_metadata, extract_agentic_job_fields, extract_effort_field, extract_prompt_context_fields, merge_extra_body
from revenium_middleware._core.cache_tokens import extract_cache_tokens
from revenium_middleware._core.config import is_selective_metering_enabled
from revenium_middleware._core.context import is_inside_decorated_function
from revenium_middleware._core import submit_ai_event
from revenium_middleware._core.log_sanitize import sanitize_for_logging
from revenium_middleware._core.patch_registry import register_patch
from .context import metadata_context
from .hooks import execute_metadata_hooks
from . import trace_fields

logger = logging.getLogger("revenium_middleware.extension")

COMPLETION_ENTRY_POINTS = ("completion", "text_completion", "completion_with_retries")
ASYNC_COMPLETION_ENTRY_POINTS = ("acompletion", "atext_completion", "acompletion_with_retries")
EMBEDDING_ENTRY_POINTS = ("embedding",)
ASYNC_EMBEDDING_ENTRY_POINTS = ("aembedding",)

CHAT_OPERATION = "CHAT"
EMBED_OPERATION = "EMBED"

_metered_call_in_progress: ContextVar[bool] = ContextVar("revenium_litellm_metered_call", default=False)
_stream_without_usage_warned = False


class _PreparedCall(NamedTuple):
    kwargs: Dict[str, Any]
    usage_metadata: Dict[str, Any]
    request_time_dt: datetime.datetime
    is_streaming: bool
    caller_requested_usage: bool


def _should_pass_through():
    if _metered_call_in_progress.get():
        return True
    return is_selective_metering_enabled() and not is_inside_decorated_function()


def _merged_usage_metadata(kwargs):
    explicit_metadata = kwargs.pop("usage_metadata", {}) if "usage_metadata" in kwargs else {}
    usage_metadata = {**metadata_context.get(), **explicit_metadata}
    try:
        usage_metadata = execute_metadata_hooks(usage_metadata)
    except Exception as e:
        logger.error(f"Error executing metadata hooks: {e}. Continuing with unmodified metadata.")
    logger.debug("Usage metadata (merged from context and kwargs, after hooks): {}".format(usage_metadata))
    return usage_metadata


def _with_forced_usage_reporting(kwargs):
    """Return a copy of ``kwargs`` asking LiteLLM to end the stream with a
    usage chunk, plus whether the caller had asked for that chunk themselves.

    Safe for every provider: LiteLLM exempts ``stream_options`` from provider
    param validation and drops it for providers that do not take it, while its
    own stream wrapper still appends the usage chunk.
    """
    caller_options = kwargs.get("stream_options")
    if not isinstance(caller_options, Mapping):
        caller_options = {}
    forced = dict(kwargs)
    forced["stream_options"] = {**caller_options, "include_usage": True}
    return forced, caller_options.get("include_usage") is True


def _prepare_call(args, kwargs):
    usage_metadata = _merged_usage_metadata(kwargs)
    request_time_dt = datetime.datetime.now(datetime.timezone.utc)
    logger.debug(
        "Calling LiteLLM with args: %s, kwargs: %s",
        sanitize_for_logging(args),
        sanitize_for_logging(kwargs),
    )
    is_streaming = bool(kwargs.get("stream", False))
    caller_requested_usage = True
    if is_streaming:
        kwargs, caller_requested_usage = _with_forced_usage_reporting(kwargs)
    return _PreparedCall(kwargs, usage_metadata, request_time_dt, is_streaming, caller_requested_usage)


def _call_owning_metering(wrapped, args, kwargs):
    token = _metered_call_in_progress.set(True)
    try:
        return wrapped(*args, **kwargs)
    finally:
        _metered_call_in_progress.reset(token)


async def _await_owning_metering(wrapped, args, kwargs):
    token = _metered_call_in_progress.set(True)
    try:
        return await wrapped(*args, **kwargs)
    finally:
        _metered_call_in_progress.reset(token)


def _meter_completion(response, call):
    if call.is_streaming:
        return handle_streaming_response(response, call.request_time_dt, call.usage_metadata,
                                         call.caller_requested_usage)
    return handle_response(response, call.request_time_dt, call.usage_metadata, False)


def completion_wrapper(wrapped, _, args, kwargs):
    if _should_pass_through():
        return wrapped(*args, **kwargs)
    call = _prepare_call(args, kwargs)
    return _meter_completion(_call_owning_metering(wrapped, args, call.kwargs), call)


def acompletion_wrapper(wrapped, _, args, kwargs):
    if _should_pass_through():
        return wrapped(*args, **kwargs)
    return _metered_acompletion(wrapped, args, kwargs)


async def _metered_acompletion(wrapped, args, kwargs):
    call = _prepare_call(args, kwargs)
    return _meter_completion(await _await_owning_metering(wrapped, args, call.kwargs), call)


def embedding_wrapper(wrapped, _, args, kwargs):
    if _should_pass_through():
        return wrapped(*args, **kwargs)
    call = _prepare_call(args, kwargs)
    response = _call_owning_metering(wrapped, args, call.kwargs)
    return handle_response(response, call.request_time_dt, call.usage_metadata, False, EMBED_OPERATION)


def aembedding_wrapper(wrapped, _, args, kwargs):
    if _should_pass_through():
        return wrapped(*args, **kwargs)
    return _metered_aembedding(wrapped, args, kwargs)


async def _metered_aembedding(wrapped, args, kwargs):
    call = _prepare_call(args, kwargs)
    response = await _await_owning_metering(wrapped, args, call.kwargs)
    return handle_response(response, call.request_time_dt, call.usage_metadata, False, EMBED_OPERATION)


_WRAPPERS_BY_ENTRY_POINT = {
    **{name: completion_wrapper for name in COMPLETION_ENTRY_POINTS},
    **{name: acompletion_wrapper for name in ASYNC_COMPLETION_ENTRY_POINTS},
    **{name: embedding_wrapper for name in EMBEDDING_ENTRY_POINTS},
    **{name: aembedding_wrapper for name in ASYNC_EMBEDDING_ENTRY_POINTS},
}

for _entry_point, _wrapper in _WRAPPERS_BY_ENTRY_POINT.items():
    if register_patch(f"litellm.{_entry_point}"):
        wrapt.wrap_function_wrapper("litellm", _entry_point, _wrapper)


def _warn_stream_without_usage():
    global _stream_without_usage_warned
    if _stream_without_usage_warned:
        return
    _stream_without_usage_warned = True
    logger.warning(
        "A LiteLLM stream ended without reporting usage although stream_options.include_usage was "
        "requested, so the call was not metered. Further streams like it are skipped without this warning."
    )


DELTA_CONTENT_FIELDS = ("content", "tool_calls", "function_call", "reasoning_content", "thinking_blocks", "audio")


def _field(obj, name):
    if isinstance(obj, Mapping):
        return obj.get(name)
    return getattr(obj, name, None)


def _choice_carries_content(choice):
    if _field(choice, "finish_reason"):
        return True
    delta = _field(choice, "delta")
    return any(_field(delta, name) for name in DELTA_CONTENT_FIELDS)


def _carries_content(chunk):
    return any(_choice_carries_content(choice) for choice in (getattr(chunk, "choices", None) or ()))


def _usage_litellm_attached(chunk):
    """The usage LiteLLM attaches to the final chunk of a stream sent without
    stream_options, or None when it is absent or all zero."""
    hidden_params = getattr(chunk, "_hidden_params", None)
    usage = hidden_params.get("usage") if isinstance(hidden_params, Mapping) else None
    if getattr(usage, "prompt_tokens", 0) or getattr(usage, "completion_tokens", 0):
        return usage
    return None


class _StreamMeter:
    """Collects a LiteLLM stream's chunks and meters the call once."""

    def __init__(self, request_time_dt, usage_metadata, caller_requested_usage):
        self._request_time_dt = request_time_dt
        self._usage_metadata = usage_metadata
        self._caller_requested_usage = caller_requested_usage
        self._last_chunk = None
        self._usage_chunk = None
        self._finished = False

    def observe(self, chunk):
        """Record ``chunk`` and return whether it belongs to the caller."""
        self._last_chunk = chunk
        if getattr(chunk, "usage", None) is None:
            return True
        self._usage_chunk = chunk
        # LiteLLM moves provider usage off content chunks and appends it as a
        # chunk of its own only when include_usage is set, so a usage-only chunk
        # exists because we asked for it unless the caller did too.
        return self._caller_requested_usage or _carries_content(chunk)

    def finish(self, completed):
        if self._finished:
            return
        self._finished = True
        if self._last_chunk is None:
            return
        usage = None if self._usage_chunk is not None else _usage_litellm_attached(self._last_chunk)
        if self._usage_chunk is None and usage is None and completed:
            _warn_stream_without_usage()
            return
        try:
            handle_response(self._usage_chunk or self._last_chunk, self._request_time_dt,
                            self._usage_metadata, True, usage=usage)
        except Exception as e:
            logger.warning("Error metering interrupted/completed stream: %s", e)


class _MeteredStream(wrapt.ObjectProxy):
    """A LiteLLM stream that meters itself once it ends, is closed, or is abandoned.

    It proxies the stream LiteLLM returned, so ``isinstance(stream,
    CustomStreamWrapper)`` still holds for the Router and the Responses bridge.
    """

    def __init__(self, stream, meter):
        super().__init__(stream)
        self._self_meter = meter

    def __iter__(self):
        return self

    def __next__(self):
        while True:
            try:
                chunk = next(self.__wrapped__)
            except StopIteration:
                self._self_meter.finish(completed=True)
                raise
            except Exception:
                self._self_meter.finish(completed=False)
                raise
            if self._self_meter.observe(chunk):
                return chunk

    def __aiter__(self):
        return self

    async def __anext__(self):
        while True:
            try:
                chunk = await self.__wrapped__.__anext__()
            except StopAsyncIteration:
                self._self_meter.finish(completed=True)
                raise
            except Exception:
                self._self_meter.finish(completed=False)
                raise
            if self._self_meter.observe(chunk):
                return chunk

    def close(self):
        self._self_meter.finish(completed=False)
        closer = getattr(self.__wrapped__, "close", None)
        if not callable(closer):
            return
        try:
            closer()
        except Exception as e:
            logger.debug("Error closing LiteLLM stream: %s", e)

    async def aclose(self):
        self._self_meter.finish(completed=False)
        closer = getattr(self.__wrapped__, "aclose", None)
        if not callable(closer):
            return
        try:
            await closer()
        except Exception as e:
            logger.debug("Error closing async LiteLLM stream: %s", e)

    def __del__(self):
        # An abandoned stream (break, then garbage collection) never reaches
        # StopIteration; finish() only schedules the metering thread.
        try:
            self._self_meter.finish(completed=False)
        except Exception:
            pass


def handle_streaming_response(stream, request_time_dt, usage_metadata, caller_requested_usage=True):
    """Wrap a LiteLLM stream so the call is metered once, from its usage chunk."""
    return _MeteredStream(stream, _StreamMeter(request_time_dt, usage_metadata, caller_requested_usage))


def handle_response(response, request_time_dt, usage_metadata, is_streaming, operation_type=CHAT_OPERATION,
                    usage=None):
    """
    Process a complete response (either streaming or non-streaming) and send metering data.
    Returns the original response.
    """
    if get_client() is None:
        return response  # metering disabled (no API key configured)

    async def metering_call():
        response_time_dt = datetime.datetime.now(datetime.timezone.utc)
        response_time = response_time_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        request_duration = (response_time_dt - request_time_dt).total_seconds() * 1000
        request_time = request_time_dt.strftime("%Y-%m-%dT%H:%M:%SZ")

        # Generate a unique ID if not present in response
        response_id = getattr(response, 'id', None) or f"litellm_client-{datetime.datetime.now().timestamp()}"

        # Extract token counts from LiteLLM response
        reported_usage = usage if usage is not None else getattr(response, 'usage', None)
        prompt_tokens = getattr(reported_usage, 'prompt_tokens', 0) or 0
        completion_tokens = getattr(reported_usage, 'completion_tokens', 0) or 0
        cache_read_tokens, cache_creation_tokens = extract_cache_tokens(reported_usage)
        total_tokens = prompt_tokens + completion_tokens

        logger.debug(
            "LiteLLM completion token usage - prompt: %d, completion: %d, "
            "cache read: %d, cache creation: %d, total: %d",
            prompt_tokens, completion_tokens, cache_read_tokens,
            cache_creation_tokens, total_tokens
        )

        finish_reason = getattr(response, 'finish_reason', None)

        finish_reason_map = {
            "stop": "END",
            "length": "TOKEN_LIMIT",
            "error": "ERROR",
            "cancelled": "CANCELLED",
            "tool_calls": "END_SEQUENCE",
            "function_calls": "END_SEQUENCE"
        }

        stop_reason = finish_reason_map.get(finish_reason, "END")  # type: ignore
        try:
            if shutdown_event.is_set():
                logger.warning("Skipping metering call during shutdown")
                return
            logger.debug("Metering call to Revenium for completion %s", response_id)

            # Create subscriber object from usage metadata
            subscriber = extract_subscriber_from_metadata(usage_metadata)

            organization_name, product_name = extract_org_and_product(usage_metadata)
            meta = extract_common_metadata(usage_metadata)
            agentic_fields = extract_agentic_job_fields(usage_metadata)
            extra_body = merge_extra_body(None, agentic_fields)

            completion_args = {
                "cache_creation_token_count": cache_creation_tokens,
                "cache_read_token_count": cache_read_tokens,
                "input_token_cost": None,
                "output_token_cost": None,
                "total_cost": None,
                "output_token_count": completion_tokens,
                "cost_type": "AI",
                "model": getattr(response, 'model', 'litellm-model'),
                "input_token_count": prompt_tokens,
                "provider": "LITELLM",
                "model_source": "LITELLM",
                "reasoning_token_count": 0,
                "request_time": request_time,
                "response_time": response_time,
                "completion_start_time": response_time,
                "request_duration": int(request_duration),
                "stop_reason": stop_reason,
                "total_token_count": total_tokens,
                "transaction_id": response_id,
                "trace_id": meta["trace_id"],
                "task_type": meta["task_type"],
                "subscriber": subscriber if subscriber else None,
                "organization_name": organization_name,
                "subscription_id": meta["subscription_id"],
                "product_name": product_name,
                "agent": meta["agent"],
                "response_quality_score": meta["response_quality_score"],
                "is_streamed": is_streaming,
                "operation_type": operation_type,
                "system_fingerprint": getattr(response, 'system_fingerprint', None),
                "middleware_source": "PYTHON"
            }

            # Add trace visualization fields (v0.3.0+)
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

            # Trace type field (with validation)
            trace_type = (
                usage_metadata.get("trace_type") or
                usage_metadata.get("traceType") or
                trace_fields.get_trace_type()
            )
            if trace_type:
                if usage_metadata.get("trace_type") or usage_metadata.get("traceType"):
                    trace_type = trace_fields.validate_trace_type(trace_type)
                if trace_type:
                    completion_args["trace_type"] = trace_type

            # Trace name field (with validation)
            trace_name = (
                usage_metadata.get("trace_name") or
                usage_metadata.get("traceName") or
                trace_fields.get_trace_name()
            )
            if trace_name:
                if usage_metadata.get("trace_name") or usage_metadata.get("traceName"):
                    trace_name = trace_fields.validate_trace_name(trace_name)
                if trace_name:
                    completion_args["trace_name"] = trace_name

            # Ticket ID field (FRONT-1545)
            ticket_id = trace_fields.get_ticket_id(usage_metadata)
            if ticket_id:
                completion_args["ticket_id"] = ticket_id

            agent_version = trace_fields.get_agent_version(usage_metadata)
            if agent_version:
                completion_args["agent_version"] = agent_version

            # Reasoning effort level (caller-supplied, forwarded verbatim)
            completion_args.update(extract_effort_field(usage_metadata))
            completion_args.update(extract_prompt_context_fields(usage_metadata))

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

            # Log the arguments at debug level
            logger.debug("Calling client.ai.create_completion with args: %s", completion_args)

            # The client.ai.create_completion method is not async, so don't use await
            result = submit_ai_event("completion", {**completion_args, "extra_body": extra_body})
            logger.debug("Metering call result: %s", result)
        except Exception as e:
            if not shutdown_event.is_set():
                logger.warning(f"Error in metering call: {str(e)}")
                # Log the full traceback for better debugging
                import traceback
                logger.warning(f"Traceback: {traceback.format_exc()}")

    logger.debug("Handling LiteLLM response: {}".format(response))
    thread = run_async_in_thread(metering_call())
    logger.debug("Metering thread started: %s", thread)
    
    # Return the original response
    return response
