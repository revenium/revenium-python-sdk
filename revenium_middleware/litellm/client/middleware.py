import wrapt
import httpx
import inspect
import openai
import litellm
import logging
import datetime
from collections.abc import Mapping
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from types import SimpleNamespace
from typing import Any, Dict, NamedTuple
from revenium_middleware import client, get_client, run_async_in_thread, shutdown_event
from revenium_middleware._core.call_ownership import (
    LITELLM, claim_call_for_transport, owned_by_outer_wrapper, peek_callback_metadata, take_callback_metadata
)
from revenium_middleware._core.context import merge_metadata, next_idempotency_key, overlay_metadata, snapshot_metadata
from revenium_middleware._core.langchain_settings import langchain_usage_metadata
from revenium_middleware._core.enforcement import (
    check_enforcement, checked_by_outer_wrapper, is_bypass_enabled, is_circuit_breaker_enabled
)
from revenium_middleware._core.subscriber import extract_subscriber_from_metadata
from revenium_middleware._core.fields import extract_org_and_product, extract_common_metadata, extract_agentic_job_fields, extract_effort_field, extract_prompt_context_fields, extract_skill_fields, merge_extra_body
from revenium_middleware._core.cache_tokens import (
    billable_input_tokens,
    cache_creation_inside_litellm_prompt,
    extract_cache_creation_ttl_counts,
    extract_cache_tokens,
    prompt_tokens_fold_cache,
    without_openai_cache_writes,
)
from revenium_middleware._core.config import Config, is_capture_prompts_enabled, is_selective_metering_enabled
from revenium_middleware._core.perplexity_cost import reported_cost_fields
from revenium_middleware._core.usage_iterations import (
    TOTALS_ARE_EXECUTOR_ONLY,
    TOTALS_UNRECOGNISED,
    billed_iteration_usage,
    billed_iterations,
    executor_cache_creation_ttl_counts,
    iteration_totals_shape,
    submit_iteration_records,
)
from revenium_middleware._core.prompt_extraction import extract_prompt_data_if_enabled
from revenium_middleware._core.failed_calls import (
    ERROR_STOP_REASON, exception_in_flight, failed_transaction_id, failure_of, raised_from_any,
)
from revenium_middleware._core.context import is_inside_decorated_function
from revenium_middleware._core import submit_ai_event
from revenium_middleware._core.log_sanitize import sanitize_for_logging
from revenium_middleware._core.call_context import CallContext
from revenium_middleware._core.metering import current_client, recorded_under
from revenium_middleware._core.patch_registry import wrap_once
from revenium_middleware._core.selective_metering import reported_if_unmetered
from revenium_middleware.litellm.proxy._metering_owner import (
    LITELLM_PROVIDER, guardrail_meters_request, proxy_meters_request
)
from .context import metadata_context
from .hooks import execute_metadata_hooks
from . import trace_fields

logger = logging.getLogger("revenium_middleware.extension")

COMPLETION_ENTRY_POINTS = ("completion", "text_completion", "completion_with_retries")
ASYNC_COMPLETION_ENTRY_POINTS = ("acompletion", "atext_completion", "acompletion_with_retries")
EMBEDDING_ENTRY_POINTS = ("embedding",)
ASYNC_EMBEDDING_ENTRY_POINTS = ("aembedding",)
UNRECORDED_ENTRY_POINTS = (
    "image_generation", "aimage_generation", "image_edit", "aimage_edit", "transcription", "atranscription",
    "video_generation", "avideo_generation", "search", "asearch",
)

CHAT_OPERATION = "CHAT"
EMBED_OPERATION = "EMBED"

PERPLEXITY_UPSTREAM = "perplexity"
OPENAI_COMPATIBLE_UPSTREAMS = frozenset({"openai", "openai_like", "custom_openai"})

_metered_call_in_progress: ContextVar[bool] = ContextVar("revenium_litellm_metered_call", default=False)
_stream_without_usage_warned = False


class _PreparedCall(NamedTuple):
    kwargs: Dict[str, Any]
    usage_metadata: Dict[str, Any]
    request_time_dt: datetime.datetime
    is_streaming: bool
    caller_requested_usage: bool
    model: Any = None
    metering_client: Any = None


def _selectively_unmetered(metered_by_proxy, kwargs):
    skipped = is_selective_metering_enabled() and not is_inside_decorated_function()
    if skipped and not metered_by_proxy(kwargs):
        reported_if_unmetered(True)
    return skipped


def _take_explicit_metadata(kwargs):
    """Pop the Revenium-only ``usage_metadata`` keyword, which LiteLLM would forward to the provider."""
    explicit_metadata = kwargs.pop("usage_metadata", None)
    return explicit_metadata if isinstance(explicit_metadata, Mapping) else {}


def _merged_usage_metadata(explicit_metadata, callback_metadata=None):
    call_metadata = overlay_metadata(metadata_context.get(), explicit_metadata)
    usage_metadata = merge_metadata(overlay_metadata(callback_metadata, call_metadata))
    try:
        usage_metadata = snapshot_metadata(execute_metadata_hooks(usage_metadata))
    except Exception as e:
        logger.error(f"Error executing metadata hooks: {e}. Continuing with unmodified metadata.")
    logger.debug("Usage metadata (merged from context and kwargs, after hooks): %s", sanitize_for_logging(usage_metadata))
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


def _prepare_call(args, kwargs, callback_metadata=None, model=None):
    explicit_metadata = _take_explicit_metadata(kwargs)
    usage_metadata = _merged_usage_metadata(explicit_metadata, callback_metadata)
    if _checks_caps_here(kwargs):
        check_enforcement(_caps_identity(usage_metadata, explicit_metadata), model=_cap_check_model(model, kwargs),
                          provider=LITELLM_PROVIDER)
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
    return _PreparedCall(kwargs, usage_metadata, request_time_dt, is_streaming, caller_requested_usage, model,
                         current_client())


def _checks_caps_here(kwargs):
    """Whether this wrapper runs the spending-cap check for the call: the
    circuit breaker is on and the call is not one the LiteLLM proxy is
    serving, whose cap check belongs to the proxy guardrail."""
    if not is_circuit_breaker_enabled() or is_bypass_enabled():
        return False
    return not isinstance(kwargs.get("proxy_server_request"), Mapping)


def _caps_identity(usage_metadata, explicit_metadata):
    """The identity the cap check sees. With no ``usage_metadata`` keyword, a
    caller named in LangChain's per-call settings stands in for it, as in the
    OpenAI wrapper, which stands aside for a LiteLLM call made through it."""
    if explicit_metadata:
        return usage_metadata
    return overlay_metadata(usage_metadata, langchain_usage_metadata())


def _check_unmetered_call(wrapped, args, kwargs):
    """Drop the caller's Revenium metadata from a call selective metering
    leaves unmetered and run the spending-cap check on it. No record is sent,
    so the LangChain caller's attribution is read without being taken."""
    explicit_metadata = _take_explicit_metadata(kwargs)
    if not _checks_caps_here(kwargs):
        return
    usage_metadata = _merged_usage_metadata(explicit_metadata, peek_callback_metadata(LITELLM))
    check_enforcement(_caps_identity(usage_metadata, explicit_metadata),
                      model=_cap_check_model(_requested_model(wrapped, args, kwargs), kwargs),
                      provider=LITELLM_PROVIDER)


def _call_unmetered(wrapped, args, kwargs):
    _check_unmetered_call(wrapped, args, kwargs)
    return _call_owning_metering(wrapped, args, kwargs)


async def _await_unmetered(wrapped, args, kwargs):
    _check_unmetered_call(wrapped, args, kwargs)
    return await _await_owning_metering(wrapped, args, kwargs)


def _inner_check_scope(kwargs):
    return checked_by_outer_wrapper() if _checks_caps_here(kwargs) else nullcontext()


def _call_owning_metering(wrapped, args, kwargs):
    token = _metered_call_in_progress.set(True)
    try:
        with owned_by_outer_wrapper(), _inner_check_scope(kwargs):
            return wrapped(*args, **kwargs)
    finally:
        _metered_call_in_progress.reset(token)


async def _await_owning_metering(wrapped, args, kwargs):
    token = _metered_call_in_progress.set(True)
    try:
        with owned_by_outer_wrapper(), _inner_check_scope(kwargs):
            return await wrapped(*args, **kwargs)
    finally:
        _metered_call_in_progress.reset(token)


def _prepare_completion(wrapped, args, kwargs):
    """A completion is the call a LangChain model makes, so it carries the
    attached callback's attribution."""
    return _prepare_call(args, kwargs, take_callback_metadata(LITELLM), _requested_model(wrapped, args, kwargs))


def _requested_model(wrapped, args, kwargs):
    """The ``model`` argument, wherever the entry point takes it: first for
    ``completion``, second for ``text_completion``, keyword-only for the rest.

    A keyword wins before binding, because an entry point taking ``**kwargs``
    binds ``model`` inside that dictionary rather than under its own name."""
    if "model" in kwargs:
        return kwargs["model"]
    try:
        return inspect.signature(wrapped).bind_partial(*args, **kwargs).arguments.get("model")
    except (TypeError, ValueError):
        return None


def _cap_check_model(model, kwargs):
    """The model as LiteLLM routes it: a bare name chosen by ``custom_llm_provider``
    takes that vendor's prefix, so the cap check sees which vendor serves it."""
    vendor = kwargs.get("custom_llm_provider")
    if not (isinstance(vendor, str) and vendor and isinstance(model, str)):
        return model
    if model.lower().startswith(f"{vendor.lower()}/"):
        return model
    return f"{vendor}/{model}"


def _upstream_provider(call):
    """The LiteLLM provider that served ``call``, resolved from the request.

    Read from the request because a non-streamed response does not carry it:
    ``_hidden_params["custom_llm_provider"]`` is set on stream chunks only.
    LiteLLM has just resolved the same arguments to make the call, so this
    resolution cannot newly fail; if it does, the response decides alone.
    """
    if not call.model:
        return None
    try:
        from litellm import get_llm_provider

        return get_llm_provider(
            call.model,
            custom_llm_provider=call.kwargs.get("custom_llm_provider"),
            api_base=call.kwargs.get("api_base"),
        )[1]
    except Exception as e:  # noqa: BLE001 - metering must never break the caller
        logger.debug("Could not resolve the LiteLLM provider for this call: %s", e)
        return None


def _served_by_perplexity(call, upstream_provider):
    """Whether Perplexity answered ``call``: through LiteLLM's Perplexity provider,
    or through an OpenAI-compatible route pointed at Perplexity's address. Any
    other provider is billed by its own tokens whatever its address says."""
    if upstream_provider == PERPLEXITY_UPSTREAM:
        return True
    api_base = call.kwargs.get("api_base")
    return upstream_provider in OPENAI_COMPATIBLE_UPSTREAMS and PERPLEXITY_UPSTREAM in str(api_base or "").lower()


def _captured_prompt_fields(request_kwargs, response=None, accumulated_content=None):
    """The prompt capture fields for a chat row, empty when capture is off.

    The same extraction the OpenAI integration records, since LiteLLM takes
    OpenAI-shaped messages and returns OpenAI-shaped responses.
    """
    system_prompt, input_messages, output_response, prompts_truncated = extract_prompt_data_if_enabled(
        request_kwargs, response=response, accumulated_content=accumulated_content
    )
    fields = {
        "system_prompt": system_prompt,
        "input_messages": input_messages,
        "output_response": output_response,
        "prompts_truncated": prompts_truncated,
    }
    return {name: value for name, value in fields.items() if value is not None}


def _iteration_request(request_kwargs):
    """The request fields a pass record reads: the advisor tool names the advisor's model.

    Never the whole LiteLLM call: its ``model`` is the routed name, such as
    ``anthropic/claude-...``, which no price matches.
    """
    tools = (request_kwargs or {}).get("tools")
    return {"tools": tools} if tools else None


def _meter_completion(response, call):
    upstream_provider = _upstream_provider(call)
    served_by_perplexity = _served_by_perplexity(call, upstream_provider)
    if call.is_streaming:
        with recorded_under(call.metering_client):
            metered = handle_streaming_response(response, call.request_time_dt, call.usage_metadata,
                                                call.caller_requested_usage, request_kwargs=call.kwargs,
                                                upstream_provider=upstream_provider, call=call,
                                                served_by_perplexity=served_by_perplexity)
    else:
        metered = handle_response(response, call.request_time_dt, call.usage_metadata, False,
                                  captured_prompts=_captured_prompt_fields(call.kwargs, response=response),
                                  upstream_provider=upstream_provider,
                                  iteration_request=_iteration_request(call.kwargs),
                                  served_by_perplexity=served_by_perplexity)
    if get_client() is not None:
        claim_call_for_transport(LITELLM, None if call.is_streaming else getattr(response, "id", None))
    return metered


_NO_SERVER_ANSWERED = (openai.APIConnectionError, httpx.TransportError)


def _failed_response(model):
    return SimpleNamespace(id=failed_transaction_id(), model=model,
                           usage=SimpleNamespace(prompt_tokens=0, completion_tokens=0, total_tokens=0))


def _record_failure(failure, call, last_chunk=None, usage=None):
    """Send the ERROR record of a failed call, under the metering client
    current when the call started: from the chunks a broken-off stream
    delivered, or with zero usage when nothing arrived."""
    if failure is None:
        return
    try:
        with recorded_under(call.metering_client):
            handle_response(last_chunk or _failed_response(call.model), call.request_time_dt, call.usage_metadata,
                            call.is_streaming, usage=usage, failure=failure)
    except Exception as e:
        logger.warning("Could not record a failed LiteLLM call: %s", e)


def _failure_of(error, outer):
    """LiteLLM gives a connection error a status of 500 and a timeout one of
    408 that no server answered, so neither names an HTTP status. On its
    ``openai/`` and ``anthropic/`` providers it raises a refused connection as
    an ``InternalServerError``, with the connection error only in its chain."""
    if raised_from_any(error, outer, _NO_SERVER_ANSWERED):
        return failure_of(error, outer, status=None)
    return failure_of(error, outer)


@contextmanager
def _recording_failure(call):
    """Record a call the provider refused, then let the caller's exception through unchanged."""
    outer = exception_in_flight()
    try:
        yield
    except Exception as error:
        _record_failure(_failure_of(error, outer), call)
        raise


def _stream_for_caller(stream, caller_requested_usage):
    if caller_requested_usage:
        return stream
    return _ObservedStream(stream, _ForcedUsageHider())


def _leave_to_guardrail(wrapped, args, kwargs):
    """Make a call the proxy guardrail or the deprecated callback meters, without metering it here.

    Both meter a stream from the response LiteLLM assembles, which holds the
    provider's token counts only if the stream ended with a usage chunk, and
    a proxy configured with ``always_include_stream_usage: false`` does not
    ask for one. So the stream still gets the usage request this wrapper adds
    to the streams it meters itself, hidden from the caller the same way.
    """
    with owned_by_outer_wrapper():
        if not kwargs.get("stream"):
            return wrapped(*args, **kwargs)
        forced_kwargs, caller_requested_usage = _with_forced_usage_reporting(kwargs)
        return _stream_for_caller(wrapped(*args, **forced_kwargs), caller_requested_usage)


async def _aleave_to_guardrail(wrapped, args, kwargs):
    with owned_by_outer_wrapper():
        if not kwargs.get("stream"):
            return await wrapped(*args, **kwargs)
        forced_kwargs, caller_requested_usage = _with_forced_usage_reporting(kwargs)
        return _stream_for_caller(await wrapped(*args, **forced_kwargs), caller_requested_usage)


def completion_wrapper(wrapped, _, args, kwargs):
    if _metered_call_in_progress.get():
        return wrapped(*args, **kwargs)
    if _selectively_unmetered(proxy_meters_request, kwargs):
        return _call_unmetered(wrapped, args, kwargs)
    if proxy_meters_request(kwargs):
        return _leave_to_guardrail(wrapped, args, kwargs)
    call = _prepare_completion(wrapped, args, kwargs)
    with _recording_failure(call):
        response = _call_owning_metering(wrapped, args, call.kwargs)
    return _meter_completion(response, call)


def acompletion_wrapper(wrapped, _, args, kwargs):
    if _metered_call_in_progress.get():
        return wrapped(*args, **kwargs)
    if _selectively_unmetered(proxy_meters_request, kwargs):
        return _await_unmetered(wrapped, args, kwargs)
    if proxy_meters_request(kwargs):
        return _aleave_to_guardrail(wrapped, args, kwargs)
    return _metered_acompletion(wrapped, args, kwargs)


async def _metered_acompletion(wrapped, args, kwargs):
    call = _prepare_completion(wrapped, args, kwargs)
    with _recording_failure(call):
        response = await _await_owning_metering(wrapped, args, call.kwargs)
    return _meter_completion(response, call)


def embedding_wrapper(wrapped, _, args, kwargs):
    if _metered_call_in_progress.get():
        return wrapped(*args, **kwargs)
    if _selectively_unmetered(guardrail_meters_request, kwargs):
        return _call_unmetered(wrapped, args, kwargs)
    if guardrail_meters_request(kwargs):
        return _leave_to_guardrail(wrapped, args, kwargs)
    call = _prepare_call(args, kwargs, model=_requested_model(wrapped, args, kwargs))
    response = _call_owning_metering(wrapped, args, call.kwargs)
    return handle_response(response, call.request_time_dt, call.usage_metadata, False, EMBED_OPERATION)


def aembedding_wrapper(wrapped, _, args, kwargs):
    if _metered_call_in_progress.get():
        return wrapped(*args, **kwargs)
    if _selectively_unmetered(guardrail_meters_request, kwargs):
        return _await_unmetered(wrapped, args, kwargs)
    if guardrail_meters_request(kwargs):
        return _aleave_to_guardrail(wrapped, args, kwargs)
    return _metered_aembedding(wrapped, args, kwargs)


async def _metered_aembedding(wrapped, args, kwargs):
    call = _prepare_call(args, kwargs, model=_requested_model(wrapped, args, kwargs))
    response = await _await_owning_metering(wrapped, args, call.kwargs)
    return handle_response(response, call.request_time_dt, call.usage_metadata, False, EMBED_OPERATION)


_WRAPPERS_BY_ENTRY_POINT = {
    **{name: completion_wrapper for name in COMPLETION_ENTRY_POINTS},
    **{name: acompletion_wrapper for name in ASYNC_COMPLETION_ENTRY_POINTS},
    **{name: embedding_wrapper for name in EMBEDDING_ENTRY_POINTS},
    **{name: aembedding_wrapper for name in ASYNC_EMBEDDING_ENTRY_POINTS},
}

for _entry_point, _wrapper in _WRAPPERS_BY_ENTRY_POINT.items():
    wrap_once(f"litellm.{_entry_point}", "litellm", _entry_point, _wrapper)


def unrecorded_call_wrapper(wrapped, _, args, kwargs):
    """Drop the Revenium-only ``usage_metadata`` keyword from a call the SDK does not record.

    LiteLLM forwards keywords it does not know inside the provider request.
    """
    kwargs.pop("usage_metadata", None)
    return wrapped(*args, **kwargs)


# Older LiteLLM releases the package still supports lack some of these calls.
for _entry_point in UNRECORDED_ENTRY_POINTS:
    if hasattr(litellm, _entry_point):
        wrap_once(f"litellm.{_entry_point}", "litellm", _entry_point, unrecorded_call_wrapper)


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


def _belongs_to_caller(chunk, caller_requested_usage):
    if getattr(chunk, "usage", None) is None:
        return True
    # LiteLLM moves provider usage off content chunks and appends it as a
    # chunk of its own only when include_usage is set, so a usage-only chunk
    # exists because we asked for it unless the caller did too.
    return caller_requested_usage or _carries_content(chunk)


class _ForcedUsageHider:
    """Hides the usage chunk this wrapper requested from a stream the proxy guardrail meters."""

    def observe(self, chunk):
        return _belongs_to_caller(chunk, False)

    def finish(self, completed, error=None):
        pass


def _reply_text(chunk):
    choices = getattr(chunk, "choices", None) or ()
    content = _field(_field(choices[0], "delta"), "content") if choices else None
    return content if isinstance(content, str) else ""


class _ReplyCollector:
    """Collects a streamed reply for prompt capture, keeping one character past the capture limit.

    The extra character is what tells the extraction the reply was cut, so
    the row is truncated and flagged exactly as an unbounded reply would be.
    """

    def __init__(self):
        self._parts = []
        self._room = Config.MAX_PROMPT_LENGTH + 1

    def add(self, text):
        if self._room > 0 and text:
            kept = text[:self._room]
            self._parts.append(kept)
            self._room -= len(kept)

    def take(self):
        """Return the collected reply and release it."""
        text = "".join(self._parts)
        self._parts = []
        return text


def _iterations_litellm_attached(chunk):
    """The ``usage.iterations`` LiteLLM keeps on a chunk's hidden usage, which its usage chunk drops."""
    hidden_params = getattr(chunk, "_hidden_params", None)
    usage = hidden_params.get("usage") if isinstance(hidden_params, Mapping) else None
    iterations = _field(usage, "iterations")
    return iterations if isinstance(iterations, list) and iterations else None


def _iterations_the_stream_accumulated(stream):
    """The ``iterations`` LiteLLM's Anthropic stream iterator gathered from ``message_delta``.

    An async stream attaches them to no chunk, so this LiteLLM internal is the
    only place they survive; any other route or release leaves it absent.
    """
    cumulative = getattr(getattr(stream, "completion_stream", None), "_cumulative_usage", None)
    iterations = cumulative.get("iterations") if isinstance(cumulative, Mapping) else None
    return iterations if isinstance(iterations, list) and iterations else None


class _StreamMeter:
    """Collects a LiteLLM stream's chunks and meters the call once."""

    def __init__(self, request_time_dt, usage_metadata, caller_requested_usage, request_kwargs=None,
                 upstream_provider=None, stream=None, call=None, served_by_perplexity=False):
        self._call = call
        self._served_by_perplexity = served_by_perplexity
        self._stream = stream
        self._iterations = None
        self._request_time_dt = request_time_dt
        self._metering_client = current_client()
        self._call_context = CallContext()
        self._upstream_provider = upstream_provider
        self._usage_metadata = usage_metadata
        self._caller_requested_usage = caller_requested_usage
        self._request_kwargs = request_kwargs
        self._reply = _ReplyCollector() if request_kwargs and is_capture_prompts_enabled() else None
        self._last_chunk = None
        self._usage_chunk = None
        self._finished = False
        self._outer = exception_in_flight()

    def observe(self, chunk):
        """Record ``chunk`` and return whether it belongs to the caller."""
        self._last_chunk = chunk
        if getattr(chunk, "usage", None) is not None:
            self._usage_chunk = chunk
        self._iterations = _iterations_litellm_attached(chunk) or self._iterations
        if self._reply is not None:
            self._reply.add(_reply_text(chunk))
        return _belongs_to_caller(chunk, self._caller_requested_usage)

    def _captured_prompts(self):
        if self._reply is None:
            return {}
        return _captured_prompt_fields(self._request_kwargs, accumulated_content=self._reply.take())

    def finish(self, completed, error=None):
        """Meter the call once. A stream left unfinished is metered from what
        arrived; one broken off by ``error`` is recorded as an error."""
        if self._finished:
            return
        self._finished = True
        self._call_context.run(self._meter, completed, error)

    def _meter(self, completed, error):
        failure = _failure_of(error, self._outer) if error is not None and self._call is not None else None
        usage = None if self._usage_chunk is not None else _usage_litellm_attached(self._last_chunk)
        if failure is not None:
            _record_failure(failure, self._call, self._usage_chunk or self._last_chunk, usage)
            return
        if self._last_chunk is None:
            return
        if self._usage_chunk is None and usage is None and completed:
            _warn_stream_without_usage()
            return
        try:
            with recorded_under(self._metering_client):
                handle_response(self._usage_chunk or self._last_chunk, self._request_time_dt,
                                self._usage_metadata, True, usage=usage, captured_prompts=self._captured_prompts(),
                                upstream_provider=self._upstream_provider,
                                usage_iterations=self._iterations or _iterations_the_stream_accumulated(self._stream),
                                iteration_request=_iteration_request(self._request_kwargs),
                                served_by_perplexity=self._served_by_perplexity
                                and (completed or self._usage_chunk is not None))
        except Exception as e:
            logger.warning("Error metering interrupted/completed stream: %s", e)


class _ObservedStream(wrapt.ObjectProxy):
    """A LiteLLM stream that shows each chunk to an observer and tells it when the stream ends.

    The observer decides which chunks reach the caller; a ``_StreamMeter``
    meters the call once the stream ends, is closed, or is abandoned.

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
            except Exception as error:
                self._self_meter.finish(completed=False, error=error)
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
            except Exception as error:
                self._self_meter.finish(completed=False, error=error)
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


def handle_streaming_response(stream, request_time_dt, usage_metadata, caller_requested_usage=True,
                              request_kwargs=None, upstream_provider=None, call=None, served_by_perplexity=False):
    """Wrap a LiteLLM stream so the call is metered once, from its usage chunk.

    With ``call``, a stream broken off by an exception is recorded as an error."""
    return _ObservedStream(
        stream, _StreamMeter(request_time_dt, usage_metadata, caller_requested_usage, request_kwargs,
                             upstream_provider, stream, call, served_by_perplexity)
    )


class _MainRecordUsage(NamedTuple):
    prompt_tokens: int
    completion_tokens: int
    cache_read_tokens: int
    cache_creation_tokens: int
    cache_creation_ttl_counts: Dict[str, int]
    billed_passes: Any = None


def _main_record_usage(reported_usage, usage_iterations=None):
    """The counts the main record carries, and the passes to send as records of their own.

    LiteLLM sums every advisor and compaction pass into its totals. Those are
    taken back out, so each pass is priced once, at its own model, on its own
    record. Totals of a shape this cannot account for are sent as reported,
    with no pass records, so a pass is never billed twice.
    """
    cache_read_tokens, cache_creation_tokens = extract_cache_tokens(reported_usage)
    reported = _MainRecordUsage(
        getattr(reported_usage, 'prompt_tokens', 0) or 0,
        getattr(reported_usage, 'completion_tokens', 0) or 0,
        cache_read_tokens,
        cache_creation_tokens,
        extract_cache_creation_ttl_counts(reported_usage),
    )
    iterations = usage_iterations if usage_iterations is not None else getattr(reported_usage, 'iterations', None)
    if not billed_iterations(iterations):
        return reported
    shape = iteration_totals_shape(reported.prompt_tokens, reported.completion_tokens, iterations)
    if shape == TOTALS_UNRECOGNISED:
        logger.debug("LiteLLM usage totals match none of its %d iterations' sums; sending one record",
                     len(iterations))
        return reported
    if shape == TOTALS_ARE_EXECUTOR_ONLY:
        return reported._replace(billed_passes=iterations)
    billed = billed_iteration_usage(iterations)
    return _MainRecordUsage(
        max(reported.prompt_tokens - billed.gross_prompt_tokens, 0),
        max(reported.completion_tokens - billed.output_tokens, 0),
        max(reported.cache_read_tokens - billed.cache_read_tokens, 0),
        max(reported.cache_creation_tokens - billed.cache_creation_tokens, 0),
        executor_cache_creation_ttl_counts(iterations) or reported.cache_creation_ttl_counts,
        iterations,
    )


def _submit_main_record(payload, idempotency_key):
    """Send the main record; a key is drawn only when passes are billed, so their keys derive from it.

    Every other call leaves the draw to submit_ai_event, as before.
    """
    if idempotency_key is None:
        return submit_ai_event("completion", payload)
    return submit_ai_event("completion", payload, idempotency_key=idempotency_key)


def handle_response(response, request_time_dt, usage_metadata, is_streaming, operation_type=CHAT_OPERATION,
                    usage=None, captured_prompts=None, upstream_provider=None, usage_iterations=None,
                    iteration_request=None, failure=None, served_by_perplexity=False):
    """
    Process a complete response (either streaming or non-streaming) and send metering data.
    Returns the original response.

    ``captured_prompts`` holds the prompt capture fields for the row, which
    only a chat call with capture on has. ``upstream_provider`` is the LiteLLM
    provider resolved from the request; it decides, ahead of the response's
    own hidden params, whether ``prompt_tokens`` holds the cache tokens too.
    ``usage_iterations`` stands in for ``usage.iterations`` when a stream's
    usage chunk dropped them; ``iteration_request`` is what the pass records
    read the advisor's model from. ``failure`` marks the record of a failed
    call: stop reason ERROR, with its error reason and its own key.
    ``served_by_perplexity`` sends the cost Perplexity reported, whose total
    holds its request and search fees, instead of leaving it to token pricing.
    """
    if current_client() is None:
        return response  # metering disabled (no API key configured)

    response_time_dt = datetime.datetime.now(datetime.timezone.utc)
    response_time = response_time_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    request_duration = (response_time_dt - request_time_dt).total_seconds() * 1000
    request_time = request_time_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    response_id = getattr(response, 'id', None) or f"litellm_client-{response_time_dt.timestamp()}"

    async def metering_call():
        # Extract token counts from LiteLLM response
        reported_usage = usage if usage is not None else getattr(response, 'usage', None)
        main_usage = _main_record_usage(reported_usage, usage_iterations)
        prompt_tokens = main_usage.prompt_tokens
        completion_tokens = main_usage.completion_tokens
        cache_read_tokens = main_usage.cache_read_tokens
        cache_creation_tokens = main_usage.cache_creation_tokens
        total_tokens = prompt_tokens + completion_tokens
        upstream_sources = (
            {"custom_llm_provider": upstream_provider},
            getattr(response, "_hidden_params", None),
            # The caller called LiteLLM directly, so this is the deployment;
            # a stream's usage chunk names it nowhere else.
            {"model": getattr(response, "model", None)},
        )
        input_tokens = billable_input_tokens(
            without_openai_cache_writes(prompt_tokens, reported_usage),
            cache_read_tokens,
            cache_creation_inside_litellm_prompt(cache_creation_tokens, *upstream_sources),
            prompt_tokens_fold_cache(*upstream_sources),
        )

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

        stop_reason = ERROR_STOP_REASON if failure else finish_reason_map.get(finish_reason, "END")  # type: ignore
        main_payload = None
        main_key = None
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
                **main_usage.cache_creation_ttl_counts,
                "cache_read_token_count": cache_read_tokens,
                "input_token_cost": None,
                "output_token_cost": None,
                "total_cost": None,
                "output_token_count": completion_tokens,
                "cost_type": "AI",
                "model": getattr(response, 'model', 'litellm-model'),
                "input_token_count": input_tokens,
                "provider": LITELLM_PROVIDER,
                "model_source": LITELLM_PROVIDER,
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
                "middleware_source": "PYTHON",
                **(captured_prompts or {}),
            }
            if served_by_perplexity:
                completion_args.update(reported_cost_fields(reported_usage))

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
            completion_args.update(extract_skill_fields(usage_metadata))

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
            retry_number = trace_fields.get_retry_number(usage_metadata)
            completion_args["retry_number"] = retry_number

            # Log the arguments at debug level
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("Calling client.ai.create_completion with args: %s", sanitize_for_logging(completion_args))

            main_payload = {**completion_args, "extra_body": extra_body}
            if failure is not None:
                main_payload.update(failure.record_fields())
            main_key = failure.idempotency_key if failure else (
                next_idempotency_key() if main_usage.billed_passes else None)
            result = _submit_main_record(main_payload, main_key)
            logger.debug("Metering call result: %s", result)
        except Exception as e:
            if not shutdown_event.is_set():
                logger.warning(f"Error in metering call: {str(e)}")
                # Log the full traceback for better debugging
                import traceback
                logger.warning(f"Traceback: {traceback.format_exc()}")
        finally:
            submit_iteration_records(submit_ai_event, main_payload, main_usage.billed_passes, iteration_request,
                                     main_key)

    logger.debug("Handling LiteLLM response: %s", sanitize_for_logging(response))
    thread = run_async_in_thread(metering_call())
    logger.debug("Metering thread started: %s", thread)
    
    # Return the original response
    return response
