"""Revenium budget guardrail for the LiteLLM proxy.

``ReveniumGuardrail`` is a LiteLLM ``CustomGuardrail`` that enforces a budget
before the proxied call and meters usage after it:

* **pre_call** -- runs the SDK's own circuit breaker
  (``revenium_middleware._core.enforcement.check_enforcement``) against the
  request's subscriber/organization attribution and turns a
  ``BudgetExceededError`` into the HTTP 429 LiteLLM expects. Every other
  failure fails **open**. The check is safe to await on the proxy's event loop
  because it never waits on Revenium: it judges the request against the rules
  already cached and leaves refreshing them to the SDK's poller thread, so an
  unreachable, slow or misconfigured enforcement endpoint neither refuses a
  call nor delays it.
* **post_call** -- meters the completion (success and failure alike) through
  the SDK's metering client, exactly as the deprecated ``MiddlewareHandler``
  callback does: ``submit_ai_event`` dispatched via ``run_async_in_thread``,
  cache-token fields via ``extract_cache_tokens``, ``x-revenium-*`` header
  attribution, ``effort``, and ``agenticJob*`` tagging. Nothing this hook does
  can fail the proxied call -- the post-call hook runs in-band, so an
  unexpected response shape would otherwise turn a successful LLM call into a
  client-facing error.

Streaming takes two paths, because LiteLLM dispatches the end of a stream by
sniffing the response object
(``ProxyBaseLLMRequestProcessing._arm_deferred_stream_dispatch``, verified
against litellm 1.101.0):

* A ``/v1/chat/completions`` stream arrives as a ``CustomStreamWrapper``, and
  the vendor arms a closure that runs ``_run_deferred_stream_guardrails``,
  which calls ``async_post_call_success_hook`` with the assembled response and
  its full usage totals. So that route needs no streaming hook of ours. It
  must also not gain one: the vendor skips the success hook for any guardrail
  defining ``async_post_call_streaming_iterator_hook`` in its own ``__dict__``,
  assuming it already scanned the stream, so declaring an iterator hook here
  would *cost* us the assembled totals rather than gain us anything.
* A native ``anthropic_messages`` stream -- ``/v1/messages`` with
  ``stream: true``, which is what Claude Code speaks -- arrives as a plain
  async iterator, and the vendor arms a different closure that enqueues a
  ready-made logging coroutine. ``_run_deferred_stream_guardrails`` is never
  reached and ``async_post_call_success_hook`` never fires, so on that route
  this class meters from ``async_log_success_event``, the ``CustomLogger``
  surface the logging dispatch does reach. A streamed ``/v1/responses`` call
  (``aresponses``, what Codex CLI speaks) ends the same way, so it is metered
  there too. That method is gated to exactly these two streamed call types,
  because LiteLLM calls it for every completed request.

``tests/test_litellm/test_proxy_guardrail_streaming.py`` pins every part of
that contract, including the arming step itself.

Deliberate exclusion: the classification layer from the private
``revenium-litellm-guardrail`` package (``_jobs.py`` -- classifier model calls,
conversation arcs, ``job-taxonomy.json``, ``job-value-card.json``) is **not**
part of this module. Agentic-job attribution here comes only from what the
caller declares: ``x-revenium-agentic-job-*`` headers and virtual-key metadata.
Inferring a job by calling an LLM from inside a proxy hook is a product
decision with its own cost and latency profile, and it does not belong in the
SDK's metering path.

Usage in the LiteLLM proxy ``config.yaml``. ``guardrails`` is a **top-level**
key, not a member of ``litellm_settings``: the nested spelling reaches
LiteLLM's legacy v1 loader (``init_guardrails.initialize_guardrails``), which
expects a mapping and exits the proxy at startup with
``GuardrailItem() argument after ** must be a mapping, not str``::

    guardrails:
      - guardrail_name: "revenium"
        litellm_params:
          guardrail: revenium_middleware.litellm.proxy.guardrail.ReveniumGuardrail
          mode:
            - "pre_call"
            - "post_call"
          default_on: true

``mode`` must be a list to enable both hooks; a single string restricts the
guardrail to that one event type.
"""

import sys

if sys.version_info < (3, 10):  # pragma: no cover - guarded at import time
    raise ImportError(
        "revenium_middleware.litellm.proxy.guardrail requires Python 3.10 or "
        "newer, because it runs inside the LiteLLM proxy "
        '(pip install "revenium-python-sdk[litellm-proxy]"), which does not '
        "support older interpreters. The client-side integration "
        "(revenium_middleware.litellm.client) and the deprecated "
        "revenium_middleware.litellm.proxy.middleware callback remain "
        "available on Python 3.8+."
    )

import collections
import datetime
import importlib.metadata
import logging
import threading
import time
import uuid
from collections.abc import Mapping

from fastapi import HTTPException
from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.integrations.custom_logger import CustomLogger
from litellm.types.guardrails import GuardrailEventHooks

from revenium_middleware._core import get_client, run_async_in_thread, submit_ai_event
from revenium_middleware._core.cache_tokens import (
    billable_input_tokens,
    cache_creation_inside_litellm_prompt,
    extract_cache_creation_ttl_counts,
    extract_cache_tokens,
    get_usage_field,
    total_priced_tokens,
    without_openai_cache_writes,
)
from revenium_middleware._core.config import is_shared_call_id_enabled, shared_call_id_setting
from revenium_middleware._core.enforcement import check_enforcement, normalize_email
from revenium_middleware._core.exceptions import BudgetExceededError
from revenium_middleware._core.failed_calls import is_spending_cap_refusal
from revenium_middleware._core.fields import merge_extra_body
from revenium_middleware._core.log_sanitize import sanitize_for_logging

from ._metering_owner import LITELLM_PROVIDER, register_metering_guardrail
from .middleware import (
    ANTHROPIC_MESSAGES_CALL_TYPE,
    prompt_tokens_fold_cache,
    _extract_agentic_job_from_headers,
    _extract_organization_name,
    _extract_product_name,
    SUBSCRIBER_EMAIL_HEADER,
    SUBSCRIBER_ID_HEADER,
    extract_request_headers,
    format_utc_timestamp,
    header_then_metadata,
    key_team_alias,
    log_organization_pin_setting,
    pinned_key_team,
    present_or_none,
    read_shared_call_id,
    record_shared_call_id,
    resolve_transaction_id,
    trace_fields_from_headers,
)

logger = logging.getLogger("revenium_middleware.extension")

# Response class name -> the metering operation type it represents. LiteLLM
# hands the post-call hook whatever shape the endpoint returned, and only
# ModelResponse carries the ChatCompletion fields the payload below assumes.
# Anything absent here falls back to CHAT, which is how every request was
# treated before this map existed -- a wrong label is recoverable, an exception
# raised from this in-band hook is not.
_OPERATION_TYPES = {
    "ModelResponse": "CHAT",
    "EmbeddingResponse": "EMBED",
    "RerankResponse": "RERANK",
    "ImageResponse": "IMAGE",
    "TranscriptionResponse": "AUDIO",
}

# Virtual-key metadata keys carrying declared agentic-job fields, mapped to the
# Revenium REST field names. Headers win; these are the fallback for a proxy
# that pins a job on the key rather than on every request.
_KEY_METADATA_JOB_FIELDS = {
    "revenium_agentic_job_id": "agenticJobId",
    "revenium_agentic_job_name": "agenticJobName",
    "revenium_agentic_job_type": "agenticJobType",
    "revenium_agentic_job_version": "agenticJobVersion",
}

_POST_CALL_EVENT = "post_call"
_PRE_CALL_EVENT = "pre_call"

# The missing-id counter. Metering runs off run_async_in_thread, so this is
# touched from more than one thread and needs the lock.
_shared_call_id_lock = threading.Lock()
_shared_call_id_missing_count = 0
_shared_call_id_last_warned_at = None
_SHARED_CALL_ID_WARN_INTERVAL_SECONDS = 60.0


def _headers_hook_available():
    """True when this LiteLLM can return plugin-supplied response headers.

    Asked of the vendor base class, never of ``ReveniumGuardrail``, which
    always declares the method: asking our own class would report the shared id
    active on a LiteLLM that cannot deliver it, and the metered row's
    transaction id would change while the client received no header to copy.
    """
    return hasattr(CustomLogger, "async_post_call_response_headers_hook")


def _note_missing_shared_call_id():
    """Count a metered call that should have carried a minted id and did not.

    A pre-call hook that quietly stopped running looks exactly like a return of
    the double counting the flag was turned on to fix, with nothing in the
    proxy log to say so. Counted every time, said at most once a minute, with
    the cumulative count so one line carries the scale.

    Called only for a call the mint would have covered; see
    ``_was_eligible_for_a_minted_id``. Every other route reaching here without
    an id is the design working, not a fault, and saying so would be a false
    report of double counting.
    """
    global _shared_call_id_missing_count, _shared_call_id_last_warned_at
    now = time.monotonic()
    with _shared_call_id_lock:
        _shared_call_id_missing_count += 1
        count = _shared_call_id_missing_count
        last = _shared_call_id_last_warned_at
        if last is not None and now - last < _SHARED_CALL_ID_WARN_INTERVAL_SECONDS:
            return
        _shared_call_id_last_warned_at = now
    logger.warning(
        "ReveniumGuardrail: REVENIUM_LITELLM_SHARED_CALL_ID is on but no shared "
        "call id reached the metered row (%s so far). Claude Code's own usage "
        "record cannot be matched to this one, so the call is counted twice. "
        "Check that this guardrail's mode includes 'pre_call'.",
        count,
    )


# The LiteLLM call type both Anthropic-shaped routes carry, one of the two
# ``async_log_success_event`` meters (the set is
# ``_STREAMED_CALL_TYPES_METERED_FROM_THE_LOGGING_EVENT``). A literal rather than
# ``litellm.types.utils.CallTypes.anthropic_messages``: importing the vendor
# enum here would let a rename break proxy startup, which is a worse failure
# than a missed row. ``test_proxy_guardrail_streaming`` asserts the two still
# agree, so a rename fails a test instead of quietly stopping the metering.
_ANTHROPIC_MESSAGES_CALL_TYPE = "anthropic_messages"
_RESPONSES_CALL_TYPE = "aresponses"

# The streams whose end reaches only ``async_log_success_event``: an
# Anthropic-shaped one, and a Responses one on every provider (verified on
# litellm 1.104.2). LiteLLM bridges a Responses call to a provider without a
# native Responses API through ``litellm.acompletion``, but still logs it under
# ``aresponses``.
_STREAMED_CALL_TYPES_METERED_FROM_THE_LOGGING_EVENT = frozenset(
    {_ANTHROPIC_MESSAGES_CALL_TYPE, _RESPONSES_CALL_TYPE}
)

# The call types whose streams end in one of this guardrail's metering hooks:
# a chat stream through the post-call hook LiteLLM runs on the assembled
# response, the rest through ``async_log_success_event``. A streamed
# /v1/completions call reaches neither (verified on litellm 1.104.2), so for
# that the client wrapper is the only meter there is.
_STREAMED_CALL_TYPES_METERED_HERE = frozenset(
    {"acompletion"} | _STREAMED_CALL_TYPES_METERED_FROM_THE_LOGGING_EVENT
)

# How many recently metered calls the duplicate guard remembers. Each entry is
# one short id, so this costs a few hundred kilobytes and covers far more calls
# than a proxy has in flight at once. Forgetting an old entry can only ever
# allow a duplicate row, never suppress a real one.
_METERED_CALL_MEMORY = 4096

# The duplicate guard's key, and the source of a record's id when the response
# carries none: one value per request, written by this process's pre-call hook.
# The prefix is minted at startup, so a value a caller plants in their own
# request body is never mistaken for one, and every request's pre-call
# overwrites whatever was planted before either success path reads it. The
# failure hook also reads it as proof pre-call ran (``_reached_pre_call``), so
# pre-call must write it on every request it runs on.
_METER_KEY_FIELD = "revenium_meter_key"
_METER_KEY_PREFIX = uuid.uuid4().hex + ":"

# Revenium stores a record's model in a varchar(255) column.
_MODEL_TEXT_MAX_LENGTH = 255

# LiteLLM call type -> Revenium operation type, for the failure path, where
# there is no response object to read the shape off. Only the non-chat entries
# are listed: everything absent here is CHAT, which is both the common case and
# the safe default (a wrong label is recoverable, a missing row is not).
_CALL_TYPE_OPERATIONS = {
    "embedding": "EMBED",
    "aembedding": "EMBED",
    "rerank": "RERANK",
    "arerank": "RERANK",
    "image_generation": "IMAGE",
    "aimage_generation": "IMAGE",
    "image_edit": "IMAGE",
    "aimage_edit": "IMAGE",
    "transcription": "AUDIO",
    "atranscription": "AUDIO",
    "speech": "AUDIO",
    "aspeech": "AUDIO",
}

# Request-path suffix -> Revenium operation type, the fallback when no call
# type is recoverable. Deliberately our own table rather than LiteLLM's
# API_ROUTE_TO_CALL_TYPES: the operation vocabulary is Revenium's, and a vendor
# map that grows a route we have no operation for would silently mislabel it.
_ROUTE_OPERATIONS = (
    ("/embeddings", "EMBED"),
    ("/rerank", "RERANK"),
    ("/images/generations", "IMAGE"),
    ("/images/edits", "IMAGE"),
    ("/audio/transcriptions", "AUDIO"),
    ("/audio/speech", "AUDIO"),
)


def _as_mapping(value):
    """Return ``value`` when it reads like a mapping, otherwise an empty dict.

    LiteLLM fills ``hidden_params`` and ``metadata`` with whatever the calling
    route had; a non-mapping there must not abort an in-band hook.
    """
    return value if hasattr(value, "get") else {}


def _upstream_model(data):
    """The model the proxy called for this request, else the request's own ``model``.

    On a proxy the request's ``model`` is the operator's alias for a
    deployment, which Revenium's model catalog does not know, so a row named
    after it is priced at $0. LiteLLM records the deployment's model on the
    call's logging object. ``metadata`` is never read for it: on
    ``/v1/messages`` that is the caller's own request body, so a
    ``deployment`` there is whatever the caller claims.
    """
    call_details = _as_mapping(getattr(data.get("litellm_logging_obj"), "model_call_details", None))
    model = call_details.get("model")
    if not (isinstance(model, str) and model.strip()):
        model = data.get("model")
    if isinstance(model, str):
        return model[:_MODEL_TEXT_MAX_LENGTH]
    return model


def _resolve_usage(response):
    """Return the usage object carried by a response of any shape.

    Only chat-shaped responses reliably carry one: an ``ImageResponse`` may have
    none at all, and a dict-shaped payload exposes it by key rather than by
    attribute. Returning ``None`` lets the token accessors fall back to zero,
    which is the better failure -- the transaction still records a model, a cost
    type and a timestamp.
    """
    usage = getattr(response, "usage", None)
    if usage is not None:
        return usage
    try:
        return response["usage"]
    except (TypeError, KeyError, IndexError):
        return None


def _token_counts(usage, cache_read_tokens, cache_creation_tokens, upstream_sources):
    """Return ``(input, completion, total)`` for a usage object of any shape.

    ``input`` is what Revenium prices at the model's input rate; the cache
    counts are handed in rather than re-read, because the payload builder has
    already extracted them for the fields they are reported in, and the two
    must be the same numbers. ``upstream_sources`` are the LiteLLM surfaces
    that decide whether this upstream's prompt count contains them, and how
    much of the write -- see ``billable_input_tokens`` for what each wrong
    answer costs.

    Read through ``get_usage_field`` -- the same tolerant accessor the callback
    uses -- so attribute-style and dict-style usage objects agree, and a usage
    object reporting only the parts still yields a sensible total.

    The OpenAI spelling is preferred, and the Anthropic one is the fallback.
    A non-streamed ``/v1/messages`` call hands the post-call hook the
    provider's own JSON rather than a ``litellm.ModelResponse``, and its usage
    is spelled ``input_tokens`` / ``output_tokens``; reading only the OpenAI
    spelling zeroed every billable count on that route while the row itself
    looked healthy. Anthropic reports cache tokens *beside* ``input_tokens``,
    so that count is already the input-rate one whatever the caller answered --
    exactly what the direct Anthropic integration meters for the same call.
    """
    prompt = get_usage_field(usage, "prompt_tokens", 0) or 0
    if prompt:
        input_tokens = billable_input_tokens(
            without_openai_cache_writes(prompt, usage),
            cache_read_tokens,
            cache_creation_inside_litellm_prompt(cache_creation_tokens, *upstream_sources),
            prompt_tokens_fold_cache(*upstream_sources),
        )
    else:
        input_tokens = without_openai_cache_writes(get_usage_field(usage, "input_tokens", 0) or 0, usage)
    completion = get_usage_field(usage, "completion_tokens", 0) or 0
    total = get_usage_field(usage, "total_tokens", 0) or 0
    if not completion:
        completion = get_usage_field(usage, "output_tokens", 0) or 0
    return input_tokens, completion, total or total_priced_tokens(
        input_tokens, completion, cache_read_tokens, cache_creation_tokens
    )


def _elapsed_ms(start_time, end_time):
    """Milliseconds between two timestamps, or 0 when either is unusable.

    The logging event is handed the call's own start and end times, which is
    the only duration available on that path when the hidden params carry no
    ``_response_ms``.
    """
    try:
        return max((end_time - start_time).total_seconds() * 1000, 0)
    except (TypeError, AttributeError):
        return 0


def _stamp_meter_key(data):
    """Write a fresh duplicate-guard key into the metadata LiteLLM owns for the route.

    The failure hook also reads this stamp as proof pre-call ran
    (``_reached_pre_call``); without it a failed call is not recorded.

    Exactly one container. On the routes in LiteLLM's ``LITELLM_METADATA_ROUTES``
    (``/v1/responses``, ``/v1/messages``) its own metadata is
    ``litellm_metadata`` and ``metadata`` is the caller's body field, which
    ``/v1/responses`` forwards to the provider. Always assigned, never
    ``setdefault``: a planted value must not survive.
    """
    for key in ("litellm_metadata", "metadata"):
        container = data.get(key)
        if isinstance(container, Mapping):
            if hasattr(container, "__setitem__"):
                container[_METER_KEY_FIELD] = _new_meter_key()
            return
    if "litellm_metadata" not in data and "metadata" not in data:
        data["metadata"] = {_METER_KEY_FIELD: _new_meter_key()}


def _new_meter_key():
    """This process's prefix followed by a fresh UUID, the request's own id."""
    return _METER_KEY_PREFIX + str(uuid.uuid4())


def _meter_key(container):
    """The duplicate-guard key this process stamped on a request, or ``None``."""
    container = _as_mapping(container)
    for key in ("litellm_metadata", "metadata"):
        value = _as_mapping(container.get(key)).get(_METER_KEY_FIELD)
        if isinstance(value, str) and value.startswith(_METER_KEY_PREFIX):
            return value
    return None


def _reached_pre_call(request_data):
    """Whether the Revenium guardrail's pre-call ran on this request in this process.

    LiteLLM runs pre-call only after authentication. A pass-through route moves
    the stamped ``metadata`` into ``litellm_params``.
    """
    return (
        _meter_key(request_data) is not None
        or _meter_key(_as_mapping(request_data).get("litellm_params")) is not None
    )


def _request_transaction_id(meter_key):
    """The transaction id of a request's record when no provider id or minted id applies, or None.

    Only the per-request half goes on the wire. The prefix is what tells a key
    this process stamped from one a caller planted, so no record may reveal it.
    """
    return meter_key[len(_METER_KEY_PREFIX):] if meter_key else None


class _MeteredCalls:
    """The calls already metered, so no completed call is metered twice.

    LiteLLM hands the same completed call to more than one of our hooks:
    ``async_post_call_success_hook`` and ``async_log_success_event`` both fire
    on a streamed ``/v1/chat/completions`` call, and a provider whose
    ``anthropic_messages`` stream arrived as a ``CustomStreamWrapper`` would
    make both fire on the Anthropic route too. The route gate on the logging
    event covers the shapes seen today; this covers the ones that change
    underneath us.

    Keyed on the per-request value the pre-call hook stamps
    (``_stamp_meter_key``), never on LiteLLM's ``litellm_call_id``: LiteLLM
    takes that from the caller's ``x-litellm-call-id`` header unvalidated, so
    keying on it let any caller repeat one id and have every later call go
    unrecorded. Nor on the row's transaction id: the two hooks resolve that
    from different sources and can land on different values for one call.
    """

    def __init__(self, capacity=_METERED_CALL_MEMORY):
        self._capacity = capacity
        self._seen = collections.OrderedDict()
        self._lock = threading.Lock()

    def claim(self, call_id):
        """True when this caller is the first to meter ``call_id``.

        An empty id is always granted: with nothing to identify the call by,
        suppressing a row would risk discarding a real one.
        """
        if not call_id:
            return True
        with self._lock:
            if call_id in self._seen:
                return False
            self._seen[call_id] = None
            while len(self._seen) > self._capacity:
                self._seen.popitem(last=False)
            return True


def _call_type(request_kwargs):
    """The LiteLLM call type of the route serving ``request_kwargs``.

    Read off the proxy's logging object rather than inferred from the entry
    point, because the Anthropic-shaped route reaches ``litellm.acompletion``
    through a bridge that relabels the logging object as ``anthropic_messages``.
    """
    call_type = getattr(request_kwargs.get("litellm_logging_obj"), "call_type", None)
    return getattr(call_type, "value", call_type)


def _litellm_call_id(*sources):
    """LiteLLM's own correlation id, from the first source that carries one."""
    for source in sources:
        call_id = _as_mapping(source).get("litellm_call_id")
        if call_id:
            return call_id
    return None


def _meter_identity(response, shared_call_id=None, request_transaction_id=None):
    """Return ``(transaction_id, operation_type)`` for a response of any shape.

    ``EmbeddingResponse`` and friends have no ``.id`` -- reading it unguarded is
    what turned a successful ``/v1/embeddings`` call into a client-facing error,
    because a ``CustomGuardrail`` post-call hook runs in-band. So the id is
    resolved through a chain and, when nothing real survives it, a fresh UUID
    rather than a constant -- see ``resolve_transaction_id`` for why a shared
    sentinel is worse than no id at all.

    LiteLLM's ``litellm_call_id`` is never a link: LiteLLM takes it from the
    caller's ``x-litellm-call-id`` header unvalidated, and Revenium keeps one
    record per organization and transaction id, so a caller repeating one
    header would have every later id-less call merged into the first record.

    Args:
        shared_call_id: The id this process minted for this request, when the
            shared call id is on. It outranks the provider's own response id
            because that id never reaches Claude Code, so a row keyed on it can
            never collide with the telemetry row the customer is billed twice
            for.
        request_transaction_id: The request's own id from the key the pre-call
            hook stamped (``_request_transaction_id``), used when the response
            carries no id of its own. The logging event reads the same key, so
            when the response has no provider id the two paths file one call
            under one id.
    """
    return (
        resolve_transaction_id(
            shared_call_id, getattr(response, "id", None), request_transaction_id
        ),
        _OPERATION_TYPES.get(type(response).__name__, "CHAT"),
    )


def _operation_type_from_route(route):
    """Map a request path (or full URL) onto a Revenium operation type."""
    if not isinstance(route, str) or not route:
        return None
    # A full URL from metadata["endpoint"] carries a query string and a host;
    # only the path decides the operation.
    path = route.split("?", 1)[0].rstrip("/")
    for suffix, operation in _ROUTE_OPERATIONS:
        if path.endswith(suffix):
            return operation
    return None


# LiteLLM's native Anthropic messages route, the only route the mint runs on.
# The pass-through route ends in the same two segments but is a different call
# type (``pass_through_endpoint``) that never mints, and its own docstring
# names it ``{PROXY_BASE_URL}/anthropic/v1/messages``
# (``proxy/anthropic_endpoints/endpoints.py:99``), so it is excluded by name.
_ANTHROPIC_MESSAGES_PATH = "/v1/messages"
_ANTHROPIC_PASS_THROUGH_PATH = "/anthropic/v1/messages"


def _route_is_anthropic_messages(route):
    """True when a request path (or full URL) is the Anthropic messages route."""
    if not isinstance(route, str) or not route:
        return False
    # A full URL from metadata["endpoint"] carries a query string and a host;
    # only the path decides the route.
    path = route.split("?", 1)[0].rstrip("/")
    return path.endswith(_ANTHROPIC_MESSAGES_PATH) and not path.endswith(
        _ANTHROPIC_PASS_THROUGH_PATH
    )


def _was_eligible_for_a_minted_id(request_data, metadata, user_api_key_dict):
    """True when this call is one the pre-call hook would have minted an id for.

    The mint is gated on ``call_type == "anthropic_messages"``, but the success
    hook is handed no call type, so without this the missing-id counter would
    fire on every embedding, chat-completion and pass-through call on a proxy
    that also serves the Anthropic route, telling the operator those calls are
    counted twice when they were never part of the mechanism at all.

    Read from LiteLLM's own record of the request, never from the minted keys:
    those are absent in exactly the case the counter exists to report, a
    pre-call hook that quietly stopped running. Most authoritative first, the
    same two sources and the same order ``_failure_operation_type`` uses. When
    none of them resolves, the answer is no: a missing warning is a diagnostic
    gap, a false one sends an operator hunting a break that is not there.
    """
    call_type = _as_mapping(request_data.get("standard_logging_object")).get("call_type")
    if isinstance(call_type, str) and call_type:
        return call_type == ANTHROPIC_MESSAGES_CALL_TYPE
    for candidate in (
        metadata.get("endpoint"),
        getattr(user_api_key_dict, "request_route", None),
    ):
        if _route_is_anthropic_messages(candidate):
            return True
    return False


def _failure_operation_type(request_data, metadata, user_api_key_dict):
    """Operation type for a failed call, where there is no response to read.

    Hard-coding CHAT here mislabelled every failed embedding, rerank, image and
    transcription request -- the success path has read the response class since
    this module existed, so the two disagreed about the same endpoint. Three
    sources, most authoritative first: the call type LiteLLM recorded on the
    standard logging object, the endpoint recorded in request metadata, and the
    route the calling key was authorized against. CHAT only when none of them
    resolves.
    """
    call_type = _as_mapping(request_data.get("standard_logging_object")).get("call_type")
    if isinstance(call_type, str):
        operation = _CALL_TYPE_OPERATIONS.get(call_type)
        if operation:
            return operation
        # A known call type that maps to no non-chat operation IS chat; do not
        # fall through to the route, which would re-decide a settled answer.
        if call_type:
            return "CHAT"
    for candidate in (
        metadata.get("endpoint"),
        getattr(user_api_key_dict, "request_route", None),
    ):
        operation = _operation_type_from_route(candidate)
        if operation:
            return operation
    return "CHAT"


def _key_metadata(metadata, user_api_key_dict):
    """Custom metadata attached to the calling virtual key."""
    key_metadata = _as_mapping(metadata.get("user_api_key_metadata"))
    if not key_metadata:
        key_metadata = _as_mapping(getattr(user_api_key_dict, "metadata", None))
    return key_metadata


def _key_owner_id(key_metadata):
    return str(key_metadata.get("revenium_user_id", "") or "")


def _key_owner_email(metadata, user_api_key_dict, key_metadata):
    return (
        metadata.get("user_api_key_user_email", "")
        or getattr(user_api_key_dict, "user_email", "")
        or key_metadata.get("email", "")
        or ""
    )


def _same_email(first, second):
    return (
        isinstance(first, str)
        and isinstance(second, str)
        and normalize_email(first) == normalize_email(second)
    )


def _build_subscriber(headers, metadata, user_api_key_dict, key_metadata):
    """Build the metering subscriber block.

    Fallback chain, most explicit first: ``x-revenium-*`` headers, then LiteLLM
    request metadata, then the ``UserAPIKeyAuth`` fields, then the virtual key's
    own ``revenium_*`` custom metadata.

    ``x-revenium-subscriber-id`` and ``x-revenium-subscriber-email`` are taken
    as the caller states them, with no check against the key: a virtual key
    shared by several people trusts each of them to name themselves, and
    budget enforcement keys per-person and department caps on that email.
    Without either, the call is attributed to the key's owner.

    The key owner's ``revenium_user_id`` is used only while the subscriber is
    still the key owner: no per-call email, or one that normalizes to the
    owner's own. A per-call email naming someone else gets no id, because
    grouped enforcement matches ``subscriber.id`` before the email, so the
    owner's id would check the owner's balance instead of the caller's.
    """
    subscriber = {}
    per_call_id = header_then_metadata(headers, metadata, SUBSCRIBER_ID_HEADER)
    per_call_email = header_then_metadata(headers, metadata, SUBSCRIBER_EMAIL_HEADER)
    owner_email = _key_owner_email(metadata, user_api_key_dict, key_metadata)
    names_someone_else = per_call_email and not _same_email(per_call_email, owner_email)
    subscriber_id = per_call_id or (
        "" if names_someone_else else _key_owner_id(key_metadata)
    )
    subscriber_email = per_call_email or owner_email
    credential_name = (
        metadata.get("user_api_key_alias", "")
        or getattr(user_api_key_dict, "key_alias", "")
        or key_metadata.get("revenium_key_name", "")
        or ""
    )
    if subscriber_id:
        subscriber["id"] = subscriber_id
    if subscriber_email:
        subscriber["email"] = subscriber_email
    if credential_name:
        subscriber["credential"] = {
            "name": credential_name,
            "value": credential_name,
        }
    return subscriber


_REQUEST_BODY_VALUE_TYPES = (Mapping, str, bytes, list, int, float)


def _key_team_alias(user_api_key_dict, data):
    """The calling key's team alias, read only where LiteLLM itself wrote it.

    The ``UserAPIKeyAuth`` object decides whenever a hook has one. Without it
    (the logging path) the alias comes from the logging kwargs'
    ``litellm_params`` first, because LiteLLM spreads request params into the
    kwargs' top level, where ``metadata`` can be the caller's body. The logging
    path takes its ``user_api_key_auth`` from that ``metadata``, so a value a
    JSON body can carry is never treated as the auth object.
    """
    if user_api_key_dict is not None and not isinstance(user_api_key_dict, _REQUEST_BODY_VALUE_TYPES):
        return present_or_none(getattr(user_api_key_dict, "team_alias", None))
    litellm_params = _as_mapping(data).get("litellm_params")
    if hasattr(litellm_params, "get"):
        return key_team_alias(litellm_params)
    return key_team_alias(data)


# LiteLLM stamps this on a metadata container when its ``missing_session_id``
# policy minted the session id itself rather than taking one from the caller.
_SESSION_ID_GENERATED_KEY = "litellm_session_id_generated"


def _litellm_session_id(container):
    """The LiteLLM session id the caller supplied, or None.

    ``container`` is a hook's request surface: ``data`` / ``request_data`` on the
    guardrail hooks, ``litellm_params`` on the logging path, never the logging
    kwargs' top level, where LiteLLM spreads provider params and ``metadata``
    can be the caller's body. LiteLLM fills ``litellm_session_id`` only from
    caller input (the ``x-litellm-session-id`` / ``x-litellm-trace-id`` headers
    or the request body), and mirrors it into ``session_id`` in the route's
    metadata container. ``litellm_metadata`` is read before ``metadata`` because
    on the routes that use it, ``metadata`` is the caller's own body.

    LiteLLM's per-request trace id is deliberately not read: it mints a fresh
    uuid4 for every request, so using it would make every call its own trace.
    For the same reason, a session id LiteLLM minted itself (its opt-in
    ``missing_session_id: generate`` policy) is ignored when LiteLLM marked it.

    Only a non-blank string counts; anything else falls through to the next
    source, because this runs in-band and a non-string must never reach the wire.
    """
    container = _as_mapping(container)
    metadata_containers = [
        _as_mapping(container.get(key)) for key in ("litellm_metadata", "metadata")
    ]
    if any(level.get(_SESSION_ID_GENERATED_KEY) for level in metadata_containers):
        return None
    candidates = (
        container.get("litellm_session_id"),
        *(level.get("session_id") for level in metadata_containers),
    )
    for candidate in candidates:
        if isinstance(candidate, str) and present_or_none(candidate) is not None:
            return candidate
    return None


def _litellm_trace_name(container):
    """The trace name the caller gave LiteLLM, or None.

    LiteLLM takes a trace name as ``metadata.trace_name``, one of its trace
    control fields alongside ``session_id``, and on the routes that use
    ``litellm_metadata`` it promotes the caller's value there. ``container`` is
    the same per-hook request surface ``_litellm_session_id`` reads, and
    ``litellm_metadata`` is read first for the same reason.

    Only a non-blank string counts, because this runs in-band and a non-string
    must never reach the wire.
    """
    container = _as_mapping(container)
    for key in ("litellm_metadata", "metadata"):
        candidate = _as_mapping(container.get(key)).get("trace_name")
        if isinstance(candidate, str) and present_or_none(candidate) is not None:
            return candidate
    return None


def _organization_name(data, headers, metadata, user_api_key_dict, key_metadata):
    """Organization attribution, headers first then key/team identity.

    When the operator pins the organization to the key's team
    (``REVENIUM_LITELLM_PIN_ORGANIZATION_TO_KEY_TEAM``) a key with a team alias
    decides it before any header is read.

    The header half is ``middleware._extract_organization_name`` -- the same
    implementation the callback uses, including its deprecation warning for
    ``x-revenium-organization-id`` -- rather than a second copy of it.
    """
    pinned = pinned_key_team(_key_team_alias(user_api_key_dict, data))
    if pinned is not None:
        return pinned
    organization_name = _extract_organization_name(headers, metadata)
    if organization_name is None:
        organization_name = present_or_none(getattr(user_api_key_dict, "team_alias", None))
    if organization_name is None:
        organization_name = present_or_none(key_metadata.get("revenium_organization_name"))
    return organization_name


def _job_fields(headers, key_metadata):
    """Declared agentic-job fields for the metering ``extra_body``.

    Headers win outright -- a caller that knows its job is never second-guessed
    -- and virtual-key metadata is the fallback for a proxy that pins a job on
    the key. The job id is required: a name or type without one identifies no
    job, so partial declarations yield nothing.

    No inference happens here; see this module's docstring on the classification
    layer left out of the SDK.
    """
    fields = dict(_extract_agentic_job_from_headers(headers))
    for source_key, wire_name in _KEY_METADATA_JOB_FIELDS.items():
        value = key_metadata.get(source_key)
        if value and wire_name not in fields:
            fields[wire_name] = value
    if not fields.get("agenticJobId"):
        return {}
    job_type = fields.get("agenticJobType")
    if isinstance(job_type, str):
        # Revenium lower-cases job type on ingest; do it here too so the value
        # sent matches what analytics group on.
        fields["agenticJobType"] = job_type.strip().lower()
    return fields


def _enforcement_metadata(data, headers, metadata, user_api_key_dict):
    """Describe the request the way ``check_enforcement`` reads a caller.

    ``check_enforcement`` resolves the caller through
    ``extract_subscriber_from_metadata`` (the nested ``subscriber`` block) and
    ``subscriber_credential``; the remaining keys are the attribution the rest
    of the SDK uses for the same request and are carried so a future rule
    dimension finds them already populated.
    """
    key_metadata = _key_metadata(metadata, user_api_key_dict)
    subscriber = _build_subscriber(headers, metadata, user_api_key_dict, key_metadata)
    credential = (subscriber.get("credential") or {}).get("value") or ""
    usage_metadata = {
        "model": data.get("model"),
        "organization_name": _organization_name(
            data, headers, metadata, user_api_key_dict, key_metadata
        ),
        "product_name": _extract_product_name(headers),
        "task_type": headers.get("x-revenium-task-type"),
        "agent": headers.get("x-revenium-agent"),
        "subscriber_credential": credential,
    }
    credential_alias = trace_fields_from_headers(headers).get("credential_alias")
    if credential_alias:
        usage_metadata["credential_alias"] = credential_alias
    if subscriber:
        usage_metadata["subscriber"] = subscriber
    return usage_metadata


def _budget_exception(error, model):
    """Render a ``BudgetExceededError`` as the 429 LiteLLM returns to the caller.

    LiteLLM serializes ``HTTPException.detail`` straight into the response body,
    so the envelope here *is* the client-visible error contract.
    """
    return HTTPException(
        status_code=429,
        detail={
            "error": {
                "message": error.message or "Budget exceeded",
                "type": "budget_exceeded",
                "guardrail": "revenium",
                "model": model,
                "budgets": [
                    {
                        "name": error.rule_name or "",
                        "ruleId": error.rule_id,
                        "threshold": error.threshold,
                        "currentValue": error.current_value,
                        "resetsAt": error.resets_at,
                    }
                ],
            }
        },
    )


def _is_own_cap_refusal(error):
    """Whether ``error`` is the 429 this guardrail raised for a spending cap.

    The exception chain is the first signal. The 429's body is the second, for
    a LiteLLM release that hands the failure hook a new exception without it.
    """
    if is_spending_cap_refusal(error):
        return True
    detail = error.detail if isinstance(error, HTTPException) else None
    body = detail.get("error") if isinstance(detail, dict) else None
    return (
        isinstance(body, dict)
        and body.get("type") == "budget_exceeded"
        and body.get("guardrail") == "revenium"
    )


# Returned by _event_hooks when the configured mode cannot be resolved to a
# fixed set of hook names -- a Mode object or dict for per-tag routing, where
# which hooks fire depends on the request. Distinct from None, which means the
# caller set no event_hook at all and every hook fires on every request. Folding
# the two together is what let a per-tag guardrail claim ownership of metering
# it does not perform on every request.
_UNKNOWN_HOOKS = object()


def _event_hooks(event_hook):
    """Normalize LiteLLM's ``event_hook`` into a set of hook names.

    Returns:
        ``None`` when no event hook is configured, which in LiteLLM means every
        hook fires on every request; a set of hook names for a string, a
        ``GuardrailEventHooks`` member, or a list of those; and
        ``_UNKNOWN_HOOKS`` for a per-tag ``Mode`` (or dict), where the hooks
        that fire depend on the request's tags and cannot be known here.
    """
    if event_hook is None:
        return None
    values = event_hook if isinstance(event_hook, (list, tuple, set)) else [event_hook]
    names = set()
    for value in values:
        value = getattr(value, "value", value)
        if not isinstance(value, str):
            return _UNKNOWN_HOOKS
        names.add(value)
    return names


class ReveniumGuardrail(CustomGuardrail):
    """Budget enforcement and usage metering for the LiteLLM proxy.

    This is the one LiteLLM proxy integration Revenium supports.
    ``revenium_middleware.litellm.proxy.middleware.MiddlewareHandler`` -- the
    ``CustomLogger`` callback -- is deprecated: it meters but never enforces,
    and a proxy that enables both meters every call twice. When this guardrail
    is configured to run on every request (``default_on: true`` with
    ``post_call`` among its modes) the callback suppresses its own metering so a
    proxy mid-migration is not double-billed; see ``_metering_owner``.

    Environment variables (all read by the SDK core, not by this class):
        ``REVENIUM_METERING_API_KEY``, ``REVENIUM_METERING_BASE_URL`` -- the
            metering client. Without an API key the SDK builds no client and
            nothing is metered; enforcement is unaffected.
        ``REVENIUM_CIRCUIT_BREAKER_ENABLED`` -- opt in to pre-call enforcement.
            Disabled by default, which makes the pre-call hook a no-op.
        ``REVENIUM_TEAM_ID``, ``REVENIUM_CB_POLL_INTERVAL_SECONDS``,
            ``REVENIUM_CB_FAIL_MODE`` -- see
            ``revenium_middleware._core.enforcement``.
        ``REVENIUM_LITELLM_SHARED_CALL_ID`` -- mint one identifier per proxied
            ``/v1/messages`` request and return it as the ``request-id``
            response header, so Claude Code's own usage record and this proxy's
            describe one call rather than two. On by default; ``false`` opts
            out: each ``/v1/messages`` record then carries a per-request id the
            proxy assigns, never Anthropic's ``msg_`` id. It covers the
            proxy's ``/v1/messages`` route only, so Claude Code has to be
            pointed at the proxy root (``ANTHROPIC_BASE_URL=<proxy>``). LiteLLM's
            Anthropic pass-through at ``<proxy>/anthropic/v1/messages`` mints
            nothing, warns about nothing, and leaves the call counted twice.
            See the README's "Counting a Claude Code call once" section.
        ``REVENIUM_LITELLM_PIN_ORGANIZATION_TO_KEY_TEAM`` -- off by default.
            When on, the calling virtual key's team alias decides the
            organization for the budget check and the record, and the
            ``x-revenium-organization-name`` / ``-id`` headers apply only to
            keys with no team alias.
    """

    def __init__(self, **kwargs):
        """Initialize the guardrail.

        Args:
            **kwargs: Forwarded to ``CustomGuardrail.__init__``. LiteLLM injects
                the ``config.yaml`` ``litellm_params`` here at proxy startup.
        """
        super().__init__(**kwargs)
        self._metered_calls = _MeteredCalls()
        log_organization_pin_setting()
        default_on = bool(getattr(self, "default_on", False))
        hooks = _event_hooks(getattr(self, "event_hook", None))
        if hooks is _UNKNOWN_HOOKS:
            # Per-tag routing: post_call may fire for some requests and not
            # others, so this guardrail cannot be the sole metering path.
            runs_post_call_on_every_request = False
        elif hooks is None:
            runs_post_call_on_every_request = True
        else:
            runs_post_call_on_every_request = _POST_CALL_EVENT in hooks
        self._meters_every_request = default_on and runs_post_call_on_every_request
        # The shared call id is live only where every part of the mechanism can
        # run. Three conditions, each closing a hole a green test run hides:
        #
        # * the flag, which is on unless set to false;
        # * a LiteLLM that can return plugin-supplied response headers, because
        #   on one that cannot, no header reaches Claude Code and changing the
        #   metered id would give the customer two records under a new id
        #   instead of two under the old one;
        # * a mode that runs this guardrail's pre-call hook on every request.
        #   LiteLLM gates the pre-call hook on mode (proxy/utils.py:1328) but
        #   runs the response-headers hook for any callback whose leaf class
        #   declares it (:3007 and :2149-2151), so without this the headers
        #   hook and the metered row would read back whatever a caller planted
        #   in their own request body.
        #
        # _UNKNOWN_HOOKS is a bare object(), so "pre_call in hooks" raises
        # TypeError on it and would take proxy startup down from inside
        # __init__. It is checked first, and it resolves to inactive for the
        # same reason it resolves metering ownership to false: under a per-tag
        # Mode the pre-call hook fires for some requests and not others.
        pre_call_runs_every_request = hooks is not _UNKNOWN_HOOKS and (
            hooks is None or _PRE_CALL_EVENT in hooks
        )
        shared_call_id_flag = is_shared_call_id_enabled()
        self._shared_id_active = bool(
            shared_call_id_flag
            and _headers_hook_available()
            and pre_call_runs_every_request
        )
        # A meter key is only proof when the pre-call hook overwrote it on this
        # very request; where that hook may not run, a found key is whatever
        # the caller planted, and claiming nothing records the call instead.
        self._meter_key_trusted = bool(
            pre_call_runs_every_request and self._meters_every_request
        )
        if shared_call_id_flag:
            try:
                litellm_version = importlib.metadata.version("litellm")
            except Exception:  # pragma: no cover - packaging metadata is absent
                litellm_version = "unknown"
            logger.info(
                "ReveniumGuardrail: REVENIUM_LITELLM_SHARED_CALL_ID is on "
                "(active=%s, response_headers_hook=%s, litellm=%s, mode=%r)",
                self._shared_id_active,
                _headers_hook_available(),
                litellm_version,
                getattr(self, "event_hook", None),
            )
            if not self._shared_id_active:
                logger.warning(
                    "ReveniumGuardrail: REVENIUM_LITELLM_SHARED_CALL_ID is on "
                    "but no shared call id can be minted here "
                    "(response_headers_hook=%s, mode=%r). Claude Code's own "
                    "usage record cannot be matched to this proxy's, so calls "
                    "seen by both are counted twice. Fix: LiteLLM 1.93.0 or "
                    "newer, and mode: [\"pre_call\", \"post_call\"].",
                    _headers_hook_available(),
                    getattr(self, "event_hook", None),
                )
        else:
            logger.info(
                "ReveniumGuardrail: REVENIUM_LITELLM_SHARED_CALL_ID is off "
                "(value=%.40r); calls Claude Code also reports are stored "
                "twice. Unset it or set it to true to turn it on.",
                sanitize_for_logging((shared_call_id_setting() or "").strip()),
            )
        if default_on and hooks is _UNKNOWN_HOOKS:
            logger.info(
                "ReveniumGuardrail: mode=%r selects hooks per request, so this "
                "guardrail does not claim metering ownership -- a deprecated "
                "MiddlewareHandler callback, if configured, keeps metering and "
                "calls this guardrail also meters would be counted twice. "
                "Remove the callback, or configure mode as a plain list.",
                getattr(self, "event_hook", None),
            )
        self._warn_when_failed_calls_go_unrecorded(hooks)
        if self._meters_every_request:
            # Claiming ownership only under this configuration is deliberate:
            # a guardrail applied per request could otherwise silence the
            # callback for requests it never runs on, losing metering entirely.
            register_metering_guardrail(self)
        if get_client() is None:
            # The SDK builds no metering client without REVENIUM_METERING_API_KEY.
            # Said once and loudly at startup, because the alternative is silence
            # on every request and no hint as to why.
            logger.error(
                "ReveniumGuardrail: REVENIUM_METERING_API_KEY is unset, so the "
                "SDK built no metering client -- nothing will be metered. "
                "Budget enforcement still works; only metering is disabled."
            )
        logger.info(
            "ReveniumGuardrail initialized (guardrail_name=%s, event_hook=%s, "
            "default_on=%s, owns_metering=%s)",
            getattr(self, "guardrail_name", None),
            getattr(self, "event_hook", None),
            getattr(self, "default_on", False),
            self._meters_every_request,
        )

    def _warn_when_failed_calls_go_unrecorded(self, hooks):
        """Say at startup when no request will carry the pre-call stamp the failure hook needs."""
        scan_raw_request = bool(getattr(self, "scan_raw_request", False))
        post_call_without_pre_call = (
            hooks is not None
            and hooks is not _UNKNOWN_HOOKS
            and _POST_CALL_EVENT in hooks
            and _PRE_CALL_EVENT not in hooks
        )
        if post_call_without_pre_call or scan_raw_request:
            logger.warning(
                "ReveniumGuardrail: failed calls are not recorded unless the "
                "Revenium guardrail's pre_call hook runs on the request itself "
                "(mode=%r, scan_raw_request=%s). Add \"pre_call\" to mode and leave "
                "scan_raw_request off to record them.",
                getattr(self, "event_hook", None),
                scan_raw_request,
            )

    async def async_pre_call_hook(self, user_api_key_dict, cache, data, call_type):
        """Enforce the caller's budget before the proxied call.

        Outcomes:

        1. Allowed -- no tripped rule applies (and the no-op case where the
           circuit breaker is disabled).
        2. Blocked -- a tripped rule applies: raises ``HTTPException`` 429.
        3. Failed open -- anything else went wrong in the check: logged,
           request allowed. Enforcement never takes the proxy down with it.

        Calls the synchronous ``check_enforcement`` directly on the event loop,
        with no ``to_thread``, because that check does no network I/O and no
        sleeping (BACK-3917); a thread hop would add latency and an executor
        dependency to every request for nothing.

        Args:
            user_api_key_dict: LiteLLM ``UserAPIKeyAuth`` for the calling key.
            cache: LiteLLM's ``DualCache`` (unused; enforcement keeps its own).
            data: The request body dict.
            call_type: The LiteLLM call type, e.g. ``"completion"``.

        Returns:
            ``data`` unchanged when the request is allowed.

        Raises:
            HTTPException: 429 when a tripped enforcement rule blocks the call.
        """
        logger.debug(
            "ReveniumGuardrail.async_pre_call_hook fired (call_type=%s)", call_type
        )
        # Before the mint and before enforcement, in a try of its own, for the
        # reasons given on the mint below. Always: the failure hook reads it as
        # proof the request authenticated; the duplicate guard only where trusted.
        try:
            _stamp_meter_key(data)
        except Exception as error:
            logger.warning(
                "ReveniumGuardrail: could not stamp the duplicate-guard key "
                "(%s: %s) -- the call proceeds; it is metered unguarded on "
                "success and a failure is not recorded",
                type(error).__name__,
                error,
            )
        # Minted before enforcement, so a call blocked with a 429 still carries
        # an id and the ordering never makes the mint conditional on the budget
        # decision. In a try of its own, separate from the enforcement block
        # below, so neither can break the other: LiteLLM's pre-call loop catches
        # only SensitiveDataRouteException (proxy/utils.py:1864), so anything
        # else raised here becomes the customer's error response, and a str
        # return is an HTTP 400 rejection (:1164-1179).
        if self._shared_id_active and call_type == ANTHROPIC_MESSAGES_CALL_TYPE:
            try:
                call_id = str(uuid.uuid4())
                # Both keys, because LiteLLM picks one per route and the live
                # 2026-09-15 proxy run confirmed both reach the response-headers
                # hook and both post-call paths with identical values.
                record_shared_call_id(
                    data.setdefault("litellm_metadata", {}), call_id, call_type
                )
                record_shared_call_id(
                    data.setdefault("metadata", {}), call_id, call_type
                )
            except Exception as error:
                logger.warning(
                    "ReveniumGuardrail: could not mint a shared call id "
                    "(%s: %s) -- the call proceeds and is metered under the "
                    "provider's response id",
                    type(error).__name__,
                    error,
                )
        metadata = _as_mapping(data.get("metadata"))
        # NOT metadata["headers"]: on /v1/messages LiteLLM fills
        # litellm_metadata instead. See extract_request_headers.
        headers = _as_mapping(extract_request_headers(data))
        try:
            usage_metadata = _enforcement_metadata(
                data, headers, metadata, user_api_key_dict
            )
            # The request's model is the operator's alias for a deployment;
            # the model metered is known only after routing.
            check_enforcement(usage_metadata, model=None, provider=LITELLM_PROVIDER)
        except BudgetExceededError as error:
            logger.warning(
                "ReveniumGuardrail: rule '%s' breached (%s/%s) -- blocking",
                error.rule_name,
                error.current_value,
                error.threshold,
            )
            raise _budget_exception(error, data.get("model")) from error
        except Exception as error:
            logger.warning(
                "ReveniumGuardrail: budget check failed (%s: %s) -- failing open",
                type(error).__name__,
                error,
            )
            return data
        logger.debug("ReveniumGuardrail: no tripped rule applies -- allowing")
        return data

    async def async_post_call_response_headers_hook(
        self,
        data,
        user_api_key_dict,
        response=None,
        request_headers=None,
        litellm_call_info=None,
        **kwargs,
    ):
        """Hand the caller the id this process minted for their request.

        Claude Code copies the ``request-id`` response header onto the usage
        record it reports to Revenium itself, so returning the same value we
        meter under is the whole mechanism: the two records collide on the
        transaction id and Revenium's team-scoped duplicate gate keeps one.
        Our bare ``request-id`` wins over the upstream provider's, which
        LiteLLM demotes to ``llm_provider-request-id`` on a streamed response
        and drops entirely on a non-streamed one.

        Declared on this class and not on a mixin: LiteLLM decides whether a
        callback customizes response headers by reading the leaf class's own
        ``__dict__`` (``proxy/utils.py:2149-2151``), so an inherited override
        never fires. That check also means declaring this hook costs every
        operator one call per request whether or not the flag is on, which is
        why the body returns immediately when it is off.

        ``litellm_call_info`` is declared by name rather than swallowed by
        ``**kwargs``: the dispatcher picks the call shape by reading
        ``inspect.signature`` and caches the answer per class
        (``proxy/utils.py:378-383``). ``**kwargs`` then absorbs whatever a later
        release adds, so a vendor upgrade cannot turn this into a ``TypeError``
        raised at a customer.

        Returns:
            ``{"request-id": id, "x-revenium-transaction-id": id}``, or ``None``
            when the shared call id is not live here, when there is no response,
            or when no id this process minted is on the request. ``None`` leaves
            LiteLLM's own headers untouched, which is what keeps the provider's
            ``request-id`` intact on the pass-through route.
        """
        try:
            if not self._shared_id_active or response is None:
                return None
            call_id = read_shared_call_id(data, self._shared_id_active)
            if not call_id:
                return None
            return {
                "request-id": call_id,
                "x-revenium-transaction-id": call_id,
            }
        except Exception as error:
            logger.warning(
                "ReveniumGuardrail: could not build the shared call id response "
                "headers (%s: %s) -- the response is returned unchanged",
                type(error).__name__,
                error,
            )
            return None

    def meters_request(self, request_kwargs):
        """Whether this guardrail meters the LiteLLM call ``request_kwargs`` makes.

        The client wrapper asks this on every patched LiteLLM call and stands
        aside when it is true, so a wrong True loses a row while a wrong False
        only sends a second one. Every case this cannot settle answers False.

        It is true only when all three hold:

        * The proxy is serving the call. LiteLLM attaches
          ``proxy_server_request`` to every request it receives and strips one a
          caller sends, so a direct call from the same process, such as a
          background job or another callback, never carries it.
        * A streamed call is of a type whose end reaches a metering hook here
          (``_STREAMED_CALL_TYPES_METERED_HERE``).
        * LiteLLM runs this guardrail's post-call hooks for the request, which
          it does not for a key or team that opted out of it.

        Args:
            request_kwargs: The keyword arguments the patched LiteLLM entry
                point was called with, which on the proxy is the request body
                the router forwarded.
        """
        try:
            if not isinstance(request_kwargs.get("proxy_server_request"), Mapping):
                return False
            if (
                request_kwargs.get("stream")
                and _call_type(request_kwargs) not in _STREAMED_CALL_TYPES_METERED_HERE
            ):
                return False
            return (
                self.should_run_guardrail(
                    data=request_kwargs, event_type=GuardrailEventHooks.post_call
                )
                is True
            )
        except Exception as error:
            # Broad on purpose: this runs inside the caller's LiteLLM call, where
            # anything raised would fail it, and the vendor method it consults
            # changes between releases. Answering False keeps the call metered.
            logger.debug(
                "ReveniumGuardrail: could not tell whether this call is metered "
                "here (%s: %s) -- leaving it to the client wrapper",
                type(error).__name__,
                error,
            )
            return False

    async def async_post_call_success_hook(self, data, user_api_key_dict, response):
        """Meter a successful proxied call.

        For a streamed request LiteLLM calls this with the *assembled* response
        once the stream completes; see this module's docstring.

        Args:
            data: The request body dict, with metadata at ``data["metadata"]``
                (a guardrail hook's shape, not the callback's
                ``litellm_params.metadata``).
            user_api_key_dict: LiteLLM ``UserAPIKeyAuth`` for the calling key.
            response: The provider response, of whatever shape the route
                returned.
        """
        try:
            logger.debug("ReveniumGuardrail.async_post_call_success_hook fired")
            metadata = _as_mapping(data.get("metadata"))
            # NOT metadata["headers"]: on /v1/messages LiteLLM fills
            # litellm_metadata instead. See extract_request_headers.
            headers = _as_mapping(extract_request_headers(data))

            hidden_params = _as_mapping(metadata.get("hidden_params"))
            if not hidden_params:
                hidden_params = _as_mapping(getattr(response, "_hidden_params", None))
            duration_ms = hidden_params.get("_response_ms") or 0
            mediation_latency = hidden_params.get("litellm_overhead_time_ms") or 0

            # Identity first, claim second. The claim is keyed on the meter key
            # the pre-call hook stamped, which both success paths read for one
            # call and no caller can choose; see _MeteredCalls.
            shared_call_id = read_shared_call_id(data, self._shared_id_active)
            if (
                self._shared_id_active
                and not shared_call_id
                and _was_eligible_for_a_minted_id(data, metadata, user_api_key_dict)
            ):
                _note_missing_shared_call_id()

            meter_key = self._trusted_meter_key(data)
            if not self._metered_calls.claim(meter_key):
                logger.debug(
                    "ReveniumGuardrail: this call was already metered from the "
                    "logging event -- not metering it a second time"
                )
                return

            transaction_id, operation_type = _meter_identity(
                response, shared_call_id, _request_transaction_id(meter_key)
            )

            end_time = datetime.datetime.now(datetime.timezone.utc)
            created_ts = getattr(response, "created", None)
            if created_ts:
                start_time = datetime.datetime.fromtimestamp(
                    created_ts, tz=datetime.timezone.utc
                )
            else:
                start_time = end_time - datetime.timedelta(milliseconds=duration_ms)

            completion_args = self._completion_args(
                data=data,
                headers=headers,
                metadata=metadata,
                hidden_params=hidden_params,
                user_api_key_dict=user_api_key_dict,
                usage=_resolve_usage(response),
                transaction_id=transaction_id,
                operation_type=operation_type,
                request_time=start_time,
                response_time=end_time,
                request_duration=duration_ms,
                mediation_latency=mediation_latency,
                stop_reason="END",
                litellm_session_id=_litellm_session_id(data),
                litellm_trace_name=_litellm_trace_name(data),
                # LiteLLM-populated surfaces only: data["metadata"] (and so
                # hidden_params) is the caller's own request body on
                # /v1/messages, and a caller must not be able to name the
                # upstream that decides the billed input count.
                upstream_sources=(
                    getattr(response, "_hidden_params", None),
                    data.get("litellm_params"),
                    getattr(data.get("litellm_logging_obj"), "model_call_details", None),
                ),
            )
            self._submit(completion_args, "success")
        except Exception as error:
            # This hook runs IN-BAND: LiteLLM turns anything raised here into
            # the client's response, so an unexpected payload shape would
            # convert a successful LLM call into an error. Metering is an
            # observability concern and must never be able to do that.
            logger.error(
                "ReveniumGuardrail: post-call metering failed (%s: %s) -- "
                "response returned to the caller unchanged",
                type(error).__name__,
                error,
            )

    def _trusted_meter_key(self, container):
        """The duplicate-guard key for this request, or ``None`` to record it unguarded."""
        return _meter_key(container) if self._meter_key_trusted else None

    def _meters_logged_request(self, litellm_params):
        """Whether this guardrail meters the request a logging event reports.

        The same ownership ``meters_request`` answers for the client wrapper,
        read from the logging path's ``litellm_params``, which carries the
        request's ``metadata`` and ``litellm_metadata`` where LiteLLM's guardrail
        check looks for a key's opt-out.
        """
        return self._meters_every_request and (
            self.should_run_guardrail(data=litellm_params, event_type=GuardrailEventHooks.post_call)
            is True
        )

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        """Meter a streamed Anthropic-route or Responses call, which no other hook sees.

        ``/v1/messages`` with ``stream: true`` is the one route where LiteLLM
        never calls ``async_post_call_success_hook``: the vendor arms
        ``_on_deferred_native_stream_complete`` for a native
        ``anthropic_messages`` iterator, which enqueues a logging coroutine and
        never reaches ``_run_deferred_stream_guardrails``. See this module's
        docstring. That route is what Claude Code speaks, so without this method
        a customer's whole Claude Code spend goes unmetered. A streamed
        ``/v1/responses`` call (Codex CLI) ends the same way.

        A Responses stream is metered here only when this guardrail meters the
        request (``meters_request``): otherwise the client wrapper meters one
        LiteLLM bridges through ``litellm.acompletion``, and a row here would be
        its second.

        LiteLLM calls this for **every** completed request, so the route gate is
        load-bearing: without it, every call already metered by the post-call
        hook would get a second row. The duplicate guard behind the gate is the
        belt to that pair of braces, for a provider whose Anthropic stream
        arrives in a shape that arms the guardrail closure after all.

        Nothing here can fail the proxied call -- this runs after the client
        already has the response -- but LiteLLM logs and counts a raising
        callback, so the body is guarded anyway.

        Args:
            kwargs: LiteLLM's ``model_call_details``: the call type, the stream
                flag and ``litellm_call_id`` at the top level, with metadata
                under ``litellm_params`` rather than at the top level as a
                guardrail hook receives it.
            response_obj: The response LiteLLM assembled from the finished
                stream, carrying the full usage totals.
            start_time: When the call started.
            end_time: When the stream completed.
        """
        try:
            call_type = kwargs.get("call_type")
            if (
                call_type not in _STREAMED_CALL_TYPES_METERED_FROM_THE_LOGGING_EVENT
                or not kwargs.get("stream")
            ):
                return
            litellm_params = _as_mapping(kwargs.get("litellm_params"))
            if call_type == _RESPONSES_CALL_TYPE and not self._meters_logged_request(litellm_params):
                return
            logger.debug(
                "ReveniumGuardrail.async_log_success_event fired for a streamed %s call",
                call_type,
            )
            metadata = _as_mapping(litellm_params.get("metadata"))
            # NOT metadata["headers"]: this is the one route on which LiteLLM
            # fills litellm_metadata and leaves metadata as the caller's own
            # request body, so reading metadata alone drops every documented
            # x-revenium-* header here and takes a forged one where the caller
            # planted it. See extract_request_headers.
            headers = _as_mapping(extract_request_headers(litellm_params))
            hidden_params = _as_mapping(metadata.get("hidden_params"))
            if not hidden_params:
                hidden_params = _as_mapping(getattr(response_obj, "_hidden_params", None))

            # Identity first, claim second, on the same meter key the post-call
            # hook claims, so either recognises the other's claim.
            shared_call_id = read_shared_call_id(litellm_params, self._shared_id_active)
            meter_key = self._trusted_meter_key(litellm_params)
            transaction_id = self._stream_transaction_id(
                shared_call_id, _request_transaction_id(meter_key)
            )

            if not self._metered_calls.claim(meter_key):
                logger.debug(
                    "ReveniumGuardrail: this call was already metered from the "
                    "post-call hook -- not metering it a second time"
                )
                return

            request_time = start_time or datetime.datetime.now(datetime.timezone.utc)
            response_time = end_time or request_time
            duration_ms = hidden_params.get("_response_ms") or _elapsed_ms(
                request_time, response_time
            )

            completion_args = self._completion_args(
                data=kwargs,
                headers=headers,
                metadata=metadata,
                # There is no ``UserAPIKeyAuth`` object on the logging path.
                # Every field this payload reads off one is also present in
                # request metadata as a flat ``user_api_key_*`` key, which is
                # where the payload builder looks first.
                user_api_key_dict=metadata.get("user_api_key_auth"),
                hidden_params=hidden_params,
                usage=_resolve_usage(response_obj),
                transaction_id=transaction_id,
                operation_type=_OPERATION_TYPES.get(
                    type(response_obj).__name__, "CHAT"
                ),
                request_time=request_time,
                response_time=response_time,
                request_duration=duration_ms,
                mediation_latency=hidden_params.get("litellm_overhead_time_ms") or 0,
                stop_reason="END",
                # litellm_params, not kwargs: LiteLLM spreads provider params
                # across the kwargs' top level, which the caller can shape.
                litellm_session_id=_litellm_session_id(litellm_params),
                litellm_trace_name=_litellm_trace_name(litellm_params),
                upstream_sources=(
                    getattr(response_obj, "_hidden_params", None),
                    kwargs,
                    litellm_params,
                ),
                # Known from the gate above, rather than re-derived from hidden
                # params that this path does not always carry.
                is_streamed=True,
            )
            self._submit(completion_args, "success")
        except Exception as error:
            logger.error(
                "ReveniumGuardrail: streamed Anthropic metering failed (%s: %s)",
                type(error).__name__,
                error,
            )

    def _stream_transaction_id(self, shared_call_id, request_transaction_id):
        """The transaction id for a streamed Anthropic row.

        This is ``async_post_call_success_hook``'s identity chain with its
        second link, the response's own ``.id``, deliberately dropped. On this
        path ``response_obj`` is a ``ModelResponse`` LiteLLM assembled from the
        Anthropic stream and its ``.id`` is the provider's ``msg_`` id, so
        metering on it would give the streamed row a different identity shape
        from the non-streamed row on the same route -- that one is handed a raw
        dict with no ``.id`` to read and falls through to the request's own id.
        Both rows on the route therefore carry the request's own id.

        The first link is the same as the post-call hook's and outranks
        everything: the id this proxy minted for the request and handed back as
        the ``request-id`` response header. Claude Code copies that header onto
        its own usage record, and this is the only route Claude Code speaks, so
        a streamed row filed under anything else can never collide with the
        telemetry row the customer is billed twice for.

        Args:
            shared_call_id: The minted id, or ``None`` when the flag is off or
                nothing this process minted is on the request.
            request_transaction_id: The request's own id from the stamped meter
                key, or ``None`` when no trusted key is on the request, which
                leaves a fresh UUID per row. Never LiteLLM's ``litellm_call_id``,
                which the caller chooses (see ``_meter_identity``).
        """
        return resolve_transaction_id(shared_call_id, request_transaction_id)

    async def async_post_call_failure_hook(
        self,
        request_data,
        original_exception,
        user_api_key_dict,
        traceback_str=None,
    ):
        """Meter a failed proxied call with ``stop_reason="ERROR"``.

        A request LiteLLM refused before the Revenium guardrail's pre-call ran
        on it (a wrong, missing, expired or blocked key, or any other refusal
        during authentication) sends no record.

        A call this guardrail's spending cap refused never reached a provider,
        so, as with the client integrations, it sends no record.

        Args:
            request_data: The request body dict.
            original_exception: The exception that failed the call.
            user_api_key_dict: LiteLLM ``UserAPIKeyAuth`` for the calling key.
            traceback_str: Optional traceback text from LiteLLM.
        """
        try:
            if not _reached_pre_call(request_data):
                logger.debug(
                    "ReveniumGuardrail: failed request never reached the pre-call "
                    "hook; no failed-call record"
                )
                return
            logger.debug("ReveniumGuardrail.async_post_call_failure_hook fired")
            if _is_own_cap_refusal(original_exception):
                logger.debug("ReveniumGuardrail: a spending cap refused the call; no failed-call record")
                return
            metadata = _as_mapping(request_data.get("metadata"))
            # NOT metadata["headers"]: on /v1/messages LiteLLM fills
            # litellm_metadata instead. See extract_request_headers.
            headers = _as_mapping(extract_request_headers(request_data))
            hidden_params = _as_mapping(metadata.get("hidden_params"))
            if not hidden_params:
                hidden_params = _as_mapping(
                    getattr(original_exception, "_hidden_params", None)
                )

            now = datetime.datetime.now(datetime.timezone.utc)
            completion_args = self._completion_args(
                data=request_data,
                headers=headers,
                metadata=metadata,
                hidden_params=hidden_params,
                user_api_key_dict=user_api_key_dict,
                usage=_resolve_usage(original_exception),
                # Never a constant: a shared sentinel collides in Revenium's
                # (organization, transactionId) dedup, so every failed call
                # after the first would be discarded as a duplicate and the
                # failure-rate signal would vanish.
                # The ":err:<8 hex>" suffix is minted per event, never per
                # guardrail instance or per process. Correlation still comes
                # from the prefix, but a router retry hands the failed attempt
                # and the paid attempt the same litellm_call_id, so without the
                # suffix the failure took the success's identity and the paid
                # row was dropped as a duplicate at zero cost. No success
                # identity can contain the marker: those are a provider response
                # id or a bare UUID. Correlation by prefix therefore holds only
                # where the success carries the provider's id.
                transaction_id="{0}:err:{1}".format(
                    resolve_transaction_id(
                        getattr(original_exception, "id", None),
                        _litellm_call_id(
                            request_data,
                            hidden_params,
                            getattr(original_exception, "_hidden_params", None),
                        ),
                    ),
                    uuid.uuid4().hex[:8],
                ),
                operation_type=_failure_operation_type(
                    request_data, metadata, user_api_key_dict
                ),
                request_time=now,
                response_time=now,
                request_duration=0,
                mediation_latency=hidden_params.get("litellm_overhead_time_ms") or 0,
                stop_reason="ERROR",
                litellm_session_id=_litellm_session_id(request_data),
                litellm_trace_name=_litellm_trace_name(request_data),
                upstream_sources=(
                    getattr(original_exception, "_hidden_params", None),
                    request_data.get("litellm_params"),
                    getattr(
                        request_data.get("litellm_logging_obj"),
                        "model_call_details",
                        None,
                    ),
                ),
            )
            self._submit(completion_args, "failure")
        except Exception as error:
            logger.error(
                "ReveniumGuardrail: failure-path metering failed (%s: %s)",
                type(error).__name__,
                error,
            )

    def _completion_args(
        self,
        *,
        data,
        headers,
        metadata,
        hidden_params,
        user_api_key_dict,
        usage,
        transaction_id,
        operation_type,
        request_time,
        response_time,
        request_duration,
        mediation_latency,
        stop_reason,
        upstream_sources,
        is_streamed=None,
        litellm_session_id=None,
        litellm_trace_name=None,
    ):
        """Build the completion metering payload shared by all three hooks.

        Args:
            hidden_params: The hidden params the calling hook already resolved
                -- request metadata first, then the response's own
                ``_hidden_params``. Re-deriving them here from ``metadata``
                alone lost the stream flag for every response that carried it
                only on itself, silently metering a streamed call as
                non-streamed.
            upstream_sources: The LiteLLM-populated surfaces, in lookup order,
                that say whether this upstream's ``prompt_tokens`` contains the
                cache tokens the row also reports in their own fields. Chosen
                per hook from what LiteLLM handed it, because no two hooks see
                the same surfaces, and a wrong answer in either direction costs
                money -- see ``billable_input_tokens``.
            is_streamed: Whether this row is a stream, when the caller already
                knows. Left unset, it is read out of the hidden params, which
                is the only source the post-call hooks have.
                ``async_log_success_event`` knows from the call it was gated
                on, and says so rather than depending on a nested key the
                logging path does not always carry.
            litellm_session_id: The session id the caller gave LiteLLM, as the
                calling hook resolved it from its own request surface with
                ``_litellm_session_id``. It is the trace id when no
                ``x-revenium-trace-id`` header was sent. Resolved by the hook
                because the logging path's ``data`` is not a LiteLLM-owned
                surface at its top level.
            litellm_trace_name: The trace name the caller gave LiteLLM,
                resolved by the hook with ``_litellm_trace_name``. It is the
                trace name when no ``x-revenium-trace-name`` header was sent,
                and outranks the host's ``REVENIUM_TRACE_NAME``. When none of
                the three is set, ``litellm_session_id`` names the trace, but
                only when it is also the trace id.

        Returns:
            A ``(completion_args, extra_body)`` tuple -- ``extra_body`` carries
            the ``agenticJob*`` fields, which the metering API has no typed
            parameters for.
        """
        key_metadata = _key_metadata(metadata, user_api_key_dict)
        cache_read_tokens, cache_creation_tokens = extract_cache_tokens(usage)
        input_tokens, completion_tokens, total_tokens = _token_counts(
            usage, cache_read_tokens, cache_creation_tokens, upstream_sources
        )
        if is_streamed is None:
            is_streamed = bool(
                _as_mapping(_as_mapping(hidden_params).get("optional_params")).get(
                    "stream", False
                )
            )
        subscriber = _build_subscriber(
            headers, metadata, user_api_key_dict, key_metadata
        )
        # Both callers reach here, and only one of them arrives in UTC: the
        # post-call hook builds aware values, while the logging event is handed
        # LiteLLM's naive local start_time and end_time. format_utc_timestamp
        # is what keeps the published "Z" honest for both.
        response_time_str = format_utc_timestamp(response_time)

        # Order: the x-revenium-trace-id header, the LiteLLM session id the
        # caller supplied, then Claude Code's session id. A session id groups
        # calls into one trace, and Claude Code stamps its own on its telemetry
        # rows as their trace id, so one session reads as one trace on both
        # records. LiteLLM's own per-request trace id is never a fallback: it
        # mints a fresh uuid4 for every request, which would make every call
        # its own trace. Never the transaction id: a session covers many calls,
        # and keying the duplicate gate on it would fold a whole session into
        # one record and under-bill it.
        trace_id = (headers.get("x-revenium-trace-id")
                    or litellm_session_id
                    or headers.get("x-claude-code-session-id"))
        # The session id names the trace only when it is the trace: under a
        # caller's own x-revenium-trace-id it would label that trace with an id
        # it does not carry.
        trace_fields = trace_fields_from_headers(
            headers,
            litellm_trace_name=litellm_trace_name,
            default_trace_name=(
                litellm_session_id
                if litellm_session_id and trace_id == litellm_session_id
                else None
            ),
        )

        completion_args = {
            "cache_creation_token_count": cache_creation_tokens,
            **extract_cache_creation_ttl_counts(usage),
            "cache_read_token_count": cache_read_tokens,
            "input_token_cost": None,
            "output_token_cost": None,
            "total_cost": None,
            "output_token_count": completion_tokens,
            "cost_type": "AI",
            "model": _upstream_model(data),
            "input_token_count": input_tokens,
            "provider": LITELLM_PROVIDER,
            "model_source": LITELLM_PROVIDER,
            "reasoning_token_count": 0,
            "request_time": format_utc_timestamp(request_time),
            "response_time": response_time_str,
            "completion_start_time": response_time_str,
            "request_duration": request_duration,
            "time_to_first_token": request_duration,
            "stop_reason": stop_reason,
            "total_token_count": total_tokens,
            "transaction_id": transaction_id,
            "trace_id": trace_id,
            "task_type": headers.get("x-revenium-task-type"),
            "subscriber": subscriber if subscriber else None,
            "organization_name": _organization_name(
                data, headers, metadata, user_api_key_dict, key_metadata
            ),
            "subscription_id": headers.get("x-revenium-subscription-id"),
            "product_name": _extract_product_name(headers),
            "agent": headers.get("x-revenium-agent"),
            # Present only when the caller sent the header: create_completion
            # drops NotGiven but keeps an explicit None, which would reach the
            # wire as "effort": null instead of being omitted.
            **(
                {"effort": headers["x-revenium-effort"]}
                if "x-revenium-effort" in headers
                else {}
            ),
            "response_quality_score": headers.get("x-revenium-response-quality-score"),
            "is_streamed": bool(is_streamed),
            "operation_type": operation_type,
            "mediation_latency": mediation_latency,
            **trace_fields,
            "middleware_source": "GUARDRAIL",
        }
        extra_body = merge_extra_body(None, _job_fields(headers, key_metadata))
        return completion_args, extra_body

    @staticmethod
    def _submit(built_args, label):
        """Dispatch the metering row off the request path.

        ``run_async_in_thread`` is how the callback submits today: the row goes
        out on the SDK's own metering thread so neither the proxy's event loop
        nor the caller's latency carries it, and a metering failure is logged
        rather than raised.
        """
        completion_args, extra_body = built_args
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("ReveniumGuardrail: metering (%s) with args: %s", label, sanitize_for_logging(completion_args))

        async def metering_call():
            try:
                result = submit_ai_event(
                    "completion", {**completion_args, "extra_body": extra_body}
                )
                logger.debug("ReveniumGuardrail: metering result (%s): %s", label, result)
            except Exception as error:
                logger.error(
                    "ReveniumGuardrail: metering call failed (%s): %s", label, error
                )

        run_async_in_thread(metering_call())
