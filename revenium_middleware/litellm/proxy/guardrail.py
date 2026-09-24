"""Revenium budget guardrail for the LiteLLM proxy.

``ReveniumGuardrail`` is a LiteLLM ``CustomGuardrail`` that enforces a budget
before the proxied call and meters usage after it:

* **pre_call** -- runs the SDK's own circuit breaker
  (``revenium_middleware._core.enforcement.check_enforcement``) against the
  request's subscriber/organization attribution and turns a
  ``BudgetExceededError`` into the HTTP 429 LiteLLM expects. Every other
  failure fails **open**: an unreachable or misbehaving enforcement path never
  blocks a call.
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
  surface the logging dispatch does reach. That method is gated to exactly this
  route, because LiteLLM calls it for every completed request.

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

from fastapi import HTTPException
from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.integrations.custom_logger import CustomLogger

from revenium_middleware._core import get_client, run_async_in_thread, submit_ai_event
from revenium_middleware._core.cache_tokens import (
    billable_input_tokens,
    extract_cache_tokens,
    get_usage_field,
    total_priced_tokens,
)
from revenium_middleware._core.config import is_shared_call_id_enabled
from revenium_middleware._core.enforcement import check_enforcement
from revenium_middleware._core.exceptions import BudgetExceededError
from revenium_middleware._core.fields import merge_extra_body

from ._metering_owner import register_metering_guardrail
from .middleware import (
    ANTHROPIC_MESSAGES_CALL_TYPE,
    prompt_tokens_fold_cache,
    _extract_agentic_job_from_headers,
    _extract_organization_name,
    _extract_product_name,
    extract_request_headers,
    format_utc_timestamp,
    read_shared_call_id,
    record_shared_call_id,
    resolve_transaction_id,
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


# The LiteLLM call type both Anthropic-shaped routes carry, and the only one
# ``async_log_success_event`` meters. A literal rather than
# ``litellm.types.utils.CallTypes.anthropic_messages``: importing the vendor
# enum here would let a rename break proxy startup, which is a worse failure
# than a missed row. ``test_proxy_guardrail_streaming`` asserts the two still
# agree, so a rename fails a test instead of quietly stopping the metering.
_ANTHROPIC_MESSAGES_CALL_TYPE = "anthropic_messages"

# How many recently metered calls the duplicate guard remembers. Each entry is
# one short id, so this costs a few hundred kilobytes and covers far more calls
# than a proxy has in flight at once. Forgetting an old entry can only ever
# allow a duplicate row, never suppress a real one.
_METERED_CALL_MEMORY = 4096

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


def _token_counts(usage, cache_read_tokens, cache_creation_tokens, prompt_folds_cache):
    """Return ``(input, completion, total)`` for a usage object of any shape.

    ``input`` is what Revenium prices at the model's input rate; the cache
    counts are handed in rather than re-read, because the payload builder has
    already extracted them for the fields they are reported in, and the two
    must be the same numbers. ``prompt_folds_cache`` is the caller's answer to
    whether this upstream's prompt count contains them -- see
    ``billable_input_tokens`` for what each wrong answer costs.

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
            prompt, cache_read_tokens, cache_creation_tokens, prompt_folds_cache
        )
    else:
        input_tokens = get_usage_field(usage, "input_tokens", 0) or 0
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


class _MeteredCalls:
    """The calls already metered, so no completed call is metered twice.

    LiteLLM hands the same completed call to more than one of our hooks:
    ``async_post_call_success_hook`` and ``async_log_success_event`` both fire
    on a streamed ``/v1/chat/completions`` call, and a provider whose
    ``anthropic_messages`` stream arrived as a ``CustomStreamWrapper`` would
    make both fire on the Anthropic route too. The route gate on the logging
    event covers the shapes seen today; this covers the ones that change
    underneath us.

    Keyed on LiteLLM's own call id rather than on the row's transaction id.
    The two hooks resolve that id from different sources and can land on
    different values for one call, which is exactly the case a transaction-id
    key would fail to catch.
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


def _litellm_call_id(*sources):
    """LiteLLM's own correlation id, from the first source that carries one."""
    for source in sources:
        call_id = _as_mapping(source).get("litellm_call_id")
        if call_id:
            return call_id
    return None


def _meter_identity(response, hidden_params, data=None, shared_call_id=None):
    """Return ``(transaction_id, operation_type)`` for a response of any shape.

    ``EmbeddingResponse`` and friends have no ``.id`` -- reading it unguarded is
    what turned a successful ``/v1/embeddings`` call into a client-facing error,
    because a ``CustomGuardrail`` post-call hook runs in-band. So the id is
    resolved through a chain and, when nothing real survives it, a fresh UUID
    rather than a constant -- see ``resolve_transaction_id`` for why a shared
    sentinel is worse than no id at all.

    Args:
        shared_call_id: The id this process minted for this request, when the
            shared call id is on. It outranks the provider's own response id
            because that id never reaches Claude Code, so a row keyed on it can
            never collide with the telemetry row the customer is billed twice
            for. ``None`` restores the chain exactly as it is with the flag off.
    """
    return (
        resolve_transaction_id(
            shared_call_id,
            getattr(response, "id", None),
            # Three places carry LiteLLM's correlation id, and the request
            # metadata is not where LiteLLM puts it -- check the response's own
            # hidden params and the request body too, or the fallback to a real
            # id never fires and every row gets an unrelated UUID.
            _litellm_call_id(
                hidden_params, getattr(response, "_hidden_params", None), data
            ),
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


def _build_subscriber(headers, metadata, user_api_key_dict, key_metadata):
    """Build the metering subscriber block.

    Fallback chain, most explicit first: ``x-revenium-*`` headers, then LiteLLM
    request metadata, then the ``UserAPIKeyAuth`` fields, then the virtual key's
    own ``revenium_*`` custom metadata.
    """
    subscriber = {}
    subscriber_id = (
        metadata.get("x-revenium-subscriber-id", "")
        or headers.get("x-revenium-subscriber-id")
        or str(key_metadata.get("revenium_user_id", "") or "")
        or ""
    )
    subscriber_email = (
        metadata.get("user_api_key_user_email", "")
        or getattr(user_api_key_dict, "user_email", "")
        or key_metadata.get("email", "")
        or ""
    )
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


def _organization_name(headers, metadata, user_api_key_dict, key_metadata):
    """Organization attribution, headers first then key/team identity.

    The header half is ``middleware._extract_organization_name`` -- the same
    implementation the callback uses, including its deprecation warning for
    ``x-revenium-organization-id`` -- rather than a second copy of it.
    """
    organization_name = _extract_organization_name(headers, metadata)
    if organization_name is None:
        organization_name = getattr(user_api_key_dict, "team_alias", None)
    if organization_name is None:
        organization_name = key_metadata.get("revenium_organization_name")
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
            headers, metadata, user_api_key_dict, key_metadata
        ),
        "product_name": _extract_product_name(headers),
        "task_type": headers.get("x-revenium-task-type"),
        "agent": headers.get("x-revenium-agent"),
        "subscriber_credential": credential,
    }
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
            out and keeps LiteLLM's own response id. It covers the
            proxy's ``/v1/messages`` route only, so Claude Code has to be
            pointed at the proxy root (``ANTHROPIC_BASE_URL=<proxy>``). LiteLLM's
            Anthropic pass-through at ``<proxy>/anthropic/v1/messages`` mints
            nothing, warns about nothing, and leaves the call counted twice.
            See the README's "Counting a Claude Code call once" section.
    """

    def __init__(self, **kwargs):
        """Initialize the guardrail.

        Args:
            **kwargs: Forwarded to ``CustomGuardrail.__init__``. LiteLLM injects
                the ``config.yaml`` ``litellm_params`` here at proxy startup.
        """
        super().__init__(**kwargs)
        self._metered_calls = _MeteredCalls()
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
        shared_call_id_flag = is_shared_call_id_enabled()
        self._shared_id_active = bool(
            shared_call_id_flag
            and _headers_hook_available()
            and hooks is not _UNKNOWN_HOOKS
            and (hooks is None or _PRE_CALL_EVENT in hooks)
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
        if default_on and hooks is _UNKNOWN_HOOKS:
            logger.info(
                "ReveniumGuardrail: mode=%r selects hooks per request, so this "
                "guardrail does not claim metering ownership -- a deprecated "
                "MiddlewareHandler callback, if configured, keeps metering and "
                "calls this guardrail also meters would be counted twice. "
                "Remove the callback, or configure mode as a plain list.",
                getattr(self, "event_hook", None),
            )
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

    async def async_pre_call_hook(self, user_api_key_dict, cache, data, call_type):
        """Enforce the caller's budget before the proxied call.

        Outcomes:

        1. Allowed -- no tripped rule applies (and the no-op case where the
           circuit breaker is disabled).
        2. Blocked -- a tripped rule applies: raises ``HTTPException`` 429.
        3. Failed open -- anything else went wrong in the check: logged,
           request allowed. Enforcement never takes the proxy down with it.

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
            check_enforcement(usage_metadata)
        except BudgetExceededError as error:
            logger.warning(
                "ReveniumGuardrail: rule '%s' breached (%s/%s) -- blocking",
                error.rule_name,
                error.current_value,
                error.threshold,
            )
            raise _budget_exception(error, data.get("model"))
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

            # Identity first, claim second. The minted id, when the flag is
            # on and one arrived, is the id both success paths resolve for the
            # same call, so it is the only key on which the duplicate guard can
            # recognise a call the logging event already metered. Without one
            # the key falls back to LiteLLM's own call id, which is what
            # BACK-3199 shipped: never the stored transaction id, because the
            # two hooks reach different values for it on the same call.
            shared_call_id = read_shared_call_id(data, self._shared_id_active)
            if (
                self._shared_id_active
                and not shared_call_id
                and _was_eligible_for_a_minted_id(data, metadata, user_api_key_dict)
            ):
                _note_missing_shared_call_id()

            if not self._metered_calls.claim(
                shared_call_id
                or _litellm_call_id(
                    hidden_params, getattr(response, "_hidden_params", None), data
                )
            ):
                logger.debug(
                    "ReveniumGuardrail: this call was already metered from the "
                    "logging event -- not metering it a second time"
                )
                return

            transaction_id, operation_type = _meter_identity(
                response, hidden_params, data, shared_call_id
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
                # LiteLLM-populated surfaces only: data["metadata"] (and so
                # hidden_params) is the caller's own request body on
                # /v1/messages, and a caller must not be able to name the
                # upstream that decides the billed input count.
                prompt_folds_cache=prompt_tokens_fold_cache(
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

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        """Meter a streamed Anthropic-route call, which no other hook sees.

        ``/v1/messages`` with ``stream: true`` is the one route where LiteLLM
        never calls ``async_post_call_success_hook``: the vendor arms
        ``_on_deferred_native_stream_complete`` for a native
        ``anthropic_messages`` iterator, which enqueues a logging coroutine and
        never reaches ``_run_deferred_stream_guardrails``. See this module's
        docstring. That route is what Claude Code speaks, so without this method
        a customer's whole Claude Code spend goes unmetered.

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
            if (
                kwargs.get("call_type") != _ANTHROPIC_MESSAGES_CALL_TYPE
                or not kwargs.get("stream")
            ):
                return
            logger.debug(
                "ReveniumGuardrail.async_log_success_event fired for a streamed "
                "Anthropic call"
            )
            litellm_params = _as_mapping(kwargs.get("litellm_params"))
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

            # Identity first, claim second, exactly as the post-call hook does.
            # The minted id is the value both success paths resolve for one
            # call, so it is the only key on which either can recognise the
            # other's claim; without one the key falls back to LiteLLM's own
            # call id, which is what this method shipped with.
            shared_call_id = read_shared_call_id(litellm_params, self._shared_id_active)
            transaction_id = self._stream_transaction_id(
                kwargs, litellm_params, metadata, hidden_params, shared_call_id
            )

            if not self._metered_calls.claim(
                shared_call_id
                or _litellm_call_id(kwargs, litellm_params, metadata, hidden_params)
            ):
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
                prompt_folds_cache=prompt_tokens_fold_cache(
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

    def _stream_transaction_id(
        self, kwargs, litellm_params, metadata, hidden_params, shared_call_id=None
    ):
        """The transaction id for a streamed Anthropic row.

        This is ``async_post_call_success_hook``'s identity chain with its
        second link, the response's own ``.id``, deliberately dropped. On this
        path ``response_obj`` is a ``ModelResponse`` LiteLLM assembled from the
        Anthropic stream and its ``.id`` is the provider's ``msg_`` id, so
        metering on it would give the streamed row a different identity shape
        from the non-streamed row on the same route -- that one is handed a raw
        dict with no ``.id`` to read and already falls through to LiteLLM's
        correlation id.

        The first link is the same as the post-call hook's and outranks
        everything: the id this proxy minted for the request and handed back as
        the ``request-id`` response header. Claude Code copies that header onto
        its own usage record, and this is the only route Claude Code speaks, so
        a streamed row filed under anything else can never collide with the
        telemetry row the customer is billed twice for.

        Args:
            shared_call_id: The minted id, or ``None`` when the flag is off or
                nothing this process minted is on the request. ``None`` restores
                the chain exactly as it was before the shared call id existed.
        """
        return resolve_transaction_id(
            shared_call_id,
            _litellm_call_id(kwargs, litellm_params, metadata, hidden_params),
        )

    async def async_post_call_failure_hook(
        self,
        request_data,
        original_exception,
        user_api_key_dict,
        traceback_str=None,
    ):
        """Meter a failed proxied call with ``stop_reason="ERROR"``.

        Args:
            request_data: The request body dict.
            original_exception: The exception that failed the call.
            user_api_key_dict: LiteLLM ``UserAPIKeyAuth`` for the calling key.
            traceback_str: Optional traceback text from LiteLLM.
        """
        try:
            logger.debug("ReveniumGuardrail.async_post_call_failure_hook fired")
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
                # id, a litellm_call_id or a bare UUID.
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
                prompt_folds_cache=prompt_tokens_fold_cache(
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
        prompt_folds_cache,
        is_streamed=None,
    ):
        """Build the completion metering payload shared by all three hooks.

        Args:
            hidden_params: The hidden params the calling hook already resolved
                -- request metadata first, then the response's own
                ``_hidden_params``. Re-deriving them here from ``metadata``
                alone lost the stream flag for every response that carried it
                only on itself, silently metering a streamed call as
                non-streamed.
            prompt_folds_cache: Whether this upstream's ``prompt_tokens``
                contains the cache tokens the row also reports in their own
                fields. Resolved per hook from what LiteLLM handed it, because
                no two hooks see the same surfaces, and wrong in either
                direction costs money -- see ``billable_input_tokens``.
            is_streamed: Whether this row is a stream, when the caller already
                knows. Left unset, it is read out of the hidden params, which
                is the only source the post-call hooks have.
                ``async_log_success_event`` knows from the call it was gated
                on, and says so rather than depending on a nested key the
                logging path does not always carry.

        Returns:
            A ``(completion_args, extra_body)`` tuple -- ``extra_body`` carries
            the ``agenticJob*`` fields, which the metering API has no typed
            parameters for.
        """
        key_metadata = _key_metadata(metadata, user_api_key_dict)
        cache_read_tokens, cache_creation_tokens = extract_cache_tokens(usage)
        input_tokens, completion_tokens, total_tokens = _token_counts(
            usage, cache_read_tokens, cache_creation_tokens, prompt_folds_cache
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

        completion_args = {
            "cache_creation_token_count": cache_creation_tokens,
            "cache_read_token_count": cache_read_tokens,
            "input_token_cost": None,
            "output_token_cost": None,
            "total_cost": None,
            "output_token_count": completion_tokens,
            "cost_type": "AI",
            "model": data.get("model"),
            "input_token_count": input_tokens,
            "provider": "LITELLM",
            "model_source": "LITELLM",
            "reasoning_token_count": 0,
            "request_time": format_utc_timestamp(request_time),
            "response_time": response_time_str,
            "completion_start_time": response_time_str,
            "request_duration": request_duration,
            "time_to_first_token": request_duration,
            "stop_reason": stop_reason,
            "total_token_count": total_tokens,
            "transaction_id": transaction_id,
            # Claude Code stamps its session id on its own telemetry rows as
            # their trace id, so reading it here is what makes one session
            # read as one trace on both records. Never the transaction id: a
            # session covers many calls, and keying the duplicate gate on it
            # would fold a whole session into one record and under-bill it.
            "trace_id": (headers.get("x-revenium-trace-id")
                         or headers.get("x-claude-code-session-id")),
            "task_type": headers.get("x-revenium-task-type"),
            "subscriber": subscriber if subscriber else None,
            "organization_name": _organization_name(
                headers, metadata, user_api_key_dict, key_metadata
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
        logger.debug("ReveniumGuardrail: metering (%s) with args: %s", label, completion_args)

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
