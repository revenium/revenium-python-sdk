"""Deprecated LiteLLM proxy callback.

``MiddlewareHandler`` is a ``CustomLogger`` that meters proxied calls but never
enforces a budget. It is superseded by
``revenium_middleware.litellm.proxy.guardrail.ReveniumGuardrail``, which adds
pre-call budget enforcement and covers everything this callback does. It keeps
working for this release; see the README's "Migrating from the callback"
section.

A proxy that enables both would meter every call twice. When the guardrail is
configured to run on every request it claims metering ownership (see
``_metering_owner``) and this callback stops submitting rows, so a proxy
mid-migration is not double-billed.
"""

import datetime
import logging
import uuid
import warnings

from litellm.integrations.custom_logger import CustomLogger
from revenium_middleware import client, get_client, run_async_in_thread
from revenium_middleware._core.fields import merge_extra_body
from revenium_middleware._core.cache_tokens import (
    billable_input_tokens,
    extract_cache_tokens,
    get_usage_field,
)
from revenium_middleware._core.config import is_shared_call_id_enabled
from revenium_middleware._core import submit_ai_event

from ._metering_owner import guardrail_owns_metering

logger = logging.getLogger("revenium_middleware.extension")

# Revenium's metering API reads these as instants, and the trailing "Z" in the
# pattern is a literal that promises UTC.
_TIMESTAMP_FORMAT = "%Y-%m-%dT%H:%M:%SZ"


def format_utc_timestamp(value):
    """A metering timestamp, always the UTC instant behind the trailing ``Z``.

    ``strftime`` renders whatever fields the object carries and copies the
    ``Z`` through as a literal, so formatting a naive datetime publishes a
    local wall-clock reading labelled UTC. The proxy hooks receive both kinds:
    the post-call path builds aware UTC values of its own, while LiteLLM hands
    every ``CustomLogger`` event -- the streamed Anthropic route among them --
    naive ``start_time`` and ``end_time`` taken from the proxy machine's local
    clock. On a proxy outside UTC that skewed every gateway row by the machine's
    offset, which was survivable only while the Claude Code row carried the
    right time beside it. Under the shared call id (BACK-2399) the gateway row
    is the one the duplicate check keeps, so the skew becomes the customer's
    only recorded time.

    ``astimezone`` handles both inputs: it converts an aware value, and reads a
    naive one against the platform's local zone, which is precisely what
    LiteLLM's naive values are.

    Args:
        value: A ``datetime``, aware or naive.

    Returns:
        The instant as ``YYYY-MM-DDTHH:MM:SSZ``.
    """
    return value.astimezone(datetime.timezone.utc).strftime(_TIMESTAMP_FORMAT)


def resolve_transaction_id(*candidates):
    """The first usable correlation id, or a fresh UUID -- never a constant.

    A constant sentinel is not a missing id, it is a *colliding* one. Revenium's
    transaction engine dedups on (organization, transactionId) and passes
    non-UUID strings through unchanged, so every failed call from a tenant
    sharing one sentinel after the first is acknowledged as a duplicate and
    never stored -- the failure-rate signal disappears exactly when it matters.
    A per-event UUID keeps each row distinct; correlation is still preferred
    when the request carries something real (LiteLLM's ``litellm_call_id``, or
    the response's own id).

    Args:
        *candidates: Correlation ids in order of preference. Non-strings are
            coerced; empty and ``None`` are skipped.

    Returns:
        A non-empty transaction id string.
    """
    for candidate in candidates:
        if candidate:
            text = str(candidate).strip()
            if text:
                return text
    return str(uuid.uuid4())

DEPRECATION_MESSAGE = (
    "revenium_middleware.litellm.proxy.middleware.MiddlewareHandler (the "
    "litellm_settings.callbacks entry 'proxy_handler_instance') is deprecated. "
    "Use revenium_middleware.litellm.proxy.guardrail.ReveniumGuardrail instead: "
    "it adds pre-call budget enforcement and meters everything this callback "
    "does. Remove the callback when you enable the guardrail -- running both "
    "meters every call twice. See the 'Migrating from the callback' section of "
    "the README."
)

# Said once per process, not once per metered call: the hazard is a
# configuration mistake, and repeating it per request would bury it.
_dedup_warning_emitted = False


def _warn_metering_is_owned_by_the_guardrail():
    """Log, once, that this callback is deferring to the guardrail."""
    global _dedup_warning_emitted
    if _dedup_warning_emitted:
        return
    _dedup_warning_emitted = True
    logger.warning(
        "MiddlewareHandler: a ReveniumGuardrail configured to run on every "
        "request is active, so this deprecated callback is NOT metering -- "
        "the guardrail meters instead. Running both would double-count every "
        "call. Remove the callbacks entry from litellm_settings."
    )


def _extract_organization_name(headers, metadata):
    organization_name = headers.get("x-revenium-organization-name")
    if organization_name is None:
        old_value = headers.get("x-revenium-organization-id")
        if old_value:
            logger.warning(
                "Header 'x-revenium-organization-id' is deprecated. "
                "Use 'x-revenium-organization-name' instead."
            )
        organization_name = old_value
    if organization_name is None:
        organization_name = metadata.get('user_api_key_team_alias', '') or None
    return organization_name


def extract_request_headers(container):
    """The inbound request headers, from whichever key this route filled.

    LiteLLM writes them to ``litellm_metadata`` on the routes in its
    ``LITELLM_METADATA_ROUTES`` tuple (``/v1/messages`` among them, which is the
    route Claude Code speaks) and to ``metadata`` everywhere else, and it writes
    ``proxy_server_request`` wholesale on every route. Reading ``metadata``
    alone therefore saw an empty dict on the Anthropic messages route and
    dropped every documented ``x-revenium-*`` header on it.

    ``proxy_server_request`` is preferred over ``metadata`` because LiteLLM
    assigns that dict itself on every route, while on the Anthropic messages
    route ``metadata`` is the caller's own request body and a caller can seed a
    ``headers`` mapping in it. Reading it second would hand a caller the
    subscriber, organization, product, subscription and agentic-job tags of
    their choosing.

    Every level is checked with ``hasattr(value, "get")`` rather than ``or {}``
    because a caller can put a non-dict under either metadata key and LiteLLM's
    own guard admits it. A truthy non-mapping passes ``or {}`` and then raises,
    and neither this callback nor the guardrail's hooks turn that into anything
    an operator sees. They turn it into a missing metering row.

    Public rather than underscored: ``guardrail.py`` imports it. It also stays
    Python 3.8 compatible, because this module does and the guardrail is the
    3.10+ side of that import.

    Args:
        container: A guardrail hook's ``data`` or a callback's
            ``litellm_params``. The first non-empty headers mapping wins.

    Returns:
        The headers mapping, or an empty dict when no tier carries one.
    """
    container = container if hasattr(container, "get") else {}
    for key in ("litellm_metadata", "proxy_server_request", "metadata"):
        level = container.get(key)
        level = level if hasattr(level, "get") else {}
        headers = level.get("headers")
        headers = headers if hasattr(headers, "get") else {}
        if headers:
            return headers
    return {}


# The keys the shared per-call id is stamped under, in both metadata dicts.
# Public rather than underscored: guardrail.py imports all four, the way it
# already imports resolve_transaction_id and extract_request_headers.
REVENIUM_CALL_ID_KEY = "revenium_call_id"
REVENIUM_CALL_TYPE_KEY = "revenium_call_type"
REVENIUM_CALL_MINT_KEY = "revenium_call_mint"
ANTHROPIC_MESSAGES_CALL_TYPE = "anthropic_messages"

# LiteLLM ``custom_llm_provider`` values whose usage conversion folds the cache
# buckets into ``prompt_tokens`` *and* whose rows the platform stores as sent.
# Anthropic is the one provider Revenium never normalizes server-side (see
# billable_input_tokens), so it is the one provider whose overlap this row may
# remove. Bedrock and Vertex Claude are absent deliberately: their LiteLLM
# transformations are not AnthropicConfig's, so their prompt counts are left
# exactly as LiteLLM built them until someone has measured one.
_CACHE_FOLDING_PROVIDERS = frozenset({"anthropic", "anthropic_text"})


def prompt_tokens_fold_cache(*sources):
    """True when this call's ``prompt_tokens`` is an Anthropic count plus cache.

    Only the upstream decides, never the numbers: an OpenAI-shaped provider
    reports ``cached_tokens`` inside a prompt count the platform itself nets
    out, so subtracting here would net it out twice and price the input leg at
    zero. See ``billable_input_tokens`` for both failure modes.

    ``custom_llm_provider`` is the field to read, and which surface carries it
    depends on the hook, so every caller hands in the ones it has and the first
    that names a provider decides -- the response's own before the request's.
    Only surfaces LiteLLM fills qualify: ``data["metadata"]`` on ``/v1/messages``
    is the caller's request body, so a value there would let a caller pick the
    arithmetic that sets their billed input count (see BACK-3190 for the same
    rule on headers).
    Measured against a live litellm 1.102.0 proxy: the logging event's
    ``kwargs`` and its ``litellm_params`` both carry it (also on the 1.93.0
    floor), while on the non-streamed ``/v1/messages`` route the post-call hook
    is handed a raw provider body with no ``_hidden_params`` and a request dict
    that names no provider -- there it is ``data["litellm_logging_obj"]``'s
    ``model_call_details`` that carries both the provider and the call type.
    An ``anthropic_messages`` call type says the same thing about a route
    LiteLLM only ever serves from an Anthropic-shaped upstream, and covers a
    surface carrying the call type but no provider.

    With nothing to read, the prompt count stays as LiteLLM built it: reporting
    a gross count costs the cache overlap twice for Anthropic only, while
    reporting a net one costs the whole input leg for every pool the platform
    normalizes.
    """
    for source in sources:
        level = source if hasattr(source, "get") else {}
        provider = level.get("custom_llm_provider")
        if provider:
            return str(provider).lower() in _CACHE_FOLDING_PROVIDERS
        if level.get("call_type") == ANTHROPIC_MESSAGES_CALL_TYPE:
            return True
    logger.debug(
        "No upstream provider on this event; reporting prompt_tokens as LiteLLM "
        "built it rather than assuming the cache tokens are inside it"
    )
    return False

# Minted once per proxy process, and the reason a read can tell a value this
# process minted from one a caller sent. See read_shared_call_id.
_MINT_NONCE = uuid.uuid4().hex


def record_shared_call_id(container, call_id, call_type):
    """Stamp a minted call id, its call type and this process's mint nonce.

    All three are written, because the id alone proves nothing: on
    ``/v1/messages`` the container the proxy hands us is the caller's own
    request body, so a caller can seed the id key themselves (LiteLLM strips
    only a fixed list of keys from the two metadata dicts, and
    ``litellm/proxy/litellm_pre_call_utils.py:2107-2125`` says as much in its
    own comment). The call type and the nonce are what make the read a fact.

    Silently does nothing for a container that is not a writable mapping: a
    caller can put a string or a list under either metadata key, and this runs
    inside a hook that must never raise.

    Args:
        container: The ``metadata`` or ``litellm_metadata`` dict to stamp.
        call_id: The minted id, a fresh uuid4 per request.
        call_type: The LiteLLM call type the mint ran for.
    """
    if not hasattr(container, "get") or not hasattr(container, "__setitem__"):
        return
    container[REVENIUM_CALL_ID_KEY] = call_id
    container[REVENIUM_CALL_TYPE_KEY] = call_type
    container[REVENIUM_CALL_MINT_KEY] = _MINT_NONCE


def read_shared_call_id(container, active):
    """The id this process minted for this request, or ``None``.

    Three gates, all required, because presence is never proof:

    * ``active`` is false unless the flag is on and this proxy's configuration
      can actually mint (the guardrail resolves that once at construction).
      Without it, a ``mode`` that never runs the pre-call hook would read back
      whatever the caller planted.
    * The recorded call type must be the Anthropic messages route. The mint
      never runs elsewhere, so a value found on another route came from the
      caller.
    * The recorded nonce must equal this process's. A caller who guesses both
      key names and the call type still cannot produce 32 hex digits minted at
      startup.

    A forged id that got through would let a caller hand two paid calls one
    transaction id; Revenium's team-scoped duplicate gate would drop the
    second, which is a false merge and a zero bill in one move.

    Every level is checked with ``hasattr(value, "get")`` rather than ``or {}``,
    the same shape ``extract_request_headers`` uses and for the same reason: a
    truthy non-mapping passes ``or {}`` and then raises, and this runs in hooks
    whose only visible failure is a metering row that never happened.

    Args:
        container: A guardrail hook's ``data`` or a callback's
            ``litellm_params``.
        active: Whether the shared call id is live for this proxy.

    Returns:
        The minted id, or ``None``.
    """
    if not active:
        return None
    container = container if hasattr(container, "get") else {}
    for key in ("litellm_metadata", "metadata"):
        level = container.get(key)
        level = level if hasattr(level, "get") else {}
        if level.get(REVENIUM_CALL_MINT_KEY) != _MINT_NONCE:
            continue
        if level.get(REVENIUM_CALL_TYPE_KEY) != ANTHROPIC_MESSAGES_CALL_TYPE:
            continue
        call_id = level.get(REVENIUM_CALL_ID_KEY)
        if call_id:
            return call_id
    return None


def _extract_product_name(headers):
    product_name = headers.get("x-revenium-product-name")
    if product_name is None:
        old_value = headers.get("x-revenium-product-id")
        if old_value:
            logger.warning(
                "Header 'x-revenium-product-id' is deprecated. "
                "Use 'x-revenium-product-name' instead."
            )
        product_name = old_value
    return product_name


def _extract_agentic_job_from_headers(headers):
    mapping = {
        "agenticJobId": "x-revenium-agentic-job-id",
        "agenticJobName": "x-revenium-agentic-job-name",
        "agenticJobType": "x-revenium-agentic-job-type",
        "agenticJobVersion": "x-revenium-agentic-job-version",
    }
    result = {}
    for wire_name, header_name in mapping.items():
        value = headers.get(header_name)
        if value:
            result[wire_name] = value
    return result


class MiddlewareHandler(CustomLogger):
    """Deprecated metering callback. See the module docstring."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # stacklevel=2 points at the line that constructed the handler -- for
        # an operator that is their own config, not this file.
        warnings.warn(DEPRECATION_MESSAGE, DeprecationWarning, stacklevel=2)
        # Also logged, because a LiteLLM proxy runs with warnings suppressed
        # more often than not, and its operator reads logs.
        logger.warning(DEPRECATION_MESSAGE)

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        if guardrail_owns_metering():
            _warn_metering_is_owned_by_the_guardrail()
            return  # the guardrail meters this call; metering it here would double-count
        if get_client() is None:
            return  # metering disabled (no API key configured)
        # log: key, user, model, prompt, response, tokens, cost
        # Access kwargs passed to litellm.completion()
        # pprint.pprint(kwargs['litellm_params'])

        model = kwargs.get("model", None)

        # Access litellm_params passed to litellm.completion(), example access `metadata`
        litellm_params = kwargs.get("litellm_params", {})
        # Still bound, and still the source for the virtual key's own fields
        # (the subscriber id, user_api_key_* and hidden_params below). Guarded
        # with the same hasattr shape extract_request_headers uses: a caller can
        # put a non-dict here, LiteLLM's own guard admits it, and this callback
        # has no wrapper of its own -- an AttributeError raised here is not an
        # error an operator sees, it is a metering row that never happened.
        metadata = litellm_params.get("metadata", {})
        metadata = metadata if hasattr(metadata, "get") else {}
        # NOT metadata["headers"]: LiteLLM fills litellm_metadata instead on
        # /v1/messages, the route Claude Code speaks. See extract_request_headers.
        headers = extract_request_headers(litellm_params)
        # The id ReveniumGuardrail's pre-call hook minted for this request, when
        # the flag is on. This callback sees it in the configuration where the
        # guardrail does not own metering (no default_on), which is the one
        # configuration where this callback is what writes the row, so leaving
        # the identity on the provider id here would lose the match exactly
        # where it is needed. The nonce and call type the guardrail recorded are
        # what make this safe to read from a container a caller can seed; see
        # read_shared_call_id.
        shared_call_id = read_shared_call_id(
            litellm_params, is_shared_call_id_enabled()
        )

        response = response_obj
        # tokens used in response. Read every field -- cache and base token
        # counts alike -- through the same tolerant accessor (see the
        # failure path below for why: reading some fields directly and
        # others through a tolerant accessor is what caused the two
        # preceding review rounds to each find another zeroed field).
        usage = response_obj["usage"]
        cache_read_tokens, cache_creation_tokens = extract_cache_tokens(usage)
        # An unconverted Anthropic body stays unpriced here on purpose: the
        # guardrail owns that route.
        input_tokens = billable_input_tokens(
            get_usage_field(usage, "prompt_tokens", 0),
            cache_read_tokens,
            cache_creation_tokens,
            prompt_tokens_fold_cache(
                getattr(response_obj, "_hidden_params", None), kwargs, litellm_params
            ),
        )
        completion_tokens = get_usage_field(usage, "completion_tokens", 0)
        total_tokens = get_usage_field(usage, "total_tokens", 0)

        # Create subscriber object from metadata and headers
        subscriber = {}

        # Extract subscriber information from metadata and headers
        subscriber_id = metadata.get('x-revenium-subscriber-id', '') or headers.get("x-revenium-subscriber-id")
        subscriber_email = metadata.get('user_api_key_user_email', '')
        credential_name = metadata.get('user_api_key_alias', '')
        credential_value = metadata.get('user_api_key_alias', '')

        if subscriber_id:
            subscriber["id"] = subscriber_id
        if subscriber_email:
            subscriber["email"] = subscriber_email
        if credential_name or credential_value:
            subscriber["credential"] = {
                "name": credential_name,
                "value": credential_value
            }

        organization_name = _extract_organization_name(headers, metadata)
        product_name = _extract_product_name(headers)
        agentic_fields = _extract_agentic_job_from_headers(headers)
        extra_body = merge_extra_body(None, agentic_fields)

        completion_args = {
            "cache_creation_token_count": cache_creation_tokens,
            "cache_read_token_count": cache_read_tokens,
            "input_token_cost": None,
            "output_token_cost": None,
            "total_cost": None,
            "output_token_count": completion_tokens,
            "cost_type": "AI",
            "model": model,
            "input_token_count": input_tokens,
            "provider": "LITELLM",
            "model_source": "LITELLM",
            "reasoning_token_count": 0,
            "request_time": format_utc_timestamp(start_time),
            "response_time": format_utc_timestamp(end_time),
            "completion_start_time": format_utc_timestamp(end_time),
            "request_duration": (end_time - start_time).total_seconds() * 1000,
            "time_to_first_token": (end_time - start_time).total_seconds() * 1000,
            "stop_reason": "END",
            "total_token_count": total_tokens,
            # "minted or response.id", never resolve_transaction_id(...): that
            # helper coerces with str(...).strip(), so wrapping this would
            # change the stored value's type for a non-string provider id even
            # with the flag off, and flag-off output has to stay identical.
            "transaction_id": shared_call_id or response.id,
            # Claude Code stamps its session id on its own telemetry rows as
            # their trace id, so reading it here is what makes one session read
            # as one trace on both records. Never the transaction id: a session
            # covers many calls, and keying the duplicate gate on it would fold
            # a whole session into one record and under-bill it.
            "trace_id": (headers.get("x-revenium-trace-id")
                         or headers.get("x-claude-code-session-id")),
            "task_type": headers.get("x-revenium-task-type"),
            "subscriber": subscriber if subscriber else None,
            "organization_name": organization_name,
            "subscription_id": headers.get("x-revenium-subscription-id"),
            "product_name": product_name,
            "agent": headers.get("x-revenium-agent"),
            # Present only when the caller sent the header: create_completion
            # drops NotGiven but keeps an explicit None, which would reach the
            # wire as "effort": null instead of being omitted.
            **({"effort": headers["x-revenium-effort"]}
               if "x-revenium-effort" in headers else {}),
            "response_quality_score": headers.get("x-revenium-response-quality-score"),
            "is_streamed": metadata.get('hidden_params', {}).get('optional_params', {}).get('stream', False),
            "operation_type": "CHAT",
            "mediation_latency": metadata.get('hidden_params', {}).get('litellm_overhead_time_ms', 0),
            "middleware_source": "PROXY",
        }

        logger.debug("Calling client.ai.create_completion with args: %s", completion_args)

        async def metering_call():
            try:
                result = submit_ai_event("completion", {**completion_args, "extra_body": extra_body})
                logger.debug("Proxy metering call result: %s", result)
            except Exception as e:
                logger.error("Proxy metering call failed: %s", e)

        run_async_in_thread(metering_call())

        return

    async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
        if guardrail_owns_metering():
            _warn_metering_is_owned_by_the_guardrail()
            return  # the guardrail meters this call; metering it here would double-count
        if get_client() is None:
            return  # metering disabled (no API key configured)
        # log: key, user, model, prompt, error, tokens, cost
        # Access kwargs passed to litellm.completion()
        # pprint.pprint(kwargs['litellm_params'])

        model = kwargs.get("model", None)

        # Access litellm_params passed to litellm.completion(), example access `metadata`
        litellm_params = kwargs.get("litellm_params", {})
        # Still bound, and still the source for the virtual key's own fields
        # (the subscriber id, user_api_key_* and hidden_params below). Guarded
        # with the same hasattr shape extract_request_headers uses: a caller can
        # put a non-dict here, LiteLLM's own guard admits it, and this callback
        # has no wrapper of its own -- an AttributeError raised here is not an
        # error an operator sees, it is a metering row that never happened.
        metadata = litellm_params.get("metadata", {})
        metadata = metadata if hasattr(metadata, "get") else {}
        # NOT metadata["headers"]: LiteLLM fills litellm_metadata instead on
        # /v1/messages, the route Claude Code speaks. See extract_request_headers.
        headers = extract_request_headers(litellm_params)

        # For failures, we may not have usage information. Read every field --
        # cache and base token counts alike -- through the same tolerant
        # accessor rather than normalizing `usage` to a dict first: that
        # normalization silently zeroed every field (not just cache) for
        # attribute-style usage objects, producing a payload with real cache
        # counts alongside zeroed prompt/completion/total tokens for the same
        # call.
        usage = getattr(response_obj, "usage", None)
        cache_read_tokens, cache_creation_tokens = extract_cache_tokens(usage)
        # An unconverted Anthropic body stays unpriced here on purpose: the
        # guardrail owns that route.
        input_tokens = billable_input_tokens(
            get_usage_field(usage, "prompt_tokens", 0),
            cache_read_tokens,
            cache_creation_tokens,
            prompt_tokens_fold_cache(
                getattr(response_obj, "_hidden_params", None), kwargs, litellm_params
            ),
        )
        completion_tokens = get_usage_field(usage, "completion_tokens", 0)
        total_tokens = get_usage_field(usage, "total_tokens", 0)

        error_message = str(response_obj)
        error_type = type(response_obj).__name__

        # Create subscriber object from metadata and headers
        subscriber = {}

        # Extract subscriber information from metadata and headers
        subscriber_id = metadata.get('x-revenium-subscriber-id', '') or headers.get("x-revenium-subscriber-id")
        subscriber_email = metadata.get('user_api_key_user_email', '')
        credential_name = metadata.get('user_api_key_alias', '')
        credential_value = metadata.get('user_api_key_hash', '')

        if subscriber_id:
            subscriber["id"] = subscriber_id
        if subscriber_email:
            subscriber["email"] = subscriber_email
        if credential_name or credential_value:
            subscriber["credential"] = {
                "name": credential_name,
                "value": credential_value
            }

        organization_name = _extract_organization_name(headers, metadata)
        product_name = _extract_product_name(headers)
        agentic_fields = _extract_agentic_job_from_headers(headers)
        extra_body = merge_extra_body(None, agentic_fields)

        completion_args = {
            "cache_creation_token_count": cache_creation_tokens,
            "cache_read_token_count": cache_read_tokens,
            "input_token_cost": None,
            "output_token_cost": None,
            "total_cost": None,
            "output_token_count": completion_tokens,
            "cost_type": "AI",
            "model": model,
            "input_token_count": input_tokens,
            "provider": "LITELLM",
            "model_source": "LITELLM",
            "reasoning_token_count": 0,
            "request_time": format_utc_timestamp(start_time),
            "response_time": format_utc_timestamp(end_time),
            "completion_start_time": format_utc_timestamp(end_time),
            "request_duration": (end_time - start_time).total_seconds() * 1000,
            "time_to_first_token": (end_time - start_time).total_seconds() * 1000,
            "stop_reason": "ERROR",
            "total_token_count": total_tokens,
            # Never a constant: a shared sentinel collides in Revenium's
            # (organization, transactionId) dedup and drops every failed call
            # after the first. See resolve_transaction_id.
            #
            # The ":err:<8 hex>" suffix is minted per event, never per handler
            # or per process. Correlation still comes from the prefix, but a
            # router retry hands the failed attempt and the paid attempt the
            # same litellm_call_id, so without the suffix the failure took the
            # success's identity and the paid row was dropped as a duplicate at
            # zero cost. No success identity can contain the marker: those are a
            # provider response id, a litellm_call_id or a bare UUID.
            "transaction_id": "{0}:err:{1}".format(
                resolve_transaction_id(
                    getattr(response_obj, "id", None),
                    kwargs.get("litellm_call_id"),
                    litellm_params.get("litellm_call_id"),
                ),
                uuid.uuid4().hex[:8],
            ),
            # Claude Code stamps its session id on its own telemetry rows as
            # their trace id, so reading it here is what makes one session read
            # as one trace on both records. Never the transaction id: a session
            # covers many calls, and keying the duplicate gate on it would fold
            # a whole session into one record and under-bill it.
            "trace_id": (headers.get("x-revenium-trace-id")
                         or headers.get("x-claude-code-session-id")),
            "task_type": headers.get("x-revenium-task-type"),
            "subscriber": subscriber if subscriber else None,
            "organization_name": organization_name,
            "subscription_id": headers.get("x-revenium-subscription-id"),
            "product_name": product_name,
            "agent": headers.get("x-revenium-agent"),
            # Present only when the caller sent the header: create_completion
            # drops NotGiven but keeps an explicit None, which would reach the
            # wire as "effort": null instead of being omitted.
            **({"effort": headers["x-revenium-effort"]}
               if "x-revenium-effort" in headers else {}),
            "response_quality_score": headers.get("x-revenium-response-quality-score"),
            "is_streamed": metadata.get('hidden_params', {}).get('optional_params', {}).get('stream', False),
            "operation_type": "CHAT",
            "middleware_source": "PROXY",
        }

        logger.debug("Calling client.ai.create_completion with args (failure): %s", completion_args)

        async def metering_call():
            try:
                result = submit_ai_event("completion", {**completion_args, "extra_body": extra_body})
                logger.debug("Result from create_completion (failure): %s", result)
            except Exception as e:
                logger.error("Error logging failure event: %s", e)

        run_async_in_thread(metering_call())


# Built on first access rather than at import, so merely importing the SDK's
# LiteLLM subpackage does not emit a DeprecationWarning at a guardrail user who
# never asked for the callback. LiteLLM resolves the
# "…middleware.proxy_handler_instance" string in litellm_settings.callbacks by
# importing this module and reading the attribute, which is exactly the moment
# the operator should hear about the deprecation.
_proxy_handler_instance = None


def __getattr__(name):
    global _proxy_handler_instance
    if name == "proxy_handler_instance":
        if _proxy_handler_instance is None:
            _proxy_handler_instance = MiddlewareHandler()
        return _proxy_handler_instance
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
