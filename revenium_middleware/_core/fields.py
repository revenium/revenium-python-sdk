import logging
import os
import threading
import warnings
from typing import Any, Callable, Dict, Mapping, Optional, Set, Tuple

from .config import Config
from .context import get_agentic_job_fields

logger = logging.getLogger(__name__)

# Dedup set for deprecated-alias logger.warning so high-volume callers don't get a flood
# per call. Python's warnings module already dedupes warnings.warn by call site, but
# logger.warning has no built-in dedup. The lock makes the check-then-add atomic so
# concurrent threads can't both observe "not seen" and emit duplicate startup warnings.
_WARNED_DEPRECATED_FIELDS: Set[Tuple[str, str]] = set()
_WARNED_DEPRECATED_LOCK = threading.Lock()

AGENTIC_JOB_FIELD_MAP = {
    "agenticJobId": ("agentic_job_id", "agenticJobId"),
    "agenticJobName": ("agentic_job_name", "agenticJobName"),
    "agenticJobType": ("agentic_job_type", "agenticJobType"),
    "agenticJobVersion": ("agentic_job_version", "agenticJobVersion"),
}

_AGENTIC_JOB_ENV_MAP = {
    "agenticJobId": Config.ENV_REVENIUM_AGENTIC_JOB_ID,
    "agenticJobName": Config.ENV_REVENIUM_AGENTIC_JOB_NAME,
    "agenticJobType": Config.ENV_REVENIUM_AGENTIC_JOB_TYPE,
    "agenticJobVersion": Config.ENV_REVENIUM_AGENTIC_JOB_VERSION,
}

# Keys are the typed snake_case params on create_completion; values are the
# accepted usage_metadata aliases in precedence order.
SKILL_FIELD_MAP = {
    "skill_invocation_trigger": ("skill_invocation_trigger", "skillInvocationTrigger"),
    "skill_kind": ("skill_kind", "skillKind"),
    "skill_marketplace_name": ("skill_marketplace_name", "skillMarketplaceName"),
    "skill_name": ("skill_name", "skillName"),
    "skill_plugin_name": ("skill_plugin_name", "skillPluginName"),
    "skill_source": ("skill_source", "skillSource"),
}

# Keys are the typed snake_case params on create_completion; values are the
# accepted usage_metadata aliases in precedence order. Coding assistant
# attribution is caller-supplied only -- the middleware cannot infer it, so
# there is deliberately no env-var fallback map beside this one.
CODING_ASSISTANT_FIELD_MAP = {
    "coding_assistant_account_uuid": ("coding_assistant_account_uuid", "codingAssistantAccountUuid"),
}

_SKILL_ENV_MAP = {
    "skill_invocation_trigger": Config.ENV_REVENIUM_SKILL_INVOCATION_TRIGGER,
    "skill_kind": Config.ENV_REVENIUM_SKILL_KIND,
    "skill_marketplace_name": Config.ENV_REVENIUM_SKILL_MARKETPLACE_NAME,
    "skill_name": Config.ENV_REVENIUM_SKILL_NAME,
    "skill_plugin_name": Config.ENV_REVENIUM_SKILL_PLUGIN_NAME,
    "skill_source": Config.ENV_REVENIUM_SKILL_SOURCE,
}

# Keys are the typed snake_case params on create_image / create_video; values are
# the accepted usage_metadata aliases in precedence order. These carry media
# lineage for edit-and-regenerate flows, so they are per-call only (no env var).
MEDIA_LINEAGE_FIELD_MAP = {
    "source_transaction_id": ("source_transaction_id", "sourceTransactionId"),
}

# Keys are the typed snake_case params on create_completion; values are the
# accepted usage_metadata aliases in precedence order. The reasoning effort
# level is caller-supplied only -- like the coding assistant attribution there
# is deliberately no env-var fallback, because the level is a property of one
# request, not of the process. The value is passed through verbatim: no
# allow-list, no lowercasing, no coercion onto a known level. The backend owns
# validation (at most 16 characters matching ^[A-Za-z0-9_-]+$), so a vendor's
# next level reaches Revenium instead of being dropped client-side.
EFFORT_FIELD_MAP = {
    "effort": ("effort",),
}

# Keys are the typed snake_case params on the AI metering methods; values are
# the accepted usage_metadata aliases in precedence order. costType is
# deliberately absent: it is a pass-through wire field, never populated here.
SERVICE_TIER_FIELD_MAP = {
    "actual_service_tier": ("actual_service_tier", "actualServiceTier"),
    "requested_service_tier": ("requested_service_tier", "requestedServiceTier"),
    "pricing_tier": ("pricing_tier", "pricingTier"),
    "subscription_tier": ("subscription_tier", "subscriptionTier"),
    "cost_multiplier": ("cost_multiplier", "costMultiplier"),
    "priority_tier": ("priority_tier", "priorityTier"),
}

# Not every AI endpoint accepts every tier field: subscription_tier and
# cost_multiplier exist only on /v2/ai/completions, priority_tier only on
# /v2/ai/video and /v2/ai/images. Forwarding an unsupported one would be a
# TypeError against the typed create_* methods, so callers name their endpoint.
SERVICE_TIER_FIELDS_BY_ENDPOINT = {
    "completion": (
        "actual_service_tier",
        "requested_service_tier",
        "pricing_tier",
        "subscription_tier",
        "cost_multiplier",
    ),
    "audio": (
        "actual_service_tier",
        "requested_service_tier",
        "pricing_tier",
    ),
    "video": (
        "actual_service_tier",
        "requested_service_tier",
        "pricing_tier",
        "priority_tier",
    ),
    "image": (
        "actual_service_tier",
        "requested_service_tier",
        "pricing_tier",
        "priority_tier",
    ),
}


def extract_field_with_fallback(
    source: Mapping[str, Any],
    new_snake: str,
    new_camel: str,
    old_snake: str,
    old_camel: str,
    field_label: str,
) -> Any:
    value = source.get(new_snake)
    if value is None:
        value = source.get(new_camel)
    if value is None:
        value = source.get(old_snake)
    if value is None:
        value = source.get(old_camel)

    if source.get(old_snake) or source.get(old_camel):
        if not (source.get(new_snake) or source.get(new_camel)):
            msg = (
                "Fields '%s' and '%s' are deprecated and are no longer "
                "accepted by the Revenium backend. The SDK is translating to "
                "'%s' for this call. Use '%s' or '%s' instead. The "
                "input-layer aliases will be removed in the next major release."
            )
            pair = (old_snake, new_snake)
            with _WARNED_DEPRECATED_LOCK:
                should_log = pair not in _WARNED_DEPRECATED_FIELDS
                if should_log:
                    _WARNED_DEPRECATED_FIELDS.add(pair)
            if should_log:
                logger.warning(msg, old_camel, old_snake, new_camel, new_camel, new_snake)
            warnings.warn(msg % (old_camel, old_snake, new_camel, new_camel, new_snake), DeprecationWarning, stacklevel=3)

    return value


def extract_with_aliases(source: Mapping[str, Any], snake: str, camel: str) -> Any:
    value = source.get(snake)
    if value is None:
        value = source.get(camel)
    return value


def extract_organization_name(source: Mapping[str, Any]) -> Any:
    return extract_field_with_fallback(
        source,
        "organization_name", "organizationName",
        "organization_id", "organizationId",
        "organization",
    )


def extract_product_name(source: Mapping[str, Any]) -> Any:
    return extract_field_with_fallback(
        source,
        "product_name", "productName",
        "product_id", "productId",
        "product",
    )


def extract_org_and_product(source: Mapping[str, Any]) -> tuple:
    return extract_organization_name(source), extract_product_name(source)


def extract_common_metadata(source: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "trace_id": extract_with_aliases(source, "trace_id", "traceId"),
        "task_type": extract_with_aliases(source, "task_type", "taskType"),
        "subscription_id": extract_with_aliases(source, "subscription_id", "subscriptionId"),
        "agent": extract_with_aliases(source, "agent", "agent"),
        "response_quality_score": extract_with_aliases(source, "response_quality_score", "responseQualityScore"),
    }


def _resolve_field_map(
    source: Mapping[str, Any],
    field_map: Mapping[str, tuple],
    fallback: Optional[Callable[[str], Any]] = None,
) -> Dict[str, Any]:
    """Resolve a field family from caller metadata.

    Each field is read from ``source`` under its aliases in precedence order
    (snake_case before camelCase); when every alias misses, the optional
    ``fallback(name)`` supplies the value (contextvar or env lookup, per
    family). Absent fields are omitted entirely (never emitted as None).
    """
    result = {}
    for name, aliases in field_map.items():
        value = None
        for alias in aliases:
            value = source.get(alias)
            if value is not None:
                break
        if value is None and fallback is not None:
            value = fallback(name)
        if value is not None:
            result[name] = value
    return result


def extract_agentic_job_fields(source: Mapping[str, Any]) -> Dict[str, Any]:
    """Resolve agentic job fields with per-field precedence:

    explicit source metadata (snake then camel alias) > job contextvar >
    ``REVENIUM_AGENTIC_JOB_*`` env var. Each field resolves independently.
    """
    context_fields = get_agentic_job_fields() or {}

    def _fallback(wire_name: str) -> Any:
        value = context_fields.get(wire_name)
        if value is None:
            value = os.getenv(_AGENTIC_JOB_ENV_MAP[wire_name]) or None
        return value

    return _resolve_field_map(source, AGENTIC_JOB_FIELD_MAP, _fallback)


def extract_skill_fields(source: Mapping[str, Any]) -> Dict[str, Any]:
    """Resolve skill attribution fields with per-field precedence:

    explicit source metadata (snake then camel alias) >
    ``REVENIUM_SKILL_*`` env var. Each field resolves independently and
    absent fields are omitted entirely (never emitted as None). Returns
    snake_case keys matching the typed create_completion parameters.
    """
    return _resolve_field_map(
        source,
        SKILL_FIELD_MAP,
        lambda param_name: os.getenv(_SKILL_ENV_MAP[param_name]) or None,
    )


def extract_coding_assistant_fields(source: Mapping[str, Any]) -> Dict[str, Any]:
    """Resolve coding assistant attribution fields from caller metadata.

    Each field resolves from its snake_case key then its camelCase alias, and
    absent fields are omitted entirely (never emitted as None). Returns
    snake_case keys matching the typed create_completion parameters. Unlike the
    skill and agentic job families there is no env-var fallback: this is
    caller-supplied attribution the middleware has no way to infer.
    """
    return _resolve_field_map(source, CODING_ASSISTANT_FIELD_MAP)


def extract_effort_field(source: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """Resolve the reasoning effort level from caller-supplied metadata.

    Reads ``effort`` from the ``usage_metadata`` mapping and returns it as the
    snake_case keyword the typed ``create_completion`` parameter expects. An
    absent effort is omitted entirely (never emitted as None) so the typed
    client keeps its NotGiven default and nothing extra reaches the wire.

    The value is forwarded verbatim. There is no client-side allow-list and no
    normalization: an unrecognised but well-formed level (a vendor's next tier)
    is passed through, and a value the backend rejects fails visibly on the
    metering call rather than being silently dropped here.
    """
    if not source:
        return {}
    return _resolve_field_map(source, EFFORT_FIELD_MAP)


def extract_service_tier_fields(
    source: Mapping[str, Any],
    endpoint: str = "completion",
) -> Dict[str, Any]:
    """Resolve the service-tier and pricing fields for one AI endpoint.

    Each field is read from the source metadata with snake_case taking
    precedence over its camelCase alias, exactly as ``organization_id`` /
    ``organizationId`` resolve. Absent fields are omitted entirely (never
    emitted as None) so the typed create_* methods keep their NotGiven
    default and nothing extra reaches the wire.

    Args:
        source: The caller-supplied ``usage_metadata`` mapping.
        endpoint: One of ``completion``, ``audio``, ``video`` or ``image``;
            selects which fields that path's params type accepts.

    Returns:
        Dictionary of snake_case keyword arguments for the metering call.
    """
    accepted = SERVICE_TIER_FIELDS_BY_ENDPOINT[endpoint]
    return _resolve_field_map(
        source,
        {name: SERVICE_TIER_FIELD_MAP[name] for name in accepted},
    )


def extract_media_lineage_fields(source: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """Resolve media lineage fields from caller-supplied metadata.

    Each field resolves from its snake_case key then its camelCase alias, and
    absent fields are omitted entirely (never emitted as None) so an unset
    field keeps the typed client's NotGiven default instead of reaching the
    wire as an explicit null. Returns snake_case keys matching the typed
    ``create_image`` / ``create_video`` parameters.
    """
    if not source:
        return {}
    return _resolve_field_map(source, MEDIA_LINEAGE_FIELD_MAP)


def merge_extra_body(existing: Optional[Dict[str, Any]], agentic_fields: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not agentic_fields:
        return existing if existing else None
    merged = dict(existing) if existing else {}
    for k, v in agentic_fields.items():
        merged.setdefault(k, v)
    return merged
