import logging
import threading
import warnings
from typing import Any, Dict, Mapping, Optional, Set, Tuple

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


def extract_agentic_job_fields(source: Mapping[str, Any]) -> Dict[str, Any]:
    result = {}
    for wire_name, aliases in AGENTIC_JOB_FIELD_MAP.items():
        for alias in aliases:
            value = source.get(alias)
            if value is not None:
                result[wire_name] = value
                break
    return result


def merge_extra_body(existing: Optional[Dict[str, Any]], agentic_fields: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not agentic_fields:
        return existing if existing else None
    merged = dict(existing) if existing else {}
    for k, v in agentic_fields.items():
        merged.setdefault(k, v)
    return merged
