import logging
from typing import Any, Dict, Mapping, Optional

logger = logging.getLogger(__name__)

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
            logger.warning(
                "Fields '%s' and '%s' are deprecated. Use '%s' or '%s' instead. "
                "The old fields will be removed in a future version.",
                old_camel, old_snake, new_camel, new_snake,
            )

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
