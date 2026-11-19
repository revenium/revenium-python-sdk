"""The inference geography goes only on a Claude record Anthropic itself bills.

Only Anthropic's own API applies a US-inference rate, so the field is left off
a Claude record that Bedrock, Foundry, Google Cloud or a LiteLLM route bills.
How Revenium itself treats the field on such a record is tracked by BACK-5560.
"""

import logging
from typing import Any, Dict, Mapping

from revenium_middleware._core.cache_tokens import names_claude_model

logger = logging.getLogger("revenium_middleware")

__all__ = ["without_inference_geo_anthropic_does_not_bill"]

_INFERENCE_GEO = "inference_geo"
_ANTHROPIC = "anthropic"


def _label(record: Mapping[str, Any], field: str) -> str:
    return str(record.get(field) or "").strip().lower()


def _serves_claude(record: Mapping[str, Any]) -> bool:
    return _label(record, "model_source") == _ANTHROPIC or names_claude_model(record.get("model"))


def _billed_by_anthropic(record: Mapping[str, Any]) -> bool:
    return _label(record, "provider") == _ANTHROPIC


def without_inference_geo_anthropic_does_not_bill(record: Mapping[str, Any]) -> Dict[str, Any]:
    """A copy of a completion ``record`` without ``inference_geo`` when it is a
    Claude call another party bills; any other record is copied whole."""
    if record.get(_INFERENCE_GEO) is None or _billed_by_anthropic(record) or not _serves_claude(record):
        return dict(record)
    logger.debug("inference_geo left off a Claude record that Anthropic does not bill")
    return {field: value for field, value in record.items() if field != _INFERENCE_GEO}
