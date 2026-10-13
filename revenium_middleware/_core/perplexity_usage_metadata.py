from collections.abc import Mapping
from typing import Any, Dict


def _as_metadata(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def take_usage_metadata(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Remove the caller's Revenium metadata from a Perplexity request and return it.

    It is accepted as a ``usage_metadata`` keyword or inside ``extra_body``;
    neither may reach Perplexity. Where both carry a field, the keyword wins.
    The caller's own ``extra_body`` dict is left unchanged: ``kwargs`` gets a
    copy without the metadata, so a retry with the same dict keeps its attribution.
    """
    top_level = _as_metadata(kwargs.pop("usage_metadata", None))
    from_extra_body: Dict[str, Any] = {}
    extra_body = kwargs.get("extra_body")
    if isinstance(extra_body, dict):
        extra_body = kwargs["extra_body"] = dict(extra_body)
        from_extra_body = _as_metadata(extra_body.pop("usage_metadata", None))
    return {**from_extra_body, **top_level}
