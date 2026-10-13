"""Stop reason of an OpenAI Responses API response, shared by the integrations that meter one."""
from typing import Any

_RESPONSES_INCOMPLETE_REASONS = {
    "max_output_tokens": "TOKEN_LIMIT",
    "content_filter": "ERROR",
}


def _get_value(source: Any, key: str, default: Any = None) -> Any:
    if source is None:
        return default
    if isinstance(source, dict):
        return source.get(key, default)
    return getattr(source, key, default)


def responses_stop_reason(response: Any) -> str:
    """Map a Responses API ``status`` and ``incomplete_details.reason`` to a
    Revenium stop reason. Values are compared, never truth-tested, so an
    unknown status or reason, or a test double, reads as END."""
    status = _get_value(response, "status")
    if status == "failed":
        return "ERROR"
    if status == "incomplete":
        reason = _get_value(_get_value(response, "incomplete_details"), "reason")
        if reason in _RESPONSES_INCOMPLETE_REASONS:
            return _RESPONSES_INCOMPLETE_REASONS[reason]
    return "END"
