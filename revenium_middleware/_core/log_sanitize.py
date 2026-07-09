"""
Shared log sanitization for all Revenium provider middlewares.

Debug logging must never emit raw request kwargs: auth material (API keys,
bearer tokens, headers) and prompt/message content (potential PII) are
redacted before anything hits application logs.

Usage:
    logger.debug("Calling wrapped function with args: %s, kwargs: %s",
                 sanitize_for_logging(args), sanitize_for_logging(kwargs))
"""

from typing import Any

__all__ = ["sanitize_for_logging"]

# Maximum recursion depth when sanitizing nested structures.
MAX_SANITIZATION_DEPTH: int = 8

# Strings longer than this are truncated (they may embed sensitive data).
MAX_LOG_STRING_LENGTH: int = 100

# Case-insensitive substring markers: any dict key containing one of these
# has its value replaced with "[REDACTED]".
SENSITIVE_KEY_MARKERS = frozenset(
    {
        "api_key",
        "apikey",
        "api-key",
        "secret",
        "token",
        "password",
        "credential",
        "authorization",
        "auth",
        "bearer",
        "private",
    }
)

# Case-insensitive exact-match keys whose values carry prompt/message content
# (potential PII). Values are summarized without recursing into them.
CONTENT_FIELD_NAMES = frozenset(
    {
        "messages",
        "prompt",
        "system",
        "input",
        "contents",
    }
)

_REDACTED = "[REDACTED]"


def _summarize_content(value: Any) -> str:
    """Summarize prompt/message content without exposing (or recursing into) it."""
    if isinstance(value, (list, tuple)):
        return f"[REDACTED: {len(value)} items]"
    return _REDACTED


def sanitize_for_logging(data: Any, max_depth: int = MAX_SANITIZATION_DEPTH) -> Any:
    """
    Sanitize data for secure logging.

    - Values under keys containing a sensitive marker (api_key, token,
      authorization, ...) are replaced with "[REDACTED]".
    - Values under prompt/message content keys (messages, prompt, system,
      input, contents) are summarized without recursing into them.
    - Long strings are truncated to avoid leaking embedded payloads.

    Args:
        data: Data to sanitize (dict, list/tuple, or primitive)
        max_depth: Maximum recursion depth to prevent runaway recursion

    Returns:
        Sanitized data safe for logging
    """
    if max_depth <= 0:
        return "[MAX_DEPTH_REACHED]"

    if isinstance(data, dict):
        sanitized = {}
        for key, value in data.items():
            key_lower = str(key).lower()
            if key_lower in CONTENT_FIELD_NAMES:
                sanitized[key] = _summarize_content(value)
            elif any(marker in key_lower for marker in SENSITIVE_KEY_MARKERS):
                sanitized[key] = _REDACTED
            else:
                sanitized[key] = sanitize_for_logging(value, max_depth - 1)
        return sanitized
    elif isinstance(data, (list, tuple)):
        return [sanitize_for_logging(item, max_depth - 1) for item in data]
    elif isinstance(data, str) and len(data) > MAX_LOG_STRING_LENGTH:
        # Truncate very long strings that might contain sensitive data
        return data[:MAX_LOG_STRING_LENGTH] + "...[TRUNCATED]"
    else:
        return data
