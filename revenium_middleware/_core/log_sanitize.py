"""
Shared log sanitization for all Revenium provider middlewares.

Debug logging must never emit raw request kwargs, responses or usage records:
auth material (API keys, bearer tokens, credentials, emails) and prompt,
message or reply content (potential PII) are redacted before anything hits
application logs.

Rules, per dict entry (keys are compared lowercased, without ``_`` and ``-``,
so ``system_prompt`` and ``systemPrompt`` match alike):

- A string that looks like a secret or an email is redacted under any key.
- Attribution fields (organization, product, model, trace names, ...) print
  verbatim up to 64 characters.
- Under a sensitive key (api key, token, credential, email, any ``...key``)
  the value is redacted; a number under a ``...count`` or ``...tokens`` key
  still prints.
- Under a content key (messages, prompt, instructions, ...) the value is
  summarized without recursing into it.
- Any other string with whitespace, or longer than 64 characters, is free
  text and is redacted with its length only.
- Numbers, booleans and None print; any other object prints as its type name,
  never its repr.

Usage:
    logger.debug("Calling wrapped function with args: %s, kwargs: %s",
                 sanitize_for_logging(args), sanitize_for_logging(kwargs))
"""

import re
from collections.abc import Mapping
from typing import Any

from revenium_middleware._core.secret_shapes import (
    SECRET_LABEL_ENDINGS,
    SECRET_SHAPES,
    contains_secret_shape,
    mask_jwts,
    mask_secret_shapes,
)

__all__ = ["sanitize_for_logging", "redact_error_text"]

# Maximum recursion depth when sanitizing nested structures.
MAX_SANITIZATION_DEPTH: int = 8

# Strings longer than this, or containing whitespace, are treated as free text.
MAX_LOG_STRING_LENGTH: int = 64

# Substring markers on the normalized key: the value is replaced with "[REDACTED]".
SENSITIVE_KEY_MARKERS = frozenset(
    {
        "apikey",
        "secret",
        "token",
        "password",
        "credential",
        "authorization",
        "auth",
        "bearer",
        "private",
        "email",
    }
)

# Normalized keys whose values carry prompt, message or reply content.
CONTENT_FIELD_NAMES = frozenset(
    {
        "messages",
        "prompt",
        "system",
        "input",
        "contents",
        "content",
        "instructions",
        "systeminstruction",
        "query",
        "text",
        "parts",
        "suffix",
        "output",
        "outputresponse",
        "completion",
    }
)

# Substring markers on the normalized key for content fields not listed above.
CONTENT_KEY_MARKERS = frozenset({"prompt", "message", "instruction"})

# Normalized keys whose short values are printed as is, spaces included.
ATTRIBUTION_FIELD_NAMES = frozenset(
    {
        "organizationname",
        "organizationid",
        "productname",
        "productid",
        "agent",
        "taskid",
        "tasktype",
        "tracename",
        "traceid",
        "transactionname",
        "transactionid",
        "model",
        "provider",
        "stopreason",
        "environment",
        "region",
        "operationtype",
        "subscriptionid",
        "credentialname",
    }
)

# Value prefixes of credentials, from SecurityConfig.SENSITIVE_PATTERNS plus
# the Revenium, Google and legacy hosted-API key shapes.
SECRET_VALUE_PREFIXES = ("sk-", "pk-", "Bearer ", "Basic ", "Token ", "rev_sk_", "rev_mk_", "hak_", "AIza")

_SECRET_VALUE_PATTERN = re.compile(
    r"(?<![A-Za-z0-9])(?:" + "|".join(re.escape(prefix) for prefix in SECRET_VALUE_PREFIXES) + ")"
)
_EMAIL_PATTERN = re.compile(r"(?<![^\s@])[^\s@]+@[^\s@]+\.[^\s@]+")
_WHITESPACE_PATTERN = re.compile(r"\s")

_REDACTED = "[REDACTED]"
_COUNT_KEY_SUFFIXES = ("count", "tokens")


def _normalize_key(key: Any) -> str:
    return str(key).lower().replace("_", "").replace("-", "")


def _is_number_or_none(value: Any) -> bool:
    return value is None or isinstance(value, (bool, int, float))


def _looks_secret(text: str) -> bool:
    return bool(_SECRET_VALUE_PATTERN.search(text) or _EMAIL_PATTERN.search(text)) or contains_secret_shape(text)


def _is_sensitive_key(key: str) -> bool:
    return key.endswith("key") or any(marker in key for marker in SENSITIVE_KEY_MARKERS)


def _is_content_key(key: str) -> bool:
    return key in CONTENT_FIELD_NAMES or any(marker in key for marker in CONTENT_KEY_MARKERS)


def _summarize_content(value: Any) -> str:
    """Summarize prompt/message content without exposing (or recursing into) it."""
    if isinstance(value, (list, tuple)):
        return f"[REDACTED: {len(value)} items]"
    return _REDACTED


def _sanitize_string(text: str) -> str:
    if _looks_secret(text):
        return _REDACTED
    if len(text) > MAX_LOG_STRING_LENGTH or _WHITESPACE_PATTERN.search(text):
        return f"[REDACTED: {len(text)} chars]"
    return text


def _sanitize_attribution(value: Any, max_depth: int) -> Any:
    if isinstance(value, str) and not _looks_secret(value) and len(value) <= MAX_LOG_STRING_LENGTH:
        return value
    return _sanitize_value(value, min(1, max_depth - 1))


def _sanitize_entry(key: Any, value: Any, max_depth: int) -> Any:
    normalized = _normalize_key(key)
    if normalized in ATTRIBUTION_FIELD_NAMES:
        return _sanitize_attribution(value, max_depth)
    if _is_sensitive_key(normalized):
        if _is_number_or_none(value) and normalized.endswith(_COUNT_KEY_SUFFIXES):
            return value
        return _REDACTED
    if _is_content_key(normalized) and not _is_number_or_none(value):
        return _summarize_content(value)
    return _sanitize_value(value, max_depth - 1)


def _sanitize_value(data: Any, max_depth: int) -> Any:
    if max_depth <= 0:
        return "[MAX_DEPTH_REACHED]"
    if isinstance(data, Mapping):
        return {key: _sanitize_entry(key, value, max_depth) for key, value in data.items()}
    if isinstance(data, (list, tuple)):
        return [_sanitize_value(item, max_depth - 1) for item in data]
    if isinstance(data, str):
        return _sanitize_string(data)
    if _is_number_or_none(data):
        return data
    return f"[{type(data).__name__}]"


def sanitize_for_logging(data: Any, max_depth: int = MAX_SANITIZATION_DEPTH) -> Any:
    """
    Return a copy of ``data`` that is safe to log, following the module rules.

    The input is never modified: callers log the result and send the original.

    Args:
        data: Data to sanitize (mapping, list/tuple, or primitive)
        max_depth: Maximum recursion depth to prevent runaway recursion

    Returns:
        Sanitized data safe for logging
    """
    return _sanitize_value(data, max_depth)


# The second branch matches a ?name or &name with no "=" and is put back
# unchanged by _redact_query_value. It consumes the whole run in one step, so
# the scan stays linear instead of retrying every position inside a long run.
_QUERY_VALUE_PATTERN = re.compile(r"""([?&][^=&#\s'"]+=)[^&#\s'"]*|[?&][^=&#\s'"]+""")
_AUTH_SCHEME_PATTERN = re.compile(r"((?<![A-Za-z0-9_](?<!\\[nrt]))(?:Bearer|Basic|Token)\s+)[^\s'\",]+", re.IGNORECASE)
_URL_USERINFO_PATTERN = re.compile(r"(://)[^/\s:@]*:[^/\s@]+@")
# Kept out of SENSITIVE_KEY_MARKERS, which also decides which debug-log keys are hidden.
_ERROR_TEXT_SECRET_NAMES = (SENSITIVE_KEY_MARKERS | set(SECRET_LABEL_ENDINGS) | {"signature"}) - {"apikey"}
_SENSITIVE_NAME_ALTERNATION = "|".join(
    [r"api[_-]?key", r"key(?=[\"']?\s*[:=])"] + sorted(re.escape(name) for name in _ERROR_TEXT_SECRET_NAMES)
)
# The sensitive name must end the label, so "SYSTEM-SECRET-INSTRUCTIONS: never"
# keeps its word. Python 3.10 has no atomic groups, so (?=(?P<word>...))(?P=word)
# stands in for one, and the lookbehind starts a match only at the start of a
# word. Without both, repeated sensitive names make this scan cubic.
_SENSITIVE_ASSIGNMENT_PATTERN = re.compile(
    r"""((?<![\w-])["']?(?=[\w-]*?(?:""" + _SENSITIVE_NAME_ALTERNATION + r""")(?![\w-]))(?=(?P<word>[\w-]+))(?P=word)["']?\s*[:=]\s*)"""
    r"""(?:"[^"]*"|'[^']*'|[^\s,;&'"}]+)""",
    re.IGNORECASE,
)
_SECRET_SHAPED_WORD_PATTERN = re.compile(
    r"(?<![A-Za-z0-9](?<!\\[nrt]))(?:"
    + "|".join(re.escape(prefix) for prefix in SECRET_VALUE_PREFIXES if not prefix.endswith(" "))
    + r")[^\s'\",;&]*"
)
_EMAIL_WORD_CHAR = r"""[^\s@'",;:<>()]"""
_EMAIL_WORD_PATTERN = re.compile(
    r"(?<!" + _EMAIL_WORD_CHAR + r")" + _EMAIL_WORD_CHAR + r"+@" + _EMAIL_WORD_CHAR + r"+\." + _EMAIL_WORD_CHAR + r"+"
)


def _redact_query_value(match: re.Match) -> str:
    key = match.group(1)
    return match.group(0) if key is None else key + _REDACTED


def redact_error_text(text: str, max_len: int) -> str:
    """Redact credentials and personal data from exception text, then cap it at ``max_len``.

    The shared secret shapes (vendor keys, card numbers) and JSON Web Tokens,
    query-string values, bearer tokens, URL user info, ``key=value`` and
    ``"key": "value"`` pairs under sensitive names, emails and secret-shaped
    words are replaced with "[REDACTED]". Redaction runs before the cap so a
    cut never leaves part of a secret behind.
    """
    redacted = mask_secret_shapes(text, SECRET_SHAPES, lambda kept: _REDACTED)
    redacted = mask_jwts(redacted, _REDACTED)
    redacted = _QUERY_VALUE_PATTERN.sub(_redact_query_value, redacted)
    redacted = _AUTH_SCHEME_PATTERN.sub(r"\1" + _REDACTED, redacted)
    redacted = _URL_USERINFO_PATTERN.sub(r"\1" + _REDACTED + "@", redacted)
    redacted = _SENSITIVE_ASSIGNMENT_PATTERN.sub(r"\1" + _REDACTED, redacted)
    redacted = _EMAIL_WORD_PATTERN.sub(_REDACTED, redacted)
    redacted = _SECRET_SHAPED_WORD_PATTERN.sub(_REDACTED, redacted)
    return redacted[:max_len]


def address_without_credentials(url) -> str:
    """Scheme, host, port and path of an ``httpx.URL``: never its userinfo, query or fragment."""
    port = f":{url.port}" if url.port is not None else ""
    return f"{url.scheme}://{url.host}{port}{url.path}"
