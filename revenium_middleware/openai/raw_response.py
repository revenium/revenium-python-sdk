"""Metering for the raw-response forms of OpenAI streaming.

chat.completions.with_raw_response.create(stream=True) and
with_streaming_response.create(stream=True), and their Responses API
counterparts, go through the patched create, but the SDK hands the caller its
own APIResponse wrapper instead of a Stream. The caller must get that wrapper
back unchanged, with the body still unread, so the body is metered from its
bytes as the caller reads them: through parse(), read(), iter_bytes(),
iter_text() or iter_lines() alike, since httpx routes all of them through
iter_bytes(). iter_raw() bypasses it and is not metered.
"""

import importlib

from revenium_middleware._core.sse_body_tap import tap_async_sse_body, tap_sse_body  # noqa: F401


def _load_classes(candidates):
    found = []
    for module_name, class_name in candidates:
        try:
            found.append(getattr(importlib.import_module(module_name), class_name))
        except (ImportError, AttributeError):
            continue
    return tuple(found)


_RAW_RESPONSE_TYPES = _load_classes((
    ("openai._response", "BaseAPIResponse"),
    ("openai._legacy_response", "LegacyAPIResponse"),
))
_PARSED_TYPES = _load_classes((
    ("openai", "Stream"),
    ("openai", "AsyncStream"),
))


def _looks_like_sdk_raw_response(value):
    # Restricted to classes the openai package defines so test doubles such
    # as MagicMock, which answer every hasattr, keep taking the Stream path.
    return (
        type(value).__module__.split(".")[0] == "openai"
        and callable(getattr(value, "parse", None))
        and hasattr(value, "http_response")
        and not isinstance(value, _PARSED_TYPES)
    )


def is_raw_response(value):
    if _RAW_RESPONSE_TYPES and isinstance(value, _RAW_RESPONSE_TYPES):
        return True
    return _looks_like_sdk_raw_response(value)
