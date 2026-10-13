"""Metering for the raw-response forms of Perplexity chat calls, native and
OpenAI-compatible alike.

``with_raw_response`` and ``with_streaming_response`` go through the patched
create, but the SDK hands the caller its own response wrapper instead of the
parsed result, and the native sync ``with_streaming_response`` hands it a
context manager that has not made the request yet. The caller gets that object
back unchanged, with its body as the SDK left it. Usage is read from the body:
at once when the SDK already read it, otherwise from the bytes as the caller
reads them, since httpx routes parse(), read() and the iter_* readers through
iter_bytes() / aiter_bytes().
"""

import importlib
import json
import logging
from types import SimpleNamespace

from revenium_middleware._core.sse_body_tap import _as_attributes

logger = logging.getLogger("revenium_middleware.perplexity")


def _load_classes(candidates):
    found = []
    for module_name, class_name in candidates:
        try:
            found.append(getattr(importlib.import_module(module_name), class_name))
        except (ImportError, AttributeError):
            continue
    return tuple(found)


_RAW_RESPONSE_TYPES = _load_classes((
    ("perplexity._response", "BaseAPIResponse"),
    ("openai._response", "BaseAPIResponse"),
    ("openai._legacy_response", "LegacyAPIResponse"),
))
_LAZY_RESPONSE_TYPES = _load_classes((
    ("perplexity._response", "ResponseContextManager"),
))


def is_raw_response(value):
    return bool(_RAW_RESPONSE_TYPES) and isinstance(value, _RAW_RESPONSE_TYPES)


def is_lazy_response(value):
    return bool(_LAZY_RESPONSE_TYPES) and isinstance(value, _LAZY_RESPONSE_TYPES)


def decoded_body(body):
    """The JSON body as a response-shaped object; an empty one when it is not JSON."""
    try:
        return _as_attributes(json.loads(body))
    except ValueError as e:
        logger.debug("Perplexity response body is not JSON, so it carries no usage: %s", e)
        return SimpleNamespace()


def is_body_read(raw_response):
    return getattr(raw_response.http_response, "is_stream_consumed", True)


class _WholeBodyTap:
    """Hands the body to ``on_body`` once, when the caller's first read of it
    reaches the end. A read abandoned part-way hands over nothing."""

    def __init__(self, on_body):
        self._on_body = on_body
        self.started = False

    def finish(self, chunks):
        try:
            self._on_body(b"".join(chunks))
        except Exception as e:
            logger.warning("Error metering a Perplexity raw response: %s", e)

    def pieces(self, body):
        self.started = True
        chunks = []
        for chunk in body:
            chunks.append(chunk)
            yield chunk
        self.finish(chunks)

    async def apieces(self, body):
        self.started = True
        chunks = []
        async for chunk in body:
            chunks.append(chunk)
            yield chunk
        self.finish(chunks)


def tap_whole_body(raw_response, on_body):
    http_response = raw_response.http_response
    read_bytes = http_response.iter_bytes
    tap = _WholeBodyTap(on_body)

    def iter_bytes(chunk_size=None):
        if tap.started:
            return read_bytes(chunk_size)
        return tap.pieces(read_bytes(chunk_size))

    http_response.iter_bytes = iter_bytes


def tap_async_whole_body(raw_response, on_body):
    http_response = raw_response.http_response
    read_bytes = http_response.aiter_bytes
    tap = _WholeBodyTap(on_body)

    def aiter_bytes(chunk_size=None):
        if tap.started:
            return read_bytes(chunk_size)
        return tap.apieces(read_bytes(chunk_size))

    http_response.aiter_bytes = aiter_bytes


class LazyMeteredResponse:
    """Stands in for the SDK's not-yet-entered response context manager, and
    sets metering up on the response it yields when the caller enters it."""

    def __init__(self, manager, on_start, on_enter):
        self._manager = manager
        self._on_start = on_start
        self._on_enter = on_enter

    def __enter__(self):
        self._on_start()
        raw_response = self._manager.__enter__()
        self._on_enter(raw_response)
        return raw_response

    def __exit__(self, exc_type, exc_val, exc_tb):
        return self._manager.__exit__(exc_type, exc_val, exc_tb)

    def __getattr__(self, name):
        return getattr(self._manager, name)
