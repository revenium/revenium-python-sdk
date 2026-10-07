"""Metering for the SDK's raw-response call forms.

messages.with_raw_response.create and messages.with_streaming_response.create,
and their parse() and beta.messages counterparts, go through the patched
methods, but the SDK hands the caller its own APIResponse wrapper instead of a
Message or a Stream. The caller must get that wrapper back unchanged, body
included:

- with_raw_response has already read the body, so usage comes from parse(),
  which the SDK caches, so the caller later receives the same Message.
- with_streaming_response leaves the body unread for the caller to consume or
  abandon, and stream=True bodies are read incrementally, so those are metered
  from the bytes as the caller reads them, through parse() or iter_bytes() alike.
"""

import importlib
import inspect
import json
import logging
from types import SimpleNamespace

logger = logging.getLogger("revenium_middleware.anthropic")


def _load_classes(candidates):
    found = []
    for module_name, class_name in candidates:
        try:
            found.append(getattr(importlib.import_module(module_name), class_name))
        except (ImportError, AttributeError):
            continue
    return tuple(found)


_RAW_RESPONSE_TYPES = _load_classes((
    ("anthropic._response", "BaseAPIResponse"),
    ("anthropic._legacy_response", "LegacyAPIResponse"),
))
_PARSED_TYPES = _load_classes((
    ("anthropic.types", "Message"),
    ("anthropic", "Stream"),
    ("anthropic", "AsyncStream"),
))


def _looks_like_sdk_raw_response(value):
    # Restricted to classes the anthropic package defines so test doubles such
    # as MagicMock, which answer every hasattr, keep taking the Message path.
    return (
        type(value).__module__.split(".")[0] == "anthropic"
        and callable(getattr(value, "parse", None))
        and hasattr(value, "http_response")
        and not isinstance(value, _PARSED_TYPES)
    )


def is_raw_response(value):
    if _RAW_RESPONSE_TYPES and isinstance(value, _RAW_RESPONSE_TYPES):
        return True
    return _looks_like_sdk_raw_response(value)


def is_body_read(raw_response):
    return getattr(raw_response.http_response, "is_stream_consumed", True)


def parsed_message(raw_response):
    try:
        return raw_response.parse()
    except Exception as e:
        logger.warning("Could not parse Anthropic raw response for metering: %s", e)
        return None


async def async_parsed_message(raw_response):
    try:
        parsed = raw_response.parse()
        if inspect.isawaitable(parsed):
            parsed = await parsed
        return parsed
    except Exception as e:
        logger.warning("Could not parse Anthropic raw response for metering: %s", e)
        return None


def _as_attributes(value):
    if isinstance(value, dict):
        return SimpleNamespace(**{key: _as_attributes(item) for key, item in value.items()})
    if isinstance(value, list):
        return [_as_attributes(item) for item in value]
    return value


class _SSEUsageDecoder:
    """Feeds the events of an SSE body into a StreamUsageState."""

    def __init__(self, state, finalize):
        self._state = state
        self._finalize_cb = finalize
        self._pending = b""
        self._data_lines = []

    def feed(self, chunk):
        *lines, self._pending = (self._pending + chunk).split(b"\n")
        for line in lines:
            self._read_line(line.rstrip(b"\r"))

    def finish(self):
        self._read_line(self._pending.rstrip(b"\r"))
        self._read_line(b"")
        self._finalize_cb(self._state)

    def _read_line(self, line):
        if not line:
            self._dispatch()
        elif line.startswith(b"data:"):
            self._data_lines.append(line[len(b"data:"):].lstrip(b" "))

    def _dispatch(self):
        if not self._data_lines:
            return
        data, self._data_lines = b"\n".join(self._data_lines), []
        try:
            event = json.loads(data)
        except ValueError:
            return
        self._state.ingest(_as_attributes(event))


class _MessageBodyDecoder:
    """Rebuilds the message from a JSON body once the caller has read all of it;
    a body abandoned part-way is not metered."""

    def __init__(self, on_message, message_type):
        self._on_message = on_message
        self._message_type = message_type
        self._chunks = []

    def feed(self, chunk):
        self._chunks.append(chunk)

    def finish(self):
        try:
            message = self._message_type.model_validate(json.loads(b"".join(self._chunks)))
        except Exception as e:
            logger.debug("Anthropic response body was not a complete message; not metered: %s", e)
            return
        self._on_message(message)


class _BodyTap:
    def __init__(self, decoder):
        self._decoder = decoder
        self._finished = False

    def feed(self, chunk):
        if not self._finished:
            self._decoder.feed(chunk)

    def finish(self):
        if self._finished:
            return
        self._finished = True
        try:
            self._decoder.finish()
        except Exception as e:
            logger.warning("Error finalizing raw-response metering: %s", e)


def _tap_body(raw_response, decoder):
    http_response = raw_response.http_response
    read_bytes = http_response.iter_bytes
    tap = _BodyTap(decoder)

    def iter_bytes(*args, **kwargs):
        try:
            for chunk in read_bytes(*args, **kwargs):
                tap.feed(chunk)
                yield chunk
        finally:
            tap.finish()

    http_response.iter_bytes = iter_bytes


def _tap_async_body(raw_response, decoder):
    http_response = raw_response.http_response
    read_bytes = http_response.aiter_bytes
    tap = _BodyTap(decoder)

    async def aiter_bytes(*args, **kwargs):
        try:
            async for chunk in read_bytes(*args, **kwargs):
                tap.feed(chunk)
                yield chunk
        finally:
            tap.finish()

    http_response.aiter_bytes = aiter_bytes


def tap_sse_body(raw_response, state, finalize):
    _tap_body(raw_response, _SSEUsageDecoder(state, finalize))


def tap_async_sse_body(raw_response, state, finalize):
    _tap_async_body(raw_response, _SSEUsageDecoder(state, finalize))


def tap_message_body(raw_response, on_message, message_type):
    _tap_body(raw_response, _MessageBodyDecoder(on_message, message_type))


def tap_async_message_body(raw_response, on_message, message_type):
    _tap_async_body(raw_response, _MessageBodyDecoder(on_message, message_type))
