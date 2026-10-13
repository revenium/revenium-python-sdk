"""Metering of a raw-response SSE body as the caller reads it, shared by the
OpenAI and Perplexity integrations.

This module lives in `_core` because importing `revenium_middleware.openai`
patches the OpenAI client, which a Perplexity-only install must not do.
"""

import json
import logging
from types import SimpleNamespace

logger = logging.getLogger("revenium_middleware.extension")

# The terminators openai's own SSE decoder ends an event on
# (openai/_streaming.py SSEDecoder._iter_chunks), so the events split here are
# the events the caller's parse() sees.
_EVENT_TERMINATORS = (b"\r\r", b"\n\n", b"\r\n\r\n")
_DONE = "[DONE]"


def _as_attributes(value):
    if isinstance(value, dict):
        return SimpleNamespace(**{key: _as_attributes(item) for key, item in value.items()})
    if isinstance(value, list):
        return [_as_attributes(item) for item in value]
    return value


def _parse_event(raw_event):
    data_lines = []
    for line in raw_event.splitlines():
        if line.startswith(b"data:"):
            value = line[len(b"data:"):]
            data_lines.append(value[1:] if value.startswith(b" ") else value)
    if not data_lines:
        return None
    data = b"\n".join(data_lines).decode("utf-8")
    if data == _DONE:
        return None
    try:
        return _as_attributes(json.loads(data))
    except ValueError:
        return None


class _SSEBodyRewriter:
    """Splits an SSE body into events, hands each parsed event to ``on_event``,
    and leaves out of the bytes the events ``drop_event`` matches.

    With no ``drop_event`` the caller's chunks pass through untouched. With
    one, the body is re-emitted event by event, each event verbatim. Any
    failure while reading an event keeps that event, and the rest of the body
    then passes through untouched and unmetered: metering never changes or
    breaks what the caller reads.
    """

    def __init__(self, on_event, drop_event):
        self._on_event = on_event
        self._drop_event = drop_event
        self._pending = b""
        self._reading = True

    def rewrite(self, chunks):
        for chunk in chunks:
            kept = self._feed(chunk)
            if kept:
                yield kept
        tail = self._flush()
        if tail:
            yield tail

    async def arewrite(self, chunks):
        async for chunk in chunks:
            kept = self._feed(chunk)
            if kept:
                yield kept
        tail = self._flush()
        if tail:
            yield tail

    def _feed(self, chunk):
        if not self._reading:
            return chunk
        kept = []
        lines = chunk.splitlines(keepends=True)
        for index, line in enumerate(lines):
            self._pending += line
            if not self._pending.endswith(_EVENT_TERMINATORS):
                continue
            event, self._pending = self._pending, b""
            if self._keeps(event):
                kept.append(event)
            if not self._reading:
                kept.extend(lines[index + 1:])
                break
        return chunk if self._drop_event is None else b"".join(kept)

    def _flush(self):
        tail, self._pending = self._pending, b""
        kept = not (tail and self._reading) or self._keeps(tail)
        return tail if kept and self._drop_event is not None else b""

    def _keeps(self, raw_event):
        try:
            event = _parse_event(raw_event)
            if event is None:
                return True
            self._on_event(event)
            return self._drop_event is None or not self._drop_event(event)
        except Exception as e:
            logger.warning("Stream event could not be metered; metering stopped for this body: %s", e)
            self._reading = False
            return True


class _BodyTap:
    """Runs ``on_finish`` once for the first read of a body: when that read
    ends, however it ends, or when the caller closes the response mid-read.

    httpx closes the response itself as the read reaches the end of the
    body, before the last bytes reach the rewriter, so a close only finishes
    the read while the caller holds it paused between pieces.
    """

    def __init__(self, on_event, on_finish, on_error=None):
        self._on_event = on_event
        self._on_finish = on_finish
        self._on_error = on_error
        self.started = False
        self.paused = False
        self.finished = False

    def observe(self, event):
        if not self.finished:
            self._on_event(event)

    def finish(self):
        if self.finished:
            return
        self.finished = True
        try:
            self._on_finish()
        except Exception as e:
            logger.warning("Error finalizing raw-response stream metering: %s", e)

    def failed(self, error):
        if self._on_error is None or self.finished:
            return
        try:
            self._on_error(error)
        except Exception as e:
            logger.warning("Error recording a failed raw-response stream: %s", e)

    def closed_by_caller(self):
        if self.started and self.paused:
            self.finish()

    def pieces(self, body):
        self.started = True
        try:
            for piece in body:
                self.paused = True
                yield piece
                self.paused = False
        except Exception as error:
            self.failed(error)
            raise
        finally:
            self.paused = False
            self.finish()

    async def apieces(self, body):
        self.started = True
        try:
            async for piece in body:
                self.paused = True
                yield piece
                self.paused = False
        except Exception as error:
            self.failed(error)
            raise
        finally:
            self.paused = False
            self.finish()


def _in_pieces_of(size, chunks):
    buffer = b""
    for chunk in chunks:
        buffer += chunk
        while len(buffer) >= size:
            yield buffer[:size]
            buffer = buffer[size:]
    if buffer:
        yield buffer


async def _async_in_pieces_of(size, chunks):
    buffer = b""
    async for chunk in chunks:
        buffer += chunk
        while len(buffer) >= size:
            yield buffer[:size]
            buffer = buffer[size:]
    if buffer:
        yield buffer


def tap_sse_body(raw_response, on_event, on_finish, drop_event=None, on_error=None):
    """Meter a sync raw-response stream body as the caller reads it.

    The first read is the one metered; ``on_finish`` runs once, when that
    read ends or the caller closes the response. A later read (parse() after
    read(), which replays the stored body) gets the bytes the first read
    produced. ``on_error`` gets the exception that broke off that read.
    """
    http_response = raw_response.http_response
    read_bytes = http_response.iter_bytes
    close = http_response.close
    tap = _BodyTap(on_event, on_finish, on_error)

    def iter_bytes(chunk_size=None):
        if tap.started:
            yield from read_bytes(chunk_size)
            return
        body = _SSEBodyRewriter(tap.observe, drop_event).rewrite(read_bytes())
        yield from tap.pieces(body if chunk_size is None else _in_pieces_of(chunk_size, body))

    def close_and_finish():
        try:
            close()
        finally:
            tap.closed_by_caller()

    http_response.iter_bytes = iter_bytes
    http_response.close = close_and_finish


def tap_async_sse_body(raw_response, on_event, on_finish, drop_event=None, on_error=None):
    """The async twin of ``tap_sse_body``, on ``aiter_bytes`` and ``aclose``."""
    http_response = raw_response.http_response
    read_bytes = http_response.aiter_bytes
    aclose = http_response.aclose
    tap = _BodyTap(on_event, on_finish, on_error)

    async def aiter_bytes(chunk_size=None):
        if tap.started:
            async for chunk in read_bytes(chunk_size):
                yield chunk
            return
        body = _SSEBodyRewriter(tap.observe, drop_event).arewrite(read_bytes())
        if chunk_size is not None:
            body = _async_in_pieces_of(chunk_size, body)
        async for chunk in tap.apieces(body):
            yield chunk

    async def aclose_and_finish():
        try:
            await aclose()
        finally:
            tap.closed_by_caller()

    http_response.aiter_bytes = aiter_bytes
    http_response.aclose = aclose_and_finish
