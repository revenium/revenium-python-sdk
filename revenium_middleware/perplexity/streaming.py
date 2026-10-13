"""Metered proxies for Perplexity chat streams, native and OpenAI-compatible alike.

Both clients return a Stainless ``Stream`` / ``AsyncStream`` of OpenAI-shaped chunks
with usage on the last one. The proxies keep that object's interface (iteration,
context manager, ``close()``, ``.response``) and meter the call once: when a read
reaches the end, on ``close()``, or when the caller drops the stream part-way.
"""
import datetime
import logging
import weakref
from types import SimpleNamespace
from typing import Any, Callable

from revenium_middleware._core.call_context import CallContext
from revenium_middleware._core.metering import current_client, recorded_under
from revenium_middleware._core.sse_body_tap import tap_async_sse_body, tap_sse_body

from .raw_response import (
    LazyMeteredResponse,
    decoded_body,
    is_body_read,
    is_lazy_response,
    is_raw_response,
    tap_async_whole_body,
    tap_whole_body,
)

logger = logging.getLogger("revenium_middleware.perplexity")

MeterCallback = Callable[[Any], None]
_RESPONSE_EVENTS_WITH_USAGE = frozenset({"response.completed", "response.incomplete", "response.failed"})


class StreamMeter:
    """Meters one stream exactly once, from the last chunk that carried usage.

    ``on_complete`` receives a response-shaped summary, so non-streamed extraction reads it
    unchanged: ``usage`` and one choice with the last ``finish_reason`` for a chat stream, or
    ``usage``, ``status`` and ``incomplete_details`` of the terminal event's response for a
    Responses stream.
    """

    def __init__(self, on_complete: MeterCallback):
        self._on_complete = on_complete
        self._metering_client = current_client()
        self._call_context = CallContext()
        self._usage = None
        self._finish_reason = None
        self._terminal_response = None
        self._saw_chunk = False
        self._metered = False

    def observe(self, chunk) -> None:
        self._saw_chunk = True
        if getattr(chunk, 'type', None) in _RESPONSE_EVENTS_WITH_USAGE:
            self._terminal_response = getattr(chunk, 'response', None)
        self._usage = getattr(chunk, 'usage', None) or self._responses_usage(chunk) or self._usage
        for choice in getattr(chunk, 'choices', None) or ():
            self._finish_reason = getattr(choice, 'finish_reason', None) or self._finish_reason

    @staticmethod
    def _responses_usage(chunk):
        if getattr(chunk, 'type', None) not in _RESPONSE_EVENTS_WITH_USAGE:
            return None
        return getattr(getattr(chunk, 'response', None), 'usage', None)

    def meter_when_dropped(self, proxy) -> None:
        """Meter once ``proxy`` is collected, for a caller that stops reading without closing it.

        The record is built in the context the stream was created in, not the one that happens to drop it,
        so it keeps the caller's job and trace fields.
        """
        finalizer = weakref.finalize(proxy, self._meter_dropped)
        # A stream still referenced at exit was not abandoned, and its record could only be refused by the
        # SDK's own exit flush, which may already have run.
        finalizer.atexit = False

    def _meter_dropped(self) -> None:
        try:
            self.meter()
        except Exception as e:
            logger.warning("Error metering a dropped Perplexity stream: %s", e)

    def meter(self) -> None:
        if self._metered or not self._saw_chunk:
            return
        self._metered = True
        self._call_context.run_recorded_under(self._metering_client, self._on_complete, self._summary())

    def _summary(self):
        if self._terminal_response is not None:
            return SimpleNamespace(
                usage=self._usage,
                status=getattr(self._terminal_response, 'status', None),
                incomplete_details=getattr(self._terminal_response, 'incomplete_details', None),
                choices=[],
            )
        return SimpleNamespace(usage=self._usage, choices=[SimpleNamespace(finish_reason=self._finish_reason)])


class MeteredStream:
    def __init__(self, stream, meter: StreamMeter):
        self._stream = stream
        self._iterator = iter(stream)
        self._meter = meter
        meter.meter_when_dropped(self)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            chunk = next(self._iterator)
        except BaseException:
            self._meter.meter()
            raise
        self._meter.observe(chunk)
        return chunk

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        self._meter.meter()
        try:
            self._stream.close()
        except Exception as e:
            logger.debug("Error closing Perplexity stream: %s", e)

    def __getattr__(self, name):
        return getattr(self._stream, name)


class AsyncMeteredStream:
    def __init__(self, stream, meter: StreamMeter):
        self._stream = stream
        self._iterator = stream.__aiter__()
        self._meter = meter
        meter.meter_when_dropped(self)

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            chunk = await self._iterator.__anext__()
        except BaseException:
            self._meter.meter()
            raise
        self._meter.observe(chunk)
        return chunk

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def close(self):
        self._meter.meter()
        try:
            await self._stream.close()
        except Exception as e:
            logger.debug("Error closing async Perplexity stream: %s", e)

    async def aclose(self):
        await self.close()

    def __getattr__(self, name):
        return getattr(self._stream, name)


def _stream_meter(call) -> StreamMeter:
    with recorded_under(call.metering_client):
        return StreamMeter(call.meter)


def _claim_for_transport(transport: str, response_id) -> None:
    from revenium_middleware import get_client
    from revenium_middleware._core.call_ownership import claim_call_for_transport

    if get_client() is not None:
        claim_call_for_transport(transport, response_id)


def _meter_whole_body(call, raw_response, asynchronous: bool):
    """Meter a non-streamed raw response; return its response id when it is already known."""
    if is_body_read(raw_response):
        response = decoded_body(raw_response.http_response.content)
        call.meter(response)
        return getattr(response, 'id', None)

    def meter_under_the_call_key(body):
        with recorded_under(call.metering_client):
            call.meter(decoded_body(body))

    tap = tap_async_whole_body if asynchronous else tap_whole_body
    tap(raw_response, meter_under_the_call_key)
    return None


def _meter_raw_response(call, raw_response, transport: str, asynchronous: bool) -> None:
    """Meter the SDK's raw response from its body, leaving the body as the SDK left it."""
    try:
        if call.is_streaming:
            meter = _stream_meter(call)
            tap = tap_async_sse_body if asynchronous else tap_sse_body
            tap(raw_response, meter.observe, meter.meter)
            response_id = None
        else:
            response_id = _meter_whole_body(call, raw_response, asynchronous)
    except Exception as e:
        logger.warning("Error setting up Perplexity raw-response metering: %s", e)
        return
    _claim_for_transport(transport, response_id)


def _restart_clock(call):
    def on_start():
        call.request_time_dt = datetime.datetime.now(datetime.timezone.utc)
    return on_start


def _meter_on_enter(call, transport: str):
    def on_enter(raw_response):
        _meter_raw_response(call, raw_response, transport, asynchronous=False)
    return on_enter


def meter_call(call, response, transport: str, stream_class=MeteredStream):
    """Meter ``call``'s response (a stream when it ends), then tell an
    attached LangChain callback that the ``transport`` wrap metered it, once
    its record is on its way (never when metering is off).

    A raw response goes back to the caller as the SDK made it; the async call
    sites say they are async by passing ``AsyncMeteredStream``, since the
    OpenAI client returns the same raw type from sync and async calls."""
    if is_lazy_response(response):
        return LazyMeteredResponse(response, _restart_clock(call), _meter_on_enter(call, transport))
    if is_raw_response(response):
        _meter_raw_response(call, response, transport, asynchronous=stream_class is AsyncMeteredStream)
        return response
    if call.is_streaming:
        response = stream_class(response, _stream_meter(call))
        response_id = None
    else:
        call.meter(response)
        response_id = getattr(response, 'id', None)
    _claim_for_transport(transport, response_id)
    return response
