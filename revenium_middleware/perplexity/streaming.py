"""Metered proxies for Perplexity chat streams, native and OpenAI-compatible alike.

Both clients return a Stainless ``Stream`` / ``AsyncStream`` of OpenAI-shaped chunks
with usage on the last one. The proxies keep that object's interface (iteration,
context manager, ``close()``, ``.response``) and meter the call once: when a read
reaches the end, on ``close()``, or when the caller drops the stream part-way.
"""
import contextvars
import logging
import weakref
from types import SimpleNamespace
from typing import Any, Callable

logger = logging.getLogger("revenium_middleware.perplexity")

MeterCallback = Callable[[Any], None]


class StreamMeter:
    """Meters one stream exactly once, from the last chunk that carried usage.

    ``on_complete`` receives a response-shaped summary (``usage`` and one choice
    with the last ``finish_reason``), so non-streamed extraction reads it unchanged.
    """

    def __init__(self, on_complete: MeterCallback):
        self._on_complete = on_complete
        self._usage = None
        self._finish_reason = None
        self._saw_chunk = False
        self._metered = False

    def observe(self, chunk) -> None:
        self._saw_chunk = True
        self._usage = getattr(chunk, 'usage', None) or self._usage
        for choice in getattr(chunk, 'choices', None) or ():
            self._finish_reason = getattr(choice, 'finish_reason', None) or self._finish_reason

    def meter_when_dropped(self, proxy) -> None:
        """Meter once ``proxy`` is collected, for a caller that stops reading without closing it.

        The record is built in the context the stream was created in, not the one that happens to drop it,
        so it keeps the caller's job and trace fields.
        """
        finalizer = weakref.finalize(proxy, self._meter_dropped, contextvars.copy_context())
        # A stream still referenced at exit was not abandoned, and its record could only be refused by the
        # SDK's own exit flush, which may already have run.
        finalizer.atexit = False

    def _meter_dropped(self, creation_context: contextvars.Context) -> None:
        try:
            creation_context.run(self.meter)
        except Exception as e:
            logger.warning("Error metering a dropped Perplexity stream: %s", e)

    def meter(self) -> None:
        if self._metered or not self._saw_chunk:
            return
        self._metered = True
        self._on_complete(SimpleNamespace(
            usage=self._usage,
            choices=[SimpleNamespace(finish_reason=self._finish_reason)],
        ))


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
