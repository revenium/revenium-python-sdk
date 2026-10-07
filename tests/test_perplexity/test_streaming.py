"""Stream proxy lifecycle: a failing close never masks the caller's exception (BACK-3607), and a stream
the caller drops part-way is still metered once (BACK-3915)."""
import asyncio
import gc
import threading
from types import SimpleNamespace

import pytest

from revenium_middleware.perplexity.streaming import AsyncMeteredStream, MeteredStream, StreamMeter

CHUNKS = ["first", "second"]


class _CallerError(Exception):
    pass


class _FailingCloseStream:
    def __iter__(self):
        return iter(CHUNKS)

    def close(self):
        raise RuntimeError("connection already reset")


class _AsyncFailingCloseStream:
    def __aiter__(self):
        return self._chunks()

    async def _chunks(self):
        for chunk in CHUNKS:
            yield chunk

    async def close(self):
        raise RuntimeError("connection already reset")


def _meter(records):
    return StreamMeter(records.append)


def test_a_failing_close_on_the_provider_stream_is_swallowed_after_metering():
    records = []
    stream = MeteredStream(_FailingCloseStream(), _meter(records))
    next(stream)

    stream.close()

    assert len(records) == 1


def test_the_callers_exception_survives_a_failing_close_inside_with():
    records = []

    with pytest.raises(_CallerError):
        with MeteredStream(_FailingCloseStream(), _meter(records)) as stream:
            next(stream)
            raise _CallerError

    assert len(records) == 1


def test_a_failing_async_close_on_the_provider_stream_is_swallowed_after_metering():
    records = []

    async def consume():
        stream = AsyncMeteredStream(_AsyncFailingCloseStream(), _meter(records))
        await stream.__anext__()
        await stream.aclose()

    asyncio.run(consume())

    assert len(records) == 1


def test_the_callers_exception_survives_a_failing_close_inside_async_with():
    records = []

    async def consume():
        async with AsyncMeteredStream(_AsyncFailingCloseStream(), _meter(records)) as stream:
            await stream.__anext__()
            raise _CallerError

    with pytest.raises(_CallerError):
        asyncio.run(consume())

    assert len(records) == 1


USAGE = SimpleNamespace(prompt_tokens=11, completion_tokens=7, total_tokens=18)
USAGE_CHUNKS = [
    SimpleNamespace(choices=[SimpleNamespace(finish_reason=None)], usage=None),
    SimpleNamespace(choices=[SimpleNamespace(finish_reason="stop")], usage=None),
    SimpleNamespace(choices=[], usage=USAGE),
]


class _ProviderStream:
    """Like a Stainless stream, its iterator holds the stream, so only the cycle collector frees it."""

    def __init__(self):
        self._iterator = self._chunks()
        self.closed = False

    def _chunks(self):
        yield from USAGE_CHUNKS

    def __iter__(self):
        return self._iterator

    def close(self):
        self.closed = True


class _AsyncProviderStream:
    def __init__(self):
        self._iterator = self._chunks()

    async def _chunks(self):
        for chunk in USAGE_CHUNKS:
            yield chunk

    def __aiter__(self):
        return self._iterator

    async def close(self):
        pass


def _metered_usage(records):
    gc.collect()
    return [record.usage for record in records]


def test_a_stream_dropped_after_its_usage_chunk_is_metered_once():
    records = []
    stream = MeteredStream(_ProviderStream(), _meter(records))
    for _ in USAGE_CHUNKS:
        next(stream)
    assert records == []

    del stream

    assert _metered_usage(records) == [USAGE]


def test_a_stream_left_to_go_out_of_scope_after_its_usage_chunk_is_metered_once():
    records = []

    def read_up_to_the_usage_chunk():
        stream = MeteredStream(_ProviderStream(), _meter(records))
        for chunk in stream:
            if chunk.usage:
                break

    read_up_to_the_usage_chunk()

    assert _metered_usage(records) == [USAGE]


def test_a_stream_dropped_on_another_thread_is_metered_once():
    records = []
    holder = [MeteredStream(_ProviderStream(), _meter(records))]
    for _ in USAGE_CHUNKS:
        next(holder[0])

    dropper = threading.Thread(target=holder.clear)
    dropper.start()
    dropper.join()

    assert _metered_usage(records) == [USAGE]


def test_a_stream_read_to_the_end_and_dropped_is_metered_once():
    records = []
    stream = MeteredStream(_ProviderStream(), _meter(records))
    assert list(stream) == USAGE_CHUNKS

    del stream

    assert _metered_usage(records) == [USAGE]


def test_a_closed_stream_dropped_afterwards_is_metered_once():
    records = []
    stream = MeteredStream(_ProviderStream(), _meter(records))
    for _ in USAGE_CHUNKS:
        next(stream)
    stream.close()

    del stream

    assert _metered_usage(records) == [USAGE]


def test_a_stream_dropped_before_any_chunk_sends_nothing():
    records = []
    stream = MeteredStream(_ProviderStream(), _meter(records))

    del stream

    assert _metered_usage(records) == []


def test_a_failing_meter_on_a_dropped_stream_is_logged_not_raised(caplog):
    def failing_meter(_summary):
        raise RuntimeError("metering client unavailable")

    stream = MeteredStream(_ProviderStream(), StreamMeter(failing_meter))
    next(stream)

    del stream
    gc.collect()

    assert "Error metering a dropped Perplexity stream" in caplog.text


def test_an_async_stream_dropped_after_its_usage_chunk_is_metered_once():
    records = []

    async def read_up_to_the_usage_chunk():
        stream = AsyncMeteredStream(_AsyncProviderStream(), _meter(records))
        async for chunk in stream:
            if chunk.usage:
                break

    asyncio.run(read_up_to_the_usage_chunk())

    assert _metered_usage(records) == [USAGE]


def test_an_async_stream_read_to_the_end_and_closed_is_metered_once():
    records = []

    async def read_and_close():
        stream = AsyncMeteredStream(_AsyncProviderStream(), _meter(records))
        chunks = [chunk async for chunk in stream]
        await stream.close()
        return chunks

    assert asyncio.run(read_and_close()) == USAGE_CHUNKS

    assert _metered_usage(records) == [USAGE]
