"""A failing close on the provider stream never masks the caller's exception (BACK-3607)."""
import asyncio

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
