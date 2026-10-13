"""BACK-4292: a ChatOpenAI stream to Perplexity is read on its way through, never changed.

The reader keeps the last charge Perplexity reported, and an early stop
still closes the stream it wraps.
"""
import asyncio
import contextlib
from types import SimpleNamespace

import pytest

from revenium_middleware._core.call_ownership import CallbackMetadata
from revenium_middleware._core.perplexity_cost import AsyncReportedCostReadingStream, read_reported_usage
from revenium_middleware.openai.middleware import _keep_what_perplexity_reported

_client_utils = pytest.importorskip("langchain_openai.chat_models._client_utils")


def _chunk(total_cost):
    return SimpleNamespace(usage=SimpleNamespace(cost={"total_cost": total_cost}))


def test_the_last_charge_on_the_stream_is_kept_even_after_non_zero_interim_charges():
    publication = CallbackMetadata({}, None, "run")
    chunks = [_chunk(0.001), _chunk(0.002), _chunk(0.005005)]
    stream = read_reported_usage(iter(chunks), lambda item: _keep_what_perplexity_reported((publication,), item))
    assert list(stream) == chunks
    assert publication.reported_cost == {"total_cost": 0.005005}


class _OpenAIStyleAsyncStream:
    """Like openai's AsyncStream: it closes with ``close`` and has no ``aclose``."""

    def __init__(self, items):
        self._items = items
        self.closed = False

    async def _iterate(self):
        for item in self._items:
            yield item

    def __aiter__(self):
        return self._iterate()

    async def close(self):
        self.closed = True


def test_a_consumer_stopping_early_under_a_chunk_timeout_closes_the_wrapped_stream():
    wrapped = _OpenAIStyleAsyncStream([_chunk(0), _chunk(0.005005)])
    reader = AsyncReportedCostReadingStream(wrapped, lambda item: None)

    async def take_the_first_chunk():
        chunks = _client_utils._astream_with_chunk_timeout(reader, 5)
        async with contextlib.aclosing(chunks):
            async for chunk in chunks:
                return chunk

    asyncio.run(take_the_first_chunk())
    assert wrapped.closed
