"""Breaking out of a stream early must still dispatch metering.

Covers the sync chat StreamWrapper, the async chat wrapper, and the
Responses-API wrapper. Abandoning the iterator (break + GC) previously fired
no metering event at all.
"""
import asyncio
import datetime
import gc
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from revenium_middleware.openai import middleware as mw

NOW = datetime.datetime.now(datetime.timezone.utc)


def drain_coroutine(coro):
    coro.close()
    return MagicMock()


def make_chunk():
    chunk = MagicMock()
    chunk.usage.prompt_tokens = 3
    chunk.usage.completion_tokens = 5
    chunk.usage.total_tokens = 8
    return chunk


def make_usageless_chunk(i):
    return SimpleNamespace(
        id="chatcmpl-test",
        model="gpt-4o-mini",
        choices=[SimpleNamespace(delta=SimpleNamespace(content=f"tok{i}"), finish_reason=None)],
        system_fingerprint=None,
    )


@patch("revenium_middleware.openai.middleware.run_async_in_thread", side_effect=drain_coroutine)
class TestSyncChatEarlyTermination:
    def test_early_break_dispatches_metering_once(self, mock_run):
        wrapper = mw.handle_streaming_response(iter([make_chunk() for _ in range(5)]), NOW, {}, None, {})
        for i, _ in enumerate(wrapper):
            if i == 1:
                break
        del wrapper
        gc.collect()

        assert mock_run.call_count == 1

    def test_early_break_without_usage_chunk_still_dispatches(self, mock_run):
        chunks = [make_usageless_chunk(i) for i in range(5)]
        wrapper = mw.handle_streaming_response(iter(chunks), NOW, {}, None, {})
        for i, _ in enumerate(wrapper):
            if i == 1:
                break
        del wrapper
        gc.collect()

        assert mock_run.call_count == 1

    def test_full_consumption_dispatches_exactly_once(self, mock_run):
        wrapper = mw.handle_streaming_response(iter([make_chunk() for _ in range(3)]), NOW, {}, None, {})
        for _ in wrapper:
            pass
        del wrapper
        gc.collect()

        assert mock_run.call_count == 1


@patch("revenium_middleware.openai.middleware.run_async_in_thread", side_effect=drain_coroutine)
class TestAsyncChatEarlyTermination:
    class FakeAsyncStream:
        def __init__(self, chunks):
            self._it = iter(chunks)

        def __aiter__(self):
            return self

        async def __anext__(self):
            try:
                return next(self._it)
            except StopIteration:
                raise StopAsyncIteration

    def test_early_break_dispatches_metering_once(self, mock_run):
        async def scenario():
            wrapper = mw._wrap_async_stream(
                self.FakeAsyncStream([make_chunk() for _ in range(5)]), NOW, {})
            count = 0
            async for _ in wrapper:
                count += 1
                if count == 2:
                    break
            return wrapper

        wrapper = asyncio.run(scenario())
        del wrapper
        gc.collect()

        assert mock_run.call_count == 1

    def test_early_break_without_usage_chunk_still_dispatches(self, mock_run):
        async def scenario():
            wrapper = mw._wrap_async_stream(
                self.FakeAsyncStream([make_usageless_chunk(i) for i in range(5)]), NOW, {})
            count = 0
            async for _ in wrapper:
                count += 1
                if count == 2:
                    break
            return wrapper

        wrapper = asyncio.run(scenario())
        del wrapper
        gc.collect()

        assert mock_run.call_count == 1


@patch("revenium_middleware.openai.middleware.run_async_in_thread", side_effect=drain_coroutine)
class TestResponsesEarlyTermination:
    def test_early_break_dispatches_metering_once(self, mock_run):
        wrapper = mw.handle_streaming_responses(iter([make_chunk() for _ in range(5)]), NOW, {})
        for i, _ in enumerate(wrapper):
            if i == 1:
                break
        del wrapper
        gc.collect()

        assert mock_run.call_count == 1
