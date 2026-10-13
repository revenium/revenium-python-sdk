"""Breaking out of a stream early must still dispatch metering.

Covers the sync chat StreamWrapper, the async chat wrapper, and the
Responses-API wrapper. Abandoning the iterator (break + GC) previously fired
no metering event at all.
"""
import asyncio
import datetime
import gc
import threading
import time
import weakref
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from revenium_middleware.openai import middleware as mw

from .responses_stream_events import typed_events

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
        wrapper = mw.handle_streaming_responses(iter(typed_events()), NOW, {})
        for i, _ in enumerate(wrapper):
            if i == 1:
                break
        del wrapper
        gc.collect()

        assert mock_run.call_count == 1

    def test_full_consumption_dispatches_exactly_once(self, mock_run):
        wrapper = mw.handle_streaming_responses(iter(typed_events()), NOW, {})
        for _ in wrapper:
            pass
        wrapper.close()
        del wrapper
        gc.collect()

        assert mock_run.call_count == 1

    def test_async_early_break_dispatches_metering_once(self, mock_run):
        async def scenario():
            wrapper = mw._wrap_async_responses_stream(
                TestAsyncChatEarlyTermination.FakeAsyncStream(typed_events()), NOW, {})
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


class FailingAsyncEventStream:
    """An async event stream that raises after ``fail_after`` events and records its close."""

    def __init__(self, events, fail_after=None, close_error=None):
        self._it = iter(events)
        self._remaining = fail_after
        self._close_error = close_error
        self.close_calls = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._remaining == 0:
            raise ConnectionError("stream dropped")
        if self._remaining is not None:
            self._remaining -= 1
        try:
            return next(self._it)
        except StopIteration:
            raise StopAsyncIteration

    async def close(self):
        self.close_calls += 1
        if self._close_error is not None:
            raise self._close_error


@patch("revenium_middleware.openai.middleware.submit_ai_event", return_value=SimpleNamespace(id="evt"))
@patch("revenium_middleware.openai.middleware.current_client", lambda: object())
@patch("revenium_middleware.openai.middleware.run_async_in_thread")
class TestAsyncResponsesStreamCleanup:
    @staticmethod
    def _run_metering(mock_run):
        def run(coro):
            thread = threading.Thread(target=asyncio.run, args=(coro,))
            thread.start()
            thread.join()
            return thread
        mock_run.side_effect = run

    def test_mid_stream_error_meters_once_and_closes_the_stream(self, mock_run, mock_submit):
        self._run_metering(mock_run)
        source = FailingAsyncEventStream(typed_events(), fail_after=2)

        async def scenario():
            wrapper = mw._wrap_async_responses_stream(source, NOW, {})
            async for _ in wrapper:
                pass

        with pytest.raises(ConnectionError):
            asyncio.run(scenario())

        assert source.close_calls == 1
        assert mock_submit.call_count == 1
        assert mock_submit.call_args[0][1]["total_token_count"] == 0

    def test_close_after_an_error_does_not_close_twice(self, mock_run, mock_submit):
        self._run_metering(mock_run)
        source = FailingAsyncEventStream(typed_events(), fail_after=1)

        async def scenario():
            async with mw._wrap_async_responses_stream(source, NOW, {}) as wrapper:
                async for _ in wrapper:
                    pass

        with pytest.raises(ConnectionError):
            asyncio.run(scenario())

        assert source.close_calls == 1
        assert mock_submit.call_count == 1

    def test_failing_close_does_not_escape_async_with(self, mock_run, mock_submit):
        self._run_metering(mock_run)
        source = FailingAsyncEventStream(typed_events(), close_error=RuntimeError("close failed"))

        async def scenario():
            async with mw._wrap_async_responses_stream(source, NOW, {}) as wrapper:
                return [event async for event in wrapper]

        assert len(asyncio.run(scenario())) == len(typed_events())
        assert source.close_calls == 1
        assert mock_submit.call_count == 1
        assert mock_submit.call_args[0][1]["total_token_count"] == 100


LONG_STREAM_CHUNKS = 10_000
LONG_STREAM_USAGE = SimpleNamespace(prompt_tokens=11, completion_tokens=LONG_STREAM_CHUNKS,
                                    total_tokens=11 + LONG_STREAM_CHUNKS)


class _Chunk:
    """A chat chunk that can be weakly referenced, unlike SimpleNamespace."""

    def __init__(self, finish_reason=None, usage=None):
        self.id = "chatcmpl-long"
        self.model = "gpt-4o-mini"
        self.system_fingerprint = None
        self.choices = [SimpleNamespace(delta=SimpleNamespace(content="x"), finish_reason=finish_reason)]
        self.usage = usage


def _long_stream(refs):
    for number in range(LONG_STREAM_CHUNKS):
        last = number == LONG_STREAM_CHUNKS - 1
        chunk = _Chunk("length", LONG_STREAM_USAGE) if last else _Chunk()
        refs.append(weakref.ref(chunk))
        yield chunk


async def _async_long_stream(refs):
    for chunk in _long_stream(refs):
        yield chunk


def _run_metering_now(coro):
    thread = threading.Thread(target=asyncio.run, args=(coro,))
    thread.start()
    thread.join()
    return thread


@pytest.fixture
def sent():
    records = []
    with patch.object(mw, "current_client", lambda: object()), \
         patch.object(mw, "submit_ai_event", side_effect=lambda _kind, args: records.append(args)), \
         patch.object(mw, "run_async_in_thread", side_effect=_run_metering_now):
        yield records


def _consume_sync(refs):
    wrapper = mw.handle_streaming_response(_long_stream(refs), NOW, {}, None, {})
    for _ in wrapper:
        pass
    return wrapper


def _consume_async(refs):
    async def consume():
        wrapper = mw._wrap_async_stream(_async_long_stream(refs), NOW, {})
        async for _ in wrapper:
            pass
        return wrapper
    return asyncio.run(consume())


class TestLongStreamRetention:
    @pytest.mark.parametrize("consume", [_consume_sync, _consume_async], ids=["sync", "async"])
    def test_consumed_chunks_are_released(self, consume, sent):
        refs = []

        wrapper = consume(refs)
        gc.collect()

        assert len(refs) == LONG_STREAM_CHUNKS
        assert sum(ref() is not None for ref in refs) <= 1
        assert wrapper is not None
        assert len(sent) == 1
        assert sent[0]["total_token_count"] == LONG_STREAM_USAGE.total_tokens
        assert sent[0]["output_token_count"] == LONG_STREAM_CHUNKS
        assert sent[0]["stop_reason"] == "TOKEN_LIMIT"


FIRST_TOKEN_WAIT = datetime.timedelta(seconds=1)
RAW_READ_PAUSE_SECONDS = 0.2


def _stream_responses_sync(events):
    for _ in mw.handle_streaming_responses(iter(events), NOW - FIRST_TOKEN_WAIT, {}):
        pass
    return FIRST_TOKEN_WAIT


def _stream_responses_async(events):
    async def consume():
        stream = TestAsyncChatEarlyTermination.FakeAsyncStream(events)
        async for _ in mw._wrap_async_responses_stream(stream, NOW - FIRST_TOKEN_WAIT, {}):
            pass
    asyncio.run(consume())
    return FIRST_TOKEN_WAIT


def _stream_responses_raw(events):
    from .test_openai_sdk_entry_points import FakeOpenAI, RESPONSES_REQUEST

    raw = FakeOpenAI().sync_client().responses.with_raw_response.create(**RESPONSES_REQUEST, stream=True)
    time.sleep(RAW_READ_PAUSE_SECONDS)
    list(raw.parse())
    return datetime.timedelta(seconds=RAW_READ_PAUSE_SECONDS)


class TestResponsesStreamFirstToken:
    @pytest.mark.parametrize("stream_responses", [_stream_responses_sync, _stream_responses_async,
                                                  _stream_responses_raw], ids=["sync", "async", "raw_response"])
    def test_a_streamed_response_records_its_first_token_time(self, sent, stream_responses):
        """The first output text is timed from the request, as on the chat stream (BACK-4413)."""
        waited = stream_responses(typed_events())

        assert len(sent) == 1
        assert waited.total_seconds() * 1000 <= sent[0]["time_to_first_token"] <= sent[0]["request_duration"]

    def test_a_stream_without_output_text_records_zero(self, sent):
        events = [event for event in typed_events() if event.type != "response.output_text.delta"]

        _stream_responses_sync(events)

        assert len(sent) == 1
        assert sent[0]["time_to_first_token"] == 0
