"""LiteLLM client metering: async calls, streams that did not request usage, and exactly-once."""
import asyncio
import gc
import logging
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("litellm")

import litellm  # noqa: E402
from litellm.litellm_core_utils.streaming_handler import CustomStreamWrapper  # noqa: E402

from revenium_middleware.litellm.client import middleware as mw  # noqa: E402

MODEL = "gpt-4o-mini"
PROVIDER_WITHOUT_STREAM_OPTIONS = "anthropic/claude-sonnet-4-5"
EMBEDDING_MODEL = "text-embedding-3-small"
MESSAGES = [{"role": "user", "content": "hi there"}]


def completion_kwargs(**extra):
    return {"model": MODEL, "messages": MESSAGES, "mock_response": "hello world", **extra}


def embedding_kwargs():
    return {"model": EMBEDDING_MODEL, "input": ["hi there"], "mock_response": [0.1, 0.2]}


def run_in_own_thread(coro):
    thread = threading.Thread(target=asyncio.run, args=(coro,))
    thread.start()
    thread.join()
    return thread


@pytest.fixture
def payloads():
    recorded = []
    with patch.object(mw, "run_async_in_thread", side_effect=run_in_own_thread), \
            patch.object(mw, "submit_ai_event", side_effect=lambda op, args: recorded.append(args)):
        yield recorded


@pytest.fixture
def warned_once_reset(monkeypatch):
    monkeypatch.setattr(mw, "_stream_without_usage_warned", False)


def assert_one_metered_call(payloads, operation_type="CHAT", streamed=False):
    assert len(payloads) == 1, payloads
    payload = payloads[0]
    assert payload["operation_type"] == operation_type
    assert payload["is_streamed"] is streamed
    assert payload["input_token_count"] > 0
    if operation_type == "CHAT":
        assert payload["output_token_count"] > 0
    return payload


def error_records(caplog):
    return [r for r in caplog.records if r.levelno >= logging.ERROR]


def warnings_about_missing_usage(caplog):
    return [r for r in caplog.records if r.levelno == logging.WARNING and "without reporting usage" in r.message]


async def drain(stream):
    return [chunk async for chunk in stream]


class TestAsyncCompletion:
    def test_non_streamed_acompletion_meters_once(self, payloads):
        asyncio.run(litellm.acompletion(**completion_kwargs()))

        assert_one_metered_call(payloads)

    def test_streamed_acompletion_with_usage_meters_once_and_keeps_the_usage_chunk(self, payloads):
        async def call():
            stream = await litellm.acompletion(**completion_kwargs(stream=True, stream_options={"include_usage": True}))
            return await drain(stream)

        chunks = asyncio.run(call())

        assert_one_metered_call(payloads, streamed=True)
        assert getattr(chunks[-1], "usage", None) is not None

    def test_streamed_acompletion_without_usage_meters_once_and_hides_the_usage_chunk(self, payloads, caplog):
        async def call():
            stream = await litellm.acompletion(**completion_kwargs(stream=True))
            return await drain(stream)

        with caplog.at_level(logging.DEBUG):
            chunks = asyncio.run(call())

        assert_one_metered_call(payloads, streamed=True)
        assert all(getattr(chunk, "usage", None) is None for chunk in chunks)
        assert error_records(caplog) == []

    def test_async_stream_is_still_a_litellm_stream(self, payloads):
        async def call():
            stream = await litellm.acompletion(**completion_kwargs(stream=True))
            is_litellm_stream = isinstance(stream, CustomStreamWrapper)
            await drain(stream)
            return is_litellm_stream

        assert asyncio.run(call()) is True

    def test_async_stream_closed_early_meters_once(self, payloads):
        async def call():
            stream = await litellm.acompletion(**completion_kwargs(stream=True))
            await stream.__anext__()
            await stream.aclose()

        asyncio.run(call())
        gc.collect()

        assert len(payloads) == 1
        assert payloads[0]["is_streamed"] is True

    def test_atext_completion_and_acompletion_with_retries_meter_once_each(self, payloads):
        asyncio.run(litellm.atext_completion(model=MODEL, prompt="hi there", mock_response="hello world"))
        assert_one_metered_call(payloads)
        payloads.clear()

        asyncio.run(litellm.acompletion_with_retries(**completion_kwargs()))
        assert_one_metered_call(payloads)


class TestStreamWithoutRequestedUsage:
    def test_sync_stream_meters_once_from_the_requested_usage_without_error_logs(self, payloads, caplog):
        with caplog.at_level(logging.DEBUG):
            chunks = list(litellm.completion(**completion_kwargs(stream=True)))

        assert_one_metered_call(payloads, streamed=True)
        assert all(getattr(chunk, "usage", None) is None for chunk in chunks)
        assert error_records(caplog) == []

    def test_caller_sees_the_same_chunks_as_without_the_middleware(self, payloads):
        chunks = list(litellm.completion(**completion_kwargs(stream=True)))

        texts = [chunk.choices[0].delta.content for chunk in chunks]
        assert texts == ["hel", "lo ", "wor", "ld", None]
        assert chunks[-1].choices[0].finish_reason == "stop"

    def test_usage_is_requested_and_the_callers_own_stream_options_are_kept(self, payloads):
        wrapped = MagicMock(return_value=iter([]))

        mw.completion_wrapper(wrapped, None, (), {"model": MODEL, "stream": True, "stream_options": {"x": 1}})

        assert wrapped.call_args.kwargs["stream_options"] == {"x": 1, "include_usage": True}

    def test_non_streamed_call_is_sent_unchanged(self, payloads):
        wrapped = MagicMock(return_value=SimpleNamespace(id="x", usage=None))

        mw.completion_wrapper(wrapped, None, (), {"model": MODEL})

        assert "stream_options" not in wrapped.call_args.kwargs

    def test_stream_that_ends_without_usage_emits_nothing_and_warns_once(self, payloads, caplog, warned_once_reset):
        chunk = SimpleNamespace(id="x", choices=[])

        with caplog.at_level(logging.DEBUG):
            for _ in range(2):
                wrapped = MagicMock(return_value=iter([chunk, chunk]))
                list(mw.completion_wrapper(wrapped, None, (), {"model": MODEL, "stream": True}))

        assert payloads == []
        assert error_records(caplog) == []
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING and "without reporting usage" in r.message]
        assert len(warnings) == 1

    def test_metering_a_response_without_usage_never_raises(self, payloads):
        mw.handle_response(SimpleNamespace(id="x", model=MODEL), mw.datetime.datetime.now(mw.datetime.timezone.utc),
                           {}, True)

        assert payloads[0]["input_token_count"] == 0


class TestExactlyOnce:
    def test_completion_meters_once(self, payloads):
        litellm.completion(**completion_kwargs())

        assert_one_metered_call(payloads)

    def test_sync_stream_with_usage_meters_once(self, payloads):
        list(litellm.completion(**completion_kwargs(stream=True, stream_options={"include_usage": True})))

        assert_one_metered_call(payloads, streamed=True)

    def test_batch_completion_meters_once(self, payloads):
        litellm.batch_completion(model=MODEL, messages=[MESSAGES], mock_response="hello world")

        assert_one_metered_call(payloads)

    def test_text_completion_meters_once(self, payloads):
        litellm.text_completion(model=MODEL, prompt="hi there", mock_response="hello world")

        assert_one_metered_call(payloads)

    def test_completion_with_retries_meters_once(self, payloads):
        litellm.completion_with_retries(**completion_kwargs())

        assert_one_metered_call(payloads)

    def test_completion_with_retries_over_the_wrapped_completion_meters_once(self, payloads):
        litellm.completion_with_retries(**completion_kwargs(), original_function=litellm.completion)

        assert_one_metered_call(payloads)

    def test_responses_is_not_metered(self, payloads):
        litellm.responses(model=MODEL, input="hi there", mock_response="hello world")

        assert payloads == []


class TestEmbedding:
    def test_embedding_meters_once_as_embed(self, payloads):
        litellm.embedding(**embedding_kwargs())

        payload = assert_one_metered_call(payloads, operation_type="EMBED")
        assert payload["model"] == EMBEDDING_MODEL
        assert payload["output_token_count"] == 0

    def test_aembedding_meters_once_as_embed(self, payloads):
        asyncio.run(litellm.aembedding(**embedding_kwargs()))

        assert_one_metered_call(payloads, operation_type="EMBED")


class TestStreamUsageIsRequestedForEveryProvider:
    def test_an_anthropic_stream_gets_the_option(self, payloads):
        wrapped = MagicMock(return_value=iter([]))

        mw.completion_wrapper(wrapped, None, (), {"model": PROVIDER_WITHOUT_STREAM_OPTIONS, "stream": True})

        assert wrapped.call_args.kwargs["stream_options"] == {"include_usage": True}

    def test_an_anthropic_stream_meters_from_the_appended_usage_chunk(self, payloads, caplog):
        with caplog.at_level(logging.DEBUG):
            chunks = list(litellm.completion(
                model=PROVIDER_WITHOUT_STREAM_OPTIONS, messages=MESSAGES, mock_response="hello world", stream=True))

        assert [chunk.choices[0].delta.content for chunk in chunks][:4] == ["hel", "lo ", "wor", "ld"]
        assert all(getattr(chunk, "usage", None) is None for chunk in chunks)
        assert_one_metered_call(payloads, streamed=True)
        assert error_records(caplog) == []

    def test_a_stream_without_a_usage_chunk_is_metered_from_the_usage_litellm_attaches(self, payloads):
        final = SimpleNamespace(id="msg_1", model="claude-sonnet-4-5", choices=[],
                                _hidden_params={"usage": SimpleNamespace(prompt_tokens=11, completion_tokens=7)})
        wrapped = MagicMock(return_value=iter([SimpleNamespace(id="msg_1", choices=[], _hidden_params={}), final]))

        list(mw.completion_wrapper(wrapped, None, (), {"model": PROVIDER_WITHOUT_STREAM_OPTIONS, "stream": True}))

        assert len(payloads) == 1
        assert (payloads[0]["input_token_count"], payloads[0]["output_token_count"]) == (11, 7)

    def test_all_zero_attached_usage_counts_as_none(self, payloads, caplog, warned_once_reset):
        final = SimpleNamespace(id="msg_1", choices=[],
                                _hidden_params={"usage": SimpleNamespace(prompt_tokens=0, completion_tokens=0)})
        wrapped = MagicMock(return_value=iter([final]))

        with caplog.at_level(logging.DEBUG):
            list(mw.completion_wrapper(wrapped, None, (), {"model": MODEL, "stream": True}))

        assert payloads == []
        assert len(warnings_about_missing_usage(caplog)) == 1


class TestInjectedUsageChunkHiding:
    NOW = mw.datetime.datetime.now(mw.datetime.timezone.utc)
    USAGE = SimpleNamespace(prompt_tokens=11, completion_tokens=7)

    def stream(self, chunks):
        return list(mw.handle_streaming_response(iter(chunks), self.NOW, {}, caller_requested_usage=False))

    def test_a_usage_only_chunk_is_hidden(self, payloads):
        usage_only = SimpleNamespace(id="c1", choices=[SimpleNamespace(delta=SimpleNamespace(content=None),
                                                                       finish_reason=None)], usage=self.USAGE)

        assert self.stream([usage_only]) == []
        assert len(payloads) == 1

    def test_a_chunk_with_usage_and_content_reaches_the_caller_with_its_usage(self, payloads):
        both = SimpleNamespace(id="c1", choices=[SimpleNamespace(delta=SimpleNamespace(content="hello"),
                                                                 finish_reason=None)], usage=self.USAGE)

        received = self.stream([both])

        assert received == [both]
        assert received[0].usage is self.USAGE
        assert (payloads[0]["input_token_count"], payloads[0]["output_token_count"]) == (11, 7)

    def test_a_chunk_with_usage_and_a_finish_reason_reaches_the_caller(self, payloads):
        final = SimpleNamespace(id="c1", choices=[{"delta": {}, "finish_reason": "stop"}], usage=self.USAGE)

        assert self.stream([final]) == [final]


class TestFailingUnderlyingClose:
    NOW = mw.datetime.datetime.now(mw.datetime.timezone.utc)

    class SyncStream:
        def __init__(self):
            self.chunks = iter([SimpleNamespace(id="c1", choices=[])])

        def __next__(self):
            return next(self.chunks)

        def close(self):
            raise RuntimeError("transport already gone")

    class AsyncStream:
        def __init__(self):
            self.chunks = iter([SimpleNamespace(id="c1", choices=[])])

        async def __anext__(self):
            try:
                return next(self.chunks)
            except StopIteration:
                raise StopAsyncIteration

        async def aclose(self):
            raise RuntimeError("transport already gone")

    def test_close_does_not_raise_and_still_meters_once(self, payloads):
        stream = mw.handle_streaming_response(self.SyncStream(), self.NOW, {})
        next(stream)

        stream.close()

        assert len(payloads) == 1

    def test_aclose_does_not_raise_and_still_meters_once(self, payloads):
        async def call():
            stream = mw.handle_streaming_response(self.AsyncStream(), self.NOW, {})
            await stream.__anext__()
            await stream.aclose()

        asyncio.run(call())

        assert len(payloads) == 1
