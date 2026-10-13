"""LiteLLM client metering: async calls, streams that did not request usage, and exactly-once."""
import asyncio
import contextvars
import gc
import logging
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("litellm")

import httpx  # noqa: E402
import litellm  # noqa: E402
import openai  # noqa: E402
from litellm.litellm_core_utils.streaming_handler import CustomStreamWrapper  # noqa: E402

import revenium_middleware.openai  # noqa: E402,F401
import revenium_middleware.perplexity  # noqa: E402,F401
from revenium_middleware._core import metering_pool  # noqa: E402
from revenium_middleware._core.patch_registry import is_patched  # noqa: E402
from revenium_middleware.litellm.client import middleware as mw  # noqa: E402

from . import fake_openai_upstream  # noqa: E402

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
            patch.object(mw, "submit_ai_event", side_effect=lambda op, args, **kwargs: recorded.append(args)):
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


OPENAI_SDK_MODEL = "openai/gpt-4.1-nano"
ORGANIZATION = "AcmeCorp"
# Long enough for the metering pool on a loaded CI runner; a passing run
# returns as soon as the expected rows land.
ROW_TIMEOUT_SECONDS = 15.0
# A second row from another wrapper can trail the one the test waits for.
TRAILING_ROW_GRACE_SECONDS = 0.5


@pytest.fixture
def openai_compatible_upstream():
    server = fake_openai_upstream.start_upstream()
    yield fake_openai_upstream.api_base(server)
    server.shutdown()
    server.server_close()


def openai_sdk_completion_kwargs(api_base, model=OPENAI_SDK_MODEL, **extra):
    """A LiteLLM call that LiteLLM serves through the OpenAI SDK, which the OpenAI integration patches."""
    return {
        "model": model,
        "messages": MESSAGES,
        "api_base": api_base,
        "api_key": "sk-fake",
        "usage_metadata": {"organization_name": ORGANIZATION},
        **extra,
    }


def sync_openai_sdk_completion(api_base, **extra):
    return litellm.completion(**openai_sdk_completion_kwargs(api_base, **extra))


def async_openai_sdk_completion(api_base, **extra):
    return asyncio.run(litellm.acompletion(**openai_sdk_completion_kwargs(api_base, **extra)))


def rows_once_settled(mock_revenium_client, expected_count=1):
    def rows():
        return [call.kwargs for call in mock_revenium_client.ai.create_completion.call_args_list]

    deadline = time.monotonic() + ROW_TIMEOUT_SECONDS
    while time.monotonic() < deadline and len(rows()) < expected_count:
        metering_pool.wait_until_idle(0.5)
        time.sleep(0.05)
    time.sleep(TRAILING_ROW_GRACE_SECONDS)
    metering_pool.wait_until_idle(ROW_TIMEOUT_SECONDS)
    return rows()


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

    @pytest.mark.parametrize("call", [sync_openai_sdk_completion, async_openai_sdk_completion], ids=["sync", "async"])
    def test_a_call_served_through_the_openai_sdk_is_recorded_once_with_the_callers_metadata(
        self, call, openai_compatible_upstream, mock_revenium_client
    ):
        call(api_base=openai_compatible_upstream)

        rows = rows_once_settled(mock_revenium_client)
        assert [(row["provider"], row.get("organization_name")) for row in rows] == [("LITELLM", ORGANIZATION)]

    def test_a_stream_served_through_the_openai_sdk_reaches_the_caller_and_is_recorded_once(
        self, openai_compatible_upstream, mock_revenium_client
    ):
        stream = sync_openai_sdk_completion(api_base=openai_compatible_upstream, stream=True)

        assert "".join(chunk.choices[0].delta.content or "" for chunk in stream if chunk.choices) == "hi"
        rows = rows_once_settled(mock_revenium_client)
        assert [(row["provider"], row.get("organization_name")) for row in rows] == [("LITELLM", ORGANIZATION)]

    def test_a_perplexity_call_served_through_the_openai_sdk_is_recorded_once_with_the_callers_metadata(
        self, openai_compatible_upstream, mock_revenium_client
    ):
        sync_openai_sdk_completion(model="perplexity/sonar", api_base=openai_compatible_upstream + "/perplexity")

        rows = rows_once_settled(mock_revenium_client)
        assert [(row["provider"], row.get("organization_name")) for row in rows] == [("LITELLM", ORGANIZATION)]

    @pytest.mark.parametrize("model, upstream_path, patch_path", [
        pytest.param(OPENAI_SDK_MODEL, "", "openai.resources.chat.completions.Completions.create", id="openai"),
        pytest.param("perplexity/sonar", "/perplexity",
                     "perplexity:openai.resources.chat.completions.Completions.create", id="perplexity"),
    ])
    @pytest.mark.parametrize("stream", [False, True], ids=["plain", "streamed"])
    @pytest.mark.parametrize("entry_point", ["completion", "acompletion"])
    def test_a_proxied_call_served_through_the_openai_sdk_leaves_no_openai_row(
        self, entry_point, stream, model, upstream_path, patch_path, openai_compatible_upstream,
        mock_revenium_client, monkeypatch
    ):
        """The gateway meters a proxied call itself, so the OpenAI and Perplexity
        integrations patched under LiteLLM's OpenAI SDK call stand aside (BACK-4249)."""
        monkeypatch.setattr(mw, "proxy_meters_request", lambda kwargs: True)
        served = []
        serve = fake_openai_upstream._Handler.do_POST

        def counted(handler):
            served.append(handler.path)
            serve(handler)

        monkeypatch.setattr(fake_openai_upstream._Handler, "do_POST", counted)
        kwargs = openai_sdk_completion_kwargs(
            openai_compatible_upstream + upstream_path, model=model, stream=stream,
            proxy_server_request={"url": "http://proxy/v1/chat/completions", "body": {}}, metadata={})

        if entry_point == "completion":
            response = litellm.completion(**kwargs)
            if stream:
                list(response)
        else:
            async def call():
                response = await litellm.acompletion(**kwargs)
                if stream:
                    await drain(response)
            asyncio.run(call())

        assert is_patched(patch_path)
        assert len(served) == 1
        assert rows_once_settled(mock_revenium_client, expected_count=0) == []

    def test_an_openai_call_from_a_context_copied_during_a_litellm_call_is_metered_after_it_returns(
        self, openai_compatible_upstream, mock_revenium_client
    ):
        """An async LiteLLM success callback runs in a task that copies the
        call's context, so a call it makes after the LiteLLM call returned is
        its own and must be metered."""
        copied_contexts = []

        def copying_entry_point(*args, **kwargs):
            copied_contexts.append(contextvars.copy_context())
            return litellm.completion.__wrapped__(*args, **kwargs)

        mw.completion_wrapper(copying_entry_point, None, (), completion_kwargs())
        openai_client = openai.OpenAI(base_url=openai_compatible_upstream, api_key="sk-fake")
        copied_contexts[0].run(openai_client.chat.completions.create, model="gpt-4.1-nano", messages=MESSAGES)

        rows = rows_once_settled(mock_revenium_client, expected_count=2)
        assert sorted(row["provider"] for row in rows) == ["LITELLM", "OPENAI"]


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


class TestSelectivePassThrough:
    """Selective metering on, outside @revenium_meter (BACK-3991)."""

    WRAPPERS = {
        "completion": (mw.completion_wrapper, False),
        "acompletion": (mw.acompletion_wrapper, True),
        "embedding": (mw.embedding_wrapper, False),
        "aembedding": (mw.aembedding_wrapper, True),
    }

    METADATA = {"organizationName": "AcmeCorp", "subscriber": {"email": "dev@acme.example"}}

    @pytest.mark.parametrize("usage_metadata", [METADATA, None], ids=["metadata", "none"])
    @pytest.mark.parametrize("entry_point", list(WRAPPERS))
    def test_selective_pass_through_drops_usage_metadata(self, entry_point, usage_metadata, monkeypatch, payloads):
        monkeypatch.setenv("REVENIUM_SELECTIVE_METERING", "true")
        wrapper, is_async = self.WRAPPERS[entry_point]
        seen = []

        def wrapped(*_args, **kwargs):
            seen.append(kwargs)
            return "provider response"

        async def awrapped(*args, **kwargs):
            return wrapped(*args, **kwargs)

        result = wrapper(awrapped if is_async else wrapped, None, (), {"model": MODEL, "usage_metadata": usage_metadata})
        if is_async:
            result = asyncio.run(result)

        assert result == "provider response"
        assert seen == [{"model": MODEL}]
        assert payloads == []


class TestRefusedCompletion:
    """BACK-4239: a failed LiteLLM client call sends one ERROR record."""

    @staticmethod
    def assert_one_error_record(payloads, *, streamed):
        assert len(payloads) == 1, payloads
        record = payloads[0]
        assert record["stop_reason"] == "ERROR"
        assert (record["input_token_count"], record["output_token_count"], record["total_token_count"]) == (0, 0, 0)
        assert record["model"] == MODEL
        assert record["is_streamed"] is streamed
        return record

    @pytest.mark.parametrize("call", [
        pytest.param(lambda: litellm.completion(**completion_kwargs(mock_response="litellm.RateLimitError")),
                     id="completion"),
        pytest.param(lambda: asyncio.run(litellm.acompletion(**completion_kwargs(mock_response="litellm.RateLimitError"))),
                     id="acompletion"),
    ])
    def test_refused_completion_sends_one_error_record(self, payloads, call):
        with pytest.raises(litellm.RateLimitError):
            call()

        record = self.assert_one_error_record(payloads, streamed=False)
        assert "RateLimitError" in record["error_reason"]
        assert "HTTP 429" in record["error_reason"]
        assert record.get("error_code") == 429

    @pytest.fixture(params=[
        pytest.param(lambda: litellm.APIConnectionError(message="upstream unreachable", llm_provider="openai", model=MODEL),
                     id="connection_error"),
        pytest.param(lambda: litellm.Timeout(message="upstream timed out", model=MODEL, llm_provider="openai"),
                     id="timeout"),
    ])
    def error(self, request):
        return request.param()

    def assert_no_invented_status(self, payloads, error, *, streamed):
        record = self.assert_one_error_record(payloads, streamed=streamed)
        assert type(error).__name__ in record["error_reason"]
        assert "(HTTP" not in record["error_reason"]
        assert "error_code" not in record

    def test_a_connection_error_sends_no_error_code(self, payloads, error):
        """BACK-4406: LiteLLM gives a connection error a status of 500 that no server answered."""

        def unreachable(*_args, **_kwargs):
            raise error

        with pytest.raises(type(error)):
            mw.completion_wrapper(unreachable, None, (), {"model": MODEL, "messages": MESSAGES})

        self.assert_no_invented_status(payloads, error, streamed=False)

    def test_a_stream_cut_off_by_a_connection_error_sends_no_error_code(self, payloads, error):
        def stream_cut_off_after_one_chunk(*_args, **_kwargs):
            yield litellm.ModelResponseStream(model=MODEL, choices=[
                litellm.utils.StreamingChoices(delta=litellm.utils.Delta(content="hi"))])
            raise error

        stream = mw.completion_wrapper(stream_cut_off_after_one_chunk, None, (),
                                       {"model": MODEL, "messages": MESSAGES, "stream": True})
        with pytest.raises(type(error)):
            list(stream)

        self.assert_no_invented_status(payloads, error, streamed=True)

    def test_stream_failing_before_first_chunk_sends_one_error_record(self, payloads):
        def stream_refused_at_its_first_read(*_args, **_kwargs):
            raise ConnectionError("upstream refused the stream")
            yield  # pragma: no cover

        stream = mw.completion_wrapper(stream_refused_at_its_first_read, None, (),
                                       {"model": MODEL, "messages": MESSAGES, "stream": True})
        with pytest.raises(ConnectionError):
            list(stream)

        record = self.assert_one_error_record(payloads, streamed=True)
        assert "ConnectionError" in record["error_reason"]

    def test_stream_failing_part_way_is_recorded_as_error(self, payloads):
        def stream_cut_off_after_one_chunk(*_args, **_kwargs):
            yield litellm.ModelResponseStream(model=MODEL, choices=[
                litellm.utils.StreamingChoices(delta=litellm.utils.Delta(content="hi"))])
            raise ConnectionError("connection reset mid-stream")

        stream = mw.completion_wrapper(stream_cut_off_after_one_chunk, None, (),
                                       {"model": MODEL, "messages": MESSAGES, "stream": True})
        with pytest.raises(ConnectionError):
            list(stream)

        assert len(payloads) == 1, payloads
        assert payloads[0]["stop_reason"] == "ERROR"
        assert "ConnectionError" in payloads[0]["error_reason"]

    def test_stream_failing_after_hidden_usage_records_those_tokens(self, payloads):
        chunk = litellm.ModelResponseStream(model=MODEL, choices=[
            litellm.utils.StreamingChoices(delta=litellm.utils.Delta(content="hi"))])
        chunk._hidden_params["usage"] = litellm.Usage(prompt_tokens=11, completion_tokens=7, total_tokens=18)

        def stream_cut_off_after_a_chunk_carrying_hidden_usage(*_args, **_kwargs):
            yield chunk
            raise ConnectionError("connection reset mid-stream")

        stream = mw.completion_wrapper(stream_cut_off_after_a_chunk_carrying_hidden_usage, None, (),
                                       {"model": MODEL, "messages": MESSAGES, "stream": True})
        with pytest.raises(ConnectionError):
            list(stream)

        assert len(payloads) == 1, payloads
        record = payloads[0]
        assert record["stop_reason"] == "ERROR"
        assert (record["input_token_count"], record["output_token_count"]) == (11, 7)

    def test_a_failing_proxied_stream_raises_the_providers_error(self, payloads, monkeypatch):
        monkeypatch.setattr(mw, "proxy_meters_request", lambda _kwargs: True)

        def stream_cut_off_after_one_chunk(*_args, **_kwargs):
            yield litellm.ModelResponseStream(model=MODEL, choices=[
                litellm.utils.StreamingChoices(delta=litellm.utils.Delta(content="hi"))])
            raise ConnectionError("connection reset mid-stream")

        stream = mw.completion_wrapper(stream_cut_off_after_one_chunk, None, (),
                                       {"model": MODEL, "messages": MESSAGES, "stream": True})
        with pytest.raises(ConnectionError, match="connection reset mid-stream"):
            list(stream)

        assert payloads == []

    def test_a_failing_proxied_async_stream_raises_the_providers_error(self, payloads, monkeypatch):
        monkeypatch.setattr(mw, "proxy_meters_request", lambda _kwargs: True)

        async def stream_cut_off_after_one_chunk():
            yield litellm.ModelResponseStream(model=MODEL, choices=[
                litellm.utils.StreamingChoices(delta=litellm.utils.Delta(content="hi"))])
            raise ConnectionError("connection reset mid-stream")

        async def open_stream(*_args, **_kwargs):
            return stream_cut_off_after_one_chunk()

        async def consume():
            stream = await mw.acompletion_wrapper(open_stream, None, (),
                                                  {"model": MODEL, "messages": MESSAGES, "stream": True})
            return [chunk async for chunk in stream]

        with pytest.raises(ConnectionError, match="connection reset mid-stream"):
            asyncio.run(consume())

        assert payloads == []

    def test_an_abandoned_stream_is_not_recorded_as_error(self, payloads):
        stream = litellm.completion(**completion_kwargs(stream=True))
        next(iter(stream))
        stream.close()

        assert len(payloads) == 1, payloads
        assert payloads[0]["stop_reason"] != "ERROR"


class _AlwaysFails(BaseHTTPRequestHandler):
    def do_POST(self):
        self.rfile.read(int(self.headers.get("content-length", 0)))
        self.send_response(500)
        self.send_header("content-type", "application/json")
        self.end_headers()
        self.wfile.write(b'{"type": "error", "error": {"type": "api_error", "message": "upstream broke"}}')

    def log_message(self, *_args):
        pass


@pytest.fixture
def closed_port_base():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    return f"http://127.0.0.1:{port}"


@pytest.fixture
def failing_server_base():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _AlwaysFails)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()
    server.server_close()


def real_litellm_call(model, api_base, *, stream):
    response = litellm.completion(model=model, messages=MESSAGES, api_base=api_base, api_key="sk-fake",
                                  stream=stream, num_retries=0, max_retries=0)
    if stream:
        list(response)


REAL_PROVIDER_CALLS = [
    pytest.param("openai/gpt-4o-mini", "/v1", False, id="openai"),
    pytest.param("openai/gpt-4o-mini", "/v1", True, id="openai_stream"),
    pytest.param("anthropic/claude-sonnet-4-5", "", False, id="anthropic"),
    pytest.param("anthropic/claude-sonnet-4-5", "", True, id="anthropic_stream"),
]


class TestRealLiteLLMProviderFailures:
    """BACK-4406: LiteLLM raises a refused connection on its ``openai/`` and
    ``anthropic/`` providers as an ``InternalServerError`` with status 500."""

    @pytest.mark.parametrize("model, path, stream", REAL_PROVIDER_CALLS)
    def test_a_refused_connection_sends_no_error_code(self, payloads, closed_port_base, model, path, stream):
        with pytest.raises(litellm.InternalServerError):
            real_litellm_call(model, closed_port_base + path, stream=stream)

        assert len(payloads) == 1, payloads
        record = payloads[0]
        assert record["stop_reason"] == "ERROR"
        assert "InternalServerError" in record["error_reason"]
        assert "(HTTP" not in record["error_reason"]
        assert "error_code" not in record

    @pytest.mark.parametrize("model, path, stream", REAL_PROVIDER_CALLS)
    def test_a_500_the_provider_answered_sends_it_as_the_error_code(self, payloads, failing_server_base,
                                                                    model, path, stream):
        with pytest.raises(litellm.InternalServerError):
            real_litellm_call(model, failing_server_base + path, stream=stream)

        assert len(payloads) == 1, payloads
        assert payloads[0]["error_code"] == 500
        assert "(HTTP 500)" in payloads[0]["error_reason"]

    def test_a_500_answered_while_the_caller_handles_a_connection_error_keeps_its_code(
            self, payloads, failing_server_base):
        try:
            raise openai.APIConnectionError(request=httpx.Request("POST", failing_server_base))
        except openai.APIConnectionError:
            with pytest.raises(litellm.InternalServerError):
                real_litellm_call("openai/gpt-4o-mini", failing_server_base + "/v1", stream=False)

        assert len(payloads) == 1, payloads
        assert payloads[0]["error_code"] == 500
