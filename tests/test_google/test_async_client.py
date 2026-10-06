"""The async google-genai client is metered like its sync twin: one payload per call (BACK-3605).

Calls run through the real google-genai client against the entry-point matrix's
local HTTP stub, so the chat paths prove that AsyncChat reaches the wrapped
AsyncModels exactly once rather than being wrapped a second time.
"""
import asyncio
import gc
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from revenium_middleware import revenium_meter
from revenium_middleware.google.google_ai import middleware as genai_mw

pytestmark = pytest.mark.skipif(genai_mw is None, reason="google-genai SDK not installed")

from entry_points import calls  # noqa: E402

MODEL = calls.GEMINI_MODEL
STUB_TOKENS = {"input_token_count": calls.INPUT_TOKENS, "output_token_count": calls.OUTPUT_TOKENS}


def payloads(recorder):
    calls.wait_for_metering()
    return calls.recorded_payloads(recorder)


def completion_kwargs(recorder):
    return [c.kwargs for c in recorder.ai.create_completion.call_args_list]


def assert_one_completion(recorder, **expected):
    recorded = payloads(recorder)
    assert len(recorded) == 1, recorded
    assert recorded[0]["operation"] == "completion"
    assert recorded[0]["model"] == expected.pop("model", MODEL)
    for field, value in {**STUB_TOKENS, **expected}.items():
        assert recorded[0][field] == value, (field, recorded[0])


async def drain(stream):
    async for _chunk in stream:
        pass


class TestAsyncModels:
    def test_generate_content_sends_one_payload_with_the_stub_usage(self, mock_revenium_client):
        response = asyncio.run(calls._genai().aio.models.generate_content(model=MODEL, contents="hi"))

        assert response.text == "hi"
        assert_one_completion(mock_revenium_client, cache_read_token_count=calls.CACHE_READ_TOKENS)
        assert completion_kwargs(mock_revenium_client)[0]["is_streamed"] is False

    def test_generate_content_stream_sends_one_streamed_payload_at_stream_end(self, mock_revenium_client):
        async def run():
            stream = await calls._genai().aio.models.generate_content_stream(model=MODEL, contents="hi")
            assert payloads(mock_revenium_client) == []
            await drain(stream)

        asyncio.run(run())

        assert_one_completion(mock_revenium_client)
        assert completion_kwargs(mock_revenium_client)[0]["is_streamed"] is True

    def test_embed_content_sends_one_embed_payload_without_tokens(self, mock_revenium_client):
        asyncio.run(calls._genai().aio.models.embed_content(model=calls.GEMINI_EMBED_MODEL, contents="hi"))

        recorded = payloads(mock_revenium_client)
        assert len(recorded) == 1, recorded
        assert recorded[0]["operation_type"] == "EMBED"
        assert recorded[0]["model"] == calls.GEMINI_EMBED_MODEL
        assert not recorded[0]["input_token_count"]

    def test_generate_images_sends_one_image_payload(self, mock_revenium_client):
        asyncio.run(calls._genai_vertex_express().aio.models.generate_images(
            model=calls.IMAGEN_MODEL, prompt="a cat"))

        recorded = payloads(mock_revenium_client)
        assert recorded == [{"operation": "image", "model": calls.IMAGEN_MODEL, "provider": "Google",
                             "requested_image_count": 1, "actual_image_count": 1}]

    def test_usage_metadata_reaches_the_payload_and_not_the_sdk(self, mock_revenium_client):
        asyncio.run(calls._genai().aio.models.generate_content(
            model=MODEL, contents="hi", usage_metadata={"trace_id": "trace-async"}))

        assert [kw["trace_id"] for kw in completion_kwargs(mock_revenium_client)] == ["trace-async"]

    def test_selective_metering_skips_the_call_and_still_drops_usage_metadata(
            self, mock_revenium_client, monkeypatch):
        monkeypatch.setenv("REVENIUM_SELECTIVE_METERING", "true")

        asyncio.run(calls._genai().aio.models.generate_content(
            model=MODEL, contents="hi", usage_metadata={"trace_id": "unmetered"}))

        assert payloads(mock_revenium_client) == []

    def test_a_metering_failure_never_reaches_the_caller(self, mock_revenium_client):
        with patch.object(genai_mw, "create_google_ai_metering_call", side_effect=RuntimeError("boom")):
            response = asyncio.run(calls._genai().aio.models.generate_content(model=MODEL, contents="hi"))

        assert response.text == "hi"


class TestAsyncStreamTermination:
    def test_early_break_still_meters_once(self, mock_revenium_client):
        async def run():
            stream = await calls._genai().aio.models.generate_content_stream(model=MODEL, contents="hi")
            async for _chunk in stream:
                break
            del stream
            gc.collect()

        asyncio.run(run())

        assert len(payloads(mock_revenium_client)) == 1

    def test_aclose_meters_once_and_ends_iteration(self, mock_revenium_client):
        async def run():
            stream = await calls._genai().aio.models.generate_content_stream(model=MODEL, contents="hi")
            await stream.__anext__()
            await stream.aclose()
            with pytest.raises(StopAsyncIteration):
                await stream.__anext__()
            await stream.aclose()

        asyncio.run(run())

        assert len(payloads(mock_revenium_client)) == 1

    def test_async_context_manager_meters_once(self, mock_revenium_client):
        async def run():
            async with await calls._genai().aio.models.generate_content_stream(
                    model=MODEL, contents="hi") as stream:
                await drain(stream)

        asyncio.run(run())

        assert len(payloads(mock_revenium_client)) == 1

    def test_a_provider_error_mid_stream_is_raised_and_metered_once(self, mock_revenium_client):
        chunk = calls_chunk()

        async def dropped_stream():
            yield chunk
            raise ConnectionError("dropped")

        async def run():
            with pytest.raises(ConnectionError):
                await drain(genai_mw.AsyncStreamWrapper(dropped_stream(), genai_mw._now(), {}))

        asyncio.run(run())

        assert_one_completion(mock_revenium_client)


class TestMeteringFailuresNeverReachAsyncCallers:
    def test_a_failure_before_the_call_sends_it_unmetered(self, mock_revenium_client):
        with patch.object(genai_mw, "_metering_skipped", side_effect=RuntimeError("boom")):
            response = asyncio.run(calls._genai().aio.models.generate_content(
                model=MODEL, contents="hi", usage_metadata={"trace_id": "t"}))

        assert response.text == "hi"
        assert payloads(mock_revenium_client) == []

    def test_a_stream_setup_failure_returns_the_provider_stream_unmetered(self, mock_revenium_client):
        async def run():
            stream = await calls._genai().aio.models.generate_content_stream(model=MODEL, contents="hi")
            assert not isinstance(stream, genai_mw.AsyncStreamWrapper)
            await drain(stream)

        with patch.object(genai_mw, "detect_vision_content", side_effect=RuntimeError("boom")):
            asyncio.run(run())

        assert payloads(mock_revenium_client) == []

    def test_a_failure_after_a_streamed_call_is_not_raised(self, mock_revenium_client):
        async def run():
            await drain(await calls._genai().aio.models.generate_content_stream(model=MODEL, contents="hi"))

        with patch.object(genai_mw, "create_google_ai_metering_call", side_effect=RuntimeError("boom")):
            asyncio.run(run())

    def test_a_failure_before_a_sync_call_sends_it_unmetered(self, mock_revenium_client):
        with patch.object(genai_mw, "_metering_skipped", side_effect=RuntimeError("boom")):
            response = calls._genai().models.generate_content(model=MODEL, contents="hi")

        assert response.text == "hi"
        assert payloads(mock_revenium_client) == []


class TestSelectiveMeteringIsDecidedWhenAwaited:
    @pytest.fixture(autouse=True)
    def selective(self, monkeypatch):
        monkeypatch.setenv("REVENIUM_SELECTIVE_METERING", "true")

    def test_created_outside_and_awaited_inside_a_metered_scope_is_metered(self, mock_revenium_client):
        @revenium_meter()
        async def metered(pending):
            return await pending

        asyncio.run(metered(calls._genai().aio.models.generate_content(model=MODEL, contents="hi")))

        assert len(payloads(mock_revenium_client)) == 1

    def test_a_stream_created_outside_and_awaited_inside_a_metered_scope_is_metered(self, mock_revenium_client):
        @revenium_meter()
        async def metered(pending):
            await drain(await pending)

        asyncio.run(metered(calls._genai().aio.models.generate_content_stream(model=MODEL, contents="hi")))

        assert len(payloads(mock_revenium_client)) == 1

    def test_created_inside_and_awaited_outside_a_metered_scope_is_not_metered(self, mock_revenium_client):
        @revenium_meter()
        async def create():
            return calls._genai().aio.models.generate_content(model=MODEL, contents="hi")

        async def run():
            await (await create())

        asyncio.run(run())

        assert payloads(mock_revenium_client) == []


class _ClosableStream:
    def __init__(self, chunks):
        self._chunks = iter(chunks)
        self.close_calls = 0

    def __iter__(self):
        return self

    def __next__(self):
        return next(self._chunks)

    def close(self):
        self.close_calls += 1


class TestAbandonedStreamsAreReleased:
    def test_a_collected_sync_stream_closes_the_provider_stream_and_meters_once(self, mock_revenium_client):
        provider_stream = _ClosableStream([calls_chunk(), calls_chunk()])
        wrapper = genai_mw.handle_streaming_response(provider_stream, genai_mw._now(), {})
        next(wrapper)
        del wrapper
        gc.collect()

        assert provider_stream.close_calls == 1
        assert len(payloads(mock_revenium_client)) == 1

    def test_a_collected_async_stream_leaves_the_provider_generator_to_the_loop_finalizer(
            self, mock_revenium_client):
        # __del__ cannot await aclose(); asyncio's async-generator finalizer hook
        # closes google-genai's generator once the wrapper drops the last reference.
        closed = []

        async def provider_stream():
            try:
                yield calls_chunk()
                yield calls_chunk()
            finally:
                closed.append(True)

        async def run():
            wrapper = genai_mw.AsyncStreamWrapper(provider_stream(), genai_mw._now(), {})
            await wrapper.__anext__()
            del wrapper
            gc.collect()
            for _ in range(3):
                await asyncio.sleep(0)

        asyncio.run(run())

        assert closed == [True]
        assert len(payloads(mock_revenium_client)) == 1


def calls_chunk():
    return SimpleNamespace(
        text="hi", model_version=MODEL, candidates=[],
        usage_metadata=SimpleNamespace(prompt_token_count=calls.INPUT_TOKENS,
                                       candidates_token_count=calls.OUTPUT_TOKENS,
                                       total_token_count=calls.INPUT_TOKENS + calls.OUTPUT_TOKENS,
                                       cached_content_token_count=0))


class TestAsyncChats:
    def test_each_send_message_is_metered_once(self, mock_revenium_client):
        async def run():
            chat = calls._genai().aio.chats.create(model=MODEL)
            await chat.send_message("hi")
            await chat.send_message("again")

        asyncio.run(run())

        recorded = payloads(mock_revenium_client)
        assert len(recorded) == 2, recorded
        assert all(p["input_token_count"] == calls.INPUT_TOKENS for p in recorded)

    def test_send_message_stream_is_metered_once(self, mock_revenium_client):
        async def run():
            chat = calls._genai().aio.chats.create(model=MODEL)
            await drain(await chat.send_message_stream("hi"))

        asyncio.run(run())

        assert_one_completion(mock_revenium_client)
        assert completion_kwargs(mock_revenium_client)[0]["is_streamed"] is True


def _sync_generate_content():
    calls._genai().models.generate_content(model=MODEL, contents="hi")


def _sync_generate_content_stream():
    for _chunk in calls._genai().models.generate_content_stream(model=MODEL, contents="hi"):
        pass


def _sync_chat_send_message():
    calls._genai().chats.create(model=MODEL).send_message("hi")


def _sync_chat_send_message_stream():
    for _chunk in calls._genai().chats.create(model=MODEL).send_message_stream("hi"):
        pass


@pytest.mark.parametrize("call", [
    _sync_generate_content, _sync_generate_content_stream, _sync_chat_send_message, _sync_chat_send_message_stream,
])
def test_sync_controls_still_send_one_payload(call, mock_revenium_client):
    call()

    assert_one_completion(mock_revenium_client)


def test_sync_embed_and_images_controls_still_send_one_payload_each(mock_revenium_client):
    calls._genai().models.embed_content(model=calls.GEMINI_EMBED_MODEL, contents="hi")
    calls._genai_vertex_express().models.generate_images(model=calls.IMAGEN_MODEL, prompt="a cat")

    recorded = payloads(mock_revenium_client)
    assert sorted(p["operation"] for p in recorded) == ["completion", "image"]
    assert [p["operation_type"] for p in recorded if p["operation"] == "completion"] == ["EMBED"]


def test_sync_selective_metering_skip_also_drops_usage_metadata(mock_revenium_client, monkeypatch):
    monkeypatch.setenv("REVENIUM_SELECTIVE_METERING", "true")

    calls._genai().models.generate_content(model=MODEL, contents="hi", usage_metadata={"trace_id": "unmetered"})

    assert payloads(mock_revenium_client) == []
