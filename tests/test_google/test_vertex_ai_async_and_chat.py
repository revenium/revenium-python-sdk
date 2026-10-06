"""Vertex AI async calls and chat-session messages each produce exactly one metering record (BACK-3610).

The models are real vertexai objects whose prediction clients are the
entry-point matrix stubs, so the SDK's own request building, chat history and
response parsing run; only the gapic call is faked. Runs in the vertex
environment (``.[google-vertex]``), where vertexai is installed.
"""
import asyncio
import inspect
import json

import pytest

pytest.importorskip("vertexai")

import revenium_middleware.google.vertex_ai.middleware  # noqa: E402,F401
from entry_points import calls  # noqa: E402


def payloads(recorder):
    calls.wait_for_metering()
    return calls.recorded_payloads(recorder)


def completion_kwargs(recorder):
    calls.wait_for_metering()
    return [recorded.kwargs for recorded in recorder.ai.create_completion.call_args_list]


def assert_one_chat_payload(recorder):
    assert payloads(recorder) == [{
        "operation": "completion",
        "model": calls.GEMINI_MODEL,
        "provider": "Google",
        "operation_type": "CHAT",
        "input_token_count": calls.INPUT_TOKENS,
        "output_token_count": calls.OUTPUT_TOKENS,
        "cache_read_token_count": 0,
        "reasoning_token_count": 0,
    }]


async def drain(stream):
    return [chunk async for chunk in stream]


class TestGenerateContentAsync:
    def test_a_non_streamed_call_is_metered_once(self, mock_revenium_client):
        response = asyncio.run(calls.vertex_model().generate_content_async("hi"))

        assert response.text == "hi"
        assert_one_chat_payload(mock_revenium_client)

    def test_a_streamed_call_is_metered_once_when_the_stream_ends(self, mock_revenium_client):
        async def run():
            return await drain(await calls.vertex_model().generate_content_async("hi", stream=True))

        chunks = asyncio.run(run())

        assert "".join(chunk.text for chunk in chunks) == "hi"
        assert_one_chat_payload(mock_revenium_client)
        assert completion_kwargs(mock_revenium_client)[0]["is_streamed"] is True

    def test_concurrent_calls_are_each_metered(self, mock_revenium_client):
        async def run():
            model = calls.vertex_model()
            await asyncio.gather(model.generate_content_async("a"), model.generate_content_async("b"))

        asyncio.run(run())

        assert len(payloads(mock_revenium_client)) == 2


class TestGetEmbeddingsAsync:
    def test_is_metered_once_with_the_token_count(self, mock_revenium_client):
        embeddings = asyncio.run(calls.vertex_embedding_model().get_embeddings_async(["hi"]))

        assert embeddings[0].values == [0.1, 0.2]
        assert payloads(mock_revenium_client) == [{
            "operation": "completion",
            "model": calls.VERTEX_EMBED_MODEL,
            "provider": "Google",
            "operation_type": "EMBED",
            "input_token_count": calls.INPUT_TOKENS,
            "output_token_count": 0,
            "cache_read_token_count": 0,
            "reasoning_token_count": 0,
        }]


class TestChatSessionSendMessage:
    def test_one_message_is_one_record_and_the_history_still_grows(self, mock_revenium_client):
        chat = calls.vertex_model().start_chat()

        chat.send_message("hi")

        assert [content.role for content in chat.history] == ["user", "model"]
        assert_one_chat_payload(mock_revenium_client)

    def test_each_message_of_a_conversation_is_its_own_record(self, mock_revenium_client):
        chat = calls.vertex_model().start_chat()

        chat.send_message("hi")
        chat.send_message("again")

        assert len(payloads(mock_revenium_client)) == 2

    def test_a_streamed_message_is_metered_once(self, mock_revenium_client):
        chat = calls.vertex_model().start_chat()

        chunks = list(chat.send_message("hi", stream=True))

        assert len(chunks) == 2
        assert_one_chat_payload(mock_revenium_client)
        assert completion_kwargs(mock_revenium_client)[0]["is_streamed"] is True

    def test_usage_metadata_is_taken_off_the_call_and_reported(self, mock_revenium_client):
        calls.vertex_model().start_chat().send_message("hi", usage_metadata={"trace_id": "trace-3610"})

        assert completion_kwargs(mock_revenium_client)[0]["trace_id"] == "trace-3610"


class TestUsageMetadata:
    def test_the_call_overrides_the_model_key_by_key_under_either_spelling(self, mock_revenium_client):
        model = calls.vertex_model()
        model._revenium_usage_metadata = {"traceId": "model-trace", "task_type": "model-task"}

        model.generate_content("hi", usage_metadata={"trace_id": "call-trace"})

        record = completion_kwargs(mock_revenium_client)[0]
        assert (record["trace_id"], record["task_type"]) == ("call-trace", "model-task")

    def test_a_chat_message_merges_model_session_and_call_metadata(self, mock_revenium_client):
        model = calls.vertex_model()
        model._revenium_usage_metadata = {"trace_id": "model-trace", "task_type": "model-task", "agent": "model-agent"}
        chat = model.start_chat()
        chat._revenium_usage_metadata = {"task_type": "session-task", "agent": "session-agent"}

        chat.send_message("hi", usage_metadata={"agent": "call-agent"})

        record = completion_kwargs(mock_revenium_client)[0]
        assert (record["trace_id"], record["task_type"], record["agent"]) == (
            "model-trace", "session-task", "call-agent")


class TestChatPromptCapture:
    def test_the_captured_input_carries_the_earlier_turns(self, mock_revenium_client, monkeypatch):
        monkeypatch.setenv("REVENIUM_CAPTURE_PROMPTS", "true")
        chat = calls.vertex_model().start_chat()

        chat.send_message("first question")
        chat.send_message("second question")

        captured = json.loads(completion_kwargs(mock_revenium_client)[1]["input_messages"])
        assert [(turn["role"], turn["parts"][0]["text"]) for turn in captured] == [
            ("user", "first question"), ("model", "hi"), ("user", "second question")]

    def test_a_first_message_captures_only_itself(self, mock_revenium_client, monkeypatch):
        monkeypatch.setenv("REVENIUM_CAPTURE_PROMPTS", "true")

        calls.vertex_model().start_chat().send_message("only question")

        captured = json.loads(completion_kwargs(mock_revenium_client)[0]["input_messages"])
        assert captured == [{"role": "user", "parts": [{"text": "only question"}]}]


class TestChatSessionSendMessageAsync:
    def test_a_non_streamed_message_is_metered_once(self, mock_revenium_client):
        chat = calls.vertex_model().start_chat()

        asyncio.run(chat.send_message_async("hi"))

        assert len(chat.history) == 2
        assert_one_chat_payload(mock_revenium_client)

    def test_a_streamed_message_is_metered_once(self, mock_revenium_client):
        chat = calls.vertex_model().start_chat()

        async def run():
            return await drain(await chat.send_message_async("hi", stream=True))

        assert len(asyncio.run(run())) == 2
        assert_one_chat_payload(mock_revenium_client)


class TestNestedPublicCallsAreNotMeteredTwice:
    """A vertexai release that routes ChatSession through the wrapped public methods must not double-count."""

    def test_a_chat_message_routed_through_generate_content(self, mock_revenium_client, monkeypatch):
        public_model, chat_model = calls.vertex_model(), calls.vertex_model()
        monkeypatch.setattr(chat_model, "_generate_content",
                            lambda **kwargs: public_model.generate_content(**kwargs))

        chat_model.start_chat().send_message("hi")

        assert_one_chat_payload(mock_revenium_client)

    def test_a_streamed_chat_message_routed_through_generate_content(self, mock_revenium_client, monkeypatch):
        public_model, chat_model = calls.vertex_model(), calls.vertex_model()
        monkeypatch.setattr(chat_model, "_generate_content_streaming",
                            lambda **kwargs: public_model.generate_content(stream=True, **kwargs))

        list(chat_model.start_chat().send_message("hi", stream=True))

        assert_one_chat_payload(mock_revenium_client)

    def test_an_async_chat_message_routed_through_generate_content_async(self, mock_revenium_client,
                                                                         monkeypatch):
        public_model, chat_model = calls.vertex_model(), calls.vertex_model()
        monkeypatch.setattr(chat_model, "_generate_content_async",
                            lambda **kwargs: public_model.generate_content_async(**kwargs))

        asyncio.run(chat_model.start_chat().send_message_async("hi"))

        assert_one_chat_payload(mock_revenium_client)

    def test_the_next_call_after_a_metered_one_is_metered_again(self, mock_revenium_client):
        model = calls.vertex_model()

        model.generate_content("hi")
        asyncio.run(model.generate_content_async("hi"))

        assert len(payloads(mock_revenium_client)) == 2


class TestRegistration:
    def test_the_preview_chat_session_reuses_the_wrap_it_inherits(self):
        from vertexai.generative_models import ChatSession
        from vertexai.preview.generative_models import ChatSession as PreviewChatSession

        for method in ("send_message", "send_message_async"):
            assert method not in vars(PreviewChatSession)
            assert inspect.getattr_static(PreviewChatSession, method) is vars(ChatSession)[method]

    def test_no_video_generation_wrap_is_registered(self):
        from revenium_middleware._core import patch_registry

        assert not [key for key in patch_registry._patched if "VideoGenerationModel" in key]
