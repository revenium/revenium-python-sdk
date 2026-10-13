"""Vertex AI async calls and chat-session messages each produce exactly one metering record (BACK-3610).

The models are real vertexai objects whose prediction clients are the
entry-point matrix stubs, so the SDK's own request building, chat history and
response parsing run; only the gapic call is faked. Runs in the vertex
environment (``.[google-vertex]``), where vertexai is installed.
"""
import asyncio
import gc
import importlib
import inspect
import json
import threading
from types import SimpleNamespace

import pytest

pytest.importorskip("vertexai")

import revenium_middleware.google.vertex_ai.middleware  # noqa: E402,F401
from revenium_middleware._core.decorators import revenium_metadata  # noqa: E402
from entry_points import calls  # noqa: E402

PNG_1X1 = bytes.fromhex("89504e470d0a1a0a0000000d4948445200000001000000010806000000")


def payloads(recorder):
    calls.wait_for_metering()
    return calls.recorded_payloads(recorder)


def completion_kwargs(recorder):
    calls.wait_for_metering()
    return [recorded.kwargs for recorded in recorder.ai.create_completion.call_args_list]


def image_kwargs(recorder):
    calls.wait_for_metering()
    return [recorded.kwargs for recorded in recorder.ai.create_image.call_args_list]


def _call_imagen(name, model, kwargs, seen=None):
    from revenium_middleware.google.vertex_ai import middleware as vertex_middleware

    impl = {"generate_images": vertex_middleware._imagen_generate_images_impl,
            "edit_image": vertex_middleware._imagen_edit_image_impl}[name]

    def wrapped(*_args, **call_kwargs):
        if seen is not None:
            seen.append(call_kwargs)
        return SimpleNamespace(images=[object()])

    return impl(wrapped, model, ("prompt",), dict(kwargs))


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


class _BatchEndpointClient:
    TOKEN_COUNTS = (5, 120, 300)

    @classmethod
    def response(cls):
        from google.cloud.aiplatform_v1.types import PredictResponse
        predictions = [{"embeddings": {"values": [0.1], "statistics": {"token_count": count, "truncated": False}}}
                       for count in cls.TOKEN_COUNTS]
        return PredictResponse.from_json(json.dumps({"predictions": predictions}))

    def predict(self, endpoint, instances, parameters, timeout=None):
        return self.response()


class _BatchEndpointAsyncClient:
    async def predict(self, endpoint, instances, parameters, timeout=None):
        return _BatchEndpointClient.response()


class TestGetEmbeddingsBatch:
    @pytest.mark.parametrize("mode", ["sync", "async"])
    def test_every_text_in_the_batch_is_billed(self, mock_revenium_client, mode):
        model = calls.vertex_embedding_model()
        model._endpoint._prediction_client_value = _BatchEndpointClient()
        model._endpoint._prediction_async_client_value = _BatchEndpointAsyncClient()
        texts = ["a", "b", "c"]

        if mode == "sync":
            embeddings = model.get_embeddings(texts)
        else:
            embeddings = asyncio.run(model.get_embeddings_async(texts))

        assert len(embeddings) == 3
        recorded = payloads(mock_revenium_client)
        assert [(p["operation_type"], p["input_token_count"], p["output_token_count"]) for p in recorded] == [
            ("EMBED", sum(_BatchEndpointClient.TOKEN_COUNTS), 0)]


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

    def test_decorator_metadata_reaches_the_record(self, mock_revenium_client):
        @revenium_metadata(organizationName="DecOrg", trace_id="dec-trace")
        def decorated():
            calls.vertex_model().generate_content("hi")

        decorated()

        record = completion_kwargs(mock_revenium_client)[0]
        assert (record.get("organization_name"), record.get("trace_id")) == ("DecOrg", "dec-trace")

    def test_the_model_and_the_call_override_the_decorator(self, mock_revenium_client):
        @revenium_metadata(trace_id="dec-trace", task_type="dec-task", agent="dec-agent")
        def decorated():
            model = calls.vertex_model()
            model._revenium_usage_metadata = {"task_type": "model-task"}
            model.generate_content("hi", usage_metadata={"agent": "call-agent"})

        decorated()

        record = completion_kwargs(mock_revenium_client)[0]
        assert (record.get("trace_id"), record.get("task_type"), record.get("agent")) == (
            "dec-trace", "model-task", "call-agent")

    @pytest.mark.parametrize("name", ["generate_images", "edit_image"])
    def test_decorator_metadata_reaches_the_image_record(self, mock_revenium_client, name):
        @revenium_metadata(organizationName="AcmeCorp", trace_id="dec-trace")
        def decorated():
            _call_imagen(name, SimpleNamespace(_model_id="imagen-3.0-generate-002"), {})

        decorated()

        record = image_kwargs(mock_revenium_client)[0]
        assert (record.get("organization_name"), record.get("trace_id")) == ("AcmeCorp", "dec-trace")

    @pytest.mark.parametrize("name", ["generate_images", "edit_image"])
    def test_the_image_call_beats_the_model_and_the_decorator_and_is_not_forwarded(self, mock_revenium_client, name):
        seen = []

        @revenium_metadata(trace_id="dec-trace", task_type="dec-task", agent="dec-agent")
        def decorated():
            model = SimpleNamespace(_model_id="imagen-3.0-generate-002",
                                    _revenium_usage_metadata={"task_type": "model-task"})
            _call_imagen(name, model, {"usage_metadata": {"agent": "call-agent"}}, seen)

        decorated()

        record = image_kwargs(mock_revenium_client)[0]
        assert (record.get("trace_id"), record.get("task_type"), record.get("agent")) == (
            "dec-trace", "model-task", "call-agent")
        assert seen == [{}]

    def test_per_call_trace_and_skill_fields_win_over_the_environment(self, mock_revenium_client, monkeypatch):
        monkeypatch.setenv("REVENIUM_TRANSACTION_NAME", "deployment-default")
        monkeypatch.setenv("REVENIUM_RETRY_NUMBER", "1")
        monkeypatch.setenv("REVENIUM_SKILL_NAME", "env-skill")
        per_call = {"parent_transaction_id": "call-parent", "transaction_name": "Generate Response",
                    "retry_number": 3, "skill_name": "md-skill", "skill_source": "plugin"}

        calls.vertex_model().generate_content("hi", usage_metadata=per_call)

        record = completion_kwargs(mock_revenium_client)[0]
        assert {name: record.get(name) for name in per_call} == per_call

    def test_a_quality_score_of_zero_reaches_the_record(self, mock_revenium_client):
        calls.vertex_model().generate_content("hi", usage_metadata={"response_quality_score": 0.0})

        record = completion_kwargs(mock_revenium_client)[0]
        assert "response_quality_score" in record
        assert record["response_quality_score"] == 0.0


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


class TestVisionContent:
    def test_an_image_call_is_metered_once_marked_as_vision(self, mock_revenium_client):
        from vertexai.generative_models import Part

        calls.vertex_model().generate_content([Part.from_data(data=PNG_1X1, mime_type="image/png"), "hi"])

        sent = completion_kwargs(mock_revenium_client)
        assert len(sent) == 1
        assert "has_vision_content" not in sent[0]
        assert (sent[0].get("extra_body") or {}).get("hasVisionContent") is True


class TestProviderErrors:
    def test_a_provider_error_reaches_the_caller_unchanged(self, mock_revenium_client):
        from google.api_core.exceptions import ResourceExhausted

        class RateLimitedPredictionClient:
            def generate_content(self, *args, **kwargs):
                raise ResourceExhausted("quota")

        model = calls.vertex_model()
        calls.cache_vertex_prediction_clients(model, RateLimitedPredictionClient())

        with pytest.raises(ResourceExhausted):
            model.generate_content("hi")


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


def _vertex_pass_through_wrappers():
    from revenium_middleware.google.vertex_ai import middleware as vertex_middleware
    return {
        "generate_content": vertex_middleware.generate_content_wrapper_impl,
        "generate_content_async": vertex_middleware.generate_content_async_wrapper_impl,
        "send_message": vertex_middleware.send_message_wrapper_impl,
        "get_embeddings": vertex_middleware._sync_metered(vertex_middleware._EmbeddingCall),
        "generate_images": vertex_middleware._imagen_generate_images_impl,
        "edit_image": vertex_middleware._imagen_edit_image_impl,
    }


@pytest.mark.parametrize("usage_metadata", [{"organizationName": "AcmeCorp"}, None], ids=["metadata", "none"])
@pytest.mark.parametrize("name", list(_vertex_pass_through_wrappers()))
def test_selective_pass_through_drops_usage_metadata(name, usage_metadata, monkeypatch):
    """Selective metering on, outside @revenium_meter: not metered, and the
    Revenium-only keyword never reaches the Vertex AI SDK (BACK-3991)."""
    monkeypatch.setenv("REVENIUM_SELECTIVE_METERING", "true")
    seen = []

    def wrapped(*_args, **kwargs):
        seen.append(kwargs)
        return "provider response"

    wrapper = _vertex_pass_through_wrappers()[name]

    async def awaited(*args, **kwargs):
        return wrapped(*args, **kwargs)

    if name.endswith("_async"):
        result = asyncio.run(wrapper(awaited, object(), ("hi",), {"usage_metadata": usage_metadata}))
    else:
        result = wrapper(wrapped, object(), ("hi",), {"usage_metadata": usage_metadata})
    assert result == "provider response"
    assert seen == [{}]


# --- A scripted model: each request the SDK sends gets the next scripted response ---

def _answer(input_tokens, output_tokens, text="It is sunny.", finish="STOP"):
    return {"usage": (input_tokens, output_tokens), "parts": [{"text": text}], "finish": finish}


def _tool_call(input_tokens, output_tokens, city, name="get_weather"):
    return {"usage": (input_tokens, output_tokens),
            "parts": [{"function_call": {"name": name, "args": {"city": city}}}], "finish": "STOP"}


def _gapic_response(module_path, spec):
    types = importlib.import_module(calls._VERTEX_GAPIC_TYPES[module_path])
    body = {"model_version": calls.GEMINI_MODEL}
    if spec.get("usage"):
        input_tokens, output_tokens = spec["usage"]
        body["usage_metadata"] = {"prompt_token_count": input_tokens, "candidates_token_count": output_tokens,
                                  "total_token_count": input_tokens + output_tokens}
    if spec.get("blocked"):
        body["prompt_feedback"] = {"block_reason": "SAFETY"}
    else:
        candidate = {"index": 0, "finish_reason": spec.get("finish", "FINISH_REASON_UNSPECIFIED")}
        if spec.get("parts") is not None:
            candidate["content"] = {"role": "model", "parts": spec["parts"]}
        body["candidates"] = [candidate]
    return types.GenerateContentResponse(body)


class _ScriptedPredictionClient:
    """Serves the next scripted request per call; a list in the script is one streamed request's chunks."""

    def __init__(self, module_path, script):
        self._module_path = module_path
        self._script = list(script)
        self.served = []

    def next_request(self):
        spec = self._script.pop(0)
        self.served.append(spec)
        chunks = spec if isinstance(spec, list) else [spec]
        return [_gapic_response(self._module_path, chunk) for chunk in chunks]

    def generate_content(self, request, **kwargs):
        return self.next_request()[-1]

    def stream_generate_content(self, request, **kwargs):
        return iter(self.next_request())


class _ScriptedPredictionAsyncClient:
    def __init__(self, scripted):
        self._scripted = scripted

    async def generate_content(self, request, **kwargs):
        return self._scripted.generate_content(request)

    async def stream_generate_content(self, request, **kwargs):
        return calls._async_iterate(self._scripted.next_request())


def _scripted_model(script, module_path=calls.VERTEX_GA, tools=None):
    calls._vertex_init()
    model = importlib.import_module(module_path).GenerativeModel(calls.GEMINI_MODEL, tools=tools)
    client = _ScriptedPredictionClient(module_path, script)
    calls.cache_vertex_prediction_clients(model, client, _ScriptedPredictionAsyncClient(client))
    return model, client


def get_weather(city: str):
    """The weather in a city."""
    return {"forecast": f"sunny in {city}"}


def _tools(*functions):
    from vertexai.preview.generative_models import FunctionDeclaration, Tool
    return [Tool(function_declarations=[FunctionDeclaration.from_func(function) for function in functions])]


def _vision_by_tokens(recorder):
    return sorted(
        ((r["input_token_count"], r["output_token_count"]), (r.get("extra_body") or {}).get("hasVisionContent") is True)
        for r in completion_kwargs(recorder))


def _function_calling_chat(script, *functions, max_calls=5):
    from vertexai.preview.generative_models import AutomaticFunctionCallingResponder
    model, client = _scripted_model(script, calls.VERTEX_PREVIEW, _tools(*(functions or (get_weather,))))
    responder = AutomaticFunctionCallingResponder(max_automatic_function_calls=max_calls)
    return model.start_chat(responder=responder), client


TWO_TOOL_ROUNDS = (_tool_call(1000, 20, "Paris"), _tool_call(1100, 20, "Rome"),
                   _answer(1200, 30, "Paris and Rome are sunny."))


def _token_counts(recorder):
    return sorted((p["input_token_count"], p["output_token_count"]) for p in payloads(recorder))


class TestAutomaticFunctionCallingChatTurn:
    """A chat turn in which Google's SDK runs the tools makes one model request per round (BACK-4314)."""

    def test_every_model_request_of_the_turn_is_its_own_record(self, mock_revenium_client):
        chat, client = _function_calling_chat(TWO_TOOL_ROUNDS)

        response = chat.send_message("Weather in Paris and Rome?")

        assert response.text == "Paris and Rome are sunny."
        assert len(chat.history) == 6
        recorded = payloads(mock_revenium_client)
        assert len(client.served) == len(recorded) == 3
        assert _token_counts(mock_revenium_client) == [(1000, 20), (1100, 20), (1200, 30)]
        assert {p["operation_type"] for p in recorded} == {"CHAT"}
        sent = completion_kwargs(mock_revenium_client)
        assert [record["stop_reason"] for record in sent] == ["END"] * 3
        assert len({record["transaction_id"] for record in sent}) == 3

    def test_each_record_carries_the_prompt_of_its_own_request(self, mock_revenium_client, monkeypatch):
        monkeypatch.setenv("REVENIUM_CAPTURE_PROMPTS", "true")
        chat, _ = _function_calling_chat(TWO_TOOL_ROUNDS)

        chat.send_message("Weather in Paris and Rome?")

        sent = sorted(completion_kwargs(mock_revenium_client), key=lambda record: record["input_token_count"])
        roles = [[turn["role"] for turn in json.loads(record["input_messages"])] for record in sent]
        assert roles == [["user"], ["user", "model", "user"], ["user", "model", "user", "model", "user"]]

    def test_a_turn_whose_model_calls_no_tool_is_one_record(self, mock_revenium_client):
        chat, client = _function_calling_chat([_answer(1000, 30)])

        chat.send_message("Weather in Paris?")

        assert len(client.served) == 1
        assert _token_counts(mock_revenium_client) == [(1000, 30)]

    def test_the_turn_is_checked_against_caps_once(self, mock_revenium_client, monkeypatch):
        from revenium_middleware.google.vertex_ai import middleware as vertex_middleware
        checks = []
        monkeypatch.setattr(vertex_middleware, "check_enforcement", lambda *args, **kwargs: checks.append(kwargs))
        chat, _ = _function_calling_chat(TWO_TOOL_ROUNDS)

        chat.send_message("Weather in Paris and Rome?")

        assert len(checks) == 1
        assert len(payloads(mock_revenium_client)) == 3

    def test_a_tool_that_raises_keeps_the_rounds_before_it(self, mock_revenium_client):
        def get_weather(city: str):
            """The weather in a city."""
            raise KeyError(city)

        chat, client = _function_calling_chat(TWO_TOOL_ROUNDS, get_weather)

        with pytest.raises(RuntimeError) as excinfo:
            chat.send_message("Weather in Paris and Rome?")

        assert type(excinfo.value.__cause__) is KeyError
        assert len(client.served) == 1
        assert _token_counts(mock_revenium_client) == [(1000, 20)]

    def test_a_tool_that_calls_another_model_adds_one_record_under_the_chat(self, mock_revenium_client,
                                                                             monkeypatch):
        from revenium_middleware.google.vertex_ai import middleware as vertex_middleware
        checks = []
        monkeypatch.setattr(vertex_middleware, "check_enforcement", lambda *args, **kwargs: checks.append(kwargs))

        def get_weather(city: str):
            """The weather in a city."""
            calls.vertex_model().generate_content(f"Weather in {city}?")
            return {"forecast": "sunny"}

        chat, _ = _function_calling_chat((_tool_call(1000, 20, "Paris"), _answer(1200, 30)), get_weather)

        chat.send_message("Weather in Paris?", usage_metadata={"trace_id": "chat-trace"})

        assert _token_counts(mock_revenium_client) == [
            (calls.INPUT_TOKENS, calls.OUTPUT_TOKENS), (1000, 20), (1200, 30)]
        assert {record["trace_id"] for record in completion_kwargs(mock_revenium_client)} == {"chat-trace"}
        assert len(checks) == 1

    def test_a_nested_call_with_an_image_is_marked_as_vision_but_the_text_chat_is_not(self, mock_revenium_client):
        from vertexai.generative_models import Part

        def get_weather(city: str):
            """The weather in a city."""
            calls.vertex_model().generate_content(
                [Part.from_data(data=PNG_1X1, mime_type="image/png"), f"Weather in {city}?"])
            return {"forecast": "sunny"}

        chat, _ = _function_calling_chat((_tool_call(1000, 20, "Paris"), _answer(1200, 30)), get_weather)

        chat.send_message("Weather in Paris?")

        assert _vision_by_tokens(mock_revenium_client) == [
            ((calls.INPUT_TOKENS, calls.OUTPUT_TOKENS), True), ((1000, 20), False), ((1200, 30), False)]

    def test_a_nested_text_call_is_not_marked_as_vision_under_an_image_chat(self, mock_revenium_client):
        from vertexai.generative_models import Part

        def get_weather(city: str):
            """The weather in a city."""
            calls.vertex_model().generate_content(f"Weather in {city}?")
            return {"forecast": "sunny"}

        chat, _ = _function_calling_chat((_tool_call(1000, 20, "Paris"), _answer(1200, 30)), get_weather)

        chat.send_message([Part.from_data(data=PNG_1X1, mime_type="image/png"), "Weather in Paris?"])

        assert _vision_by_tokens(mock_revenium_client) == [
            ((calls.INPUT_TOKENS, calls.OUTPUT_TOKENS), False), ((1000, 20), True), ((1200, 30), True)]

    def test_one_failed_record_never_stops_the_others_or_reaches_the_caller(self, mock_revenium_client,
                                                                            monkeypatch):
        from revenium_middleware.google.vertex_ai import middleware as vertex_middleware
        extract = vertex_middleware.extract_vertex_ai_usage_data
        attempts = []

        def failing_once(*args, **kwargs):
            attempts.append(1)
            if len(attempts) == 1:
                raise RuntimeError("extraction failed")
            return extract(*args, **kwargs)

        monkeypatch.setattr(vertex_middleware, "extract_vertex_ai_usage_data", failing_once)
        chat, _ = _function_calling_chat(TWO_TOOL_ROUNDS)

        response = chat.send_message("Weather in Paris and Rome?")

        assert response.text == "Paris and Rome are sunny."
        assert len(payloads(mock_revenium_client)) == 2

    def test_two_chats_in_two_threads_each_record_their_own_rounds(self, mock_revenium_client):
        def run(trace_id):
            chat, _ = _function_calling_chat(TWO_TOOL_ROUNDS)
            chat.send_message("Weather in Paris and Rome?", usage_metadata={"trace_id": trace_id})

        threads = [threading.Thread(target=run, args=(trace_id,)) for trace_id in ("trace-a", "trace-b")]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        sent = completion_kwargs(mock_revenium_client)
        for trace_id in ("trace-a", "trace-b"):
            assert sorted(record["input_token_count"] for record in sent if record["trace_id"] == trace_id) == [
                1000, 1100, 1200]

    def test_concurrent_async_calls_each_carry_their_own_metadata(self, mock_revenium_client):
        model, _ = _scripted_model([_answer(1000, 30), _answer(2000, 40)])

        async def run():
            await asyncio.gather(model.generate_content_async("a", usage_metadata={"trace_id": "trace-a"}),
                                 model.generate_content_async("b", usage_metadata={"trace_id": "trace-b"}))

        asyncio.run(run())

        sent = completion_kwargs(mock_revenium_client)
        assert len(sent) == 2
        by_trace = {record["trace_id"]: record["input_token_count"] for record in sent}
        assert set(by_trace) == {"trace-a", "trace-b"}
        assert sorted(by_trace.values()) == [1000, 2000]

    def test_an_async_turn_that_asks_for_a_tool_is_one_request_and_one_record(self, mock_revenium_client):
        chat, client = _function_calling_chat(TWO_TOOL_ROUNDS)

        asyncio.run(chat.send_message_async("Weather in Paris and Rome?"))

        assert len(client.served) == 1
        assert _token_counts(mock_revenium_client) == [(1000, 20)]


def _send(chat, mode, message="Tell me something."):
    if mode == "sync":
        return chat.send_message(message)
    return asyncio.run(chat.send_message_async(message))


def _drain_stream(chat, mode, message="Tell me something."):
    """Every chunk the stream yields; the stream wrapper is dropped and collected before returning or raising."""
    if mode == "sync":
        stream = chat.send_message(message, stream=True)
        try:
            return list(stream)
        finally:
            del stream
            gc.collect()

    async def run():
        stream = await chat.send_message_async(message, stream=True)
        try:
            return await drain(stream)
        finally:
            del stream
            gc.collect()
    return asyncio.run(run())


class TestChatTurnRejectedAfterGoogleAnswered:
    """Google bills an answer the SDK then rejects, so the record is sent before the error reaches the caller
    (BACK-4315)."""

    @pytest.mark.parametrize("mode", ["sync", "async"])
    @pytest.mark.parametrize("finish, stop_reason", [
        ("SAFETY", "ERROR"), ("MAX_TOKENS", "TOKEN_LIMIT"), ("PROHIBITED_CONTENT", "ERROR")])
    def test_a_rejected_answer_is_recorded_and_the_error_still_reaches_the_caller(
            self, mock_revenium_client, finish, stop_reason, mode):
        from vertexai.generative_models import ResponseValidationError
        model, _ = _scripted_model([_answer(1000, 30, "partial", finish=finish)])
        chat = model.start_chat()

        with pytest.raises(ResponseValidationError) as excinfo:
            _send(chat, mode)

        assert type(excinfo.value) is ResponseValidationError
        assert chat.history == []
        assert _token_counts(mock_revenium_client) == [(1000, 30)]
        assert completion_kwargs(mock_revenium_client)[0]["stop_reason"] == stop_reason

    def test_a_function_call_limit_records_every_request_it_made(self, mock_revenium_client):
        chat, client = _function_calling_chat([_tool_call(1000, 20, "Paris")] * 5, max_calls=2)

        with pytest.raises(RuntimeError, match="maximum number of automatic function calls") as excinfo:
            chat.send_message("Weather everywhere?")

        assert type(excinfo.value) is RuntimeError
        recorded = payloads(mock_revenium_client)
        assert len(client.served) == len(recorded) == 3
        assert sum(p["input_token_count"] for p in recorded) == 3000

    def test_a_metering_failure_never_replaces_googles_error(self, mock_revenium_client, monkeypatch):
        from revenium_middleware.google.vertex_ai import middleware as vertex_middleware
        from vertexai.generative_models import ResponseValidationError

        def failing(*args, **kwargs):
            raise RuntimeError("extraction failed")

        monkeypatch.setattr(vertex_middleware, "extract_vertex_ai_usage_data", failing)
        model, _ = _scripted_model([_answer(1000, 30, "partial", finish="SAFETY")])

        with pytest.raises(ResponseValidationError) as excinfo:
            model.start_chat().send_message("Tell me something.")

        assert type(excinfo.value) is ResponseValidationError

    @pytest.mark.parametrize("mode", ["sync", "async"])
    @pytest.mark.parametrize("chunks, tokens, stop_reason", [
        pytest.param([_answer(1000, 30, "partial", finish="SAFETY")], (1000, 30), "ERROR", id="one_chunk"),
        pytest.param([{"parts": [{"text": "Once upon "}]}, _answer(1000, 30, "a time", finish="SAFETY")],
                     (1000, 30), "ERROR", id="two_chunks"),
        pytest.param([{"usage": (1000, 30), "finish": "SAFETY"}], (1000, 30), "ERROR", id="safety_without_content"),
        pytest.param([{"usage": (1000, 0), "blocked": True}], (1000, 0), None, id="prompt_blocked"),
    ])
    def test_a_rejected_streamed_answer_records_the_tokens_the_stream_reported(
            self, mock_revenium_client, chunks, tokens, stop_reason, mode):
        from vertexai.generative_models import ResponseValidationError
        model, _ = _scripted_model([chunks])

        with pytest.raises(ResponseValidationError) as excinfo:
            _drain_stream(model.start_chat(), mode)

        assert type(excinfo.value) is ResponseValidationError
        assert _token_counts(mock_revenium_client) == [tokens]
        sent = completion_kwargs(mock_revenium_client)[0]
        assert sent["is_streamed"] is True
        if stop_reason is not None:
            assert sent["stop_reason"] == stop_reason

    @pytest.mark.parametrize("entry_point", ["generate_content", "send_message"])
    def test_a_streamed_function_call_reaches_the_caller_and_is_recorded_once(self, mock_revenium_client,
                                                                              entry_point):
        script = [[_tool_call(1000, 20, "Paris")]]
        if entry_point == "generate_content":
            model, client = _scripted_model(script, calls.VERTEX_PREVIEW, _tools(get_weather))
            stream = model.generate_content("Weather in Paris?", stream=True)
        else:
            chat, client = _function_calling_chat(script)
            stream = chat.send_message("Weather in Paris?", stream=True)

        chunks = list(stream)

        assert [call.name for call in chunks[0].candidates[0].function_calls] == ["get_weather"]
        assert len(client.served) == 1
        assert _token_counts(mock_revenium_client) == [(1000, 20)]
        assert completion_kwargs(mock_revenium_client)[0]["is_streamed"] is True


# --- Images a Gemini image model makes ---

def _reports_image_tokens():
    from google.cloud.aiplatform_v1.types import GenerateContentResponse
    return "candidates_tokens_details" in GenerateContentResponse.UsageMetadata.meta.fields


def _image_chunk(parts, usage=True):
    from google.cloud.aiplatform_v1 import types
    body = {"model_version": calls.GEMINI_MODEL,
            "candidates": [{"index": 0, "content": {"role": "model", "parts": parts}, "finish_reason": "STOP"}]}
    if usage:
        body["usage_metadata"] = {"prompt_token_count": 10, "candidates_token_count": 1413, "total_token_count": 1423,
                                  "candidates_tokens_details": [{"modality": "IMAGE", "token_count": 1120}]}
    return types.GenerateContentResponse(body)


def _vertex_image_part(thought=False):
    part = {"inline_data": {"mime_type": "image/png", "data": PNG_1X1}}
    if thought:
        part["thought"] = True
    return part


class _ImagePredictionClient:
    def __init__(self, chunks, then_raise=None):
        self._chunks = chunks
        self._then_raise = then_raise

    def generate_content(self, request, **kwargs):
        return self._chunks[-1]

    def stream_generate_content(self, request, **kwargs):
        yield from self._chunks
        if self._then_raise is not None:
            raise self._then_raise


class _ImagePredictionAsyncClient:
    def __init__(self, chunks):
        self._chunks = chunks

    async def generate_content(self, request, **kwargs):
        return self._chunks[-1]

    async def stream_generate_content(self, request, **kwargs):
        return calls._async_iterate(self._chunks)


def _image_model(chunks, then_raise=None):
    calls._vertex_init()
    model = importlib.import_module(calls.VERTEX_GA).GenerativeModel(calls.GEMINI_MODEL)
    calls.cache_vertex_prediction_clients(
        model, _ImagePredictionClient(chunks, then_raise), _ImagePredictionAsyncClient(chunks))
    return model


def _assert_vertex_split(recorder, image_count=1):
    completions, images = completion_kwargs(recorder), image_kwargs(recorder)
    assert len(completions) == 1, completions
    assert len(images) == 1, images
    completion, image = completions[0], images[0]
    assert (completion["input_token_count"], completion["output_token_count"], completion["total_token_count"]) \
        == (10, 293, 303)
    assert completion["operation_type"] == "CHAT"
    assert image["actual_image_count"] == image_count
    assert image["transaction_id"] == completion["transaction_id"] + "-image"
    return completion


@pytest.mark.skipif("not _reports_image_tokens()",
                    reason="this google-cloud-aiplatform's UsageMetadata has no candidates_tokens_details")
class TestVertexImageAnswers:
    """An image a Gemini image model makes through vertexai is metered as an image record (BACK-4334)."""

    def test_a_vertex_image_answer_sends_the_image_as_an_image_record(self, mock_revenium_client):
        _image_model([_image_chunk([_vertex_image_part()])]).generate_content("A red circle.")

        _assert_vertex_split(mock_revenium_client)

    def test_a_streamed_vertex_image_answer_sends_the_image_as_an_image_record(self, mock_revenium_client):
        chunks = [_image_chunk([_vertex_image_part()], usage=False), _image_chunk([{"text": "done"}])]

        async def run():
            return await drain(await _image_model(chunks).generate_content_async("A red circle.", stream=True))

        asyncio.run(run())

        assert _assert_vertex_split(mock_revenium_client)["is_streamed"] is True

    def test_a_vertex_thought_image_is_not_counted(self, mock_revenium_client):
        parts = [_vertex_image_part(thought=True), _vertex_image_part()]
        _image_model([_image_chunk(parts)]).generate_content("A red circle.")

        _assert_vertex_split(mock_revenium_client, image_count=1)

    def test_a_vertex_stream_that_fails_after_an_image_chunk_sends_no_image_record(self, mock_revenium_client):
        model = _image_model([_image_chunk([_vertex_image_part()])], then_raise=ConnectionError("dropped"))

        with pytest.raises(ConnectionError):
            list(model.generate_content("A red circle.", stream=True))

        completions = completion_kwargs(mock_revenium_client)
        assert [(c["output_token_count"], c["total_token_count"]) for c in completions] == [(1413, 1423)]
        assert image_kwargs(mock_revenium_client) == []
