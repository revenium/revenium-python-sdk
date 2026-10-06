"""Every LangChain model call is metered exactly once, under the right
provider (BACK-3582, and BACK-3583's callback-plus-middleware criterion).

Real langchain-openai and langchain-anthropic run over stubbed transports, so
no call leaves the process: OpenAI traffic is answered by a patched httpx
transport and Anthropic traffic by an httpx2 mock client. Every stub response
carries a fresh provider id, as the live APIs do.
"""
import asyncio
import json
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock, patch

import pytest

httpx = pytest.importorskip("httpx")
langchain_openai = pytest.importorskip("langchain_openai")
anthropic = pytest.importorskip("anthropic")
httpx2 = pytest.importorskip("httpx2")
langchain_anthropic = pytest.importorskip("langchain_anthropic")

from langchain_core.language_models.chat_models import BaseChatModel  # noqa: E402
from langchain_core.messages import AIMessage  # noqa: E402
from langchain_core.outputs import ChatGeneration, ChatResult  # noqa: E402

import revenium_middleware.openai  # noqa: E402,F401  installs the OpenAI patches
import revenium_middleware.anthropic  # noqa: E402,F401  installs the Anthropic patches
from revenium_middleware._core.call_ownership import (  # noqa: E402
    ANTHROPIC, OLLAMA, OPENAI, claim_call_for_transport, claimed_by_transport, reset_claimed_response_ids,
    transport_claim_mark, _ClaimedResponseIds,
)
from revenium_middleware.openai.langchain import ReveniumCallbackHandler, wrap  # noqa: E402
from revenium_middleware.openai.langchain import _model_call  # noqa: E402

INPUT_TOKENS = 13
OUTPUT_TOKENS = 5
OPENAI_ALIAS = "gpt-4o-mini"
OPENAI_DATED = "gpt-4o-mini-2024-07-18"
ANTHROPIC_ALIAS = "claude-sonnet-4-5"
ANTHROPIC_DATED = "claude-sonnet-4-5-20250929"
OPENAI_HOST = "api.openai.com"
ANTHROPIC_HOST = "anthropic.stub.invalid"


def _sse(events):
    return "".join(f"data: {json.dumps(event)}\n\n" for event in events) + "data: [DONE]\n\n"


class StubOpenAI:
    def __init__(self):
        self.ids = []
        self._lock = threading.Lock()

    def respond(self, request):
        assert request.url.host == OPENAI_HOST, f"unexpected upstream {request.url}"
        body = json.loads(request.content or b"{}")
        response_id = f"chatcmpl-{uuid.uuid4().hex}"
        with self._lock:
            self.ids.append(response_id)
        usage = {"prompt_tokens": INPUT_TOKENS, "completion_tokens": OUTPUT_TOKENS,
                 "total_tokens": INPUT_TOKENS + OUTPUT_TOKENS}
        if not body.get("stream"):
            response = httpx.Response(200, json={
                "id": response_id, "object": "chat.completion", "created": 1, "model": OPENAI_DATED,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"},
                             "finish_reason": "stop"}],
                "usage": usage})
        else:
            base = {"id": response_id, "object": "chat.completion.chunk", "created": 1, "model": OPENAI_DATED}
            events = [
                dict(base, choices=[{"index": 0, "delta": {"role": "assistant", "content": "hi"},
                                     "finish_reason": None}]),
                dict(base, choices=[{"index": 0, "delta": {}, "finish_reason": "stop"}]),
            ]
            if (body.get("stream_options") or {}).get("include_usage"):
                events.append(dict(base, choices=[], usage=usage))
            response = httpx.Response(200, headers={"content-type": "text/event-stream"},
                                      content=_sse(events).encode())
        response.request = request
        return response


class StubAnthropic:
    def __init__(self):
        self.ids = []
        self._lock = threading.Lock()

    def _respond(self, request):
        assert request.url.host == ANTHROPIC_HOST, f"unexpected upstream {request.url}"
        message_id = f"msg_{uuid.uuid4().hex}"
        with self._lock:
            self.ids.append(message_id)
        usage = {"input_tokens": INPUT_TOKENS, "output_tokens": OUTPUT_TOKENS,
                 "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}
        if not json.loads(request.content).get("stream"):
            return httpx2.Response(200, headers={"content-type": "application/json", "request-id": "req_stub"},
                                   json={"id": message_id, "type": "message", "role": "assistant",
                                         "model": ANTHROPIC_DATED, "content": [{"type": "text", "text": "hi"}],
                                         "stop_reason": "end_turn", "stop_sequence": None, "usage": usage})
        events = [
            ("message_start", {"type": "message_start", "message": {
                "id": message_id, "type": "message", "role": "assistant", "model": ANTHROPIC_DATED,
                "content": [], "stop_reason": None, "stop_sequence": None,
                "usage": {**usage, "output_tokens": 1}}}),
            ("content_block_start", {"type": "content_block_start", "index": 0,
                                     "content_block": {"type": "text", "text": ""}}),
            ("content_block_delta", {"type": "content_block_delta", "index": 0,
                                     "delta": {"type": "text_delta", "text": "hi"}}),
            ("content_block_stop", {"type": "content_block_stop", "index": 0}),
            ("message_delta", {"type": "message_delta",
                               "delta": {"stop_reason": "end_turn", "stop_sequence": None}, "usage": usage}),
            ("message_stop", {"type": "message_stop"}),
        ]
        body = "".join(f"event: {name}\ndata: {json.dumps(data)}\n\n" for name, data in events)
        return httpx2.Response(200, headers={"content-type": "text/event-stream", "request-id": "req_stub"},
                               content=body.encode())

    def chat(self, **kwargs):
        model = langchain_anthropic.ChatAnthropic(model=ANTHROPIC_ALIAS, max_tokens=16, max_retries=0,
                                                  api_key="sk-ant-test", base_url=f"http://{ANTHROPIC_HOST}",
                                                  **kwargs)
        client_kwargs = dict(api_key="sk-ant-test", base_url=f"http://{ANTHROPIC_HOST}", max_retries=0)
        model.__dict__["_client"] = anthropic.Anthropic(
            http_client=httpx2.Client(transport=httpx2.MockTransport(self._respond)), **client_kwargs)
        model.__dict__["_async_client"] = anthropic.AsyncAnthropic(
            http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(self._respond)), **client_kwargs)
        return model


def _run_now(coroutine):
    thread = threading.Thread(target=lambda: asyncio.run(coroutine))
    thread.start()
    thread.join(timeout=10)
    return thread


def _run_func_now(coro_func, *args, **kwargs):
    _run_now(coro_func(*args, **kwargs))
    return MagicMock()


@pytest.fixture(autouse=True)
def _isolated(monkeypatch):
    monkeypatch.setenv("REVENIUM_METERING_API_KEY", "hak_test_langchain")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    reset_claimed_response_ids()
    yield
    reset_claimed_response_ids()


@pytest.fixture
def records():
    captured = []
    lock = threading.Lock()

    def record(operation, payload, *args, **kwargs):
        with lock:
            captured.append(payload)
        return MagicMock(id="recorded", status_code=201)

    with patch("revenium_middleware.openai.middleware.submit_ai_event", side_effect=record), \
            patch("revenium_middleware.openai.middleware.run_async_in_thread", side_effect=_run_now), \
            patch("revenium_middleware.anthropic.middleware._get_thread_safe_client", return_value=MagicMock()), \
            patch("revenium_middleware.anthropic.middleware.submit_ai_event", side_effect=record), \
            patch("revenium_middleware.anthropic.middleware._safe_run_async_in_thread",
                  side_effect=_run_func_now):
        yield captured


@pytest.fixture
def openai_stub(monkeypatch):
    stub = StubOpenAI()
    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", lambda self, request: stub.respond(request))

    async def handle_async(self, request):
        return stub.respond(request)

    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", handle_async)
    return stub


@pytest.fixture
def anthropic_stub():
    return StubAnthropic()


@pytest.fixture
def without_anthropic_middleware(monkeypatch):
    """The Anthropic client as it is when revenium_middleware.anthropic was
    never imported (the patches are process-wide once another test imports it)."""
    for cls in (anthropic.resources.messages.messages.Messages,
                anthropic.resources.messages.messages.AsyncMessages):
        for name in ("create", "stream"):
            attribute = cls.__dict__[name]
            monkeypatch.setattr(cls, name, getattr(attribute, "__wrapped__", attribute))


def chat_openai(**kwargs):
    return langchain_openai.ChatOpenAI(model=OPENAI_ALIAS, api_key="sk-test", max_retries=0, **kwargs)


def with_callback(**kwargs):
    return chat_openai(callbacks=[ReveniumCallbackHandler()], **kwargs)


def custom_clients():
    return dict(http_client=httpx.Client(), http_async_client=httpx.AsyncClient())


def stream(model):
    return "".join(chunk.text for chunk in model.stream("hi"))


def astream(model):
    async def call():
        return "".join([chunk.text async for chunk in model.astream("hi")])
    return asyncio.run(call())


def invoke(model):
    return model.invoke("hi").text


def ainvoke(model):
    return asyncio.run(model.ainvoke("hi")).text


def assert_one_record(stub, records, *, provider, streamed, transport):
    assert len(stub.ids) == 1, "the stub must serve exactly one upstream request"
    assert len(records) == 1, records
    record = records[0]
    assert record["input_token_count"] == INPUT_TOKENS
    assert record["output_token_count"] == OUTPUT_TOKENS
    assert record["provider"] == provider
    assert record["is_streamed"] is streamed
    if transport:
        assert record["transaction_id"] == stub.ids[0]
    else:
        assert record["transaction_id"].startswith("langchain-")
    return record


class TestChatOpenAIWithCallback:
    @pytest.mark.parametrize("call, build, streamed", [
        pytest.param(invoke, with_callback, False, id="invoke-callbacks-in-constructor"),
        pytest.param(ainvoke, with_callback, False, id="ainvoke-callbacks-in-constructor"),
        pytest.param(invoke, lambda: wrap(chat_openai()), False, id="invoke-via-wrap"),
        pytest.param(stream, with_callback, True, id="stream-usage-unset-default-client"),
        pytest.param(stream, lambda: with_callback(**custom_clients()), True,
                     id="stream-usage-unset-custom-client"),
        pytest.param(stream, lambda: with_callback(stream_usage=False), True, id="stream-usage-false"),
        pytest.param(astream, with_callback, True, id="astream-usage-unset-default-client"),
        pytest.param(astream, lambda: with_callback(**custom_clients()), True,
                     id="astream-usage-unset-custom-client"),
        pytest.param(astream, lambda: with_callback(stream_usage=False), True, id="astream-usage-false"),
        pytest.param(ainvoke, lambda: with_callback(streaming=True), True, id="ainvoke-streaming-true"),
    ])
    def test_one_record_from_the_transport(self, openai_stub, records, call, build, streamed):
        assert call(build()) == "hi"
        record = assert_one_record(openai_stub, records, provider="OPENAI", streamed=streamed, transport=True)
        assert record["model"] == OPENAI_DATED

    def test_callbacks_in_config(self, openai_stub, records):
        result = chat_openai().invoke("hi", config={"callbacks": [ReveniumCallbackHandler()]})
        assert result.text == "hi"
        assert_one_record(openai_stub, records, provider="OPENAI", streamed=False, transport=True)

    def test_the_stream_usage_settings_under_test_are_the_ones_named(self):
        assert chat_openai().stream_usage is True
        assert chat_openai(**custom_clients()).stream_usage is None


class TestChatOpenAIWithoutCallback:
    @pytest.mark.parametrize("call, streamed", [(invoke, False), (ainvoke, False), (stream, True), (astream, True)])
    def test_transport_record_only(self, openai_stub, records, call, streamed):
        assert call(chat_openai()) == "hi"
        assert_one_record(openai_stub, records, provider="OPENAI", streamed=streamed, transport=True)


@pytest.mark.usefixtures("without_anthropic_middleware")
class TestChatAnthropicCallbackAlone:
    @pytest.mark.parametrize("call, streamed", [(invoke, False), (ainvoke, False), (stream, True), (astream, True)])
    def test_one_anthropic_record_from_the_callback(self, anthropic_stub, records, call, streamed):
        model = anthropic_stub.chat(callbacks=[ReveniumCallbackHandler()])
        assert call(model) == "hi"
        record = assert_one_record(anthropic_stub, records, provider="ANTHROPIC", streamed=streamed,
                                   transport=False)
        assert record["model_source"] == "ANTHROPIC"
        assert record["model"] == ANTHROPIC_DATED


class TestChatAnthropicMiddlewareAndCallback:
    @pytest.mark.parametrize("chat_kwargs", [
        pytest.param({}, id="messages"),
        pytest.param({"betas": ["token-efficient-tools-2025-02-19"]}, id="beta-messages"),
    ])
    @pytest.mark.parametrize("call, streamed", [(invoke, False), (ainvoke, False), (stream, True), (astream, True)])
    def test_one_record_from_the_transport(self, anthropic_stub, records, call, streamed, chat_kwargs):
        model = anthropic_stub.chat(callbacks=[ReveniumCallbackHandler()], **chat_kwargs)
        assert call(model) == "hi"
        record = assert_one_record(anthropic_stub, records, provider="ANTHROPIC", streamed=streamed,
                                   transport=True)
        assert record["model"] == ANTHROPIC_DATED


def _assert_records_match(records, *, transport_ids, callback_count):
    assert sorted(r["transaction_id"] for r in records if not r["transaction_id"].startswith("langchain-")) \
        == sorted(transport_ids)
    assert sum(r["transaction_id"].startswith("langchain-") for r in records) == callback_count
    assert all(r["input_token_count"] == INPUT_TOKENS for r in records)


class TestConcurrentCalls:
    def test_gathered_openai_calls_each_send_one_record(self, openai_stub, records):
        async def main():
            model = with_callback()

            async def drain():
                return "".join([chunk.text async for chunk in model.astream("hi")])

            return await asyncio.gather(model.ainvoke("a"), model.ainvoke("b"), drain(), drain())

        asyncio.run(main())
        assert len(openai_stub.ids) == 4
        assert len(records) == 4
        _assert_records_match(records, transport_ids=openai_stub.ids, callback_count=0)

    @pytest.mark.usefixtures("without_anthropic_middleware")
    def test_gathered_transport_and_callback_only_calls_do_not_mix(self, openai_stub, anthropic_stub, records):
        openai_model = with_callback()
        anthropic_model = anthropic_stub.chat(callbacks=[ReveniumCallbackHandler()])

        async def main():
            async def drain(model):
                return "".join([chunk.text async for chunk in model.astream("hi")])

            return await asyncio.gather(openai_model.ainvoke("a"), anthropic_model.ainvoke("b"),
                                        drain(openai_model), drain(anthropic_model))

        asyncio.run(main())
        assert len(records) == 4
        _assert_records_match(records, transport_ids=openai_stub.ids, callback_count=2)
        assert {r["provider"] for r in records if r["transaction_id"].startswith("langchain-")} == {"ANTHROPIC"}

    @pytest.mark.usefixtures("without_anthropic_middleware")
    def test_threaded_calls_each_send_one_record(self, openai_stub, anthropic_stub, records):
        openai_model = with_callback()
        anthropic_model = anthropic_stub.chat(callbacks=[ReveniumCallbackHandler()])
        calls = [lambda: invoke(openai_model), lambda: stream(openai_model),
                 lambda: invoke(anthropic_model), lambda: stream(anthropic_model)] * 3

        with ThreadPoolExecutor(max_workers=6) as pool:
            assert all(result == "hi" for result in pool.map(lambda call: call(), calls))

        assert len(records) == 12
        _assert_records_match(records, transport_ids=openai_stub.ids, callback_count=6)


class TestProviderLabels:
    @pytest.mark.parametrize("serialized, invocation_params, metadata, expected", [
        ({"id": ["langchain", "chat_models", "openai", "ChatOpenAI"]}, {"_type": "openai-chat"},
         {"ls_provider": "openai"}, "OPENAI"),
        ({"id": ["langchain", "chat_models", "azure_openai", "AzureChatOpenAI"]}, {"_type": "azure-openai-chat"},
         {}, "Azure"),
        ({"id": ["langchain", "chat_models", "anthropic", "ChatAnthropic"]}, {"_type": "anthropic-chat"},
         {}, "ANTHROPIC"),
        ({"id": ["langchain_ollama", "chat_models", "ChatOllama"]}, {}, {}, "OLLAMA"),
    ])
    def test_known_model_classes(self, serialized, invocation_params, metadata, expected):
        assert _model_call.provider_metadata_for(serialized, invocation_params, metadata)["provider"] == expected

    def test_unknown_provider_keeps_the_default(self):
        labels = _model_call.provider_metadata_for(
            {"id": ["x", "ChatSomething"]}, {"_type": "something-chat"}, {"ls_provider": "something"})
        assert labels is None


class TestCallOwnership:
    def test_a_claim_in_the_same_context_is_seen(self):
        mark = transport_claim_mark()
        assert not claimed_by_transport(mark, {OPENAI})
        claim_call_for_transport(OPENAI)
        assert claimed_by_transport(mark, {OPENAI})
        assert claimed_by_transport(mark, None)

    def test_any_provider_in_the_scope_counts(self):
        mark = transport_claim_mark()
        claim_call_for_transport(OPENAI)
        assert claimed_by_transport(mark, {OLLAMA, OPENAI})

    def test_a_claim_for_another_provider_is_not_seen(self):
        mark = transport_claim_mark()
        claim_call_for_transport(OPENAI)
        assert not claimed_by_transport(mark, {ANTHROPIC})

    def test_a_claimed_id_answers_every_asker(self):
        mark = transport_claim_mark()

        async def claim_in_child_task():
            claim_call_for_transport(OPENAI, "chatcmpl-child")

        asyncio.run(claim_in_child_task())
        assert not claimed_by_transport(mark, {OPENAI}, ["chatcmpl-other"])
        assert claimed_by_transport(mark, {OPENAI}, ["chatcmpl-child"])
        assert claimed_by_transport(mark, {OPENAI}, ["chatcmpl-child"])

    def test_oldest_ids_are_evicted(self):
        ids = _ClaimedResponseIds(capacity=2)
        for response_id in ("a", "b", "c"):
            ids.add(response_id)
        assert not ids.contains_any(["a"])
        assert ids.contains_any(["b"]) and ids.contains_any(["c"])


class TestReviewRoundOne:
    def test_two_handlers_on_one_ainvoke_send_one_record(self, openai_stub, records):
        model = with_callback()
        result = asyncio.run(model.ainvoke("hi", config={"callbacks": [ReveniumCallbackHandler()]}))
        assert result.text == "hi"
        assert_one_record(openai_stub, records, provider="OPENAI", streamed=False, transport=True)

    def test_anthropic_dispatch_failure_keeps_the_callback_record(self, anthropic_stub, records):
        model = anthropic_stub.chat(callbacks=[ReveniumCallbackHandler()])
        with patch("revenium_middleware.anthropic.middleware._safe_run_async_in_thread", return_value=None):
            assert invoke(model) == "hi"
        record = assert_one_record(anthropic_stub, records, provider="ANTHROPIC", streamed=False,
                                   transport=False)
        assert record["model"] == ANTHROPIC_DATED

    def test_openai_dispatch_failure_keeps_the_callback_record(self, openai_stub, records):
        dispatched = []

        def fail_first_dispatch(coroutine):
            if not dispatched:
                dispatched.append(coroutine)
                coroutine.close()
                return None
            return _run_now(coroutine)

        with patch("revenium_middleware.openai.middleware.run_async_in_thread", side_effect=fail_first_dispatch):
            assert invoke(with_callback()) == "hi"
        assert_one_record(openai_stub, records, provider="OPENAI", streamed=False, transport=False)

    @pytest.mark.usefixtures("without_anthropic_middleware")
    def test_openai_call_between_anthropic_chunks_does_not_hide_the_anthropic_record(
            self, openai_stub, anthropic_stub, records):
        anthropic_model = anthropic_stub.chat(callbacks=[ReveniumCallbackHandler()])
        text = ""
        for chunk in anthropic_model.stream("hi"):
            text += chunk.text
            if chunk.text:
                assert invoke(chat_openai()) == "hi"
        assert text == "hi"
        assert len(records) == 2
        by_provider = {record["provider"]: record for record in records}
        assert by_provider["OPENAI"]["transaction_id"] == openai_stub.ids[0]
        assert by_provider["ANTHROPIC"]["transaction_id"].startswith("langchain-")
        assert by_provider["ANTHROPIC"]["is_streamed"] is True

    def test_token_hook_never_raises(self, caplog):
        handler = ReveniumCallbackHandler()
        run_id = uuid.uuid4()
        handler._active_runs[run_id] = {}
        asyncio.run(handler.on_llm_new_token("hi", run_id=run_id))
        assert "on_llm_new_token" in caplog.text


class OllamaCompatibleChatOpenAI(langchain_openai.ChatOpenAI):
    """How an Ollama integration built on ChatOpenAI reports itself: an
    OpenAI-compatible endpoint, served through the OpenAI client."""

    def _get_ls_params(self, stop=None, **kwargs):
        return {**super()._get_ls_params(stop=stop, **kwargs), "ls_provider": "ollama"}


class ChatOllamaStandIn(BaseChatModel):
    """A callback-only Ollama model: no Revenium transport wrap sees its calls."""

    @property
    def _llm_type(self):
        return "chat-ollama"

    def _get_ls_params(self, stop=None, **kwargs):
        return {"ls_provider": "ollama", "ls_model_name": "llama3", "ls_model_type": "chat"}

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        message = AIMessage(content="hi", response_metadata={"model_name": "llama3:8b"},
                            usage_metadata={"input_tokens": INPUT_TOKENS, "output_tokens": OUTPUT_TOKENS,
                                            "total_tokens": INPUT_TOKENS + OUTPUT_TOKENS})
        return ChatResult(generations=[ChatGeneration(message=message)])


class TestOllamaLabelledModels:
    @pytest.mark.parametrize("call, streamed", [(invoke, False), (ainvoke, False), (stream, True), (astream, True)])
    def test_ollama_labelled_openai_client_sends_the_transport_record(self, openai_stub, records, call, streamed):
        model = OllamaCompatibleChatOpenAI(model=OPENAI_ALIAS, api_key="sk-test", max_retries=0,
                                           callbacks=[ReveniumCallbackHandler()])
        assert call(model) == "hi"
        assert_one_record(openai_stub, records, provider="OPENAI", streamed=streamed, transport=True)

    def test_callback_only_ollama_model_sends_its_own_record(self, records):
        model = ChatOllamaStandIn(callbacks=[ReveniumCallbackHandler()])
        assert invoke(model) == "hi"
        assert len(records) == 1
        record = records[0]
        assert record["provider"] == "OLLAMA"
        assert record["transaction_id"].startswith("langchain-")
        assert record["input_token_count"] == INPUT_TOKENS
        assert record["model"] == "llama3:8b"
