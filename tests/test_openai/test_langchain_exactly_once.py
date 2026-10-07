"""Every LangChain model call is metered exactly once, under the right
provider (BACK-3582, and BACK-3583's callback-plus-middleware criterion), and
the one record keeps the attribution given to the callback (BACK-3913).

Real langchain-openai and langchain-anthropic run over stubbed transports, so
no call leaves the process: OpenAI traffic is answered by a patched httpx
transport and Anthropic traffic by an httpx2 mock client. Every stub response
carries a fresh provider id, as the live APIs do.
"""
import asyncio
import contextvars
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
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage  # noqa: E402
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult  # noqa: E402
from pydantic import BaseModel  # noqa: E402

import revenium_middleware.openai  # noqa: E402,F401  installs the OpenAI patches
import revenium_middleware.anthropic  # noqa: E402,F401  installs the Anthropic patches
from revenium_middleware._core.call_ownership import (  # noqa: E402
    ANTHROPIC, OLLAMA, OPENAI, claim_call_for_transport, claimed_by_transport, publish_callback_metadata,
    reset_claimed_response_ids, take_callback_metadata, transport_claim_mark, with_callback_metadata,
    _ClaimedResponseIds, _published_in_context,
)
from revenium_middleware.openai import middleware as openai_middleware  # noqa: E402
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
        self.content = "hi"
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
                "choices": [{"index": 0, "message": {"role": "assistant", "content": self.content},
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
    publications = _published_in_context.set(())
    yield
    _published_in_context.reset(publications)
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


CALLBACK_METADATA = {
    "subscriber": {"id": "sub-callback", "email": "callback@example.com"},
    "organizationName": "org-callback",
    "traceId": "trace-callback",
    "taskType": "task-callback",
    "prompt_id": "prompt-callback",
}


def attributed_handler():
    return ReveniumCallbackHandler(usage_metadata=CALLBACK_METADATA)


def assert_callback_attribution(record, *, trace_id="trace-callback"):
    assert record["subscriber"] == CALLBACK_METADATA["subscriber"]
    assert record["organization_name"] == "org-callback"
    assert record["trace_id"] == trace_id
    assert record["task_type"] == "task-callback"
    assert record["prompt_id"] == "prompt-callback"


def assert_no_callback_attribution(record):
    assert not record.get("subscriber")
    assert not record.get("organization_name")
    assert not record.get("trace_id")


def attributed_openai(**kwargs):
    return chat_openai(callbacks=[attributed_handler()], **kwargs)


class Answer(BaseModel):
    answer: str


class TestCallbackMetadataOnTheTransportRecord:
    @pytest.mark.parametrize("call, build, streamed", [
        pytest.param(invoke, attributed_openai, False, id="invoke"),
        pytest.param(ainvoke, attributed_openai, False, id="ainvoke"),
        pytest.param(stream, attributed_openai, True, id="stream"),
        pytest.param(astream, attributed_openai, True, id="astream"),
        pytest.param(ainvoke, lambda: attributed_openai(streaming=True), True, id="ainvoke-streaming-true"),
        pytest.param(invoke, lambda: attributed_openai(include_response_headers=True), False,
                     id="invoke-with-raw-response"),
        pytest.param(ainvoke, lambda: attributed_openai(include_response_headers=True), False,
                     id="ainvoke-with-raw-response"),
        pytest.param(invoke, lambda: wrap(chat_openai(), usage_metadata=CALLBACK_METADATA), False,
                     id="invoke-via-wrap"),
    ])
    def test_chat_openai(self, openai_stub, records, call, build, streamed):
        assert call(build()) == "hi"
        assert_callback_attribution(
            assert_one_record(openai_stub, records, provider="OPENAI", streamed=streamed, transport=True))

    def test_callbacks_in_config(self, openai_stub, records):
        assert chat_openai().invoke("hi", config={"callbacks": [attributed_handler()]}).text == "hi"
        assert_callback_attribution(
            assert_one_record(openai_stub, records, provider="OPENAI", streamed=False, transport=True))

    @pytest.mark.parametrize("handlers", [
        pytest.param(lambda: [attributed_handler(), ReveniumCallbackHandler()], id="attributed-first"),
        pytest.param(lambda: [ReveniumCallbackHandler(), attributed_handler()], id="attributed-last"),
    ])
    @pytest.mark.parametrize("call", [invoke, ainvoke])
    def test_a_handler_without_metadata_does_not_hide_one_with_it(self, openai_stub, records, call, handlers):
        assert call(chat_openai(callbacks=handlers())) == "hi"
        assert_callback_attribution(
            assert_one_record(openai_stub, records, provider="OPENAI", streamed=False, transport=True))

    @pytest.mark.parametrize("run", [
        pytest.param(lambda model: model.invoke("hi"), id="invoke"),
        pytest.param(lambda model: asyncio.run(model.ainvoke("hi")), id="ainvoke"),
    ])
    def test_chat_openai_structured_output_through_parse(self, openai_stub, records, run):
        openai_stub.content = '{"answer": "hi"}'
        model = attributed_openai().with_structured_output(Answer, method="json_schema")
        assert run(model) == Answer(answer="hi")
        assert_callback_attribution(
            assert_one_record(openai_stub, records, provider="OPENAI", streamed=False, transport=True))

    @pytest.mark.parametrize("chat_kwargs", [
        pytest.param({}, id="messages"),
        pytest.param({"betas": ["token-efficient-tools-2025-02-19"]}, id="beta-messages"),
    ])
    @pytest.mark.parametrize("call, streamed", [(invoke, False), (ainvoke, False), (stream, True), (astream, True)])
    def test_chat_anthropic(self, anthropic_stub, records, call, streamed, chat_kwargs):
        model = anthropic_stub.chat(callbacks=[attributed_handler()], **chat_kwargs)
        assert call(model) == "hi"
        assert_callback_attribution(
            assert_one_record(anthropic_stub, records, provider="ANTHROPIC", streamed=streamed, transport=True))

    @pytest.mark.usefixtures("without_anthropic_middleware")
    def test_the_callback_record_still_carries_it_when_no_transport_meters(self, anthropic_stub, records):
        assert invoke(anthropic_stub.chat(callbacks=[attributed_handler()])) == "hi"
        assert_callback_attribution(
            assert_one_record(anthropic_stub, records, provider="ANTHROPIC", streamed=False, transport=False))

    @pytest.mark.parametrize("call", [invoke, ainvoke, stream, astream])
    def test_metadata_on_the_openai_call_wins(self, openai_stub, records, call):
        model = attributed_openai().bind(usage_metadata={"trace_id": "trace-call"})
        assert call(model) == "hi"
        record = assert_one_record(openai_stub, records, provider="OPENAI",
                                   streamed=call in (stream, astream), transport=True)
        assert_callback_attribution(record, trace_id="trace-call")

    @pytest.mark.parametrize("call", [invoke, ainvoke, stream, astream])
    def test_metadata_on_the_anthropic_call_wins(self, anthropic_stub, records, call):
        model = anthropic_stub.chat(callbacks=[attributed_handler()]).bind(usage_metadata={"trace_id": "trace-call"})
        assert call(model) == "hi"
        record = assert_one_record(anthropic_stub, records, provider="ANTHROPIC",
                                   streamed=call in (stream, astream), transport=True)
        assert_callback_attribution(record, trace_id="trace-call")

    def test_a_later_call_without_the_callback_does_not_inherit_it(self, openai_stub, records):
        assert invoke(attributed_openai()) == "hi"
        assert invoke(chat_openai()) == "hi"
        assert len(records) == 2
        assert_callback_attribution(records[0])
        assert_no_callback_attribution(records[1])

    def test_a_later_async_call_in_the_same_task_does_not_inherit_it(self, openai_stub, records):
        async def calls():
            await attributed_openai().ainvoke("hi")
            async for _ in attributed_openai().astream("hi"):
                pass
            await chat_openai().ainvoke("hi")

        asyncio.run(calls())
        assert len(records) == 3
        assert_callback_attribution(records[0])
        assert_callback_attribution(records[1])
        assert_no_callback_attribution(records[2])

    def test_an_anthropic_callback_does_not_attribute_an_openai_call(self, openai_stub, anthropic_stub, records):
        anthropic_model = anthropic_stub.chat(callbacks=[attributed_handler()])
        for chunk in anthropic_model.stream("hi"):
            if chunk.text:
                assert invoke(chat_openai()) == "hi"
        by_provider = {record["provider"]: record for record in records}
        assert_callback_attribution(by_provider["ANTHROPIC"])
        assert_no_callback_attribution(by_provider["OPENAI"])


class TestCallbackMetadataPublication:
    def test_nothing_is_published_outside_a_callback_run(self):
        assert take_callback_metadata(OPENAI) == {}

    def test_a_provider_in_scope_takes_it(self):
        publish_callback_metadata({"traceId": "t"}, frozenset({OLLAMA, OPENAI}))
        assert take_callback_metadata(ANTHROPIC) == {}
        assert take_callback_metadata(OLLAMA) == {"traceId": "t"}

    def test_an_unknown_scope_serves_every_provider(self):
        publish_callback_metadata({"traceId": "t"})
        assert take_callback_metadata(ANTHROPIC) == {"traceId": "t"}

    def test_it_is_taken_once(self):
        publish_callback_metadata({"traceId": "t"}, frozenset({OLLAMA, OPENAI}), "run")
        assert take_callback_metadata(OPENAI) == {"traceId": "t"}
        assert take_callback_metadata(OPENAI) == {}
        assert take_callback_metadata(OLLAMA) == {}

    def test_taking_it_closes_every_handler_publication_for_the_call(self):
        publish_callback_metadata({"traceId": "t"}, frozenset({OPENAI}), "run")
        publish_callback_metadata({}, frozenset({OPENAI}), "run")
        assert take_callback_metadata(OPENAI) == {"traceId": "t"}
        assert take_callback_metadata(OPENAI) == {}

    def test_an_empty_publication_for_another_call_takes_nothing_from_it(self):
        publish_callback_metadata({"traceId": "outer"}, frozenset({OPENAI}), "outer")
        publish_callback_metadata({}, frozenset({OPENAI}), "inner")
        assert take_callback_metadata(OPENAI) == {}
        assert take_callback_metadata(OPENAI) == {"traceId": "outer"}

    def test_the_latest_open_call_is_taken_first(self):
        publish_callback_metadata({"traceId": "first"}, frozenset({OPENAI}), "first")
        publish_callback_metadata({"traceId": "second"}, frozenset({OPENAI}), "second")
        assert take_callback_metadata(OPENAI) == {"traceId": "second"}
        assert take_callback_metadata(OPENAI) == {"traceId": "first"}
        assert take_callback_metadata(OPENAI) == {}

    def test_a_withdrawn_publication_is_not_taken(self):
        publish_callback_metadata({"traceId": "t"}, frozenset({OPENAI})).withdraw()
        assert take_callback_metadata(OPENAI) == {}

    def test_withdrawing_from_another_context_still_closes_it(self):
        publication = publish_callback_metadata({"traceId": "t"}, frozenset({OPENAI}))
        contextvars.copy_context().run(publication.withdraw)
        assert take_callback_metadata(OPENAI) == {}

    def test_taking_it_in_a_child_context_closes_it_for_the_parent(self):
        publish_callback_metadata({"traceId": "t"}, frozenset({OPENAI}))
        assert contextvars.copy_context().run(take_callback_metadata, OPENAI) == {"traceId": "t"}
        assert take_callback_metadata(OPENAI) == {}

    def test_a_publication_does_not_leak_out_of_its_context(self):
        contextvars.copy_context().run(publish_callback_metadata, {"traceId": "t"})
        assert take_callback_metadata(OPENAI) == {}

    @pytest.mark.parametrize("call_metadata, expected", [
        ({"trace_id": "call"}, {"organizationName": "org", "trace_id": "call"}),
        (None, {"traceId": "callback", "organizationName": "org"}),
    ])
    def test_the_call_metadata_wins_under_either_spelling(self, call_metadata, expected):
        publish_callback_metadata({"traceId": "callback", "organizationName": "org"})
        assert with_callback_metadata(OPENAI, call_metadata) == expected

    def test_the_responses_api_call_reads_it(self):
        publish_callback_metadata({"traceId": "t"}, frozenset({OPENAI}))
        call = openai_middleware._begin_responses_call(MagicMock(), {"usage_metadata": {"taskType": "x"}})
        assert call.usage_metadata["traceId"] == "t"
        assert call.usage_metadata["taskType"] == "x"


def plain_openai_call():
    import openai
    openai.OpenAI(api_key="sk-test", max_retries=0).chat.completions.create(
        model=OPENAI_ALIAS, messages=[{"role": "user", "content": "hi"}])


async def plain_async_openai_call():
    import openai
    await openai.AsyncOpenAI(api_key="sk-test", max_retries=0).chat.completions.create(
        model=OPENAI_ALIAS, messages=[{"role": "user", "content": "hi"}])


def plain_anthropic_call(stub):
    client = anthropic.Anthropic(http_client=httpx2.Client(transport=httpx2.MockTransport(stub._respond)),
                                 api_key="sk-ant-test", base_url=f"http://{ANTHROPIC_HOST}", max_retries=0)
    client.messages.create(model=ANTHROPIC_ALIAS, max_tokens=16, messages=[{"role": "user", "content": "hi"}])


@pytest.fixture
def budget_checks():
    checked = []
    with patch("revenium_middleware.openai.middleware.check_enforcement",
               side_effect=lambda usage_metadata=None: checked.append(dict(usage_metadata or {}))):
        yield checked


def assert_no_callback_fields(usage_metadata):
    assert not set(CALLBACK_METADATA) & {key for key, value in usage_metadata.items() if value}


class StreamingChatOllamaStandIn(ChatOllamaStandIn):
    """A streaming callback-only Ollama model: its client is one the SDK does
    not meter, so no transport call ever takes its publication."""

    def _stream(self, messages, stop=None, run_manager=None, **kwargs):
        for text in ("h", "i"):
            chunk = ChatGenerationChunk(message=AIMessageChunk(content=text))
            if run_manager:
                run_manager.on_llm_new_token(text, chunk=chunk)
            yield chunk

    async def _astream(self, messages, stop=None, run_manager=None, **kwargs):
        for text in ("h", "i"):
            chunk = ChatGenerationChunk(message=AIMessageChunk(content=text))
            if run_manager:
                await run_manager.on_llm_new_token(text, chunk=chunk)
            yield chunk


class TestCallbackAttributionStaysWithItsModelCall:
    def test_a_plain_openai_call_between_stream_chunks_gets_none_of_it(self, openai_stub, records, budget_checks):
        for chunk in attributed_openai().stream("hi"):
            if chunk.text:
                plain_openai_call()
        stream_id, plain_id = openai_stub.ids
        by_id = {record["transaction_id"]: record for record in records}
        assert_callback_attribution(by_id[stream_id])
        assert_no_callback_attribution(by_id[plain_id])
        stream_check, plain_check = budget_checks
        assert stream_check["organizationName"] == "org-callback"
        assert_no_callback_fields(plain_check)

    def test_a_plain_async_openai_call_between_astream_chunks_gets_none_of_it(
            self, openai_stub, records, budget_checks):
        async def calls():
            async for chunk in attributed_openai().astream("hi"):
                if chunk.text:
                    await plain_async_openai_call()

        asyncio.run(calls())
        stream_id, plain_id = openai_stub.ids
        by_id = {record["transaction_id"]: record for record in records}
        assert_callback_attribution(by_id[stream_id])
        assert_no_callback_attribution(by_id[plain_id])
        stream_check, plain_check = budget_checks
        assert stream_check["organizationName"] == "org-callback"
        assert_no_callback_fields(plain_check)

    @pytest.mark.parametrize("chat_kwargs", [
        pytest.param({}, id="messages"),
        pytest.param({"betas": ["token-efficient-tools-2025-02-19"]}, id="beta-messages"),
    ])
    def test_a_plain_anthropic_call_between_stream_chunks_gets_none_of_it(
            self, anthropic_stub, records, chat_kwargs):
        for chunk in anthropic_stub.chat(callbacks=[attributed_handler()], **chat_kwargs).stream("hi"):
            if chunk.text:
                plain_anthropic_call(anthropic_stub)
        stream_id, plain_id = anthropic_stub.ids
        by_id = {record["transaction_id"]: record for record in records}
        assert_callback_attribution(by_id[stream_id])
        assert_no_callback_attribution(by_id[plain_id])

    def test_a_stream_no_transport_meters_withdraws_it_at_its_first_token(self, openai_stub, records, budget_checks):
        for chunk in StreamingChatOllamaStandIn(callbacks=[attributed_handler()]).stream("hi"):
            if chunk.text:
                plain_openai_call()
                break
        (plain_id,) = openai_stub.ids
        assert_no_callback_attribution({r["transaction_id"]: r for r in records}[plain_id])
        assert_no_callback_fields(budget_checks[-1])

    def test_an_astream_no_transport_meters_withdraws_it_at_its_first_token(
            self, openai_stub, records, budget_checks):
        async def calls():
            async for chunk in StreamingChatOllamaStandIn(callbacks=[attributed_handler()]).astream("hi"):
                if chunk.text:
                    await plain_async_openai_call()
                    break

        asyncio.run(calls())
        (plain_id,) = openai_stub.ids
        assert_no_callback_attribution({r["transaction_id"]: r for r in records}[plain_id])
        assert_no_callback_fields(budget_checks[-1])

    @pytest.mark.parametrize("generate", [
        pytest.param(lambda model, batch: model.generate(batch), id="generate"),
        pytest.param(lambda model, batch: asyncio.run(model.agenerate(batch)), id="agenerate"),
    ])
    def test_every_call_of_a_batch_keeps_it(self, openai_stub, records, generate):
        generate(attributed_openai(), [[HumanMessage("a")], [HumanMessage("b")]])
        assert len(records) == 2
        for record in records:
            assert_callback_attribution(record)
