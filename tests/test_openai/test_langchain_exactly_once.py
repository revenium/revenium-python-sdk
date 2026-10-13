"""Every LangChain model call is metered exactly once, under the right
provider (BACK-3582, and BACK-3583's callback-plus-middleware criterion), and
the one record keeps the attribution given to the callback (BACK-3913).

Real langchain-openai and langchain-anthropic run over stubbed transports, so
no call leaves the process: OpenAI traffic is answered by patched httpx and
httpx2 transports (openai 3 sends through httpx2) and Anthropic traffic by an
httpx2 mock client. Every stub response carries a fresh provider id, as the
live APIs do.
"""
import asyncio
import contextvars
import gc
import json
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock, patch

import pytest

httpx = pytest.importorskip("httpx")
langchain_openai = pytest.importorskip("langchain_openai")
anthropic = pytest.importorskip("anthropic")
openai_sdk = pytest.importorskip("openai")
httpx2 = pytest.importorskip("httpx2")
langchain_anthropic = pytest.importorskip("langchain_anthropic")

from langchain_core.callbacks import BaseCallbackHandler  # noqa: E402
from langchain_core.language_models import llms as langchain_core_llms  # noqa: E402
from langchain_core.language_models.chat_models import BaseChatModel  # noqa: E402
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage  # noqa: E402
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult, GenerationChunk  # noqa: E402
from langchain_core.prompt_values import ChatPromptValue  # noqa: E402
from pydantic import BaseModel  # noqa: E402

import revenium_middleware.openai  # noqa: E402,F401  installs the OpenAI patches
import revenium_middleware.anthropic  # noqa: E402,F401  installs the Anthropic patches
from revenium_middleware import idempotency_key as idempotency_block  # noqa: E402
from revenium_middleware import revenium_meter, revenium_metadata  # noqa: E402
from revenium_middleware._core.call_ownership import (  # noqa: E402
    ANTHROPIC, OLLAMA, OPENAI, claim_call_for_transport, claimed_by_transport, publish_callback_metadata,
    reset_claimed_response_ids, take_callback_metadata, transport_claim_mark, with_callback_metadata,
    _ClaimedResponseIds, _published_in_context,
)
from revenium_middleware.openai import middleware as openai_middleware  # noqa: E402
from revenium_middleware.openai.langchain import ReveniumCallbackHandler, attach_to, wrap  # noqa: E402
from revenium_middleware.openai.langchain import _model_call  # noqa: E402

from entry_points import mistral  # noqa: E402

from .openai_transport import CHUNKED_STREAMS, answer_every_transport, http_library_of  # noqa: E402
from .responses_stream_events import event_payloads, usage_payload  # noqa: E402
from .test_openai_sdk_entry_points import in_chunks_of  # noqa: E402

INPUT_TOKENS = 13
OUTPUT_TOKENS = 5
OPENAI_ALIAS = "gpt-4o-mini"
OPENAI_DATED = "gpt-4o-mini-2024-07-18"
ANTHROPIC_ALIAS = "claude-sonnet-4-5"
ANTHROPIC_DATED = "claude-sonnet-4-5-20250929"
OPENAI_HOST = "api.openai.com"
ANTHROPIC_HOST = "anthropic.stub.invalid"
AZURE_HOST = "azure.stub.invalid"
TEXT_MODEL = "gpt-3.5-turbo-instruct"


def _sse(events):
    return "".join(f"data: {json.dumps(event)}\n\n" for event in events) + "data: [DONE]\n\n"


class StubOpenAI:
    def __init__(self):
        self.ids = []
        self.content = "hi"
        self.omit_null_fields = False
        self.ignore_include_usage = False
        self.bodies = []
        self.closed_streams = 0
        self._lock = threading.Lock()

    def respond(self, request):
        assert request.url.host in (OPENAI_HOST, AZURE_HOST), f"unexpected upstream {request.url}"
        http = http_library_of(request)
        body = json.loads(request.content or b"{}")
        response_id = f"chatcmpl-{uuid.uuid4().hex}"
        with self._lock:
            self.ids.append(response_id)
            self.bodies.append(body)
        usage = {"prompt_tokens": INPUT_TOKENS, "completion_tokens": OUTPUT_TOKENS,
                 "total_tokens": INPUT_TOKENS + OUTPUT_TOKENS}
        if request.url.path.endswith("/completions") and not request.url.path.endswith("/chat/completions"):
            response = self._text_completion(response_id, body, usage, http)
        elif request.url.path.endswith("/responses"):
            response = self._responses_stream(response_id, http)
        elif not body.get("stream"):
            response = http.Response(200, json={
                "id": response_id, "object": "chat.completion", "created": 1, "model": OPENAI_DATED,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": self.content},
                             "finish_reason": "stop"}],
                "usage": usage})
        else:
            base = {"id": response_id, "object": "chat.completion.chunk", "created": 1, "model": OPENAI_DATED}
            content_choice = {"index": 0, "delta": {"role": "assistant", "content": "hi"}}
            if not self.omit_null_fields:
                content_choice["finish_reason"] = None
            events = [
                dict(base, choices=[content_choice]),
                dict(base, choices=[{"index": 0, "delta": {}, "finish_reason": "stop"}]),
            ]
            if (body.get("stream_options") or {}).get("include_usage"):
                events.append(dict(base, choices=[], usage=usage))
            response = self._event_stream(_sse(events), http)
        response.request = request
        return response

    def _usage_requested(self, body):
        return not self.ignore_include_usage and bool((body.get("stream_options") or {}).get("include_usage"))

    def _text_completion(self, response_id, body, usage, http):
        base = {"id": response_id, "object": "text_completion", "created": 1, "model": TEXT_MODEL}

        def choice(text, finish_reason):
            return {"index": 0, "text": text, "logprobs": None, "finish_reason": finish_reason}

        if not body.get("stream"):
            return http.Response(200, json=dict(base, choices=[choice(self.content, "stop")], usage=usage))
        events = [dict(base, choices=[choice("hi", None)]), dict(base, choices=[choice("", "stop")])]
        if self._usage_requested(body):
            events.append(dict(base, choices=[], usage=usage))
        return self._closing_event_stream(_sse(events), http)

    def _closing_event_stream(self, body, http):
        stub = self

        class ClosingStream(CHUNKED_STREAMS[http]):
            def close(self):
                with stub._lock:
                    stub.closed_streams += 1

        return http.Response(200, headers={"content-type": "text/event-stream"},
                             stream=ClosingStream(in_chunks_of(64)(body.encode())))

    @staticmethod
    def _event_stream(body, http):
        # Served in pieces, as a live stream arrives; a bytes body would be read
        # whole on construction.
        return http.Response(200, headers={"content-type": "text/event-stream"},
                              stream=CHUNKED_STREAMS[http](in_chunks_of(64)(body.encode())))

    def _responses_stream(self, response_id, http):
        events = event_payloads(usage=usage_payload(input_tokens=INPUT_TOKENS, output_tokens=OUTPUT_TOKENS))
        for event in events:
            if "response" in event:
                event["response"].update(id=response_id, model=OPENAI_DATED)
        body = "".join(f"event: {event['type']}\ndata: {json.dumps(event)}\n\n" for event in events)
        return self._event_stream(body, http)


class StubAnthropic:
    def __init__(self, usage=None):
        self.ids = []
        self._lock = threading.Lock()
        self._usage = usage or {"input_tokens": INPUT_TOKENS, "output_tokens": OUTPUT_TOKENS,
                                "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}

    def _respond(self, request):
        assert request.url.host == ANTHROPIC_HOST, f"unexpected upstream {request.url}"
        message_id = f"msg_{uuid.uuid4().hex}"
        with self._lock:
            self.ids.append(message_id)
        usage = self._usage
        # The live API sends the cache-write split on message_start only.
        delta_usage = {name: value for name, value in usage.items() if name != "cache_creation"}
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
                               "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                               "usage": delta_usage}),
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


def _run_now_in_caller_context(coroutine):
    context = contextvars.copy_context()
    thread = threading.Thread(target=lambda: context.run(asyncio.run, coroutine))
    thread.start()
    thread.join(timeout=10)
    return thread


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
    answer_every_transport(monkeypatch, stub.respond)
    return stub


@pytest.fixture
def anthropic_stub():
    return StubAnthropic()


def unwrap_anthropic_middleware(monkeypatch):
    for cls in (anthropic.resources.messages.messages.Messages,
                anthropic.resources.messages.messages.AsyncMessages,
                anthropic.resources.beta.messages.messages.Messages,
                anthropic.resources.beta.messages.messages.AsyncMessages):
        for name in ("create", "stream"):
            attribute = cls.__dict__[name]
            monkeypatch.setattr(cls, name, getattr(attribute, "__wrapped__", attribute))


@pytest.fixture
def without_anthropic_middleware(monkeypatch):
    """The Anthropic client as it is when revenium_middleware.anthropic was
    never imported (the patches are process-wide once another test imports it)."""
    unwrap_anthropic_middleware(monkeypatch)


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


def stream_chunks(model):
    return list(model.stream("hi"))


def astream_chunks(model):
    async def call():
        return [chunk async for chunk in model.astream("hi")]
    return asyncio.run(call())


class TestChatOpenAIWithResponseHeaders:
    """include_response_headers sends LangChain through with_raw_response (BACK-3999)."""

    @pytest.mark.parametrize("collect", [stream_chunks, astream_chunks], ids=["stream", "astream"])
    @pytest.mark.parametrize("build", [
        pytest.param(lambda: with_callback(include_response_headers=True), id="with-callback"),
        pytest.param(lambda: chat_openai(include_response_headers=True), id="without-callback"),
        pytest.param(lambda: with_callback(include_response_headers=True, **custom_clients()),
                     id="with-callback-custom-client"),
    ])
    def test_streamed_answer_is_metered_once(self, openai_stub, records, collect, build):
        chunks = collect(build())

        assert "".join(chunk.text for chunk in chunks) == "hi"
        assert chunks[0].response_metadata["headers"]
        record = assert_one_record(openai_stub, records, provider="OPENAI", streamed=True, transport=True)
        assert record["model"] == OPENAI_DATED

    @pytest.mark.parametrize("collect", [stream_chunks, astream_chunks], ids=["stream", "astream"])
    def test_usage_chunk_stays_hidden_when_stream_usage_is_off(self, openai_stub, records, collect):
        chunks = collect(with_callback(include_response_headers=True, stream_usage=False))

        assert "".join(chunk.text for chunk in chunks) == "hi"
        assert not any(chunk.usage_metadata for chunk in chunks)
        assert_one_record(openai_stub, records, provider="OPENAI", streamed=True, transport=True)

    @pytest.mark.parametrize("collect", [stream_chunks, astream_chunks], ids=["stream", "astream"])
    @pytest.mark.parametrize("settings", [
        pytest.param({}, id="stream-usage-default"),
        pytest.param({"stream_usage": False}, id="stream-usage-false"),
    ])
    def test_a_stream_whose_chunks_omit_null_fields_is_metered_once(self, openai_stub, records, collect, settings):
        """Grok leaves finish_reason out of its content chunks (BACK-4254)."""
        openai_stub.omit_null_fields = True

        chunks = collect(with_callback(include_response_headers=True, **settings))

        assert "".join(chunk.text for chunk in chunks) == "hi"
        assert_one_record(openai_stub, records, provider="OPENAI", streamed=True, transport=True)
        if settings.get("stream_usage") is False:
            assert not any(chunk.usage_metadata for chunk in chunks)

    def test_responses_api_stream_is_metered_once(self, openai_stub, records):
        chunks = stream_chunks(with_callback(use_responses_api=True, include_response_headers=True))

        assert "".join(chunk.text for chunk in chunks) == "hi"
        assert chunks[0].response_metadata["headers"]
        assert_one_record(openai_stub, records, provider="OPENAI", streamed=True, transport=True)


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


def cached_usage(input_tokens):
    """A one-hour cache write of 3,000 and a cache read of 2,000 beside ``input_tokens`` uncached."""
    return {"input_tokens": input_tokens, "output_tokens": 6, "cache_read_input_tokens": 2000,
            "cache_creation_input_tokens": 3000,
            "cache_creation": {"ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 3000}}


CONTRADICTION_WARNING = "Usage counts contradict each other"
UNCACHED_INPUTS = [pytest.param(16, id="small-input"), pytest.param(9000, id="large-input")]


def the_one_callback_record(stub, records):
    assert len(stub.ids) == 1, "the stub must serve exactly one upstream request"
    assert len(records) == 1, records
    assert records[0]["transaction_id"].startswith("langchain-")
    return records[0]


class TestChatAnthropicPromptCacheIsPricedOnce:
    """A LangChain Claude call metered by the callback alone prices its cache once (BACK-3993).

    langchain-anthropic's streamed ``usage_metadata`` puts the cache read and
    write inside ``input_tokens``; Revenium prices them again from their own
    fields, and nets nothing for Claude server-side.
    """

    @pytest.mark.usefixtures("without_anthropic_middleware")
    @pytest.mark.parametrize("uncached", UNCACHED_INPUTS)
    @pytest.mark.parametrize("call", [stream, astream])
    def test_a_streamed_callback_record_counts_cache_tokens_once(self, records, call, uncached, caplog):
        stub = StubAnthropic(cached_usage(uncached))
        with caplog.at_level("WARNING", logger="revenium_middleware"):
            assert call(stub.chat(callbacks=[ReveniumCallbackHandler()])) == "hi"

        record = the_one_callback_record(stub, records)
        assert record["is_streamed"] is True
        assert record["input_token_count"] == uncached
        assert record["cache_read_token_count"] == 2000
        assert record["cache_creation_token_count"] == 3000
        assert record["total_token_count"] == uncached + 6 + 2000 + 3000
        assert CONTRADICTION_WARNING not in caplog.text

    @pytest.mark.usefixtures("without_anthropic_middleware")
    @pytest.mark.parametrize("uncached", UNCACHED_INPUTS)
    @pytest.mark.parametrize("call", [invoke, ainvoke])
    def test_a_non_streamed_callback_record_carries_the_one_hour_split(self, records, call, uncached, caplog):
        stub = StubAnthropic(cached_usage(uncached))
        with caplog.at_level("WARNING", logger="revenium_middleware"):
            assert call(stub.chat(callbacks=[ReveniumCallbackHandler()])) == "hi"

        record = the_one_callback_record(stub, records)
        assert record["input_token_count"] == uncached
        assert record["cache_creation_token_count"] == 3000
        assert record["cache_creation1h_token_count"] == 3000
        assert record["cache_creation5m_token_count"] == 0
        assert CONTRADICTION_WARNING not in caplog.text

    @pytest.mark.parametrize("uncached", UNCACHED_INPUTS)
    @pytest.mark.parametrize("call", [invoke, ainvoke, stream, astream])
    def test_the_callback_record_prices_like_the_transport_record(self, records, monkeypatch, call, uncached):
        with monkeypatch.context() as callback_only:
            unwrap_anthropic_middleware(callback_only)
            assert call(StubAnthropic(cached_usage(uncached)).chat(callbacks=[ReveniumCallbackHandler()])) == "hi"
        assert call(StubAnthropic(cached_usage(uncached)).chat(callbacks=[ReveniumCallbackHandler()])) == "hi"

        callback_record, transport_record = records
        assert callback_record["transaction_id"].startswith("langchain-")
        assert not transport_record["transaction_id"].startswith("langchain-")
        priced = ("input_token_count", "cache_read_token_count", "cache_creation_token_count")
        assert {field: callback_record[field] for field in priced} == {
            field: transport_record[field] for field in priced
        }
        assert transport_record["cache_creation1h_token_count"] == 3000
        if call in (stream, astream):
            # Anthropic sends the split on message_start, which langchain-anthropic
            # drops, so a streamed callback record cannot carry it.
            assert "cache_creation5m_token_count" not in callback_record
            assert "cache_creation1h_token_count" not in callback_record
        else:
            for field in ("cache_creation5m_token_count", "cache_creation1h_token_count"):
                assert callback_record[field] == transport_record[field]


ADVISOR_MODEL = "claude-opus-5"
BILLED_PASSES = [
    {"type": "message", "input_tokens": 5, "output_tokens": 2},
    {"type": "advisor_message", "model": ADVISOR_MODEL, "input_tokens": 823, "output_tokens": 1612},
    {"type": "compaction", "input_tokens": 1800, "output_tokens": 350},
    {"type": "message", "input_tokens": INPUT_TOKENS - 5, "output_tokens": 3},
]
MESSAGE_PASSES_ONLY = [BILLED_PASSES[0], BILLED_PASSES[3]]
MESSAGES_APIS = [pytest.param({}, id="messages"),
                 pytest.param({"betas": ["advisor-tool-2026-03-01"]}, id="beta-messages")]


def usage_with_passes(iterations):
    return {"input_tokens": INPUT_TOKENS, "output_tokens": OUTPUT_TOKENS, "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0, "iterations": iterations}


def pass_ids(main_id):
    return [main_id, f"{main_id}-iteration-1", f"{main_id}-iteration-2"]


def assert_pass_records(records, main_id, *, compaction_model=ANTHROPIC_ALIAS):
    assert len(records) == 3, records
    main, advisor, compaction = records
    assert [record["transaction_id"] for record in records] == pass_ids(main_id)
    assert (main["input_token_count"], main["output_token_count"]) == (INPUT_TOKENS, OUTPUT_TOKENS)
    assert advisor["operation_subtype"] == "advisor_message"
    assert advisor["model"] == ADVISOR_MODEL
    assert (advisor["input_token_count"], advisor["output_token_count"], advisor["total_token_count"]) == (
        823, 1612, 2435)
    assert compaction["operation_subtype"] == "compaction"
    assert compaction["model"] == compaction_model
    assert (compaction["input_token_count"], compaction["output_token_count"],
            compaction["total_token_count"]) == (1800, 350, 2150)
    for extra in (advisor, compaction):
        assert extra["parent_transaction_id"] == main_id
        assert extra["provider"] == "ANTHROPIC"
        assert extra["model_source"] == "ANTHROPIC"
        assert extra["is_streamed"] is main["is_streamed"]
    assert sum(r["input_token_count"] for r in records) == INPUT_TOKENS + 823 + 1800
    assert sum(r["output_token_count"] for r in records) == OUTPUT_TOKENS + 1612 + 350


class StreamingTap(BaseCallbackHandler):
    """The kind of handler astream_events and a LangGraph run streamed in messages
    mode attach. LangChain streams any call that carries one, an invoke included."""

    def tap_output_iter(self, run_id, output):
        return output

    def tap_output_aiter(self, run_id, output):
        return output


def invoke_under_a_streaming_tap(model):
    return model.invoke("hi", config={"callbacks": [StreamingTap()]}).text


def callback_only_chat_with_passes(**chat_kwargs):
    return StubAnthropic(usage_with_passes(BILLED_PASSES)).chat(callbacks=[ReveniumCallbackHandler()], **chat_kwargs)


class TestChatAnthropicAdvisorAndCompactionPasses:
    """A ChatAnthropic call metered by the callback alone sends one record per advisor
    or compaction pass when LangChain does not stream the call."""

    @pytest.mark.usefixtures("without_anthropic_middleware")
    @pytest.mark.parametrize("chat_kwargs", MESSAGES_APIS)
    @pytest.mark.parametrize("call", [invoke, ainvoke])
    def test_a_non_streamed_callback_call_bills_each_pass(self, records, call, chat_kwargs):
        assert call(callback_only_chat_with_passes(**chat_kwargs)) == "hi"
        assert records[0]["transaction_id"].startswith("langchain-")
        assert_pass_records(records, records[0]["transaction_id"])

    @pytest.mark.usefixtures("without_anthropic_middleware")
    def test_callback_pass_records_take_their_transaction_id_as_idempotency_key(self):
        sent = []

        def record(operation, payload, idempotency_key=None):
            sent.append((payload["transaction_id"], idempotency_key))
            return MagicMock(id="recorded", status_code=201)

        with patch("revenium_middleware.openai.middleware.submit_ai_event", side_effect=record), \
                patch("revenium_middleware.openai.middleware.run_async_in_thread", side_effect=_run_now):
            invoke(callback_only_chat_with_passes())
        main_id = sent[0][0]
        assert sent == [(main_id, None), (f"{main_id}-iteration-1", f"{main_id}-iteration-1"),
                        (f"{main_id}-iteration-2", f"{main_id}-iteration-2")]

    @pytest.mark.usefixtures("without_anthropic_middleware")
    def test_callback_pass_records_follow_the_idempotency_block_key(self):
        metering_client = MagicMock()
        metering_client.ai.create_completion = MagicMock(return_value=MagicMock(id="recorded", status_code=201))
        model = callback_only_chat_with_passes()
        with patch("revenium_middleware._core.metering_submission.current_client", lambda: metering_client), \
                patch("revenium_middleware.openai.middleware.run_async_in_thread",
                      side_effect=_run_now_in_caller_context):
            with idempotency_block("blk"):
                invoke(model)
                invoke(model)
        sent = [call.kwargs["extra_headers"]["Idempotency-Key"]
                for call in metering_client.ai.create_completion.call_args_list]
        assert sent == ["blk", "blk-iteration-1", "blk-iteration-2", "blk:2", "blk:2-iteration-1", "blk:2-iteration-2"]

    @pytest.mark.usefixtures("without_anthropic_middleware")
    def test_callback_pass_records_are_sent_when_the_main_record_fails(self):
        sent = []

        def reject_main_record(operation, payload, idempotency_key=None):
            sent.append(payload["transaction_id"])
            if "-iteration-" not in payload["transaction_id"]:
                raise ValueError("main record rejected")
            return MagicMock(id="recorded", status_code=201)

        with patch("revenium_middleware.openai.middleware.submit_ai_event", side_effect=reject_main_record), \
                patch("revenium_middleware.openai.middleware.run_async_in_thread", side_effect=_run_now):
            assert invoke(callback_only_chat_with_passes()) == "hi"
        assert sent[0].startswith("langchain-")
        assert sent == pass_ids(sent[0])

    @pytest.mark.usefixtures("without_anthropic_middleware")
    @pytest.mark.parametrize("chat_kwargs", MESSAGES_APIS)
    @pytest.mark.parametrize("call", [stream, astream, invoke_under_a_streaming_tap])
    def test_a_streamed_callback_call_sends_only_the_main_record(self, records, call, chat_kwargs):
        """The documented gap: langchain-anthropic keeps no usage.iterations on a result it
        streamed, whatever the caller called. See the LangChain section of README.md."""
        stub = StubAnthropic(usage_with_passes(BILLED_PASSES))
        assert call(stub.chat(callbacks=[ReveniumCallbackHandler()], **chat_kwargs)) == "hi"
        assert the_one_callback_record(stub, records)["is_streamed"] is True

    @pytest.mark.usefixtures("without_anthropic_middleware")
    @pytest.mark.parametrize("iterations", [None, [], MESSAGE_PASSES_ONLY], ids=["none", "empty", "message_only"])
    @pytest.mark.parametrize("call", [invoke, ainvoke])
    def test_a_callback_call_without_billed_passes_sends_one_record(self, records, call, iterations):
        stub = StubAnthropic(usage_with_passes(iterations))
        assert call(stub.chat(callbacks=[ReveniumCallbackHandler()])) == "hi"
        record = the_one_callback_record(stub, records)
        assert (record["input_token_count"], record["output_token_count"]) == (INPUT_TOKENS, OUTPUT_TOKENS)

    @pytest.mark.parametrize("chat_kwargs", MESSAGES_APIS)
    @pytest.mark.parametrize("call", [invoke, ainvoke, stream, astream])
    def test_the_transport_bills_each_pass_once_when_it_also_meters(self, records, call, chat_kwargs):
        stub = StubAnthropic(usage_with_passes(BILLED_PASSES))
        assert call(stub.chat(callbacks=[ReveniumCallbackHandler()], **chat_kwargs)) == "hi"
        assert_pass_records(records, stub.ids[0])


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


def openai_text(**kwargs):
    return wrap(langchain_openai.OpenAI(model=TEXT_MODEL, api_key="sk-test", max_retries=0, **kwargs))


def azure_text(api_version="2024-10-21", **kwargs):
    return wrap(langchain_openai.AzureOpenAI(model=TEXT_MODEL, azure_deployment="instruct-dep",
                                             api_version=api_version, azure_endpoint=f"https://{AZURE_HOST}",
                                             api_key="sk-test", max_retries=0, **kwargs))


def text_stream(model):
    return "".join(model.stream("hi"))


def text_astream(model):
    async def call():
        return "".join([chunk async for chunk in model.astream("hi")])
    return asyncio.run(call())


def text_invoke(model):
    return model.invoke("hi")


def text_ainvoke(model):
    return asyncio.run(model.ainvoke("hi"))


NO_USAGE_WARNING = "No usage data found in LLM response"


class StreamedTextModelStandIn(langchain_core_llms.LLM):
    """A streamed text model whose calls no Revenium wrap sees and whose stream carries no usage."""

    @property
    def _llm_type(self):
        return "streamed-text-stand-in"

    def _call(self, prompt, stop=None, run_manager=None, **kwargs):
        return "hi"

    def _stream(self, prompt, stop=None, run_manager=None, **kwargs):
        for text in ("h", "i"):
            chunk = GenerationChunk(text=text)
            if run_manager:
                run_manager.on_llm_new_token(text, chunk=chunk)
            yield chunk


class RawResponseTextModel(langchain_core_llms.LLM):
    """A text model that streams the legacy completions endpoint through with_raw_response."""

    @property
    def _llm_type(self):
        return "raw-response-text-model"

    def _get_ls_params(self, stop=None, **kwargs):
        return {"ls_provider": "openai", "ls_model_name": TEXT_MODEL, "ls_model_type": "llm"}

    def _call(self, prompt, stop=None, run_manager=None, **kwargs):
        client = openai_sdk.OpenAI(api_key="sk-test", max_retries=0)
        raw = client.completions.with_raw_response.create(model=TEXT_MODEL, prompt=prompt, stream=True)
        assert isinstance(raw, openai_sdk._legacy_response.LegacyAPIResponse), type(raw)
        return "".join(chunk.choices[0].text for chunk in raw.parse() if chunk.choices)


class TestTextCompletionClasses:
    """BACK-4350: LangChain's OpenAI and AzureOpenAI text-completion classes record their streamed calls."""

    @pytest.mark.parametrize("call, build_kwargs", [
        pytest.param(text_stream, {}, id="stream"),
        pytest.param(text_astream, {}, id="astream"),
        pytest.param(text_invoke, {"streaming": True}, id="invoke-streaming-true"),
        pytest.param(text_ainvoke, {"streaming": True}, id="ainvoke-streaming-true", marks=pytest.mark.xfail(
            strict=True, reason="langchain-core's BaseLLM.agenerate starts each run in its own asyncio.gather task, "
                                "so the run's published metadata never reaches the client call")),
    ])
    @pytest.mark.parametrize("build, provider", [
        pytest.param(openai_text, "OPENAI", id="openai"),
        pytest.param(azure_text, "Azure", id="azure"),
    ])
    def test_a_streamed_text_completion_call_is_recorded_once_with_its_tokens(
            self, openai_stub, records, build, provider, call, build_kwargs):
        assert call(build(**build_kwargs)) == "hi"

        assert_one_record(openai_stub, records, provider=provider, streamed=True, transport=False)
        assert openai_stub.bodies[0]["stream_options"] == {"include_usage": True}

    def test_an_openai_sdk_without_stream_options_is_left_unwrapped(self, monkeypatch):
        import openai.resources.completions as completions_module
        from revenium_middleware._core import patch_registry
        from revenium_middleware.openai.langchain import _text_completion_usage as capture

        def create(self, *, model, prompt, stream=False):
            return None

        async def acreate(self, *, model, prompt, stream=False):
            return None

        keys = {f"{capture.CAPTURE_KEY_PREFIX}{capture._COMPLETIONS_MODULE}.{attribute}"
                for attribute, _ in capture._TARGETS}
        monkeypatch.setattr(completions_module.Completions, "create", create)
        monkeypatch.setattr(completions_module.AsyncCompletions, "create", acreate)
        monkeypatch.setattr(patch_registry, "_patched", set(patch_registry._patched) - keys)

        capture.install_text_completion_usage_capture()

        assert completions_module.Completions.create is create
        assert completions_module.AsyncCompletions.create is acreate
        assert not keys & patch_registry._patched

    def test_a_text_completion_stream_without_usage_is_still_recorded(self, openai_stub, records, caplog):
        openai_stub.ignore_include_usage = True

        with caplog.at_level("WARNING", logger="revenium_middleware"):
            assert text_stream(openai_text()) == "hi"

        assert len(records) == 1, records
        assert records[0]["input_token_count"] == 0
        assert records[0]["output_token_count"] == 0
        assert records[0]["is_streamed"] is True
        assert "requested" in caplog.text

    def test_two_handlers_on_a_text_completion_stream_without_usage_send_one_record(self, openai_stub, records):
        openai_stub.ignore_include_usage = True

        assert "".join(openai_text().stream("hi", config={"callbacks": [ReveniumCallbackHandler()]})) == "hi"

        assert len(records) == 1, records
        assert records[0]["input_token_count"] == 0

    def test_the_requested_usage_chunk_stays_out_of_a_text_completion_stream(self, openai_stub, records):
        assert list(openai_text().stream("hi")) == ["hi", ""]

    @pytest.mark.parametrize("call", [text_invoke, text_ainvoke], ids=["invoke", "ainvoke"])
    def test_a_plain_text_completion_call_is_still_recorded_once(self, openai_stub, records, call):
        assert call(openai_text()) == "hi"

        assert_one_record(openai_stub, records, provider="OPENAI", streamed=False, transport=False)
        assert "stream_options" not in openai_stub.bodies[0]

    def test_a_bare_client_completions_stream_outside_langchain_is_left_alone(self, openai_stub, records):
        client = openai_sdk.OpenAI(api_key="sk-test", max_retries=0)

        chunks = list(client.completions.create(model=TEXT_MODEL, prompt="hi", stream=True))

        assert [chunk.choices[0].text for chunk in chunks] == ["hi", ""]
        assert "stream_options" not in openai_stub.bodies[0]
        assert records == []

    def test_an_old_azure_api_version_text_stream_gets_no_stream_options(self, openai_stub, records, caplog):
        """Azure refuses stream_options before api-version 2024-09-01-preview, so it is never sent there."""
        with caplog.at_level("WARNING", logger="revenium_middleware"):
            assert text_stream(azure_text(api_version="2024-05-01-preview")) == "hi"

        assert "stream_options" not in openai_stub.bodies[0]
        assert records == []
        assert NO_USAGE_WARNING in caplog.text

    def test_a_custom_streamed_text_model_without_capture_still_sends_no_record(self, records, caplog):
        with caplog.at_level("WARNING", logger="revenium_middleware"):
            assert text_stream(wrap(StreamedTextModelStandIn())) == "hi"

        assert records == []
        assert NO_USAGE_WARNING in caplog.text

    def test_an_abandoned_text_completion_stream_is_closed_and_not_recorded(self, openai_stub, records):
        chunks = openai_text().stream("hi")
        assert next(chunks) == "hi"

        chunks.close()
        del chunks
        gc.collect()

        assert openai_stub.closed_streams == 1
        assert records == []

    def test_a_raw_response_text_completion_call_is_untouched(self, openai_stub, records):
        assert text_invoke(wrap(RawResponseTextModel())) == "hi"

        assert "stream_options" not in openai_stub.bodies[0]


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
               side_effect=lambda usage_metadata=None, **_keywords: checked.append(dict(usage_metadata or {}))):
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


class TestAttachToOnASharedModel:
    """BACK-4261: attaching per request on one shared model must not stack handlers."""

    @staticmethod
    def revenium_handlers(llm):
        return [c for c in llm.callbacks if isinstance(c, ReveniumCallbackHandler)]

    def test_attaching_per_request_keeps_one_revenium_handler(self, openai_stub, records):
        llm = chat_openai()
        for i in range(50):
            attach_to(llm, usage_metadata={"subscriber": {"id": f"user_{i}"}})
        assert len(self.revenium_handlers(llm)) == 1

        llm.invoke("hi")
        assert len(records) == 1
        assert records[0]["subscriber"] == {"id": "user_49"}

    def test_attach_keeps_the_customers_own_callbacks(self, openai_stub, records):
        own = BaseCallbackHandler()
        llm = chat_openai(callbacks=[own])
        attach_to(llm, usage_metadata={"subscriber": {"id": "first"}})
        attach_to(llm, usage_metadata={"subscriber": {"id": "second"}})
        assert own in llm.callbacks
        assert len(self.revenium_handlers(llm)) == 1

    def test_overlapping_requests_named_with_revenium_metadata_keep_their_person(self, monkeypatch, records):
        stub = StubOpenAI()
        both_in_flight = threading.Barrier(2, timeout=10)

        def respond_once_both_are_in_flight(request):
            both_in_flight.wait()
            return stub.respond(request)

        answer_every_transport(monkeypatch, respond_once_both_are_in_flight)
        llm = chat_openai()
        attach_to(llm, usage_metadata={"organizationName": "org-shared"})

        def handle_request(user_id):
            @revenium_metadata(subscriber={"id": user_id})
            def answer():
                return llm.invoke("hi")
            return answer()

        with ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(handle_request, ["alice", "bob"]))

        assert len(records) == 2
        assert {record["subscriber"]["id"] for record in records} == {"alice", "bob"}
        assert all(record["organization_name"] == "org-shared" for record in records)

    def test_an_attached_and_a_per_call_handler_record_a_chat_openai_call_once(self, openai_stub, records):
        """BACK-4318: the attached handler's fields are the ones kept, as on callback-recorded models."""
        llm = chat_openai()
        attach_to(llm, usage_metadata={"organizationName": "org-startup"})
        per_call = ReveniumCallbackHandler(usage_metadata={"subscriber": {"id": "per-call"}})
        assert llm.invoke("hi", config={"callbacks": [per_call]}).text == "hi"
        assert len(records) == 1, records
        assert records[0]["organization_name"] == "org-startup"
        assert not records[0].get("subscriber")


def three_prompts():
    return [[HumanMessage(content=text)] for text in ("a", "b", "c")]


BATCH_CALLS = {
    "generate": lambda model: model.generate(three_prompts()),
    "generate_prompt": lambda model: model.generate_prompt(
        [ChatPromptValue(messages=messages) for messages in three_prompts()]),
    "agenerate": lambda model: asyncio.run(model.agenerate(three_prompts())),
}


class Delegating(BaseChatModel):
    """A model whose answer comes from another model it calls without passing its callbacks on."""

    inner: BaseChatModel

    @property
    def _llm_type(self):
        return "delegating"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        return ChatResult(generations=[ChatGeneration(message=self.inner.invoke(messages))])


class TestBatchedCallsEachCarryTheCallbackMetadata:
    """BACK-4309: the last call of a streamed synchronous batch lost the caller's tags."""

    @pytest.mark.parametrize("call", sorted(BATCH_CALLS))
    @pytest.mark.parametrize("streaming", [True, False], ids=["streamed", "not-streamed"])
    def test_every_call_of_a_batch_carries_it(self, openai_stub, records, streaming, call):
        BATCH_CALLS[call](wrap(chat_openai(streaming=streaming), usage_metadata=CALLBACK_METADATA))

        assert len(openai_stub.ids) == 3
        assert sorted(record["transaction_id"] for record in records) == sorted(openai_stub.ids)
        assert [record.get("organization_name") for record in records] == ["org-callback"] * 3
        assert [record.get("trace_id") for record in records] == ["trace-callback"] * 3
        assert [record.get("subscriber") for record in records] == [CALLBACK_METADATA["subscriber"]] * 3

    def test_a_model_that_delegates_to_another_model_keeps_the_callers_tags(self, openai_stub, records):
        model = wrap(Delegating(inner=chat_openai()), usage_metadata=CALLBACK_METADATA)

        assert model.invoke("hi").text == "hi"
        assert [record.get("organization_name") for record in records] == ["org-callback"]


def metered(call):
    """``call`` made inside a @revenium_meter function."""
    return revenium_meter()(call)


def metered_async(coroutine_function):
    """``coroutine_function`` awaited inside an async @revenium_meter function."""
    handle = revenium_meter()(coroutine_function)
    return lambda model: asyncio.run(handle(model))


async def _ainvoke_text(model):
    return (await model.ainvoke("hi")).text


async def _astream_text(model):
    return "".join([chunk.text async for chunk in model.astream("hi")])


OUTSIDE_CALLS = [
    pytest.param(invoke, id="invoke"),
    pytest.param(ainvoke, id="ainvoke"),
    pytest.param(stream, id="stream"),
    pytest.param(astream, id="astream"),
]
INSIDE_CALLS = [
    pytest.param(metered(invoke), False, id="invoke"),
    pytest.param(metered_async(_ainvoke_text), False, id="ainvoke"),
    pytest.param(metered(stream), True, id="stream"),
    pytest.param(metered_async(_astream_text), True, id="astream"),
]


@pytest.fixture
def mistral_requests():
    pytest.importorskip("langchain_mistralai")
    return []


def chat_mistralai(requests):
    return mistral.chat_mistralai([ReveniumCallbackHandler()], during_request=lambda: requests.append(1))


def assert_one_mistral_record(requests, records):
    assert len(requests) == 1, "the stub must serve exactly one upstream request"
    assert len(records) == 1, records
    assert records[0]["provider"] == "MISTRAL"
    assert records[0]["transaction_id"].startswith("langchain-")
    assert (records[0]["input_token_count"], records[0]["output_token_count"]) == (
        mistral.INPUT_TOKENS, mistral.OUTPUT_TOKENS)


class TestSelectiveMetering:
    """BACK-4320: with selective metering on, a LangChain call outside a
    @revenium_meter function sends no record, as the README says, and one
    inside it sends exactly one. The call is judged when it starts."""

    @pytest.fixture(autouse=True)
    def _selective_metering(self, monkeypatch):
        monkeypatch.setenv("REVENIUM_SELECTIVE_METERING", "true")

    @pytest.mark.parametrize("call", OUTSIDE_CALLS)
    def test_a_chat_openai_call_outside_a_metered_function_sends_no_record(self, openai_stub, records, call):
        assert call(with_callback()) == "hi"

        assert len(openai_stub.ids) == 1
        assert records == []

    def test_a_wrapped_chat_openai_call_outside_a_metered_function_sends_no_record(self, openai_stub, records):
        assert invoke(wrap(chat_openai())) == "hi"

        assert len(openai_stub.ids) == 1
        assert records == []

    @pytest.mark.parametrize("call", OUTSIDE_CALLS)
    def test_a_chat_mistralai_call_outside_a_metered_function_sends_no_record(self, mistral_requests, records, call):
        assert call(chat_mistralai(mistral_requests)) == mistral.ANSWER

        assert len(mistral_requests) == 1
        assert records == []

    @pytest.mark.parametrize("call", [
        pytest.param(text_invoke, id="invoke"),
        pytest.param(text_stream, id="stream"),
    ])
    def test_a_text_completion_call_outside_a_metered_function_sends_no_record(self, openai_stub, records, call):
        assert call(openai_text()) == "hi"

        assert len(openai_stub.ids) == 1
        assert records == []

    @pytest.mark.parametrize("call, streamed", INSIDE_CALLS)
    def test_a_chat_openai_call_inside_a_metered_function_is_recorded_once(self, openai_stub, records, call, streamed):
        assert call(with_callback()) == "hi"

        assert_one_record(openai_stub, records, provider="OPENAI", streamed=streamed, transport=True)

    def test_a_wrapped_chat_openai_call_inside_a_metered_function_is_recorded_once(self, openai_stub, records):
        assert metered(invoke)(wrap(chat_openai())) == "hi"

        assert_one_record(openai_stub, records, provider="OPENAI", streamed=False, transport=True)

    @pytest.mark.parametrize("call, streamed", INSIDE_CALLS)
    def test_a_chat_mistralai_call_inside_a_metered_function_is_recorded_once(
            self, mistral_requests, records, call, streamed):
        assert call(chat_mistralai(mistral_requests)) == mistral.ANSWER

        assert_one_mistral_record(mistral_requests, records)
        assert records[0]["is_streamed"] is streamed

    @pytest.mark.parametrize("call, streamed", [
        pytest.param(metered(text_invoke), False, id="invoke"),
        pytest.param(metered(text_stream), True, id="stream"),
    ])
    def test_a_text_completion_call_inside_a_metered_function_is_recorded_once(
            self, openai_stub, records, call, streamed):
        assert call(openai_text()) == "hi"

        assert_one_record(openai_stub, records, provider="OPENAI", streamed=streamed, transport=False)

    def test_a_chat_openai_stream_started_inside_a_metered_function_is_recorded_once_when_read_outside(
            self, openai_stub, records):
        first, rest = metered(started_stream)(with_callback())

        assert first.text + "".join(chunk.text for chunk in rest) == "hi"
        assert_one_record(openai_stub, records, provider="OPENAI", streamed=True, transport=True)

    def test_a_chat_mistralai_stream_started_inside_a_metered_function_is_recorded_once_when_read_outside(
            self, mistral_requests, records):
        first, rest = metered(started_stream)(chat_mistralai(mistral_requests))

        assert first.text + "".join(chunk.text for chunk in rest) == mistral.ANSWER
        assert_one_mistral_record(mistral_requests, records)


def started_stream(model):
    """A stream whose first chunk is read here and the rest by the caller."""
    chunks = model.stream("hi")
    return next(chunks), chunks
