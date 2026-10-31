"""client.beta.messages and messages.parse() are metered once per call (BACK-3606).

The real anthropic package runs over an httpx2 stub. The beta response carries
a beta-only content block and the beta-only usage fields, which the standard
Message model rejects, so the beta path is exercised on its own types.
"""
import asyncio
import contextvars
import hashlib
import json
import threading
from unittest.mock import MagicMock, patch

import pytest

anthropic = pytest.importorskip("anthropic")
httpx2 = pytest.importorskip("httpx2")

import revenium_middleware.anthropic  # noqa: E402,F401  installs the patches
from revenium_middleware import idempotency_key as idempotency_block  # noqa: E402
from revenium_middleware import revenium_metadata  # noqa: E402
from revenium_middleware.anthropic.provider import Provider  # noqa: E402
from revenium_middleware._core.call_ownership import (  # noqa: E402
    ANTHROPIC, claimed_by_transport, reset_claimed_response_ids, transport_claim_mark,
)

DATED_MODEL = "claude-sonnet-4-5-20250929"
MESSAGE_ID = "msg_beta_stub"
INPUT_TOKENS = 13
OUTPUT_TOKENS = 5
CACHE_READ_TOKENS = 2
STUB_HOST = "anthropic.stub.invalid"
REQUEST = dict(model="claude-sonnet-4-5", max_tokens=16, messages=[{"role": "user", "content": "hi"}])
BETA_REQUEST = dict(REQUEST, betas=["mcp-client-2025-04-04"])

_USAGE = {
    "input_tokens": INPUT_TOKENS,
    "output_tokens": OUTPUT_TOKENS,
    "cache_creation_input_tokens": 0,
    "cache_read_input_tokens": CACHE_READ_TOKENS,
    "server_tool_use": {"web_search_requests": 1, "web_fetch_requests": 0},
    "service_tier": "standard",
    "speed": "standard",
    "iterations": None,
}
_MCP_BLOCK = {"type": "mcp_tool_use", "id": "mcptoolu_1", "name": "search", "server_name": "docs", "input": {}}
_MESSAGE = {
    "id": MESSAGE_ID, "type": "message", "role": "assistant", "model": DATED_MODEL,
    "content": [_MCP_BLOCK, {"type": "text", "text": "hi"}], "stop_reason": "end_turn", "stop_sequence": None,
    "usage": _USAGE,
}


def _sse_body(start_usage, delta_usage):
    events = [
        ("message_start", {"type": "message_start", "message": {
            **_MESSAGE, "content": [], "stop_reason": None, "usage": start_usage}}),
        ("content_block_start", {"type": "content_block_start", "index": 0,
                                 "content_block": {"type": "text", "text": ""}}),
        ("content_block_delta", {"type": "content_block_delta", "index": 0,
                                 "delta": {"type": "text_delta", "text": "hi"}}),
        ("content_block_stop", {"type": "content_block_stop", "index": 0}),
        ("message_delta", {"type": "message_delta", "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                           "usage": delta_usage}),
        ("message_stop", {"type": "message_stop"}),
    ]
    return "".join(f"event: {name}\ndata: {json.dumps(data)}\n\n" for name, data in events).encode()


def _is_beta_request(request):
    # anthropic 2.3.0 stopped adding `?beta=true` to client.beta.messages URLs, so a beta call is told apart by the
    # `anthropic-beta` header that `betas=` sets on every version.
    return "anthropic-beta" in request.headers


class StubAnthropic:
    def __init__(self, usage=_USAGE, start_usage=None, delta_usage=None):
        self.requests = []
        self._message_body = json.dumps({**_MESSAGE, "usage": usage}).encode()
        self._standard_message_body = json.dumps(
            {**_MESSAGE, "content": [{"type": "text", "text": "hi"}], "usage": usage}).encode()
        self._sse_body = _sse_body(start_usage or {**usage, "output_tokens": 1},
                                   delta_usage or {"output_tokens": OUTPUT_TOKENS})

    def _respond(self, request):
        assert request.url.host == STUB_HOST
        self.requests.append(request)
        if json.loads(request.content).get("stream"):
            return httpx2.Response(200, headers={"content-type": "text/event-stream"}, content=self._sse_body)
        body = self._message_body if _is_beta_request(request) else self._standard_message_body
        return httpx2.Response(200, headers={"content-type": "application/json"}, content=body)

    def sync_client(self):
        return anthropic.Anthropic(api_key="sk-ant-test", base_url=f"http://{STUB_HOST}", max_retries=0,
                                   http_client=httpx2.Client(transport=httpx2.MockTransport(self._respond)))

    def async_client(self):
        return anthropic.AsyncAnthropic(api_key="sk-ant-test", base_url=f"http://{STUB_HOST}", max_retries=0,
                                        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(self._respond)))


class UnbufferedStreamStub(StubAnthropic):
    """Serves the event stream line by line, so the response stays open until it is drained or closed."""

    def _respond(self, request):
        assert request.url.host == STUB_HOST
        self.requests.append(request)
        self.stream_response = httpx2.Response(200, headers={"content-type": "text/event-stream"},
                                               content=iter(self._sse_body.splitlines(keepends=True)))
        return self.stream_response


def _run_metering_synchronously(coro_func, *args, **kwargs):
    thread = threading.Thread(target=lambda: asyncio.run(coro_func(*args, **kwargs)))
    thread.start()
    thread.join(timeout=10)
    return thread


@pytest.fixture
def stub():
    return StubAnthropic()


@pytest.fixture
def records():
    captured = []

    def record(operation, payload, *args, **kwargs):
        captured.append(payload)
        return MagicMock(status_code=201)

    with patch("revenium_middleware.anthropic.middleware._get_thread_safe_client", return_value=MagicMock()), \
            patch("revenium_middleware.anthropic.middleware.submit_ai_event", side_effect=record), \
            patch("revenium_middleware.anthropic.middleware._safe_run_async_in_thread",
                  side_effect=_run_metering_synchronously):
        yield captured


def assert_one_record(stub, records, *, streamed, provider="ANTHROPIC"):
    assert len(stub.requests) == 1
    assert len(records) == 1, records
    record = records[0]
    assert record["input_token_count"] == INPUT_TOKENS
    assert record["output_token_count"] == OUTPUT_TOKENS
    assert record["cache_read_token_count"] == CACHE_READ_TOKENS
    assert record["provider"] == provider
    assert record["model"] == DATED_MODEL
    assert record["transaction_id"] == MESSAGE_ID
    assert record["is_streamed"] is streamed
    return record


def _beta(client):
    return client.beta.messages


def _standard(client):
    return client.messages


def _drain_sync_create(messages, request):
    return list(messages.create(stream=True, **request))


def _drain_sync_stream(messages, request):
    with messages.stream(**request) as stream:
        return list(stream)


async def _drain_async_create(messages, request):
    return [event async for event in await messages.create(stream=True, **request)]


async def _drain_async_stream(messages, request):
    async with messages.stream(**request) as stream:
        return [event async for event in stream]


RESOURCES = [pytest.param(_beta, BETA_REQUEST, id="beta"), pytest.param(_standard, REQUEST, id="standard")]


@pytest.mark.parametrize("resource, request_kwargs", RESOURCES)
class TestOneRecordPerCall:
    def test_create_sync(self, stub, records, resource, request_kwargs):
        message = resource(stub.sync_client()).create(**request_kwargs)
        assert message.id == MESSAGE_ID
        assert_one_record(stub, records, streamed=False)

    def test_create_async(self, stub, records, resource, request_kwargs):
        message = asyncio.run(resource(stub.async_client()).create(**request_kwargs))
        assert message.id == MESSAGE_ID
        assert_one_record(stub, records, streamed=False)

    def test_create_stream_sync(self, stub, records, resource, request_kwargs):
        events = _drain_sync_create(resource(stub.sync_client()), request_kwargs)
        assert events[-1].type == "message_stop"
        assert_one_record(stub, records, streamed=True)

    def test_create_stream_async(self, stub, records, resource, request_kwargs):
        events = asyncio.run(_drain_async_create(resource(stub.async_client()), request_kwargs))
        assert events[-1].type == "message_stop"
        assert_one_record(stub, records, streamed=True)

    def test_stream_helper_sync(self, stub, records, resource, request_kwargs):
        _drain_sync_stream(resource(stub.sync_client()), request_kwargs)
        assert_one_record(stub, records, streamed=True)

    def test_stream_helper_async(self, stub, records, resource, request_kwargs):
        asyncio.run(_drain_async_stream(resource(stub.async_client()), request_kwargs))
        assert_one_record(stub, records, streamed=True)

    def test_parse_sync(self, stub, records, resource, request_kwargs):
        message = resource(stub.sync_client()).parse(**request_kwargs)
        assert message.id == MESSAGE_ID
        assert_one_record(stub, records, streamed=False)

    def test_parse_async(self, stub, records, resource, request_kwargs):
        message = asyncio.run(resource(stub.async_client()).parse(**request_kwargs))
        assert message.id == MESSAGE_ID
        assert_one_record(stub, records, streamed=False)


class TestBetaRawResponse:
    def test_raw_create_returns_the_sdk_response(self, stub, records):
        response = stub.sync_client().beta.messages.with_raw_response.create(**BETA_REQUEST)
        assert isinstance(response.parse(), anthropic.types.beta.BetaMessage)
        assert_one_record(stub, records, streamed=False)

    def test_streaming_response_body_decodes_as_a_beta_message(self, stub, records):
        with stub.sync_client().beta.messages.with_streaming_response.create(**BETA_REQUEST) as response:
            assert response.parse().content[0].type == "mcp_tool_use"
        assert_one_record(stub, records, streamed=False)

    def test_async_streaming_response_body_decodes_as_a_beta_message(self, stub, records):
        async def call():
            messages = stub.async_client().beta.messages
            async with messages.with_streaming_response.create(**BETA_REQUEST) as response:
                return await response.parse()

        assert asyncio.run(call()).content[0].type == "mcp_tool_use"
        assert_one_record(stub, records, streamed=False)


class TestTransportClaim:
    """The LangChain callback stays silent for a call the transport metered."""

    @pytest.fixture(autouse=True)
    def _fresh_ledger(self):
        reset_claimed_response_ids()
        yield
        reset_claimed_response_ids()

    @pytest.mark.parametrize("call", [
        pytest.param(lambda client: _beta(client).create(**BETA_REQUEST), id="create"),
        pytest.param(lambda client: _beta(client).parse(**BETA_REQUEST), id="parse"),
        pytest.param(lambda client: _drain_sync_create(_beta(client), BETA_REQUEST), id="create-stream"),
        pytest.param(lambda client: _drain_sync_stream(_beta(client), BETA_REQUEST), id="stream-helper"),
    ])
    def test_sync_beta_calls_claim_the_record(self, stub, records, call):
        mark = transport_claim_mark()
        call(stub.sync_client())
        assert claimed_by_transport(mark, {ANTHROPIC})

    @pytest.mark.parametrize("call", [
        pytest.param(lambda client: _beta(client).create(**BETA_REQUEST), id="create"),
        pytest.param(lambda client: _drain_async_create(_beta(client), BETA_REQUEST), id="create-stream"),
        pytest.param(lambda client: _drain_async_stream(_beta(client), BETA_REQUEST), id="stream-helper"),
    ])
    def test_async_beta_calls_claim_the_record(self, stub, records, call):
        async def run():
            mark = transport_claim_mark()
            await call(stub.async_client())
            return claimed_by_transport(mark, {ANTHROPIC})

        assert asyncio.run(run())


class TestBedrockClientsStayOnTheSdkTransport:
    @pytest.fixture
    def bedrock_detected(self):
        with patch("revenium_middleware.anthropic.middleware.detect_provider", return_value=Provider.BEDROCK), \
                patch("revenium_middleware.anthropic.middleware._handle_bedrock_request") as fast_create, \
                patch("revenium_middleware.anthropic.middleware._handle_bedrock_stream_request") as fast_stream:
            yield fast_create, fast_stream

    @pytest.mark.parametrize("call, streamed", [
        pytest.param(lambda client: _beta(client).create(**BETA_REQUEST), False, id="beta-create"),
        pytest.param(lambda client: _beta(client).parse(**BETA_REQUEST), False, id="beta-parse"),
        pytest.param(lambda client: _standard(client).parse(**REQUEST), False, id="standard-parse"),
        pytest.param(lambda client: _drain_sync_stream(_beta(client), BETA_REQUEST), True, id="beta-stream"),
    ])
    def test_metered_under_the_aws_label_without_rerouting(self, stub, records, bedrock_detected, call, streamed):
        call(stub.sync_client())
        fast_create, fast_stream = bedrock_detected
        fast_create.assert_not_called()
        fast_stream.assert_not_called()
        assert_one_record(stub, records, streamed=streamed, provider="AWS")


TOOL_STEP_ID = "msg_tool_step"
FINAL_STEP_ID = "msg_final_step"
_TOOL_USE_BLOCK = {"type": "tool_use", "id": "toolu_1", "name": "lookup", "input": {"city": "Paris"}}


def _sends_a_tool_result(request_body):
    last = request_body["messages"][-1]
    return isinstance(last["content"], list) and any(block.get("type") == "tool_result" for block in last["content"])


def _tool_step_message(final):
    if final:
        return {**_MESSAGE, "id": FINAL_STEP_ID, "content": [{"type": "text", "text": "sunny"}]}
    return {**_MESSAGE, "id": TOOL_STEP_ID, "content": [_TOOL_USE_BLOCK], "stop_reason": "tool_use"}


def _tool_step_events(message):
    block = message["content"][0]
    if block["type"] == "tool_use":
        opening = {**block, "input": {}}
        delta = {"type": "input_json_delta", "partial_json": json.dumps(block["input"])}
    else:
        opening = {"type": "text", "text": ""}
        delta = {"type": "text_delta", "text": block["text"]}
    events = [
        ("message_start", {"type": "message_start", "message": {
            **message, "content": [], "stop_reason": None, "usage": {**_USAGE, "output_tokens": 1}}}),
        ("content_block_start", {"type": "content_block_start", "index": 0, "content_block": opening}),
        ("content_block_delta", {"type": "content_block_delta", "index": 0, "delta": delta}),
        ("content_block_stop", {"type": "content_block_stop", "index": 0}),
        ("message_delta", {"type": "message_delta",
                           "delta": {"stop_reason": message["stop_reason"], "stop_sequence": None},
                           "usage": {"output_tokens": OUTPUT_TOKENS}}),
        ("message_stop", {"type": "message_stop"}),
    ]
    return "".join(f"event: {name}\ndata: {json.dumps(data)}\n\n" for name, data in events).encode()


class ToolRunnerStub(StubAnthropic):
    """Answers the first request with a tool call and the one carrying its result with the final reply."""

    def _respond(self, request):
        assert request.url.host == STUB_HOST
        self.requests.append(request)
        body = json.loads(request.content)
        message = _tool_step_message(final=_sends_a_tool_result(body))
        if body.get("stream"):
            return httpx2.Response(200, headers={"content-type": "text/event-stream"},
                                   content=_tool_step_events(message))
        return httpx2.Response(200, headers={"content-type": "application/json"}, content=json.dumps(message).encode())


@anthropic.beta_tool
def lookup(city: str) -> str:
    """Look up the weather in a city."""
    return "sunny"


@anthropic.beta_async_tool(name="lookup")
async def async_lookup(city: str) -> str:
    """Look up the weather in a city."""
    return "sunny"


def _run_sync_tool_runner(client, stream):
    runner = client.beta.messages.tool_runner(tools=[lookup], stream=stream, **BETA_REQUEST)
    for step in runner:
        if stream:
            list(step)


async def _run_async_tool_runner(client, stream):
    runner = client.beta.messages.tool_runner(tools=[async_lookup], stream=stream, **BETA_REQUEST)
    async for step in runner:
        if stream:
            [event async for event in step]


class TestBetaToolRunner:
    """tool_runner drives beta parse (or beta stream) once per step, so each step is one record."""

    @pytest.fixture
    def runner_stub(self):
        return ToolRunnerStub()

    def assert_one_record_per_step(self, runner_stub, records, streamed):
        assert len(runner_stub.requests) == 2
        assert [record["transaction_id"] for record in records] == [TOOL_STEP_ID, FINAL_STEP_ID], records
        for record in records:
            assert record["input_token_count"] == INPUT_TOKENS
            assert record["output_token_count"] == OUTPUT_TOKENS
            assert record["is_streamed"] is streamed

    @pytest.mark.parametrize("stream", [False, True], ids=["parse-steps", "stream-steps"])
    def test_sync(self, runner_stub, records, stream):
        _run_sync_tool_runner(runner_stub.sync_client(), stream)
        self.assert_one_record_per_step(runner_stub, records, stream)

    @pytest.mark.parametrize("stream", [False, True], ids=["parse-steps", "stream-steps"])
    def test_async(self, runner_stub, records, stream):
        asyncio.run(_run_async_tool_runner(runner_stub.async_client(), stream))
        self.assert_one_record_per_step(runner_stub, records, stream)


def _send_sync(thread):
    thread.send()


def _stream_sync(thread):
    with thread.stream() as stream:
        list(stream)


async def _send_async(thread):
    await thread.send()


async def _stream_async(thread):
    async with thread.stream() as stream:
        [event async for event in stream]


@pytest.mark.skipif(not hasattr(anthropic.resources.beta.Messages, "thread"),
                    reason="client.beta.messages.thread arrived in anthropic 2.3.0")
class TestBetaMessageThread:
    """A thread sends nothing itself; each turn is one beta messages.create call, so one record per turn."""

    @pytest.mark.parametrize("turn, streamed", [(_send_sync, False), (_stream_sync, True)], ids=["send", "stream"])
    def test_sync_turn(self, stub, records, turn, streamed):
        turn(stub.sync_client().beta.messages.thread(**BETA_REQUEST))
        assert_one_record(stub, records, streamed=streamed)

    @pytest.mark.parametrize("turn, streamed", [(_send_async, False), (_stream_async, True)], ids=["send", "stream"])
    def test_async_turn(self, stub, records, turn, streamed):
        asyncio.run(turn(stub.async_client().beta.messages.thread(**BETA_REQUEST)))
        assert_one_record(stub, records, streamed=streamed)

    def test_each_turn_carries_decorator_metadata(self, stub, records):
        @revenium_metadata(organization_id="org-thread")
        def two_turns():
            thread = stub.sync_client().beta.messages.thread(**BETA_REQUEST)
            thread.send()
            thread.send({"role": "user", "content": "again"})

        two_turns()
        assert len(stub.requests) == 2
        assert [record["organization_name"] for record in records] == ["org-thread", "org-thread"]


ADVISOR_MODEL = "claude-opus-5"
FIRST_PASS_INPUT_TOKENS = 5
_ITERATIONS = [
    {"type": "message", "input_tokens": FIRST_PASS_INPUT_TOKENS, "output_tokens": 2,
     "cache_creation_input_tokens": 0, "cache_read_input_tokens": CACHE_READ_TOKENS},
    {"type": "advisor_message", "model": ADVISOR_MODEL, "input_tokens": 823, "output_tokens": 1612,
     "cache_creation_input_tokens": 40, "cache_read_input_tokens": 7,
     "cache_creation": {"ephemeral_5m_input_tokens": 40, "ephemeral_1h_input_tokens": 0}},
    {"type": "compaction", "input_tokens": 1800, "output_tokens": 350},
    {"type": "message", "input_tokens": INPUT_TOKENS - FIRST_PASS_INPUT_TOKENS, "output_tokens": 3,
     "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0},
]
_MESSAGE_ONLY_ITERATIONS = [_ITERATIONS[0], _ITERATIONS[3]]
_EXECUTOR_CACHE_SPLIT = {"ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 0}


def iteration_stub(iterations=_ITERATIONS, start_input_tokens=FIRST_PASS_INPUT_TOKENS, stub_class=StubAnthropic):
    """A turn whose executor ran several passes: message_start carries only the first
    pass's input, and the cumulative input and the iterations arrive on message_delta."""
    usage = {**_USAGE, "cache_creation": _EXECUTOR_CACHE_SPLIT, "iterations": iterations}
    return stub_class(
        usage,
        start_usage={**usage, "input_tokens": start_input_tokens, "output_tokens": 1, "iterations": None},
        delta_usage={"output_tokens": OUTPUT_TOKENS, "input_tokens": INPUT_TOKENS,
                     "cache_creation_input_tokens": 0, "cache_read_input_tokens": CACHE_READ_TOKENS,
                     "iterations": iterations},
    )


def _create_sync(messages, request):
    messages.create(**request)


def _create_async(messages, request):
    asyncio.run(messages.create(**request))


def _create_stream_sync(messages, request):
    _drain_sync_create(messages, request)


def _create_stream_async(messages, request):
    asyncio.run(_drain_async_create(messages, request))


def _stream_helper_sync(messages, request):
    _drain_sync_stream(messages, request)


def _stream_helper_async(messages, request):
    asyncio.run(_drain_async_stream(messages, request))


def _parse_sync(messages, request):
    messages.parse(**request)


def _parse_async(messages, request):
    asyncio.run(messages.parse(**request))


_SYNC_FORMS = {_create_sync, _create_stream_sync, _stream_helper_sync, _parse_sync}
_STREAMED_FORMS = {_create_stream_sync, _create_stream_async, _stream_helper_sync, _stream_helper_async}
_FORMS = [_create_sync, _create_async, _create_stream_sync, _create_stream_async,
          _stream_helper_sync, _stream_helper_async, _parse_sync, _parse_async]


def _call(form, resource, stub, request):
    client = stub.sync_client() if form in _SYNC_FORMS else stub.async_client()
    form(resource(client), request)


_ITERATION_CELLS = [
    pytest.param(form, resource, request, id=f"{form.__name__.lstrip('_')}-{resource_id}")
    for resource, request, resource_id in [(_beta, BETA_REQUEST, "beta"), (_standard, REQUEST, "standard")]
    for form in _FORMS
]


def _read_text_stream(stream):
    return list(stream.text_stream)


def _read_final_message(stream):
    return stream.get_final_message()


def _read_final_text(stream):
    return stream.get_final_text()


def _read_until_done(stream):
    return stream.until_done()


def _read_one_chunk_then_final_message(stream):
    next(iter(stream.text_stream))
    return stream.get_final_message()


async def _aread_text_stream(stream):
    return [chunk async for chunk in stream.text_stream]


async def _aread_final_message(stream):
    return await stream.get_final_message()


async def _aread_final_text(stream):
    return await stream.get_final_text()


async def _aread_until_done(stream):
    return await stream.until_done()


_SYNC_READS = [_read_text_stream, _read_final_message, _read_final_text, _read_until_done,
               _read_one_chunk_then_final_message]
_ASYNC_READS = [_aread_text_stream, _aread_final_message, _aread_final_text, _aread_until_done]


def _read_stream_helper(read, resource, stub, request):
    if read in _SYNC_READS:
        with resource(stub.sync_client()).stream(**request) as stream:
            read(stream)
        return

    async def run():
        async with resource(stub.async_client()).stream(**request) as stream:
            await read(stream)
    asyncio.run(run())

_SSE_EVENT_TYPES = ["message_start", "content_block_start", "content_block_delta", "content_block_stop",
                    "message_delta", "message_stop"]


class _PlainEventIterator:
    """The helper's events behind __next__ and __anext__ only: an iterator that is not a generator."""

    def __init__(self, events):
        self._events = events

    def __iter__(self):
        return self

    def __next__(self):
        return next(self._events)

    def __aiter__(self):
        return self

    def __anext__(self):
        return self._events.__anext__()


class _AssignedOnce:
    """An attribute that takes the vendor's own assignment and refuses every later one."""

    def __get__(self, instance, owner):
        return instance.__dict__["_assigned_once"]

    def __set__(self, instance, value):
        if "_assigned_once" in instance.__dict__:
            raise AttributeError("read-only")
        instance.__dict__["_assigned_once"] = value


def _serve_a_plain_event_iterator(monkeypatch, stream_class):
    events = stream_class.__stream__
    monkeypatch.setattr(stream_class, "__stream__", lambda self: _PlainEventIterator(events(self)))


def _refuse_a_new_event_iterator(monkeypatch, stream_class):
    monkeypatch.setattr(stream_class, "_iterator", _AssignedOnce(), raising=False)


_INHERITED_FIELDS = ("trace_id", "subscriber", "organization_name", "product_name", "agent", "task_type",
                     "provider", "model_source", "is_streamed", "operation_type", "request_time")


def assert_iteration_records(stub, records, *, streamed, provider="ANTHROPIC", compaction_model=REQUEST["model"]):
    assert len(stub.requests) == 1
    assert len(records) == 3, records
    main, advisor, compaction = records
    assert_one_record(stub, [main], streamed=streamed, provider=provider)
    assert main["input_token_count"] == INPUT_TOKENS

    assert advisor["transaction_id"] == f"{MESSAGE_ID}-iteration-1"
    assert advisor["parent_transaction_id"] == MESSAGE_ID
    assert advisor["operation_subtype"] == "advisor_message"
    assert advisor["model"] == ADVISOR_MODEL
    assert (advisor["input_token_count"], advisor["output_token_count"], advisor["total_token_count"]) == (
        823, 1612, 2435)
    assert advisor["cache_read_token_count"] == 7
    assert advisor["cache_creation_token_count"] == 40
    assert advisor["cache_creation5m_token_count"] == 40
    assert advisor["cache_creation1h_token_count"] == 0

    assert compaction["transaction_id"] == f"{MESSAGE_ID}-iteration-2"
    assert compaction["parent_transaction_id"] == MESSAGE_ID
    assert compaction["operation_subtype"] == "compaction"
    assert compaction["model"] == compaction_model
    assert (compaction["input_token_count"], compaction["output_token_count"],
            compaction["total_token_count"]) == (1800, 350, 2150)
    assert compaction["cache_read_token_count"] == 0
    assert compaction["cache_creation_token_count"] == 0
    assert "cache_creation5m_token_count" not in compaction
    assert "cache_creation1h_token_count" not in compaction

    for extra in (advisor, compaction):
        for field in _INHERITED_FIELDS:
            assert extra[field] == main[field], field
        assert extra["system_prompt"] is None
        assert extra["input_messages"] is None
        assert extra["output_response"] is None

    assert sum(r["input_token_count"] for r in records) == INPUT_TOKENS + 823 + 1800
    assert sum(r["output_token_count"] for r in records) == OUTPUT_TOKENS + 1612 + 350
    return records


@pytest.fixture
def submissions():
    """Every submit_ai_event call as (payload, idempotency_key)."""
    captured = []

    def record(operation, payload, idempotency_key=None):
        captured.append((payload, idempotency_key))
        return MagicMock(status_code=201)

    with patch("revenium_middleware.anthropic.middleware._get_thread_safe_client", return_value=MagicMock()), \
            patch("revenium_middleware.anthropic.middleware.submit_ai_event", side_effect=record), \
            patch("revenium_middleware.anthropic.middleware._safe_run_async_in_thread",
                  side_effect=_run_metering_synchronously):
        yield captured


def _run_metering_synchronously_in_caller_context(coro_func, *args, **kwargs):
    context = contextvars.copy_context()
    thread = threading.Thread(target=lambda: context.run(asyncio.run, coro_func(*args, **kwargs)))
    thread.start()
    thread.join(timeout=10)
    return thread


class TestUsageIterations:
    """BACK-3997: advisor and compaction passes are billed outside the top-level usage."""

    @pytest.mark.parametrize("form, resource, request_kwargs", _ITERATION_CELLS)
    def test_advisor_and_compaction_get_their_own_records(self, records, form, resource, request_kwargs):
        stub = iteration_stub()
        _call(form, resource, stub, request_kwargs)
        assert_iteration_records(stub, records, streamed=form in _STREAMED_FORMS)

    @pytest.mark.parametrize("form, resource, request_kwargs", [
        pytest.param(_create_sync, _beta, BETA_REQUEST, id="create_sync-beta"),
        pytest.param(_create_stream_sync, _standard, REQUEST, id="create_stream_sync-standard"),
    ])
    def test_pass_records_carry_no_captured_prompt_with_capture_on(self, records, monkeypatch, form, resource,
                                                                    request_kwargs):
        monkeypatch.setenv("REVENIUM_CAPTURE_PROMPTS", "true")
        _call(form, resource, iteration_stub(), dict(request_kwargs, system="SECRET SYSTEM"))

        main, advisor, compaction = records
        for field in ("system_prompt", "input_messages", "output_response"):
            assert main[field] is not None, field
            assert advisor[field] is None, field
            assert compaction[field] is None, field

    @pytest.mark.parametrize("read", _SYNC_READS + _ASYNC_READS, ids=lambda read: read.__name__.lstrip("_"))
    @pytest.mark.parametrize("resource, request_kwargs", RESOURCES)
    def test_stream_helper_bills_the_passes_however_the_stream_is_read(self, records, resource, request_kwargs, read):
        stub = iteration_stub()
        _read_stream_helper(read, resource, stub, request_kwargs)
        assert_iteration_records(stub, records, streamed=True)

    @pytest.mark.parametrize("iterations", [None, [], _MESSAGE_ONLY_ITERATIONS],
                             ids=["none", "empty", "message_only"])
    @pytest.mark.parametrize("form", [_stream_helper_sync, _stream_helper_async], ids=["sync", "async"])
    @pytest.mark.parametrize("resource, request_kwargs", RESOURCES)
    def test_stream_helper_sends_one_record_without_billed_iterations(self, records, form, resource,
                                                                      request_kwargs, iterations):
        stub = iteration_stub(iterations=iterations, start_input_tokens=INPUT_TOKENS)
        _call(form, resource, stub, request_kwargs)
        assert_one_record(stub, records, streamed=True)

    @pytest.mark.parametrize("resource, request_kwargs", RESOURCES)
    def test_stream_helper_left_before_the_final_usage_sends_only_the_main_record(self, records, resource,
                                                                                  request_kwargs):
        stub = iteration_stub(stub_class=UnbufferedStreamStub)
        with resource(stub.sync_client()).stream(**request_kwargs) as stream:
            for event in stream:
                if event.type == "content_block_delta":
                    break
            assert not stub.stream_response.is_closed
        assert stub.stream_response.is_closed
        assert [record["transaction_id"] for record in records] == [MESSAGE_ID]

    @pytest.mark.parametrize("disturb", [_serve_a_plain_event_iterator, _refuse_a_new_event_iterator],
                             ids=["not_a_generator", "not_assignable"])
    @pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
    def test_stream_helper_with_an_unexpected_event_iterator_still_streams_and_sends_one_record(
            self, records, monkeypatch, asynchronous, disturb):
        from anthropic.lib.streaming import AsyncMessageStream, MessageStream
        disturb(monkeypatch, AsyncMessageStream if asynchronous else MessageStream)
        stub = iteration_stub()
        if asynchronous:
            events = asyncio.run(_drain_async_stream(stub.async_client().messages, REQUEST))
        else:
            events = _drain_sync_stream(stub.sync_client().messages, REQUEST)
        assert [event.type for event in events if event.type in _SSE_EVENT_TYPES] == _SSE_EVENT_TYPES
        assert_one_record(stub, records, streamed=True)

    @pytest.mark.parametrize("form", [_create_stream_sync, _create_stream_async], ids=["sync", "async"])
    @pytest.mark.parametrize("resource, request_kwargs", RESOURCES)
    def test_raw_stream_main_record_counts_input_from_message_delta(self, records, form, resource, request_kwargs):
        stub = iteration_stub(iterations=_MESSAGE_ONLY_ITERATIONS)
        _call(form, resource, stub, request_kwargs)
        assert_one_record(stub, records, streamed=True)

    @pytest.mark.parametrize("iterations", [None, [], _MESSAGE_ONLY_ITERATIONS],
                             ids=["none", "empty", "message_only"])
    @pytest.mark.parametrize("form", [_create_sync, _create_stream_sync], ids=["create", "create_stream"])
    def test_no_extra_records_without_billed_iterations(self, records, form, iterations):
        stub = iteration_stub(iterations=iterations, start_input_tokens=INPUT_TOKENS)
        _call(form, _beta, stub, BETA_REQUEST)
        assert_one_record(stub, records, streamed=form in _STREAMED_FORMS)

    def test_iteration_records_carry_their_transaction_id_as_idempotency_key(self, submissions):
        _call(_create_sync, _beta, iteration_stub(), BETA_REQUEST)
        assert [key for _, key in submissions] == [
            None, f"{MESSAGE_ID}-iteration-1", f"{MESSAGE_ID}-iteration-2"]

    @pytest.mark.parametrize("form, resource, request_kwargs", [
        pytest.param(_create_sync, _beta, BETA_REQUEST, id="create_sync-beta"),
        pytest.param(_stream_helper_sync, _standard, REQUEST, id="stream_helper_sync-standard"),
    ])
    def test_idempotency_block_numbering_is_unchanged_by_iteration_records(self, form, resource, request_kwargs):
        metering_client = MagicMock()
        metering_client.ai.create_completion = MagicMock(return_value=MagicMock(status_code=201))
        stub = iteration_stub()
        with patch("revenium_middleware._core.metering_submission.current_client", lambda: metering_client), \
                patch("revenium_middleware.anthropic.middleware._get_thread_safe_client",
                      return_value=metering_client), \
                patch("revenium_middleware.anthropic.middleware._safe_run_async_in_thread",
                      side_effect=_run_metering_synchronously_in_caller_context):
            with idempotency_block("blk"):
                _call(form, resource, stub, request_kwargs)
                _call(form, resource, stub, request_kwargs)
        sent = [(call.kwargs["transaction_id"], call.kwargs["extra_headers"]["Idempotency-Key"])
                for call in metering_client.ai.create_completion.call_args_list]
        assert sent == [
            (MESSAGE_ID, "blk"),
            (f"{MESSAGE_ID}-iteration-1", "blk-iteration-1"),
            (f"{MESSAGE_ID}-iteration-2", "blk-iteration-2"),
            (MESSAGE_ID, "blk:2"),
            (f"{MESSAGE_ID}-iteration-1", "blk:2-iteration-1"),
            (f"{MESSAGE_ID}-iteration-2", "blk:2-iteration-2"),
        ]

    def test_over_long_block_key_hashes_the_iteration_keys(self):
        metering_client = MagicMock()
        metering_client.ai.create_completion = MagicMock(return_value=MagicMock(status_code=201))
        block_key = "k" * 255
        digest = hashlib.sha256(block_key.encode()).hexdigest()
        with patch("revenium_middleware._core.metering_submission.current_client", lambda: metering_client), \
                patch("revenium_middleware.anthropic.middleware._get_thread_safe_client",
                      return_value=metering_client), \
                patch("revenium_middleware.anthropic.middleware._safe_run_async_in_thread",
                      side_effect=_run_metering_synchronously_in_caller_context):
            with idempotency_block(block_key):
                _call(_create_sync, _beta, iteration_stub(), BETA_REQUEST)
        keys = [call.kwargs["extra_headers"]["Idempotency-Key"]
                for call in metering_client.ai.create_completion.call_args_list]
        assert keys == [block_key, f"{digest}-iteration-1", f"{digest}-iteration-2"]
        assert all(len(key) <= 255 for key in keys)

    @pytest.mark.parametrize("key_length, hashed", [(255 - len("-iteration-1"), False),
                                                    (255 - len("-iteration-1") + 1, True)])
    def test_iteration_idempotency_key_boundary(self, key_length, hashed):
        from revenium_middleware._core.usage_iterations import iteration_idempotency_key
        key = "k" * key_length
        result = iteration_idempotency_key(key, 1)
        assert len(result) <= 255
        expected = hashlib.sha256(key.encode()).hexdigest() if hashed else key
        assert result == f"{expected}-iteration-1"

    def test_iteration_records_still_sent_when_main_submit_fails(self, submissions):
        def reject_main_record(operation, payload, idempotency_key=None):
            submissions.append((payload, idempotency_key))
            if payload["transaction_id"] == MESSAGE_ID:
                raise ValueError("main record rejected")
            return MagicMock(status_code=201)

        with patch("revenium_middleware.anthropic.middleware.submit_ai_event", side_effect=reject_main_record):
            message = iteration_stub().sync_client().beta.messages.create(**BETA_REQUEST)
        assert message.id == MESSAGE_ID
        assert [payload["transaction_id"] for payload, _ in submissions] == [
            MESSAGE_ID, f"{MESSAGE_ID}-iteration-1", f"{MESSAGE_ID}-iteration-2"]

    def test_zero_token_iteration_still_recorded(self, records):
        silent_advisor = {"type": "advisor_message", "model": ADVISOR_MODEL, "input_tokens": 0, "output_tokens": 0}
        stub = iteration_stub(iterations=[_ITERATIONS[0], silent_advisor, _ITERATIONS[3]])
        _call(_create_sync, _beta, stub, BETA_REQUEST)
        assert len(records) == 2, records
        advisor = records[1]
        assert advisor["operation_subtype"] == "advisor_message"
        assert (advisor["input_token_count"], advisor["output_token_count"], advisor["total_token_count"]) == (0, 0, 0)

    def test_advisor_record_model_is_iteration_model_verbatim_on_bedrock(self, records):
        with patch("revenium_middleware.anthropic.middleware.detect_provider", return_value=Provider.BEDROCK), \
                patch("revenium_middleware.anthropic.middleware._handle_bedrock_request") as fast_create:
            stub = iteration_stub()
            _call(_create_sync, _beta, stub, BETA_REQUEST)
        fast_create.assert_not_called()
        assert_iteration_records(stub, records, streamed=False, provider="AWS")

    def test_streaming_response_raw_stream_meters_iterations(self, records):
        stub = iteration_stub()
        with stub.sync_client().beta.messages.with_streaming_response.create(stream=True, **BETA_REQUEST) as response:
            b"".join(response.iter_bytes())
        assert_iteration_records(stub, records, streamed=True)
