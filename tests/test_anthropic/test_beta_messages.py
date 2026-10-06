"""client.beta.messages and messages.parse() are metered once per call (BACK-3606).

The real anthropic package runs over an httpx2 stub. The beta response carries
a beta-only content block and the beta-only usage fields, which the standard
Message model rejects, so the beta path is exercised on its own types.
"""
import asyncio
import json
import threading
from unittest.mock import MagicMock, patch

import pytest

anthropic = pytest.importorskip("anthropic")
httpx2 = pytest.importorskip("httpx2")

import revenium_middleware.anthropic  # noqa: E402,F401  installs the patches
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
_SSE_EVENTS = [
    ("message_start", {"type": "message_start", "message": {
        **_MESSAGE, "content": [], "stop_reason": None, "usage": {**_USAGE, "output_tokens": 1}}}),
    ("content_block_start", {"type": "content_block_start", "index": 0,
                             "content_block": {"type": "text", "text": ""}}),
    ("content_block_delta", {"type": "content_block_delta", "index": 0,
                             "delta": {"type": "text_delta", "text": "hi"}}),
    ("content_block_stop", {"type": "content_block_stop", "index": 0}),
    ("message_delta", {"type": "message_delta", "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                       "usage": {"output_tokens": OUTPUT_TOKENS}}),
    ("message_stop", {"type": "message_stop"}),
]
SSE_BODY = "".join(f"event: {name}\ndata: {json.dumps(data)}\n\n" for name, data in _SSE_EVENTS).encode()
MESSAGE_BODY = json.dumps(_MESSAGE).encode()
STANDARD_MESSAGE_BODY = json.dumps({**_MESSAGE, "content": [{"type": "text", "text": "hi"}]}).encode()


class StubAnthropic:
    def __init__(self):
        self.requests = []

    def _respond(self, request):
        assert request.url.host == STUB_HOST
        self.requests.append(request)
        if json.loads(request.content).get("stream"):
            return httpx2.Response(200, headers={"content-type": "text/event-stream"}, content=SSE_BODY)
        body = MESSAGE_BODY if "beta=true" in str(request.url) else STANDARD_MESSAGE_BODY
        return httpx2.Response(200, headers={"content-type": "application/json"}, content=body)

    def sync_client(self):
        return anthropic.Anthropic(api_key="sk-ant-test", base_url=f"http://{STUB_HOST}", max_retries=0,
                                   http_client=httpx2.Client(transport=httpx2.MockTransport(self._respond)))

    def async_client(self):
        return anthropic.AsyncAnthropic(api_key="sk-ant-test", base_url=f"http://{STUB_HOST}", max_retries=0,
                                        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(self._respond)))


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
