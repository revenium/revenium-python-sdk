"""Claude beta calls through the Bedrock and Vertex AI clients are metered and capped (BACK-4243).

``anthropic.lib.bedrock`` and ``anthropic.lib.vertex`` copy the first-party beta
``create``, ``parse`` and ``stream`` functions into their own classes when
``anthropic`` is imported, so wrapping the first-party class alone leaves them
bare. The real anthropic clients run here over an httpx2 stub that answers as
Bedrock (AWS event stream) or Vertex AI (server-sent events) would.
"""
import asyncio
import importlib
import inspect
import json
import pkgutil
import sys
from unittest.mock import MagicMock, patch

import pytest

anthropic = pytest.importorskip("anthropic")
httpx2 = pytest.importorskip("httpx2")
wrapt = pytest.importorskip("wrapt")

import revenium_middleware.anthropic  # noqa: E402,F401  installs the patches
from entry_points.caps import breached_rule, scoped, seed_rules  # noqa: E402
from revenium_middleware._core.exceptions import BudgetExceededError  # noqa: E402
from test_patch_coverage import wrapper_owners  # noqa: E402

from .test_beta_messages import _run_metering_synchronously, _sends_a_tool_result, async_lookup, lookup  # noqa: E402
from .test_bedrock_sync_client import _SCRUBBED_ENV, event_stream_frame  # noqa: E402

MESSAGE_ID = "msg_platform_beta"
TOOL_STEP_ID = "msg_platform_tool_step"
FINAL_STEP_ID = "msg_platform_final_step"
RESPONSE_MODEL = "claude-haiku-4-5-20251001"
INPUT_TOKENS = 21
OUTPUT_TOKENS = 8
BETAS = ["context-management-2025-06-27"]
MESSAGES = [{"role": "user", "content": "hi"}]
BEDROCK_REQUEST = dict(model="us.anthropic.claude-haiku-4-5-20251001-v1:0", max_tokens=16, messages=MESSAGES,
                       betas=BETAS)
VERTEX_REQUEST = dict(model="claude-haiku-4-5@20251001", max_tokens=16, messages=MESSAGES, betas=BETAS)
FOUNDRY_REQUEST = dict(model=RESPONSE_MODEL, max_tokens=16, messages=MESSAGES, betas=BETAS)
RULE_ID = 4243

_TOOL_USE_BLOCK = {"type": "tool_use", "id": "toolu_1", "name": "lookup", "input": {"city": "Paris"}}


def _message(message_id=MESSAGE_ID, content=None, stop_reason="end_turn"):
    return {
        "id": message_id, "type": "message", "role": "assistant", "model": RESPONSE_MODEL,
        "content": content if content is not None else [{"type": "text", "text": "hello"}],
        "stop_reason": stop_reason, "stop_sequence": None,
        "usage": {"input_tokens": INPUT_TOKENS, "output_tokens": OUTPUT_TOKENS},
    }


def _stream_events(message):
    events = [{"type": "message_start", "message": {
        **message, "content": [], "stop_reason": None, "usage": {"input_tokens": INPUT_TOKENS, "output_tokens": 1}}}]
    for index, block in enumerate(message["content"]):
        if block["type"] == "tool_use":
            start = {**block, "input": {}}
            delta = {"type": "input_json_delta", "partial_json": json.dumps(block["input"])}
        else:
            start = {**block, "text": ""}
            delta = {"type": "text_delta", "text": block["text"]}
        events += [
            {"type": "content_block_start", "index": index, "content_block": start},
            {"type": "content_block_delta", "index": index, "delta": delta},
            {"type": "content_block_stop", "index": index},
        ]
    return events + [
        {"type": "message_delta", "delta": {"stop_reason": message["stop_reason"], "stop_sequence": None},
         "usage": {"output_tokens": OUTPUT_TOKENS}},
        {"type": "message_stop"},
    ]


class PlatformStub:
    """Answers as Bedrock or Vertex AI would, and records every request the client sends."""

    def __init__(self, platform, tool_runner=False):
        self.platform = platform
        self.tool_runner = tool_runner
        self.requests = []

    def _reply_to(self, body):
        if not self.tool_runner:
            return _message()
        if _sends_a_tool_result(body):
            return _message(FINAL_STEP_ID)
        return _message(TOOL_STEP_ID, content=[_TOOL_USE_BLOCK], stop_reason="tool_use")

    def _is_stream(self, request, body):
        if self.platform == "bedrock":
            return request.url.path.endswith("/invoke-with-response-stream")
        return bool(body.get("stream"))

    def respond(self, request):
        self.requests.append(request)
        body = json.loads(request.content)
        message = self._reply_to(body)
        if not self._is_stream(request, body):
            return httpx2.Response(200, headers={"content-type": "application/json"},
                                   content=json.dumps(message).encode())
        events = _stream_events(message)
        if self.platform == "bedrock":
            return httpx2.Response(200, headers={"content-type": "application/vnd.amazon.eventstream"},
                                   content=b"".join(event_stream_frame(event) for event in events))
        sse = "".join(f"event: {event['type']}\ndata: {json.dumps(event)}\n\n" for event in events)
        return httpx2.Response(200, headers={"content-type": "text/event-stream"}, content=sse.encode())

    def client(self, asynchronous):
        transport = httpx2.MockTransport(self.respond)
        http_client = httpx2.AsyncClient(transport=transport) if asynchronous else httpx2.Client(transport=transport)
        if self.platform == "bedrock":
            cls = anthropic.AsyncAnthropicBedrock if asynchronous else anthropic.AnthropicBedrock
            return cls(aws_access_key="AKIATESTONLY", aws_secret_key="test-only", aws_region="us-east-1",
                       max_retries=0, http_client=http_client)
        if self.platform == "vertex":
            cls = anthropic.AsyncAnthropicVertex if asynchronous else anthropic.AnthropicVertex
            return cls(region="us-east5", project_id="test-project", access_token="test-token",
                       max_retries=0, http_client=http_client)
        cls = anthropic.AsyncAnthropicFoundry if asynchronous else anthropic.AnthropicFoundry
        return cls(api_key="test-key", resource="test-resource", max_retries=0, http_client=http_client)


PLATFORMS = {
    "bedrock": (BEDROCK_REQUEST, "AWS"),
    "vertex": (VERTEX_REQUEST, "ANTHROPIC"),
    "foundry": (FOUNDRY_REQUEST, "Foundry"),
}


@pytest.fixture(autouse=True)
def _scrubbed_env(monkeypatch):
    for name in _SCRUBBED_ENV:
        monkeypatch.delenv(name, raising=False)


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


def _sync_create(messages, request):
    messages.create(**request)


def _sync_create_stream(messages, request):
    list(messages.create(stream=True, **request))


def _sync_stream_helper(messages, request):
    with messages.stream(**request) as stream:
        list(stream)


def _sync_parse(messages, request):
    messages.parse(**request)


def _sync_raw_create(messages, request):
    messages.with_raw_response.create(**request).parse()


def _sync_raw_create_stream(messages, request):
    list(messages.with_raw_response.create(stream=True, **request).parse())


def _sync_streaming_response_create(messages, request):
    with messages.with_streaming_response.create(**request) as response:
        response.parse()


async def _async_create(messages, request):
    await messages.create(**request)


async def _async_create_stream(messages, request):
    [event async for event in await messages.create(stream=True, **request)]


async def _async_stream_helper(messages, request):
    async with messages.stream(**request) as stream:
        [event async for event in stream]


async def _async_parse(messages, request):
    await messages.parse(**request)


async def _async_raw_create(messages, request):
    await (await messages.with_raw_response.create(**request)).parse()


async def _async_raw_create_stream(messages, request):
    stream = await (await messages.with_raw_response.create(stream=True, **request)).parse()
    [event async for event in stream]


async def _async_streaming_response_create(messages, request):
    async with messages.with_streaming_response.create(**request) as response:
        await response.parse()


FORMS = [
    pytest.param(_sync_create, False, id="sync-create"),
    pytest.param(_sync_create_stream, True, id="sync-create-stream"),
    pytest.param(_sync_stream_helper, True, id="sync-stream-helper"),
    pytest.param(_sync_parse, False, id="sync-parse"),
    pytest.param(_sync_raw_create, False, id="sync-raw-create"),
    pytest.param(_sync_raw_create_stream, True, id="sync-raw-create-stream"),
    pytest.param(_sync_streaming_response_create, False, id="sync-streaming-response-create"),
    pytest.param(_async_create, False, id="async-create"),
    pytest.param(_async_create_stream, True, id="async-create-stream"),
    pytest.param(_async_stream_helper, True, id="async-stream-helper"),
    pytest.param(_async_parse, False, id="async-parse"),
    pytest.param(_async_raw_create, False, id="async-raw-create"),
    pytest.param(_async_raw_create_stream, True, id="async-raw-create-stream"),
    pytest.param(_async_streaming_response_create, False, id="async-streaming-response-create"),
]


def _call(stub, form, request):
    if inspect.iscoroutinefunction(form):
        asyncio.run(form(stub.client(asynchronous=True).beta.messages, request))
    else:
        form(stub.client(asynchronous=False).beta.messages, request)


def assert_one_record(stub, records, *, provider, streamed, transaction_id=MESSAGE_ID):
    assert len(stub.requests) == 1
    assert len(records) == 1, records
    record = records[0]
    assert record["provider"] == provider
    assert record["input_token_count"] == INPUT_TOKENS
    assert record["output_token_count"] == OUTPUT_TOKENS
    assert record["transaction_id"] == transaction_id
    assert record["is_streamed"] is streamed


@pytest.mark.parametrize("platform", ["bedrock", "vertex"])
@pytest.mark.parametrize("form, streamed", FORMS)
def test_every_platform_beta_call_sends_one_record_with_the_platform_label(records, platform, form, streamed):
    stub = PlatformStub(platform)
    request, provider = PLATFORMS[platform]
    _call(stub, form, request)
    assert_one_record(stub, records, provider=provider, streamed=streamed)


@pytest.mark.parametrize("platform", ["bedrock", "vertex"])
@pytest.mark.parametrize("form, streamed", FORMS)
def test_a_breached_cap_refuses_every_platform_beta_call_before_the_cloud(
        monkeypatch, records, platform, form, streamed):
    stub = PlatformStub(platform)
    request, provider = PLATFORMS[platform]
    seed_rules(monkeypatch, breached_rule(RULE_ID, scoped("PROVIDER", "IS", provider)))
    with pytest.raises(BudgetExceededError):
        _call(stub, form, request)
    assert stub.requests == []
    assert records == []


def _plain_create(asynchronous):
    if asynchronous:
        return lambda client, request: asyncio.run(client.messages.create(**_without_betas(request)))
    return lambda client, request: client.messages.create(**_without_betas(request))


def _beta_create(asynchronous):
    if asynchronous:
        return lambda client, request: asyncio.run(client.beta.messages.create(**request))
    return lambda client, request: client.beta.messages.create(**request)


def _without_betas(request):
    return {key: value for key, value in request.items() if key != "betas"}


@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
def test_a_breached_cap_on_the_dash_spelled_claude_model_refuses_a_vertex_at_spelled_call(
        monkeypatch, records, asynchronous):
    """Vertex AI names the model ``claude-haiku-4-5@20251001``; the cap names it
    ``claude-haiku-4-5-20251001`` (BACK-4307)."""
    stub = PlatformStub("vertex")
    seed_rules(monkeypatch, breached_rule(RULE_ID, scoped("MODEL", "IS", RESPONSE_MODEL)))

    with pytest.raises(BudgetExceededError):
        _plain_create(asynchronous)(stub.client(asynchronous), VERTEX_REQUEST)

    assert stub.requests == []
    assert records == []


@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("platform, call", [
    pytest.param("bedrock", _plain_create, id="bedrock-plain"),
    pytest.param("vertex", _plain_create, id="vertex-plain"),
    pytest.param("foundry", _beta_create, id="foundry-beta"),
])
def test_plain_and_foundry_calls_still_send_exactly_one_record(records, platform, call, asynchronous):
    stub = PlatformStub(platform)
    request, provider = PLATFORMS[platform]
    call(asynchronous)(stub.client(asynchronous), request)
    assert_one_record(stub, records, provider=provider, streamed=False)


def _run_sync_tool_runner(client, request, stream):
    runner = client.beta.messages.tool_runner(tools=[lookup], stream=stream, **request)
    for step in runner:
        if stream:
            list(step)


async def _run_async_tool_runner(client, request, stream):
    runner = client.beta.messages.tool_runner(tools=[async_lookup], stream=stream, **request)
    async for step in runner:
        if stream:
            [event async for event in step]


@pytest.mark.parametrize("platform", ["bedrock", "vertex"])
@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("stream", [False, True], ids=["parse-steps", "stream-steps"])
def test_tool_runner_on_platform_clients_sends_one_record_per_round(records, platform, asynchronous, stream):
    stub = PlatformStub(platform, tool_runner=True)
    request, provider = PLATFORMS[platform]
    client = stub.client(asynchronous)
    if asynchronous:
        asyncio.run(_run_async_tool_runner(client, request, stream))
    else:
        _run_sync_tool_runner(client, request, stream)
    assert len(stub.requests) == 2
    assert [record["transaction_id"] for record in records] == [TOOL_STEP_ID, FINAL_STEP_ID], records
    for record in records:
        assert record["provider"] == provider
        assert record["is_streamed"] is stream


METERED_METHODS = ("create", "parse", "stream")
METERED_CLASSES = (
    "anthropic.resources.messages.messages.Messages",
    "anthropic.resources.messages.messages.AsyncMessages",
    "anthropic.resources.beta.messages.messages.Messages",
    "anthropic.resources.beta.messages.messages.AsyncMessages",
)


def _class_at(path):
    module_name, _, class_name = path.rpartition(".")
    return getattr(importlib.import_module(module_name), class_name)


def _unwrapped_originals():
    originals = {}
    for path in METERED_CLASSES:
        cls = _class_at(path)
        for method in METERED_METHODS:
            value = vars(cls).get(method)
            if isinstance(value, wrapt.FunctionWrapper):
                originals[id(value.__wrapped__)] = f"{path}.{method}"
    return originals


def _import_every_anthropic_module():
    for module_info in pkgutil.walk_packages(anthropic.__path__, "anthropic.", onerror=lambda name: None):
        try:
            importlib.import_module(module_info.name)
        except Exception:
            continue


def test_no_anthropic_class_holds_an_unwrapped_copy_of_a_metered_method():
    _import_every_anthropic_module()
    originals = _unwrapped_originals()
    assert originals, "the first-party Messages classes are not wrapped"
    copies = []
    for module_name, module in list(sys.modules.items()):
        if not (module_name == "anthropic" or module_name.startswith("anthropic.")):
            continue
        for cls in list(vars(module).values()):
            if not inspect.isclass(cls) or cls.__module__ != module_name:
                continue
            copies += [f"{module_name}.{cls.__qualname__}.{name} copies {originals[id(value)]}"
                       for name, value in vars(cls).items() if id(value) in originals]
    assert copies == []


CLIENT_FACTORIES = [
    pytest.param(lambda: anthropic.Anthropic(api_key="test-key"), id="Anthropic"),
    pytest.param(lambda: anthropic.AsyncAnthropic(api_key="test-key"), id="AsyncAnthropic"),
    pytest.param(lambda: PlatformStub("bedrock").client(False), id="AnthropicBedrock"),
    pytest.param(lambda: PlatformStub("bedrock").client(True), id="AsyncAnthropicBedrock"),
    pytest.param(lambda: PlatformStub("vertex").client(False), id="AnthropicVertex"),
    pytest.param(lambda: PlatformStub("vertex").client(True), id="AsyncAnthropicVertex"),
    pytest.param(lambda: PlatformStub("foundry").client(False), id="AnthropicFoundry"),
    pytest.param(lambda: PlatformStub("foundry").client(True), id="AsyncAnthropicFoundry"),
]


@pytest.mark.parametrize("make_client", CLIENT_FACTORIES)
@pytest.mark.parametrize("method", METERED_METHODS)
def test_every_client_beta_method_has_exactly_one_revenium_wrapper(make_client, method):
    resource_class = type(make_client().beta.messages)
    attribute = inspect.getattr_static(resource_class, method)
    owners = [owner for owner in wrapper_owners(attribute) if owner.startswith("revenium_middleware.anthropic")]
    assert len(owners) == 1, (resource_class, method, owners)
