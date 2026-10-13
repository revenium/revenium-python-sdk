"""Sync Claude calls on Bedrock through the Anthropic SDK's own clients.

BACK-4006: a sync ``AnthropicBedrock`` call sends, signs and returns exactly
what the same call sends, signs and returns without Revenium, and is metered
once. A plain ``Anthropic`` client pointed at a Bedrock URL cannot sign its
own requests, so it keeps the middleware's boto3 shortcut.

BACK-4010: a non-streamed Bedrock call records the subscriber in the nested
form the README documents, as well as the old flat keys.

"Without Revenium" is the SDK's own method called outside the wrapper.

The real anthropic package runs over an httpx2 stub that answers as Bedrock
does, and the boto3 client the middleware would otherwise build is replaced
by a fake, so nothing reaches AWS whichever path serves the call.
"""
import asyncio
import base64
import binascii
import io
import json
import logging
import struct
import threading
from unittest.mock import MagicMock, patch

import pytest

anthropic = pytest.importorskip("anthropic")
httpx2 = pytest.importorskip("httpx2")

import revenium_middleware.anthropic  # noqa: E402,F401  installs the patches
from anthropic.resources.messages.messages import Messages  # noqa: E402

REGION = "eu-west-1"
BEDROCK_HOST = f"bedrock-runtime.{REGION}.amazonaws.com"
BEDROCK_URL = f"https://{BEDROCK_HOST}"
QUALIFIED_MODEL = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
RESPONSE_MODEL = "claude-haiku-4-5-20251001"
MESSAGE_ID = "msg_bdrk_01"
INPUT_TOKENS = 12
OUTPUT_TOKENS = 7
REQUEST = dict(model=QUALIFIED_MODEL, max_tokens=64, messages=[{"role": "user", "content": "hi"}])
SUBSCRIBER = {"id": "user-42", "email": "dev@acme.example"}

_SCRUBBED_ENV = (
    "AWS_BEARER_TOKEN_BEDROCK", "ANTHROPIC_BEDROCK_BASE_URL", "AWS_PROFILE", "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN", "REVENIUM_BEDROCK_DISABLE", "REVENIUM_SELECTIVE_METERING",
    "ANTHROPIC_API_KEY",
)


def bedrock_message(content=None, stop_reason="end_turn"):
    return {
        "id": MESSAGE_ID, "type": "message", "role": "assistant", "model": RESPONSE_MODEL,
        "content": content if content is not None else [{"type": "text", "text": "hello"}],
        "stop_reason": stop_reason, "stop_sequence": None,
        "usage": {"input_tokens": INPUT_TOKENS, "output_tokens": OUTPUT_TOKENS},
    }


def stream_events(message):
    """The Anthropic stream events Bedrock sends for ``message``, one block per content item."""
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
    events += [
        {"type": "message_delta", "delta": {"stop_reason": message["stop_reason"], "stop_sequence": None},
         "usage": {"output_tokens": OUTPUT_TOKENS}},
        {"type": "message_stop"},
    ]
    return events


def _event_stream_header(name, value):
    name, value = name.encode(), value.encode()
    return struct.pack("!B", len(name)) + name + struct.pack("!BH", 7, len(value)) + value


def event_stream_frame(event):
    """One ``application/vnd.amazon.eventstream`` chunk frame, CRCs included, as Bedrock streams it."""
    headers = (_event_stream_header(":event-type", "chunk")
               + _event_stream_header(":content-type", "application/json")
               + _event_stream_header(":message-type", "event"))
    payload = json.dumps({"bytes": base64.b64encode(json.dumps(event).encode()).decode()}).encode()
    prelude = struct.pack("!II", 12 + len(headers) + len(payload) + 4, len(headers))
    prelude += struct.pack("!I", binascii.crc32(prelude) & 0xFFFFFFFF)
    frame = prelude + headers + payload
    return frame + struct.pack("!I", binascii.crc32(frame) & 0xFFFFFFFF)


class BedrockStub:
    """Answers as Bedrock would, and records every request the client sends."""

    def __init__(self):
        self.message = bedrock_message()
        self.status = 200
        self.requests = []

    def respond(self, request):
        self.requests.append(request)
        if self.status != 200:
            return httpx2.Response(self.status, headers={"content-type": "application/json"},
                                   content=json.dumps({"message": "Too many requests"}).encode())
        if request.url.path.endswith("/invoke-with-response-stream"):
            body = b"".join(event_stream_frame(event) for event in stream_events(self.message))
            return httpx2.Response(200, headers={"content-type": "application/vnd.amazon.eventstream"}, content=body)
        return httpx2.Response(200, headers={"content-type": "application/json"},
                               content=json.dumps(self.message).encode())

    def http_client(self):
        return httpx2.Client(transport=httpx2.MockTransport(self.respond))

    def bodies(self):
        return [json.loads(request.content) for request in self.requests]


class FakeBedrockRuntime:
    """Stands in for the boto3 client the middleware's own Bedrock shortcut builds."""

    def __init__(self, stub):
        self.stub = stub
        self.calls = []

    def invoke_model(self, **kwargs):
        self.calls.append(kwargs)
        return {"body": io.BytesIO(json.dumps(self.stub.message).encode())}

    def invoke_model_with_response_stream(self, **kwargs):
        self.calls.append(kwargs)
        return {"body": [{"chunk": {"bytes": json.dumps(event).encode()}}
                         for event in stream_events(self.stub.message)]}


def _run_metering_synchronously(coro_func, *args, **kwargs):
    thread = threading.Thread(target=lambda: asyncio.run(coro_func(*args, **kwargs)))
    thread.start()
    thread.join(timeout=10)
    return thread


@pytest.fixture(autouse=True)
def clean_aws_env(monkeypatch):
    for name in _SCRUBBED_ENV:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def stub():
    return BedrockStub()


@pytest.fixture(autouse=True)
def fake_boto3(stub):
    fake = FakeBedrockRuntime(stub)
    with patch("revenium_middleware.anthropic.bedrock_adapter.get_bedrock_client", return_value=fake):
        yield fake


@pytest.fixture
def records():
    captured = []

    def record(operation, payload, *args, **kwargs):
        captured.append(payload)
        return MagicMock(status_code=201)

    with patch("revenium_middleware.anthropic.middleware._get_thread_safe_client", return_value=MagicMock()), \
            patch("revenium_middleware.anthropic.middleware.submit_ai_event", side_effect=record), \
            patch("revenium_middleware._core.submit_ai_event", side_effect=record), \
            patch("revenium_middleware.anthropic.middleware._safe_run_async_in_thread",
                  side_effect=_run_metering_synchronously):
        yield captured


def anthropic_bedrock_client(stub, **overrides):
    options = dict(aws_access_key="test-access-key", aws_secret_key="test-secret-key", aws_region=REGION)
    options.update(overrides)
    return anthropic.AnthropicBedrock(max_retries=0, http_client=stub.http_client(), **options)


def plain_client_with_a_bedrock_url(stub, base_url=BEDROCK_URL):
    return anthropic.Anthropic(api_key="test-key", base_url=base_url, max_retries=0, http_client=stub.http_client())


CLIENT_KINDS = [
    pytest.param(anthropic_bedrock_client, id="anthropic-bedrock"),
    pytest.param(plain_client_with_a_bedrock_url, id="plain-bedrock-url"),
]


@pytest.mark.parametrize("make_client", CLIENT_KINDS)
@pytest.mark.parametrize("usage_metadata, subscriber", [
    pytest.param({"subscriber": SUBSCRIBER, "organizationName": "AcmeCorp"}, SUBSCRIBER, id="nested"),
    pytest.param({"subscriber_id": "user-42", "subscriber_email": "dev@acme.example"}, SUBSCRIBER, id="flat"),
    pytest.param({"subscriber_id": "user-42", "subscriber_email": "dev@acme.example",
                  "subscriber_credential_name": "key-name", "subscriber_credential": "key-value"},
                 {**SUBSCRIBER, "credential": {"name": "key-name", "value": "key-value"}}, id="flat-credential"),
])
def test_sync_bedrock_create_records_the_subscriber(stub, records, make_client, usage_metadata, subscriber):
    make_client(stub).messages.create(usage_metadata=usage_metadata, **REQUEST)

    assert len(records) == 1, records
    assert records[0]["provider"] == "AWS"
    assert records[0]["subscriber"] == subscriber


TOOL_USE_BLOCK = {"type": "tool_use", "id": "toolu_1", "name": "lookup", "input": {"city": "Paris"}}
TOOLS = [{"name": "lookup", "description": "Weather for a city",
          "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}}}]
TOOL_REQUEST = dict(REQUEST, tools=TOOLS, tool_choice={"type": "any"})
STUB_ANTHROPIC_HOST = "anthropic.stub.invalid"


def create_without_revenium(client, **request):
    return Messages.create.__wrapped__(client.messages, **request)


def stream_without_revenium(client, **request):
    return Messages.stream.__wrapped__(client.messages, **request)


def only_request(stub):
    assert len(stub.requests) == 1, [str(request.url) for request in stub.requests]
    return stub.requests[0]


def take_baseline(stub, call):
    """Run ``call`` without Revenium, then hand back its result and the one request it sent."""
    result = call()
    request = only_request(stub)
    stub.requests.clear()
    return result, request


def native_stop_reason(stub_message, records):
    """The stop reason the middleware records for ``stub_message`` on a direct Anthropic call."""
    client = anthropic.Anthropic(
        api_key="test-key", base_url=f"http://{STUB_ANTHROPIC_HOST}", max_retries=0,
        http_client=httpx2.Client(transport=httpx2.MockTransport(
            lambda request: httpx2.Response(200, json=stub_message))))
    client.messages.create(**REQUEST)
    return records.pop()["stop_reason"]


def blocks(message):
    return [block.model_dump(exclude_none=True) for block in message.content]


def assert_metered_once_through_the_sdk(records, fake_boto3, *, streamed=False):
    assert fake_boto3.calls == []
    assert len(records) == 1, records
    record = records[0]
    assert record["provider"] == "AWS"
    assert record["is_streamed"] is streamed
    assert record["input_token_count"] == INPUT_TOKENS
    assert record["output_token_count"] == OUTPUT_TOKENS
    return record


def test_tools_and_tool_choice_reach_bedrock_and_the_tool_call_comes_back(stub, records, fake_boto3):
    stub.message = bedrock_message([TOOL_USE_BLOCK], stop_reason="tool_use")
    expected_stop_reason = native_stop_reason(stub.message, records)
    baseline, baseline_request = take_baseline(
        stub, lambda: create_without_revenium(anthropic_bedrock_client(stub), **TOOL_REQUEST))

    message = anthropic_bedrock_client(stub).messages.create(usage_metadata={"trace_id": "t-1"}, **TOOL_REQUEST)

    sent = only_request(stub)
    assert json.loads(sent.content) == json.loads(baseline_request.content)
    assert sent.url == baseline_request.url
    assert blocks(message) == blocks(baseline) == [TOOL_USE_BLOCK]
    assert message.stop_reason == baseline.stop_reason == "tool_use"
    record = assert_metered_once_through_the_sdk(records, fake_boto3)
    assert record["stop_reason"] != "END"
    assert record["stop_reason"] == expected_stop_reason


def test_list_form_system_prompt_and_metadata_are_sent_verbatim(stub, records, fake_boto3):
    request = dict(REQUEST, metadata={"user_id": "u1"},
                   system=[{"type": "text", "text": "Be brief.", "cache_control": {"type": "ephemeral"}}])
    _, baseline_request = take_baseline(
        stub, lambda: create_without_revenium(anthropic_bedrock_client(stub), **request))

    anthropic_bedrock_client(stub).messages.create(**request)

    assert json.loads(only_request(stub).content) == json.loads(baseline_request.content)
    assert_metered_once_through_the_sdk(records, fake_boto3)


def test_a_reply_cut_off_at_max_tokens_reports_max_tokens(stub, records, fake_boto3):
    stub.message = bedrock_message(stop_reason="max_tokens")
    baseline, _ = take_baseline(stub, lambda: create_without_revenium(anthropic_bedrock_client(stub), **REQUEST))

    message = anthropic_bedrock_client(stub).messages.create(**REQUEST)

    only_request(stub)
    assert message.stop_reason == baseline.stop_reason == "max_tokens"
    record = assert_metered_once_through_the_sdk(records, fake_boto3)
    assert record["stop_reason"] == "TOKEN_LIMIT"


def test_create_with_stream_true_yields_bedrock_events_and_meters_once(stub, records, fake_boto3):
    baseline_stream, baseline_request = take_baseline(
        stub, lambda: create_without_revenium(anthropic_bedrock_client(stub), stream=True, **REQUEST))
    baseline_events = [event.type for event in baseline_stream]

    stream = anthropic_bedrock_client(stub).messages.create(stream=True, **REQUEST)

    sent = only_request(stub)
    assert sent.url.path.endswith("/invoke-with-response-stream")
    assert sent.url == baseline_request.url
    assert json.loads(sent.content) == json.loads(baseline_request.content)
    assert [event.type for event in stream] == baseline_events
    assert baseline_events[-1] == "message_stop"
    assert_metered_once_through_the_sdk(records, fake_boto3, streamed=True)


def test_raw_streamed_create_returns_bedrocks_bytes_and_meters_once(stub, records, fake_boto3):
    bedrock_body = b"".join(event_stream_frame(event) for event in stream_events(stub.message))

    with anthropic_bedrock_client(stub).messages.with_streaming_response.create(stream=True, **REQUEST) as response:
        body = b"".join(response.iter_bytes())

    assert only_request(stub).url.path.endswith("/invoke-with-response-stream")
    assert body == bedrock_body
    assert_metered_once_through_the_sdk(records, fake_boto3, streamed=True)


def test_messages_stream_returns_the_tool_call(stub, records, fake_boto3):
    stub.message = bedrock_message([TOOL_USE_BLOCK], stop_reason="tool_use")

    def drain(stream_manager):
        with stream_manager as stream:
            return stream.get_final_message()

    baseline, baseline_request = take_baseline(
        stub, lambda: drain(stream_without_revenium(anthropic_bedrock_client(stub), **TOOL_REQUEST)))

    final = drain(anthropic_bedrock_client(stub).messages.stream(**TOOL_REQUEST))

    sent = only_request(stub)
    assert sent.url == baseline_request.url
    assert json.loads(sent.content) == json.loads(baseline_request.content)
    assert blocks(final) == blocks(baseline) == [TOOL_USE_BLOCK]
    assert final.stop_reason == baseline.stop_reason == "tool_use"
    assert_metered_once_through_the_sdk(records, fake_boto3, streamed=True)


def _sigv4_scope(request):
    authorization = request.headers["authorization"]
    credential = authorization.split("Credential=", 1)[1].split(",", 1)[0]
    key_id, _date, region, service, terminator = credential.split("/")
    return key_id, region, service, terminator


def test_the_clients_access_keys_sign_the_request(stub, records, fake_boto3):
    _, baseline_request = take_baseline(
        stub, lambda: create_without_revenium(anthropic_bedrock_client(stub), **REQUEST))

    anthropic_bedrock_client(stub).messages.create(**REQUEST)

    sent = only_request(stub)
    assert sent.url.host == baseline_request.url.host == BEDROCK_HOST
    assert _sigv4_scope(sent) == _sigv4_scope(baseline_request) == (
        "test-access-key", REGION, "bedrock", "aws4_request")
    assert_metered_once_through_the_sdk(records, fake_boto3)


def test_the_clients_profile_signs_the_request(stub, records, fake_boto3, tmp_path, monkeypatch):
    profile = "revenium-bedrock-sync-client-test"
    credentials = tmp_path / "credentials"
    credentials.write_text(f"[{profile}]\naws_access_key_id = profile-access-key\n"
                           "aws_secret_access_key = profile-secret-key\n")
    config = tmp_path / "config"
    config.write_text("")
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", str(credentials))
    monkeypatch.setenv("AWS_CONFIG_FILE", str(config))

    def client():
        return anthropic.AnthropicBedrock(aws_profile=profile, aws_region=REGION, max_retries=0,
                                          http_client=stub.http_client())

    _, baseline_request = take_baseline(stub, lambda: create_without_revenium(client(), **REQUEST))

    client().messages.create(**REQUEST)

    assert _sigv4_scope(only_request(stub)) == _sigv4_scope(baseline_request) == (
        "profile-access-key", REGION, "bedrock", "aws4_request")
    assert_metered_once_through_the_sdk(records, fake_boto3)


def test_the_clients_bedrock_api_key_is_sent_as_bearer(stub, records, fake_boto3):
    def client():
        return anthropic.AnthropicBedrock(api_key="test-bearer", aws_region=REGION, max_retries=0,
                                          http_client=stub.http_client())

    _, baseline_request = take_baseline(stub, lambda: create_without_revenium(client(), **REQUEST))

    client().messages.create(**REQUEST)

    assert only_request(stub).headers["authorization"] == baseline_request.headers["authorization"] == (
        "Bearer test-bearer")
    assert_metered_once_through_the_sdk(records, fake_boto3)


def test_a_vpc_endpoint_base_url_is_used(stub, records, fake_boto3):
    vpc_host = f"vpce-1.bedrock-runtime.{REGION}.vpce.amazonaws.com"
    _, baseline_request = take_baseline(stub, lambda: create_without_revenium(
        anthropic_bedrock_client(stub, base_url=f"https://{vpc_host}"), **REQUEST))

    anthropic_bedrock_client(stub, base_url=f"https://{vpc_host}").messages.create(**REQUEST)

    assert only_request(stub).url.host == baseline_request.url.host == vpc_host
    assert_metered_once_through_the_sdk(records, fake_boto3)


@pytest.mark.parametrize("model, raw_path", [
    pytest.param("claude-opus-4-7", b"/model/anthropic.claude-opus-4-7/invoke", id="documented-short-name"),
    pytest.param(QUALIFIED_MODEL, f"/model/{QUALIFIED_MODEL}/invoke".encode(), id="qualified-id"),
    pytest.param("arn:aws:bedrock:us-east-1:123456789012:inference-profile/" + QUALIFIED_MODEL,
                 f"/model/arn:aws:bedrock:us-east-1:123456789012:inference-profile%2F{QUALIFIED_MODEL}/invoke".encode(),
                 id="inference-profile-arn"),
])
def test_documented_short_model_names_still_resolve(stub, records, fake_boto3, model, raw_path):
    anthropic_bedrock_client(stub).messages.create(**dict(REQUEST, model=model))

    assert only_request(stub).url.raw_path == raw_path
    assert_metered_once_through_the_sdk(records, fake_boto3)


def test_bedrock_error_reaches_the_caller_as_without_the_sdk(stub, records, fake_boto3):
    stub.status = 429
    with pytest.raises(anthropic.RateLimitError):
        create_without_revenium(anthropic_bedrock_client(stub), **REQUEST)
    only_request(stub)
    stub.requests.clear()

    with pytest.raises(anthropic.RateLimitError):
        anthropic_bedrock_client(stub).messages.create(**REQUEST)

    only_request(stub)
    assert fake_boto3.calls == []
    assert len(records) == 1, records
    assert (records[0]["stop_reason"], records[0]["provider"]) == ("ERROR", "AWS")
    assert "RateLimitError (HTTP 429)" in records[0]["error_reason"]


def test_a_plain_client_with_a_bedrock_url_keeps_the_fast_path(stub, records, fake_boto3):
    message = plain_client_with_a_bedrock_url(stub).messages.create(**REQUEST)

    assert stub.requests == []
    assert len(fake_boto3.calls) == 1
    assert message.content[0].text == "hello"
    assert len(records) == 1, records
    assert records[0]["provider"] == "AWS"


def test_a_plain_client_with_an_amazonaws_proxy_url_keeps_its_model_name(stub, records, fake_boto3):
    client = plain_client_with_a_bedrock_url(stub, base_url="https://abc.execute-api.us-east-1.amazonaws.com")

    message = client.messages.create(**dict(REQUEST, model="claude-sonnet-4-5"))

    assert stub.requests == []
    assert [call["modelId"] for call in fake_boto3.calls] == ["anthropic.claude-sonnet-4-5"]
    assert message.model == "claude-sonnet-4-5"
    assert len(records) == 1, records
    assert records[0]["model"] == "claude-sonnet-4-5"


# ---- BACK-4251: a cap on a Claude model judges the model the record carries ----

from entry_points.caps import breached_rule, record_cap_checks, scoped, seed_rules  # noqa: E402
from revenium_middleware._core.enforcement import _metered_model_forms  # noqa: E402
from revenium_middleware._core.exceptions import BudgetExceededError  # noqa: E402

from . import test_platform_beta_messages as platform_beta_messages  # noqa: E402  after event_stream_frame, which it imports back

CAP_RULE_ID = 4251
REGIONAL_MODEL = QUALIFIED_MODEL
BASE_MODEL = "anthropic.claude-haiku-4-5-20251001-v1:0"
MAPPED_SHORT_NAME = "global.claude-opus-4-7"
MAPPED_RESPONSE_MODEL = "claude-opus-4-7"
BEDROCK_REPORTED_MODEL = BASE_MODEL
MAPPED_BEDROCK_REPORTED_MODEL = "anthropic.claude-opus-4-7"


def model_cap(operator, value=None, values=None):
    entry = scoped("MODEL", operator, value)
    if values is not None:
        entry["values"] = values
    return breached_rule(CAP_RULE_ID, entry)


def unmatched_cap():
    return breached_rule(CAP_RULE_ID, scoped("AGENT", "IS", "nobody"))


def sync_create(client, **request):
    return client.messages.create(**request)


def sync_create_streamed(client, **request):
    return list(client.messages.create(stream=True, **request))


def sync_stream(client, **request):
    with client.messages.stream(**request) as stream:
        return stream.get_final_message()


def sync_parse(client, **request):
    return client.messages.parse(**request)


def sync_raw_create(client, **request):
    return client.messages.with_raw_response.create(**request).parse()


SYNC_SURFACES = {
    "create": sync_create,
    "create-stream-true": sync_create_streamed,
    "stream": sync_stream,
    "parse": sync_parse,
    "with-raw-response-create": sync_raw_create,
}


def async_anthropic_bedrock_client(stub):
    return anthropic.AsyncAnthropicBedrock(
        aws_access_key="test-access-key", aws_secret_key="test-secret-key", aws_region=REGION, max_retries=0,
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(stub.respond)))


async def async_create(client, **request):
    return await client.messages.create(**request)


async def async_stream(client, **request):
    async with client.messages.stream(**request) as stream:
        return await stream.get_final_message()


ASYNC_SURFACES = {
    "async-create": async_create,
    "async-stream": async_stream,
}


def call_sync(name, stub, **request):
    return SYNC_SURFACES[name](anthropic_bedrock_client(stub), **request)


def call_async(name, stub, **request):
    return asyncio.run(ASYNC_SURFACES[name](async_anthropic_bedrock_client(stub), **request))


def call_surface(name, stub, **request):
    if name in ASYNC_SURFACES:
        return call_async(name, stub, **request)
    return call_sync(name, stub, **request)


CLAUDE_MODEL_CAPS = [
    pytest.param(("IS", RESPONSE_MODEL, None), id="is"),
    pytest.param(("IN", None, [RESPONSE_MODEL, "claude-other"]), id="in"),
    pytest.param(("STARTS_WITH", "claude-haiku", None), id="starts-with"),
    pytest.param(("ENDS_WITH", "4-5-20251001", None), id="ends-with"),
]


@pytest.mark.parametrize("surface", list(SYNC_SURFACES))
@pytest.mark.parametrize("cap_filter", CLAUDE_MODEL_CAPS)
@pytest.mark.parametrize("model", [REGIONAL_MODEL, BASE_MODEL])
def test_a_breached_cap_on_the_claude_model_refuses_the_anthropic_bedrock_call(
        stub, records, fake_boto3, monkeypatch, surface, cap_filter, model):
    operator, value, values = cap_filter
    seed_rules(monkeypatch, model_cap(operator, value, values))

    with pytest.raises(BudgetExceededError):
        call_sync(surface, stub, **dict(REQUEST, model=model))

    assert stub.requests == []
    assert fake_boto3.calls == []
    assert records == []


# Only the sync create and stream wrappers swap a short name for its Bedrock id
# before sending; parse and the async surfaces send the name as given.
MAPPING_SURFACES = ["create", "create-stream-true", "stream", "with-raw-response-create"]


@pytest.mark.parametrize("surface", MAPPING_SURFACES)
def test_a_short_model_name_is_judged_as_the_bedrock_model_it_maps_to(stub, records, fake_boto3, monkeypatch, surface):
    stub.message = dict(stub.message, model=MAPPED_RESPONSE_MODEL)
    seed_rules(monkeypatch, model_cap("IS", MAPPED_RESPONSE_MODEL))

    with pytest.raises(BudgetExceededError):
        call_sync(surface, stub, **dict(REQUEST, model=MAPPED_SHORT_NAME))

    assert stub.requests == []
    assert records == []


@pytest.mark.parametrize("surface", list(ASYNC_SURFACES))
def test_an_async_anthropic_bedrock_call_under_a_breached_claude_model_cap_is_refused(
        stub, records, monkeypatch, surface):
    seed_rules(monkeypatch, model_cap("IS", RESPONSE_MODEL))

    with pytest.raises(BudgetExceededError):
        call_async(surface, stub, **REQUEST)

    assert stub.requests == []
    assert records == []


@pytest.mark.parametrize("surface", ["create", "stream", "async-create", "async-stream"])
@pytest.mark.parametrize("capped_model", [RESPONSE_MODEL, BEDROCK_REPORTED_MODEL])
def test_a_selectively_unmetered_bedrock_call_under_a_claude_model_cap_is_refused(
        stub, records, fake_boto3, monkeypatch, surface, capped_model):
    monkeypatch.setenv("REVENIUM_SELECTIVE_METERING", "true")
    seed_rules(monkeypatch, model_cap("IS", capped_model))

    with pytest.raises(BudgetExceededError):
        call_surface(surface, stub, **REQUEST)

    assert stub.requests == []
    assert records == []


@pytest.mark.parametrize("surface", [*SYNC_SURFACES, *ASYNC_SURFACES])
def test_a_cap_on_another_claude_model_lets_the_bedrock_call_through(stub, records, fake_boto3, monkeypatch, surface):
    seed_rules(monkeypatch, model_cap("IS", "claude-sonnet-4-5-20250929"))

    call_surface(surface, stub, **REQUEST)

    only_request(stub)
    assert len(records) == 1, records


@pytest.mark.parametrize("capped_model", [BEDROCK_REPORTED_MODEL, RESPONSE_MODEL])
@pytest.mark.parametrize("surface", [*SYNC_SURFACES, *ASYNC_SURFACES])
def test_a_breached_cap_on_either_the_recorded_bedrock_id_or_the_claude_name_refuses_the_call(
        stub, records, fake_boto3, monkeypatch, surface, capped_model):
    """The record carries the regional id the call was sent with; a cap may
    also name the bare id Bedrock answers with or the Claude model."""
    stub.message = dict(stub.message, model=BEDROCK_REPORTED_MODEL)
    seed_rules(monkeypatch, model_cap("IS", capped_model))

    with pytest.raises(BudgetExceededError):
        call_surface(surface, stub, **REQUEST)

    assert stub.requests == []
    assert fake_boto3.calls == []
    assert records == []


BEDROCK_ID_CAPS = [
    pytest.param(("IN", None, [BEDROCK_REPORTED_MODEL, "x"]), id="in"),
    pytest.param(("STARTS_WITH", "anthropic.claude-haiku", None), id="starts-with"),
    pytest.param(("ENDS_WITH", "20251001-v1:0", None), id="ends-with"),
]


@pytest.mark.parametrize("cap_filter", BEDROCK_ID_CAPS)
@pytest.mark.parametrize("surface", [*SYNC_SURFACES, *ASYNC_SURFACES])
def test_a_breached_cap_matching_the_recorded_bedrock_id_refuses_the_call(
        stub, records, fake_boto3, monkeypatch, surface, cap_filter):
    operator, value, values = cap_filter
    stub.message = dict(stub.message, model=BEDROCK_REPORTED_MODEL)
    seed_rules(monkeypatch, model_cap(operator, value, values))

    with pytest.raises(BudgetExceededError):
        call_surface(surface, stub, **REQUEST)

    assert stub.requests == []
    assert fake_boto3.calls == []
    assert records == []


@pytest.mark.parametrize("surface", [*SYNC_SURFACES, *ASYNC_SURFACES])
def test_a_breached_cap_on_every_model_except_the_claude_name_refuses_the_bedrock_call(
        stub, records, fake_boto3, monkeypatch, surface):
    """The platform compares the recorded Bedrock id, which is not the Claude
    name, so it counts this call toward the cap (BACK-4281)."""
    stub.message = dict(stub.message, model=BEDROCK_REPORTED_MODEL)
    seed_rules(monkeypatch, model_cap("IS_NOT", RESPONSE_MODEL))

    with pytest.raises(BudgetExceededError):
        call_surface(surface, stub, **REQUEST)

    assert stub.requests == []
    assert fake_boto3.calls == []
    assert records == []


@pytest.mark.parametrize("surface", ["create", "stream", "async-create", "async-stream"])
def test_a_selectively_unmetered_bedrock_call_under_a_cap_on_every_model_except_the_claude_name_is_refused(
        stub, records, fake_boto3, monkeypatch, surface):
    monkeypatch.setenv("REVENIUM_SELECTIVE_METERING", "true")
    seed_rules(monkeypatch, model_cap("IS_NOT", RESPONSE_MODEL))

    with pytest.raises(BudgetExceededError):
        call_surface(surface, stub, **REQUEST)

    assert stub.requests == []
    assert records == []


@pytest.mark.parametrize("surface", [*SYNC_SURFACES, *ASYNC_SURFACES])
def test_a_breached_cap_on_every_model_except_the_recorded_regional_id_lets_the_bedrock_call_through(
        stub, records, fake_boto3, monkeypatch, surface):
    stub.message = dict(stub.message, model=BEDROCK_REPORTED_MODEL)
    seed_rules(monkeypatch, model_cap("IS_NOT", REGIONAL_MODEL))

    call_surface(surface, stub, **REQUEST)

    only_request(stub)
    assert [record["model"] for record in records] == [REGIONAL_MODEL]


@pytest.mark.parametrize("surface", [*SYNC_SURFACES, *ASYNC_SURFACES])
def test_a_breached_cap_on_every_model_except_the_bare_bedrock_id_refuses_a_regional_call(
        stub, records, fake_boto3, monkeypatch, surface):
    """The platform compares the recorded regional id, which is not the bare
    id, so it counts this call toward the cap, as it does for boto3 (BACK-4308)."""
    stub.message = dict(stub.message, model=BEDROCK_REPORTED_MODEL)
    seed_rules(monkeypatch, model_cap("IS_NOT", BEDROCK_REPORTED_MODEL))

    with pytest.raises(BudgetExceededError):
        call_surface(surface, stub, **REQUEST)

    assert stub.requests == []
    assert records == []


@pytest.mark.parametrize("surface", [*SYNC_SURFACES, *ASYNC_SURFACES])
def test_a_cap_on_another_model_lets_the_bedrock_call_through_and_records_the_regional_id_it_was_sent_with(
        stub, records, fake_boto3, monkeypatch, surface):
    stub.message = dict(stub.message, model=BEDROCK_REPORTED_MODEL)
    seed_rules(monkeypatch, model_cap("IS", "claude-opus-4-7"))

    call_surface(surface, stub, **REQUEST)

    only_request(stub)
    assert [record["model"] for record in records] == [REGIONAL_MODEL]


@pytest.mark.parametrize("surface, model, recorded_model, reported_model, claude_name", [
    *[pytest.param(surface, REGIONAL_MODEL, REGIONAL_MODEL, BEDROCK_REPORTED_MODEL, RESPONSE_MODEL,
                   id=f"qualified-id-{surface}")
      for surface in [*SYNC_SURFACES, *ASYNC_SURFACES]],
    *[pytest.param(surface, MAPPED_SHORT_NAME, "global." + MAPPED_BEDROCK_REPORTED_MODEL,
                   MAPPED_BEDROCK_REPORTED_MODEL, MAPPED_RESPONSE_MODEL, id=f"mapped-short-name-{surface}")
      for surface in MAPPING_SURFACES],
])
def test_the_bedrock_cap_check_judges_the_recorded_id_the_bare_bedrock_id_and_the_claude_name(
        stub, records, fake_boto3, monkeypatch, surface, model, recorded_model, reported_model, claude_name):
    """The check judges the id the record carries, the bare id Bedrock answers
    with and the Claude name for every Claude model id the SDK can name.
    Provisioned-throughput ARNs are left out on purpose: they do not name
    their model, so the SDK cannot judge them by the record's model before the
    call (BACK-4253). Application inference profile ARNs are judged by the
    model they route to (BACK-4305)."""
    stub.message = dict(stub.message, model=reported_model)
    seed_rules(monkeypatch, unmatched_cap())
    checks = record_cap_checks(monkeypatch)

    call_surface(surface, stub, **dict(REQUEST, model=model))

    assert [record["model"] for record in records] == [recorded_model]
    assert [str(check.model) for check in checks] == [recorded_model]
    assert [_metered_model_forms(check.model) for check in checks] == [{recorded_model, reported_model, claude_name}]


def test_the_plain_client_fast_path_is_judged_by_the_model_it_records(stub, records, fake_boto3, monkeypatch):
    seed_rules(monkeypatch, unmatched_cap())
    checks = record_cap_checks(monkeypatch)

    plain_client_with_a_bedrock_url(stub).messages.create(**REQUEST)

    assert len(fake_boto3.calls) == 1
    assert len(records) == 1, records
    assert [check.model for check in checks] == [records[0]["model"]] == [QUALIFIED_MODEL]


def test_a_direct_anthropic_call_is_judged_by_its_own_model_name(records, monkeypatch):
    seed_rules(monkeypatch, unmatched_cap())
    checks = record_cap_checks(monkeypatch)
    client = anthropic.Anthropic(
        api_key="test-key", base_url=f"http://{STUB_ANTHROPIC_HOST}", max_retries=0,
        http_client=httpx2.Client(transport=httpx2.MockTransport(
            lambda request: httpx2.Response(200, json=bedrock_message()))))

    client.messages.create(**dict(REQUEST, model=RESPONSE_MODEL))

    assert [check.model for check in checks] == [RESPONSE_MODEL]
    assert len(records) == 1, records
    assert records[0]["provider"] == "ANTHROPIC"


GEOGRAPHY_PREFIXES = ["us.", "eu.", "apac.", "global.", ""]


def _raw_boto3_cases():
    for operation in ["InvokeModel", "Converse"]:
        for prefix in GEOGRAPHY_PREFIXES:
            for capped, cap_name in [(BASE_MODEL, "bedrock-id"), (RESPONSE_MODEL, "claude-name")]:
                spelled = prefix or "bare."
                control = "-control" if not prefix and capped == BASE_MODEL else ""
                yield pytest.param(operation, prefix + BASE_MODEL, capped,
                                   id=f"{operation}-{spelled}-capped-on-{cap_name}{control}")


def boto3_call():
    """Stands in for the botocore call the transport wraps."""
    return MagicMock(return_value={"ResponseMetadata": {"RequestId": "req-1"}})


def call_raw_boto3(monkeypatch, wrapped, model_id, operation="Converse"):
    """One raw boto3 Bedrock call through the transport, ``wrapped`` being the call boto3 would make."""
    from types import SimpleNamespace

    from revenium_middleware.anthropic import bedrock_transport

    monkeypatch.setenv("REVENIUM_BEDROCK_TRANSPORT", "1")
    instance = SimpleNamespace(meta=SimpleNamespace(service_model=SimpleNamespace(service_name="bedrock-runtime"),
                                                    region_name=REGION))
    with patch.object(bedrock_transport, "_observe_response"):
        bedrock_transport._make_api_call_wrapper(wrapped, instance, (operation, {"modelId": model_id}), {})


@pytest.mark.parametrize("operation, requested, capped", list(_raw_boto3_cases()))
def test_a_breached_cap_on_the_bedrock_id_or_the_claude_name_refuses_a_raw_boto3_call(
        monkeypatch, operation, requested, capped):
    """BACK-4304: a raw boto3 call is refused by a cap on the bare Bedrock id or
    the Claude name, whichever geography id it names. Its record keeps the
    requested id, which carries the regional price."""
    seed_rules(monkeypatch, model_cap("IS", capped))
    wrapped = boto3_call()

    with pytest.raises(BudgetExceededError):
        call_raw_boto3(monkeypatch, wrapped, requested, operation)

    wrapped.assert_not_called()


def test_a_raw_boto3_call_under_a_cap_on_every_model_except_the_claude_name_is_still_refused(monkeypatch):
    """The platform compares the recorded ``us.`` id, which is not the Claude name."""
    seed_rules(monkeypatch, model_cap("IS_NOT", RESPONSE_MODEL))
    wrapped = boto3_call()

    with pytest.raises(BudgetExceededError):
        call_raw_boto3(monkeypatch, wrapped, REGIONAL_MODEL)

    wrapped.assert_not_called()


def test_a_raw_boto3_call_under_a_cap_on_every_model_except_the_bare_id_is_still_refused(monkeypatch):
    """The platform compares the recorded ``us.`` id, which is not the bare id."""
    seed_rules(monkeypatch, model_cap("IS_NOT", BASE_MODEL))
    wrapped = boto3_call()

    with pytest.raises(BudgetExceededError):
        call_raw_boto3(monkeypatch, wrapped, REGIONAL_MODEL)

    wrapped.assert_not_called()


def test_a_raw_boto3_call_under_a_cap_on_every_model_except_its_own_id_goes_through(monkeypatch):
    seed_rules(monkeypatch, model_cap("IS_NOT", REGIONAL_MODEL))
    wrapped = boto3_call()

    call_raw_boto3(monkeypatch, wrapped, REGIONAL_MODEL)

    wrapped.assert_called_once()


@pytest.mark.parametrize("model, expected", [
    ("us.anthropic.claude-haiku-4-5-20251001-v1:0", "claude-haiku-4-5-20251001"),
    ("anthropic.claude-haiku-4-5-20251001-v1:0", "claude-haiku-4-5-20251001"),
    ("anthropic.claude-3-5-sonnet-20241022-v2:0", "claude-3-5-sonnet-20241022"),
    ("global.anthropic.claude-opus-4-7", "claude-opus-4-7"),
    ("arn:aws:bedrock:us-east-1::foundation-model/anthropic.claude-3-sonnet-20240229-v1:0", "claude-3-sonnet-20240229"),
    ("claude-haiku-4-5", "claude-haiku-4-5"),
    ("arn:aws:bedrock:us-east-1:123456789012:application-inference-profile/abc",
     "arn:aws:bedrock:us-east-1:123456789012:application-inference-profile/abc"),
    ("anthropic.claude-v2:1", "anthropic.claude-v2:1"),
    (None, None),
])
def test_a_bedrock_model_id_names_the_claude_model_it_serves(model, expected):
    from revenium_middleware.anthropic.bedrock_adapter import anthropic_model_name

    assert anthropic_model_name(model) == expected


# ---- BACK-4305: an AnthropicBedrock call by application inference profile is capped by the model it routes to ----

from .test_bedrock_transport import PROFILE_ARN, FakeControlPlane, install_resolver, profile_warnings  # noqa: E402

PROFILE_REQUEST = dict(REQUEST, model=PROFILE_ARN)
PROFILE_REGION = "us-east-1"


@pytest.fixture
def control_plane(monkeypatch):
    """Routes ``PROFILE_ARN`` to ``BASE_MODEL`` through a fresh resolver the transport shares."""
    fake = FakeControlPlane()
    install_resolver(monkeypatch, lambda runtime, region: fake)
    return fake


@pytest.mark.parametrize("selective", [False, True], ids=["metered", "selectively-unmetered"])
@pytest.mark.parametrize("capped", [BASE_MODEL, RESPONSE_MODEL], ids=["bedrock-id", "claude-name"])
@pytest.mark.parametrize("surface", list(SYNC_SURFACES))
def test_a_breached_cap_on_the_claude_model_refuses_an_anthropic_bedrock_call_by_application_profile(
        stub, records, fake_boto3, control_plane, monkeypatch, surface, capped, selective):
    if selective:
        monkeypatch.setenv("REVENIUM_SELECTIVE_METERING", "true")
    seed_rules(monkeypatch, model_cap("IS", capped))

    with pytest.raises(BudgetExceededError):
        call_sync(surface, stub, **PROFILE_REQUEST)

    assert stub.requests == []
    assert fake_boto3.calls == []
    assert records == []
    assert control_plane.lookups == [PROFILE_ARN]


def test_an_anthropic_bedrock_profile_whose_lookup_fails_is_answered_as_before_with_one_warning(
        stub, records, control_plane, monkeypatch, caplog):
    caplog.set_level(logging.WARNING, logger="revenium_middleware")
    control_plane.errors[PROFILE_ARN] = ["AccessDeniedException"]
    seed_rules(monkeypatch, model_cap("IS", RESPONSE_MODEL))

    call_sync("create", stub, **PROFILE_REQUEST)
    call_sync("create", stub, **PROFILE_REQUEST)

    assert len(stub.requests) == 2
    assert len(records) == 2, records
    assert control_plane.lookups == [PROFILE_ARN]
    warnings = profile_warnings(caplog)
    assert len(warnings) == 1
    message = warnings[0].getMessage()
    assert PROFILE_ARN in message
    assert "bedrock:GetInferenceProfile" in message
    assert "cannot be capped by model" in message
    assert "neither metered" not in message


def test_an_anthropic_bedrock_profile_is_looked_up_once_shared_with_the_transport(
        stub, records, control_plane, monkeypatch):
    seed_rules(monkeypatch, unmatched_cap())
    wrapped = boto3_call()

    call_sync("create", stub, **PROFILE_REQUEST)
    call_raw_boto3(monkeypatch, wrapped, PROFILE_ARN)

    only_request(stub)
    wrapped.assert_called_once()
    assert control_plane.lookups == [PROFILE_ARN]


def capture_control_plane_clients(monkeypatch):
    """Keeps the real lookup client factory, capturing what it asks botocore for."""
    import botocore.session

    from revenium_middleware.anthropic.inference_profiles import _control_plane_client

    fake = FakeControlPlane()
    created = []

    def create_client(self, service_name, **kwargs):
        created.append((service_name, kwargs))
        return fake

    install_resolver(monkeypatch, _control_plane_client)
    monkeypatch.setattr(botocore.session.Session, "create_client", create_client)
    return created


def test_the_anthropic_bedrock_profile_lookup_signs_with_the_clients_own_keys(stub, records, monkeypatch):
    created = capture_control_plane_clients(monkeypatch)
    seed_rules(monkeypatch, model_cap("IS", RESPONSE_MODEL))

    with pytest.raises(BudgetExceededError):
        call_sync("create", stub, **PROFILE_REQUEST)

    assert [(service, kwargs["aws_access_key_id"], kwargs["region_name"]) for service, kwargs in created] == [
        ("bedrock", "test-access-key", PROFILE_REGION)]
    assert stub.requests == []


@pytest.mark.parametrize("bedrock_api_key", ["argument", "environment"])
def test_an_anthropic_bedrock_client_on_a_bedrock_api_key_makes_no_profile_lookup(
        stub, records, monkeypatch, caplog, bedrock_api_key):
    caplog.set_level(logging.WARNING, logger="revenium_middleware")
    created = capture_control_plane_clients(monkeypatch)
    seed_rules(monkeypatch, model_cap("IS", RESPONSE_MODEL))
    credentials = dict(aws_access_key=None, aws_secret_key=None)
    if bedrock_api_key == "argument":
        credentials["api_key"] = "test-bedrock-api-key"
    else:
        monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "test-bedrock-api-key")

    anthropic_bedrock_client(stub, **credentials).messages.create(**PROFILE_REQUEST)

    assert created == []
    only_request(stub)
    assert len(records) == 1, records
    assert len(profile_warnings(caplog)) == 1


@pytest.mark.parametrize("surface", list(ASYNC_SURFACES))
def test_an_async_anthropic_bedrock_call_by_a_profile_already_looked_up_is_refused(
        stub, records, control_plane, monkeypatch, surface):
    seed_rules(monkeypatch, unmatched_cap())
    call_sync("create", stub, **PROFILE_REQUEST)
    stub.requests.clear()
    records.clear()
    seed_rules(monkeypatch, model_cap("IS", RESPONSE_MODEL))

    with pytest.raises(BudgetExceededError):
        call_async(surface, stub, **PROFILE_REQUEST)

    assert stub.requests == []
    assert records == []
    assert control_plane.lookups == [PROFILE_ARN]


@pytest.mark.parametrize("surface", list(ASYNC_SURFACES))
def test_an_async_anthropic_bedrock_call_by_a_profile_not_yet_looked_up_is_answered_without_a_lookup(
        stub, records, control_plane, monkeypatch, surface):
    seed_rules(monkeypatch, model_cap("IS", RESPONSE_MODEL))

    call_async(surface, stub, **PROFILE_REQUEST)

    only_request(stub)
    assert len(records) == 1, records
    assert control_plane.lookups == []


def lookups_denied_to_anthropic_bedrock(monkeypatch, *, raw_boto3_denied=False):
    """A resolver whose lookups are denied for every ``AnthropicBedrock`` client, and for raw boto3 if asked."""
    from revenium_middleware.anthropic.inference_profiles import AnthropicBedrockCredentials

    denied, allowed = FakeControlPlane(), FakeControlPlane()
    denied.errors[PROFILE_ARN] = ["AccessDeniedException"] * 5
    raw_boto3 = denied if raw_boto3_denied else allowed
    install_resolver(monkeypatch, lambda runtime, region: (
        denied if isinstance(runtime, AnthropicBedrockCredentials) else raw_boto3))
    return denied, allowed


@pytest.mark.parametrize("credentials", ["bedrock-api-key", "denied-aws-keys"])
def test_a_denied_anthropic_bedrock_profile_lookup_does_not_stop_an_authorized_raw_boto3_call_being_capped(
        stub, records, monkeypatch, credentials):
    denied, allowed = lookups_denied_to_anthropic_bedrock(monkeypatch)
    seed_rules(monkeypatch, model_cap("IS", RESPONSE_MODEL))
    client_options = (dict(api_key="test-bedrock-api-key", aws_access_key=None, aws_secret_key=None)
                      if credentials == "bedrock-api-key" else {})
    wrapped = boto3_call()

    anthropic_bedrock_client(stub, **client_options).messages.create(**PROFILE_REQUEST)
    with pytest.raises(BudgetExceededError):
        call_raw_boto3(monkeypatch, wrapped, PROFILE_ARN)

    only_request(stub)
    wrapped.assert_not_called()
    assert allowed.lookups == [PROFILE_ARN]


def test_denied_profile_lookups_by_anthropic_bedrock_and_raw_boto3_each_warn_what_they_cost(
        stub, records, monkeypatch, caplog):
    caplog.set_level(logging.WARNING, logger="revenium_middleware")
    denied, _ = lookups_denied_to_anthropic_bedrock(monkeypatch, raw_boto3_denied=True)
    seed_rules(monkeypatch, model_cap("IS", RESPONSE_MODEL))
    wrapped = boto3_call()

    for _ in range(2):
        call_sync("create", stub, **PROFILE_REQUEST)
        call_raw_boto3(monkeypatch, wrapped, PROFILE_ARN)

    assert len(stub.requests) == 2
    assert wrapped.call_count == 2
    assert denied.lookups == [PROFILE_ARN, PROFILE_ARN]
    messages = [w.getMessage() for w in profile_warnings(caplog)]
    assert len(messages) == 2, messages
    assert "cannot be capped by model" in messages[0]
    assert "neither metered nor capped" in messages[1]


def test_a_profile_resolved_by_raw_boto3_caps_a_later_anthropic_bedrock_call_on_a_bedrock_api_key(
        stub, records, monkeypatch):
    _, allowed = lookups_denied_to_anthropic_bedrock(monkeypatch)
    seed_rules(monkeypatch, unmatched_cap())
    call_raw_boto3(monkeypatch, boto3_call(), PROFILE_ARN)
    seed_rules(monkeypatch, model_cap("IS", RESPONSE_MODEL))
    client = anthropic_bedrock_client(stub, api_key="test-bedrock-api-key", aws_access_key=None, aws_secret_key=None)

    with pytest.raises(BudgetExceededError):
        client.messages.create(**PROFILE_REQUEST)

    assert stub.requests == []
    assert allowed.lookups == [PROFILE_ARN]


# ---- BACK-4308: a Bedrock call is recorded under the profile it was sent with ----

from urllib.parse import quote  # noqa: E402

SYSTEM_PROFILE_ARN = "arn:aws:bedrock:us-east-1:123456789012:inference-profile/" + REGIONAL_MODEL
FOUNDATION_MODEL_ARN = "arn:aws:bedrock:us-east-1::foundation-model/" + BASE_MODEL
SHORT_REQUESTED_MODEL = "claude-haiku-4-5"
COMPACTION_PASS = {"type": "compaction", "input_tokens": 30, "output_tokens": 5}

REQUESTED_AND_RECORDED_IDS = [
    pytest.param(REGIONAL_MODEL, REGIONAL_MODEL, id="us"),
    pytest.param("eu." + BASE_MODEL, "eu." + BASE_MODEL, id="eu"),
    pytest.param("apac." + BASE_MODEL, "apac." + BASE_MODEL, id="apac"),
    pytest.param("global." + BASE_MODEL, "global." + BASE_MODEL, id="global"),
    pytest.param(BASE_MODEL, BASE_MODEL, id="bare"),
    pytest.param(SYSTEM_PROFILE_ARN, REGIONAL_MODEL, id="system-profile-arn"),
    pytest.param(FOUNDATION_MODEL_ARN, BASE_MODEL, id="foundation-model-arn"),
]


def sent_model_path(request):
    return request.url.raw_path.decode().split("/")[2]


@pytest.mark.parametrize("requested, recorded", REQUESTED_AND_RECORDED_IDS)
@pytest.mark.parametrize("surface", [*SYNC_SURFACES, *ASYNC_SURFACES])
def test_a_regional_profile_call_is_recorded_under_the_profile_it_was_sent_with(
        stub, records, fake_boto3, surface, requested, recorded):
    """Bedrock answers with the bare model id, which drops the regional profile
    AWS prices the call by. A profile ARN is recorded as the id after its last
    ``/``, the name the platform prices."""
    stub.message = dict(stub.message, model=BEDROCK_REPORTED_MODEL)

    call_surface(surface, stub, **dict(REQUEST, model=requested))

    assert sent_model_path(only_request(stub)) == quote(requested, safe=":")
    assert [record["model"] for record in records] == [recorded]


@pytest.mark.parametrize("surface", ["create", "async-create"])
def test_a_regional_profile_calls_compaction_pass_is_recorded_under_the_same_profile(
        stub, records, fake_boto3, surface):
    usage = dict(stub.message["usage"], iterations=[COMPACTION_PASS])
    stub.message = dict(stub.message, model=BEDROCK_REPORTED_MODEL, usage=usage)

    call_surface(surface, stub, **REQUEST)

    assert [record["model"] for record in records] == [REGIONAL_MODEL, REGIONAL_MODEL]


@pytest.mark.parametrize("surface", MAPPING_SURFACES)
def test_an_alias_is_recorded_as_the_bedrock_profile_it_maps_to(stub, records, fake_boto3, surface):
    stub.message = dict(stub.message, model="anthropic.claude-opus-4-7")

    call_sync(surface, stub, **dict(REQUEST, model="us.claude-opus-4-7"))

    assert sent_model_path(only_request(stub)) == "us.anthropic.claude-opus-4-7"
    assert [record["model"] for record in records] == ["us.anthropic.claude-opus-4-7"]


def direct_anthropic_reply(request):
    return httpx2.Response(200, json=bedrock_message())


def sync_direct_create(**request):
    client = anthropic.Anthropic(
        api_key="test-key", base_url=f"http://{STUB_ANTHROPIC_HOST}", max_retries=0,
        http_client=httpx2.Client(transport=httpx2.MockTransport(direct_anthropic_reply)))
    return client.messages.create(**request)


async def async_direct_create(**request):
    client = anthropic.AsyncAnthropic(
        api_key="test-key", base_url=f"http://{STUB_ANTHROPIC_HOST}", max_retries=0,
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(direct_anthropic_reply)))
    return await client.messages.create(**request)


@pytest.mark.parametrize("call", [
    pytest.param(sync_direct_create, id="sync"),
    pytest.param(lambda **request: asyncio.run(async_direct_create(**request)), id="async"),
])
def test_a_direct_anthropic_call_is_recorded_under_the_model_the_response_names(records, call):
    call(**dict(REQUEST, model=SHORT_REQUESTED_MODEL))

    assert [(record["provider"], record["model"]) for record in records] == [("ANTHROPIC", RESPONSE_MODEL)]


def test_a_vertex_call_is_recorded_under_the_model_the_response_names(records):
    platform_beta_messages.PlatformStub("vertex").client(asynchronous=False).messages.create(
        **dict(REQUEST, model="claude-haiku-4-5@20251001"))

    assert [record["model"] for record in records] == [RESPONSE_MODEL]


def test_with_bedrock_support_switched_off_the_cap_judges_the_name_the_record_carries(
        stub, records, fake_boto3, monkeypatch):
    monkeypatch.setenv("REVENIUM_BEDROCK_DISABLE", "1")
    stub.message = dict(stub.message, model=BEDROCK_REPORTED_MODEL)
    seed_rules(monkeypatch, unmatched_cap())
    checks = record_cap_checks(monkeypatch)

    anthropic_bedrock_client(stub).messages.create(**REQUEST)

    only_request(stub)
    assert [str(check.model) for check in checks] == [record["model"] for record in records] == [
        BEDROCK_REPORTED_MODEL]


TICKET_FORMS = [
    pytest.param(sync_create, False, id="create"),
    pytest.param(sync_stream, True, id="stream"),
]


@pytest.mark.parametrize("make_client", CLIENT_KINDS)
@pytest.mark.parametrize("call, streamed", TICKET_FORMS)
def test_a_bedrock_call_carries_its_ticket_id_streamed_or_not(
        stub, records, monkeypatch, make_client, call, streamed):
    monkeypatch.delenv("REVENIUM_TICKET_ID", raising=False)

    call(make_client(stub), usage_metadata={"ticket_id": "JIRA-123"}, **REQUEST)

    assert len(records) == 1, records
    assert records[0].get("ticket_id") == "JIRA-123"
    assert records[0]["is_streamed"] is streamed


@pytest.mark.parametrize("make_client", CLIENT_KINDS)
@pytest.mark.parametrize("call, streamed", TICKET_FORMS)
def test_a_numeric_ticket_id_drops_the_ticket_not_the_record(
        stub, records, monkeypatch, make_client, call, streamed):
    monkeypatch.delenv("REVENIUM_TICKET_ID", raising=False)

    call(make_client(stub), usage_metadata={"ticket_id": 123}, **REQUEST)

    assert len(records) == 1, records
    assert records[0].get("ticket_id") is None
    assert records[0]["is_streamed"] is streamed


# ---- BACK-4447: prompt capture on the plain client's Bedrock fast path ----

from revenium_middleware.anthropic.config import Config  # noqa: E402

PROMPT_FIELDS = ("system_prompt", "input_messages", "output_response", "prompts_truncated")
CAPTURE_REQUEST = dict(REQUEST, system="Be brief.")


def capture_create(client, **request):
    return client.messages.create(**request)


def capture_stream(client, **request):
    with client.messages.stream(**request) as stream:
        return "".join(stream.text_stream)


CAPTURE_CALLS = [
    pytest.param(capture_create, False, id="create"),
    pytest.param(capture_stream, True, id="stream"),
]


def prompt_fields(record):
    return {name: record[name] for name in PROMPT_FIELDS if name in record}


@pytest.mark.parametrize("call, streamed", CAPTURE_CALLS)
def test_a_plain_client_with_a_bedrock_url_records_the_prompt_and_reply_like_anthropic_bedrock(
        stub, records, fake_boto3, monkeypatch, call, streamed):
    monkeypatch.setenv("REVENIUM_CAPTURE_PROMPTS", "true")
    call(anthropic_bedrock_client(stub), **CAPTURE_REQUEST)
    expected = prompt_fields(records.pop())

    call(plain_client_with_a_bedrock_url(stub), **CAPTURE_REQUEST)

    assert len(fake_boto3.calls) == 1
    assert len(records) == 1, records
    assert records[0]["is_streamed"] is streamed
    assert expected == {
        "system_prompt": "Be brief.",
        "input_messages": json.dumps([{"role": "user", "content": "hi"}]),
        "output_response": "hello",
        "prompts_truncated": False,
    }
    assert prompt_fields(records[0]) == expected


@pytest.mark.parametrize("call, streamed", CAPTURE_CALLS)
def test_a_plain_client_with_a_bedrock_url_records_no_prompt_with_capture_off(
        stub, records, fake_boto3, monkeypatch, call, streamed):
    monkeypatch.delenv("REVENIUM_CAPTURE_PROMPTS", raising=False)

    call(plain_client_with_a_bedrock_url(stub), **CAPTURE_REQUEST)

    assert len(fake_boto3.calls) == 1
    assert len(records) == 1, records
    assert prompt_fields(records[0]) == {}


def test_a_long_system_prompt_on_the_bedrock_fast_path_is_cut_and_flagged(stub, records, fake_boto3, monkeypatch):
    monkeypatch.setenv("REVENIUM_CAPTURE_PROMPTS", "true")

    plain_client_with_a_bedrock_url(stub).messages.create(
        **dict(REQUEST, system="A" * (Config.MAX_PROMPT_LENGTH + 100)))

    assert len(records) == 1, records
    system_prompt = records[0].get("system_prompt")
    assert system_prompt is not None
    assert len(system_prompt) == Config.MAX_PROMPT_LENGTH
    assert system_prompt.endswith("...[TRUNCATED]")
    assert records[0]["prompts_truncated"] is True


def test_a_selectively_unmetered_plain_client_call_captures_nothing(stub, records, fake_boto3, monkeypatch):
    monkeypatch.setenv("REVENIUM_CAPTURE_PROMPTS", "true")
    monkeypatch.setenv("REVENIUM_SELECTIVE_METERING", "true")

    with patch("revenium_middleware.anthropic.middleware.captured_prompt_fields") as in_middleware, \
            patch("revenium_middleware.anthropic.prompt_extractor.captured_prompt_fields") as in_adapter:
        plain_client_with_a_bedrock_url(stub).messages.create(**CAPTURE_REQUEST)

    in_middleware.assert_not_called()
    in_adapter.assert_not_called()
    assert records == []


@pytest.mark.parametrize("call, streamed", CAPTURE_CALLS)
def test_a_failing_prompt_extraction_on_the_bedrock_fast_path_still_sends_the_usage_record(
        stub, records, fake_boto3, monkeypatch, call, streamed):
    monkeypatch.setenv("REVENIUM_CAPTURE_PROMPTS", "true")

    with patch("revenium_middleware.anthropic.prompt_extractor.extract_prompt_data_if_enabled",
               side_effect=TypeError("object of type 'int' has no len()")):
        call(plain_client_with_a_bedrock_url(stub), **CAPTURE_REQUEST)

    assert len(records) == 1, records
    assert records[0]["is_streamed"] is streamed
    assert prompt_fields(records[0]) == {}
