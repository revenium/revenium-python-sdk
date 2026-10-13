"""Claude through LangChain's Bedrock Converse and Mantle classes stores only uncached input (BACK-4210).

``ChatBedrockConverse`` (langchain-aws 1.4.6 and later) and ``ChatAnthropicMantle``
put the cache read and write inside ``usage_metadata.input_tokens``, the way
``ChatAnthropic`` does. The callback record counted them again at the cache
rates, so a cached Claude call was billed twice.

Real langchain-aws classes run over stub clients, so no call leaves the process.
Nova on Converse pools the same way but is outside this fix, and ``ChatBedrock``
(invoke) shares the ``amazon_bedrock`` provider key with Converse but already
reports a net count: both keep the input count they had.
"""
import asyncio
import json
import logging
import threading
import uuid
from unittest.mock import MagicMock, patch

import pytest

langchain_aws = pytest.importorskip("langchain_aws")
anthropic = pytest.importorskip("anthropic")
httpx2 = pytest.importorskip("httpx2")

from langchain_aws import ChatBedrock, ChatBedrockConverse  # noqa: E402
from langchain_aws.chat_models.anthropic import ChatAnthropicMantle  # noqa: E402

import revenium_middleware.openai  # noqa: E402,F401  installs the OpenAI patches the callback meters through
from revenium_middleware._core.call_ownership import reset_claimed_response_ids  # noqa: E402
from revenium_middleware.openai.langchain import ReveniumCallbackHandler  # noqa: E402
from revenium_middleware.openai.langchain import _model_call  # noqa: E402

from .test_langchain_exactly_once import unwrap_anthropic_middleware  # noqa: E402

CONVERSE_CLAUDE = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
CONVERSE_NOVA = "us.amazon.nova-micro-v1:0"
INVOKE_CLAUDE = "anthropic.claude-3-haiku-20240307-v1:0"
MANTLE_CLAUDE = "anthropic.claude-haiku-4-5"
MANTLE_HOST = "bedrock-mantle.us-east-1.api.aws"
CLAUDE_PROFILE_ARN = "arn:aws:bedrock:us-east-1:123456789012:application-inference-profile/cnbclaude"
RAW_INPUT = 12

STYLES = ("invoke", "stream")
CACHE_CASES = {
    "read": {"cache_read": 4000},
    "write": {"cache_write": 4000},
}


def _run_now(coroutine):
    thread = threading.Thread(target=lambda: asyncio.run(coroutine))
    thread.start()
    thread.join(timeout=10)
    return thread


@pytest.fixture(autouse=True)
def _isolated(monkeypatch):
    monkeypatch.setenv("REVENIUM_METERING_API_KEY", "hak_test_langchain")
    reset_claimed_response_ids()
    yield
    reset_claimed_response_ids()


@pytest.fixture
def records():
    captured = []

    def record(operation, payload, *args, **kwargs):
        captured.append(payload)
        return MagicMock(id="recorded", status_code=201)

    with patch("revenium_middleware.openai.middleware.submit_ai_event", side_effect=record), \
            patch("revenium_middleware.openai.middleware.run_async_in_thread", side_effect=_run_now):
        yield captured


def only_record(records):
    assert len(records) == 1, records
    return records[0]


def converse_usage(cache_read=0, cache_write=0, cache_details=None):
    usage = {
        "inputTokens": RAW_INPUT,
        "outputTokens": 5,
        "totalTokens": RAW_INPUT + cache_read + cache_write + 5,
        "cacheReadInputTokens": cache_read,
        "cacheWriteInputTokens": cache_write,
    }
    if cache_details:
        usage["cacheDetails"] = cache_details
    return usage


class StubConverseClient:
    """The boto3 ``bedrock-runtime`` client, answering Converse and ConverseStream."""

    def __init__(self, usage):
        self.usage = usage
        self.meta = MagicMock()

    def converse(self, **kwargs):
        return {
            "output": {"message": {"role": "assistant", "content": [{"text": "ok"}]}},
            "stopReason": "end_turn",
            "usage": dict(self.usage),
            "metrics": {"latencyMs": 10},
            "ResponseMetadata": {"RequestId": str(uuid.uuid4()), "HTTPHeaders": {}},
        }

    def converse_stream(self, **kwargs):
        return {
            "stream": [
                {"messageStart": {"role": "assistant"}},
                {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"text": "ok"}}},
                {"contentBlockStop": {"contentBlockIndex": 0}},
                {"messageStop": {"stopReason": "end_turn"}},
                {"metadata": {"usage": dict(self.usage), "metrics": {"latencyMs": 10}}},
            ],
            "ResponseMetadata": {"RequestId": str(uuid.uuid4()), "HTTPHeaders": {}},
        }


def converse(model, usage, **kwargs):
    return ChatBedrockConverse(model=model, region_name="us-east-1", client=StubConverseClient(usage),
                               bedrock_client=MagicMock(), callbacks=[ReveniumCallbackHandler()], **kwargs)


def call(llm, style):
    if style == "stream":
        return "".join(chunk.text for chunk in llm.stream("hi"))
    return llm.invoke("hi").text


def assert_net(record, cache_read=0, cache_write=0):
    assert record["input_token_count"] == RAW_INPUT
    assert record["cache_read_token_count"] == cache_read
    assert record["cache_creation_token_count"] == cache_write


@pytest.mark.parametrize("case", sorted(CACHE_CASES))
@pytest.mark.parametrize("style", STYLES)
def test_converse_claude_counts_only_uncached_input(style, case, records):
    cache = CACHE_CASES[case]
    call(converse(CONVERSE_CLAUDE, converse_usage(**cache)), style)

    assert_net(only_record(records), **cache)


@pytest.mark.parametrize("style", STYLES)
def test_a_converse_claude_write_with_its_ttl_split_is_counted_once_at_the_write_rate(style, records):
    """langchain-aws moves a TTL-split write out of ``cache_creation`` into the
    ephemeral keys; the record still carries the whole write and its split."""
    usage = converse_usage(cache_write=4692, cache_details=[{"ttl": "5m", "inputTokens": 4692}])
    call(converse(CONVERSE_CLAUDE, usage), style)

    record = only_record(records)
    assert_net(record, cache_write=4692)
    assert record["cache_creation5m_token_count"] == 4692


@pytest.mark.parametrize("style", STYLES)
def test_a_claude_inference_profile_on_converse_is_netted(style, records):
    """The profile ARN names no model; the base model langchain-aws resolves does."""
    llm = converse(CLAUDE_PROFILE_ARN, converse_usage(cache_read=4000), provider="anthropic",
                   base_model_id="anthropic.claude-haiku-4-5-20251001-v1:0")
    call(llm, style)

    assert_net(only_record(records), cache_read=4000)


@pytest.mark.parametrize("style", STYLES)
def test_converse_nova_keeps_its_input_count(style, records):
    call(converse(CONVERSE_NOVA, converse_usage(cache_read=4000)), style)

    record = only_record(records)
    assert record["input_token_count"] == RAW_INPUT + 4000
    assert record["cache_read_token_count"] == 4000
    assert record["cache_creation_token_count"] == 0


@pytest.mark.parametrize("style", STYLES)
def test_converse_before_langchain_aws_1_4_6_is_not_netted(style, records, monkeypatch):
    """langchain-aws 1.0.0 to 1.4.5 reported Bedrock's own ``inputTokens``, already net.

    The installed langchain-aws still pools the stub's net Converse usage into ``input_tokens``, so a record
    left at the pooled count shows the gate held.
    """
    monkeypatch.setattr(_model_call, "_installed_langchain_aws_version", lambda: "1.4.5", raising=False)
    call(converse(CONVERSE_CLAUDE, converse_usage(cache_read=4000)), style)

    record = only_record(records)
    assert record["input_token_count"] == RAW_INPUT + 4000
    assert record["cache_read_token_count"] == 4000


class StubInvokeClient:
    """The boto3 client ``ChatBedrock`` calls InvokeModel on, counts in the headers."""

    def __init__(self):
        self.meta = MagicMock()

    def invoke_model(self, **kwargs):
        body = {
            "id": "msg_" + uuid.uuid4().hex, "type": "message", "role": "assistant", "model": INVOKE_CLAUDE,
            "content": [{"type": "text", "text": "ok"}], "stop_reason": "end_turn", "stop_sequence": None,
            "usage": {"input_tokens": RAW_INPUT, "output_tokens": 5, "cache_read_input_tokens": 4000,
                      "cache_creation_input_tokens": 0},
        }
        headers = {
            "x-amzn-bedrock-input-token-count": str(RAW_INPUT),
            "x-amzn-bedrock-output-token-count": "5",
            "x-amzn-bedrock-cache-read-input-token-count": "4000",
        }
        return {
            "body": MagicMock(read=MagicMock(return_value=json.dumps(body).encode())),
            "ResponseMetadata": {"RequestId": str(uuid.uuid4()), "HTTPHeaders": headers},
        }


def test_chat_bedrock_invoke_claude_is_not_netted(records, caplog):
    """ChatBedrock shares ``amazon_bedrock`` with Converse but reports Bedrock's net count."""
    caplog.set_level(logging.WARNING, logger="revenium_middleware")
    llm = ChatBedrock(model_id=INVOKE_CLAUDE, region_name="us-east-1", client=StubInvokeClient(),
                      bedrock_client=MagicMock(), callbacks=[ReveniumCallbackHandler()])
    llm.invoke("hi")

    record = only_record(records)
    assert record["input_token_count"] == RAW_INPUT
    assert record["cache_read_token_count"] == 4000
    assert "Usage counts contradict each other" not in caplog.text


class StubMantle:
    """Anthropic Messages on the Bedrock Mantle host, plain and streamed."""

    def __init__(self, usage):
        self.usage = usage

    def respond(self, request):
        assert request.url.host == MANTLE_HOST, f"unexpected upstream {request.url}"
        message_id = "msg_" + uuid.uuid4().hex
        message = {"id": message_id, "type": "message", "role": "assistant", "model": MANTLE_CLAUDE,
                   "content": [{"type": "text", "text": "ok"}], "stop_reason": "end_turn",
                   "stop_sequence": None, "usage": self.usage}
        if not json.loads(request.content).get("stream"):
            return httpx2.Response(200, headers={"content-type": "application/json"}, json=message)
        start = dict(message, content=[], stop_reason=None, usage={**self.usage, "output_tokens": 1})
        events = [
            ("message_start", {"type": "message_start", "message": start}),
            ("content_block_start", {"type": "content_block_start", "index": 0,
                                     "content_block": {"type": "text", "text": ""}}),
            ("content_block_delta", {"type": "content_block_delta", "index": 0,
                                     "delta": {"type": "text_delta", "text": "ok"}}),
            ("content_block_stop", {"type": "content_block_stop", "index": 0}),
            ("message_delta", {"type": "message_delta",
                               "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                               "usage": self.usage}),
            ("message_stop", {"type": "message_stop"}),
        ]
        body = "".join(f"event: {name}\ndata: {json.dumps(data)}\n\n" for name, data in events)
        return httpx2.Response(200, headers={"content-type": "text/event-stream"}, content=body.encode())


def mantle(usage):
    llm = ChatAnthropicMantle(model=MANTLE_CLAUDE, region_name="us-east-1", bedrock_api_key="stub-key",
                              max_tokens=16, max_retries=0, callbacks=[ReveniumCallbackHandler()])
    client_kwargs = dict(api_key="stub-key", base_url=f"https://{MANTLE_HOST}/anthropic", max_retries=0)
    llm.__dict__["_client"] = anthropic.Anthropic(
        http_client=httpx2.Client(transport=httpx2.MockTransport(StubMantle(usage).respond)), **client_kwargs)
    return llm


@pytest.mark.parametrize("case", sorted(CACHE_CASES))
@pytest.mark.parametrize("style", STYLES)
def test_mantle_claude_counts_only_uncached_input(style, case, records, monkeypatch, caplog):
    unwrap_anthropic_middleware(monkeypatch)
    cache = CACHE_CASES[case]
    usage = {"input_tokens": RAW_INPUT, "output_tokens": 5,
             "cache_read_input_tokens": cache.get("cache_read", 0),
             "cache_creation_input_tokens": cache.get("cache_write", 0)}
    caplog.set_level(logging.WARNING, logger="revenium_middleware")
    call(mantle(usage), style)

    assert_net(only_record(records), **cache)
    assert "Usage counts contradict each other" not in caplog.text
