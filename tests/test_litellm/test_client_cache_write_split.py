"""LiteLLM client calls to Claude are priced like the Anthropic integration (BACK-3992).

Anthropic charges a one-hour cache write at twice the input rate and a
five-minute write at 1.25 times. LiteLLM hands the split over under
``prompt_tokens_details.cache_creation_token_details`` and folds every cache
token into ``prompt_tokens``. The row must carry the split and count only the
uncached prompt as input, or the write is priced at the five-minute rate and
billed a second time inside the input count.

Every call here is a real ``litellm.completion`` or ``litellm.acompletion``
against a local Anthropic-shaped upstream, so the usage is LiteLLM's own.
"""
import asyncio
import datetime
import logging
import threading
from types import SimpleNamespace
from unittest.mock import patch

import pytest

pytest.importorskip("litellm")

import litellm  # noqa: E402
from litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini import VertexGeminiConfig  # noqa: E402
from litellm.llms.vertex_ai.vertex_ai_partner_models.anthropic.transformation import (  # noqa: E402
    VertexAIAnthropicConfig,
)

from revenium_middleware._core import cache_tokens  # noqa: E402
from revenium_middleware.litellm.client import middleware as mw  # noqa: E402

from . import fake_anthropic_upstream, fake_bedrock_upstream, fake_openai_upstream  # noqa: E402

ANTHROPIC_MODEL = "anthropic/claude-haiku-4-5"
MESSAGES = [{"role": "user", "content": "Say ok"}]
CONTRADICTION_WARNING = "Usage counts contradict each other"


def anthropic_usage(input_tokens=16, cache_read=0, five_minute=0, one_hour=3000):
    return {
        "input_tokens": input_tokens,
        "output_tokens": 6,
        "cache_read_input_tokens": cache_read,
        "cache_creation_input_tokens": five_minute + one_hour,
        "cache_creation": {"ephemeral_5m_input_tokens": five_minute, "ephemeral_1h_input_tokens": one_hour},
    }


def run_in_own_thread(coro):
    thread = threading.Thread(target=asyncio.run, args=(coro,))
    thread.start()
    thread.join()
    return thread


@pytest.fixture
def payloads():
    recorded = []
    with patch.object(mw, "run_async_in_thread", side_effect=run_in_own_thread), \
            patch.object(mw, "submit_ai_event", side_effect=lambda op, args: recorded.append(args)):
        yield recorded


@pytest.fixture
def anthropic_upstream():
    servers = []

    def start(usage):
        server = fake_anthropic_upstream.start_upstream(usage)
        servers.append(server)
        return fake_anthropic_upstream.api_base(server)

    yield start
    fake_anthropic_upstream.stop_upstreams(servers)


def call_kwargs(api_base, **extra):
    return {"model": ANTHROPIC_MODEL, "messages": MESSAGES, "api_base": api_base, "api_key": "sk-ant-fake", **extra}


def sync_call(api_base):
    return litellm.completion(**call_kwargs(api_base))


def sync_stream(api_base):
    return list(litellm.completion(**call_kwargs(api_base, stream=True)))


def async_call(api_base):
    return asyncio.run(litellm.acompletion(**call_kwargs(api_base)))


def async_stream(api_base):
    async def drain():
        stream = await litellm.acompletion(**call_kwargs(api_base, stream=True))
        return [chunk async for chunk in stream]

    return asyncio.run(drain())


CALL_STYLES = {
    "completion": sync_call,
    "completion-stream": sync_stream,
    "acompletion": async_call,
    "acompletion-stream": async_stream,
}


def only_row(payloads):
    assert len(payloads) == 1, payloads
    return payloads[0]


@pytest.mark.parametrize("style", sorted(CALL_STYLES))
def test_one_hour_cache_write_is_sent_as_one_hour(style, payloads, anthropic_upstream):
    CALL_STYLES[style](anthropic_upstream(anthropic_usage()))

    row = only_row(payloads)
    assert row["input_token_count"] == 16
    assert row["cache_creation_token_count"] == 3000
    assert row["cache_creation1h_token_count"] == 3000
    assert row["cache_creation5m_token_count"] == 0


def test_five_minute_cache_write_stays_five_minute(payloads, anthropic_upstream):
    sync_call(anthropic_upstream(anthropic_usage(five_minute=3000, one_hour=0)))

    row = only_row(payloads)
    assert row["input_token_count"] == 16
    assert row["cache_creation_token_count"] == 3000
    assert row["cache_creation5m_token_count"] == 3000
    assert row["cache_creation1h_token_count"] == 0


def test_mixed_cache_write_keeps_both_buckets(payloads, anthropic_upstream):
    sync_call(anthropic_upstream(anthropic_usage(five_minute=1000, one_hour=2000)))

    row = only_row(payloads)
    assert row["input_token_count"] == 16
    assert row["cache_creation_token_count"] == 3000
    assert row["cache_creation5m_token_count"] == 1000
    assert row["cache_creation1h_token_count"] == 2000


def test_large_uncached_input_is_netted_once(payloads, anthropic_upstream, caplog):
    """LiteLLM reports 14,000 prompt tokens; 9,000 of them are uncached input."""
    with caplog.at_level(logging.WARNING):
        response = sync_call(anthropic_upstream(anthropic_usage(input_tokens=9000, cache_read=2000)))

    assert response.usage.prompt_tokens == 14000
    row = only_row(payloads)
    assert row["input_token_count"] == 9000
    assert row["cache_read_token_count"] == 2000
    assert row["cache_creation_token_count"] == 3000
    assert row["cache_creation1h_token_count"] == 3000
    assert CONTRADICTION_WARNING not in caplog.text


def test_a_forwarding_entry_point_still_nets_the_cache(payloads, anthropic_upstream):
    """An entry point taking ``*args, **kwargs`` still names the model it was handed."""
    def forwarding_entry_point(*args, **kwargs):
        return litellm.completion.__wrapped__(*args, **kwargs)

    mw.completion_wrapper(forwarding_entry_point, None, (), call_kwargs(anthropic_upstream(anthropic_usage())))

    row = only_row(payloads)
    assert row["input_token_count"] == 16
    assert row["cache_creation1h_token_count"] == 3000


def test_openai_shaped_prompt_count_stays_gross(payloads, monkeypatch):
    """The platform nets OpenAI cache reads itself, so the row keeps LiteLLM's gross count."""
    monkeypatch.setattr(fake_openai_upstream, "USAGE", {
        "prompt_tokens": 100,
        "completion_tokens": 10,
        "total_tokens": 110,
        "prompt_tokens_details": {"cached_tokens": 80},
    })
    server = fake_openai_upstream.start_upstream()
    try:
        litellm.completion(
            model="openai/gpt-4o-mini", messages=MESSAGES,
            api_base=fake_openai_upstream.api_base(server), api_key="sk-fake",
        )
    finally:
        server.shutdown()
        server.server_close()

    row = only_row(payloads)
    assert row["input_token_count"] == 100
    assert row["cache_read_token_count"] == 80
    assert "cache_creation5m_token_count" not in row
    assert "cache_creation1h_token_count" not in row


# --- Claude on LiteLLM's Bedrock and Vertex AI routes (BACK-4162) ---------
#
# LiteLLM folds the cache into prompt_tokens on these routes too, so a Claude
# call there is netted like a direct Anthropic one. Bedrock runs a real
# litellm.completion against a local Converse upstream; Vertex AI cannot be
# pointed at one without Google credentials, so its usage is built by
# LiteLLM's own Vertex transformations and handed to handle_response.

BEDROCK_CLAUDE = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
BEDROCK_NOVA = "us.amazon.nova-micro-v1:0"
VERTEX_CLAUDE = "claude-haiku-4-5@20251001"
VERTEX_GEMINI = "gemini-2.5-flash-lite"


@pytest.fixture
def bedrock_upstream():
    servers = []

    def start(usage):
        server = fake_bedrock_upstream.start_upstream(usage)
        servers.append(server)
        return fake_bedrock_upstream.endpoint(server)

    yield start
    for server in servers:
        server.shutdown()
        server.server_close()


def bedrock_kwargs(endpoint, model=BEDROCK_CLAUDE, **extra):
    return {
        "model": "bedrock/" + model,
        "messages": MESSAGES,
        "aws_bedrock_runtime_endpoint": endpoint,
        "aws_access_key_id": "AKIAFAKEFORTESTS",
        "aws_secret_access_key": "fake-secret-for-tests",
        "aws_region_name": "us-east-1",
        **extra,
    }


def bedrock_call(style, endpoint, model=BEDROCK_CLAUDE):
    kwargs = bedrock_kwargs(endpoint, model)
    streamed = style.endswith("-stream")
    if streamed:
        kwargs["stream"] = True
    if style.startswith("acompletion"):
        async def run():
            response = await litellm.acompletion(**kwargs)
            return [chunk async for chunk in response] if streamed else response

        return asyncio.run(run())
    response = litellm.completion(**kwargs)
    return list(response) if streamed else response


BEDROCK_CACHE_CASES = {
    "write": {"cache_write": 4700},
    "read": {"cache_read": 4700},
    "read-and-write": {"cache_read": 2000, "cache_write": 3000},
}


@pytest.mark.parametrize("case", sorted(BEDROCK_CACHE_CASES))
@pytest.mark.parametrize("style", sorted(CALL_STYLES))
def test_bedrock_claude_counts_only_uncached_input(style, case, payloads, bedrock_upstream):
    cache = BEDROCK_CACHE_CASES[case]
    bedrock_call(style, bedrock_upstream(fake_bedrock_upstream.converse_usage(input_tokens=12, **cache)))

    row = only_row(payloads)
    assert row["input_token_count"] == 12
    assert row["cache_read_token_count"] == cache.get("cache_read", 0)
    assert row["cache_creation_token_count"] == cache.get("cache_write", 0)


@pytest.mark.parametrize("style", ["completion", "completion-stream"])
def test_bedrock_nova_prompt_count_stays_gross(style, payloads, bedrock_upstream):
    """Nova folds the cache on Converse too; it is outside this fix and keeps today's count."""
    endpoint = bedrock_upstream(fake_bedrock_upstream.converse_usage(input_tokens=5, cache_read=4507))
    bedrock_call(style, endpoint, model=BEDROCK_NOVA)

    row = only_row(payloads)
    assert row["input_token_count"] == 4512
    assert row["cache_read_token_count"] == 4507
    assert row["cache_creation_token_count"] == 0


@pytest.mark.parametrize("cache_read", [0, None])
def test_bedrock_claude_without_cache_keeps_its_input(cache_read, payloads, bedrock_upstream):
    usage = fake_bedrock_upstream.converse_usage(input_tokens=900)
    usage["cacheReadInputTokens"] = cache_read
    if cache_read is None:
        del usage["cacheReadInputTokens"], usage["cacheWriteInputTokens"]
    bedrock_call("completion", bedrock_upstream(usage))

    row = only_row(payloads)
    assert row["input_token_count"] == 900
    assert row["cache_read_token_count"] == 0
    assert row["cache_creation_token_count"] == 0


def meter_directly(response, upstream_provider, usage=None, is_streaming=False):
    mw.handle_response(response, datetime.datetime.now(datetime.timezone.utc), {}, is_streaming,
                       usage=usage, upstream_provider=upstream_provider)


def vertex_claude_response(usage_block, litellm_model_name="vertex_ai/" + VERTEX_CLAUDE):
    usage = VertexAIAnthropicConfig().calculate_usage(dict(usage_block), None)
    response = litellm.ModelResponse(model=VERTEX_CLAUDE)
    response.usage = usage
    response._hidden_params["litellm_model_name"] = litellm_model_name
    return response


@pytest.mark.parametrize("litellm_model_name", [
    pytest.param("vertex_ai/" + VERTEX_CLAUDE, id="non-streamed"),
    pytest.param(None, id="streamed-usage-chunk"),
])
def test_vertex_claude_counts_only_uncached_input(litellm_model_name, payloads):
    """A streamed usage chunk names the model on ``response.model`` alone."""
    response = vertex_claude_response(
        {"input_tokens": 12, "output_tokens": 5, "cache_read_input_tokens": 2000,
         "cache_creation_input_tokens": 3000},
        litellm_model_name=litellm_model_name,
    )
    assert response.usage.prompt_tokens == 5012

    meter_directly(response, "vertex_ai", is_streaming=litellm_model_name is None)

    row = only_row(payloads)
    assert row["input_token_count"] == 12
    assert row["cache_read_token_count"] == 2000
    assert row["cache_creation_token_count"] == 3000


def test_vertex_gemini_prompt_count_stays_gross(payloads):
    usage = VertexGeminiConfig._calculate_usage({"usageMetadata": {
        "promptTokenCount": 4911, "candidatesTokenCount": 1, "totalTokenCount": 4912,
        "cachedContentTokenCount": 4084,
    }})
    response = litellm.ModelResponse(model=VERTEX_GEMINI)
    response.usage = usage
    response._hidden_params["litellm_model_name"] = "vertex_ai/" + VERTEX_GEMINI

    meter_directly(response, "vertex_ai")

    row = only_row(payloads)
    assert row["input_token_count"] == 4911
    assert row["cache_read_token_count"] == 4084


def test_cache_larger_than_the_claude_prompt_is_reported_unchanged(payloads, caplog):
    response = litellm.ModelResponse(model=BEDROCK_CLAUDE)
    usage = SimpleNamespace(prompt_tokens=10, completion_tokens=5, cache_read_input_tokens=2000,
                            cache_creation_input_tokens=3000)

    with caplog.at_level(logging.WARNING, logger="revenium_middleware"):
        meter_directly(response, "bedrock", usage=usage)

    row = only_row(payloads)
    assert row["input_token_count"] == 10
    assert row["cache_read_token_count"] == 2000
    assert row["cache_creation_token_count"] == 3000
    assert CONTRADICTION_WARNING in caplog.text


@pytest.mark.parametrize("upstream_provider, model", [
    ("anthropic", "claude-haiku-4-5"),
    ("bedrock", BEDROCK_CLAUDE),
])
def test_a_litellm_that_folds_only_the_read_has_only_the_read_taken_out(
        upstream_provider, model, payloads, monkeypatch, caplog):
    """LiteLLM 1.66 to 1.77.2 added the cache read to ``prompt_tokens`` but not the write.

    6,000 uncached input, 2,000 read and 4,700 written arrive as a prompt
    count of 8,000. Taking the write out as well would bill 1,300.
    """
    monkeypatch.setattr(cache_tokens, "_installed_litellm_version", lambda: "1.70.0", raising=False)
    response = litellm.ModelResponse(model=model)
    usage = SimpleNamespace(prompt_tokens=8000, completion_tokens=5, cache_read_input_tokens=2000,
                            cache_creation_input_tokens=4700)

    with caplog.at_level(logging.WARNING, logger="revenium_middleware"):
        meter_directly(response, upstream_provider, usage=usage)

    row = only_row(payloads)
    assert row["input_token_count"] == 6000
    assert row["cache_read_token_count"] == 2000
    assert row["cache_creation_token_count"] == 4700
    assert CONTRADICTION_WARNING not in caplog.text


def test_litellm_1_77_3_direct_anthropic_has_the_write_taken_out(payloads, monkeypatch, caplog):
    """From 1.77.3 ``AnthropicConfig.calculate_usage`` adds the write to ``prompt_tokens`` again.

    10 uncached input and 4,700 written arrive as a prompt count of 4,710.
    Leaving the write in would bill it at the input rate as well.
    """
    monkeypatch.setattr(cache_tokens, "_installed_litellm_version", lambda: "1.77.3", raising=False)
    response = litellm.ModelResponse(model="claude-haiku-4-5")
    usage = SimpleNamespace(prompt_tokens=4710, completion_tokens=5, cache_read_input_tokens=0,
                            cache_creation_input_tokens=4700)

    with caplog.at_level(logging.WARNING, logger="revenium_middleware"):
        meter_directly(response, "anthropic", usage=usage)

    row = only_row(payloads)
    assert row["input_token_count"] == 10
    assert row["cache_creation_token_count"] == 4700
    assert CONTRADICTION_WARNING not in caplog.text


def test_litellm_1_77_3_bedrock_converse_claude_has_only_the_read_taken_out(payloads, monkeypatch, caplog):
    """Converse kept adding only the read until 1.78.0."""
    monkeypatch.setattr(cache_tokens, "_installed_litellm_version", lambda: "1.77.3", raising=False)
    response = litellm.ModelResponse(model=BEDROCK_CLAUDE)
    usage = SimpleNamespace(prompt_tokens=8000, completion_tokens=5, cache_read_input_tokens=2000,
                            cache_creation_input_tokens=4700)

    with caplog.at_level(logging.WARNING, logger="revenium_middleware"):
        meter_directly(response, "bedrock", usage=usage)

    row = only_row(payloads)
    assert row["input_token_count"] == 6000
    assert row["cache_read_token_count"] == 2000
    assert row["cache_creation_token_count"] == 4700
    assert CONTRADICTION_WARNING not in caplog.text
