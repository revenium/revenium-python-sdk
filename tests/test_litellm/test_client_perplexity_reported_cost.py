"""LiteLLM client calls served by Perplexity carry the cost Perplexity reported (BACK-4356).

Perplexity's ``total_cost`` includes a flat request fee and search fees that no
token count can rebuild, so a record priced from tokens alone is hundreds of
times too low. The fixture's total is deliberately not the sum of the token
parts, so only forwarding Perplexity's own figure passes.

Every call here is a real ``litellm.completion`` or ``litellm.acompletion``
against a local Perplexity-shaped upstream, so the usage shapes are LiteLLM's
own: the cost dict on a plain answer, a bare number on a stream's usage chunk.
"""
import asyncio
import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("litellm")

import litellm  # noqa: E402

from revenium_middleware.litellm.client import middleware as mw  # noqa: E402

from . import fake_perplexity_upstream  # noqa: E402

PERPLEXITY_MODEL = "perplexity/sonar"
MESSAGES = [{"role": "user", "content": "Say ok"}]
REPORTED_TOTAL = fake_perplexity_upstream.COST["total_cost"]
REPORTED_INPUT = fake_perplexity_upstream.COST["input_tokens_cost"]
REPORTED_OUTPUT = fake_perplexity_upstream.COST["output_tokens_cost"]
COST_FIELDS = ("input_token_cost", "output_token_cost", "total_cost")


def run_in_own_thread(coro):
    thread = threading.Thread(target=asyncio.run, args=(coro,))
    thread.start()
    thread.join()
    return thread


@pytest.fixture
def payloads():
    recorded = []
    with patch.object(mw, "run_async_in_thread", side_effect=run_in_own_thread), \
            patch.object(mw, "submit_ai_event", side_effect=lambda op, args, **_: recorded.append(args)):
        yield recorded


@pytest.fixture(scope="module")
def upstream():
    server = fake_perplexity_upstream.start_upstream()
    yield server
    server.shutdown()
    server.server_close()


def call_kwargs(api_base, model=PERPLEXITY_MODEL, **extra):
    return {"model": model, "messages": MESSAGES, "api_base": api_base, "api_key": "pplx-fake", **extra}


def sync_call(**kwargs):
    return [litellm.completion(**kwargs)]


def sync_stream(**kwargs):
    return list(litellm.completion(stream=True, **kwargs))


def async_call(**kwargs):
    return [asyncio.run(litellm.acompletion(**kwargs))]


def async_stream(**kwargs):
    async def drain():
        stream = await litellm.acompletion(stream=True, **kwargs)
        return [chunk async for chunk in stream]

    return asyncio.run(drain())


CALL_STYLES = {
    "completion": sync_call,
    "completion-stream": sync_stream,
    "acompletion": async_call,
    "acompletion-stream": async_stream,
}
PLAIN_STYLES = ["acompletion", "completion"]
STREAM_STYLES = ["acompletion-stream", "completion-stream"]


def only_row(payloads):
    assert len(payloads) == 1, payloads
    return payloads[0]


def sent_cost(row):
    return {field: row.get(field) for field in COST_FIELDS}


@pytest.mark.parametrize("style", PLAIN_STYLES)
def test_a_perplexity_answer_through_litellm_is_recorded_at_perplexitys_reported_cost(style, payloads, upstream):
    CALL_STYLES[style](**call_kwargs(fake_perplexity_upstream.api_base(upstream)))

    row = only_row(payloads)
    assert row["provider"] == "LITELLM"
    assert row["is_streamed"] is False
    assert sent_cost(row) == {
        "input_token_cost": pytest.approx(REPORTED_INPUT),
        "output_token_cost": pytest.approx(REPORTED_OUTPUT),
        "total_cost": pytest.approx(REPORTED_TOTAL),
    }


@pytest.mark.parametrize("style", STREAM_STYLES)
def test_a_streamed_perplexity_answer_through_litellm_is_recorded_at_perplexitys_reported_cost(
        style, payloads, upstream):
    chunks = CALL_STYLES[style](**call_kwargs(fake_perplexity_upstream.api_base(upstream)))

    row = only_row(payloads)
    assert row["is_streamed"] is True
    assert row["total_cost"] == pytest.approx(REPORTED_TOTAL)
    assert row["input_token_cost"] in (None, pytest.approx(REPORTED_INPUT))
    assert all(chunk.choices for chunk in chunks), "the usage chunk the SDK asked for reached the caller"


@pytest.mark.parametrize("style", sorted(CALL_STYLES))
def test_an_openai_route_at_perplexitys_address_is_recorded_at_perplexitys_reported_cost(style, payloads, upstream):
    perplexity_address = fake_perplexity_upstream.api_base(upstream, "/perplexity")

    CALL_STYLES[style](**call_kwargs(perplexity_address, model="openai/sonar"))

    assert only_row(payloads)["total_cost"] == pytest.approx(REPORTED_TOTAL)


@pytest.mark.parametrize("style", sorted(CALL_STYLES))
def test_a_non_perplexity_litellm_call_still_sends_no_cost(style, payloads, upstream):
    CALL_STYLES[style](**call_kwargs(fake_perplexity_upstream.api_base(upstream), model="openai/gpt-4o-mini"))

    assert sent_cost(only_row(payloads)) == {"input_token_cost": None, "output_token_cost": None, "total_cost": None}


def _wire_body_of(call):
    from revenium_middleware._metering import ReveniumMetering

    real_client = ReveniumMetering(api_key="test-key")
    mock_post = MagicMock(return_value=SimpleNamespace(id="evt-1"))
    with patch.object(real_client.ai, "_post", mock_post), \
            patch("revenium_middleware._core.metering.client", real_client):
        call()
        deadline = time.monotonic() + 5.0
        while not mock_post.called and time.monotonic() < deadline:
            time.sleep(0.01)
    assert mock_post.call_count == 1
    return mock_post.call_args.kwargs["body"]


@pytest.mark.parametrize("style", ["completion", "completion-stream"])
def test_perplexitys_reported_cost_reaches_the_wire_as_totalCost(style, upstream):
    body = _wire_body_of(lambda: CALL_STYLES[style](**call_kwargs(fake_perplexity_upstream.api_base(upstream))))

    assert body["totalCost"] == pytest.approx(REPORTED_TOTAL)
    assert body["provider"] == "LITELLM"
    assert "requestCost" not in body
    assert "searchQueriesCost" not in body


@pytest.mark.parametrize("provider, api_base, expected", [
    pytest.param("perplexity", None, True, id="perplexity-provider"),
    pytest.param("openai", "https://api.perplexity.ai", True, id="openai-route-at-perplexity"),
    pytest.param("openai_like", "https://api.perplexity.ai", True, id="openai-like-route-at-perplexity"),
    pytest.param("openai", "https://api.openai.com/v1", False, id="openai-at-openai"),
    pytest.param("anthropic", "https://perplexity-gateway.example.com", False, id="anthropic-at-perplexity-named-host"),
    pytest.param("azure", "https://perplexity.openai.azure.com", False, id="azure-at-perplexity-named-host"),
    pytest.param("hosted_vllm", "https://perplexity.internal", False, id="vllm-at-perplexity-named-host"),
    pytest.param("openai", None, False, id="openai-without-address"),
])
def test_only_perplexity_or_an_openai_route_at_its_address_counts_as_served_by_perplexity(
        provider, api_base, expected):
    assert mw._served_by_perplexity(SimpleNamespace(kwargs={"api_base": api_base}), provider) is expected


def close_sync_stream_after_first_chunk(**kwargs):
    stream = litellm.completion(stream=True, **kwargs)
    next(iter(stream))
    stream.close()


def close_async_stream_after_first_chunk(**kwargs):
    async def open_take_one_and_close():
        stream = await litellm.acompletion(stream=True, **kwargs)
        await stream.__anext__()
        await stream.aclose()

    asyncio.run(open_take_one_and_close())


@pytest.mark.parametrize("close_early", [close_sync_stream_after_first_chunk, close_async_stream_after_first_chunk],
                         ids=["close", "aclose"])
def test_a_perplexity_stream_closed_before_its_charge_arrives_is_not_recorded_as_free(
        close_early, payloads, upstream):
    close_early(**call_kwargs(fake_perplexity_upstream.api_base(upstream)))

    assert only_row(payloads)["total_cost"] is None
