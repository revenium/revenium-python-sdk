"""Each Perplexity record carries the cost Perplexity reported for the call (BACK-3988).

Perplexity returns its own charge in ``usage.cost``, and its ``total_cost`` already includes the
flat per-request fee. The fixture's total is deliberately not the sum of the listed parts, so only
forwarding ``total_cost`` as reported passes.
"""
import asyncio
import datetime
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import httpx
import openai
import perplexity
import pytest

import revenium_middleware.openai  # noqa: F401
import revenium_middleware.perplexity  # noqa: F401
from entry_points import calls
from revenium_middleware.perplexity.perplexity_sdk import send_perplexity_metering_data

MESSAGES = [{"role": "user", "content": "hi"}]
REPORTED_COST = {"input_tokens_cost": 0.00011, "output_tokens_cost": 1e-05, "request_cost": 0.0025,
                 "search_queries_cost": 0.005, "total_cost": 0.00762}
EXPECTED = {"input_token_cost": 0.00011, "output_token_cost": 1e-05, "total_cost": 0.00762}
COST_FIELDS = ("input_token_cost", "output_token_cost", "total_cost")
USAGE_WITH_COST = {**calls._PERPLEXITY_USAGE, "cost": REPORTED_COST}
ZERO_COST_SKELETON = {"input_tokens_cost": 0.0, "output_tokens_cost": 0.0, "total_cost": 0.0}


@pytest.fixture
def recorder():
    recording_client = MagicMock()
    with patch("revenium_middleware._core.metering.client", recording_client):
        yield recording_client


def _the_one_record(recording_client):
    calls.wait_for_metering()
    records = recording_client.ai.create_completion.call_args_list
    assert len(records) == 1, records
    return records[0].kwargs


def _sent_cost(kwargs):
    return {field: kwargs[field] for field in COST_FIELDS if field in kwargs}


def _assert_reported_cost_sent(kwargs):
    assert kwargs["provider"] == "PERPLEXITY"
    assert _sent_cost(kwargs) == pytest.approx(EXPECTED)
    assert kwargs["total_cost"] != pytest.approx(REPORTED_COST["request_cost"])


def _handler(usage=USAGE_WITH_COST):
    return calls._openai_compatible_handler(calls.PERPLEXITY_MODEL, usage)


def _native(handler):
    return perplexity.Perplexity(api_key="stub", http_client=httpx.Client(transport=httpx.MockTransport(handler)))


def _async_native(handler):
    return perplexity.AsyncPerplexity(api_key="stub",
                                      http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))


def _openai_client(handler):
    return openai.OpenAI(api_key="stub", base_url=calls.PERPLEXITY_BASE_URL,
                         http_client=httpx.Client(transport=httpx.MockTransport(handler)))


def _async_openai_client(handler):
    return openai.AsyncOpenAI(api_key="stub", base_url=calls.PERPLEXITY_BASE_URL,
                              http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))


def _create(client, **kwargs):
    return client.chat.completions.create(model=calls.PERPLEXITY_MODEL, messages=MESSAGES, **kwargs)


def _consume_stream(client):
    return list(_create(client, stream=True))


def test_a_native_call_sends_the_cost_perplexity_reported(recorder):
    _create(_native(_handler()))

    _assert_reported_cost_sent(_the_one_record(recorder))


def test_an_async_native_call_sends_the_cost_perplexity_reported(recorder):
    asyncio.run(_create(_async_native(_handler())))

    _assert_reported_cost_sent(_the_one_record(recorder))


def test_a_native_stream_sends_the_cost_from_its_final_chunk(recorder):
    _consume_stream(_native(_handler()))

    kwargs = _the_one_record(recorder)
    assert kwargs["is_streamed"] is True
    _assert_reported_cost_sent(kwargs)


def test_an_openai_client_call_to_perplexity_sends_the_reported_cost(recorder):
    _create(_openai_client(_handler()))

    _assert_reported_cost_sent(_the_one_record(recorder))


def test_an_openai_client_stream_to_perplexity_sends_the_reported_cost(recorder):
    _consume_stream(_openai_client(_handler()))

    kwargs = _the_one_record(recorder)
    assert kwargs["is_streamed"] is True
    _assert_reported_cost_sent(kwargs)


def test_an_async_openai_client_call_to_perplexity_sends_the_reported_cost(recorder):
    asyncio.run(_create(_async_openai_client(_handler())))

    _assert_reported_cost_sent(_the_one_record(recorder))


def _stream_with_zero_cost_interim_chunks(request):
    """Perplexity's real stream shape: a zero-cost skeleton on interim chunks, the charge on the last."""
    base = {"id": "chatcmpl-1", "object": "chat.completion.chunk", "created": 1, "model": calls.PERPLEXITY_MODEL}
    interim_usage = {**calls._PERPLEXITY_USAGE, "cost": ZERO_COST_SKELETON}
    chunks = [
        {**base, "usage": interim_usage,
         "choices": [{"index": 0, "delta": {"role": "assistant", "content": "o"}, "finish_reason": None}]},
        {**base, "usage": interim_usage,
         "choices": [{"index": 0, "delta": {"content": "k"}, "finish_reason": None}]},
        {**base, "usage": USAGE_WITH_COST,
         "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
    ]
    body = calls._sse([(None, chunk) for chunk in chunks]) + b"data: [DONE]\n\n"
    return httpx.Response(200, content=body, headers={"content-type": "text/event-stream"})


@pytest.mark.parametrize("make_client", [
    pytest.param(_native, id="native"),
    pytest.param(_openai_client, id="openai-client"),
])
def test_a_stream_with_zero_cost_interim_chunks_sends_the_terminal_cost(recorder, make_client):
    _consume_stream(make_client(_stream_with_zero_cost_interim_chunks))

    _assert_reported_cost_sent(_the_one_record(recorder))


async def _consume_async_stream(client):
    stream = await _create(client, stream=True)
    return [chunk async for chunk in stream]


@pytest.mark.parametrize("make_client", [
    pytest.param(_async_native, id="async-native"),
    pytest.param(_async_openai_client, id="async-openai-client"),
])
def test_an_async_stream_with_zero_cost_interim_chunks_sends_the_terminal_cost(recorder, make_client):
    asyncio.run(_consume_async_stream(make_client(_stream_with_zero_cost_interim_chunks)))

    kwargs = _the_one_record(recorder)
    assert kwargs["is_streamed"] is True
    _assert_reported_cost_sent(kwargs)


def test_a_zero_reported_cost_is_sent_as_zero(recorder):
    _create(_native(_handler({**calls._PERPLEXITY_USAGE, "cost": ZERO_COST_SKELETON})))

    kwargs = _the_one_record(recorder)
    assert "total_cost" in kwargs
    assert _sent_cost(kwargs) == {"input_token_cost": 0.0, "output_token_cost": 0.0, "total_cost": 0.0}


@pytest.mark.parametrize("make_client", [
    pytest.param(_native, id="native"),
    pytest.param(_openai_client, id="openai-client"),
])
def test_a_response_without_cost_sends_no_cost_fields(recorder, make_client):
    _create(make_client(_handler(calls._PERPLEXITY_USAGE)))

    assert _sent_cost(_the_one_record(recorder)) == {}


def test_only_the_cost_fields_perplexity_reported_are_sent(recorder):
    _create(_openai_client(_handler({**calls._PERPLEXITY_USAGE, "cost": {"total_cost": 0.003}})))

    assert _sent_cost(_the_one_record(recorder)) == pytest.approx({"total_cost": 0.003})


@pytest.mark.parametrize("response", [
    pytest.param(MagicMock(), id="mock-usage"),
    pytest.param(SimpleNamespace(usage=SimpleNamespace(cost={"total_cost": "0.003"})), id="string"),
    pytest.param(SimpleNamespace(usage=SimpleNamespace(cost={"total_cost": True})), id="bool"),
    pytest.param(SimpleNamespace(usage=SimpleNamespace(cost={"total_cost": float("nan")})), id="nan"),
    pytest.param(SimpleNamespace(usage=SimpleNamespace(cost={"total_cost": float("inf")})), id="inf"),
    pytest.param(SimpleNamespace(usage=SimpleNamespace(cost=True)), id="bare-bool"),
    pytest.param(SimpleNamespace(usage=SimpleNamespace(cost=float("nan"))), id="bare-nan"),
    pytest.param(SimpleNamespace(usage=SimpleNamespace(cost=float("inf"))), id="bare-inf"),
    pytest.param(SimpleNamespace(usage=SimpleNamespace(cost="0.003")), id="bare-string"),
])
def test_a_non_numeric_cost_is_not_sent(response):
    from revenium_middleware._core.perplexity_cost import extract_reported_cost

    assert extract_reported_cost(response) == {}


@pytest.mark.parametrize("cost, expected", [
    pytest.param(0.00267, {"total_cost": 0.00267}, id="charge"),
    pytest.param(0, {"total_cost": 0.0}, id="zero"),
])
def test_a_bare_number_cost_is_taken_as_perplexitys_total(cost, expected):
    """LiteLLM flattens Perplexity's cost to the bare total on a stream's usage chunk (BACK-4356)."""
    from revenium_middleware._core.perplexity_cost import extract_reported_cost

    assert extract_reported_cost(SimpleNamespace(usage=SimpleNamespace(cost=cost))) == pytest.approx(expected)


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


def test_the_reported_cost_reaches_the_wire_as_totalCost():
    now = datetime.datetime.now(datetime.timezone.utc)
    response = SimpleNamespace(
        usage=SimpleNamespace(prompt_tokens=6, completion_tokens=1, total_tokens=7,
                              cost=SimpleNamespace(**REPORTED_COST)),
        choices=[],
    )

    body = _wire_body_of(lambda: asyncio.run(send_perplexity_metering_data(
        response=response,
        model=calls.PERPLEXITY_MODEL,
        request_time_dt=now,
        response_time_dt=now,
        transaction_id="perplexity-reported-cost-wire",
        usage_metadata={},
        is_streaming=False,
    )))

    assert body["totalCost"] == pytest.approx(0.00762)
    assert body["inputTokenCost"] == pytest.approx(0.00011)
    assert body["outputTokenCost"] == pytest.approx(1e-05)
    assert "requestCost" not in body
