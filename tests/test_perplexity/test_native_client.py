"""The native Perplexity client is metered exactly once per call (BACK-3607)."""
import asyncio
import json
from unittest.mock import MagicMock, patch

import httpx
import perplexity
import pytest

import revenium_middleware.perplexity  # noqa: F401
from entry_points import calls

MESSAGES = [{"role": "user", "content": "hi"}]


@pytest.fixture
def recorder():
    recording_client = MagicMock()
    with patch("revenium_middleware._core.metering.client", recording_client):
        yield recording_client


def _payloads(recording_client):
    calls.wait_for_metering()
    return calls.recorded_payloads(recording_client)


def _assert_one_perplexity_payload(recording_client):
    payloads = _payloads(recording_client)
    assert len(payloads) == 1, payloads
    payload = payloads[0]
    assert payload["provider"] == "PERPLEXITY"
    assert payload["model"] == calls.PERPLEXITY_MODEL
    assert (payload["input_token_count"], payload["output_token_count"]) == (calls.INPUT_TOKENS, calls.OUTPUT_TOKENS)
    return recording_client.ai.create_completion.call_args.kwargs


class _RecordingHandler:
    def __init__(self):
        self.bodies = []
        self._respond = calls._perplexity_handler()

    def __call__(self, request):
        self.bodies.append(json.loads(request.content or b"{}"))
        return self._respond(request)


def _client(handler=None):
    transport = httpx.MockTransport(handler or calls._perplexity_handler())
    return perplexity.Perplexity(api_key="stub", http_client=httpx.Client(transport=transport))


def _async_client():
    transport = httpx.MockTransport(calls._perplexity_handler())
    return perplexity.AsyncPerplexity(api_key="stub", http_client=httpx.AsyncClient(transport=transport))


def test_sync_create_is_metered_once(recorder):
    _client().chat.completions.create(model=calls.PERPLEXITY_MODEL, messages=MESSAGES)

    sent = _assert_one_perplexity_payload(recorder)
    assert sent["is_streamed"] is False


def test_async_create_is_metered_once(recorder):
    asyncio.run(_async_client().chat.completions.create(model=calls.PERPLEXITY_MODEL, messages=MESSAGES))

    _assert_one_perplexity_payload(recorder)


def test_usage_metadata_is_taken_out_of_extra_body_before_the_request(recorder):
    handler = _RecordingHandler()

    _client(handler).chat.completions.create(model=calls.PERPLEXITY_MODEL, messages=MESSAGES,
                                             extra_body={"usage_metadata": {"trace_id": "t-1"}})

    assert "usage_metadata" not in handler.bodies[0]
    assert _assert_one_perplexity_payload(recorder)["trace_id"] == "t-1"


def test_a_stream_used_as_a_context_manager_is_metered_once_from_its_usage_chunk(recorder):
    create = _client().chat.completions.create
    with create(model=calls.PERPLEXITY_MODEL, messages=MESSAGES, stream=True) as stream:
        chunks = list(stream)

    assert len(chunks) == 3
    sent = _assert_one_perplexity_payload(recorder)
    assert sent["is_streamed"] is True
    assert sent["stop_reason"] == "END"


def test_a_stream_closed_early_is_metered_once(recorder):
    stream = _client().chat.completions.create(model=calls.PERPLEXITY_MODEL, messages=MESSAGES, stream=True)
    next(stream)
    stream.close()
    stream.close()

    payloads = _payloads(recorder)
    assert len(payloads) == 1, payloads


def test_the_wrapped_stream_still_exposes_the_http_response(recorder):
    stream = _client().chat.completions.create(model=calls.PERPLEXITY_MODEL, messages=MESSAGES, stream=True)

    assert isinstance(stream.response, httpx.Response)
    stream.close()


def test_an_async_stream_used_as_a_context_manager_is_metered_once(recorder):
    async def consume():
        create = _async_client().chat.completions.create
        stream = await create(model=calls.PERPLEXITY_MODEL, messages=MESSAGES, stream=True)
        async with stream:
            return [chunk async for chunk in stream]

    assert len(asyncio.run(consume())) == 3
    assert _assert_one_perplexity_payload(recorder)["is_streamed"] is True


def test_a_stream_that_yields_nothing_is_not_metered(recorder):
    stream = _client().chat.completions.create(model=calls.PERPLEXITY_MODEL, messages=MESSAGES, stream=True)
    stream.close()

    assert _payloads(recorder) == []
