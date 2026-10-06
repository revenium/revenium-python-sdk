"""Perplexity through AsyncOpenAI is metered once, alongside the OpenAI middleware (BACK-3607)."""
import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import httpx
import pytest

import revenium_middleware.openai  # noqa: F401
import revenium_middleware.perplexity  # noqa: F401
from entry_points import calls
from revenium_middleware.perplexity import middleware as perplexity_middleware


@pytest.fixture
def recorder():
    recording_client = MagicMock()
    with patch("revenium_middleware._core.metering.client", recording_client):
        yield recording_client


def _payloads(recording_client):
    calls.wait_for_metering()
    return calls.recorded_payloads(recording_client)


def _perplexity_client():
    return calls._async_openai(base_url=calls.PERPLEXITY_BASE_URL, model=calls.PERPLEXITY_MODEL,
                               usage=calls._PERPLEXITY_USAGE)


def test_an_async_perplexity_call_is_metered_once_as_perplexity(recorder):
    asyncio.run(_perplexity_client().chat.completions.create(**calls._chat_kwargs(calls.PERPLEXITY_MODEL)))

    payloads = _payloads(recorder)
    assert [(p["provider"], p["input_token_count"], p["output_token_count"]) for p in payloads] == [
        ("PERPLEXITY", calls.INPUT_TOKENS, calls.OUTPUT_TOKENS)]


def test_an_async_perplexity_stream_is_metered_once(recorder):
    async def consume():
        stream = await _perplexity_client().chat.completions.create(
            **calls._chat_kwargs(calls.PERPLEXITY_MODEL), stream=True)
        return [chunk async for chunk in stream]

    assert len(asyncio.run(consume())) == 3
    payloads = _payloads(recorder)
    assert [p["provider"] for p in payloads] == ["PERPLEXITY"]


def test_a_plain_async_openai_call_is_metered_once_as_openai(recorder):
    asyncio.run(calls._async_openai().chat.completions.create(**calls._chat_kwargs()))

    assert [p["provider"] for p in _payloads(recorder)] == ["OPENAI"]


def test_the_async_wrapper_defers_a_non_perplexity_call_untouched():
    returned = object()
    wrapped = MagicMock(return_value=returned)
    instance = SimpleNamespace(_client=SimpleNamespace(base_url="https://api.openai.com/v1"))

    with patch.object(perplexity_middleware, "send_metering_data") as send:
        result = perplexity_middleware.async_create_wrapper(wrapped, instance, (), {"usage_metadata": {"a": 1}})

    assert result is returned
    wrapped.assert_called_once_with(usage_metadata={"a": 1})
    send.assert_not_called()


def _stream_kwargs():
    return {**calls._chat_kwargs(calls.PERPLEXITY_MODEL), "stream": True}


def test_an_async_perplexity_stream_keeps_the_openai_stream_interface(recorder):
    async def consume():
        stream = await _perplexity_client().chat.completions.create(**_stream_kwargs())
        async with stream as entered:
            assert entered is stream
            assert isinstance(stream.response, httpx.Response)
            return [chunk async for chunk in stream]

    assert len(asyncio.run(consume())) == 3
    assert [p["provider"] for p in _payloads(recorder)] == ["PERPLEXITY"]


def test_an_async_perplexity_stream_closed_early_is_metered_once(recorder):
    async def consume_one_then_close():
        stream = await _perplexity_client().chat.completions.create(**_stream_kwargs())
        await stream.__anext__()
        await stream.aclose()
        await stream.close()

    asyncio.run(consume_one_then_close())

    assert [p["provider"] for p in _payloads(recorder)] == ["PERPLEXITY"]


def test_a_sync_perplexity_stream_keeps_the_openai_stream_interface(recorder):
    client = calls._openai(base_url=calls.PERPLEXITY_BASE_URL, model=calls.PERPLEXITY_MODEL,
                           usage=calls._PERPLEXITY_USAGE)

    with client.chat.completions.create(**_stream_kwargs()) as stream:
        assert isinstance(stream.response, httpx.Response)
        chunks = list(stream)

    assert len(chunks) == 3
    payloads = _payloads(recorder)
    assert [(p["provider"], p["input_token_count"], p["output_token_count"]) for p in payloads] == [
        ("PERPLEXITY", calls.INPUT_TOKENS, calls.OUTPUT_TOKENS)]
