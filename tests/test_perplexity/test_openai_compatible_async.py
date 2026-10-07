"""Perplexity through AsyncOpenAI is metered once, alongside the OpenAI middleware (BACK-3607)."""
import asyncio
import gc
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import httpx
import pytest

import revenium_middleware.openai  # noqa: F401
import revenium_middleware.perplexity  # noqa: F401
from entry_points import calls
from revenium_middleware import JobContext
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


def _sync_perplexity_client():
    return calls._openai(base_url=calls.PERPLEXITY_BASE_URL, model=calls.PERPLEXITY_MODEL,
                         usage=calls._PERPLEXITY_USAGE)


def _metered_counts(recording_client):
    gc.collect()
    return [(p["provider"], p["input_token_count"], p["output_token_count"]) for p in _payloads(recording_client)]


PERPLEXITY_RECORD = [("PERPLEXITY", calls.INPUT_TOKENS, calls.OUTPUT_TOKENS)]


def test_a_sync_perplexity_stream_dropped_after_its_usage_chunk_is_metered_once(recorder):
    stream = _sync_perplexity_client().chat.completions.create(**_stream_kwargs())
    chunks = [next(stream) for _ in range(3)]
    assert chunks[-1].usage is not None

    del stream

    assert _metered_counts(recorder) == PERPLEXITY_RECORD


def test_a_sync_perplexity_stream_left_to_go_out_of_scope_is_metered_once(recorder):
    def read_up_to_the_usage_chunk():
        for chunk in _sync_perplexity_client().chat.completions.create(**_stream_kwargs()):
            if chunk.usage:
                break

    read_up_to_the_usage_chunk()

    assert _metered_counts(recorder) == PERPLEXITY_RECORD


def test_a_sync_perplexity_stream_read_to_the_end_then_dropped_is_metered_once(recorder):
    stream = _sync_perplexity_client().chat.completions.create(**_stream_kwargs())
    assert len(list(stream)) == 3

    del stream

    assert _metered_counts(recorder) == PERPLEXITY_RECORD


def test_an_async_perplexity_stream_dropped_after_its_usage_chunk_is_metered_once(recorder):
    async def read_up_to_the_usage_chunk():
        stream = await _perplexity_client().chat.completions.create(**_stream_kwargs())
        async for chunk in stream:
            if chunk.usage:
                break

    asyncio.run(read_up_to_the_usage_chunk())

    assert _metered_counts(recorder) == PERPLEXITY_RECORD


def _metered_job_ids(recording_client):
    gc.collect()
    calls.wait_for_metering()
    return [(recorded.kwargs.get("extra_body") or {}).get("agenticJobId")
            for recorded in recording_client.ai.create_completion.call_args_list]


def _stream_read_through_its_usage_chunk():
    stream = _sync_perplexity_client().chat.completions.create(**_stream_kwargs())
    for _ in range(3):
        next(stream)
    return stream


def test_a_stream_dropped_inside_another_job_is_metered_under_the_job_that_made_it(recorder):
    with JobContext(job_id="job-a"):
        streams = [_stream_read_through_its_usage_chunk()]
    with JobContext(job_id="job-b"):
        streams.clear()

    assert _metered_job_ids(recorder) == ["job-a"]


def test_a_stream_dropped_on_another_thread_is_metered_under_the_job_that_made_it(recorder):
    with JobContext(job_id="job-a"):
        streams = [_stream_read_through_its_usage_chunk()]

    def drop_inside_another_job():
        with JobContext(job_id="job-b"):
            streams.clear()

    dropper = threading.Thread(target=drop_inside_another_job)
    dropper.start()
    dropper.join()

    assert _metered_job_ids(recorder) == ["job-a"]
