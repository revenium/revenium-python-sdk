"""
Copyright (c) 2025 Revenium, Inc.
SPDX-License-Identifier: MIT
"""

"""
A Gemini stream sends exactly one metering record, built from its chunks, however it ends.

The wrappers are the SDK's own StreamWrapper and AsyncStreamWrapper around a
plain iterator of real google-genai response chunks. Early termination by
`break` is covered in test_stream_early_termination.py.
"""

import asyncio
import datetime
import importlib
from unittest.mock import patch

import pytest
from google.genai import types

from revenium_middleware._core.metering_pool import wait_until_idle
from revenium_middleware.google.google_ai import middleware as genai_mw

REQUEST_TIME = datetime.datetime(2026, 10, 8, 12, 0, 0, tzinfo=datetime.timezone.utc)


def make_chunk(text="tok", finish_reason=None, usage=None):
    return types.GenerateContentResponse(
        model_version="gemini-2.5-flash",
        candidates=[types.Candidate(
            content=types.Content(role="model", parts=[types.Part(text=text)]), finish_reason=finish_reason)],
        usage_metadata=types.GenerateContentResponseUsageMetadata(**usage) if usage else None,
    )


FINAL_USAGE = {"prompt_token_count": 12, "candidates_token_count": 7, "total_token_count": 19}


def full_stream():
    return [make_chunk("Hel"), make_chunk("lo"), make_chunk("!", types.FinishReason.STOP, FINAL_USAGE)]


class ProviderStream:
    """The provider's stream: yields the given chunks, then raises `error` if one is set."""

    def __init__(self, chunks, error=None):
        self._chunks = iter(chunks)
        self._error = error
        self.closed = False

    def __iter__(self):
        return self

    def __next__(self):
        try:
            return next(self._chunks)
        except StopIteration:
            if self._error:
                raise self._error
            raise

    def close(self):
        self.closed = True


class AsyncProviderStream:
    def __init__(self, chunks):
        self._chunks = iter(chunks)

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self._chunks)
        except StopIteration:
            raise StopAsyncIteration


def wrap(provider_stream):
    return genai_mw.StreamWrapper(provider_stream, REQUEST_TIME, {})


def sent_records(recorder):
    wait_until_idle(10)
    return [recorded.kwargs for recorded in recorder.ai.create_completion.call_args_list]


class TestOneRecordPerStream:
    def test_a_finished_stream_sends_one_record_with_the_final_chunks_usage(self, mock_revenium_client):
        stream = wrap(ProviderStream(full_stream()))

        assert "".join(chunk.text for chunk in stream) == "Hello!"

        sent = sent_records(mock_revenium_client)
        assert [(r["input_token_count"], r["output_token_count"], r["total_token_count"]) for r in sent] == [
            (12, 7, 19)]
        assert sent[0]["is_streamed"] is True
        assert sent[0]["model"] == "gemini-2.5-flash"

    def test_the_last_chunks_finish_reason_becomes_the_stop_reason(self, mock_revenium_client):
        chunks = [make_chunk("a"), make_chunk("b", types.FinishReason.MAX_TOKENS, FINAL_USAGE)]
        stream = wrap(ProviderStream(chunks))
        list(stream)

        assert [r["stop_reason"] for r in sent_records(mock_revenium_client)] == ["TOKEN_LIMIT"]

    def test_closing_before_the_end_sends_one_record(self, mock_revenium_client):
        provider = ProviderStream(full_stream())
        stream = wrap(provider)
        next(stream)
        stream.close()

        assert len(sent_records(mock_revenium_client)) == 1
        assert provider.closed is True

    def test_repeated_close_and_exit_send_one_record(self, mock_revenium_client):
        stream = wrap(ProviderStream(full_stream()))
        with stream:
            next(stream)
            stream.close()
            stream.close()

        assert len(sent_records(mock_revenium_client)) == 1

    def test_a_stream_with_no_chunks_sends_no_record(self, mock_revenium_client):
        stream = wrap(ProviderStream([]))
        list(stream)
        stream.close()

        assert sent_records(mock_revenium_client) == []

    def test_a_stream_longer_than_the_chunk_cap_still_sends_one_record(self, mock_revenium_client, monkeypatch):
        monkeypatch.setattr(genai_mw.StreamWrapper, "_max_chunks", 2)
        stream = wrap(ProviderStream(full_stream()))
        list(stream)

        sent = sent_records(mock_revenium_client)
        assert len(stream.chunks) == 2
        assert [r["total_token_count"] for r in sent] == [19]

    def test_an_async_stream_sends_one_record_when_it_ends(self, mock_revenium_client):
        stream = genai_mw.AsyncStreamWrapper(AsyncProviderStream(full_stream()), REQUEST_TIME, {})

        async def drain():
            return [chunk async for chunk in stream]

        assert len(asyncio.run(drain())) == 3
        assert [r["total_token_count"] for r in sent_records(mock_revenium_client)] == [19]


class TestProviderErrorsMidStream:
    def test_the_providers_error_is_re_raised_and_the_partial_stream_metered_once(self, mock_revenium_client):
        error = RuntimeError("connection reset by provider")
        chunks = [make_chunk("a", usage={"prompt_token_count": 12, "candidates_token_count": 1, "total_token_count": 13})]
        stream = wrap(ProviderStream(chunks, error=error))

        with pytest.raises(RuntimeError) as raised:
            list(stream)
        stream.close()

        assert raised.value is error
        assert [r["total_token_count"] for r in sent_records(mock_revenium_client)] == [13]

    def test_a_metering_failure_does_not_stop_the_provider_stream_closing(self, mock_revenium_client):
        provider = ProviderStream(full_stream())
        stream = wrap(provider)
        next(stream)

        with patch.object(genai_mw, "create_google_ai_metering_call", side_effect=RuntimeError("metering down")):
            stream.close()

        assert provider.closed is True


class TestTimeToFirstToken:
    def test_is_measured_from_the_request_to_the_first_chunk(self, mock_revenium_client, monkeypatch):
        monkeypatch.setattr(genai_mw, "_now", lambda: REQUEST_TIME + datetime.timedelta(milliseconds=250))
        stream = wrap(ProviderStream(full_stream()))
        list(stream)

        assert [r["time_to_first_token"] for r in sent_records(mock_revenium_client)] == [250]


@pytest.fixture
def capture_with_default_limit(monkeypatch):
    """Capture on, no length setting: google.config as it loads by default, reloaded back afterwards."""
    from revenium_middleware.google import config, prompt_extractor

    with monkeypatch.context() as scoped:
        scoped.setenv("REVENIUM_CAPTURE_PROMPTS", "true")
        scoped.delenv("REVENIUM_MAX_PROMPT_SIZE", raising=False)
        scoped.delenv("REVENIUM_MAX_PROMPT_LENGTH", raising=False)
        importlib.reload(config)
        importlib.reload(prompt_extractor)
        yield
    importlib.reload(config)
    importlib.reload(prompt_extractor)


def reply_chunks(total_length, chunk_length=1_000):
    return [make_chunk("R" * chunk_length) for _ in range(total_length // chunk_length)]


@pytest.mark.usefixtures("capture_with_default_limit")
class TestStreamedReplyCaptureLength:
    def test_a_30000_character_streamed_reply_is_recorded_whole(self, mock_revenium_client):
        list(wrap(ProviderStream(reply_chunks(30_000))))

        (sent,) = sent_records(mock_revenium_client)
        assert sent["output_response"] == "R" * 30_000
        assert sent["prompts_truncated"] is False

    def test_a_60000_character_streamed_reply_is_cut_to_50000_and_flagged(self, mock_revenium_client):
        list(wrap(ProviderStream(reply_chunks(60_000))))

        (sent,) = sent_records(mock_revenium_client)
        assert len(sent["output_response"]) == 50_000
        assert sent["output_response"].endswith("...[TRUNCATED]")
        assert sent["prompts_truncated"] is True

    def test_an_async_30000_character_streamed_reply_is_recorded_whole(self, mock_revenium_client):
        stream = genai_mw.AsyncStreamWrapper(AsyncProviderStream(reply_chunks(30_000)), REQUEST_TIME, {})

        async def drain():
            return [chunk async for chunk in stream]

        asyncio.run(drain())

        (sent,) = sent_records(mock_revenium_client)
        assert sent["output_response"] == "R" * 30_000
        assert sent["prompts_truncated"] is False
