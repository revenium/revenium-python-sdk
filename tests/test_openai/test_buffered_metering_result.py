"""A usage record the SDK buffered instead of sending is not reported as a metering failure.

``submit_ai_event`` returns None when it buffers the record: after a
retryable delivery failure, and when an overflowed task is built for later
replay. The OpenAI integration read ``result.id`` regardless, so each of those
records also logged three ``REVENIUM FAILURE`` errors with a traceback.
"""
import asyncio
import logging
import time

import httpx
import pytest

from revenium_middleware._core import metering_buffer
from revenium_middleware._core.metering_buffer import MeteringBuffer, delivery_deferred_to_buffer
from revenium_middleware._core.metering_status import get_metering_status, reset_metering_status
from revenium_middleware._metering._exceptions import APIConnectionError
from revenium_middleware.openai.middleware import log_token_usage


@pytest.fixture
def buffer(monkeypatch):
    buf = MeteringBuffer(flush_interval=3600, replay_fn=lambda event, timeout: None)
    monkeypatch.setattr(metering_buffer, "_buffer", buf)
    reset_metering_status()
    yield buf
    reset_metering_status()


def _log():
    return log_token_usage(
        response_id="chatcmpl-buffered",
        model="gpt-4o-mini",
        prompt_tokens=10,
        completion_tokens=5,
        total_tokens=15,
        cached_tokens=0,
        stop_reason="END",
        request_time="2026-10-06T00:00:00Z",
        response_time="2026-10-06T00:00:01Z",
        request_duration=1000,
        usage_metadata={},
    )


async def _log_deferred():
    with delivery_deferred_to_buffer(enqueued_at=time.time()):
        await _log()


def test_a_record_buffered_after_a_retryable_failure_logs_no_metering_failure(buffer, mock_revenium_client, caplog):
    mock_revenium_client.ai.create_completion.side_effect = APIConnectionError(
        request=httpx.Request("POST", "http://metering.test/v2/ai/completions"))

    with caplog.at_level(logging.DEBUG, logger="revenium_middleware"):
        asyncio.run(_log())

    assert "REVENIUM FAILURE" not in caplog.text
    assert buffer.stats()["size"] == 1
    assert get_metering_status().error_count == 1, "only the delivery failure itself is recorded"


def test_an_overflowed_record_built_for_replay_logs_no_metering_failure(buffer, mock_revenium_client, caplog):
    with caplog.at_level(logging.DEBUG, logger="revenium_middleware"):
        asyncio.run(_log_deferred())

    assert "REVENIUM FAILURE" not in caplog.text
    assert buffer.stats()["size"] == 1
    mock_revenium_client.ai.create_completion.assert_not_called()
    assert get_metering_status().error_count == 0
