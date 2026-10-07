"""A usage record the SDK buffered instead of sending is not reported as a metering failure.

``submit_ai_event`` returns None when it buffers the record: after a
retryable delivery failure, and when an overflowed task is built for later
replay. The Google integration read ``result.id`` regardless, logged
``REVENIUM FAILURE`` and raised ``MeteringError``, which the worker then
recorded and logged again as a failed metering thread.
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
from revenium_middleware.google.common.utils import log_image_usage, log_token_usage, log_video_usage

TIMES = {"request_time": "2026-10-06T00:00:00Z", "response_time": "2026-10-06T00:00:01Z", "request_duration": 1000}


def _log_completion():
    return log_token_usage(transaction_id="txn-google-buffered", model="gemini-2.0-flash", prompt_tokens=10,
                           completion_tokens=5, total_tokens=15, cached_tokens=0, stop_reason="END",
                           usage_metadata={}, **TIMES)


def _log_image():
    return log_image_usage(transaction_id="txn-google-image-buffered", model="imagen-3.0", requested_image_count=1,
                           actual_image_count=1, usage_metadata={}, **TIMES)


def _log_video():
    return log_video_usage(transaction_id="txn-google-video-buffered", model="veo-2.0", duration_seconds=5.0,
                           usage_metadata={}, **TIMES)


LOGGERS = {"completion": _log_completion, "image": _log_image, "video": _log_video}


@pytest.fixture
def buffer(monkeypatch):
    buf = MeteringBuffer(flush_interval=3600, replay_fn=lambda event, timeout: None)
    monkeypatch.setattr(metering_buffer, "_buffer", buf)
    reset_metering_status()
    yield buf
    reset_metering_status()


@pytest.mark.parametrize("operation", sorted(LOGGERS))
def test_a_record_buffered_after_a_retryable_failure_is_not_a_metering_failure(
    buffer, mock_revenium_client, caplog, operation
):
    unreachable = APIConnectionError(request=httpx.Request("POST", "http://metering.test/v2/ai"))
    for method in ("create_completion", "create_image", "create_video"):
        getattr(mock_revenium_client.ai, method).side_effect = unreachable

    with caplog.at_level(logging.DEBUG, logger="revenium_middleware"):
        asyncio.run(LOGGERS[operation]())

    assert "REVENIUM FAILURE" not in caplog.text
    assert buffer.stats()["size"] == 1
    assert get_metering_status().error_count == 1, "only the delivery failure itself is recorded"


@pytest.mark.parametrize("operation", sorted(LOGGERS))
def test_an_overflowed_record_built_for_replay_is_not_a_metering_failure(buffer, caplog, operation):
    async def build():
        with delivery_deferred_to_buffer(enqueued_at=time.time()):
            await LOGGERS[operation]()

    with caplog.at_level(logging.DEBUG, logger="revenium_middleware"):
        asyncio.run(build())

    assert "REVENIUM FAILURE" not in caplog.text
    assert buffer.stats()["size"] == 1
    assert get_metering_status().error_count == 0
