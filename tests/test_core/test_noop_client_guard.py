"""Provider metering callers must safely no-op when the metering client is None
(i.e., REVENIUM_METERING_API_KEY is not set), instead of raising AttributeError
that gets logged as 'REVENIUM FAILURE' on every API call."""

import asyncio
from unittest.mock import patch


def test_google_log_token_usage_noop_when_client_none():
    from revenium_middleware.google.common import utils

    with patch.object(utils, "client", None):
        result = asyncio.run(utils.log_token_usage(
            transaction_id="test-txn-id",
            model="gemini-1.5-pro",
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
            cached_tokens=0,
            stop_reason="END",
            request_time="2026-04-28T00:00:00Z",
            response_time="2026-04-28T00:00:01Z",
            request_duration=1000,
            usage_metadata={},
        ))
        assert result is None


def test_google_log_image_usage_noop_when_client_none():
    from revenium_middleware.google.common import utils

    with patch.object(utils, "client", None):
        result = asyncio.run(utils.log_image_usage(
            transaction_id="test-txn-id",
            model="imagen-3.0-generate-001",
            requested_image_count=1,
            actual_image_count=1,
            request_time="2026-04-28T00:00:00Z",
            response_time="2026-04-28T00:00:01Z",
            request_duration=1000,
            usage_metadata={},
        ))
        assert result is None


def test_google_log_video_usage_noop_when_client_none():
    from revenium_middleware.google.common import utils

    with patch.object(utils, "client", None):
        result = asyncio.run(utils.log_video_usage(
            transaction_id="test-txn-id",
            model="veo-2.0-generate-001",
            duration_seconds=5.0,
            request_time="2026-04-28T00:00:00Z",
            response_time="2026-04-28T00:00:01Z",
            request_duration=1000,
            usage_metadata={},
        ))
        assert result is None
