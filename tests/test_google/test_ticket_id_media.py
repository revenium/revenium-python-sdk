"""ticketId forwarding on the Google image/video metering paths.

Uses a real metering client (only the HTTP layer mocked) so a kwarg the
typed methods do not accept fails here instead of raising a swallowed
TypeError in production.
"""
import asyncio
import os
from unittest.mock import MagicMock, patch

from revenium_middleware.google.common.utils import log_image_usage, log_video_usage


def _run_with_real_client(coro_factory):
    from revenium_middleware._metering import ReveniumMetering

    real_client = ReveniumMetering(api_key="test-key")
    mock_post = MagicMock(return_value=type("R", (), {"id": "evt-1"})())
    with patch.object(real_client.ai, "_post", mock_post):
        with patch("revenium_middleware._core.metering.client", real_client):
            asyncio.run(coro_factory())
    assert mock_post.called, "metering call never reached the HTTP layer"
    return mock_post.call_args.kwargs["body"]


def _log_image(usage_metadata):
    return log_image_usage(
        transaction_id="txn-google-image",
        model="imagen-3.0",
        requested_image_count=1,
        actual_image_count=1,
        request_time="2026-08-12T00:00:00Z",
        response_time="2026-08-12T00:00:01Z",
        request_duration=1000,
        usage_metadata=usage_metadata,
        aspect_ratio="16:9",
    )


def _log_video(usage_metadata):
    return log_video_usage(
        transaction_id="txn-google-video",
        model="veo-2.0",
        duration_seconds=5.0,
        request_time="2026-08-12T00:00:00Z",
        response_time="2026-08-12T00:00:01Z",
        request_duration=1000,
        usage_metadata=usage_metadata,
    )


def test_image_forwards_ticket_id_to_the_wire():
    body = _run_with_real_client(lambda: _log_image({"ticket_id": "JIRA-42"}))
    assert body["ticketId"] == "JIRA-42"


def test_video_forwards_ticket_id_to_the_wire():
    body = _run_with_real_client(lambda: _log_video({"ticketId": "JIRA-43"}))
    assert body["ticketId"] == "JIRA-43"


def test_video_env_var_fallback():
    with patch.dict(os.environ, {"REVENIUM_TICKET_ID": "ENV-7"}):
        body = _run_with_real_client(lambda: _log_video({}))
    assert body["ticketId"] == "ENV-7"


def test_image_unset_ticket_id_omitted():
    with patch.dict(os.environ, {}, clear=True):
        body = _run_with_real_client(lambda: _log_image({}))
    assert "ticketId" not in body
