"""Reasoning effort forwarding for the fal provider (BACK-2710).

Only the completion endpoint carries ``effort``; the media params types do
not declare it, so the generation path is the one under test here. Uses a
real metering client (only the HTTP layer mocked) so a kwarg the typed
methods do not accept fails here instead of raising a swallowed TypeError in
production.
"""
import datetime
import time
from unittest.mock import MagicMock, patch

from revenium_middleware.fal._metering import handle_metering


def _completion_body(usage_metadata):
    from revenium_middleware._metering import ReveniumMetering

    real_client = ReveniumMetering(api_key="test-key")
    mock_post = MagicMock(return_value=type("R", (), {"id": "evt-1"})())
    with patch(
        "revenium_middleware.fal._metering.detect_media_type",
        return_value="generation",
    ):
        with patch.object(real_client.ai, "_post", mock_post):
            with patch("revenium_middleware._core.metering.client", real_client):
                handle_metering(
                    application="fal-ai/any-llm",
                    arguments={"prompt": "x"},
                    result={"output": "ok"},
                    request_time_dt=datetime.datetime.now(datetime.timezone.utc),
                    usage_metadata=usage_metadata,
                    transaction_id="fal-effort-test",
                )
                time.sleep(0.3)  # wait for the fire-and-forget metering thread
    assert mock_post.called, "metering call never reached the HTTP layer"
    return mock_post.call_args.kwargs["body"]


def test_completion_forwards_effort_to_the_wire():
    assert _completion_body({"effort": "high"})["effort"] == "high"


def test_completion_forwards_an_unrecognised_level_unchanged():
    assert _completion_body({"effort": "hyper_9"})["effort"] == "hyper_9"


def test_completion_omits_effort_when_unset():
    assert "effort" not in _completion_body({"trace_id": "no-effort-field"})
