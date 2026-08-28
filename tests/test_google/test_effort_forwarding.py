"""Reasoning effort forwarding on the Google completion path (BACK-2710).

Uses a real metering client (only the HTTP layer mocked) so a kwarg the typed
methods do not accept fails here instead of raising a swallowed TypeError in
production.
"""
import asyncio
from unittest.mock import MagicMock, patch

from revenium_middleware.google.common.utils import log_token_usage


def _wire_body(usage_metadata):
    from revenium_middleware._metering import ReveniumMetering

    real_client = ReveniumMetering(api_key="test-key")
    mock_post = MagicMock(return_value=type("R", (), {"id": "evt-1"})())
    with patch.object(real_client.ai, "_post", mock_post):
        with patch("revenium_middleware._core.metering.client", real_client):
            asyncio.run(
                log_token_usage(
                    transaction_id="txn-google-effort",
                    model="gemini-2.5-pro",
                    prompt_tokens=100,
                    completion_tokens=25,
                    total_tokens=125,
                    cached_tokens=0,
                    stop_reason="END",
                    request_time="2026-08-23T12:00:00Z",
                    response_time="2026-08-23T12:00:01Z",
                    request_duration=1000,
                    usage_metadata=usage_metadata,
                )
            )
    assert mock_post.called, "metering call never reached the HTTP layer"
    return mock_post.call_args.kwargs["body"]


def test_effort_reaches_the_wire():
    assert _wire_body({"effort": "high"})["effort"] == "high"


def test_unrecognised_level_reaches_the_wire_unchanged():
    assert _wire_body({"effort": "hyper_9"})["effort"] == "hyper_9"


def test_unset_effort_omitted_from_the_wire():
    assert "effort" not in _wire_body({"trace_id": "no-effort-field"})
