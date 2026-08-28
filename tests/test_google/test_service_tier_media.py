"""Service-tier and pricing forwarding on the Google image/video paths.

Uses a real metering client (only the HTTP layer mocked) so a kwarg the
typed methods do not accept fails here instead of raising a swallowed
TypeError in production.
"""
import asyncio
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
        request_time="2026-08-15T00:00:00Z",
        response_time="2026-08-15T00:00:01Z",
        request_duration=1000,
        usage_metadata=usage_metadata,
        aspect_ratio="16:9",
    )


def _log_video(usage_metadata):
    return log_video_usage(
        transaction_id="txn-google-video",
        model="veo-2.0",
        duration_seconds=5.0,
        request_time="2026-08-15T00:00:00Z",
        response_time="2026-08-15T00:00:01Z",
        request_duration=1000,
        usage_metadata=usage_metadata,
    )


def test_image_forwards_service_tier_fields_to_the_wire():
    body = _run_with_real_client(lambda: _log_image({
        "actual_service_tier": "flex",
        "requested_service_tier": "priority",
        "pricing_tier": "BATCH",
        "priority_tier": "high",
    }))
    assert body["actualServiceTier"] == "flex"
    assert body["requestedServiceTier"] == "priority"
    assert body["pricingTier"] == "BATCH"
    assert body["priorityTier"] == "high"


def test_video_forwards_camel_case_aliases_to_the_wire():
    body = _run_with_real_client(lambda: _log_video({
        "actualServiceTier": "flex",
        "requestedServiceTier": "priority",
        "pricingTier": "STANDARD",
        "priorityTier": "low",
    }))
    assert body["actualServiceTier"] == "flex"
    assert body["requestedServiceTier"] == "priority"
    assert body["pricingTier"] == "STANDARD"
    assert body["priorityTier"] == "low"


def test_image_unset_service_tier_fields_omitted():
    body = _run_with_real_client(lambda: _log_image({}))
    for wire_name in (
        "actualServiceTier",
        "requestedServiceTier",
        "pricingTier",
        "priorityTier",
    ):
        assert wire_name not in body, wire_name


def test_video_completion_only_fields_are_not_forwarded():
    """subscriptionTier and costMultiplier exist only on /v2/ai/completions;
    forwarding them to the video path would be a TypeError."""
    body = _run_with_real_client(lambda: _log_video({
        "subscription_tier": "enterprise",
        "cost_multiplier": 1.5,
        "pricing_tier": "BATCH",
    }))
    assert body["pricingTier"] == "BATCH"
    assert "subscriptionTier" not in body
    assert "costMultiplier" not in body


def test_completion_path_forwards_all_tier_fields():
    """The completion path additionally carries subscription_tier and
    cost_multiplier, and must not carry priority_tier."""
    from revenium_middleware.google.common import utils

    with patch.object(utils, "get_client", return_value=object()), \
            patch.object(utils, "submit_ai_event") as mock_submit:
        asyncio.run(utils.log_token_usage(
            transaction_id="txn-google-tier",
            model="gemini-2.0-flash",
            prompt_tokens=100,
            completion_tokens=10,
            total_tokens=110,
            cached_tokens=0,
            stop_reason="END",
            request_time="2026-08-15T00:00:00Z",
            response_time="2026-08-15T00:00:01Z",
            request_duration=1000,
            usage_metadata={
                "actual_service_tier": "flex",
                "requested_service_tier": "priority",
                "pricing_tier": "BATCH",
                "subscription_tier": "enterprise",
                "cost_multiplier": 1.5,
                "priority_tier": "high",
            },
        ))

    args = mock_submit.call_args[0][1]
    assert args["actual_service_tier"] == "flex"
    assert args["requested_service_tier"] == "priority"
    assert args["pricing_tier"] == "BATCH"
    assert args["subscription_tier"] == "enterprise"
    assert args["cost_multiplier"] == 1.5
    assert "priority_tier" not in args
