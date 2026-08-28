"""Service-tier and pricing forwarding for the fal provider.

Guards against a tier field resolved by the middleware but silently dropped
before the wire. Uses a real metering client (with only the HTTP layer
mocked) so a kwarg the typed methods do not accept fails here instead of
raising a swallowed TypeError in production.
"""
import datetime
import time
from unittest.mock import MagicMock, patch

from revenium_middleware.fal._metering import handle_metering

_TIER_METADATA = {
    "actual_service_tier": "flex",
    "requested_service_tier": "priority",
    "pricing_tier": "BATCH",
    "priority_tier": "high",
}


def _meter_with_real_client(application, result, usage_metadata):
    from revenium_middleware._metering import ReveniumMetering

    real_client = ReveniumMetering(api_key="test-key")
    mock_post = MagicMock(return_value=type("R", (), {"id": "evt-1"})())
    with patch.object(real_client.ai, "_post", mock_post):
        with patch("revenium_middleware._core.metering.client", real_client):
            handle_metering(
                application=application,
                arguments={"prompt": "x", "num_images": 1},
                result=result,
                request_time_dt=datetime.datetime.now(datetime.timezone.utc),
                usage_metadata=usage_metadata,
                transaction_id="fal-tier-test",
            )
            time.sleep(0.3)  # wait for the fire-and-forget metering thread
    assert mock_post.called, "metering call never reached the HTTP layer"
    return mock_post.call_args.kwargs["body"]


def _assert_shared_tiers(body):
    assert body["actualServiceTier"] == "flex"
    assert body["requestedServiceTier"] == "priority"
    assert body["pricingTier"] == "BATCH"


def test_image_forwards_service_tier_fields(mock_fal_image_response):
    body = _meter_with_real_client(
        "fal-ai/flux/dev", mock_fal_image_response, dict(_TIER_METADATA)
    )
    _assert_shared_tiers(body)
    assert body["priorityTier"] == "high"


def test_video_forwards_service_tier_fields(mock_fal_video_response):
    body = _meter_with_real_client(
        "fal-ai/kling-video/v1", mock_fal_video_response, dict(_TIER_METADATA)
    )
    _assert_shared_tiers(body)
    assert body["priorityTier"] == "high"


def test_audio_forwards_service_tier_fields(mock_fal_audio_response):
    body = _meter_with_real_client(
        "fal-ai/whisper", mock_fal_audio_response, dict(_TIER_METADATA)
    )
    _assert_shared_tiers(body)
    # priorityTier exists only on the video and image params types
    assert "priorityTier" not in body


def test_completion_forwards_service_tier_fields():
    """The completion path additionally carries subscriptionTier and
    costMultiplier, which the three media params types cannot express."""
    with patch(
        "revenium_middleware.fal._metering.detect_media_type",
        return_value="generation",
    ):
        body = _meter_with_real_client(
            "fal-ai/some-text-model",
            {"output": "hi"},
            {"subscription_tier": "enterprise", "cost_multiplier": 1.5,
             **_TIER_METADATA},
        )
    _assert_shared_tiers(body)
    assert body["subscriptionTier"] == "enterprise"
    assert body["costMultiplier"] == 1.5
    # priorityTier exists only on the video and image params types
    assert "priorityTier" not in body


def test_unset_service_tier_fields_omitted(mock_fal_image_response):
    body = _meter_with_real_client("fal-ai/flux/dev", mock_fal_image_response, {})
    for wire_name in (
        "actualServiceTier",
        "requestedServiceTier",
        "pricingTier",
        "priorityTier",
    ):
        assert wire_name not in body, wire_name
