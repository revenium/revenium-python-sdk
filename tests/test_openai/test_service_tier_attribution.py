"""Service-tier and pricing field forwarding on the OpenAI completion path.

The usage_metadata tier fields must reach the metering client as the typed
keyword arguments (not via extra_body), and unset fields must stay omitted.
"""
import asyncio
from types import SimpleNamespace
from unittest.mock import patch

from revenium_middleware.openai.middleware import log_token_usage


def _log(usage_metadata):
    asyncio.run(
        log_token_usage(
            response_id="completion-service-tier-test",
            model="gpt-4o-mini",
            prompt_tokens=100,
            completion_tokens=25,
            total_tokens=125,
            cached_tokens=0,
            stop_reason="END",
            request_time="2026-08-12T12:00:00Z",
            response_time="2026-08-12T12:00:01Z",
            request_duration=1000,
            usage_metadata=usage_metadata,
        )
    )


@patch("revenium_middleware.openai.middleware.get_client", lambda: object())
@patch("revenium_middleware.openai.middleware.submit_ai_event")
def test_usage_metadata_tier_fields_forwarded_as_typed_kwargs(mock_submit):
    mock_submit.return_value = SimpleNamespace(id="completion-service-tier-test")

    _log(
        {
            "pricing_tier": "BATCH",
            "requested_service_tier": "priority",
            "actual_service_tier": "default",
            "subscription_tier": "enterprise",
            "cost_multiplier": 0.5,
        }
    )

    payload = mock_submit.call_args[0][1]
    assert payload["pricing_tier"] == "BATCH"
    assert payload["requested_service_tier"] == "priority"
    assert payload["actual_service_tier"] == "default"
    assert payload["subscription_tier"] == "enterprise"
    assert payload["cost_multiplier"] == 0.5
    extra_body = payload.get("extra_body") or {}
    assert "pricingTier" not in extra_body


@patch("revenium_middleware.openai.middleware.get_client", lambda: object())
@patch("revenium_middleware.openai.middleware.submit_ai_event")
def test_unset_tier_fields_omitted_from_payload(mock_submit):
    mock_submit.return_value = SimpleNamespace(id="completion-service-tier-test")

    _log({"trace_id": "no-tier-fields"})

    payload = mock_submit.call_args[0][1]
    for name in (
        "pricing_tier",
        "requested_service_tier",
        "actual_service_tier",
        "subscription_tier",
        "cost_multiplier",
        "priority_tier",
    ):
        assert name not in payload
