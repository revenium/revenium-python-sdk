"""Reasoning effort forwarding on the OpenAI completion path (BACK-2710).

The usage_metadata ``effort`` level must reach the metering client as the
typed ``effort`` keyword argument (not via extra_body) and land on the wire
under the same name. The wire assertion runs through a real
``ReveniumMetering`` client with only the HTTP layer mocked, so a kwarg the
typed methods do not accept fails here instead of raising a swallowed
TypeError in production.
"""
import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from revenium_middleware.openai.middleware import log_token_usage


def _log(usage_metadata):
    return log_token_usage(
        response_id="completion-effort-test",
        model="gpt-4o-mini",
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


def _payload(usage_metadata):
    with patch("revenium_middleware.openai.middleware.get_client", lambda: object()), \
            patch("revenium_middleware.openai.middleware.submit_ai_event") as mock_submit:
        mock_submit.return_value = SimpleNamespace(id="completion-effort-test")
        asyncio.run(_log(usage_metadata))
    return mock_submit.call_args[0][1]


def _wire_body(usage_metadata):
    from revenium_middleware._metering import ReveniumMetering

    real_client = ReveniumMetering(api_key="test-key")
    mock_post = MagicMock(return_value=type("R", (), {"id": "evt-1"})())
    with patch.object(real_client.ai, "_post", mock_post):
        with patch("revenium_middleware._core.metering.client", real_client):
            asyncio.run(_log(usage_metadata))
    assert mock_post.called, "metering call never reached the HTTP layer"
    return mock_post.call_args.kwargs["body"]


def test_usage_metadata_effort_forwarded_as_typed_kwarg():
    payload = _payload({"effort": "high"})
    assert payload["effort"] == "high"
    assert "effort" not in (payload.get("extra_body") or {})


def test_unset_effort_omitted_from_payload():
    assert "effort" not in _payload({"trace_id": "no-effort-field"})


def test_effort_reaches_the_wire():
    assert _wire_body({"effort": "xhigh"})["effort"] == "xhigh"


def test_unrecognised_level_reaches_the_wire_unchanged():
    """The SDK keeps no allow-list; the backend owns validation."""
    assert _wire_body({"effort": "hyper_9"})["effort"] == "hyper_9"


def test_unset_effort_omitted_from_the_wire():
    assert "effort" not in _wire_body({"trace_id": "no-effort-field"})
