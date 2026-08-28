"""Reasoning effort forwarding on the Perplexity completion paths (BACK-2710).

Both the OpenAI-compatible wrapper (``middleware.send_metering_data``) and the
native SDK wrapper (``perplexity_sdk.send_metering_data``) build their own
completion payload, so each is asserted separately. Uses a real metering
client (only the HTTP layer mocked) so a kwarg the typed methods do not
accept fails here instead of raising a swallowed TypeError in production.
"""
import asyncio
import datetime
import time
from unittest.mock import MagicMock, patch


from revenium_middleware.perplexity.middleware import send_metering_data
from revenium_middleware.perplexity.provider import Provider


def _with_real_client(call):
    from revenium_middleware._metering import ReveniumMetering

    real_client = ReveniumMetering(api_key="test-key")
    mock_post = MagicMock(return_value=type("R", (), {"id": "evt-1"})())
    with patch.object(real_client.ai, "_post", mock_post):
        with patch("revenium_middleware._core.metering.client", real_client):
            call()
            time.sleep(0.3)  # wait for the fire-and-forget metering thread
    assert mock_post.called, "metering call never reached the HTTP layer"
    return mock_post.call_args.kwargs["body"]


def _openai_compatible_body(usage_metadata, mock_openai_response):
    return _with_real_client(
        lambda: send_metering_data(
            response=mock_openai_response,
            request_time_dt=datetime.datetime.now(datetime.timezone.utc),
            usage_metadata=usage_metadata,
            model="sonar-pro",
            provider=Provider.PERPLEXITY,
            transaction_id="perplexity-effort-test",
        )
    )


def test_openai_compatible_path_forwards_effort(mock_openai_response):
    body = _openai_compatible_body({"effort": "high"}, mock_openai_response)
    assert body["effort"] == "high"


def test_openai_compatible_path_forwards_an_unrecognised_level(mock_openai_response):
    body = _openai_compatible_body({"effort": "hyper_9"}, mock_openai_response)
    assert body["effort"] == "hyper_9"


def test_openai_compatible_path_omits_effort_when_unset(mock_openai_response):
    body = _openai_compatible_body({"trace_id": "no-effort"}, mock_openai_response)
    assert "effort" not in body


def _native_sdk_body(usage_metadata, response):
    from revenium_middleware.perplexity.perplexity_sdk import (
        send_perplexity_metering_data,
    )

    return _with_real_client(
        lambda: asyncio.run(
            send_perplexity_metering_data(
                response=response,
                model="sonar-pro",
                request_time_dt=datetime.datetime.now(datetime.timezone.utc),
                transaction_id="perplexity-native-effort-test",
                usage_metadata=usage_metadata,
                is_streaming=False,
            )
        )
    )


def test_native_sdk_path_forwards_effort(mock_openai_response):
    """The native Perplexity SDK wrapper builds its own payload."""
    body = _native_sdk_body({"effort": "xhigh"}, mock_openai_response)
    assert body["effort"] == "xhigh"


def test_native_sdk_path_omits_effort_when_unset(mock_openai_response):
    body = _native_sdk_body({"trace_id": "no-effort"}, mock_openai_response)
    assert "effort" not in body
