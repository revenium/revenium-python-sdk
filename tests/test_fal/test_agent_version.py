"""agentVersion capture and wire forwarding for the fal provider.

fal is the only provider that routes all four /v2/ai media paths from one
entry point, so it is where a field applied to completions but forgotten on
audio, image or video shows up. That is exactly how BACK-2556 escaped, and
these tests are the guard against a repeat for agentVersion.
"""
import datetime
import os
import time
from unittest.mock import MagicMock, patch

import pytest

from revenium_middleware._core.trace_fields import AGENT_VERSION_MAX_LENGTH
from revenium_middleware.fal.trace_fields import get_agent_version
from revenium_middleware.fal._metering import handle_metering


class TestAgentVersionCapture:
    def test_snake_case_alias(self):
        assert get_agent_version({'agent_version': '1.4.2'}) == '1.4.2'

    def test_camel_case_alias(self):
        assert get_agent_version({'agentVersion': '1.4.2'}) == '1.4.2'

    def test_snake_case_takes_precedence(self):
        source = {'agent_version': 'snake', 'agentVersion': 'camel'}
        assert get_agent_version(source) == 'snake'

    def test_none_when_unset(self):
        assert get_agent_version({}) is None
        assert get_agent_version() is None

    def test_no_env_var_fallback(self):
        """Unlike ticketId, the agent version is per-call attribution only."""
        with patch.dict(os.environ, {'REVENIUM_AGENT_VERSION': 'env-1'}, clear=False):
            assert get_agent_version({}) is None


class TestMediaAgentVersionForwarding:
    """Every media metering call must forward agent_version to the wire.

    Uses a real metering client (with only the HTTP layer mocked) so a kwarg
    the typed create_* methods do not accept fails here instead of raising a
    swallowed TypeError in production.
    """

    def _meter_with_real_client(self, application, result, usage_metadata):
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
                    transaction_id="fal-agent-version-test",
                )
                time.sleep(0.3)  # wait for the fire-and-forget metering thread
        assert mock_post.called, "metering call never reached the HTTP layer"
        return mock_post.call_args.kwargs["body"]

    def test_image_forwards_agent_version_to_the_wire(self, mock_fal_image_response):
        body = self._meter_with_real_client(
            "fal-ai/flux/dev", mock_fal_image_response, {"agent_version": "1.4.2"}
        )
        assert body["agentVersion"] == "1.4.2"

    def test_video_forwards_agent_version_to_the_wire(self, mock_fal_video_response):
        body = self._meter_with_real_client(
            "fal-ai/kling-video/v1", mock_fal_video_response, {"agent_version": "1.4.2"}
        )
        assert body["agentVersion"] == "1.4.2"

    def test_audio_forwards_agent_version_to_the_wire(self, mock_fal_audio_response):
        body = self._meter_with_real_client(
            "fal-ai/whisper", mock_fal_audio_response, {"agent_version": "1.4.2"}
        )
        assert body["agentVersion"] == "1.4.2"

    def test_completion_forwards_agent_version_to_the_wire(self):
        body = self._meter_with_real_client(
            "fal-ai/any-llm", {"output": "hi"}, {"agent_version": "1.4.2"}
        )
        assert body["agentVersion"] == "1.4.2"

    def test_camel_case_alias_reaches_the_wire(self, mock_fal_image_response):
        body = self._meter_with_real_client(
            "fal-ai/flux/dev", mock_fal_image_response, {"agentVersion": "1.4.2"}
        )
        assert body["agentVersion"] == "1.4.2"

    @pytest.mark.parametrize("value", [123, [1, 2], {"a": 1}])
    def test_non_string_agent_version_is_dropped_and_the_event_still_ships(
        self, value, mock_fal_image_response
    ):
        """Malformed attribution must not take the metering event with it."""
        body = self._meter_with_real_client(
            "fal-ai/flux/dev", mock_fal_image_response, {"agent_version": value}
        )
        assert "agentVersion" not in body
        assert body["transactionId"] == "fal-agent-version-test"

    def test_unset_agent_version_omitted_from_the_wire(self, mock_fal_image_response):
        body = self._meter_with_real_client("fal-ai/flux/dev", mock_fal_image_response, {})
        assert "agentVersion" not in body

    def test_over_long_agent_version_is_capped_on_the_wire(self, mock_fal_image_response):
        """Capped, not rejected -- exactly how an over-long ticketId behaves."""
        body = self._meter_with_real_client(
            "fal-ai/flux/dev",
            mock_fal_image_response,
            {"agent_version": "v" * (AGENT_VERSION_MAX_LENGTH + 20)},
        )
        assert body["agentVersion"] == "v" * AGENT_VERSION_MAX_LENGTH

    def test_agent_version_and_agentic_job_version_do_not_collide(
        self, mock_fal_image_response
    ):
        """agentVersion is the agent's version; agenticJobVersion is the job
        definition's. They travel independently and neither overwrites the
        other."""
        from revenium_middleware._metering import ReveniumMetering

        real_client = ReveniumMetering(api_key="test-key")
        mock_post = MagicMock(return_value=type("R", (), {"id": "evt-1"})())
        with patch.object(real_client.ai, "_post", mock_post):
            with patch("revenium_middleware._core.metering.client", real_client):
                handle_metering(
                    application="fal-ai/flux/dev",
                    arguments={"prompt": "x", "num_images": 1},
                    result=mock_fal_image_response,
                    request_time_dt=datetime.datetime.now(datetime.timezone.utc),
                    usage_metadata={
                        "agent_version": "agent-1.4.2",
                        "agentic_job_version": "job-9.0.0",
                    },
                    transaction_id="fal-agent-version-collision",
                )
                time.sleep(0.3)
        assert mock_post.called, "metering call never reached the HTTP layer"
        kwargs = mock_post.call_args.kwargs
        # agentVersion is a declared body field; agenticJobVersion still rides
        # extra_json. Distinct destinations, and neither value is the other's.
        assert kwargs["body"]["agentVersion"] == "agent-1.4.2"
        assert "agenticJobVersion" not in kwargs["body"]
        assert kwargs["options"]["extra_json"]["agenticJobVersion"] == "job-9.0.0"
