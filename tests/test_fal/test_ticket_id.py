"""ticketId capture for the fal provider."""
import datetime
import os
import time
from unittest.mock import patch

from revenium_middleware.fal.trace_fields import get_ticket_id
from revenium_middleware.fal._metering import handle_metering


class TestTicketIdCapture:
    def test_env_var_fallback(self):
        with patch.dict(os.environ, {'REVENIUM_TICKET_ID': 'JIRA-77'}):
            assert get_ticket_id({}) == 'JIRA-77'

    def test_metadata_takes_precedence(self):
        with patch.dict(os.environ, {'REVENIUM_TICKET_ID': 'ENV-1'}):
            assert get_ticket_id({'ticketId': 'META-1'}) == 'META-1'

    def test_none_when_unset(self):
        with patch.dict(os.environ, {}, clear=True):
            assert get_ticket_id() is None

    def test_build_common_args_excludes_ticket_id(self):
        """ticket_id is added per endpoint at each call site, not via the
        shared common-args builder."""
        from revenium_middleware.fal._metering import _build_common_args

        with patch.dict(os.environ, {'REVENIUM_TICKET_ID': 'JIRA-77'}):
            args = _build_common_args(
                application="fal-ai/flux/dev",
                request_time_dt=datetime.datetime.now(datetime.timezone.utc),
                usage_metadata={'ticketId': 'META-1'},
                transaction_id="fal-test123",
                is_streamed=False,
            )
            assert 'ticket_id' not in args


class TestMediaTicketIdForwarding:
    """Media metering calls must forward ticket_id all the way to the wire.

    Guards against an attribution field resolved by the middleware but
    silently dropped before the wire. Uses a real metering client (with only
    the HTTP layer mocked) so a kwarg the typed methods do not accept fails
    here instead of raising a swallowed TypeError in production.
    """

    def _meter_with_real_client(self, application, result, usage_metadata):
        from unittest.mock import MagicMock

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
                    transaction_id="fal-ticket-test",
                )
                time.sleep(0.3)  # wait for the fire-and-forget metering thread
        assert mock_post.called, "metering call never reached the HTTP layer"
        return mock_post.call_args.kwargs["body"]

    def test_image_forwards_ticket_id_to_the_wire(self, mock_fal_image_response):
        body = self._meter_with_real_client(
            "fal-ai/flux/dev", mock_fal_image_response, {"ticket_id": "JIRA-42"}
        )
        assert body["ticketId"] == "JIRA-42"

    def test_video_forwards_ticket_id_to_the_wire(self, mock_fal_video_response):
        body = self._meter_with_real_client(
            "fal-ai/kling-video/v1", mock_fal_video_response, {"ticket_id": "JIRA-42"}
        )
        assert body["ticketId"] == "JIRA-42"

    def test_audio_forwards_ticket_id_to_the_wire(self, mock_fal_audio_response):
        body = self._meter_with_real_client(
            "fal-ai/whisper", mock_fal_audio_response, {"ticket_id": "JIRA-42"}
        )
        assert body["ticketId"] == "JIRA-42"

    def test_unset_ticket_id_omitted_from_the_wire(self, mock_fal_image_response):
        with patch.dict(os.environ, {}, clear=True):
            body = self._meter_with_real_client(
                "fal-ai/flux/dev", mock_fal_image_response, {}
            )
        assert "ticketId" not in body

    def test_completion_forwards_ticket_id_to_the_wire(self):
        body = self._meter_with_real_client(
            "fal-ai/any-llm", {"output": "hi"}, {"ticket_id": "JIRA-42"}
        )
        assert body["ticketId"] == "JIRA-42"

    def test_completion_unset_ticket_id_omitted_from_the_wire(self):
        with patch.dict(os.environ, {}, clear=True):
            body = self._meter_with_real_client("fal-ai/any-llm", {"output": "hi"}, {})
        assert "ticketId" not in body
