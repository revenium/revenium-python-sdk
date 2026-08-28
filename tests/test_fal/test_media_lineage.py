"""sourceTransactionId forwarding on the fal image/video metering paths.

Uses a real metering client (with only the HTTP layer mocked) so a kwarg the
typed methods do not accept fails here instead of raising a swallowed
TypeError in production.
"""
import datetime
import time
from unittest.mock import MagicMock, patch


def _meter_with_real_client(application, result, usage_metadata):
    from revenium_middleware._metering import ReveniumMetering
    from revenium_middleware.fal._metering import handle_metering

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
                transaction_id="fal-lineage-test",
            )
            # The metering POST is fire-and-forget on a worker thread: poll the
            # mock with a bounded deadline instead of a fixed sleep, so a loaded
            # runner cannot flake and the happy path pays only milliseconds.
            deadline = time.monotonic() + 5.0
            while not mock_post.called and time.monotonic() < deadline:
                time.sleep(0.01)
    assert mock_post.called, "metering call never reached the HTTP layer"
    return mock_post.call_args.kwargs["body"]


class TestMediaLineageForwarding:
    """Media metering calls must forward source_transaction_id to the wire.

    Guards against a lineage field accepted from usage_metadata but silently
    dropped before the wire.
    """

    def test_image_forwards_source_transaction_id_to_the_wire(self, mock_fal_image_response):
        body = _meter_with_real_client(
            "fal-ai/flux/dev",
            mock_fal_image_response,
            {"source_transaction_id": "txn-source-1"},
        )
        assert body["sourceTransactionId"] == "txn-source-1"

    def test_video_forwards_source_transaction_id_to_the_wire(self, mock_fal_video_response):
        body = _meter_with_real_client(
            "fal-ai/kling-video/v1",
            mock_fal_video_response,
            {"sourceTransactionId": "txn-source-2"},
        )
        assert body["sourceTransactionId"] == "txn-source-2"

    def test_unset_source_transaction_id_omitted_from_the_wire(self, mock_fal_image_response):
        body = _meter_with_real_client("fal-ai/flux/dev", mock_fal_image_response, {})
        assert "sourceTransactionId" not in body
