"""Prompt, speed-mode and subagent attribution through the fal entry point (BACK-3388).

fal routes all four /v2/ai paths from one entry point, so it is where a
completion-only field leaking onto audio, image or video would surface as a
swallowed TypeError and a lost metering event.
"""
import datetime
import time
from unittest.mock import MagicMock, patch

import pytest

from revenium_middleware.fal._metering import handle_metering

METADATA = {
    "promptId": "prompt-42",
    "prompt_length": 1834,
    "querySource": "repl_main_thread",
    "speed": "fast",
    "subagentType": "general-purpose",
}

WIRE = {
    "promptId": "prompt-42",
    "promptLength": 1834,
    "querySource": "repl_main_thread",
    "speed": "fast",
    "subagentType": "general-purpose",
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
                transaction_id="fal-prompt-context-test",
            )
            deadline = time.monotonic() + 5.0
            while not mock_post.called and time.monotonic() < deadline:
                time.sleep(0.01)
    assert mock_post.called, "metering call never reached the HTTP layer"
    return mock_post.call_args.kwargs["body"]


def _meter_generation(usage_metadata):
    # detect_media_type routes only an empty application to the completion
    # branch; every named application falls back to image.
    return _meter_with_real_client("", {"output": "hi"}, usage_metadata)


def test_completion_forwards_every_field_to_the_wire():
    body = _meter_generation(METADATA)
    for wire_name, value in WIRE.items():
        assert body[wire_name] == value, wire_name


def test_completion_omits_the_fields_when_unset():
    body = _meter_generation({})
    for wire_name in WIRE:
        assert wire_name not in body, wire_name


@pytest.mark.parametrize(
    "application, fixture_name",
    [
        ("fal-ai/flux/dev", "mock_fal_image_response"),
        ("fal-ai/kling-video/v1", "mock_fal_video_response"),
        ("fal-ai/whisper", "mock_fal_audio_response"),
    ],
)
def test_media_paths_ship_without_forwarding_the_fields(application, fixture_name, request):
    body = _meter_with_real_client(application, request.getfixturevalue(fixture_name), METADATA)
    assert body["transactionId"] == "fal-prompt-context-test"
    for wire_name in ("promptId", "promptLength", "querySource", "subagentType"):
        assert wire_name not in body, wire_name
    assert body.get("speed") != "fast"
