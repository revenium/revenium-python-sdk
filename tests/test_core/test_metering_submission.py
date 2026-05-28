import uuid
from unittest.mock import MagicMock, patch

import pytest

from revenium_middleware import idempotency_key
from revenium_middleware._core.metering_submission import submit_ai_event


@pytest.fixture
def mock_client():
    """Patch the `client` import that submit_ai_event reads."""
    mock = MagicMock()
    mock.ai.create_completion = MagicMock(return_value="completion-result")
    mock.ai.create_image = MagicMock(return_value="image-result")
    mock.ai.create_video = MagicMock(return_value="video-result")
    mock.ai.create_audio = MagicMock(return_value="audio-result")
    with patch("revenium_middleware._core.metering_submission.client", mock):
        yield mock


def _extract_key(mock_method):
    """Pull the Idempotency-Key from the most recent call's kwargs."""
    kwargs = mock_method.call_args.kwargs
    return kwargs["extra_headers"]["Idempotency-Key"]


def test_submit_completion_auto_generates_uuid_v4(mock_client):
    submit_ai_event("completion", {"some_field": "value"})
    key = _extract_key(mock_client.ai.create_completion)
    parsed = uuid.UUID(key)
    assert parsed.version == 4


def test_submit_completion_uses_explicit_override(mock_client):
    submit_ai_event("completion", {"x": 1}, idempotency_key="my-explicit-key")
    assert _extract_key(mock_client.ai.create_completion) == "my-explicit-key"


def test_submit_completion_uses_contextvar_when_no_explicit(mock_client):
    with idempotency_key("ctx-key"):
        submit_ai_event("completion", {"x": 1})
    assert _extract_key(mock_client.ai.create_completion) == "ctx-key"


def test_explicit_override_wins_over_contextvar(mock_client):
    with idempotency_key("ctx-key"):
        submit_ai_event("completion", {"x": 1}, idempotency_key="explicit")
    assert _extract_key(mock_client.ai.create_completion) == "explicit"


def test_each_call_generates_distinct_uuid(mock_client):
    keys = set()
    for _ in range(3):
        submit_ai_event("completion", {"x": 1})
        keys.add(_extract_key(mock_client.ai.create_completion))
    assert len(keys) == 3


def test_image_operation_routes_to_create_image(mock_client):
    submit_ai_event("image", {"x": 1})
    mock_client.ai.create_image.assert_called_once()
    mock_client.ai.create_completion.assert_not_called()


def test_video_operation_routes_to_create_video(mock_client):
    submit_ai_event("video", {"x": 1})
    mock_client.ai.create_video.assert_called_once()
    mock_client.ai.create_completion.assert_not_called()


def test_audio_operation_routes_to_create_audio(mock_client):
    submit_ai_event("audio", {"x": 1})
    mock_client.ai.create_audio.assert_called_once()
    mock_client.ai.create_completion.assert_not_called()


def test_unknown_operation_raises_value_error(mock_client):
    with pytest.raises(ValueError, match="Unknown AI metering operation"):
        submit_ai_event("invalid", {})


def test_returns_none_when_client_unconfigured():
    with patch("revenium_middleware._core.metering_submission.client", None):
        result = submit_ai_event("completion", {"x": 1})
    assert result is None


def test_preserves_existing_extra_headers(mock_client):
    submit_ai_event(
        "completion",
        {"extra_headers": {"X-Custom": "v"}, "x": 1},
    )
    call_kwargs = mock_client.ai.create_completion.call_args.kwargs
    assert call_kwargs["extra_headers"]["X-Custom"] == "v"
    assert "Idempotency-Key" in call_kwargs["extra_headers"]


def test_preserves_existing_extra_body(mock_client):
    submit_ai_event(
        "completion",
        {"extra_body": {"agentic": "fields"}, "x": 1},
    )
    call_kwargs = mock_client.ai.create_completion.call_args.kwargs
    assert call_kwargs["extra_body"] == {"agentic": "fields"}


def test_args_dict_not_mutated(mock_client):
    args = {"extra_headers": {"X": "Y"}, "x": 1}
    snapshot = {**args, "extra_headers": dict(args["extra_headers"])}
    submit_ai_event("completion", args)
    assert args == snapshot


def test_empty_string_explicit_key_raises_value_error(mock_client):
    """Empty string explicit override is a likely caller bug; raise ValueError instead of silently breaking metering."""
    with pytest.raises(ValueError, match="must be a non-empty string"):
        submit_ai_event("completion", {}, idempotency_key="")
    mock_client.ai.create_completion.assert_not_called()


def test_extra_headers_explicitly_none_in_args(mock_client):
    """args containing extra_headers=None must be treated as no headers (not raise)."""
    submit_ai_event("completion", {"extra_headers": None, "x": 1})
    call_kwargs = mock_client.ai.create_completion.call_args.kwargs
    assert "Idempotency-Key" in call_kwargs["extra_headers"]


def test_raises_if_idempotency_key_passed_in_extra_headers(mock_client):
    """Idempotency-Key in extra_headers is a footgun — the wrapper owns this header. Fail fast."""
    with pytest.raises(ValueError, match="Pass Idempotency-Key via the idempotency_key parameter"):
        submit_ai_event(
            "completion",
            {"extra_headers": {"Idempotency-Key": "would-be-silently-overwritten"}},
        )
    mock_client.ai.create_completion.assert_not_called()
