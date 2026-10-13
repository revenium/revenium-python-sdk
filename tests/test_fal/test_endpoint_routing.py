"""
Tests for fal.ai endpoint routing.

Verifies that metering calls are routed to the correct endpoint
based on the detected media type:
- IMAGE -> client.ai.create_image
- VIDEO -> client.ai.create_video
- AUDIO -> client.ai.create_audio
- CHAT  -> client.ai.create_completion (language model endpoints, and
           unlisted endpoints that report token usage)
"""

import contextvars
import logging
import pytest
import datetime
import time
from unittest.mock import patch, MagicMock, call

from revenium_middleware._core.metering_pool import wait_until_idle
from revenium_middleware.fal._metering import handle_metering
from revenium_middleware.fal.trace_fields import normalize_model_name


def _wait_for_metering(timeout=2.0):
    """Wait briefly for the async metering thread to complete."""
    time.sleep(0.3)


class TestImageRouting:
    """Tests for image model routing to create_image."""

    def test_flux_routes_to_create_image(self, mock_revenium_client, mock_fal_image_response):
        """Flux image model should route to create_image endpoint."""
        handle_metering(
            application="fal-ai/flux/dev",
            arguments={"prompt": "sunset", "num_images": 1},
            result=mock_fal_image_response,
            request_time_dt=datetime.datetime.now(datetime.timezone.utc),
            usage_metadata={},
            transaction_id="fal-test123",
        )
        _wait_for_metering()

        mock_revenium_client.ai.create_image.assert_called_once()
        call_kwargs = mock_revenium_client.ai.create_image.call_args[1]
        assert call_kwargs["provider"] == "fal_ai"
        assert call_kwargs["model"] == "fal_ai/fal-ai/flux/dev"
        assert call_kwargs["actual_image_count"] == 1
        assert call_kwargs["requested_image_count"] == 1
        assert call_kwargs["resolution"] == "1024x1024"

    def test_multi_image_count(self, mock_revenium_client, mock_fal_multi_image_response):
        """Multiple images should report correct actual_image_count."""
        handle_metering(
            application="fal-ai/flux/dev",
            arguments={"prompt": "cats", "num_images": 3},
            result=mock_fal_multi_image_response,
            request_time_dt=datetime.datetime.now(datetime.timezone.utc),
            usage_metadata={},
            transaction_id="fal-test456",
        )
        _wait_for_metering()

        call_kwargs = mock_revenium_client.ai.create_image.call_args[1]
        assert call_kwargs["actual_image_count"] == 3
        assert call_kwargs["requested_image_count"] == 3

    def test_sdxl_routes_to_create_image(self, mock_revenium_client, mock_fal_image_response):
        """SDXL model should route to create_image."""
        handle_metering(
            application="fal-ai/sdxl",
            arguments={"prompt": "test"},
            result=mock_fal_image_response,
            request_time_dt=datetime.datetime.now(datetime.timezone.utc),
            usage_metadata={},
            transaction_id="fal-sdxl123",
        )
        _wait_for_metering()

        mock_revenium_client.ai.create_image.assert_called_once()


class TestVideoRouting:
    """Tests for video model routing to create_video."""

    def test_video_model_routes_to_create_video(self, mock_revenium_client, mock_fal_video_response):
        """Video model should route to create_video endpoint."""
        handle_metering(
            application="fal-ai/text-to-video",
            arguments={"prompt": "a cat walking"},
            result=mock_fal_video_response,
            request_time_dt=datetime.datetime.now(datetime.timezone.utc),
            usage_metadata={},
            transaction_id="fal-vid123",
        )
        _wait_for_metering()

        mock_revenium_client.ai.create_video.assert_called_once()
        call_kwargs = mock_revenium_client.ai.create_video.call_args[1]
        assert call_kwargs["provider"] == "fal_ai"
        assert call_kwargs["duration_seconds"] == 5.0

    def test_video_with_requested_duration(self, mock_revenium_client, mock_fal_video_response):
        """Video model with requested duration in arguments."""
        handle_metering(
            application="fal-ai/cogvideo",
            arguments={"prompt": "test", "duration": 10},
            result=mock_fal_video_response,
            request_time_dt=datetime.datetime.now(datetime.timezone.utc),
            usage_metadata={},
            transaction_id="fal-vid456",
        )
        _wait_for_metering()

        call_kwargs = mock_revenium_client.ai.create_video.call_args[1]
        assert call_kwargs["requested_duration_seconds"] == 10.0

    def test_kling_routes_to_create_video(self, mock_revenium_client, mock_fal_video_response):
        """Kling video model should route to create_video."""
        handle_metering(
            application="fal-ai/kling/video-gen",
            arguments={"prompt": "test"},
            result=mock_fal_video_response,
            request_time_dt=datetime.datetime.now(datetime.timezone.utc),
            usage_metadata={},
            transaction_id="fal-kling123",
        )
        _wait_for_metering()

        mock_revenium_client.ai.create_video.assert_called_once()

    @pytest.mark.parametrize(
        "application, duration, expected",
        [
            ("fal-ai/veo3.1/image-to-video", "8s", 8.0),
            ("fal-ai/veo3.1/first-last-frame-to-video", "6s", 6.0),
            ("fal-ai/luma-dream-machine/ray-2", "5s", 5.0),
            ("fal-ai/kling-video/v2.1/standard/image-to-video", "5", 5.0),
            ("fal-ai/vidu/q3/image-to-video", 5, 5.0),
            ("bytedance/seedance-2.0/text-to-video", "auto", None),
        ],
    )
    def test_every_documented_fal_duration_spelling_sends_one_video_record(
        self, mock_revenium_client, caplog, application, duration, expected
    ):
        caplog.set_level(logging.DEBUG, logger="revenium_middleware.fal")

        _meter_video(application, duration)

        assert mock_revenium_client.ai.create_video.call_count == 1
        sent = mock_revenium_client.ai.create_video.call_args.kwargs
        assert sent["requested_duration_seconds"] == expected
        if duration == "auto":
            assert _fal_records_at_or_above(caplog, logging.WARNING) == []

    def test_an_unreadable_fal_duration_still_sends_the_record_and_warns_once(
        self, mock_revenium_client, caplog
    ):
        caplog.set_level(logging.DEBUG, logger="revenium_middleware.fal")

        _meter_video("fal-ai/veo3.1/image-to-video", "long")

        assert mock_revenium_client.ai.create_video.call_count == 1
        assert mock_revenium_client.ai.create_video.call_args.kwargs["requested_duration_seconds"] is None
        duration_warnings = [
            r for r in _fal_records_at_or_above(caplog, logging.WARNING)
            if r.levelno == logging.WARNING and "duration" in r.getMessage()
        ]
        assert len(duration_warnings) == 1
        assert [r for r in _fal_records_at_or_above(caplog, logging.ERROR)] == []


def _meter_video(application, duration):
    handle_metering(
        application=application,
        arguments={"prompt": "p", "duration": duration},
        result={"video": {"url": "https://stub/v.mp4"}},
        request_time_dt=datetime.datetime.now(datetime.timezone.utc),
        usage_metadata={},
        transaction_id="fal-duration-test",
    )
    assert wait_until_idle(10)


def _fal_records_at_or_above(caplog, level):
    return [r for r in caplog.records if r.name == "revenium_middleware.fal" and r.levelno >= level]


class TestAudioRouting:
    """Tests for audio model routing to create_audio."""

    def test_tts_routes_to_create_audio(self, mock_revenium_client, mock_fal_audio_response):
        """TTS model should route to create_audio endpoint."""
        handle_metering(
            application="fal-ai/text-to-speech",
            arguments={"text": "Hello, world!"},
            result=mock_fal_audio_response,
            request_time_dt=datetime.datetime.now(datetime.timezone.utc),
            usage_metadata={},
            transaction_id="fal-tts123",
        )
        _wait_for_metering()

        mock_revenium_client.ai.create_audio.assert_called_once()
        call_kwargs = mock_revenium_client.ai.create_audio.call_args[1]
        assert call_kwargs["provider"] == "fal_ai"
        assert call_kwargs["duration_seconds"] == 8.5
        assert call_kwargs["character_count"] == 13
        assert call_kwargs["operation_subtype"] == "speech"

    def test_whisper_routes_to_create_audio(self, mock_revenium_client, mock_fal_transcription_response):
        """Whisper (STT) model should route to create_audio."""
        handle_metering(
            application="fal-ai/whisper",
            arguments={"audio_url": "https://example.com/audio.mp3"},
            result=mock_fal_transcription_response,
            request_time_dt=datetime.datetime.now(datetime.timezone.utc),
            usage_metadata={},
            transaction_id="fal-stt123",
        )
        _wait_for_metering()

        mock_revenium_client.ai.create_audio.assert_called_once()
        call_kwargs = mock_revenium_client.ai.create_audio.call_args[1]
        assert call_kwargs["duration_seconds"] == 12.3
        assert call_kwargs["operation_subtype"] == "transcription"


class TestCompletionFallback:
    """Tests for fallback to create_completion for unknown/text models."""

    def test_generation_type_falls_back_to_completion(self, mock_revenium_client):
        """Models with 'generation' media type fall back to create_completion."""
        # Patch detect_media_type to return 'generation'
        with patch(
            "revenium_middleware.fal._metering.detect_media_type",
            return_value="generation",
        ):
            handle_metering(
                application="fal-ai/some-text-model",
                arguments={"prompt": "test"},
                result={"output": "result"},
                request_time_dt=datetime.datetime.now(datetime.timezone.utc),
                usage_metadata={},
                transaction_id="fal-text123",
            )
            _wait_for_metering()

            mock_revenium_client.ai.create_completion.assert_called_once()


class TestProviderField:
    """Tests that provider is always 'fal_ai' (not 'FAL')."""

    def test_image_provider_is_fal_ai(self, mock_revenium_client, mock_fal_image_response):
        handle_metering(
            application="fal-ai/flux/dev",
            arguments={"prompt": "test"},
            result=mock_fal_image_response,
            request_time_dt=datetime.datetime.now(datetime.timezone.utc),
            usage_metadata={},
            transaction_id="fal-prov1",
        )
        _wait_for_metering()

        call_kwargs = mock_revenium_client.ai.create_image.call_args[1]
        assert call_kwargs["provider"] == "fal_ai"
        assert call_kwargs["model_source"] == "fal_ai"

    def test_video_provider_is_fal_ai(self, mock_revenium_client, mock_fal_video_response):
        handle_metering(
            application="fal-ai/text-to-video",
            arguments={"prompt": "test"},
            result=mock_fal_video_response,
            request_time_dt=datetime.datetime.now(datetime.timezone.utc),
            usage_metadata={},
            transaction_id="fal-prov2",
        )
        _wait_for_metering()

        call_kwargs = mock_revenium_client.ai.create_video.call_args[1]
        assert call_kwargs["provider"] == "fal_ai"

    def test_audio_provider_is_fal_ai(self, mock_revenium_client, mock_fal_audio_response):
        handle_metering(
            application="fal-ai/whisper",
            arguments={},
            result=mock_fal_audio_response,
            request_time_dt=datetime.datetime.now(datetime.timezone.utc),
            usage_metadata={},
            transaction_id="fal-prov3",
        )
        _wait_for_metering()

        call_kwargs = mock_revenium_client.ai.create_audio.call_args[1]
        assert call_kwargs["provider"] == "fal_ai"


REPORTED_USAGE = {"prompt_tokens": 1200, "completion_tokens": 300, "total_tokens": 1525, "cost": 0.0005}


def _meter(application, result):
    handle_metering(
        application=application,
        arguments={"prompt": "hi"},
        result=result,
        request_time_dt=datetime.datetime.now(datetime.timezone.utc),
        usage_metadata={},
        transaction_id="fal-llm-test",
    )
    _wait_for_metering()


def _the_only_record_is_a_completion(client):
    client.ai.create_image.assert_not_called()
    client.ai.create_video.assert_not_called()
    client.ai.create_audio.assert_not_called()
    client.ai.create_completion.assert_called_once()
    return client.ai.create_completion.call_args[1]


def _token_counts(completion):
    return (
        completion["input_token_count"],
        completion["output_token_count"],
        completion["total_token_count"],
    )


def _posted_request(application, result):
    """The path and body the real metering client posts, with only the HTTP layer mocked."""
    from revenium_middleware._metering import ReveniumMetering

    real_client = ReveniumMetering(api_key="test-key")
    mock_post = MagicMock(return_value=type("R", (), {"id": "evt-1"})())
    with patch.object(real_client.ai, "_post", mock_post):
        with patch("revenium_middleware._core.metering.client", real_client):
            _meter(application, result)
    assert mock_post.called, "metering call never reached the HTTP layer"
    return mock_post.call_args.args[0], mock_post.call_args.kwargs["body"]


class TestLanguageModelRouting:
    """A language model call through fal is one chat completion carrying the tokens fal reports."""

    @pytest.mark.parametrize("application", [
        "fal-ai/any-llm",
        "fal-ai/any-llm/enterprise",
        "openrouter/router",
        "openrouter/router/video",
        "openrouter/router/audio",
    ])
    def test_language_model_call_records_a_completion_with_the_tokens_fal_reports(
            self, mock_revenium_client, application):
        _meter(application, {"output": "hi", "usage": dict(REPORTED_USAGE)})

        completion = _the_only_record_is_a_completion(mock_revenium_client)
        assert _token_counts(completion) == (1200, 300, 1525)
        assert completion["provider"] == "fal_ai"
        assert completion["stop_reason"] == "END"
        assert completion["is_streamed"] is False
        assert completion["model"] == normalize_model_name(application)

    def test_missing_total_is_the_sum_of_input_and_output(self, mock_revenium_client):
        _meter("openrouter/router", {"output": "hi", "usage": {"prompt_tokens": 40, "completion_tokens": 9}})

        assert _token_counts(_the_only_record_is_a_completion(mock_revenium_client)) == (40, 9, 49)

    def test_zero_total_beside_reported_counts_is_the_sum_of_input_and_output(self, mock_revenium_client):
        usage = {"prompt_tokens": 12, "completion_tokens": 3, "total_tokens": 0}
        _meter("openrouter/router", {"output": "hi", "usage": usage})

        assert _token_counts(_the_only_record_is_a_completion(mock_revenium_client)) == (12, 3, 15)

    def test_reported_zero_output_tokens_are_sent_as_zero(self, mock_revenium_client):
        usage = {"prompt_tokens": 12, "completion_tokens": 0, "total_tokens": 12}
        _meter("openrouter/router", {"output": "", "usage": usage})

        assert _token_counts(_the_only_record_is_a_completion(mock_revenium_client)) == (12, 0, 12)

    def test_language_model_call_without_usage_records_zero_tokens(self, mock_revenium_client):
        _meter("fal-ai/any-llm", {"output": "hi"})

        assert _token_counts(_the_only_record_is_a_completion(mock_revenium_client)) == (0, 0, 0)

    @pytest.mark.parametrize("result", [None, "text"])
    def test_language_model_call_with_a_non_dict_result_records_zero_tokens(self, mock_revenium_client, result):
        _meter("fal-ai/any-llm", result)

        assert _token_counts(_the_only_record_is_a_completion(mock_revenium_client)) == (0, 0, 0)

    def test_unlisted_endpoint_reporting_token_usage_records_a_completion(self, mock_revenium_client):
        usage = {"prompt_tokens": 7, "completion_tokens": 5, "total_tokens": 12}
        _meter("fal-ai/some-new-model", {"output": "hi", "usage": usage})

        assert _token_counts(_the_only_record_is_a_completion(mock_revenium_client)) == (7, 5, 12)

    def test_responses_style_usage_keys_are_read(self, mock_revenium_client):
        _meter("openrouter/router/openai/v1/responses", {"usage": {"input_tokens": 40, "output_tokens": 9}})

        assert _token_counts(_the_only_record_is_a_completion(mock_revenium_client)) == (40, 9, 49)

    @pytest.mark.parametrize("reported, sent", [
        (12.0, 12),
        (True, 0),
        ("12", 0),
        (None, 0),
        (-5, 0),
    ])
    def test_token_values_are_sent_as_whole_non_negative_numbers(self, mock_revenium_client, reported, sent):
        _meter("openrouter/router", {"output": "hi", "usage": {"prompt_tokens": reported, "completion_tokens": 3}})

        completion = _the_only_record_is_a_completion(mock_revenium_client)
        assert _token_counts(completion) == (sent, 3, sent + 3)
        assert type(completion["input_token_count"]) is int
        assert type(completion["total_token_count"]) is int

    def test_streamed_language_model_call_is_recorded_as_streamed(self, mock_revenium_client):
        pytest.importorskip("fal_client")
        from revenium_middleware.fal._call import FalCall
        from revenium_middleware.fal.middleware import _metered_events

        events = [
            {"output": "h", "partial": True},
            {"output": "hi", "partial": False, "usage": dict(REPORTED_USAGE)},
        ]
        call = FalCall.from_request(("openrouter/router", {"prompt": "hi"}), {})

        assert list(_metered_events(iter(events), call, contextvars.copy_context())) == events
        _wait_for_metering()

        completion = _the_only_record_is_a_completion(mock_revenium_client)
        assert completion["is_streamed"] is True
        assert _token_counts(completion) == (1200, 300, 1525)

    def test_language_model_call_reaches_the_completions_endpoint_with_integer_tokens(self):
        path, body = _posted_request("fal-ai/any-llm", {"output": "hi", "usage": dict(REPORTED_USAGE)})

        assert path == "/v2/ai/completions"
        assert body["inputTokenCount"] == 1200
        assert type(body["inputTokenCount"]) is int
        assert (body["outputTokenCount"], body["totalTokenCount"]) == (300, 1525)

    def test_image_endpoint_reporting_usage_still_records_an_image(
            self, mock_revenium_client, mock_fal_image_response):
        _meter("fal-ai/flux/dev", {**mock_fal_image_response, "usage": dict(REPORTED_USAGE)})

        mock_revenium_client.ai.create_image.assert_called_once()
        assert mock_revenium_client.ai.create_image.call_args[1]["actual_image_count"] == 1
        mock_revenium_client.ai.create_completion.assert_not_called()

    def test_audio_endpoint_reporting_usage_still_records_audio(
            self, mock_revenium_client, mock_fal_transcription_response):
        _meter("fal-ai/whisper", {**mock_fal_transcription_response, "usage": dict(REPORTED_USAGE)})

        mock_revenium_client.ai.create_audio.assert_called_once()
        mock_revenium_client.ai.create_completion.assert_not_called()

    def test_endpoint_named_like_a_language_model_that_returns_images_records_an_image(
            self, mock_revenium_client, mock_fal_image_response):
        _meter("fal-ai/skillmix-xl", mock_fal_image_response)

        mock_revenium_client.ai.create_image.assert_called_once()
        mock_revenium_client.ai.create_video.assert_not_called()
        mock_revenium_client.ai.create_completion.assert_not_called()

    def test_unlisted_endpoint_without_usage_still_records_an_image(self, mock_revenium_client):
        _meter("fal-ai/some-new-model", {"output": "x"})

        mock_revenium_client.ai.create_image.assert_called_once()
        mock_revenium_client.ai.create_completion.assert_not_called()

    def test_unlisted_endpoint_with_unrecognised_usage_keys_still_records_an_image(self, mock_revenium_client):
        _meter("fal-ai/some-new-model", {"output": "x", "usage": {"credits": 3}})

        mock_revenium_client.ai.create_image.assert_called_once()
        mock_revenium_client.ai.create_completion.assert_not_called()
