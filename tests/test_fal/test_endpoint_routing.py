"""
Tests for fal.ai endpoint routing.

Verifies that metering calls are routed to the correct endpoint
based on the detected media type:
- IMAGE -> client.ai.create_image
- VIDEO -> client.ai.create_video
- AUDIO -> client.ai.create_audio
- TEXT  -> client.ai.create_completion (fallback)
"""

import pytest
import datetime
import time
from unittest.mock import patch, MagicMock, call

from revenium_middleware.fal._metering import handle_metering


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
