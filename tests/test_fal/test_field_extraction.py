"""
Tests for per-media-type field extraction from fal.ai responses.
"""

import pytest
from revenium_middleware.fal._metering import (
    extract_image_fields,
    extract_video_fields,
    extract_audio_fields,
    generate_transaction_id,
)


class TestExtractImageFields:
    """Tests for image field extraction."""

    def test_single_image_with_dimensions(self, mock_fal_image_response):
        fields = extract_image_fields(mock_fal_image_response, {})
        assert fields["actual_image_count"] == 1
        assert fields["resolution"] == "1024x1024"

    def test_multiple_images(self, mock_fal_multi_image_response):
        fields = extract_image_fields(mock_fal_multi_image_response, {"num_images": 3})
        assert fields["actual_image_count"] == 3
        assert fields["requested_image_count"] == 3

    def test_singular_image_key(self):
        result = {"image": {"url": "test.png", "width": 512, "height": 512}}
        fields = extract_image_fields(result, {})
        assert fields["actual_image_count"] == 1
        assert fields["resolution"] == "512x512"

    def test_resolution_from_arguments(self):
        result = {"images": [{"url": "test.png"}]}
        arguments = {"image_size": {"width": 768, "height": 768}}
        fields = extract_image_fields(result, arguments)
        assert fields["resolution"] == "768x768"

    def test_resolution_string_from_arguments(self):
        result = {"images": [{"url": "test.png"}]}
        arguments = {"image_size": "landscape_16_9"}
        fields = extract_image_fields(result, arguments)
        assert fields["resolution"] == "landscape_16_9"

    def test_non_dict_result(self):
        fields = extract_image_fields("not a dict", {})
        assert fields["actual_image_count"] == 1

    def test_quality_from_arguments(self):
        result = {"images": [{"url": "test.png"}]}
        fields = extract_image_fields(result, {"quality": "hd"})
        assert fields["quality"] == "hd"

    def test_aspect_ratio_from_arguments(self):
        result = {"images": [{"url": "test.png"}]}
        fields = extract_image_fields(result, {"aspect_ratio": "16:9"})
        assert fields["aspect_ratio"] == "16:9"

    def test_default_requested_count(self):
        result = {"images": [{"url": "test.png"}]}
        fields = extract_image_fields(result, {})
        assert fields["requested_image_count"] == 1


class TestExtractVideoFields:
    """Tests for video field extraction."""

    def test_duration_from_video_metadata(self, mock_fal_video_response):
        fields = extract_video_fields(mock_fal_video_response, {})
        assert fields["duration_seconds"] == 5.0

    def test_duration_fallback_to_inference_time(self):
        result = {"timings": {"inference_time": 15.0}}
        fields = extract_video_fields(result, {})
        assert fields["duration_seconds"] == 15.0

    def test_duration_fallback_to_root_inference_time(self):
        result = {"inference_time": 20.0}
        fields = extract_video_fields(result, {})
        assert fields["duration_seconds"] == 20.0

    def test_requested_duration_from_arguments(self):
        result = {"video": {"url": "test.mp4", "duration": 5.0}}
        fields = extract_video_fields(result, {"duration": 10})
        assert fields["duration_seconds"] == 5.0
        assert fields["requested_duration_seconds"] == 10.0

    def test_fps_from_arguments(self):
        result = {"video": {"url": "test.mp4", "duration": 5.0}}
        fields = extract_video_fields(result, {"fps": 30})
        assert fields["fps"] == 30

    def test_default_duration_zero(self):
        result = {"some_field": "value"}
        fields = extract_video_fields(result, {})
        assert fields["duration_seconds"] == 0.0

    def test_non_dict_result(self):
        fields = extract_video_fields("not a dict", {})
        assert fields == {}


class TestExtractAudioFields:
    """Tests for audio field extraction."""

    def test_tts_fields(self, mock_fal_audio_response):
        fields = extract_audio_fields(mock_fal_audio_response, {"text": "Hello world"})
        assert fields["duration_seconds"] == 8.5
        assert fields["character_count"] == 11
        assert fields["operation_subtype"] == "speech"

    def test_stt_fields(self, mock_fal_transcription_response):
        fields = extract_audio_fields(
            mock_fal_transcription_response,
            {"audio_url": "https://example.com/audio.mp3"},
        )
        assert fields["duration_seconds"] == 12.3
        assert fields["operation_subtype"] == "transcription"

    def test_audio_duration_from_root_level(self):
        result = {"audio_length": 7.5}
        fields = extract_audio_fields(result, {})
        assert fields["duration_seconds"] == 7.5

    def test_character_count_from_input_field(self):
        result = {"audio": {"url": "test.mp3"}}
        fields = extract_audio_fields(result, {"input": "test input text"})
        assert fields["character_count"] == 15

    def test_non_dict_result(self):
        fields = extract_audio_fields("not a dict", {})
        assert fields == {}


class TestTransactionIdGeneration:
    """Tests for transaction ID generation."""

    def test_unique_ids(self):
        ids = [generate_transaction_id() for _ in range(100)]
        assert len(ids) == len(set(ids))

    def test_id_format(self):
        tx_id = generate_transaction_id()
        assert tx_id.startswith("fal-")
        assert len(tx_id) == 20  # "fal-" + 16 hex chars
