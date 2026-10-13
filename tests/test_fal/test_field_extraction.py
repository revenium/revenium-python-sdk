"""
Tests for per-media-type field extraction from fal.ai responses.
"""

import logging

import pytest
from revenium_middleware.fal._metering import (
    extract_image_fields,
    extract_video_fields,
    extract_audio_fields,
    generate_transaction_id,
    requested_duration_seconds,
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


class TestRequestedDurationSeconds:
    """fal documents clip lengths as "8s", "5", 5 or "auto"; only a positive finite number of seconds is read."""

    @pytest.mark.parametrize(
        "value, expected",
        [
            ("8s", 8.0),
            ("8S", 8.0),
            (" 8s ", 8.0),
            ("5.5s", 5.5),
            ("8 s", 8.0),
            ("5", 5.0),
            (5, 5.0),
            (7.5, 7.5),
        ],
    )
    def test_a_number_of_seconds_is_read(self, value, expected):
        assert requested_duration_seconds(value) == expected

    @pytest.mark.parametrize(
        "value",
        [
            "0s",
            "9" * 400,
            10**400,
            float("inf"),
            float("nan"),
            True,
            -5,
            "-5s",
            "8sec",
            "1080p",
            "16:9",
            "auto",
            "long",
            [8],
        ],
    )
    def test_anything_else_is_unreadable(self, value):
        assert requested_duration_seconds(value) is None

    @pytest.mark.parametrize("value, expected", [("8s", 8.0), ("auto", None), ("long", None), ("9" * 400, None)])
    def test_extracting_video_fields_reads_the_duration_without_raising(self, value, expected):
        fields = extract_video_fields({"video": {"url": "v.mp4"}}, {"duration": value})
        assert fields.get("requested_duration_seconds") == expected

    @pytest.mark.parametrize("value", ["long", "8sec", True, -5])
    def test_an_unreadable_duration_warns_once(self, caplog, value):
        caplog.set_level(logging.DEBUG, logger="revenium_middleware.fal")
        fields = extract_video_fields({"video": {"url": "v.mp4"}}, {"duration": value})
        assert "requested_duration_seconds" not in fields
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1
        assert "duration" in warnings[0].getMessage()

    @pytest.mark.parametrize("value", ["auto", "AUTO", " Auto "])
    def test_auto_leaves_no_requested_duration_and_does_not_warn(self, caplog, value):
        caplog.set_level(logging.DEBUG, logger="revenium_middleware.fal")
        fields = extract_video_fields({"video": {"url": "v.mp4"}}, {"duration": value})
        assert "requested_duration_seconds" not in fields
        assert [r for r in caplog.records if r.levelno >= logging.WARNING] == []

    @pytest.mark.parametrize("value", ["", None, 0])
    def test_an_absent_duration_leaves_no_field_and_does_not_warn(self, caplog, value):
        caplog.set_level(logging.DEBUG, logger="revenium_middleware.fal")
        fields = extract_video_fields({"video": {"url": "v.mp4"}}, {"duration": value})
        assert "requested_duration_seconds" not in fields
        assert [r for r in caplog.records if r.levelno >= logging.WARNING] == []


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
