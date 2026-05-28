"""
Tests for multimodal metering support (Imagen, Veo, Vision).

Tests cover:
- Vision content detection in various Google AI content formats
- OperationType enum values for IMAGE and VIDEO
- Image metering function (log_image_usage)
- Video metering function (log_video_usage)
- Image/video metering call creation helpers
"""

import asyncio
import datetime
import pytest
from unittest.mock import Mock, MagicMock, patch, AsyncMock

from revenium_middleware.google.common.types import OperationType
from revenium_middleware.google.common.trace_fields import (
    detect_vision_content,
    _item_has_vision_content,
)


# --- OperationType enum tests ---


class TestOperationTypeEnum:
    """Test that IMAGE and VIDEO operation types are available."""

    def test_image_operation_type_exists(self):
        assert OperationType.IMAGE == "IMAGE"
        assert OperationType.IMAGE.value == "IMAGE"

    def test_video_operation_type_exists(self):
        assert OperationType.VIDEO == "VIDEO"
        assert OperationType.VIDEO.value == "VIDEO"

    def test_all_operation_types(self):
        assert set(OperationType) == {
            OperationType.CHAT,
            OperationType.EMBED,
            OperationType.IMAGE,
            OperationType.VIDEO,
        }


# --- Vision content detection tests ---


class TestDetectVisionContent:
    """Test detect_vision_content() for Google AI content formats."""

    def test_none_contents(self):
        assert detect_vision_content(None) is False

    def test_empty_list(self):
        assert detect_vision_content([]) is False

    def test_string_content(self):
        """Plain string content has no vision."""
        assert detect_vision_content("Hello, world!") is False

    def test_text_parts_only(self):
        """List of text-only parts has no vision."""
        parts = [{"text": "What is this?"}, {"text": "Another text"}]
        assert detect_vision_content(parts) is False

    def test_inline_data_image(self):
        """Detect inline_data with image MIME type."""
        part = {"inline_data": {"mime_type": "image/jpeg", "data": "base64data"}}
        assert detect_vision_content([part]) is True

    def test_inline_data_png(self):
        part = {"inline_data": {"mime_type": "image/png", "data": "base64data"}}
        assert detect_vision_content([part]) is True

    def test_inline_data_video(self):
        """Detect inline_data with video MIME type."""
        part = {"inline_data": {"mime_type": "video/mp4", "data": "base64data"}}
        assert detect_vision_content([part]) is True

    def test_file_data_image(self):
        """Detect file_data with image MIME type."""
        part = {
            "file_data": {
                "mime_type": "image/webp",
                "file_uri": "gs://bucket/image.webp",
            }
        }
        assert detect_vision_content([part]) is True

    def test_file_data_video(self):
        """Detect file_data with video MIME type."""
        part = {
            "file_data": {
                "mime_type": "video/quicktime",
                "file_uri": "gs://bucket/video.mov",
            }
        }
        assert detect_vision_content([part]) is True

    def test_mixed_content_with_image(self):
        """Detect image in mixed text+image content."""
        contents = [
            {"text": "What is in this image?"},
            {"inline_data": {"mime_type": "image/jpeg", "data": "base64data"}},
        ]
        assert detect_vision_content(contents) is True

    def test_content_with_parts_list(self):
        """Detect image inside a Content dict with parts."""
        content = {
            "parts": [
                {"text": "Describe this image"},
                {"inline_data": {"mime_type": "image/png", "data": "base64"}},
            ]
        }
        assert detect_vision_content([content]) is True

    def test_text_mime_type_not_vision(self):
        """Non-image/video MIME types should not be detected."""
        part = {"inline_data": {"mime_type": "text/plain", "data": "some text"}}
        assert detect_vision_content([part]) is False

    def test_audio_mime_type_not_vision(self):
        part = {"inline_data": {"mime_type": "audio/mp3", "data": "audio"}}
        assert detect_vision_content([part]) is False

    def test_genai_part_with_inline_data(self):
        """Detect vision content from google.genai Part-like objects."""
        mock_inline_data = Mock()
        mock_inline_data.mime_type = "image/jpeg"

        mock_part = Mock()
        mock_part.inline_data = mock_inline_data
        mock_part.file_data = None
        # Ensure parts is not iterable to avoid recursion
        mock_part.parts = None

        assert detect_vision_content([mock_part]) is True

    def test_genai_part_with_file_data(self):
        """Detect vision content from file_data Part objects."""
        mock_file_data = Mock()
        mock_file_data.mime_type = "video/mp4"

        mock_part = Mock()
        mock_part.inline_data = None
        mock_part.file_data = mock_file_data
        mock_part.parts = None

        assert detect_vision_content([mock_part]) is True

    def test_content_object_with_parts(self):
        """Detect vision inside a Content object with parts attribute."""
        mock_inline_data = Mock()
        mock_inline_data.mime_type = "image/png"

        mock_part = Mock()
        mock_part.inline_data = mock_inline_data
        mock_part.file_data = None
        mock_part.parts = None

        mock_content = Mock()
        mock_content.inline_data = None
        mock_content.file_data = None
        mock_content.parts = [mock_part]

        assert detect_vision_content([mock_content]) is True

    def test_single_item_not_list(self):
        """Single non-list item should be wrapped and checked."""
        part = {"inline_data": {"mime_type": "image/jpeg", "data": "base64data"}}
        assert detect_vision_content(part) is True

    def test_jpeg_bytes_magic(self):
        """Detect JPEG from raw bytes magic number."""
        jpeg_bytes = b'\xff\xd8\xff\xe0' + b'\x00' * 100
        assert detect_vision_content([jpeg_bytes]) is True

    def test_png_bytes_magic(self):
        """Detect PNG from raw bytes magic number."""
        png_bytes = b'\x89PNG' + b'\x00' * 100
        assert detect_vision_content([png_bytes]) is True

    def test_random_bytes_not_vision(self):
        """Random bytes should not be detected as vision."""
        random_bytes = b'\x00\x01\x02\x03' + b'\x00' * 100
        assert detect_vision_content([random_bytes]) is False

    def test_empty_bytes_not_vision(self):
        assert detect_vision_content([b'']) is False

    def test_type_field_image(self):
        """Detect dict with type: 'image'."""
        part = {"type": "image", "source": {"data": "base64"}}
        assert detect_vision_content([part]) is True

    def test_type_field_video(self):
        part = {"type": "video", "source": {"data": "base64"}}
        assert detect_vision_content([part]) is True

    def test_webp_bytes_magic(self):
        """Detect WebP from RIFF+WEBP magic bytes."""
        webp_bytes = b'RIFF\x00\x00\x00\x00WEBP' + b'\x00' * 100
        assert detect_vision_content([webp_bytes]) is True

    def test_riff_wav_not_vision(self):
        """RIFF container with WAV (audio) should NOT be detected as vision."""
        wav_bytes = b'RIFF\x00\x00\x00\x00WAVE' + b'\x00' * 100
        assert detect_vision_content([wav_bytes]) is False

    def test_riff_avi_not_vision(self):
        """RIFF container with AVI should NOT be detected as vision."""
        avi_bytes = b'RIFF\x00\x00\x00\x00AVI ' + b'\x00' * 100
        assert detect_vision_content([avi_bytes]) is False

    def test_recursion_depth_limit(self):
        """Deeply nested content should not cause stack overflow."""
        from revenium_middleware.google.common.trace_fields import _MAX_VISION_RECURSION_DEPTH

        # Build nested content deeper than the limit
        inner = {"inline_data": {"mime_type": "image/png", "data": "base64"}}
        for _ in range(_MAX_VISION_RECURSION_DEPTH + 5):
            inner = {"parts": [inner]}

        # Should return False (depth exceeded) instead of crashing
        assert detect_vision_content([inner]) is False


# --- Image metering tests ---


class TestImageMeteringCall:
    """Test create_image_metering_call helper."""

    @patch("revenium_middleware.google.common.utils.run_async_in_thread")
    @patch("revenium_middleware.google.common.utils.client")
    def test_create_image_metering_call(self, mock_client, mock_run_async):
        from revenium_middleware.google.common.utils import create_image_metering_call

        mock_client.api_key = None

        now = datetime.datetime.now(datetime.timezone.utc)
        create_image_metering_call(
            model="imagen-3.0-generate-001",
            requested_image_count=2,
            actual_image_count=2,
            request_time_dt=now,
            response_time_dt=now + datetime.timedelta(seconds=3),
            usage_metadata={"trace_id": "test-123"},
            operation_subtype="generation",
            aspect_ratio="16:9",
        )

        # Should have been called with the coroutine
        assert mock_run_async.called

    @patch("revenium_middleware.google.common.utils.run_async_in_thread")
    @patch("revenium_middleware.google.common.utils.client")
    def test_create_image_metering_call_with_edit_subtype(self, mock_client, mock_run_async):
        from revenium_middleware.google.common.utils import create_image_metering_call

        mock_client.api_key = None

        now = datetime.datetime.now(datetime.timezone.utc)
        create_image_metering_call(
            model="imagen-3.0-generate-001",
            requested_image_count=1,
            actual_image_count=1,
            request_time_dt=now,
            response_time_dt=now + datetime.timedelta(seconds=2),
            usage_metadata={},
            operation_subtype="edit",
        )

        assert mock_run_async.called


# --- Video metering tests ---


class TestVideoMeteringCall:
    """Test create_video_metering_call helper."""

    @patch("revenium_middleware.google.common.utils.run_async_in_thread")
    @patch("revenium_middleware.google.common.utils.client")
    def test_create_video_metering_call(self, mock_client, mock_run_async):
        from revenium_middleware.google.common.utils import create_video_metering_call

        mock_client.api_key = None

        now = datetime.datetime.now(datetime.timezone.utc)
        create_video_metering_call(
            model="veo-2.0-generate-001",
            duration_seconds=8.0,
            request_time_dt=now,
            response_time_dt=now + datetime.timedelta(seconds=30),
            usage_metadata={"trace_id": "vid-test-456"},
            operation_subtype="generation",
            resolution="1080p",
            aspect_ratio="16:9",
        )

        assert mock_run_async.called

    @patch("revenium_middleware.google.common.utils.run_async_in_thread")
    @patch("revenium_middleware.google.common.utils.client")
    def test_create_video_metering_call_with_job_id(self, mock_client, mock_run_async):
        from revenium_middleware.google.common.utils import create_video_metering_call

        mock_client.api_key = None

        now = datetime.datetime.now(datetime.timezone.utc)
        create_video_metering_call(
            model="veo-2.0-generate-001",
            duration_seconds=5.0,
            request_time_dt=now,
            response_time_dt=now + datetime.timedelta(seconds=60),
            usage_metadata={},
            video_job_id="job-abc-123",
            async_operation=True,
        )

        assert mock_run_async.called


# --- log_image_usage tests (run async via asyncio.run) ---


class TestLogImageUsage:
    """Test the async log_image_usage function."""

    @patch("revenium_middleware._core.metering_submission.client")
    @patch("revenium_middleware.google.common.utils.client")
    @patch("revenium_middleware.google.common.utils.shutdown_event")
    def test_log_image_usage_calls_create_image(self, mock_shutdown, mock_utils_client, mock_metering_client):
        from revenium_middleware.google.common.utils import log_image_usage

        mock_shutdown.is_set.return_value = False
        mock_result = Mock()
        mock_result.id = "img-result-123"
        mock_metering_client.ai.create_image.return_value = mock_result

        asyncio.run(log_image_usage(
            transaction_id="txn-img-001",
            model="imagen-3.0-generate-001",
            requested_image_count=1,
            actual_image_count=1,
            request_time="2026-02-24T10:00:00Z",
            response_time="2026-02-24T10:00:03Z",
            request_duration=3000,
            usage_metadata={"trace_id": "test-trace"},
        ))

        mock_metering_client.ai.create_image.assert_called_once()
        call_kwargs = mock_metering_client.ai.create_image.call_args[1]
        assert call_kwargs["model"] == "imagen-3.0-generate-001"
        assert call_kwargs["provider"] == "Google"
        assert call_kwargs["requested_image_count"] == 1
        assert call_kwargs["actual_image_count"] == 1
        assert call_kwargs["operation_subtype"] == "generation"
        assert call_kwargs["middleware_source"] == "python"
        assert call_kwargs["trace_id"] == "test-trace"

    @patch("revenium_middleware._core.metering_submission.client")
    @patch("revenium_middleware.google.common.utils.client")
    @patch("revenium_middleware.google.common.utils.shutdown_event")
    def test_log_image_usage_skips_during_shutdown(self, mock_shutdown, mock_utils_client, mock_metering_client):
        from revenium_middleware.google.common.utils import log_image_usage

        mock_shutdown.is_set.return_value = True

        asyncio.run(log_image_usage(
            transaction_id="txn-img-002",
            model="imagen-3.0-generate-001",
            requested_image_count=1,
            actual_image_count=1,
            request_time="2026-02-24T10:00:00Z",
            response_time="2026-02-24T10:00:03Z",
            request_duration=3000,
            usage_metadata={},
        ))

        mock_metering_client.ai.create_image.assert_not_called()

    @patch("revenium_middleware._core.metering_submission.client")
    @patch("revenium_middleware.google.common.utils.client")
    @patch("revenium_middleware.google.common.utils.shutdown_event")
    def test_log_image_usage_passes_aspect_ratio(self, mock_shutdown, mock_utils_client, mock_metering_client):
        from revenium_middleware.google.common.utils import log_image_usage

        mock_shutdown.is_set.return_value = False
        mock_result = Mock()
        mock_result.id = "img-result-ar"
        mock_metering_client.ai.create_image.return_value = mock_result

        asyncio.run(log_image_usage(
            transaction_id="txn-img-ar",
            model="imagen-3.0-generate-001",
            requested_image_count=1,
            actual_image_count=1,
            request_time="2026-02-24T10:00:00Z",
            response_time="2026-02-24T10:00:03Z",
            request_duration=3000,
            usage_metadata={},
            aspect_ratio="16:9",
        ))

        call_kwargs = mock_metering_client.ai.create_image.call_args[1]
        assert call_kwargs["aspect_ratio"] == "16:9"


# --- log_video_usage tests (run async via asyncio.run) ---


class TestLogVideoUsage:
    """Test the async log_video_usage function."""

    @patch("revenium_middleware._core.metering_submission.client")
    @patch("revenium_middleware.google.common.utils.client")
    @patch("revenium_middleware.google.common.utils.shutdown_event")
    def test_log_video_usage_calls_create_video(self, mock_shutdown, mock_utils_client, mock_metering_client):
        from revenium_middleware.google.common.utils import log_video_usage

        mock_shutdown.is_set.return_value = False
        mock_result = Mock()
        mock_result.id = "vid-result-456"
        mock_metering_client.ai.create_video.return_value = mock_result

        asyncio.run(log_video_usage(
            transaction_id="txn-vid-001",
            model="veo-2.0-generate-001",
            duration_seconds=8.0,
            request_time="2026-02-24T10:00:00Z",
            response_time="2026-02-24T10:00:30Z",
            request_duration=30000,
            usage_metadata={"trace_id": "vid-trace"},
            resolution="1080p",
            fps=24,
            video_job_id="job-xyz",
            async_operation=True,
        ))

        mock_metering_client.ai.create_video.assert_called_once()
        call_kwargs = mock_metering_client.ai.create_video.call_args[1]
        assert call_kwargs["model"] == "veo-2.0-generate-001"
        assert call_kwargs["provider"] == "Google"
        assert call_kwargs["duration_seconds"] == 8.0
        assert call_kwargs["operation_subtype"] == "generation"
        assert call_kwargs["resolution"] == "1080p"
        assert call_kwargs["fps"] == 24
        assert call_kwargs["video_job_id"] == "job-xyz"
        assert call_kwargs["async_operation"] is True
        assert call_kwargs["trace_id"] == "vid-trace"

    @patch("revenium_middleware._core.metering_submission.client")
    @patch("revenium_middleware.google.common.utils.client")
    @patch("revenium_middleware.google.common.utils.shutdown_event")
    def test_log_video_usage_skips_during_shutdown(self, mock_shutdown, mock_utils_client, mock_metering_client):
        from revenium_middleware.google.common.utils import log_video_usage

        mock_shutdown.is_set.return_value = True

        asyncio.run(log_video_usage(
            transaction_id="txn-vid-002",
            model="veo-2.0-generate-001",
            duration_seconds=5.0,
            request_time="2026-02-24T10:00:00Z",
            response_time="2026-02-24T10:01:00Z",
            request_duration=60000,
            usage_metadata={},
        ))

        mock_metering_client.ai.create_video.assert_not_called()

    @patch("revenium_middleware._core.metering_submission.client")
    @patch("revenium_middleware.google.common.utils.client")
    @patch("revenium_middleware.google.common.utils.shutdown_event")
    def test_log_video_usage_passes_aspect_ratio(self, mock_shutdown, mock_utils_client, mock_metering_client):
        from revenium_middleware.google.common.utils import log_video_usage

        mock_shutdown.is_set.return_value = False
        mock_result = Mock()
        mock_result.id = "vid-result-ar"
        mock_metering_client.ai.create_video.return_value = mock_result

        asyncio.run(log_video_usage(
            transaction_id="txn-vid-ar",
            model="veo-2.0-generate-001",
            duration_seconds=5.0,
            request_time="2026-02-24T10:00:00Z",
            response_time="2026-02-24T10:00:30Z",
            request_duration=30000,
            usage_metadata={},
            aspect_ratio="9:16",
        ))

        call_kwargs = mock_metering_client.ai.create_video.call_args[1]
        assert call_kwargs["aspect_ratio"] == "9:16"


# --- has_vision_content flag in completions tests ---


class TestHasVisionContentFlag:
    """Test that has_vision_content flag is passed through to completion args."""

    @patch("revenium_middleware._core.metering_submission.client")
    @patch("revenium_middleware.google.common.utils.client")
    @patch("revenium_middleware.google.common.utils.shutdown_event")
    def test_has_vision_content_in_completion_args(self, mock_shutdown, mock_utils_client, mock_metering_client):
        from revenium_middleware.google.common.utils import log_token_usage

        mock_shutdown.is_set.return_value = False
        mock_result = Mock()
        mock_result.id = "test-result"
        mock_metering_client.ai.create_completion.return_value = mock_result

        asyncio.run(log_token_usage(
            transaction_id="txn-vision-001",
            model="gemini-2.0-flash",
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            cached_tokens=0,
            stop_reason="END",
            request_time="2026-02-24T10:00:00Z",
            response_time="2026-02-24T10:00:02Z",
            request_duration=2000,
            usage_metadata={"has_vision_content": True},
        ))

        mock_metering_client.ai.create_completion.assert_called_once()
        call_kwargs = mock_metering_client.ai.create_completion.call_args[1]
        assert call_kwargs.get("has_vision_content") is True

    @patch("revenium_middleware._core.metering_submission.client")
    @patch("revenium_middleware.google.common.utils.client")
    @patch("revenium_middleware.google.common.utils.shutdown_event")
    def test_no_vision_content_flag_when_not_set(self, mock_shutdown, mock_utils_client, mock_metering_client):
        from revenium_middleware.google.common.utils import log_token_usage

        mock_shutdown.is_set.return_value = False
        mock_result = Mock()
        mock_result.id = "test-result"
        mock_metering_client.ai.create_completion.return_value = mock_result

        asyncio.run(log_token_usage(
            transaction_id="txn-text-001",
            model="gemini-2.0-flash",
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            cached_tokens=0,
            stop_reason="END",
            request_time="2026-02-24T10:00:00Z",
            response_time="2026-02-24T10:00:02Z",
            request_duration=2000,
            usage_metadata={},
        ))

        call_kwargs = mock_metering_client.ai.create_completion.call_args[1]
        assert "has_vision_content" not in call_kwargs


# --- Common metadata builder tests ---


class TestBuildCommonMetadataArgs:
    """Test _build_common_metadata_args helper."""

    def test_empty_metadata(self):
        from revenium_middleware.google.common.utils import _build_common_metadata_args

        result = _build_common_metadata_args({})
        # Should return empty or only env-based fields
        assert isinstance(result, dict)

    def test_with_trace_id(self):
        from revenium_middleware.google.common.utils import _build_common_metadata_args

        result = _build_common_metadata_args({"trace_id": "trace-123"})
        assert result["trace_id"] == "trace-123"

    def test_with_subscriber(self):
        from revenium_middleware.google.common.utils import _build_common_metadata_args

        result = _build_common_metadata_args({
            "subscriber": {"id": "user-1", "email": "test@example.com"}
        })
        assert result["subscriber"]["id"] == "user-1"
        assert result["subscriber"]["email"] == "test@example.com"

    def test_with_organization_name(self):
        from revenium_middleware.google.common.utils import _build_common_metadata_args

        result = _build_common_metadata_args({"organization_name": "Test Org"})
        assert result["organization_name"] == "Test Org"
