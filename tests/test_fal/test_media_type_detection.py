"""
Tests for fal.ai media type detection.

Verifies that different fal.ai model names are correctly classified
into image, video, audio, or generation types.
"""

import pytest
from revenium_middleware.fal.trace_fields import detect_media_type


class TestImageDetection:
    """Tests for image model detection."""

    def test_flux_models(self):
        assert detect_media_type("fal-ai/flux/dev") == "image"
        assert detect_media_type("fal-ai/flux/schnell") == "image"
        assert detect_media_type("fal-ai/flux-pro") == "image"
        assert detect_media_type("fal-ai/flux/dev/image-to-image") == "image"

    def test_stable_diffusion_models(self):
        assert detect_media_type("fal-ai/stable-diffusion-xl") == "image"
        assert detect_media_type("fal-ai/sdxl") == "image"

    def test_controlnet(self):
        assert detect_media_type("fal-ai/controlnet") == "image"

    def test_upscale(self):
        assert detect_media_type("fal-ai/image-upscaler") == "image"

    def test_remove_background(self):
        assert detect_media_type("fal-ai/remove-background") == "image"

    def test_inpainting(self):
        assert detect_media_type("fal-ai/inpaint-model") == "image"

    def test_unknown_defaults_to_image(self):
        """Most fal.ai models are image generators, so unknown defaults to image."""
        assert detect_media_type("fal-ai/some-new-model") == "image"


class TestVideoDetection:
    """Tests for video model detection."""

    def test_text_to_video(self):
        assert detect_media_type("fal-ai/text-to-video") == "video"

    def test_animate_diff(self):
        assert detect_media_type("fal-ai/animate-diff") == "video"

    def test_cogvideo(self):
        assert detect_media_type("fal-ai/cogvideo") == "video"

    def test_stable_video(self):
        assert detect_media_type("fal-ai/stable-video-diffusion") == "video"

    def test_kling(self):
        assert detect_media_type("fal-ai/kling/video-gen") == "video"

    def test_luma(self):
        assert detect_media_type("fal-ai/luma/dream-machine") == "video"

    def test_runway(self):
        assert detect_media_type("fal-ai/runway/gen-3") == "video"


class TestAudioDetection:
    """Tests for audio model detection."""

    def test_whisper(self):
        assert detect_media_type("fal-ai/whisper") == "audio"

    def test_tts(self):
        assert detect_media_type("fal-ai/tts") == "audio"

    def test_text_to_speech(self):
        assert detect_media_type("fal-ai/text-to-speech") == "audio"

    def test_speech_to_text(self):
        assert detect_media_type("fal-ai/speech-to-text") == "audio"

    def test_bark(self):
        assert detect_media_type("fal-ai/bark") == "audio"

    def test_music_generation(self):
        assert detect_media_type("fal-ai/music-gen") == "audio"


class TestEdgeCases:
    """Edge case tests."""

    def test_empty_string(self):
        assert detect_media_type("") == "generation"

    def test_none(self):
        assert detect_media_type(None) == "generation"

    def test_case_insensitive(self):
        assert detect_media_type("fal-ai/FLUX/DEV") == "image"
        assert detect_media_type("fal-ai/TEXT-TO-VIDEO") == "video"
        assert detect_media_type("fal-ai/WHISPER") == "audio"
