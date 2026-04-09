"""
Shared test fixtures for fal.ai middleware tests.
"""

import pytest
import os


@pytest.fixture
def mock_fal_image_response():
    """Mock fal.ai image generation response."""
    return {
        "images": [
            {
                "url": "https://fal.media/files/test/image1.png",
                "width": 1024,
                "height": 1024,
                "content_type": "image/png",
            }
        ],
        "timings": {"inference": 2.5},
        "seed": 12345,
        "prompt": "A beautiful sunset over mountains",
    }


@pytest.fixture
def mock_fal_multi_image_response():
    """Mock fal.ai response with multiple images."""
    return {
        "images": [
            {"url": "https://fal.media/files/test/image1.png", "width": 1024, "height": 1024},
            {"url": "https://fal.media/files/test/image2.png", "width": 1024, "height": 1024},
            {"url": "https://fal.media/files/test/image3.png", "width": 1024, "height": 1024},
        ],
        "seed": 12345,
    }


@pytest.fixture
def mock_fal_video_response():
    """Mock fal.ai video generation response."""
    return {
        "video": {
            "url": "https://fal.media/files/test/video.mp4",
            "content_type": "video/mp4",
            "duration": 5.0,
        },
        "timings": {"inference_time": 30.0},
    }


@pytest.fixture
def mock_fal_audio_response():
    """Mock fal.ai audio generation response (TTS)."""
    return {
        "audio": {
            "url": "https://fal.media/files/test/audio.mp3",
            "content_type": "audio/mpeg",
            "duration": 8.5,
        },
        "timings": {"inference_time": 2.0},
    }


@pytest.fixture
def mock_fal_transcription_response():
    """Mock fal.ai audio transcription response (STT)."""
    return {
        "text": "Hello world, this is a transcription test.",
        "audio_length": 12.3,
        "timings": {"inference_time": 1.5},
    }


@pytest.fixture
def env_with_api_key():
    """Set up environment with Revenium API key."""
    original = os.environ.get("REVENIUM_METERING_API_KEY")
    os.environ["REVENIUM_METERING_API_KEY"] = "test-api-key"
    yield
    if original:
        os.environ["REVENIUM_METERING_API_KEY"] = original
    else:
        os.environ.pop("REVENIUM_METERING_API_KEY", None)
