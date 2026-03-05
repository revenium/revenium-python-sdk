"""
Copyright (c) 2025 Revenium, Inc.
SPDX-License-Identifier: MIT
"""

"""
Pytest configuration and shared fixtures.

This file provides:
- Common fixtures for all tests
- Mock objects and responses
- Test configuration helpers
"""

import pytest
import os
from unittest.mock import Mock, MagicMock
from datetime import datetime, timezone


@pytest.fixture
def mock_google_ai_client():
    """Create a mock Google AI SDK client."""
    mock_client = Mock()
    mock_client.api_key = "mock-api-key-12345"
    mock_client._api_key = "mock-api-key-12345"

    # Mock models attribute
    mock_models = Mock()
    mock_client.models = mock_models

    return mock_client


@pytest.fixture
def mock_vertex_ai_client():
    """Create a mock Vertex AI SDK client."""
    mock_client = Mock()
    mock_client._vertexai = True
    mock_client._project = "mock-project-123"
    mock_client._location = "us-central1"

    return mock_client


@pytest.fixture
def mock_completion_response():
    """Create a mock completion response."""
    mock_response = Mock()
    mock_response.text = "This is a test response."

    # Mock candidates
    mock_candidate = Mock()
    mock_candidate.finish_reason = "STOP"
    mock_candidate.content = Mock()
    mock_candidate.content.parts = [Mock(text="This is a test response.")]
    mock_response.candidates = [mock_candidate]

    # Mock usage metadata
    mock_usage = Mock()
    mock_usage.prompt_token_count = 10
    mock_usage.candidates_token_count = 15
    mock_usage.total_token_count = 25
    mock_usage.cached_content_token_count = 0
    mock_response.usage_metadata = mock_usage

    return mock_response


@pytest.fixture
def mock_streaming_chunk():
    """Create a mock streaming chunk."""
    mock_chunk = Mock()
    mock_chunk.text = "chunk text"

    # Mock candidates
    mock_candidate = Mock()
    mock_candidate.finish_reason = None
    mock_candidate.content = Mock()
    mock_candidate.content.parts = [Mock(text="chunk text")]
    mock_chunk.candidates = [mock_candidate]

    # No usage metadata in intermediate chunks
    mock_chunk.usage_metadata = None

    return mock_chunk


@pytest.fixture
def mock_final_streaming_chunk():
    """Create a mock final streaming chunk with usage."""
    mock_chunk = Mock()
    mock_chunk.text = "final chunk"

    # Mock candidates with finish reason
    mock_candidate = Mock()
    mock_candidate.finish_reason = "STOP"
    mock_candidate.content = Mock()
    mock_candidate.content.parts = [Mock(text="final chunk")]
    mock_chunk.candidates = [mock_candidate]

    # Usage metadata in final chunk
    mock_usage = Mock()
    mock_usage.prompt_token_count = 50
    mock_usage.candidates_token_count = 75
    mock_usage.total_token_count = 125
    mock_usage.cached_content_token_count = 0
    mock_chunk.usage_metadata = mock_usage

    return mock_chunk


@pytest.fixture
def mock_embeddings_response():
    """Create a mock embeddings response."""
    mock_embedding = Mock()
    mock_embedding.values = [0.1, 0.2, 0.3] * 256  # 768 dimensions

    # Mock statistics (Vertex AI)
    mock_stats = Mock()
    mock_stats.token_count = 50
    mock_stats.truncated = False

    mock_response = Mock()
    mock_response.embeddings = [mock_embedding]
    mock_response.statistics = mock_stats

    return mock_response


@pytest.fixture
def sample_usage_metadata():
    """Sample usage metadata with all fields."""
    return {
        "trace_id": "test-trace-123",
        "subscriber": {
            "id": "test-user-123",
            "email": "test@example.com",
            "credential": {
                "name": "test-api-key",
                "value": "key-value-123"
            }
        },
        "organization_id": "test-org",
        "product_id": "test-product",
        "agent": "test-agent",
        "agent_version": "1.0.0",
        "task_type": "testing",
        "conversation_id": "conv-123",
        "session_id": "session-456",
        "turn_number": 1,
        "response_quality_score": 0.95
    }


@pytest.fixture
def minimal_usage_metadata():
    """Minimal usage metadata."""
    return {
        "trace_id": "test-trace-minimal",
        "subscriber": {
            "id": "test-user",
        }
    }


@pytest.fixture
def flat_usage_metadata():
    """Flat (deprecated) usage metadata format."""
    return {
        "trace_id": "test-trace-flat",
        "subscriber_id": "flat-user-123",
        "subscriber_email": "flat@example.com",
    }


@pytest.fixture
def mock_env_google_ai(monkeypatch):
    """Set up Google AI SDK environment variables."""
    monkeypatch.setenv("GOOGLE_API_KEY", "mock-google-api-key")
    monkeypatch.setenv("REVENIUM_METERING_API_KEY", "mock-revenium-key")


@pytest.fixture
def mock_env_vertex_ai(monkeypatch):
    """Set up Vertex AI environment variables."""
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "mock-project")
    monkeypatch.setenv("GOOGLE_CLOUD_LOCATION", "us-central1")
    monkeypatch.setenv("REVENIUM_METERING_API_KEY", "mock-revenium-key")


@pytest.fixture
def mock_env_clean(monkeypatch):
    """Clear all relevant environment variables."""
    env_vars = [
        "GOOGLE_API_KEY",
        "GOOGLE_CLOUD_PROJECT",
        "GOOGLE_CLOUD_LOCATION",
        "GOOGLE_APPLICATION_CREDENTIALS",
        "REVENIUM_METERING_API_KEY",
        "REVENIUM_METERING_API_URL",
        "REVENIUM_METERING_TIMEOUT",
        "GOOGLE_GENAI_USE_VERTEXAI",
    ]
    for var in env_vars:
        monkeypatch.delenv(var, raising=False)


@pytest.fixture
def freeze_time():
    """Freeze time for testing."""
    frozen_time = datetime(2025, 1, 15, 12, 0, 0, tzinfo=timezone.utc)

    class FrozenTime:
        def now(self, tz=None):
            return frozen_time

    return FrozenTime()


@pytest.fixture
def mock_revenium_api_response():
    """Mock successful Revenium API response."""
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "status": "success",
        "message": "Usage logged successfully"
    }
    mock_response.ok = True

    return mock_response


@pytest.fixture
def mock_revenium_api_error():
    """Mock Revenium API error response."""
    mock_response = Mock()
    mock_response.status_code = 500
    mock_response.json.return_value = {
        "status": "error",
        "message": "Internal server error"
    }
    mock_response.ok = False

    return mock_response


@pytest.fixture
def supported_models():
    """List of supported Google model names."""
    return [
        "gemini-2.0-flash-001",
        "gemini-1.5-pro-002",
        "gemini-1.5-pro-001",
        "gemini-1.5-flash-002",
        "gemini-1.5-flash-001",
        "gemini-pro",
        "gemini-pro-vision",
        "text-embedding-004",
        "text-embedding-005",
    ]


@pytest.fixture
def model_prefixes():
    """Model name prefixes to clean."""
    return [
        "publishers/google/models/",
        "models/",
        "google/models/",
    ]


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers",
        "unit: Unit tests (fast, no external dependencies)"
    )
    config.addinivalue_line(
        "markers",
        "e2e: End-to-end integration tests (require real API keys)"
    )
    config.addinivalue_line(
        "markers",
        "slow: Tests that take longer to run"
    )


def pytest_collection_modifyitems(config, items):
    """Automatically mark tests based on file patterns."""
    for item in items:
        # Mark E2E tests
        if "test_end_to_end" in str(item.fspath):
            if "e2e" not in item.keywords:
                item.add_marker(pytest.mark.e2e)
        # Mark other tests as unit tests
        else:
            if "unit" not in item.keywords and "e2e" not in item.keywords:
                item.add_marker(pytest.mark.unit)
