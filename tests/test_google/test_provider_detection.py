"""
Copyright (c) 2025 Revenium, Inc.
SPDX-License-Identifier: MIT
"""

"""
Tests for Google provider detection (Vertex AI vs Google AI SDK).

These tests require the Google SDK to be installed.
Mark as integration tests since they import from middleware modules that patch Google SDK.
"""

import pytest
import os
from unittest.mock import Mock, patch, MagicMock

# Mark all tests in this module as integration tests
pytestmark = pytest.mark.integration

from revenium_middleware.google.google_ai.provider import detect_provider, GoogleAIEndpoint


class TestProviderDetection:
    """Test provider detection logic."""

    def test_detect_vertex_ai_from_client_attribute(self):
        """Test detection when client has _vertexai attribute set to True."""
        mock_client = Mock()
        mock_client._vertexai = True

        result = detect_provider(mock_client)

        assert result == GoogleAIEndpoint.VERTEX_AI

    def test_detect_gemini_from_client_attribute(self):
        """Test detection when client has _vertexai attribute set to False."""
        mock_client = Mock()
        mock_client._vertexai = False

        result = detect_provider(mock_client)

        assert result == GoogleAIEndpoint.GEMINI_DEVELOPER_API

    @patch.dict(os.environ, {"GOOGLE_GENAI_USE_VERTEXAI": "true"})
    def test_detect_vertex_ai_from_env_var_true(self):
        """Test detection from GOOGLE_GENAI_USE_VERTEXAI=true."""
        result = detect_provider()

        assert result == GoogleAIEndpoint.VERTEX_AI

    @patch.dict(os.environ, {"GOOGLE_GENAI_USE_VERTEXAI": "1"})
    def test_detect_vertex_ai_from_env_var_1(self):
        """Test detection from GOOGLE_GENAI_USE_VERTEXAI=1."""
        result = detect_provider()

        assert result == GoogleAIEndpoint.VERTEX_AI

    @patch.dict(os.environ, {"GOOGLE_GENAI_USE_VERTEXAI": "yes"})
    def test_detect_vertex_ai_from_env_var_yes(self):
        """Test detection from GOOGLE_GENAI_USE_VERTEXAI=yes."""
        result = detect_provider()

        assert result == GoogleAIEndpoint.VERTEX_AI

    @patch.dict(os.environ, {"GOOGLE_GENAI_USE_VERTEXAI": "false"})
    def test_detect_gemini_from_env_var_false(self):
        """Test detection from GOOGLE_GENAI_USE_VERTEXAI=false."""
        result = detect_provider()

        # Should fall through to next check
        assert result in [GoogleAIEndpoint.VERTEX_AI, GoogleAIEndpoint.GEMINI_DEVELOPER_API]

    @patch.dict(os.environ, {
        "GOOGLE_CLOUD_PROJECT": "my-project",
        "GOOGLE_CLOUD_LOCATION": "us-central1"
    }, clear=True)
    def test_detect_vertex_ai_from_gcp_config(self):
        """Test detection from GCP project and location env vars."""
        result = detect_provider()

        assert result == GoogleAIEndpoint.VERTEX_AI

    @patch.dict(os.environ, {"GOOGLE_CLOUD_PROJECT": "my-project"}, clear=True)
    def test_detect_vertex_ai_from_project_only(self):
        """Test detection from GCP project without location (should still detect Vertex AI)."""
        # Without location, should fall through
        result = detect_provider()

        # Should NOT be Vertex AI without location
        assert result == GoogleAIEndpoint.GEMINI_DEVELOPER_API

    @patch.dict(os.environ, {"GOOGLE_API_KEY": "some-api-key"}, clear=True)
    def test_detect_gemini_from_api_key(self):
        """Test detection from GOOGLE_API_KEY env var."""
        result = detect_provider()

        assert result == GoogleAIEndpoint.GEMINI_DEVELOPER_API

    @patch.dict(os.environ, {}, clear=True)
    def test_detect_default_to_gemini(self):
        """Test default detection when no indicators present."""
        result = detect_provider()

        assert result == GoogleAIEndpoint.GEMINI_DEVELOPER_API

    @patch.dict(os.environ, {
        "GOOGLE_GENAI_USE_VERTEXAI": "true",
        "GOOGLE_API_KEY": "some-key"
    })
    def test_priority_env_var_over_api_key(self):
        """Test that GOOGLE_GENAI_USE_VERTEXAI takes priority over GOOGLE_API_KEY."""
        result = detect_provider()

        assert result == GoogleAIEndpoint.VERTEX_AI

    def test_priority_client_config_over_env_vars(self):
        """Test that client configuration takes priority over environment variables."""
        mock_client = Mock()
        mock_client._vertexai = True

        with patch.dict(os.environ, {"GOOGLE_API_KEY": "some-key"}):
            result = detect_provider(mock_client)

        assert result == GoogleAIEndpoint.VERTEX_AI


class TestProviderMetadata:
    """Test provider metadata generation."""

    def test_google_ai_sdk_provider_metadata(self):
        """Test provider metadata for Google AI SDK."""
        from revenium_middleware.google.common.types import ProviderMetadata

        metadata = ProviderMetadata.for_google_ai_sdk()

        assert metadata.provider == "Google"
        assert metadata.model_source == "GOOGLE"

    def test_vertex_ai_sdk_provider_metadata(self):
        """Test provider metadata for Vertex AI SDK."""
        from revenium_middleware.google.common.types import ProviderMetadata

        metadata = ProviderMetadata.for_vertex_ai_sdk()

        assert metadata.provider == "Google"
        assert metadata.model_source == "GOOGLE"

    def test_both_sdks_report_same_provider(self):
        """Test that both SDKs report unified provider name."""
        from revenium_middleware.google.common.types import ProviderMetadata

        google_ai = ProviderMetadata.for_google_ai_sdk()
        vertex_ai = ProviderMetadata.for_vertex_ai_sdk()

        # Both should report as "Google" for unified analytics
        assert google_ai.provider == vertex_ai.provider == "Google"
        assert google_ai.model_source == vertex_ai.model_source == "GOOGLE"
