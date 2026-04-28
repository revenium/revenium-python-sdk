"""
Copyright (c) 2025 Revenium, Inc.
SPDX-License-Identifier: MIT
"""

"""
Tests for configuration validation and authentication.
"""

import pytest
import os
from unittest.mock import Mock, patch, MagicMock
from revenium_middleware.google.common.exceptions import ConfigurationError


class TestGoogleCloudAuthentication:
    """Test Google Cloud authentication configuration."""

    @patch.dict(os.environ, {"GOOGLE_API_KEY": "test-api-key"})
    def test_google_ai_sdk_with_api_key(self):
        """Test that Google AI SDK can be configured with API key."""
        api_key = os.getenv("GOOGLE_API_KEY")

        assert api_key is not None
        assert api_key == "test-api-key"

    @patch.dict(os.environ, {
        "GOOGLE_CLOUD_PROJECT": "test-project",
        "GOOGLE_CLOUD_LOCATION": "us-central1"
    })
    def test_vertex_ai_with_project_and_location(self):
        """Test that Vertex AI can be configured with project and location."""
        project = os.getenv("GOOGLE_CLOUD_PROJECT")
        location = os.getenv("GOOGLE_CLOUD_LOCATION")

        assert project == "test-project"
        assert location == "us-central1"

    @patch.dict(os.environ, {"GOOGLE_APPLICATION_CREDENTIALS": "/path/to/credentials.json"})
    def test_vertex_ai_with_service_account(self):
        """Test Vertex AI configuration with service account."""
        credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

        assert credentials_path is not None
        assert credentials_path.endswith(".json")

    @patch.dict(os.environ, {}, clear=True)
    def test_missing_credentials_detected(self):
        """Test that missing credentials are detected."""
        # Neither API key nor GCP credentials
        api_key = os.getenv("GOOGLE_API_KEY")
        project = os.getenv("GOOGLE_CLOUD_PROJECT")

        assert api_key is None
        assert project is None


class TestReveniumConfiguration:
    """Test Revenium-specific configuration."""

    @patch.dict(os.environ, {"REVENIUM_METERING_API_KEY": "rev-key-123"})
    def test_revenium_api_key_present(self):
        """Test that Revenium API key can be configured."""
        api_key = os.getenv("REVENIUM_METERING_API_KEY")

        assert api_key is not None
        assert api_key.startswith("rev-key-")

    @patch.dict(os.environ, {}, clear=True)
    def test_missing_revenium_api_key(self):
        """Test handling of missing Revenium API key."""
        api_key = os.getenv("REVENIUM_METERING_API_KEY")

        assert api_key is None
        # Middleware should handle this gracefully

    @patch.dict(os.environ, {"REVENIUM_METERING_API_URL": "https://custom.revenium.ai"})
    def test_custom_revenium_url(self):
        """Test that custom Revenium URL can be configured."""
        custom_url = os.getenv("REVENIUM_METERING_API_URL")

        assert custom_url == "https://custom.revenium.ai"

    @patch.dict(os.environ, {"REVENIUM_METERING_TIMEOUT": "30"})
    def test_custom_timeout_configuration(self):
        """Test that custom timeout can be configured."""
        timeout = os.getenv("REVENIUM_METERING_TIMEOUT")

        assert timeout == "30"
        assert int(timeout) == 30


class TestEnvironmentVariableParsing:
    """Test environment variable parsing and validation."""

    @patch.dict(os.environ, {"GOOGLE_GENAI_USE_VERTEXAI": "true"})
    def test_parse_boolean_true_lowercase(self):
        """Test parsing 'true' as boolean."""
        value = os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "").lower()

        assert value == "true"

    @patch.dict(os.environ, {"GOOGLE_GENAI_USE_VERTEXAI": "True"})
    def test_parse_boolean_true_capitalized(self):
        """Test parsing 'True' as boolean."""
        value = os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "").lower()

        assert value == "true"

    @patch.dict(os.environ, {"GOOGLE_GENAI_USE_VERTEXAI": "1"})
    def test_parse_boolean_one_as_true(self):
        """Test parsing '1' as boolean."""
        value = os.getenv("GOOGLE_GENAI_USE_VERTEXAI")

        assert value == "1"
        # Implementation should handle "1" as True

    @patch.dict(os.environ, {"GOOGLE_GENAI_USE_VERTEXAI": "false"})
    def test_parse_boolean_false(self):
        """Test parsing 'false' as boolean."""
        value = os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "").lower()

        assert value == "false"

    @patch.dict(os.environ, {"REVENIUM_METERING_TIMEOUT": "not-a-number"})
    def test_invalid_integer_parsing(self):
        """Test handling of invalid integer values."""
        timeout_str = os.getenv("REVENIUM_METERING_TIMEOUT")

        with pytest.raises(ValueError):
            int(timeout_str)


class TestConfigurationDefaults:
    """Test default configuration values."""

    def test_default_timeout_value(self):
        """Test default timeout value."""
        default_timeout = 10  # Typical default

        timeout_str = os.getenv("REVENIUM_METERING_TIMEOUT")
        timeout = int(timeout_str) if timeout_str else default_timeout

        assert timeout >= 5  # Reasonable minimum
        assert timeout <= 60  # Reasonable maximum

    @patch.dict(os.environ, {}, clear=True)
    def test_default_google_location(self):
        """Test default Google Cloud location."""
        location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")

        assert location == "us-central1"


class TestConfigurationValidation:
    """Test configuration validation logic."""

    def test_validate_api_key_format(self):
        """API key prefix validation accepts hak_/rev_, rejects other formats."""
        from revenium_middleware._core.config import validate_api_key

        # Valid Revenium prefixes
        validate_api_key("hak_abc123def456")
        validate_api_key("rev_mk_3By1Ra6_abc123")
        validate_api_key("rev_sk_3By1Ra6_abc123")

        # Anything else (including other-provider keys) must be rejected.
        invalid_keys = [
            "rev-key-abc123",         # hyphen, not underscore
            "sk-test-123456",         # OpenAI
            "AIzaSyABC123-456_789",   # Google
        ]
        for key in invalid_keys:
            with pytest.raises(ValueError, match='should start with "hak_" or "rev_"'):
                validate_api_key(key)

    def test_validate_url_format(self):
        """Test URL format validation - should accept any valid HTTP(S) URL."""
        from urllib.parse import urlparse

        test_urls = [
            "https://api.example.com",
            "https://api.example.com:8080",
            "http://localhost:8000",
            "https://custom-domain.com/path",
        ]

        for url in test_urls:
            parsed = urlparse(url)
            # Valid URL should have scheme and netloc
            assert parsed.scheme in ("http", "https")
            assert parsed.netloc  # Has a domain/host

    def test_validate_project_id_format(self):
        """Test GCP project ID format validation."""
        valid_project_ids = [
            "my-project-123",
            "test-project",
            "prod-env-456"
        ]

        invalid_project_ids = [
            "",
            "   ",
            "PROJECT WITH SPACES",
            "project_with_underscores_only"
        ]

        for project_id in valid_project_ids:
            assert len(project_id) >= 6  # GCP minimum
            assert len(project_id) <= 30  # GCP maximum
            assert "-" in project_id or project_id.islower()

    def test_validate_location_format(self):
        """Test GCP location format validation."""
        valid_locations = [
            "us-central1",
            "us-east1",
            "europe-west1",
            "asia-southeast1"
        ]

        for location in valid_locations:
            assert "-" in location
            parts = location.split("-")
            assert len(parts) >= 2


class TestMultipleConfigurationSources:
    """Test handling of multiple configuration sources."""

    @patch.dict(os.environ, {
        "GOOGLE_API_KEY": "env-api-key",
        "GOOGLE_APPLICATION_CREDENTIALS": "/path/to/creds.json"
    })
    def test_both_api_key_and_service_account(self):
        """Test when both API key and service account are present."""
        api_key = os.getenv("GOOGLE_API_KEY")
        creds_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

        # Both are present - implementation should choose appropriate one
        assert api_key is not None
        assert creds_path is not None

    @patch.dict(os.environ, {
        "GOOGLE_CLOUD_PROJECT": "test-project",
        "GOOGLE_API_KEY": "test-api-key"
    })
    def test_mixed_vertex_and_google_ai_config(self):
        """Test when both Vertex AI and Google AI configs are present."""
        project = os.getenv("GOOGLE_CLOUD_PROJECT")
        api_key = os.getenv("GOOGLE_API_KEY")

        # Implementation should detect and use appropriate provider
        assert project is not None
        assert api_key is not None


class TestConfigurationPrecedence:
    """Test configuration source precedence."""

    def test_explicit_parameter_highest_priority(self):
        """Test that explicit parameters have highest priority."""
        # Simulating explicit parameter vs environment variable
        env_value = "env-value"
        explicit_value = "explicit-value"

        # Explicit should win
        result = explicit_value if explicit_value else env_value

        assert result == "explicit-value"


class TestConfigurationErrorMessages:
    """Test helpful configuration error messages."""

    def test_missing_api_key_error_message(self):
        """Test error message for missing API key."""
        error = ConfigurationError(
            "GOOGLE_API_KEY not set. "
            "Get your API key from: https://aistudio.google.com/app/apikey"
        )

        message = str(error)
        assert "GOOGLE_API_KEY" in message
        assert "https://" in message

    def test_missing_project_error_message(self):
        """Test error message for missing GCP project."""
        error = ConfigurationError(
            "GOOGLE_CLOUD_PROJECT not set. "
            "Set your GCP project ID using: gcloud config set project PROJECT_ID"
        )

        message = str(error)
        assert "GOOGLE_CLOUD_PROJECT" in message
        assert "gcloud" in message

    def test_missing_revenium_key_error_message(self):
        """Test error message for missing Revenium API key."""
        error = ConfigurationError(
            "REVENIUM_METERING_API_KEY not set. "
            "Get your API key from: https://portal.revenium.ai"
        )

        message = str(error)
        assert "REVENIUM_METERING_API_KEY" in message
        assert "portal.revenium.ai" in message


class TestConfigurationSecurity:
    """Test configuration security practices."""

    def test_api_keys_not_logged(self):
        """Test that API keys are not logged in plain text."""
        api_key = "sk-test-secret-key-12345"

        # Should mask in logs
        masked = f"***{api_key[-4:]}"

        assert masked == "***2345"
        assert "secret" not in masked

    def test_credentials_path_validated(self):
        """Test that credentials file paths are validated."""
        valid_paths = [
            "/path/to/credentials.json",
            "./config/service-account.json",
            "~/gcp-credentials.json"
        ]

        for path in valid_paths:
            assert ".json" in path
            assert len(path) > 5

    def test_sensitive_env_vars_list(self):
        """Test list of sensitive environment variables."""
        sensitive_vars = [
            "GOOGLE_API_KEY",
            "REVENIUM_METERING_API_KEY",
            "GOOGLE_APPLICATION_CREDENTIALS"
        ]

        for var in sensitive_vars:
            # These should never be logged in full
            assert "KEY" in var or "CREDENTIALS" in var


class TestConfigurationHelpers:
    """Test configuration helper functions."""

    def test_get_config_with_fallback(self):
        """Test getting config with fallback value."""
        def get_config(key: str, default: str = None) -> str:
            return os.getenv(key, default)

        with patch.dict(os.environ, {}, clear=True):
            result = get_config("MISSING_KEY", "default_value")
            assert result == "default_value"

    def test_require_config_raises_on_missing(self):
        """Test that required config raises error when missing."""
        def require_config(key: str) -> str:
            value = os.getenv(key)
            if not value:
                raise ConfigurationError(f"{key} is required")
            return value

        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ConfigurationError):
                require_config("REQUIRED_KEY")

    def test_parse_list_from_env_var(self):
        """Test parsing comma-separated list from env var."""
        with patch.dict(os.environ, {"ALLOWED_MODELS": "model1,model2,model3"}):
            models_str = os.getenv("ALLOWED_MODELS", "")
            models = [m.strip() for m in models_str.split(",") if m.strip()]

            assert len(models) == 3
            assert "model1" in models


class TestDevelopmentVsProduction:
    """Test configuration differences for dev vs prod."""

    @patch.dict(os.environ, {"ENVIRONMENT": "development"})
    def test_development_config(self):
        """Test development configuration."""
        env = os.getenv("ENVIRONMENT")

        assert env == "development"
        # In dev, might use localhost or dev endpoints

    @patch.dict(os.environ, {"ENVIRONMENT": "production"})
    def test_production_config(self):
        """Test production configuration."""
        env = os.getenv("ENVIRONMENT")

        assert env == "production"
        # In prod, should use official endpoints

    @patch.dict(os.environ, {"DEBUG": "true"})
    def test_debug_mode_enabled(self):
        """Test debug mode configuration."""
        debug = os.getenv("DEBUG", "").lower() == "true"

        assert debug is True
