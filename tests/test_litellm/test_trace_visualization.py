"""
Tests for trace visualization field capture and validation.
"""

import os
import pytest
from revenium_middleware.litellm.client.trace_fields import (
    get_environment,
    get_region,
    get_credential_alias,
    get_trace_type,
    get_trace_name,
    get_parent_transaction_id,
    get_transaction_name,
    get_retry_number,
    validate_trace_type,
    validate_trace_name,
    detect_operation_type
)


class TestTraceFieldCapture:
    """Test environment variable capture for trace fields."""

    def test_get_environment(self, monkeypatch):
        """Test environment field capture with priority order."""
        # Test REVENIUM_ENVIRONMENT (highest priority)
        monkeypatch.setenv("REVENIUM_ENVIRONMENT", "production")
        monkeypatch.setenv("ENVIRONMENT", "staging")
        assert get_environment() == "production"

        # Test ENVIRONMENT fallback
        monkeypatch.delenv("REVENIUM_ENVIRONMENT")
        assert get_environment() == "staging"

        # Test DEPLOYMENT_ENV fallback
        monkeypatch.delenv("ENVIRONMENT")
        monkeypatch.setenv("DEPLOYMENT_ENV", "dev")
        assert get_environment() == "dev"

    def test_get_region(self, monkeypatch):
        """Test region field capture with cloud provider fallbacks."""
        # Test REVENIUM_REGION (highest priority)
        monkeypatch.setenv("REVENIUM_REGION", "custom-region")
        monkeypatch.setenv("AWS_REGION", "us-east-1")
        assert get_region() == "custom-region"

        # Test AWS_REGION fallback
        monkeypatch.delenv("REVENIUM_REGION")
        assert get_region() == "us-east-1"

        # Test AZURE_REGION fallback
        monkeypatch.delenv("AWS_REGION")
        monkeypatch.setenv("AZURE_REGION", "eastus")
        assert get_region() == "eastus"

        # Test GCP_REGION fallback
        monkeypatch.delenv("AZURE_REGION")
        monkeypatch.setenv("GCP_REGION", "us-central1")
        assert get_region() == "us-central1"

        # Test GOOGLE_CLOUD_REGION fallback
        monkeypatch.delenv("GCP_REGION")
        monkeypatch.setenv("GOOGLE_CLOUD_REGION", "europe-west1")
        assert get_region() == "europe-west1"

    def test_get_credential_alias(self, monkeypatch):
        """Test credential alias capture."""
        monkeypatch.setenv("REVENIUM_CREDENTIAL_ALIAS", "prod-openai-key")
        assert get_credential_alias() == "prod-openai-key"

    def test_get_trace_type(self, monkeypatch):
        """Test trace type capture and validation."""
        monkeypatch.setenv("REVENIUM_TRACE_TYPE", "api-request")
        assert get_trace_type() == "api-request"

    def test_get_trace_name(self, monkeypatch):
        """Test trace name capture and validation."""
        monkeypatch.setenv("REVENIUM_TRACE_NAME", "User Authentication Flow")
        assert get_trace_name() == "User Authentication Flow"

    def test_get_parent_transaction_id(self, monkeypatch):
        """Test parent transaction ID capture."""
        monkeypatch.setenv("REVENIUM_PARENT_TRANSACTION_ID", "parent-123")
        assert get_parent_transaction_id() == "parent-123"

    def test_get_transaction_name(self, monkeypatch):
        """Test transaction name with fallback to task_type."""
        # Test env var (highest priority)
        monkeypatch.setenv("REVENIUM_TRANSACTION_NAME", "env-transaction")
        metadata = {"transactionName": "metadata-transaction", "task_type": "task"}
        assert get_transaction_name(metadata) == "env-transaction"

        # Test metadata transactionName
        monkeypatch.delenv("REVENIUM_TRANSACTION_NAME")
        assert get_transaction_name(metadata) == "metadata-transaction"

        # Test fallback to task_type
        metadata_no_name = {"task_type": "classification"}
        assert get_transaction_name(metadata_no_name) == "classification"

    def test_get_retry_number(self, monkeypatch):
        """Test retry number capture and parsing."""
        # Test valid retry number
        monkeypatch.setenv("REVENIUM_RETRY_NUMBER", "3")
        assert get_retry_number() == 3

        # Test invalid retry number (defaults to 0)
        monkeypatch.setenv("REVENIUM_RETRY_NUMBER", "invalid")
        assert get_retry_number() == 0

        # Test no retry number (defaults to 0)
        monkeypatch.delenv("REVENIUM_RETRY_NUMBER")
        assert get_retry_number() == 0


class TestTraceValidation:
    """Test validation logic for trace fields."""

    def test_validate_trace_type_valid(self):
        """Test valid trace type patterns."""
        assert validate_trace_type("customer-support") == "customer-support"
        assert validate_trace_type("api_request") == "api_request"
        assert validate_trace_type("data-analysis-2024") == "data-analysis-2024"
        assert validate_trace_type("ABC123") == "ABC123"

    def test_validate_trace_type_invalid_chars(self):
        """Test trace type with invalid characters."""
        assert validate_trace_type("invalid type!") is None
        assert validate_trace_type("type@domain") is None
        assert validate_trace_type("type with spaces") is None

    def test_validate_trace_type_too_long(self):
        """Test trace type exceeding max length."""
        long_type = "a" * 129  # 129 chars, max is 128
        assert validate_trace_type(long_type) is None

    def test_validate_trace_type_empty(self):
        """Test empty trace type."""
        assert validate_trace_type("") is None
        assert validate_trace_type(None) is None

    def test_validate_trace_name_valid(self):
        """Test valid trace name."""
        assert validate_trace_name("Customer Support Chat") == "Customer Support Chat"
        assert validate_trace_name("A" * 256) == "A" * 256  # Exactly 256 chars

    def test_validate_trace_name_truncates(self):
        """Test trace name truncation."""
        long_name = "A" * 300  # 300 chars, max is 256
        result = validate_trace_name(long_name)
        assert result == "A" * 256
        assert len(result) == 256

    def test_validate_trace_name_empty(self):
        """Test empty trace name."""
        assert validate_trace_name("") is None
        assert validate_trace_name(None) is None


class TestOperationTypeDetection:
    """Test operation type detection from method names."""

    def test_detect_chat_operation(self):
        """Test chat operation detection."""
        assert detect_operation_type("completion") == "CHAT"
        assert detect_operation_type("chat_completion") == "CHAT"
        assert detect_operation_type("acompletion") == "CHAT"

    def test_detect_embed_operation(self):
        """Test embedding operation detection."""
        assert detect_operation_type("embedding") == "EMBED"
        assert detect_operation_type("aembedding") == "EMBED"

    def test_detect_tool_call_operation(self):
        """Test tool call operation detection."""
        request_body = {"tools": [{"type": "function", "function": {"name": "get_weather"}}]}
        assert detect_operation_type("completion", request_body) == "TOOL_CALL"

    def test_detect_unknown_operation(self):
        """Test unknown operation type."""
        assert detect_operation_type("unknown_method") == "CHAT"  # Default fallback


class TestTraceFieldIntegration:
    """Test integration of trace fields with metadata."""

    def test_environment_precedence(self, monkeypatch):
        """Test that environment variables are properly captured."""
        monkeypatch.setenv("REVENIUM_ENVIRONMENT", "production")
        assert get_environment() == "production"

    def test_region_auto_detection_aws(self, monkeypatch):
        """Test AWS region auto-detection."""
        monkeypatch.setenv("AWS_REGION", "us-west-2")
        assert get_region() == "us-west-2"

    def test_region_auto_detection_azure(self, monkeypatch):
        """Test Azure region auto-detection."""
        monkeypatch.setenv("AZURE_REGION", "westus2")
        assert get_region() == "westus2"

    def test_region_auto_detection_gcp(self, monkeypatch):
        """Test GCP region auto-detection."""
        monkeypatch.setenv("GCP_REGION", "us-central1")
        assert get_region() == "us-central1"

    def test_transaction_name_fallback_chain(self, monkeypatch):
        """Test transaction name fallback chain."""
        # No env var, no metadata
        assert get_transaction_name({}) is None

        # With task_type fallback
        assert get_transaction_name({"task_type": "chat"}) == "chat"

        # With transactionName in metadata
        assert get_transaction_name({"transactionName": "Generate Response"}) == "Generate Response"

        # With transaction_name in metadata (snake_case)
        assert get_transaction_name({"transaction_name": "Analyze Sentiment"}) == "Analyze Sentiment"

        # Env var takes precedence
        monkeypatch.setenv("REVENIUM_TRANSACTION_NAME", "env-transaction")
        assert get_transaction_name({"transactionName": "metadata-transaction"}) == "env-transaction"

    def test_retry_number_parsing(self, monkeypatch):
        """Test retry number parsing from environment."""
        # Valid number
        monkeypatch.setenv("REVENIUM_RETRY_NUMBER", "5")
        assert get_retry_number() == 5

        # Zero
        monkeypatch.setenv("REVENIUM_RETRY_NUMBER", "0")
        assert get_retry_number() == 0

        # Invalid (defaults to 0)
        monkeypatch.setenv("REVENIUM_RETRY_NUMBER", "not-a-number")
        assert get_retry_number() == 0

    def test_all_fields_none_when_not_set(self):
        """Test that all fields return None when not set."""
        # Clear all environment variables
        for key in list(os.environ.keys()):
            if key.startswith("REVENIUM_"):
                del os.environ[key]

        assert get_environment() is None
        assert get_region() is None
        assert get_credential_alias() is None
        assert get_trace_type() is None
        assert get_trace_name() is None
        assert get_parent_transaction_id() is None
        assert get_transaction_name({}) is None
        assert get_retry_number() == 0  # Defaults to 0, not None

