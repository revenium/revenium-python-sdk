"""
Tests for trace visualization field capture and validation.
"""

import os
import pytest
from revenium_middleware.google.common.trace_fields import (
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
        monkeypatch.setenv("REVENIUM_CREDENTIAL_ALIAS", "prod-google-key")
        assert get_credential_alias() == "prod-google-key"

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
        monkeypatch.setenv("REVENIUM_RETRY_NUMBER", "3")
        assert get_retry_number() == 3

        # Test default value
        monkeypatch.delenv("REVENIUM_RETRY_NUMBER", raising=False)
        assert get_retry_number() == 0

    def test_get_retry_number_invalid(self, monkeypatch):
        """Test retry number with invalid value."""
        monkeypatch.setenv("REVENIUM_RETRY_NUMBER", "invalid")
        assert get_retry_number() == 0

    def test_fields_without_env_vars(self):
        """Test that fields return None when env vars not set."""
        # Clear all relevant env vars
        for key in os.environ.copy():
            if key.startswith("REVENIUM_") or key in ["ENVIRONMENT", "DEPLOYMENT_ENV", "AWS_REGION", "GCP_REGION"]:
                os.environ.pop(key, None)

        assert get_environment() is None
        assert get_region() is None
        assert get_credential_alias() is None
        assert get_trace_type() is None
        assert get_trace_name() is None
        assert get_parent_transaction_id() is None
        assert get_transaction_name() is None


class TestTraceValidation:
    """Test validation logic for trace fields."""

    def test_validate_trace_type_valid(self):
        """Test valid trace type formats."""
        assert validate_trace_type("api-request") == "api-request"
        assert validate_trace_type("user_action") == "user_action"
        assert validate_trace_type("API123") == "API123"

    def test_validate_trace_type_invalid_characters(self):
        """Test trace type with invalid characters."""
        assert validate_trace_type("api request") is None  # space
        assert validate_trace_type("api@request") is None  # special char

    def test_validate_trace_type_too_long(self):
        """Test trace type exceeding max length."""
        long_type = "a" * 129
        assert validate_trace_type(long_type) is None

    def test_validate_trace_type_max_length(self):
        """Test trace type at exactly max length."""
        max_type = "a" * 128
        assert validate_trace_type(max_type) == max_type

    def test_validate_trace_name_valid(self):
        """Test valid trace name."""
        assert validate_trace_name("User Login Flow") == "User Login Flow"
        assert validate_trace_name("API Request") == "API Request"

    def test_validate_trace_name_truncation(self):
        """Test trace name truncation at max length."""
        long_name = "a" * 300
        result = validate_trace_name(long_name)
        assert result == "a" * 256
        assert len(result) == 256

    def test_validate_trace_name_max_length(self):
        """Test trace name at exactly max length."""
        max_name = "a" * 256
        assert validate_trace_name(max_name) == max_name


class TestOperationTypeDetection:
    """Test operation type detection for Google AI methods."""

    def test_detect_generate_content(self):
        """Test detection of generate_content operations."""
        assert detect_operation_type("generate_content") == "CHAT"
        assert detect_operation_type("generate_content_async") == "CHAT"

    def test_detect_streaming(self):
        """Test detection of streaming operations."""
        assert detect_operation_type("generate_content_stream") == "CHAT"
        assert detect_operation_type("stream_generate_content") == "CHAT"

    def test_detect_embeddings(self):
        """Test detection of embedding operations."""
        assert detect_operation_type("embed_content") == "EMBED"
        assert detect_operation_type("embed_content_async") == "EMBED"

    def test_detect_count_tokens(self):
        """Test detection of token counting operations."""
        assert detect_operation_type("count_tokens") == "CHAT"
        assert detect_operation_type("count_tokens_async") == "CHAT"

    def test_detect_unknown_operation(self):
        """Test fallback for unknown operations."""
        assert detect_operation_type("unknown_method") == "CHAT"
        assert detect_operation_type("") == "CHAT"

    def test_detect_tool_calls(self):
        """Test detection of tool call operations."""
        request_with_tools = {"tools": [{"name": "search"}]}
        assert detect_operation_type("generate_content", request_with_tools) == "TOOL_CALL"


class TestTraceFieldIntegration:
    """Integration tests for trace field handling."""

    def test_environment_variable_priority(self, monkeypatch):
        """Test that REVENIUM_* env vars take priority over generic ones."""
        monkeypatch.setenv("REVENIUM_ENVIRONMENT", "production")
        monkeypatch.setenv("ENVIRONMENT", "staging")
        monkeypatch.setenv("DEPLOYMENT_ENV", "dev")
        assert get_environment() == "production"

    def test_region_cloud_provider_fallbacks(self, monkeypatch):
        """Test region fallback chain for different cloud providers."""
        # AWS
        monkeypatch.setenv("AWS_REGION", "us-east-1")
        assert get_region() == "us-east-1"

        # Azure takes priority over AWS
        monkeypatch.setenv("AZURE_REGION", "eastus")
        assert get_region() == "us-east-1"  # AWS still wins due to order

        # GCP
        monkeypatch.delenv("AWS_REGION")
        monkeypatch.delenv("AZURE_REGION")
        monkeypatch.setenv("GCP_REGION", "us-central1")
        assert get_region() == "us-central1"

        # Google Cloud
        monkeypatch.delenv("GCP_REGION")
        monkeypatch.setenv("GOOGLE_CLOUD_REGION", "europe-west1")
        assert get_region() == "europe-west1"

    def test_transaction_name_fallback_chain(self, monkeypatch):
        """Test complete fallback chain for transaction_name."""
        # Priority 1: Environment variable
        monkeypatch.setenv("REVENIUM_TRANSACTION_NAME", "env-name")
        metadata = {
            "transactionName": "camel-case-name",
            "transaction_name": "snake-case-name",
            "task_type": "fallback-task"
        }
        assert get_transaction_name(metadata) == "env-name"

        # Priority 2: transactionName (camelCase)
        monkeypatch.delenv("REVENIUM_TRANSACTION_NAME")
        assert get_transaction_name(metadata) == "camel-case-name"

        # Priority 3: transaction_name (snake_case)
        del metadata["transactionName"]
        assert get_transaction_name(metadata) == "snake-case-name"

        # Priority 4: task_type fallback
        del metadata["transaction_name"]
        assert get_transaction_name(metadata) == "fallback-task"

        # No metadata at all
        assert get_transaction_name(None) is None

