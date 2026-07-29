"""
Tests for trace visualization field capture and validation.

These are unit tests that do NOT require:
- Ollama to be running
- Real API keys
- External network calls
"""

import os
from unittest.mock import patch

import pytest
from revenium_middleware.ollama.trace_fields import (
    get_environment,
    get_region,
    get_credential_alias,
    get_trace_type,
    get_trace_name,
    get_parent_transaction_id,
    get_transaction_name,
    get_retry_number,
    get_ticket_id,
    validate_trace_type,
    validate_trace_name,
    detect_operation_type
)


@pytest.mark.unit
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

    def test_get_credential_alias(self, monkeypatch):
        """Test credential alias capture."""
        monkeypatch.setenv("REVENIUM_CREDENTIAL_ALIAS", "prod-api-key")
        assert get_credential_alias() == "prod-api-key"

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
            if key.startswith("REVENIUM_") or key in ["ENVIRONMENT", "DEPLOYMENT_ENV", "AWS_REGION"]:
                os.environ.pop(key, None)

        assert get_environment() is None
        assert get_region() is None
        assert get_credential_alias() is None
        assert get_trace_type() is None
        assert get_trace_name() is None
        assert get_parent_transaction_id() is None
        assert get_transaction_name() is None


@pytest.mark.unit
class TestTicketIdCapture:
    """Test ticketId capture (FRONT-1545)."""

    def test_env_var_fallback(self):
        with patch.dict(os.environ, {'REVENIUM_TICKET_ID': 'JIRA-77'}):
            assert get_ticket_id({}) == 'JIRA-77'

    def test_metadata_takes_precedence(self):
        with patch.dict(os.environ, {'REVENIUM_TICKET_ID': 'ENV-1'}):
            assert get_ticket_id({'ticketId': 'META-1'}) == 'META-1'

    def test_none_when_unset(self):
        with patch.dict(os.environ, {}, clear=True):
            assert get_ticket_id() is None


@pytest.mark.unit
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

    def test_validate_trace_name_truncation(self):
        """Test trace name truncation when too long."""
        long_name = "a" * 300
        result = validate_trace_name(long_name)
        assert len(result) == 256
        assert result == "a" * 256

    def test_validate_trace_name_max_length(self):
        """Test trace name at exactly max length."""
        max_name = "a" * 256
        assert validate_trace_name(max_name) == max_name


@pytest.mark.unit
class TestOperationTypeDetection:
    """Test operation type auto-detection."""

    def test_chat_completion_basic(self):
        """Test basic chat completion detection."""
        op_type = detect_operation_type('chat', {})
        assert op_type == 'CHAT'

    def test_chat_completion_with_tools(self):
        """Test chat completion with tools detection."""
        request_body = {'tools': [{'type': 'function', 'function': {}}]}
        op_type = detect_operation_type('chat', request_body)
        assert op_type == 'TOOL_CALL'

    def test_generate(self):
        """Test generate endpoint detection."""
        op_type = detect_operation_type('generate', {})
        assert op_type == 'GENERATE'

    def test_embeddings(self):
        """Test embeddings endpoint detection."""
        op_type = detect_operation_type('embeddings', {})
        assert op_type == 'EMBED'

        op_type = detect_operation_type('embed', {})
        assert op_type == 'EMBED'

    def test_unknown_endpoint(self):
        """Test unknown endpoint defaults to CHAT."""
        op_type = detect_operation_type('unknown', {})
        assert op_type == 'CHAT'

