"""
Tests for trace visualization fields.
"""
import os
import pytest
from revenium_middleware.perplexity.trace_fields import (
    get_environment,
    get_region,
    get_credential_alias,
    get_trace_type,
    get_trace_name,
    get_parent_transaction_id,
    get_transaction_name,
    get_retry_number,
    validate_trace_type,
    validate_trace_name
)


@pytest.mark.unit
class TestTraceFieldExtraction:
    """Test extraction of trace fields from environment."""

    def test_get_environment(self, monkeypatch):
        """Test environment extraction."""
        monkeypatch.setenv("REVENIUM_ENVIRONMENT", "production")
        assert get_environment() == "production"

    def test_get_region(self, monkeypatch):
        """Test region extraction."""
        monkeypatch.setenv("REVENIUM_REGION", "us-east-1")
        assert get_region() == "us-east-1"

    def test_get_credential_alias(self, monkeypatch):
        """Test credential alias extraction."""
        monkeypatch.setenv("REVENIUM_CREDENTIAL_ALIAS", "Perplexity Prod")
        assert get_credential_alias() == "Perplexity Prod"

    def test_get_trace_type(self, monkeypatch):
        """Test trace type extraction."""
        monkeypatch.setenv("REVENIUM_TRACE_TYPE", "customer_support")
        assert get_trace_type() == "customer_support"

    def test_get_trace_name(self, monkeypatch):
        """Test trace name extraction."""
        monkeypatch.setenv("REVENIUM_TRACE_NAME", "Support Ticket #123")
        assert get_trace_name() == "Support Ticket #123"

    def test_get_parent_transaction_id(self, monkeypatch):
        """Test parent transaction ID extraction."""
        monkeypatch.setenv("REVENIUM_PARENT_TRANSACTION_ID", "parent-123")
        assert get_parent_transaction_id() == "parent-123"

    def test_get_transaction_name(self, monkeypatch):
        """Test transaction name extraction."""
        monkeypatch.setenv("REVENIUM_TRANSACTION_NAME", "Answer Question")
        assert get_transaction_name() == "Answer Question"

    def test_get_retry_number_default(self):
        """Test retry number defaults to 0."""
        assert get_retry_number() == 0

    def test_get_retry_number(self, monkeypatch):
        """Test retry number extraction."""
        monkeypatch.setenv("REVENIUM_RETRY_NUMBER", "3")
        assert get_retry_number() == 3


@pytest.mark.unit
class TestTraceValidation:
    """Test trace field validation."""

    def test_validate_trace_type_valid(self):
        """Test validation of valid trace type."""
        result = validate_trace_type("customer_support-v2")
        assert result == "customer_support-v2"

    def test_validate_trace_type_invalid_chars(self):
        """Test validation rejects invalid characters."""
        result = validate_trace_type("customer@support#123")
        assert result is None

    def test_validate_trace_type_truncation(self):
        """Test validation rejects trace types exceeding max length."""
        long_type = "a" * 200
        result = validate_trace_type(long_type)
        assert result is None

    def test_validate_trace_name_valid(self):
        """Test validation of valid trace name."""
        result = validate_trace_name("Support Ticket #12345")
        assert result == "Support Ticket #12345"

    def test_validate_trace_name_truncation(self):
        """Test validation truncates long trace names."""
        long_name = "a" * 300
        result = validate_trace_name(long_name)
        assert result is not None
        assert len(result) == 256
