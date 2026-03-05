"""
Test suite for trace visualization features.

This module tests:
1. Environment variable capture (environment, region, credential_alias, etc.)
2. Trace type and trace name validation
3. Operation type detection
4. Integration with OpenAI API calls
"""

import os
import pytest
from unittest.mock import patch
from openai import OpenAI

# Import middleware (this patches OpenAI)
import revenium_middleware.openai.middleware  # noqa: F401
from revenium_middleware.openai.trace_fields import (
    get_environment, get_region, get_credential_alias,
    get_trace_type, get_trace_name, get_parent_transaction_id,
    get_transaction_name, get_retry_number, detect_operation_type,
    validate_trace_type, validate_trace_name
)


@pytest.fixture
def trace_env_vars():
    """Set up trace visualization environment variables for testing."""
    env_vars = {
        'REVENIUM_ENVIRONMENT': 'testing',
        'REVENIUM_REGION': 'us-east-1',
        'REVENIUM_CREDENTIAL_ALIAS': 'test-openai-key',
        'REVENIUM_TRACE_TYPE': 'test-workflow',
        'REVENIUM_TRACE_NAME': 'Trace Visualization Test Run',
        'REVENIUM_PARENT_TRANSACTION_ID': 'parent-txn-123',
        'REVENIUM_TRANSACTION_NAME': 'Test Chat Completion',
        'REVENIUM_RETRY_NUMBER': '0',
    }

    # Store original values
    original_values = {}
    for key in env_vars:
        original_values[key] = os.environ.get(key)

    # Set test values
    for key, value in env_vars.items():
        os.environ[key] = value

    yield env_vars

    # Restore original values
    for key, original_value in original_values.items():
        if original_value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = original_value


@pytest.fixture
def sample_usage_metadata():
    """Sample usage metadata for testing."""
    return {
        "organization_id": "org-trace-viz-test",
        "product_id": "prod-trace-viz-test",
        "trace_id": "trace-viz-test-123",
        "task_type": "testing",
    }


class TestTraceFieldCapture:
    """Test trace field capture functions."""

    def test_get_environment(self, trace_env_vars):
        """Test environment variable capture."""
        assert get_environment() == 'testing'

    def test_get_region(self, trace_env_vars):
        """Test region variable capture."""
        assert get_region() == 'us-east-1'

    def test_get_credential_alias(self, trace_env_vars):
        """Test credential alias capture."""
        assert get_credential_alias() == 'test-openai-key'

    def test_get_trace_type(self, trace_env_vars):
        """Test trace type capture and validation."""
        assert get_trace_type() == 'test-workflow'

    def test_get_trace_name(self, trace_env_vars):
        """Test trace name capture and validation."""
        assert get_trace_name() == 'Trace Visualization Test Run'

    def test_get_parent_transaction_id(self, trace_env_vars):
        """Test parent transaction ID capture."""
        assert get_parent_transaction_id() == 'parent-txn-123'

    def test_get_transaction_name(self, trace_env_vars):
        """Test transaction name capture."""
        assert get_transaction_name() == 'Test Chat Completion'

    def test_get_retry_number(self, trace_env_vars):
        """Test retry number capture."""
        assert get_retry_number() == 0

    def test_get_retry_number_invalid(self):
        """Test retry number with invalid value defaults to 0."""
        with patch.dict(os.environ, {'REVENIUM_RETRY_NUMBER': 'invalid'}):
            assert get_retry_number() == 0

    def test_fields_without_env_vars(self):
        """Test that functions return None when env vars are not set."""
        with patch.dict(os.environ, {}, clear=True):
            assert get_environment() is None
            assert get_region() is None
            assert get_credential_alias() is None
            assert get_trace_type() is None
            assert get_trace_name() is None
            assert get_parent_transaction_id() is None
            assert get_transaction_name() is None
            assert get_retry_number() == 0  # Defaults to 0


class TestTraceValidation:
    """Test trace type and trace name validation functions."""

    @pytest.mark.unit
    def test_validate_trace_type_valid(self):
        """Test validation of valid trace type."""
        valid_type = "test-workflow_123"
        assert validate_trace_type(valid_type) == valid_type

    @pytest.mark.unit
    def test_validate_trace_type_invalid_characters(self):
        """Test validation rejects invalid characters."""
        invalid_type = "test@workflow!"
        assert validate_trace_type(invalid_type) is None

    @pytest.mark.unit
    def test_validate_trace_type_too_long(self):
        """Test validation rejects trace type that's too long."""
        long_type = "a" * 150
        assert validate_trace_type(long_type) is None

    @pytest.mark.unit
    def test_validate_trace_type_max_length(self):
        """Test validation accepts trace type at max length."""
        max_length_type = "a" * 100
        assert validate_trace_type(max_length_type) == max_length_type

    @pytest.mark.unit
    def test_validate_trace_name_valid(self):
        """Test validation of valid trace name."""
        valid_name = "Test Trace Name"
        assert validate_trace_name(valid_name) == valid_name

    @pytest.mark.unit
    def test_validate_trace_name_truncation(self):
        """Test that long trace names are truncated."""
        long_name = "a" * 300
        result = validate_trace_name(long_name)
        assert result is not None
        assert len(result) == 256
        assert result == "a" * 256

    @pytest.mark.unit
    def test_validate_trace_name_max_length(self):
        """Test trace name at max length is not truncated."""
        max_length_name = "a" * 256
        assert validate_trace_name(max_length_name) == max_length_name

    @pytest.mark.unit
    def test_usage_metadata_trace_type_validation_bypass_fix(self):
        """
        Test that trace_type from usage_metadata is properly validated.

        This test verifies the fix for the validation bypass vulnerability
        where trace_type values from usage_metadata bypassed validation.

        Bug: Previously, values from usage_metadata.get('traceType') were
        used directly without validation, allowing invalid characters and
        exceeding length limits.

        Fix: Now all trace_type values are validated regardless of source.
        """
        # Test invalid characters are rejected
        invalid_types = [
            'invalid@trace',
            'invalid!type',
            'invalid trace',  # spaces
            'invalid#type',
            'invalid$type',
        ]

        for invalid_type in invalid_types:
            result = validate_trace_type(invalid_type)
            assert result is None, \
                f"trace_type '{invalid_type}' should be rejected"

        # Test exceeding max length (128 chars) is rejected
        long_type = 'a' * 150
        result = validate_trace_type(long_type)
        assert result is None, \
            "trace_type exceeding 128 chars should be rejected"

        # Test valid values are accepted
        valid_types = [
            'valid-trace',
            'valid_trace',
            'ValidTrace123',
            'trace-type_123',
        ]

        for valid_type in valid_types:
            result = validate_trace_type(valid_type)
            assert result == valid_type, \
                f"trace_type '{valid_type}' should be accepted"

    @pytest.mark.unit
    def test_usage_metadata_trace_name_validation_bypass_fix(self):
        """
        Test that trace_name from usage_metadata is properly validated.

        This test verifies the fix for the validation bypass vulnerability
        where trace_name values from usage_metadata bypassed validation.

        Bug: Previously, values from usage_metadata.get('traceName') were
        used directly without validation, allowing names >256 chars without
        truncation.

        Fix: Now all trace_name values are validated and truncated if needed.
        """
        # Test truncation for names exceeding 256 chars
        long_name = 'a' * 300
        result = validate_trace_name(long_name)
        assert result is not None, \
            "trace_name should not be None"
        assert len(result) == 256, \
            "trace_name should be truncated to 256 characters"
        assert result == 'a' * 256, \
            "trace_name should be truncated correctly"

        # Test valid names are accepted as-is
        valid_names = [
            'short-name',
            'a' * 256,  # exactly at limit
            'name with spaces',
            'name@with#special$chars',
        ]

        for valid_name in valid_names:
            result = validate_trace_name(valid_name)
            assert result == valid_name, \
                f"trace_name '{valid_name}' should be accepted as-is"


class TestOperationTypeDetection:
    """Test operation type detection for different API endpoints."""

    def test_chat_completion_basic(self):
        """Test detection of basic chat completion."""
        op_type = detect_operation_type('openai', '/v1/chat/completions', {})
        assert op_type['operationType'] == 'CHAT'
        assert op_type['operationSubtype'] is None

    def test_chat_completion_with_tools(self):
        """Test detection of chat completion with tools."""
        request_body = {'tools': [{'type': 'function'}]}
        op_type = detect_operation_type(
            'openai', '/v1/chat/completions', request_body
        )
        assert op_type['operationType'] == 'TOOL_CALL'
        assert op_type['operationSubtype'] == 'function_call'

    def test_embeddings(self):
        """Test detection of embeddings operation."""
        op_type = detect_operation_type('openai', '/v1/embeddings', {})
        assert op_type['operationType'] == 'EMBED'
        assert op_type['operationSubtype'] is None

    def test_moderations(self):
        """Test detection of moderations operation."""
        op_type = detect_operation_type('openai', '/v1/moderations', {})
        assert op_type['operationType'] == 'MODERATION'
        assert op_type['operationSubtype'] is None


@pytest.mark.e2e
class TestTraceVisualizationIntegration:
    """Integration tests with actual OpenAI API calls.

    These tests require OPENAI_API_KEY and REVENIUM_METERING_API_KEY
    to be set in the environment.
    """

    @pytest.fixture(autouse=True)
    def check_api_keys(self):
        """Check if required API keys are set."""
        if not os.getenv('OPENAI_API_KEY'):
            pytest.skip(
                "OPENAI_API_KEY not set - skipping integration test"
            )
        if not os.getenv('REVENIUM_METERING_API_KEY'):
            pytest.skip(
                "REVENIUM_METERING_API_KEY not set - skipping integration test"
            )

    def test_chat_completion_with_trace_fields(
        self, trace_env_vars, sample_usage_metadata  # noqa: ARG002
    ):
        """Test trace visualization fields captured during API call."""
        # trace_env_vars fixture sets up environment variables
        client = OpenAI()

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Say 'Trace visualization is working!' "
                        "in exactly those words."
                    )
                }
            ],
            usage_metadata=sample_usage_metadata,
            max_tokens=20
        )

        # Verify response
        assert response is not None
        assert response.choices[0].message.content is not None
        assert response.usage.total_tokens > 0

        # Verify trace fields are captured correctly
        assert get_environment() == 'testing'
        assert get_region() == 'us-east-1'
        assert get_credential_alias() == 'test-openai-key'
        assert get_trace_type() == 'test-workflow'
        assert get_trace_name() == 'Trace Visualization Test Run'
        assert get_parent_transaction_id() == 'parent-txn-123'
        assert get_transaction_name() == 'Test Chat Completion'
        assert get_retry_number() == 0
