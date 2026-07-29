"""
Test suite for trace visualization features.

This module tests:
1. Environment variable capture (environment, region, credential_alias, etc.)
2. Trace type and trace name validation
3. Operation type detection
4. Integration with Anthropic API calls
"""

import os
import pytest
from unittest.mock import patch

# Import middleware (this patches Anthropic)
import revenium_middleware.anthropic.middleware  # noqa: F401
from revenium_middleware.anthropic.trace_fields import (
    get_environment, get_region, get_credential_alias,
    get_trace_type, get_trace_name, get_parent_transaction_id,
    get_transaction_name, get_retry_number, get_ticket_id,
    detect_operation_type, validate_trace_type, validate_trace_name
)


@pytest.fixture
def trace_env_vars():
    """Set up trace visualization environment variables for testing."""
    env_vars = {
        'REVENIUM_ENVIRONMENT': 'testing',
        'REVENIUM_REGION': 'us-east-1',
        'REVENIUM_CREDENTIAL_ALIAS': 'test-anthropic-key',
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
        assert get_credential_alias() == 'test-anthropic-key'

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

    def test_get_retry_number_default(self):
        """Test retry number defaults to 0 when not set."""
        # Temporarily remove the env var
        original = os.environ.pop('REVENIUM_RETRY_NUMBER', None)
        try:
            assert get_retry_number() == 0
        finally:
            if original is not None:
                os.environ['REVENIUM_RETRY_NUMBER'] = original


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


class TestTraceValidation:
    """Test trace type and trace name validation."""

    def test_validate_trace_type_valid(self):
        """Test valid trace type passes validation."""
        assert validate_trace_type('test-workflow') == 'test-workflow'
        assert validate_trace_type('my_workflow_123') == 'my_workflow_123'

    def test_validate_trace_type_too_long(self):
        """Test trace type longer than 128 chars returns None."""
        long_type = 'a' * 129
        assert validate_trace_type(long_type) is None

    def test_validate_trace_type_invalid_chars(self):
        """Test trace type with invalid characters returns None."""
        assert validate_trace_type('test workflow') is None  # space
        assert validate_trace_type('test@workflow') is None  # @
        assert validate_trace_type('test.workflow') is None  # .

    def test_validate_trace_name_valid(self):
        """Test valid trace name passes validation."""
        assert validate_trace_name('Test Workflow') == 'Test Workflow'

    def test_validate_trace_name_truncation(self):
        """Test trace name longer than 256 chars is truncated."""
        long_name = 'a' * 300
        result = validate_trace_name(long_name)
        assert len(result) == 256
        assert result == 'a' * 256





class TestOperationTypeDetection:
    """Test operation type detection."""

    def test_detect_chat_completion(self):
        """Test detection of chat completion."""
        result = detect_operation_type('anthropic', '/messages', {})
        assert result['operationType'] == 'CHAT'
        assert result['operationSubtype'] is None

    def test_detect_tool_use(self):
        """Test detection of tool use."""
        request_body = {
            'tools': [
                {
                    'name': 'get_weather',
                    'description': 'Get weather info',
                    'input_schema': {'type': 'object'}
                }
            ]
        }
        result = detect_operation_type('anthropic', '/messages', request_body)
        assert result['operationType'] == 'TOOL_CALL'
        assert result['operationSubtype'] is None

    def test_detect_unknown_provider(self):
        """Test detection with unknown provider."""
        result = detect_operation_type('unknown', '/messages', {})
        assert result['operationType'] == 'CHAT'
        assert result['operationSubtype'] is None


class TestTraceVisualizationIntegration:
    """Test trace visualization integration with Anthropic API."""

    @pytest.mark.e2e
    @pytest.mark.skipif(
        not os.environ.get('ANTHROPIC_API_KEY'),
        reason="ANTHROPIC_API_KEY not set"
    )
    def test_trace_fields_with_real_api(self, trace_env_vars, sample_usage_metadata):
        """Test trace fields are captured with real Anthropic API call."""
        from anthropic import Anthropic

        client = Anthropic()

        # Make a simple API call with usage metadata
        response = client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=50,
            messages=[
                {"role": "user", "content": "Say 'Hello, trace visualization!'"}
            ],
            usage_metadata=sample_usage_metadata
        )

        # Verify response
        assert response.id is not None
        assert response.content is not None
        assert len(response.content) > 0

        # Note: We can't directly verify the trace fields were sent to Revenium
        # without mocking, but this test ensures the middleware doesn't break
        # the API call when trace fields are present

    def test_usage_metadata_precedence(self):
        """Test that usage_metadata values take precedence over env vars."""
        from revenium_middleware.anthropic.middleware import _extract_trace_fields

        # Set env vars
        os.environ['REVENIUM_ENVIRONMENT'] = 'env-value'
        os.environ['REVENIUM_REGION'] = 'env-region'

        # Create usage_metadata with different values
        usage_metadata = {
            'environment': 'metadata-value',
            'region': 'metadata-region',
        }

        try:
            result = _extract_trace_fields(usage_metadata)

            # Metadata values should take precedence
            assert result['environment'] == 'metadata-value'
            assert result['region'] == 'metadata-region'
        finally:
            # Clean up
            os.environ.pop('REVENIUM_ENVIRONMENT', None)
            os.environ.pop('REVENIUM_REGION', None)

    def test_camel_case_and_snake_case_support(self):
        """Test that both camelCase and snake_case are supported in usage_metadata."""
        from revenium_middleware.anthropic.middleware import _extract_trace_fields

        # Test camelCase
        usage_metadata_camel = {
            'credentialAlias': 'camel-alias',
            'traceType': 'camel-type',
            'traceName': 'Camel Name',
            'parentTransactionId': 'camel-parent',
            'transactionName': 'Camel Transaction',
            'retryNumber': 5,
        }

        result = _extract_trace_fields(usage_metadata_camel)
        assert result['credential_alias'] == 'camel-alias'
        assert result['trace_type'] == 'camel-type'
        assert result['trace_name'] == 'Camel Name'
        assert result['parent_transaction_id'] == 'camel-parent'
        assert result['transaction_name'] == 'Camel Transaction'
        assert result['retry_number'] == 5

        # Test snake_case
        usage_metadata_snake = {
            'credential_alias': 'snake-alias',
            'trace_type': 'snake-type',
            'trace_name': 'Snake Name',
            'parent_transaction_id': 'snake-parent',
            'transaction_name': 'Snake Transaction',
            'retry_number': 3,
        }

        result = _extract_trace_fields(usage_metadata_snake)
        assert result['credential_alias'] == 'snake-alias'
        assert result['trace_type'] == 'snake-type'
        assert result['trace_name'] == 'Snake Name'
        assert result['parent_transaction_id'] == 'snake-parent'
        assert result['transaction_name'] == 'Snake Transaction'
        assert result['retry_number'] == 3
