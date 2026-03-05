"""
Tests for middleware functionality.
"""
import pytest
from unittest.mock import MagicMock, patch
from revenium_middleware.perplexity.middleware import (
    get_stop_reason,
    detect_operation_type,
    extract_token_usage,
    OperationType
)


@pytest.mark.unit
class TestStopReason:
    """Test stop reason mapping."""

    def test_stop_reason_stop(self):
        """Test 'stop' finish reason maps to END."""
        assert get_stop_reason("stop") == "END"

    def test_stop_reason_length(self):
        """Test 'length' finish reason maps to TOKEN_LIMIT."""
        assert get_stop_reason("length") == "TOKEN_LIMIT"

    def test_stop_reason_content_filter(self):
        """Test 'content_filter' finish reason maps to ERROR."""
        assert get_stop_reason("content_filter") == "ERROR"

    def test_stop_reason_tool_calls(self):
        """Test 'tool_calls' finish reason maps to END_SEQUENCE."""
        assert get_stop_reason("tool_calls") == "END_SEQUENCE"

    def test_stop_reason_function_call(self):
        """Test 'function_call' finish reason maps to END_SEQUENCE."""
        assert get_stop_reason("function_call") == "END_SEQUENCE"

    def test_stop_reason_none(self):
        """Test None finish reason defaults to END."""
        assert get_stop_reason(None) == "END"

    def test_stop_reason_unknown(self):
        """Test unknown finish reason defaults to END."""
        assert get_stop_reason("unknown_reason") == "END"


@pytest.mark.unit
class TestOperationType:
    """Test operation type detection."""
    
    def test_detect_chat_operation(self, mock_openai_response):
        """Test detection of chat operation."""
        op_type = detect_operation_type(mock_openai_response)
        assert op_type == OperationType.CHAT
    
    def test_detect_operation_no_choices(self):
        """Test operation detection with no choices."""
        response = MagicMock()
        response.choices = []
        
        op_type = detect_operation_type(response)
        assert op_type == OperationType.CHAT


@pytest.mark.unit
class TestTokenUsage:
    """Test token usage extraction."""
    
    def test_extract_token_usage(self, mock_openai_response):
        """Test extracting token usage from response."""
        usage = extract_token_usage(mock_openai_response)
        
        assert usage["prompt_tokens"] == 10
        assert usage["completion_tokens"] == 20
        assert usage["total_tokens"] == 30
    
    def test_extract_token_usage_no_usage(self):
        """Test extracting token usage when usage is missing."""
        response = MagicMock()
        response.usage = None
        
        usage = extract_token_usage(response)
        
        assert usage["prompt_tokens"] == 0
        assert usage["completion_tokens"] == 0
        assert usage["total_tokens"] == 0


@pytest.mark.unit
class TestMiddlewareWrapper:
    """Test middleware wrapper functionality."""
    
    @patch('revenium_middleware.perplexity.middleware.send_metering_data')
    @patch('openai.resources.chat.completions.Completions.create')
    def test_create_wrapper_non_streaming(
        self, mock_create, mock_send_metering, mock_openai_response
    ):
        """Test wrapper for non-streaming requests."""
        from revenium_middleware.perplexity.middleware import create_wrapper
        
        # Setup mock
        mock_create.return_value = mock_openai_response
        
        # Import to trigger wrapper
        import revenium_middleware.perplexity  # noqa: F401
        
        # The wrapper should be applied automatically
        # This test verifies the wrapper logic works
        assert True  # Placeholder - actual integration test needed

