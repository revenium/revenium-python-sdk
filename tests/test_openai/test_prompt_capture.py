"""
Tests for prompt capture functionality.
"""

import json
import pytest
from unittest.mock import Mock, patch
from revenium_middleware.openai.config import Config
from revenium_middleware.openai.prompt_extractor import (
    extract_prompts_from_request,
    extract_response_content,
    extract_streaming_response_content
)

MAX_PROMPT_LENGTH = Config.MAX_PROMPT_LENGTH
TRUNCATION_MARKER = "...[TRUNCATED]"


class TestExtractPromptsFromRequest:
    """Test prompt extraction from OpenAI API requests."""
    
    def test_extract_system_and_user_messages(self):
        """Test extracting system prompt and user messages."""
        kwargs = {
            'messages': [
                {'role': 'system', 'content': 'You are a helpful assistant.'},
                {'role': 'user', 'content': 'Hello!'},
                {'role': 'assistant', 'content': 'Hi there!'},
                {'role': 'user', 'content': 'How are you?'}
            ]
        }
        
        result = extract_prompts_from_request(kwargs)
        
        assert result['systemPrompt'] == 'You are a helpful assistant.'
        assert result['promptsTruncated'] is False
        
        # Parse input messages
        input_messages = json.loads(result['inputMessages'])
        assert len(input_messages) == 3
        assert input_messages[0]['role'] == 'user'
        assert input_messages[0]['content'] == 'Hello!'
    
    def test_no_system_message(self):
        """Test when there's no system message."""
        kwargs = {
            'messages': [
                {'role': 'user', 'content': 'Hello!'}
            ]
        }
        
        result = extract_prompts_from_request(kwargs)
        
        assert result['systemPrompt'] is None
        assert result['inputMessages'] is not None
        assert result['promptsTruncated'] is False
    
    def test_empty_messages(self):
        """Test with empty messages list."""
        kwargs = {'messages': []}
        
        result = extract_prompts_from_request(kwargs)
        
        assert result['systemPrompt'] is None
        assert result['inputMessages'] is None
        assert result['promptsTruncated'] is False
    
    def test_truncation_system_prompt(self):
        """Test truncation of long system prompt."""
        long_content = 'A' * (MAX_PROMPT_LENGTH + 1000)
        kwargs = {
            'messages': [
                {'role': 'system', 'content': long_content},
                {'role': 'user', 'content': 'Hello!'}
            ]
        }

        result = extract_prompts_from_request(kwargs)

        # Should be truncated to exactly MAX_PROMPT_LENGTH (marker included)
        assert len(result['systemPrompt']) == MAX_PROMPT_LENGTH
        assert result['systemPrompt'].endswith(TRUNCATION_MARKER)
        assert result['promptsTruncated'] is True
    
    def test_truncation_input_messages(self):
        """Test truncation of long input messages."""
        # Create a large message that will exceed limit when JSON serialized
        large_message = 'B' * (MAX_PROMPT_LENGTH // 2)
        kwargs = {
            'messages': [
                {'role': 'user', 'content': large_message},
                {'role': 'user', 'content': large_message}
            ]
        }

        result = extract_prompts_from_request(kwargs)

        # Should be truncated to exactly MAX_PROMPT_LENGTH (marker included)
        assert len(result['inputMessages']) == MAX_PROMPT_LENGTH
        assert result['inputMessages'].endswith(TRUNCATION_MARKER)
        assert result['promptsTruncated'] is True
    
    def test_multimodal_content(self):
        """Test handling of multimodal content (arrays)."""
        kwargs = {
            'messages': [
                {
                    'role': 'system',
                    'content': [
                        {'type': 'text', 'text': 'You are helpful'},
                        {'type': 'image_url', 'image_url': {'url': 'http://example.com/img.jpg'}}
                    ]
                },
                {'role': 'user', 'content': 'What is this?'}
            ]
        }
        
        result = extract_prompts_from_request(kwargs)
        
        # System prompt should be JSON string for multimodal
        assert result['systemPrompt'] is not None
        assert isinstance(result['systemPrompt'], str)
        # Should be valid JSON
        json.loads(result['systemPrompt'])


class TestExtractResponseContent:
    """Test response content extraction."""
    
    def test_extract_simple_response(self):
        """Test extracting simple text response."""
        response = Mock()
        response.choices = [Mock()]
        response.choices[0].message = Mock()
        response.choices[0].message.content = 'Hello, how can I help you?'
        
        result = extract_response_content(response)
        
        assert result['outputResponse'] == 'Hello, how can I help you?'
        assert result['promptsTruncated'] is False
    
    def test_extract_long_response(self):
        """Test truncation of long response."""
        long_content = 'C' * (MAX_PROMPT_LENGTH + 1000)
        response = Mock()
        response.choices = [Mock()]
        response.choices[0].message = Mock()
        response.choices[0].message.content = long_content

        result = extract_response_content(response)

        # Should be truncated to exactly MAX_PROMPT_LENGTH (marker included)
        assert len(result['outputResponse']) == MAX_PROMPT_LENGTH
        assert result['outputResponse'].endswith(TRUNCATION_MARKER)
        assert result['promptsTruncated'] is True
    
    def test_no_content(self):
        """Test when response has no content."""
        response = Mock()
        response.choices = [Mock()]
        response.choices[0].message = Mock()
        response.choices[0].message.content = None
        
        result = extract_response_content(response)
        
        assert result['outputResponse'] is None
        assert result['promptsTruncated'] is False
    
    def test_preserve_existing_truncation_flag(self):
        """Test that existing truncation flag is preserved."""
        response = Mock()
        response.choices = [Mock()]
        response.choices[0].message = Mock()
        response.choices[0].message.content = 'Short response'
        
        result = extract_response_content(response, prompts_truncated=True)
        
        assert result['promptsTruncated'] is True


class TestExtractStreamingResponseContent:
    """Test streaming response content extraction."""

    def test_extract_streaming_content(self):
        """Test extracting accumulated streaming content."""
        accumulated = 'This is a streaming response.'

        result = extract_streaming_response_content(accumulated)

        assert result['outputResponse'] == accumulated
        assert result['promptsTruncated'] is False

    def test_truncate_long_streaming_content(self):
        """Test truncation of long streaming content."""
        long_content = 'D' * (MAX_PROMPT_LENGTH + 1000)

        result = extract_streaming_response_content(long_content)

        # Should be truncated to exactly MAX_PROMPT_LENGTH (marker included)
        assert len(result['outputResponse']) == MAX_PROMPT_LENGTH
        assert result['outputResponse'].endswith(TRUNCATION_MARKER)
        assert result['promptsTruncated'] is True


class TestPromptCaptureIntegration:
    """Integration tests for prompt capture with middleware."""

    def test_prompt_capture_disabled_by_default(self):
        from revenium_middleware._core.config import is_capture_prompts_enabled

        assert is_capture_prompts_enabled() is False

    @patch.dict('os.environ', {'REVENIUM_CAPTURE_PROMPTS': 'true'})
    def test_prompt_capture_enabled_via_env(self):
        from revenium_middleware._core.config import is_capture_prompts_enabled

        assert is_capture_prompts_enabled() is True

    @patch.dict('os.environ', {'REVENIUM_CAPTURE_PROMPTS': 'false'})
    def test_prompt_capture_disabled_via_env(self):
        from revenium_middleware._core.config import is_capture_prompts_enabled

        assert is_capture_prompts_enabled() is False

    def test_max_prompt_length_constant(self):
        """Test that MAX_PROMPT_LENGTH is set correctly."""
        from revenium_middleware.openai.config import Config

        assert Config.MAX_PROMPT_LENGTH == 50_000

