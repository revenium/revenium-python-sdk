"""
Tests for prompt capture functionality in Google middleware.

This module tests the prompt extraction and capture features for both
Google AI SDK and Vertex AI SDK.
"""

import pytest
from unittest.mock import Mock, patch
from revenium_middleware.google.config import Config
from revenium_middleware.google.prompt_extractor import (
    extract_prompts_from_request,
    extract_response_content,
    extract_streaming_response_content,
)


class TestPromptExtraction:
    """Test prompt extraction from Google Gemini API requests and responses."""

    def test_extract_system_instruction_from_dict(self):
        """Test extracting system instruction from dict format."""
        kwargs = {
            'system_instruction': {
                'parts': [{'text': 'You are a helpful assistant.'}]
            },
            'contents': 'Hello'
        }
        
        result = extract_prompts_from_request(kwargs)
        
        assert result['systemPrompt'] == 'You are a helpful assistant.'
        assert result['inputMessages'] is not None
        assert result['promptsTruncated'] is False

    def test_extract_system_instruction_from_string(self):
        """Test extracting system instruction from string format."""
        kwargs = {
            'system_instruction': 'You are a helpful assistant.',
            'contents': 'Hello'
        }
        
        result = extract_prompts_from_request(kwargs)
        
        assert result['systemPrompt'] == 'You are a helpful assistant.'
        assert result['inputMessages'] is not None

    def test_extract_contents_string(self):
        """Test extracting contents from simple string."""
        kwargs = {
            'contents': 'What is 2+2?'
        }
        
        result = extract_prompts_from_request(kwargs)
        
        assert result['inputMessages'] is not None
        assert '2+2' in result['inputMessages']
        assert result['promptsTruncated'] is False

    def test_extract_contents_list(self):
        """Test extracting contents from list format."""
        kwargs = {
            'contents': [
                {'role': 'user', 'parts': [{'text': 'Hello'}]},
                {'role': 'model', 'parts': [{'text': 'Hi there!'}]},
                {'role': 'user', 'parts': [{'text': 'How are you?'}]}
            ]
        }
        
        result = extract_prompts_from_request(kwargs)
        
        assert result['inputMessages'] is not None
        assert 'Hello' in result['inputMessages']
        assert 'How are you?' in result['inputMessages']

    def test_truncation_system_prompt(self):
        """Test that long system prompts are truncated."""
        long_text = 'A' * (Config.MAX_PROMPT_LENGTH + 1000)
        kwargs = {
            'system_instruction': long_text,
            'contents': 'Hello'
        }
        
        result = extract_prompts_from_request(kwargs)
        
        assert result['systemPrompt'] is not None
        assert len(result['systemPrompt']) <= Config.MAX_PROMPT_LENGTH
        assert result['promptsTruncated'] is True
        assert '[TRUNCATED]' in result['systemPrompt']

    def test_truncation_input_messages(self):
        """Test that long input messages are truncated."""
        long_text = 'B' * (Config.MAX_PROMPT_LENGTH + 1000)
        kwargs = {
            'contents': long_text
        }
        
        result = extract_prompts_from_request(kwargs)
        
        assert result['inputMessages'] is not None
        assert len(result['inputMessages']) <= Config.MAX_PROMPT_LENGTH
        assert result['promptsTruncated'] is True

    def test_extract_response_content(self):
        """Test extracting response content from API response."""
        # Create mock response
        mock_part = Mock()
        mock_part.text = 'The answer is 4.'
        
        mock_content = Mock()
        mock_content.parts = [mock_part]
        
        mock_candidate = Mock()
        mock_candidate.content = mock_content
        
        mock_response = Mock()
        mock_response.candidates = [mock_candidate]
        
        result = extract_response_content(mock_response)
        
        assert result['outputResponse'] == 'The answer is 4.'
        assert result['promptsTruncated'] is False

    def test_extract_response_with_text_property(self):
        """Test extracting response using .text property fallback."""
        mock_response = Mock()
        mock_response.candidates = []
        mock_response.text = 'Fallback response'
        
        result = extract_response_content(mock_response)
        
        assert result['outputResponse'] == 'Fallback response'

    def test_extract_streaming_response(self):
        """Test extracting accumulated streaming content."""
        accumulated = 'This is a streaming response.'
        
        result = extract_streaming_response_content(accumulated)
        
        assert result['outputResponse'] == accumulated
        assert result['promptsTruncated'] is False

    def test_truncation_output_response(self):
        """Test that long output responses are truncated."""
        long_response = 'C' * (Config.MAX_PROMPT_LENGTH + 1000)
        
        result = extract_streaming_response_content(long_response)
        
        assert result['outputResponse'] is not None
        assert len(result['outputResponse']) <= Config.MAX_PROMPT_LENGTH
        assert result['promptsTruncated'] is True
        assert '[TRUNCATED]' in result['outputResponse']

    def test_no_contents(self):
        """Test handling of empty contents."""
        kwargs = {}
        
        result = extract_prompts_from_request(kwargs)
        
        assert result['systemPrompt'] is None
        assert result['inputMessages'] is None
        assert result['promptsTruncated'] is False

