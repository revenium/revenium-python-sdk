"""
Tests for prompt capture functionality.
"""

import json
import pytest
from unittest.mock import Mock, patch
from revenium_middleware.anthropic.config import Config
from revenium_middleware.anthropic.prompt_extractor import (
    extract_prompts_from_request,
    extract_response_content,
    extract_streaming_response_content
)

MAX_PROMPT_LENGTH = Config.MAX_PROMPT_LENGTH
TRUNCATION_MARKER = "...[TRUNCATED]"


class TestExtractPromptsFromRequest:
    """Test prompt extraction from Anthropic API requests."""
    
    def test_extract_system_and_user_messages(self):
        """Test extracting system prompt and user messages."""
        kwargs = {
            'system': 'You are a helpful assistant.',
            'messages': [
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
            'system': long_content,
            'messages': [
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

        # The list is cut inside its texts, so it still parses and fits the limit
        stored = json.loads(result['inputMessages'])
        assert len(result['inputMessages']) <= MAX_PROMPT_LENGTH
        assert all(message['content'].endswith(TRUNCATION_MARKER) for message in stored)
        assert result['promptsTruncated'] is True

    def test_multimodal_content(self):
        """Test handling of multimodal content (arrays)."""
        kwargs = {
            'system': [
                {'type': 'text', 'text': 'You are helpful'}
            ],
            'messages': [
                {
                    'role': 'user',
                    'content': [
                        {'type': 'text', 'text': 'What is this?'},
                        {'type': 'image', 'source': {'type': 'url', 'url': 'http://example.com/img.jpg'}}
                    ]
                }
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
        text_block = Mock()
        text_block.type = 'text'
        text_block.text = 'Hello, how can I help you?'
        response.content = [text_block]
        
        result = extract_response_content(response)
        
        assert result['outputResponse'] == 'Hello, how can I help you?'
        assert result['promptsTruncated'] is False
    
    def test_extract_long_response(self):
        """Test truncation of long response."""
        long_content = 'C' * (MAX_PROMPT_LENGTH + 1000)
        response = Mock()
        text_block = Mock()
        text_block.type = 'text'
        text_block.text = long_content
        response.content = [text_block]

        result = extract_response_content(response)

        # Should be truncated to exactly MAX_PROMPT_LENGTH (marker included)
        assert len(result['outputResponse']) == MAX_PROMPT_LENGTH
        assert result['outputResponse'].endswith(TRUNCATION_MARKER)
        assert result['promptsTruncated'] is True

    def test_no_content(self):
        """Test when response has no content."""
        response = Mock()
        response.content = []

        result = extract_response_content(response)

        assert result['outputResponse'] is None
        assert result['promptsTruncated'] is False

    def test_preserve_existing_truncation_flag(self):
        """Test that existing truncation flag is preserved."""
        response = Mock()
        text_block = Mock()
        text_block.type = 'text'
        text_block.text = 'Short response'
        response.content = [text_block]

        result = extract_response_content(response, prompts_truncated=True)

        assert result['promptsTruncated'] is True

    def test_multiple_content_blocks(self):
        """Test extracting multiple content blocks."""
        response = Mock()
        text_block1 = Mock()
        text_block1.type = 'text'
        text_block1.text = 'First block'
        text_block2 = Mock()
        text_block2.type = 'text'
        text_block2.text = 'Second block'
        response.content = [text_block1, text_block2]

        result = extract_response_content(response)

        assert 'First block' in result['outputResponse']
        assert 'Second block' in result['outputResponse']
        assert result['promptsTruncated'] is False


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
        from revenium_middleware.anthropic.config import Config

        assert Config.MAX_PROMPT_LENGTH == 50_000



def test_message_capture_masks_credentials(monkeypatch, mock_revenium_client):
    """BACK-4237: an Anthropic call sends its captured prompt with credentials masked."""
    from .test_raw_response import REQUEST, StubAnthropic, _run_metering_synchronously

    monkeypatch.setenv("REVENIUM_CAPTURE_PROMPTS", "true")
    monkeypatch.setattr("revenium_middleware.anthropic.middleware._safe_run_async_in_thread",
                        _run_metering_synchronously)
    request = dict(
        REQUEST,
        system="Authorization: Bearer FAKEBEARERTOKENFAKEBEARERTOKEN0002",
        messages=[{"role": "user", "content": "My key is sk-ant-FAKE" + "3" * 24 + " and password: FAKEpassw0rd!"}],
    )

    StubAnthropic().sync_client().messages.create(**request)

    mock_revenium_client.ai.create_completion.assert_called_once()
    sent = mock_revenium_client.ai.create_completion.call_args.kwargs
    for field in ("system_prompt", "input_messages"):
        assert "FAKE" not in sent[field], (field, sent[field])
        assert "***REDACTED***" in sent[field], (field, sent[field])
    assert json.loads(sent["input_messages"])


LIMIT = 50_000


@pytest.fixture
def limit_50000(monkeypatch):
    monkeypatch.setattr("revenium_middleware.anthropic.prompt_extractor.Config.MAX_PROMPT_LENGTH", LIMIT)


def _user(content):
    return {'role': 'user', 'content': content}


@pytest.mark.usefixtures('limit_50000')
class TestMessageListLimit:
    """input_messages is whole when it fits and valid JSON within the limit when it does not."""

    def test_a_single_30000_character_message_is_kept_whole(self):
        result = extract_prompts_from_request({'messages': [_user('D' * 30_000)]})

        assert json.loads(result['inputMessages']) == [_user('D' * 30_000)]
        assert result['promptsTruncated'] is False

    def test_a_single_30000_character_text_block_is_kept_whole(self):
        message = _user([{'type': 'text', 'text': 'D' * 30_000}])

        result = extract_prompts_from_request({'messages': [message]})

        assert json.loads(result['inputMessages']) == [message]
        assert result['promptsTruncated'] is False

    def test_an_overflowing_message_list_is_valid_json_within_the_limit(self):
        tool_use = {'type': 'tool_use', 'id': 'toolu_01', 'name': 'search', 'input': {'q': 'x'}}
        tool_result = {'type': 'tool_result', 'tool_use_id': 'toolu_01', 'content': 'R' * 30_000}
        image = {'type': 'image', 'source': {'type': 'base64', 'media_type': 'image/png', 'data': 'I' * 10_000}}
        messages = [
            _user([{'type': 'text', 'text': 'T' * 30_000}]),
            {'role': 'assistant', 'content': [tool_use]},
            _user([tool_result, image]),
        ]

        result = extract_prompts_from_request({'messages': messages})

        stored = json.loads(result['inputMessages'])
        assert len(result['inputMessages']) <= LIMIT
        assert stored[1]['content'][0] == tool_use
        assert stored[2]['content'][1]['source']['media_type'] == 'image/png'
        assert result['promptsTruncated'] is True
