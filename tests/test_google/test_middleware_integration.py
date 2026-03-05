"""
Integration test to verify the middleware correctly captures prompts
from both positional and keyword arguments in realistic scenarios.

This simulates how the middleware wrappers call extract_prompt_data_if_enabled.
"""

import os
import json
import pytest
import importlib


class TestMiddlewareIntegration:
    """Integration tests for middleware prompt capture."""

    @pytest.fixture(autouse=True)
    def setup_env(self, monkeypatch):
        """Set up environment for integration tests."""
        # Set environment variables
        monkeypatch.setenv('REVENIUM_CAPTURE_PROMPTS', 'true')
        monkeypatch.setenv('REVENIUM_MAX_PROMPT_LENGTH', '1000')

        # Reload core config (where CAPTURE_PROMPTS is defined) then provider config
        from revenium_middleware._core import config as core_config
        importlib.reload(core_config)
        from revenium_middleware.google import config
        importlib.reload(config)

        # Reload the prompt_extractor module to use the new config
        from revenium_middleware.google import prompt_extractor
        importlib.reload(prompt_extractor)

        yield

    def test_simple_string_as_positional_argument(self):
        """Test: model.generate_content('What is AI?')"""
        from revenium_middleware.google.prompt_extractor import extract_prompt_data_if_enabled

        # This is what the wrapper sees
        args = ("What is AI?",)
        kwargs = {}

        system_prompt, input_messages, output_response, truncated = extract_prompt_data_if_enabled(
            kwargs, args=args
        )

        assert input_messages is not None
        parsed = json.loads(input_messages)
        assert parsed[0]['role'] == 'user'
        assert parsed[0]['parts'][0]['text'] == 'What is AI?'

    def test_string_as_keyword_argument(self):
        """Test: model.generate_content(contents='Explain ML')"""
        from revenium_middleware.google.prompt_extractor import extract_prompt_data_if_enabled

        args = ()
        kwargs = {"contents": "Explain ML"}

        system_prompt, input_messages, output_response, truncated = extract_prompt_data_if_enabled(
            kwargs, args=args
        )

        assert input_messages is not None
        parsed = json.loads(input_messages)
        assert parsed[0]['parts'][0]['text'] == 'Explain ML'

    def test_list_of_messages_as_positional_argument(self):
        """Test: model.generate_content([{'role': 'user', 'parts': [{'text': 'Hi'}]}])"""
        from revenium_middleware.google.prompt_extractor import extract_prompt_data_if_enabled

        args = ([{"role": "user", "parts": [{"text": "Hi"}]}],)
        kwargs = {}

        system_prompt, input_messages, output_response, truncated = extract_prompt_data_if_enabled(
            kwargs, args=args
        )

        assert input_messages is not None
        parsed = json.loads(input_messages)
        assert len(parsed) == 1
        assert parsed[0]['role'] == 'user'

    def test_with_system_instruction(self):
        """Test: model with system_instruction calling generate_content"""
        from revenium_middleware.google.prompt_extractor import extract_prompt_data_if_enabled

        args = ("Question",)
        kwargs = {"system_instruction": "Be helpful"}

        system_prompt, input_messages, output_response, truncated = extract_prompt_data_if_enabled(
            kwargs, args=args
        )

        assert system_prompt == "Be helpful"
        assert input_messages is not None

    def test_streaming_response_accumulation(self):
        """Test: for chunk in model.generate_content('Hi', stream=True)"""
        from revenium_middleware.google.prompt_extractor import extract_prompt_data_if_enabled

        args = ("Hi",)
        kwargs = {}
        accumulated_content = "Hello! How can I help you today?"

        system_prompt, input_messages, output_response, truncated = extract_prompt_data_if_enabled(
            kwargs, args=args, accumulated_content=accumulated_content
        )

        assert input_messages is not None
        parsed = json.loads(input_messages)
        assert parsed[0]['parts'][0]['text'] == "Hi"
        assert output_response == accumulated_content

    def test_long_content_with_valid_json_truncation(self):
        """Test that very long content is truncated while maintaining valid JSON."""
        from revenium_middleware.google.prompt_extractor import extract_prompt_data_if_enabled

        long_message = "A" * 2000  # Much longer than 1000 char limit
        args = (long_message,)
        kwargs = {}

        system_prompt, input_messages, output_response, truncated = extract_prompt_data_if_enabled(
            kwargs, args=args
        )

        assert truncated is True
        assert len(input_messages) < len(long_message)

        # Verify it's still valid JSON
        parsed = json.loads(input_messages)
        assert isinstance(parsed, list)

        # Check for truncation marker
        has_marker = any(
            '[TRUNCATED]' in str(part)
            for msg in parsed
            for part in msg.get('parts', [])
        )
        assert has_marker

