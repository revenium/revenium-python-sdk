"""
Live tests to verify prompt capture functionality with mock data.

This demonstrates that the prompt capture code correctly handles:
1. Positional arguments (contents passed as args)
2. Keyword arguments (contents passed as kwargs)
3. System instructions
4. JSON truncation without breaking JSON structure
5. Streaming accumulation with early truncation
"""

import os
import json
import pytest
import importlib


class TestPromptCaptureLive:
    """Live tests for prompt capture functionality."""

    @pytest.fixture(autouse=True)
    def setup_env(self, monkeypatch):
        """Set up environment for prompt capture tests."""
        # Set environment variables
        monkeypatch.setenv('REVENIUM_CAPTURE_PROMPTS', 'true')
        monkeypatch.setenv('REVENIUM_MAX_PROMPT_LENGTH', '500')

        # Reload core config (where CAPTURE_PROMPTS is defined) then provider config
        from revenium_middleware._core import config as core_config
        importlib.reload(core_config)
        from revenium_middleware.google import config
        importlib.reload(config)

        # Reload the prompt_extractor module to use the new config
        from revenium_middleware.google import prompt_extractor
        importlib.reload(prompt_extractor)

        yield

    def test_contents_as_positional_argument(self):
        """Test capturing contents from positional argument."""
        from revenium_middleware.google.prompt_extractor import extract_prompt_data_if_enabled

        args = ("What is the capital of France?",)
        kwargs = {"system_instruction": "You are a geography expert."}

        system_prompt, input_messages, output_response, truncated = extract_prompt_data_if_enabled(
            kwargs, args=args
        )

        assert system_prompt == "You are a geography expert."
        assert input_messages is not None
        assert not truncated

        # Verify it's valid JSON
        parsed = json.loads(input_messages)
        assert isinstance(parsed, list)
        assert parsed[0]['role'] == 'user'
        assert parsed[0]['parts'][0]['text'] == "What is the capital of France?"

    def test_contents_as_keyword_argument(self):
        """Test capturing contents from keyword argument."""
        from revenium_middleware.google.prompt_extractor import extract_prompt_data_if_enabled

        args = ()
        kwargs = {
            "contents": "Explain quantum physics",
            "system_instruction": "You are a physics professor."
        }

        system_prompt, input_messages, output_response, truncated = extract_prompt_data_if_enabled(
            kwargs, args=args
        )

        assert system_prompt == "You are a physics professor."
        assert input_messages is not None
        assert not truncated

        parsed = json.loads(input_messages)
        assert parsed[0]['parts'][0]['text'] == "Explain quantum physics"

    def test_long_content_with_json_safe_truncation(self):
        """Test that long content is truncated while maintaining valid JSON."""
        from revenium_middleware.google.prompt_extractor import extract_prompt_data_if_enabled

        long_text = "A" * 1000  # Much longer than our 500 char limit
        args = ()
        kwargs = {
            "contents": [
                {"role": "user", "parts": [{"text": long_text}]},
                {"role": "model", "parts": [{"text": "Response"}]},
                {"role": "user", "parts": [{"text": "Follow up"}]}
            ]
        }

        system_prompt, input_messages, output_response, truncated = extract_prompt_data_if_enabled(
            kwargs, args=args
        )

        assert truncated is True
        assert input_messages is not None

        # Verify it's valid JSON even after truncation
        parsed = json.loads(input_messages)
        assert isinstance(parsed, list)
        assert len(parsed) >= 1  # At least one message preserved

        # Check for truncation marker in the structure
        has_marker = any(
            '[TRUNCATED]' in str(part)
            for msg in parsed
            for part in msg.get('parts', [])
        )
        assert has_marker

    def test_streaming_response_with_early_truncation(self):
        """Test streaming response accumulation with truncation."""
        from revenium_middleware.google.prompt_extractor import extract_prompt_data_if_enabled

        # Simulate accumulated streaming content
        accumulated_content = "B" * 600  # Longer than limit
        kwargs = {"contents": "Test"}

        system_prompt, input_messages, output_response, truncated = extract_prompt_data_if_enabled(
            kwargs, args=(), accumulated_content=accumulated_content
        )

        assert output_response is not None
        assert len(output_response) == 500  # Truncated to max length
        assert truncated is True
        assert '[TRUNCATED]' in output_response

    def test_complex_nested_message_structure(self):
        """Test complex nested message structures are properly serialized."""
        from revenium_middleware.google.prompt_extractor import extract_prompt_data_if_enabled

        kwargs = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"text": "First part"},
                        {"text": "Second part"},
                        {"inline_data": "binary"}
                    ]
                },
                {
                    "role": "model",
                    "parts": [{"text": "Model response"}]
                }
            ],
            "system_instruction": {"parts": [{"text": "You are helpful"}]}
        }

        system_prompt, input_messages, output_response, truncated = extract_prompt_data_if_enabled(
            kwargs, args=()
        )

        assert system_prompt == "You are helpful"
        assert input_messages is not None

        # Verify JSON structure
        parsed = json.loads(input_messages)
        assert len(parsed) == 2
        assert len(parsed[0]['parts']) == 3
        assert parsed[0]['parts'][0]['text'] == "First part"
        assert parsed[0]['parts'][1]['text'] == "Second part"
        # Binary data should be sanitized to prevent leakage
        assert parsed[0]['parts'][2]['inline_data'] == "[BINARY_DATA]"

