"""
Basic Ollama functionality tests.

This module tests that Ollama is properly installed and working,
independent of the Revenium middleware.

These are end-to-end tests that require:
- Ollama to be running locally
- A model to be available (e.g., qwen2.5:0.5b)
- REVENIUM_METERING_API_KEY to be set
"""

import pytest
import ollama


@pytest.mark.e2e
class TestBasicOllama:
    """Test basic Ollama functionality without middleware."""

    def test_ollama_chat_works(self, setup_e2e_test, model_name):
        """Test that basic Ollama chat functionality works."""
        response = ollama.chat(
            model=model_name,
            messages=[
                {
                    'role': 'user',
                    'content': 'Say hello in one sentence.',
                },
            ]
        )

        assert 'message' in response, "Response should contain 'message' field"
        assert 'content' in response['message'], "Message should contain 'content'"
        assert response['message']['content'], "Content should not be empty"

        # Verify token usage is available
        if 'eval_count' in response:
            assert response['eval_count'] > 0, "Should have completion tokens"

    def test_ollama_generate_works(self, setup_e2e_test, model_name):
        """Test that basic Ollama generate functionality works."""
        response = ollama.generate(
            model=model_name,
            prompt='Say hello in one sentence.'
        )

        assert 'response' in response, "Response should contain 'response' field"
        assert response['response'], "Response should not be empty"

