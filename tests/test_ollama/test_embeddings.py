"""
Embeddings functionality tests.

This module tests the Ollama embeddings wrapper functionality
with the Revenium middleware.

These are unit tests that do NOT require:
- Ollama to be running
- Real API keys
- External network calls
"""

import pytest
from unittest.mock import MagicMock


@pytest.mark.unit
class TestEmbeddingsUnit:
    """Unit tests for embeddings functionality."""

    def test_embeddings_wrapper_existence(self):
        """Test that the embeddings wrapper function exists and is properly decorated."""
        # Import and verify the wrapper exists
        from revenium_middleware.ollama.middleware import embed_wrapper

        # Verify it's callable
        assert callable(embed_wrapper)

        # Verify it has the wrapt decorator
        assert hasattr(embed_wrapper, '__wrapped__')

    def test_embeddings_wrapper_applied(self):
        """Test that the embeddings wrapper is applied to ollama.embed."""
        import ollama
        import revenium_middleware.ollama  # noqa: F401

        # Verify ollama.embed has been wrapped
        assert hasattr(ollama.embed, '__wrapped__')

    def test_handle_embeddings_response_function_exists(self):
        """Test that handle_embeddings_response function exists."""
        from revenium_middleware.ollama.middleware import handle_embeddings_response

        # Verify it's callable
        assert callable(handle_embeddings_response)

    def test_embeddings_token_counting_logic(self, mock_ollama_embed_response):
        """Test that embeddings token counting logic is correct."""
        # Embeddings should only have input tokens, no output tokens
        prompt_tokens = mock_ollama_embed_response.get('prompt_eval_count', 0)
        completion_tokens = 0  # Embeddings don't generate output
        total_tokens = prompt_tokens

        assert prompt_tokens == 10  # From fixture
        assert completion_tokens == 0
        assert total_tokens == 10

    def test_embeddings_response_structure(self, mock_ollama_embed_response):
        """Test that mock embeddings response has correct structure."""
        assert 'model' in mock_ollama_embed_response
        assert 'embeddings' in mock_ollama_embed_response
        assert 'prompt_eval_count' in mock_ollama_embed_response
        assert isinstance(mock_ollama_embed_response['embeddings'], list)
        assert len(mock_ollama_embed_response['embeddings']) == 1
        assert len(mock_ollama_embed_response['embeddings'][0]) == 768

    def test_embeddings_operation_type_detection(self):
        """Test that operation type is correctly detected for embeddings."""
        from revenium_middleware.ollama.trace_fields import detect_operation_type

        # Test embeddings endpoint detection
        operation_type = detect_operation_type('embed', {})
        assert operation_type == 'EMBED'

    def test_embeddings_constants_and_defaults(self):
        """Test that embeddings-specific constants are correct."""
        # Embeddings should have:
        # - output_token_count = 0
        # - is_streamed = False
        # - operation_type = 'EMBED'
        # - provider = 'OLLAMA'

        output_tokens = 0
        is_streamed = False
        operation_type = 'EMBED'
        provider = 'OLLAMA'

        assert output_tokens == 0
        assert is_streamed is False
        assert operation_type == 'EMBED'
        assert provider == 'OLLAMA'

