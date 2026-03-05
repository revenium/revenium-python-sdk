"""
Test suite for transaction ID exposure in Revenium Ollama middleware.

This module tests that transaction IDs are properly exposed in response objects
for both chat() and generate() functions, in both streaming and non-streaming modes.

These are end-to-end tests that require:
- Ollama to be running locally
- A valid Revenium API key
- An Ollama model to be available
"""

import pytest
import ollama
import revenium_middleware.ollama

# Note: Environment setup and model_name fixture are provided by conftest.py


@pytest.mark.e2e
class TestTransactionIDNonStreaming:
    """Test transaction ID exposure in non-streaming responses."""

    def test_chat_non_streaming_has_transaction_id(self, model_name):
        """Test that non-streaming chat() responses include transaction ID."""
        response = ollama.chat(
            model=model_name,
            messages=[
                {
                    'role': 'user',
                    'content': 'Say hello in one word.',
                },
            ]
        )

        # Check for transaction ID in response
        transaction_id = getattr(response, '_revenium_transaction_id', None)

        assert transaction_id is not None, "Transaction ID should be present"
        assert transaction_id.startswith('ollama-'), (
            f"Transaction ID should start with 'ollama-', got: {transaction_id}"
        )

    def test_generate_non_streaming_has_transaction_id(self, model_name):
        """Test that non-streaming generate() responses include transaction ID."""
        response = ollama.generate(
            model=model_name,
            prompt='Say hello in one word.'
        )

        # Check for transaction ID in response
        transaction_id = getattr(response, '_revenium_transaction_id', None)

        assert transaction_id is not None, "Transaction ID should be present"
        assert transaction_id.startswith('ollama-'), (
            f"Transaction ID should start with 'ollama-', got: {transaction_id}"
        )


@pytest.mark.e2e
class TestTransactionIDStreaming:
    """Test transaction ID exposure in streaming responses."""

    def test_chat_streaming_has_transaction_id(self, model_name):
        """Test that streaming chat() responses include transaction ID in all chunks."""
        stream = ollama.chat(
            model=model_name,
            messages=[
                {
                    'role': 'user',
                    'content': 'Count to 3.',
                },
            ],
            stream=True
        )

        chunk_count = 0
        last_chunk = None
        transaction_ids = []

        for chunk in stream:
            chunk_count += 1
            last_chunk = chunk

            # Check for transaction ID in each chunk
            tid = getattr(chunk, '_revenium_transaction_id', None)

            if tid:
                transaction_ids.append(tid)

        assert chunk_count > 0, "Should receive at least one chunk"
        assert last_chunk is not None, "Should have a last chunk"

        # Check last chunk has transaction ID
        transaction_id = getattr(last_chunk, '_revenium_transaction_id', None)
        assert transaction_id is not None, "Transaction ID should be present in chunks"
        assert transaction_id.startswith('ollama-'), (
            f"Transaction ID should start with 'ollama-', got: {transaction_id}"
        )

        # Check if all chunks have the same transaction ID
        if transaction_ids:
            unique_ids = set(transaction_ids)
            assert len(unique_ids) == 1, (
                f"All chunks should have same transaction ID, got: {unique_ids}"
            )

    def test_generate_streaming_has_transaction_id(self, model_name):
        """Test that streaming generate() responses include transaction ID in all chunks."""
        stream = ollama.generate(
            model=model_name,
            prompt='Count to 3.',
            stream=True
        )

        chunk_count = 0
        last_chunk = None
        transaction_ids = []

        for chunk in stream:
            chunk_count += 1
            last_chunk = chunk

            # Check for transaction ID in each chunk
            tid = getattr(chunk, '_revenium_transaction_id', None)

            if tid:
                transaction_ids.append(tid)

        assert chunk_count > 0, "Should receive at least one chunk"
        assert last_chunk is not None, "Should have a last chunk"

        # Check last chunk has transaction ID
        transaction_id = getattr(last_chunk, '_revenium_transaction_id', None)
        assert transaction_id is not None, "Transaction ID should be present in chunks"
        assert transaction_id.startswith('ollama-'), (
            f"Transaction ID should start with 'ollama-', got: {transaction_id}"
        )

        # Check if all chunks have the same transaction ID
        if transaction_ids:
            unique_ids = set(transaction_ids)
            assert len(unique_ids) == 1, (
                f"All chunks should have same transaction ID, got: {unique_ids}"
            )


@pytest.mark.e2e
class TestBackwardCompatibility:
    """Test that existing code continues to work without modifications."""

    def test_traditional_response_access_still_works(self, model_name):
        """Test that traditional response access patterns are not broken."""
        # This is how users currently use the middleware
        response = ollama.chat(
            model=model_name,
            messages=[{'role': 'user', 'content': 'Hi'}]
        )

        # Access response in the traditional way
        content = response['message']['content']
        tokens = response.get('eval_count', 0)

        assert content is not None, "Should be able to access message content"
        assert isinstance(tokens, int), "Should be able to access token count"
        assert tokens >= 0, "Token count should be non-negative"
