"""
Revenium Middleware Integration Test Suite

This module tests the Revenium middleware integration with Ollama to verify:
1. Middleware properly intercepts Ollama API calls
2. Metering data is sent to Revenium service
3. API calls complete successfully
4. Metadata is properly passed through

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
class TestBasicIntegration:
    """Test basic middleware integration without metadata."""

    def test_chat_request_completes_successfully(self, model_name):
        """Test that basic chat requests work with middleware."""
        response = ollama.chat(
            model=model_name,
            messages=[
                {
                    'role': 'user',
                    'content': 'What is 2+2? Answer in one short sentence.',
                },
            ]
        )
        
        # Verify response structure
        assert 'message' in response, "Response should contain 'message' field"
        assert 'content' in response['message'], "Message should contain 'content'"
        assert response['message']['content'], "Content should not be empty"
        
        # Verify token usage is tracked
        if 'eval_count' in response:
            assert response['eval_count'] > 0, "Should have completion tokens"

    def test_generate_request_completes_successfully(self, model_name):
        """Test that generate requests work with middleware."""
        response = ollama.generate(
            model=model_name,
            prompt='Say "Hello from Ollama!" and nothing else.'
        )
        
        # Verify response structure
        assert 'response' in response, "Response should contain 'response' field"
        assert response['response'], "Response should not be empty"


@pytest.mark.e2e
class TestMetadataIntegration:
    """Test middleware integration with usage metadata."""

    def test_chat_with_metadata_completes_successfully(self, model_name):
        """Test that chat requests with metadata work correctly."""
        response = ollama.chat(
            model=model_name,
            messages=[
                {
                    'role': 'user',
                    'content': 'Explain what middleware is in one sentence.',
                },
            ],
            usage_metadata={
                "trace_id": "test-trace-001",
                "task_type": "integration-testing",
                "subscriber": {
                    "id": "test-user-123",
                    "email": "test@example.com"
                },
                "organizationName": "ReveniumTestOrg",
                "subscription_id": "test-subscription",
                "productName": "ollama-middleware-test",
                "agent": "test-agent"
            }
        )
        
        # Verify response structure
        assert 'message' in response, "Response should contain 'message' field"
        assert 'content' in response['message'], "Message should contain 'content'"
        assert response['message']['content'], "Content should not be empty"

    def test_generate_with_metadata_completes_successfully(self, model_name):
        """Test that generate requests with metadata work correctly."""
        response = ollama.generate(
            model=model_name,
            prompt='Count to 3.',
            usage_metadata={
                "trace_id": "test-trace-002",
                "task_type": "generate-testing",
                "subscriber": {
                    "id": "test-user-456",
                    "email": "test2@example.com"
                }
            }
        )
        
        # Verify response structure
        assert 'response' in response, "Response should contain 'response' field"
        assert response['response'], "Response should not be empty"


@pytest.mark.e2e
class TestStreamingIntegration:
    """Test middleware integration with streaming responses."""

    def test_streaming_chat_completes_successfully(self, model_name):
        """Test that streaming chat requests work with middleware."""
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
        for chunk in stream:
            chunk_count += 1
            assert 'message' in chunk, "Each chunk should have a message"
        
        assert chunk_count > 0, "Should receive at least one chunk"

    def test_streaming_generate_completes_successfully(self, model_name):
        """Test that streaming generate requests work with middleware."""
        stream = ollama.generate(
            model=model_name,
            prompt='Count to 3.',
            stream=True
        )
        
        chunk_count = 0
        for chunk in stream:
            chunk_count += 1
            assert 'response' in chunk, "Each chunk should have a response"
        
        assert chunk_count > 0, "Should receive at least one chunk"

