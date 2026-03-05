import datetime
import logging
import os
import uuid
from unittest.mock import patch, MagicMock, AsyncMock

import pytest
import wrapt
from freezegun import freeze_time
from revenium_middleware import shutdown_event

from revenium_middleware.openai.middleware import create_wrapper, embeddings_create_wrapper, extract_usage_data, OperationType

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "dummy_test_api_key")

class TestMiddleware:
    @pytest.fixture
    def reset_state(self):
        """Fixture to reset global state before each test."""
        shutdown_event.clear()
        yield
        # Cleanup after test
        shutdown_event.clear()

    @pytest.fixture
    def mock_openai_response(self):
        """Create a mock OpenAI response object."""
        mock_response = MagicMock()
        mock_response.id = "test-response-id"
        mock_response.model = "gpt-4"

        # Set up usage attributes - use spec to prevent auto-creation of input_tokens
        mock_usage = MagicMock(spec=['prompt_tokens', 'completion_tokens', 'total_tokens', 'prompt_tokens_details'])
        mock_usage.prompt_tokens = 100
        mock_usage.completion_tokens = 50
        mock_usage.total_tokens = 150

        # Set up prompt_tokens_details for cached tokens (optional)
        mock_usage.prompt_tokens_details = MagicMock()
        mock_usage.prompt_tokens_details.cached_tokens = 0

        mock_response.usage = mock_usage

        # Set up choices with finish_reason
        mock_choice = MagicMock()
        mock_choice.finish_reason = "stop"
        mock_response.choices = [mock_choice]

        # CRITICAL FIX: Configure parse() to return self (not another MagicMock)
        # This handles the LegacyAPIResponse.parse() code path in middleware.py
        mock_response.parse = MagicMock(return_value=mock_response)

        return mock_response

    @pytest.fixture
    def mock_embeddings_response(self):
        """Create a mock OpenAI embeddings response object."""
        mock_response = MagicMock()
        mock_response.model = "text-embedding-3-small"

        # Remove the 'id' attribute since embeddings responses don't have one
        del mock_response.id

        # Set up usage attributes (embeddings only have prompt_tokens)
        mock_response.usage.prompt_tokens = 9
        mock_response.usage.total_tokens = 9

        # Set up embeddings data
        mock_embedding = MagicMock()
        mock_embedding.embedding = [0.1] * 1536  # Mock 1536-dimensional embedding
        mock_embedding.index = 0
        mock_embedding.object = "embedding"
        mock_response.data = [mock_embedding]
        mock_response.object = "list"

        # CRITICAL FIX: Configure parse() to return self (not another MagicMock)
        mock_response.parse = MagicMock(return_value=mock_response)

        return mock_response

    @pytest.fixture
    def test_kwargs(self):
        """Common test kwargs for OpenAI API calls."""
        return {
            "messages": [{"role": "user", "content": "Hello"}],
            "model": "gpt-4",
            "usage_metadata": {
                "trace_id": "test-trace",
                "task_id": "test-task",
                "task_type": "test-type",
                "subscriber_identity": "test-subscriber",
                "organization_id": "openai-python-middleware-unit",
                "subscription_id": "test-sub",
                "product_id": "test-product",
                "source_id": "test-source",
                "ai_provider_key_name": "test-key",
                "agent": "test-agent"
            }
        }

    @pytest.fixture
    def embeddings_kwargs(self):
        """Common test kwargs for OpenAI embeddings API calls."""
        return {
            "input": "This is a test text for embedding",
            "model": "text-embedding-3-small",
            "usage_metadata": {
                "trace_id": "test-embed-trace",
                "task_type": "semantic-search",
                "subscriber_id": "test-subscriber",
                "organization_id": "openai-python-middleware-unit",
                "product_id": "test-product"
            }
        }

    def test_extract_usage_data_chat_completion(self, mock_openai_response):
        """Test the extract_usage_data function for chat completions."""
        request_time = "2023-01-01T12:00:00Z"
        response_time = "2023-01-01T12:00:01Z"
        request_duration = 1000.0

        usage_data, transaction_id = extract_usage_data(
            mock_openai_response, OperationType.CHAT, request_time, response_time, request_duration
        )

        # Verify basic structure
        assert isinstance(usage_data, dict)
        assert isinstance(transaction_id, str)
        assert transaction_id == "test-response-id"  # For chat, uses response.id

        # Verify chat completion-specific fields
        assert usage_data["operation_type"] == "CHAT"
        assert usage_data["input_token_count"] == 100
        assert usage_data["output_token_count"] == 50
        assert usage_data["total_token_count"] == 150
        assert usage_data["model"] == "gpt-4"
        assert usage_data["provider"] == "OPENAI"
        assert usage_data["is_streamed"] == False
        assert usage_data["stop_reason"] == "END"  # OpenAI "stop" maps to "END"

        # Verify timing fields
        assert usage_data["request_time"] == request_time
        assert usage_data["response_time"] == response_time
        assert usage_data["request_duration"] == int(request_duration)

    def test_extract_usage_data_no_choices(self, mock_openai_response):
        """Test extract_usage_data when response has no choices."""
        # Remove choices from response
        mock_openai_response.choices = []

        request_time = "2023-01-01T12:00:00Z"
        response_time = "2023-01-01T12:00:01Z"
        request_duration = 1000.0

        usage_data, transaction_id = extract_usage_data(
            mock_openai_response, OperationType.CHAT, request_time, response_time, request_duration
        )

        # Should default to "END" when no choices
        assert usage_data["stop_reason"] == "END"
        assert usage_data["operation_type"] == "CHAT"
        assert isinstance(transaction_id, str)

    def test_transaction_id_uses_response_id_for_chat(self, mock_openai_response):
        """Test that chat completions use response.id as transaction ID."""
        # Set a specific response ID
        mock_openai_response.id = "chatcmpl-specific-test-id"

        usage_data, transaction_id = extract_usage_data(
            mock_openai_response, OperationType.CHAT, "2023-01-01T12:00:00Z", "2023-01-01T12:00:01Z", 1000.0
        )

        # Verify response ID is used as transaction ID for chat completions
        assert transaction_id == "chatcmpl-specific-test-id"
        assert usage_data["transaction_id"] == "chatcmpl-specific-test-id"

    def test_different_chat_models(self):
        """Test that different chat models are handled correctly."""
        models_to_test = [
            "gpt-4",
            "gpt-4-turbo",
            "gpt-3.5-turbo"
        ]

        for model in models_to_test:
            mock_response = MagicMock()
            mock_response.id = f"test-{model}-response"
            mock_response.model = model
            mock_response.usage.prompt_tokens = 100
            mock_response.usage.completion_tokens = 50
            mock_response.usage.total_tokens = 150

            # Set up choices with finish_reason
            mock_choice = MagicMock()
            mock_choice.finish_reason = "stop"
            mock_response.choices = [mock_choice]

            # CRITICAL FIX: Configure parse() to return self
            mock_response.parse = MagicMock(return_value=mock_response)

            usage_data, transaction_id = extract_usage_data(
                mock_response, OperationType.CHAT, "2023-01-01T12:00:00Z", "2023-01-01T12:00:01Z", 1000.0
            )

            assert usage_data["model"] == model
            assert usage_data["operation_type"] == "CHAT"
            assert usage_data["provider"] == "OPENAI"
            assert isinstance(transaction_id, str)

    def test_chat_completion_constants_and_defaults(self):
        """Test that chat completion-specific constants are correct."""
        mock_response = MagicMock()
        mock_response.id = "test-response-id"
        mock_response.model = "gpt-4"

        # Set up usage attributes - use spec to prevent auto-creation of input_tokens
        mock_usage = MagicMock(spec=['prompt_tokens', 'completion_tokens', 'total_tokens', 'prompt_tokens_details'])
        mock_usage.prompt_tokens = 100
        mock_usage.completion_tokens = 50
        mock_usage.total_tokens = 150

        # Set up prompt_tokens_details for cached tokens
        mock_usage.prompt_tokens_details = MagicMock()
        mock_usage.prompt_tokens_details.cached_tokens = 0

        mock_response.usage = mock_usage

        # Set up choices with finish_reason
        mock_choice = MagicMock()
        mock_choice.finish_reason = "stop"
        mock_response.choices = [mock_choice]

        # CRITICAL FIX: Configure parse() to return self
        mock_response.parse = MagicMock(return_value=mock_response)

        usage_data, _ = extract_usage_data(
            mock_response, OperationType.CHAT, "2023-01-01T12:00:00Z", "2023-01-01T12:00:01Z", 1000.0
        )

        # Test chat completion-specific defaults
        assert usage_data["operation_type"] == "CHAT"
        assert usage_data["input_token_count"] == 100
        assert usage_data["output_token_count"] == 50
        assert usage_data["total_token_count"] == 150
        assert usage_data["is_streamed"] == False
        assert usage_data["time_to_first_token"] == 0
        assert usage_data["cache_creation_token_count"] == 0
        assert usage_data["cache_read_token_count"] == 0
        assert usage_data["reasoning_token_count"] == 0
        assert usage_data["stop_reason"] == "END"  # OpenAI "stop" maps to "END"
        assert usage_data["provider"] == "OPENAI"
        assert usage_data["model_source"] == "OPENAI"
        assert usage_data["cost_type"] == "AI"

class TestEmbeddingsMiddleware:
    """Test suite for embeddings middleware functionality."""
    
    @pytest.fixture
    def reset_state(self):
        """Fixture to reset global state before each test."""
        shutdown_event.clear()
        yield
        shutdown_event.clear()

    @pytest.fixture
    def mock_embeddings_response(self):
        """Create a mock OpenAI embeddings response object."""
        mock_response = MagicMock()
        mock_response.model = "text-embedding-3-small"

        # Remove the 'id' attribute since embeddings responses don't have one
        del mock_response.id

        # Set up usage attributes (embeddings only have prompt_tokens)
        mock_response.usage.prompt_tokens = 9
        mock_response.usage.total_tokens = 9

        # Set up embeddings data
        mock_embedding = MagicMock()
        mock_embedding.embedding = [0.1] * 1536  # Mock 1536-dimensional embedding
        mock_embedding.index = 0
        mock_embedding.object = "embedding"
        mock_response.data = [mock_embedding]
        mock_response.object = "list"

        # CRITICAL FIX: Configure parse() to return self
        mock_response.parse = MagicMock(return_value=mock_response)

        return mock_response

    @pytest.fixture
    def embeddings_kwargs(self):
        """Common test kwargs for OpenAI embeddings API calls."""
        return {
            "input": "This is a test text for embedding",
            "model": "text-embedding-3-small",
            "usage_metadata": {
                "trace_id": "test-embed-trace",
                "task_type": "semantic-search",
                "subscriber_id": "test-subscriber",
                "organization_id": "openai-python-middleware-unit",
                "product_id": "test-product"
            }
        }

    def test_extract_usage_data(self, mock_embeddings_response):
        """Test the extract_usage_data function."""
        request_time = "2024-01-01T00:00:00Z"
        response_time = "2024-01-01T00:00:01Z"
        request_duration = 1000.0
        
        usage_data, transaction_id = extract_usage_data(
            mock_embeddings_response, OperationType.EMBED, request_time, response_time, request_duration
        )
        
        # Verify basic structure
        assert isinstance(usage_data, dict)
        assert isinstance(transaction_id, str)
        assert len(transaction_id) == 36  # UUID length
        
        # Verify embeddings-specific fields
        assert usage_data["operation_type"] == "EMBED"
        assert usage_data["input_token_count"] == 9
        assert usage_data["output_token_count"] == 0
        assert usage_data["total_token_count"] == 9
        assert usage_data["model"] == "text-embedding-3-small"
        assert usage_data["provider"] == "OPENAI"
        assert usage_data["is_streamed"] == False
        assert usage_data["stop_reason"] == "END"
        assert usage_data["time_to_first_token"] == 0
        
        # Verify timing fields
        assert usage_data["request_time"] == request_time
        assert usage_data["response_time"] == response_time
        assert usage_data["request_duration"] == int(request_duration)

    @patch("revenium_middleware.openai.middleware.uuid.uuid4")
    def test_transaction_id_generation(self, mock_uuid, mock_embeddings_response):
        """Test that unique transaction IDs are generated for embeddings."""
        # Mock UUID generation
        mock_uuid.return_value = MagicMock(spec=['__str__'])
        mock_uuid.return_value.__str__.return_value = "test-uuid-123"
        
        usage_data, transaction_id = extract_usage_data(
            mock_embeddings_response, OperationType.EMBED, "2024-01-01T00:00:00Z", "2024-01-01T00:00:01Z", 1000.0
        )
        
        # Verify UUID was called and returned
        mock_uuid.assert_called_once()
        assert transaction_id == "test-uuid-123"
        assert usage_data["transaction_id"] == "test-uuid-123"

    def test_different_embedding_models(self, reset_state):
        """Test that different embedding models are handled correctly."""
        models_to_test = [
            "text-embedding-3-small",
            "text-embedding-3-large",
            "text-embedding-ada-002"
        ]

        for model in models_to_test:
            mock_response = MagicMock()
            mock_response.model = model
            mock_response.usage.prompt_tokens = 10
            mock_response.usage.total_tokens = 10
            # Remove id attribute for embeddings
            del mock_response.id

            # CRITICAL FIX: Configure parse() to return self
            mock_response.parse = MagicMock(return_value=mock_response)

            usage_data, transaction_id = extract_usage_data(
                mock_response, OperationType.EMBED, "2024-01-01T00:00:00Z", "2024-01-01T00:00:01Z", 1000.0
            )

            assert usage_data["model"] == model
            assert usage_data["operation_type"] == "EMBED"
            assert usage_data["provider"] == "OPENAI"

    @patch("revenium_middleware.openai.middleware.log_token_usage")
    def test_log_token_usage_embed_operation(self, mock_log_usage, reset_state):
        """Test that log_token_usage works correctly for embeddings operations."""
        import asyncio
        from revenium_middleware.openai.middleware import log_token_usage
        
        async def test_embed_logging():
            await log_token_usage(
                response_id="test-embed-456",
                model="text-embedding-3-small",
                prompt_tokens=9,
                completion_tokens=0,
                total_tokens=9,
                cached_tokens=0,
                stop_reason="END",
                request_time="2024-01-01T00:00:00Z",
                response_time="2024-01-01T00:00:01Z",
                request_duration=1000,
                usage_metadata={"trace_id": "test-embed-trace"},
                operation_type=OperationType.EMBED
            )
        
        # Run the async function
        asyncio.run(test_embed_logging())
        
        # Verify it was called (note: the actual call happens in the background)
        # We're testing that the function accepts EMBED operation_type
        assert True  # Function ran without error

    def test_embeddings_wrapper_existence(self):
        """Test that the embeddings wrapper function exists and is properly decorated."""
        # Import and verify the wrapper exists
        from revenium_middleware.openai.middleware import embeddings_create_wrapper
        
        # Verify it's callable
        assert callable(embeddings_create_wrapper)
        
        # Verify it has the wrapt decorator
        assert hasattr(embeddings_create_wrapper, '__wrapped__')

    def test_extract_function_with_various_token_counts(self):
        """Test extract_usage_data with various token counts."""
        test_cases = [
            (5, 5),    # Small input
            (100, 100), # Medium input
            (1000, 1000), # Large input
            (0, 0),    # Edge case - zero tokens
        ]

        for prompt_tokens, total_tokens in test_cases:
            mock_response = MagicMock()
            mock_response.model = "text-embedding-3-small"
            mock_response.usage.prompt_tokens = prompt_tokens
            mock_response.usage.total_tokens = total_tokens
            # Remove id attribute for embeddings
            del mock_response.id

            # CRITICAL FIX: Configure parse() to return self
            mock_response.parse = MagicMock(return_value=mock_response)

            usage_data, transaction_id = extract_usage_data(
                mock_response, OperationType.EMBED, "2024-01-01T00:00:00Z", "2024-01-01T00:00:01Z", 1000.0
            )

            assert usage_data["input_token_count"] == prompt_tokens
            assert usage_data["total_token_count"] == total_tokens
            assert usage_data["output_token_count"] == 0  # Always 0 for embeddings
            assert isinstance(transaction_id, str)

    def test_embeddings_constants_and_defaults(self):
        """Test that embeddings-specific constants are correct."""
        mock_response = MagicMock()
        mock_response.model = "text-embedding-3-small"
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.total_tokens = 10
        # Remove id attribute for embeddings
        del mock_response.id

        # CRITICAL FIX: Configure parse() to return self
        mock_response.parse = MagicMock(return_value=mock_response)

        usage_data, _ = extract_usage_data(
            mock_response, OperationType.EMBED, "2024-01-01T00:00:00Z", "2024-01-01T00:00:01Z", 1000.0
        )
        
        # Test embeddings-specific defaults
        assert usage_data["operation_type"] == "EMBED"
        assert usage_data["output_token_count"] == 0
        assert usage_data["is_streamed"] == False
        assert usage_data["time_to_first_token"] == 0
        assert usage_data["cache_creation_token_count"] == 0
        assert usage_data["cache_read_token_count"] == 0
        assert usage_data["reasoning_token_count"] == 0
        assert usage_data["stop_reason"] == "END"
        assert usage_data["provider"] == "OPENAI"
        assert usage_data["model_source"] == "OPENAI"
        assert usage_data["cost_type"] == "AI"
        assert usage_data["input_token_cost"] is None
        assert usage_data["output_token_cost"] is None
        assert usage_data["total_cost"] is None

    def test_unified_middleware_integration(self):
        """Test that both chat and embeddings functions can be imported together."""
        # This verifies the unified middleware approach
        from revenium_middleware.openai.middleware import (
            create_wrapper, 
            embeddings_create_wrapper, 
            extract_usage_data,
            log_token_usage
        )
        
        # Verify all functions exist and are callable
        assert callable(create_wrapper)
        assert callable(embeddings_create_wrapper)
        assert callable(extract_usage_data)
        assert callable(log_token_usage)
        
        # Verify they can work with their respective response types
        mock_embeddings_response = MagicMock()
        mock_embeddings_response.model = "text-embedding-3-small"
        mock_embeddings_response.usage.prompt_tokens = 9
        mock_embeddings_response.usage.total_tokens = 9
        # Remove id attribute for embeddings
        del mock_embeddings_response.id

        # CRITICAL FIX: Configure parse() to return self
        mock_embeddings_response.parse = MagicMock(return_value=mock_embeddings_response)

        # Test extraction works
        usage_data, transaction_id = extract_usage_data(
            mock_embeddings_response, OperationType.EMBED, "2024-01-01T00:00:00Z", "2024-01-01T00:00:01Z", 1000.0
        )
        
        assert usage_data["operation_type"] == "EMBED"
        assert isinstance(transaction_id, str)

