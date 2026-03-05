"""
Pytest configuration and shared fixtures for OpenAI middleware tests.
Shared pytest fixtures for revenium-python-sdk OpenAI tests.

This module provides properly configured mocks and fixtures to ensure
tests run without making real API calls to OpenAI or Revenium.
"""

import pytest
from unittest.mock import MagicMock, patch
from revenium_middleware import shutdown_event



def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers",
        "e2e: End-to-end integration tests (require real API keys)"
    )
    config.addinivalue_line(
        "markers",
        "unit: Unit tests (fast, no external dependencies)"
    )

@pytest.fixture(autouse=True)
def reset_global_state():
    """
    Reset global state before and after each test.

    This ensures tests don't interfere with each other through shared state.
    """
    shutdown_event.clear()
    yield
    shutdown_event.clear()


@pytest.fixture(autouse=True)
def mock_revenium_client():
    """
    Automatically mock the Revenium client for all tests.

    This prevents any real API calls to Revenium during testing.
    All metering calls will be intercepted and return a success response.
    """
    with patch('revenium_middleware.client') as mock_client:
        # Configure the mock to return success for all metering calls
        mock_client.ai.create_completion.return_value = {'status': 'success'}
        yield mock_client


@pytest.fixture
def mock_chat_completion():
    """
    Create a properly configured mock ChatCompletion response.

    This fixture creates a mock that correctly handles the parse() method
    used by LangChain's with_raw_response wrapper.

    Returns:
        MagicMock: A mock ChatCompletion object with all required attributes
    """
    # Create the actual response object
    response = MagicMock()
    response.id = "chatcmpl-test-id-123"
    response.model = "gpt-4"
    response.object = "chat.completion"
    response.created = 1234567890
    response.system_fingerprint = "fp_test"

    # Configure usage
    response.usage = MagicMock()
    response.usage.prompt_tokens = 100
    response.usage.completion_tokens = 50
    response.usage.total_tokens = 150

    # Configure prompt_tokens_details for cached tokens
    response.usage.prompt_tokens_details = MagicMock()
    response.usage.prompt_tokens_details.cached_tokens = 0

    # Configure completion_tokens_details
    response.usage.completion_tokens_details = MagicMock()
    response.usage.completion_tokens_details.reasoning_tokens = 0

    # Configure choices
    choice = MagicMock()
    choice.index = 0
    choice.finish_reason = "stop"
    choice.message = MagicMock()
    choice.message.role = "assistant"
    choice.message.content = "This is a test response."
    response.choices = [choice]

    # CRITICAL FIX: Configure parse() to return self (not another MagicMock)
    response.parse = MagicMock(return_value=response)

    return response


@pytest.fixture
def mock_legacy_api_response(mock_chat_completion):
    """
    Create a mock LegacyAPIResponse that wraps a ChatCompletion.

    This is used by LangChain's with_raw_response wrapper.
    The parse() method should return the actual ChatCompletion object.

    Args:
        mock_chat_completion: The ChatCompletion fixture

    Returns:
        MagicMock: A mock LegacyAPIResponse with parse() method
    """
    legacy_response = MagicMock()
    legacy_response.parse = MagicMock(return_value=mock_chat_completion)
    return legacy_response


@pytest.fixture
def mock_embeddings_response():
    """
    Create a properly configured mock CreateEmbeddingResponse.

    Note: Embeddings responses do NOT have an 'id' attribute.

    Returns:
        MagicMock: A mock CreateEmbeddingResponse object
    """
    response = MagicMock()
    response.model = "text-embedding-3-small"
    response.object = "list"

    # Embeddings don't have an 'id' attribute - explicitly delete it
    if hasattr(response, 'id'):
        del response.id

    # Configure usage (embeddings only have prompt_tokens)
    response.usage = MagicMock()
    response.usage.prompt_tokens = 9
    response.usage.total_tokens = 9

    # Configure embedding data
    embedding_data = MagicMock()
    embedding_data.object = "embedding"
    embedding_data.index = 0
    embedding_data.embedding = [0.1] * 1536  # 1536-dimensional vector
    response.data = [embedding_data]

    # CRITICAL FIX: Configure parse() to return self (not another MagicMock)
    response.parse = MagicMock(return_value=response)

    return response


@pytest.fixture
def standard_usage_metadata():
    """
    Standard usage metadata for testing.

    Returns:
        dict: A dictionary with common usage metadata fields
    """
    return {
        "trace_id": "test-trace-123",
        "task_type": "test-task",
        "subscriber": {
            "id": "test-subscriber-456",
            "email": "test@example.com"
        },
        "organization_id": "test-org-789",
        "product_id": "test-product",
        "agent": "test-agent"
    }


@pytest.fixture
def mock_openai_client():
    """
    Create a mock OpenAI client instance.

    This is useful for testing provider detection logic.

    Returns:
        MagicMock: A mock OpenAI client
    """
    client = MagicMock()
    client.base_url = "https://api.openai.com/v1"
    client.api_key = "sk-test-key"
    return client


@pytest.fixture
def mock_azure_client():
    """
    Create a mock Azure OpenAI client instance.

    Returns:
        MagicMock: A mock Azure OpenAI client
    """
    client = MagicMock()
    client.base_url = "https://test.openai.azure.com/"
    client.api_key = "test-azure-key"
    return client

