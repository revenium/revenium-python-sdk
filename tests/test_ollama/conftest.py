"""
Pytest configuration and shared fixtures for Revenium Ollama middleware tests.

This module provides properly configured mocks and fixtures to ensure
tests run without making real API calls to Ollama or Revenium.
"""

import os
import pytest
from unittest.mock import MagicMock, patch
from revenium_middleware import shutdown_event


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


@pytest.fixture(scope="function")
def setup_e2e_test():
    """
    Configure environment variables for end-to-end tests.

    This fixture validates that required environment variables are set
    for e2e tests that need to connect to real services.

    Required environment variables:
        - REVENIUM_METERING_API_KEY: Your Revenium API key (REQUIRED)

    Optional environment variables (with defaults):
        - REVENIUM_METERING_BASE_URL: Uses package default
        - REVENIUM_LOG_LEVEL: Defaults to INFO
    """
    # Validate required environment variables for e2e tests
    if not os.environ.get('REVENIUM_METERING_API_KEY'):
        pytest.skip(
            "REVENIUM_METERING_API_KEY environment variable is required "
            "for e2e tests. Please set it in your environment "
            "or .env file. See tests/README.md for setup instructions."
        )

    # Set default log level if not already set
    if not os.environ.get('REVENIUM_LOG_LEVEL'):
        os.environ['REVENIUM_LOG_LEVEL'] = 'INFO'

    yield


@pytest.fixture(scope="session")
def model_name():
    """
    Return the Ollama model name to use for testing.

    Using a small, fast model for testing to minimize resource usage
    and test execution time.
    """
    return 'qwen2.5:0.5b'


@pytest.fixture(scope="session")
def test_metadata():
    """
    Return sample metadata for testing metadata propagation.

    This metadata can be used in tests that verify metadata is properly
    passed through to the Revenium metering service.
    """
    return {
        "trace_id": "pytest-trace-001",
        "task_type": "automated-testing",
        "subscriber": {
            "id": "pytest-user-123",
            "email": "pytest@example.com"
        },
        "organizationName": "PytestTestOrg",
        "subscription_id": "pytest-subscription",
        "productName": "ollama-middleware-pytest",
        "agent": "pytest-agent"
    }


@pytest.fixture
def mock_ollama_chat_response():
    """
    Create a properly configured mock Ollama chat response.

    Returns:
        dict: A mock Ollama chat response with all required fields
    """
    response = {
        'model': 'qwen2.5:0.5b',
        'message': {
            'role': 'assistant',
            'content': 'This is a test response.'
        },
        'done': True,
        'done_reason': 'stop',
        'prompt_eval_count': 10,
        'eval_count': 20,
        'total_duration': 1000000000,
        'load_duration': 100000000,
        'prompt_eval_duration': 200000000,
        'eval_duration': 700000000
    }
    return response


@pytest.fixture
def mock_ollama_embed_response():
    """
    Create a properly configured mock Ollama embeddings response.

    Returns:
        dict: A mock Ollama embeddings response with all required fields
    """
    response = {
        'model': 'nomic-embed-text',
        'embeddings': [[0.1] * 768],  # 768-dimensional embedding
        'prompt_eval_count': 10,
        'total_duration': 500000000,
        'load_duration': 50000000
    }
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
        "organizationName": "TestOrg789",
        "subscription_id": "test-subscription",
        "productName": "test-product",
        "agent": "test-agent"
    }


def pytest_configure(config):
    """
    Configure pytest with custom markers.
    """
    config.addinivalue_line(
        "markers",
        "e2e: End-to-end integration tests (require real API keys and services)"
    )
    config.addinivalue_line(
        "markers",
        "unit: Unit tests (fast, no external dependencies)"
    )
