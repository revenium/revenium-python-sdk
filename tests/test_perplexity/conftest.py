"""
Pytest configuration and shared fixtures for Revenium Perplexity middleware tests.
"""
import os
import pytest
from unittest.mock import MagicMock, patch
from revenium_middleware import shutdown_event


@pytest.fixture(autouse=True)
def reset_global_state():
    """
    Reset global state before and after each test.
    """
    shutdown_event.clear()
    yield
    shutdown_event.clear()


@pytest.fixture(autouse=True)
def mock_revenium_client():
    """
    Automatically mock the Revenium client for all tests.
    
    This prevents any real API calls to Revenium during testing.
    """
    with patch('revenium_middleware.client') as mock_client:
        mock_client.ai.create_completion.return_value = {'status': 'success'}
        yield mock_client


@pytest.fixture(scope="function")
def setup_e2e_test():
    """
    Configure environment variables for end-to-end tests.
    
    Required environment variables:
        - REVENIUM_METERING_API_KEY: Your Revenium API key (REQUIRED)
        - PERPLEXITY_API_KEY: Your Perplexity API key (REQUIRED)
    """
    if not os.environ.get('REVENIUM_METERING_API_KEY'):
        pytest.skip(
            "REVENIUM_METERING_API_KEY environment variable is required "
            "for e2e tests."
        )
    
    if not os.environ.get('PERPLEXITY_API_KEY'):
        pytest.skip(
            "PERPLEXITY_API_KEY environment variable is required "
            "for e2e tests."
        )
    
    if not os.environ.get('REVENIUM_LOG_LEVEL'):
        os.environ['REVENIUM_LOG_LEVEL'] = 'INFO'
    
    yield


@pytest.fixture
def mock_openai_response():
    """
    Create a properly configured mock OpenAI/Perplexity response.
    """
    response = MagicMock()
    response.id = "chatcmpl-test-123"
    response.model = "sonar-pro"

    # Create message mock with tool_calls explicitly set to None
    message = MagicMock()
    message.role = "assistant"
    message.content = "This is a test response."
    message.tool_calls = None  # Explicitly set to None to avoid MagicMock auto-creation

    response.choices = [
        MagicMock(
            message=message,
            finish_reason="stop"
        )
    ]
    response.usage = MagicMock(
        prompt_tokens=10,
        completion_tokens=20,
        total_tokens=30
    )
    return response


@pytest.fixture
def standard_usage_metadata():
    """
    Standard usage metadata for testing.
    """
    return {
        "organization_id": "test-org-123",
        "product_id": "test-product",
        "subscription_id": "test-subscription",
        "subscriber": {
            "id": "test-user-456",
            "email": "test@example.com"
        },
        "task_type": "test-task",
        "trace_id": "test-trace-789"
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

