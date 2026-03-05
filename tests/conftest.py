"""
Pytest configuration and shared fixtures for revenium-python-sdk tests.

This module provides pytest markers and fixtures to ensure tests run
properly with appropriate mocking and state management.
"""

import pytest
from unittest.mock import patch, MagicMock


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers",
        "unit: Unit tests (fast, no external dependencies)"
    )
    config.addinivalue_line(
        "markers",
        "e2e: End-to-end integration tests (require real API keys)"
    )


def pytest_collection_modifyitems(config, items):
    """Automatically mark tests based on file patterns."""
    for item in items:
        # Mark E2E tests
        if "test_e2e" in str(item.fspath) or "test_end_to_end" in str(item.fspath):
            if "e2e" not in item.keywords:
                item.add_marker(pytest.mark.e2e)
        # Mark other tests as unit tests by default
        else:
            if "unit" not in item.keywords and "e2e" not in item.keywords:
                item.add_marker(pytest.mark.unit)


@pytest.fixture(autouse=True)
def mock_revenium_client():
    """
    Automatically mock the Revenium client for all unit tests.

    This prevents any real API calls to Revenium during testing.
    All metering calls will be intercepted and return a success response.

    For E2E tests, this fixture should be overridden or disabled.
    """
    with patch('revenium_middleware._core.metering.client') as mock_client:
        # Configure the mock to return success for all metering calls
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'status': 'success'}
        mock_client.ai.create_completion.return_value = mock_response
        yield mock_client
