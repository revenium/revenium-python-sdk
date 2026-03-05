"""
Pytest configuration for Anthropic middleware tests.
"""

import pytest


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "e2e: marks tests as end-to-end tests (may require API keys)"
    )

