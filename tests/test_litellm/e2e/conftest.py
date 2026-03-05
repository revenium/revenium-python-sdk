"""
E2E tests require real API keys and SDK dependencies.
Skip collection when running unit tests.
"""
import pytest

collect_ignore_glob = ["test_*.py"]
