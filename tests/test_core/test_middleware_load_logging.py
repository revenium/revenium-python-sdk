"""Tests for middleware load-failure logging (BACK-778).

When a provider's own SDK is installed but the Revenium middleware fails to
import, the failure must be logged at ERROR with the underlying exception so
customers can see WHICH dependency is broken. When the provider SDK itself is
absent, the quiet DEBUG log is kept — optional providers are expected to be
missing (e.g. `revenium_middleware.griptape` imports several provider
middlewares eagerly).
"""

import importlib
import logging
import sys

import pytest

from revenium_middleware._core.load_diagnostics import log_middleware_load_failure


def test_load_failure_with_sdk_installed_logs_error(caplog):
    exc = ImportError("cannot import name 'merge_metadata'")
    with caplog.at_level(logging.ERROR, logger="revenium_middleware"):
        # `openai` is installed in the test environment.
        log_middleware_load_failure("OpenAI", exc, required_packages=("openai",))
    error_records = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert len(error_records) == 1
    message = error_records[0].getMessage()
    assert "merge_metadata" in message
    assert "OpenAI" in message
    assert "NOT" in message


def test_load_failure_with_sdk_missing_logs_debug(caplog):
    exc = ImportError("No module named 'a_package_that_does_not_exist'")
    with caplog.at_level(logging.DEBUG, logger="revenium_middleware"):
        log_middleware_load_failure(
            "Imaginary", exc, required_packages=("a_package_that_does_not_exist",)
        )
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]
    debug_records = [r for r in caplog.records if r.levelno == logging.DEBUG]
    assert any("Imaginary" in r.getMessage() for r in debug_records)


def test_openai_init_logs_error_when_middleware_import_breaks(caplog):
    """Reload revenium_middleware.openai with `wrapt` blocked: openai IS
    installed, so the load failure must surface at ERROR (Run8.io scenario)."""
    import revenium_middleware.openai as openai_mw

    original_wrapt = sys.modules.get("wrapt")
    sys.modules["wrapt"] = None  # forces `import wrapt` to raise ImportError
    try:
        with caplog.at_level(logging.ERROR, logger="revenium_middleware"):
            importlib.reload(openai_mw)
        assert openai_mw.create_wrapper is None
        error_records = [r for r in caplog.records if r.levelno == logging.ERROR]
        assert any(
            "OpenAI" in r.getMessage() and "NOT" in r.getMessage()
            for r in error_records
        )
    finally:
        if original_wrapt is not None:
            sys.modules["wrapt"] = original_wrapt
        else:
            del sys.modules["wrapt"]
        importlib.reload(openai_mw)
        assert openai_mw.create_wrapper is not None
