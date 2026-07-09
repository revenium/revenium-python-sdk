"""The metering client must be (re)initializable after import.

The client was a module-level singleton snapshotted from the environment once
at import time: env vars populated later (vault bootstraps, Django settings,
import-order accidents) were silently ignored forever, and there was no
programmatic configuration API at all.
"""
from unittest.mock import MagicMock

import pytest

import revenium_middleware
from revenium_middleware import _core
from revenium_middleware._core import metering, metering_submission


@pytest.fixture()
def unconfigured_client(monkeypatch):
    """Simulate a process whose import happened before env vars were set."""
    monkeypatch.delenv("REVENIUM_METERING_API_KEY", raising=False)
    monkeypatch.delenv("REVENIUM_METERING_BASE_URL", raising=False)
    for module in (metering, metering_submission, _core, revenium_middleware):
        if hasattr(module, "client"):
            monkeypatch.setattr(module, "client", None)
    monkeypatch.setattr(metering, "_last_failed_key", None)
    return monkeypatch


def test_env_var_set_after_import_reaches_submit(unconfigured_client):
    """The audit repro: key exported after import must not disable metering forever."""
    assert metering_submission.submit_ai_event("completion", {}) is None

    unconfigured_client.setenv("REVENIUM_METERING_API_KEY", "hak_set_after_import")

    fake_client = MagicMock()
    fake_client.ai.create_completion.return_value = "metered"
    unconfigured_client.setattr(
        metering, "_build_metering_client", lambda key, url: fake_client
    )

    assert metering_submission.submit_ai_event("completion", {}) == "metered"


def test_initialize_metering_builds_client_programmatically(unconfigured_client):
    fake_client = MagicMock()
    unconfigured_client.setattr(
        metering, "_build_metering_client", lambda key, url: fake_client
    )

    assert metering.initialize_metering(api_key="hak_program", base_url="https://example.test") is True
    assert metering.get_client() is fake_client
    # Dynamic attribute readers (e.g. anthropic's thread-safe lookup) see it too.
    assert revenium_middleware.client is fake_client


def test_initialize_metering_without_key_disables_metering(unconfigured_client):
    assert metering.initialize_metering() is False
    assert metering.get_client() is None
    assert metering_submission.submit_ai_event("completion", {}) is None


def test_get_client_does_not_rebuild_when_env_still_missing(unconfigured_client):
    calls = []

    def counting_build(key, url):
        calls.append(key)
        return None

    unconfigured_client.setattr(metering, "_build_metering_client", counting_build)

    assert metering.get_client() is None
    assert metering.get_client() is None
    # No key in the environment: nothing to (re)build from.
    assert calls == []


def test_initialize_metering_is_exported_publicly():
    assert "initialize_metering" in revenium_middleware.__all__
    assert callable(revenium_middleware.initialize_metering)
