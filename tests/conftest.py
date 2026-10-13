"""
Pytest configuration and shared fixtures for revenium-python-sdk tests.

This module provides pytest markers and fixtures to ensure tests run
properly with appropriate mocking and state management.
"""

import os
import sys
import threading
import traceback

import pytest
from unittest.mock import patch, MagicMock

try:
    import freezegun
except ImportError:
    freezegun = None

# freezegun reads every attribute of every loaded module when a freeze starts.
# cohere builds its pydantic types lazily on attribute access, so that read
# imports them mid-freeze, they fail against FakeDatetime, and the failed start
# leaves the clock frozen for the rest of the run.
if freezegun is not None:
    freezegun.configure(extend_ignore_list=["cohere"])


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.uncaptured_stderr = os.fdopen(os.dup(sys.stderr.fileno()), "w")
    config.addinivalue_line(
        "markers",
        "unit: Unit tests (fast, no external dependencies)"
    )
    config.addinivalue_line(
        "markers",
        "e2e: End-to-end integration tests (require real API keys)"
    )


def pytest_unconfigure(config):
    config.uncaptured_stderr.close()


SECONDS_BETWEEN_STACK_REPORT_AND_FAULTHANDLER_EXIT = 60


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_protocol(item, nextitem):
    """Print every thread's stack, the main thread first, shortly before faulthandler ends a hung run.

    faulthandler stops after 100 threads and lists the newest first, so in a
    suite holding more than that it never reaches the main thread, which is the
    one that hung.
    """
    hang_seconds = 0.0
    if item.config.pluginmanager.has_plugin("faulthandler"):
        hang_seconds = float(item.config.getini("faulthandler_timeout") or 0)
    if hang_seconds <= SECONDS_BETWEEN_STACK_REPORT_AND_FAULTHANDLER_EXIT:
        yield
        return
    timer = threading.Timer(hang_seconds - SECONDS_BETWEEN_STACK_REPORT_AND_FAULTHANDLER_EXIT,
                            _report_every_thread, args=(item.nodeid, item.config))
    timer.daemon = True
    timer.start()
    try:
        yield
    finally:
        timer.cancel()


def _report_every_thread(nodeid, config):
    main = threading.main_thread().ident
    frames = sorted(sys._current_frames().items(), key=lambda entry: entry[0] != main)
    names = {thread.ident: thread.name for thread in threading.enumerate()}
    report = [f"\n{nodeid} is still running; {len(frames)} threads, main thread first:"]
    for ident, frame in frames:
        report.append(f"--- {names.get(ident, ident)}\n{''.join(traceback.format_stack(frame))}")
    config.uncaptured_stderr.write("\n".join(report) + "\n")
    config.uncaptured_stderr.flush()


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
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {'status': 'success'}

    mock_client = MagicMock()
    # Strings, as on a real client: spending caps read their rules with them.
    mock_client.api_key = "hak_test_dummy_key_for_ci_testing_only"
    mock_client.base_url = "https://api.test.revenium.ai/meter/"
    mock_client.ai.create_completion.return_value = mock_response
    mock_client.ai.create_image.return_value = mock_response
    mock_client.ai.create_video.return_value = mock_response
    mock_client.ai.create_audio.return_value = mock_response

    # Patch client in all modules that import it directly
    patches = [
        # get_client() resolves through _core.metering.client, so patching it
        # covers submit_ai_event and every provider guard.
        patch('revenium_middleware._core.metering.client', mock_client),
        patch('revenium_middleware.client', mock_client),
    ]

    # Conditionally patch fal module if loaded
    try:
        import revenium_middleware.fal._metering  # noqa: F401
        patches.append(patch('revenium_middleware.fal._metering.client', mock_client))
    except (ImportError, ModuleNotFoundError):
        pass

    for p in patches:
        p.start()
    yield mock_client
    for p in patches:
        p.stop()


@pytest.fixture(autouse=True)
def _reap_enforcement_poller(monkeypatch):
    """Stop any enforcement rule-poll thread a test leaves running.

    ``check_enforcement`` starts a daemon thread (``_ensure_poller_running``)
    that calls ``_refresh_cache`` every ``_DEFAULT_POLL_INTERVAL`` seconds and
    outlives the test that started it. Once the whole suite takes longer than
    one interval, that thread wakes up inside an unrelated test and refreshes
    against whatever ``httpx.get`` stub and module globals that test installed
    -- overwriting ``_cache_timestamp`` so the test's own refresh is skipped as
    "fresh", consuming responses from its stub, or setting a Retry-After
    cooldown it never asked for.

    That is a real cross-test dependency rather than a flake: it passes on a
    fast machine, where the suite finishes inside a single poll interval, and
    fails on a slower CI runner, where it does not.

    Requesting ``monkeypatch`` orders this teardown before the stubs are
    restored, so a thread on its way out cannot fall through to real ``httpx``.

    ``_poll_wakeup`` is set as well as ``_stop_event`` because the poller
    sleeps on the wake-up event between refreshes; it is cleared afterwards so
    a wake-up a test asked for cannot leak into the next one.
    """
    from revenium_middleware._core import enforcement

    def reap():
        thread = enforcement._poll_thread
        enforcement._stop_event.set()
        enforcement._poll_wakeup.set()
        if thread is not None and thread.is_alive():
            thread.join(timeout=5)
        enforcement._poll_thread = None
        enforcement._stop_event.clear()
        enforcement._poll_wakeup.clear()
        # Whose rules the cache holds is per test: a test that seeds the cache
        # by hand must not have it dropped as another team's.
        enforcement._cache_owner = None

    reap()
    yield
    reap()


@pytest.fixture(autouse=True)
def _clear_deprecated_field_warning_cache():
    """The deprecated-field logger.warning dedup in revenium_middleware._core.fields is
    module-level state. Clear it before every test so tests that assert on the warning
    aren't order-dependent on a previously-run test populating the cache."""
    from revenium_middleware._core import fields as _fields_module
    _fields_module._WARNED_DEPRECATED_FIELDS.clear()
    yield
    _fields_module._WARNED_DEPRECATED_FIELDS.clear()


@pytest.fixture(autouse=True)
def _close_delivery_circuit():
    """The delivery circuit is process-wide: a test whose stub fails three
    deliveries opens it, and every later test's metering would then be
    buffered unsent instead of reaching that test's stub."""
    from revenium_middleware._core import delivery_circuit
    delivery_circuit.reset()
    yield
    delivery_circuit.reset()
