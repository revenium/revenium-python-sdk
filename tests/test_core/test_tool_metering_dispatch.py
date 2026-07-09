"""Tool metering must never block the wrapped call.

meter_tool and report_tool_call previously issued the metering POST inline
(sync client in the decorator's finally, awaited client in the async path),
stalling the caller for the full round-trip on a slow endpoint. They must
dispatch fire-and-forget like AI-completion metering does.
"""
import asyncio
import inspect
import logging
import time
import warnings

import pytest

from revenium_middleware._metering import decorator as tool_metering
from revenium_middleware._metering.decorator import configure, meter_tool, report_tool_call

SLOW_SECONDS = 1.0
FAST_BUDGET_SECONDS = 0.5


class RecordedCall:
    def __init__(self, url, headers, json):
        self.url = url
        self.headers = headers
        self.json = json


class FakeResponse:
    def raise_for_status(self):
        return None


class SlowHttpxStub:
    """Stands in for the httpx module: every POST takes SLOW_SECONDS."""

    def __init__(self):
        self.calls = []
        stub = self

        class Client:
            def __init__(self, timeout=None):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def post(self, url, headers=None, json=None):
                time.sleep(SLOW_SECONDS)
                stub.calls.append(RecordedCall(url, headers, json))
                return FakeResponse()

        class AsyncClient:
            def __init__(self, timeout=None):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *exc):
                return False

            async def post(self, url, headers=None, json=None):
                await asyncio.sleep(SLOW_SECONDS)
                stub.calls.append(RecordedCall(url, headers, json))
                return FakeResponse()

        self.Client = Client
        self.AsyncClient = AsyncClient


@pytest.fixture()
def slow_endpoint(monkeypatch):
    stub = SlowHttpxStub()
    monkeypatch.setattr(tool_metering, "httpx", stub)
    configure(metering_url="http://metering.test", api_key="rev_mk_test")
    yield stub
    configure()  # clear overrides; resolution falls back to the environment


def wait_for_calls(stub, count=1, timeout=6.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if len(stub.calls) >= count:
            return True
        time.sleep(0.05)
    return False


def test_meter_tool_sync_returns_before_metering_completes(slow_endpoint):
    @meter_tool("audit-probe-tool")
    def fast_tool(x):
        return x * 2

    start = time.perf_counter()
    result = fast_tool(21)
    elapsed = time.perf_counter() - start

    assert result == 42
    assert elapsed < FAST_BUDGET_SECONDS, f"caller blocked for {elapsed:.2f}s"
    # The event is still delivered, just in the background.
    assert wait_for_calls(slow_endpoint)
    assert slow_endpoint.calls[0].json["toolId"] == "audit-probe-tool"


def test_report_tool_call_returns_before_metering_completes(slow_endpoint):
    start = time.perf_counter()
    report_tool_call(tool_id="manual-tool", duration_ms=5, success=True)
    elapsed = time.perf_counter() - start

    assert elapsed < FAST_BUDGET_SECONDS, f"caller blocked for {elapsed:.2f}s"
    assert wait_for_calls(slow_endpoint)
    assert slow_endpoint.calls[0].json["toolId"] == "manual-tool"


def test_meter_tool_async_returns_before_metering_completes(slow_endpoint):
    @meter_tool("async-probe-tool")
    async def fast_async_tool(x):
        return x + 1

    async def scenario():
        start = time.perf_counter()
        result = await fast_async_tool(1)
        return result, time.perf_counter() - start

    result, elapsed = asyncio.run(scenario())

    assert result == 2
    assert elapsed < FAST_BUDGET_SECONDS, f"caller blocked for {elapsed:.2f}s"
    assert wait_for_calls(slow_endpoint)
    assert slow_endpoint.calls[0].json["toolId"] == "async-probe-tool"


def test_dispatch_time_failure_never_reaches_the_caller(slow_endpoint, monkeypatch):
    # Synchronous stub: raises while _dispatch_tool_event constructs the
    # coroutine, i.e. before anything is handed to the background thread.
    def boom(**kwargs):
        raise RuntimeError("metering exploded")

    monkeypatch.setattr(tool_metering, "_send_tool_event_async", boom)

    @meter_tool("resilient-tool")
    def fast_tool():
        return "ok"

    assert fast_tool() == "ok"


def test_background_metering_failure_never_reaches_the_caller(slow_endpoint, monkeypatch, caplog):
    # Async stub: the failure only surfaces when the coroutine is awaited on
    # the metering thread, exercising the background swallow-and-log path.
    async def boom(*args, **kwargs):
        raise RuntimeError("metering exploded in background")

    monkeypatch.setattr(tool_metering, "_send_tool_event_async", boom)

    @meter_tool("resilient-tool-async")
    def fast_tool():
        return "ok"

    with caplog.at_level(logging.WARNING, logger="revenium_middleware"):
        assert fast_tool() == "ok"
        # The metering thread must catch AND log the failure -- if its
        # exception handling were removed, nothing would be logged and this
        # assertion would fail (the thread would die with stderr noise only).
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and "metering exploded in background" not in caplog.text:
            time.sleep(0.05)

    assert "metering exploded in background" in caplog.text


def test_dispatch_failure_does_not_leak_unawaited_coroutine(slow_endpoint, monkeypatch):
    # If run_async_in_thread raises after the coroutine is created, the
    # dispatcher must close() it; otherwise garbage collection emits a
    # "coroutine was never awaited" RuntimeWarning. GC timing is not
    # deterministic under pytest (log capture keeps the raised exception --
    # and through its traceback, the coroutine -- alive past the test body),
    # so capture the coroutine and assert on its state directly.
    captured = []

    def raiser(coro):
        captured.append(coro)
        raise RuntimeError("thread pool exploded")

    # _dispatch_tool_event imports the symbol lazily from the package root.
    monkeypatch.setattr("revenium_middleware.run_async_in_thread", raiser)

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        report_tool_call(tool_id="leak-guard-tool", duration_ms=1, success=True)

    assert len(captured) == 1
    coro = captured.pop()
    try:
        assert inspect.getcoroutinestate(coro) == inspect.CORO_CLOSED, (
            "dispatcher must close the coroutine when run_async_in_thread fails"
        )
    finally:
        # Idempotent; keeps a failing run from also spewing the leak warning.
        coro.close()
