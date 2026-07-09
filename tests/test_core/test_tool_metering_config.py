"""Tool metering must resolve credentials like the rest of the SDK.

configure() previously had no environment fallback at all: without an explicit
call, events were silently POSTed to http://localhost:8082 with demo-key --
fully configured production environments lost every tool event. The endpoint
path also lacked the /meter prefix the real API uses.
"""
import time

import pytest

from revenium_middleware._metering import decorator as tool_metering
from revenium_middleware._metering.decorator import configure, report_tool_call


class RecordedCall:
    def __init__(self, url, headers, json):
        self.url = url
        self.headers = headers
        self.json = json


class FakeResponse:
    def raise_for_status(self):
        return None


class HttpxStub:
    def __init__(self):
        self.calls = []
        stub = self

        class AsyncClient:
            def __init__(self, timeout=None):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *exc):
                return False

            async def post(self, url, headers=None, json=None):
                stub.calls.append(RecordedCall(url, headers, json))
                return FakeResponse()

        self.AsyncClient = AsyncClient


@pytest.fixture()
def tool_env(monkeypatch):
    stub = HttpxStub()
    monkeypatch.setattr(tool_metering, "httpx", stub)
    monkeypatch.setattr(tool_metering, "_metering_url", None)
    monkeypatch.setattr(tool_metering, "_api_key", None)
    monkeypatch.delenv("REVENIUM_METERING_API_KEY", raising=False)
    monkeypatch.delenv("REVENIUM_METERING_BASE_URL", raising=False)
    return stub, monkeypatch


def wait_for_calls(stub, count=1, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if len(stub.calls) >= count:
            return True
        time.sleep(0.05)
    return False


def test_env_vars_used_when_configure_never_called(tool_env):
    stub, monkeypatch = tool_env
    monkeypatch.setenv("REVENIUM_METERING_API_KEY", "hak_env_key")
    monkeypatch.setenv("REVENIUM_METERING_BASE_URL", "https://api.dev.example")

    report_tool_call(tool_id="env-tool", duration_ms=5, success=True)

    assert wait_for_calls(stub)
    call = stub.calls[0]
    assert call.url == "https://api.dev.example/meter/v2/tool/events"
    assert call.headers["x-api-key"] == "hak_env_key"


def test_no_key_anywhere_skips_post_instead_of_demo_localhost(tool_env):
    stub, _ = tool_env

    report_tool_call(tool_id="orphan-tool", duration_ms=5, success=True)

    time.sleep(0.4)  # give the background dispatch a chance to (not) fire
    assert stub.calls == []


def test_configure_overrides_env(tool_env):
    stub, monkeypatch = tool_env
    monkeypatch.setenv("REVENIUM_METERING_API_KEY", "hak_env_key")
    monkeypatch.setenv("REVENIUM_METERING_BASE_URL", "https://env.example")
    configure(metering_url="https://explicit.example", api_key="hak_explicit")

    report_tool_call(tool_id="explicit-tool", duration_ms=5, success=True)

    assert wait_for_calls(stub)
    call = stub.calls[0]
    assert call.url == "https://explicit.example/meter/v2/tool/events"
    assert call.headers["x-api-key"] == "hak_explicit"


def test_meter_prefix_not_duplicated(tool_env):
    stub, monkeypatch = tool_env
    monkeypatch.setenv("REVENIUM_METERING_API_KEY", "hak_env_key")
    monkeypatch.setenv("REVENIUM_METERING_BASE_URL", "https://api.revenium.ai/meter/")

    report_tool_call(tool_id="prefix-tool", duration_ms=5, success=True)

    assert wait_for_calls(stub)
    assert stub.calls[0].url == "https://api.revenium.ai/meter/v2/tool/events"


def test_default_base_url_when_only_key_present(tool_env):
    stub, monkeypatch = tool_env
    monkeypatch.setenv("REVENIUM_METERING_API_KEY", "hak_env_key")

    report_tool_call(tool_id="default-base-tool", duration_ms=5, success=True)

    assert wait_for_calls(stub)
    assert stub.calls[0].url == "https://api.revenium.ai/meter/v2/tool/events"


def test_endpoint_captured_at_dispatch_time_not_delivery_time(tool_env):
    """A configure() racing an already-dispatched event must not redirect it."""
    stub, _ = tool_env
    configure(metering_url="https://first.example", api_key="hak_first")

    report_tool_call(tool_id="race-tool", duration_ms=1, success=True)
    # Reconfigure immediately -- the event above was already resolved
    # synchronously at dispatch, so it must still go to the first endpoint.
    configure(metering_url="https://second.example", api_key="hak_second")

    assert wait_for_calls(stub)
    assert stub.calls[0].url == "https://first.example/meter/v2/tool/events"
    assert stub.calls[0].headers["x-api-key"] == "hak_first"
