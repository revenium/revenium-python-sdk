"""Tool metering must resolve credentials like the rest of the SDK.

configure() previously had no environment fallback at all: without an explicit
call, events were silently POSTed to http://localhost:8082 with demo-key --
fully configured production environments lost every tool event. The endpoint
path also lacked the /meter prefix the real API uses.
"""
import logging
import threading
import time
from collections import OrderedDict

import httpx
import pytest

import revenium_middleware
from revenium_middleware import _core
from revenium_middleware._core import metering, metering_submission, tool_event_transport
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


class ToolEventClientStub:
    """Stands in for the shared tool-event client; records each POST and the timeout it was sent with."""

    def __init__(self):
        self.calls = []
        self.timeouts = []

    def post(self, url, headers=None, json=None, timeout=None):
        self.timeouts.append(timeout)
        self.calls.append(RecordedCall(url, headers, json))
        return FakeResponse()


@pytest.fixture()
def tool_env(monkeypatch):
    stub = ToolEventClientStub()
    monkeypatch.setattr(tool_event_transport, "_client", stub)
    monkeypatch.setattr(tool_metering, "_metering_url", None)
    monkeypatch.setattr(tool_metering, "_api_key", None)
    monkeypatch.delenv("REVENIUM_METERING_API_KEY", raising=False)
    monkeypatch.delenv("REVENIUM_METERING_BASE_URL", raising=False)
    for module in (metering, metering_submission, _core, revenium_middleware):
        monkeypatch.setattr(module, "client", None, raising=False)
    monkeypatch.setattr(metering, "_last_failed_key", None)
    monkeypatch.setattr(tool_metering, "_no_key_warned", False, raising=False)
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


def test_tool_events_wait_as_long_as_the_metering_timeout_settings(tool_env):
    stub, monkeypatch = tool_env
    monkeypatch.setenv("REVENIUM_METERING_API_KEY", "hak_env_key")
    monkeypatch.setenv("REVENIUM_METERING_TIMEOUT_SECONDS", "2.5")
    monkeypatch.setenv("REVENIUM_METERING_CONNECT_TIMEOUT_SECONDS", "0.5")

    report_tool_call(tool_id="timed-tool", duration_ms=5, success=True)

    assert wait_for_calls(stub)
    assert stub.timeouts == [httpx.Timeout(2.5, connect=0.5)]


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


def test_initialize_metering_key_reaches_tool_events(tool_env):
    stub, _ = tool_env
    assert metering.initialize_metering(api_key="hak_initialized_dummy") is True

    report_tool_call(tool_id="init-tool", duration_ms=5)

    assert wait_for_calls(stub)
    assert stub.calls[0].headers["x-api-key"] == "hak_initialized_dummy"

    assert metering.initialize_metering(api_key="hak_second_dummy") is True
    report_tool_call(tool_id="reinit-tool", duration_ms=5)

    assert wait_for_calls(stub, count=2)
    assert stub.calls[1].headers["x-api-key"] == "hak_second_dummy"

    configure(api_key="hak_explicit_dummy")
    report_tool_call(tool_id="explicit-tool", duration_ms=5)

    assert wait_for_calls(stub, count=3)
    assert stub.calls[2].headers["x-api-key"] == "hak_explicit_dummy"


def test_malformed_base_url_sends_tool_events_to_the_ai_fallback(tool_env):
    stub, monkeypatch = tool_env
    monkeypatch.setenv("REVENIUM_METERING_API_KEY", "hak_env_key")
    monkeypatch.setenv("REVENIUM_METERING_BASE_URL", "localhost:9")

    report_tool_call(tool_id="malformed-base-tool", duration_ms=5)

    assert wait_for_calls(stub)
    assert stub.calls[0].url == "https://api.revenium.ai/meter/v2/tool/events"


def test_malformed_base_url_without_a_client_falls_back_to_default(tool_env):
    stub, monkeypatch = tool_env
    monkeypatch.setenv("REVENIUM_METERING_BASE_URL", "localhost:9")
    configure(api_key="hak_configured_dummy")

    report_tool_call(tool_id="no-client-tool", duration_ms=5)

    assert wait_for_calls(stub)
    assert stub.calls[0].url == "https://api.revenium.ai/meter/v2/tool/events"
    assert stub.calls[0].headers["x-api-key"] == "hak_configured_dummy"


def test_configure_with_a_key_and_no_ai_client_warns_that_ai_calls_are_not_metered(tool_env, caplog):
    """BACK-4246: configure() sets the key for tool events only; a customer relying on it for AI calls is told."""
    _stub, monkeypatch = tool_env
    monkeypatch.setattr(tool_metering, "_ai_metering_warned", False, raising=False)
    caplog.set_level(logging.WARNING, logger="revenium_middleware")

    configure(api_key="hak_configured_only_dummy")
    configure(api_key="hak_configured_only_dummy")

    warnings = [record.getMessage() for record in caplog.records if record.levelno == logging.WARNING]
    assert [w for w in warnings if "tool events only" in w] == [
        "configure() sets the Revenium key for tool events only; AI calls are not metered until "
        "REVENIUM_METERING_API_KEY is set or revenium_middleware.initialize_metering(api_key=...) is called."
    ]
    assert not [w for w in warnings if "hak_configured_only_dummy" in w]


def test_tool_events_keep_a_gateway_path_through_the_ai_client(tool_env):
    stub, monkeypatch = tool_env
    monkeypatch.setenv("REVENIUM_METERING_API_KEY", "hak_env_key")
    monkeypatch.setenv("REVENIUM_METERING_BASE_URL", "https://gw.acme.com/metering/revenium")

    report_tool_call(tool_id="gateway-tool", duration_ms=5)

    assert wait_for_calls(stub)
    assert stub.calls[0].url == "https://gw.acme.com/metering/revenium/meter/v2/tool/events"


def test_tool_events_without_an_ai_client_drop_a_query(tool_env):
    stub, monkeypatch = tool_env
    monkeypatch.setenv("REVENIUM_METERING_BASE_URL", "https://gw.acme.com/api?x=1")
    configure(api_key="hak_explicit")

    report_tool_call(tool_id="query-tool", duration_ms=5)

    assert wait_for_calls(stub)
    assert stub.calls[0].url == "https://gw.acme.com/api/meter/v2/tool/events"


def test_a_query_address_warns_once_across_tool_events(tool_env, caplog):
    stub, monkeypatch = tool_env
    monkeypatch.setattr(tool_metering, "_address_query_warned", False, raising=False)
    monkeypatch.setenv("REVENIUM_METERING_BASE_URL", "https://gw.acme.com/api?x=1")
    caplog.set_level(logging.WARNING, logger="revenium_middleware")
    configure(api_key="hak_explicit")

    report_tool_call(tool_id="query-tool-1", duration_ms=5)
    report_tool_call(tool_id="query-tool-2", duration_ms=5)

    assert wait_for_calls(stub, count=2)
    dropped = [
        record.getMessage() for record in caplog.records if "query string or fragment" in record.getMessage()
    ]
    assert len(dropped) == 1
    assert "x=1" not in dropped[0]


def test_configure_trims_a_padded_key(tool_env):
    configure(api_key="hak_tool_dummy \n")

    assert tool_metering._resolve_endpoint()[1] == "hak_tool_dummy"


def test_no_key_tool_events_warn_once_per_process(tool_env, caplog):
    """With no key anywhere, tool events are skipped with one warning per
    process, not one per event (BACK-4452 N14)."""
    stub, _ = tool_env
    caplog.set_level(logging.DEBUG, logger=tool_metering.logger.name)

    for _ in range(3):
        report_tool_call(tool_id="probe-tool", duration_ms=5, success=True)

    skipped = [r for r in caplog.records
               if r.levelno == logging.WARNING and "Tool metering skipped" in r.getMessage()]
    assert len(skipped) == 1
    assert stub.calls == []


def test_concurrent_first_tool_events_warn_once(tool_env, caplog):
    """Tool events racing on the first no-key skip still warn only once
    (BACK-4452 N14)."""
    caplog.set_level(logging.DEBUG, logger=tool_metering.logger.name)
    start = threading.Barrier(8)

    def first_event():
        start.wait()
        report_tool_call(tool_id="probe-tool", duration_ms=5, success=True)

    threads = [threading.Thread(target=first_event) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    skipped = [r for r in caplog.records
               if r.levelno == logging.WARNING and "Tool metering skipped" in r.getMessage()]
    assert len(skipped) == 1


CONFIGURE_KEY = "hak_fake_configure_key"
CUSTOMER_B_KEY = "hak_fake_customer_b_key"
CUSTOMER_B_EVENTS_URL = "https://customer-b.example/meter/v2/tool/events"
CONFIGURED_EVENTS_URL = "https://configured.example/meter/v2/tool/events"
PROCESS_EVENTS_URL = "https://process.example/meter/v2/tool/events"


@pytest.fixture()
def block_env(tool_env):
    """BACK-4408: a tool event in a metering_key() block sends a key and an address from one source."""
    stub, monkeypatch = tool_env
    monkeypatch.setattr(metering, "_scoped_clients", OrderedDict())
    return stub, monkeypatch


def sent(stub):
    assert wait_for_calls(stub)
    return [(call.url, call.headers["x-api-key"]) for call in stub.calls]


def report_tool_call_for_customer_b(base_url="https://customer-b.example"):
    with revenium_middleware.metering_key(CUSTOMER_B_KEY, base_url=base_url):
        report_tool_call(tool_id="customer-b-tool", duration_ms=5)


def test_inside_metering_key_a_configure_key_goes_to_the_process_address_not_the_block_address(block_env):
    stub, monkeypatch = block_env
    monkeypatch.setenv("REVENIUM_METERING_BASE_URL", "https://process.example")
    configure(api_key=CONFIGURE_KEY)

    report_tool_call_for_customer_b()

    assert sent(stub) == [(PROCESS_EVENTS_URL, CONFIGURE_KEY)]


def test_inside_metering_key_a_configure_key_goes_to_the_process_client_address(block_env):
    stub, monkeypatch = block_env
    process = metering._new_client("hak_fake_process_key", "https://process-client.example")
    for module in (metering, metering_submission, _core, revenium_middleware):
        monkeypatch.setattr(module, "client", process, raising=False)
    configure(api_key=CONFIGURE_KEY)

    report_tool_call_for_customer_b()

    assert sent(stub) == [("https://process-client.example/meter/v2/tool/events", CONFIGURE_KEY)]


def test_inside_metering_key_a_configure_address_alone_never_receives_the_block_key(block_env):
    stub, _ = block_env
    configure(metering_url="https://configured.example")

    report_tool_call_for_customer_b()

    assert sent(stub) == [(CUSTOMER_B_EVENTS_URL, CUSTOMER_B_KEY)]


def test_inside_metering_key_without_base_url_a_configure_address_alone_is_not_used(block_env):
    stub, monkeypatch = block_env
    monkeypatch.setenv("REVENIUM_METERING_BASE_URL", "https://process.example")
    configure(metering_url="https://configured.example")

    report_tool_call_for_customer_b(base_url=None)

    assert sent(stub) == [(PROCESS_EVENTS_URL, CUSTOMER_B_KEY)]


def test_a_streamed_tool_started_in_a_metering_key_block_keeps_the_block_address_after_it(block_env):
    stub, _ = block_env
    configure(metering_url="https://configured.example")

    @revenium_middleware.meter_tool("search-tool")
    def search():
        yield "first"
        yield "second"

    stream = search()
    with revenium_middleware.metering_key(CUSTOMER_B_KEY, base_url="https://customer-b.example"):
        assert next(stream) == "first"
    assert list(stream) == ["second"]

    assert sent(stub) == [(CUSTOMER_B_EVENTS_URL, CUSTOMER_B_KEY)]


def test_inside_metering_key_without_configure_the_block_key_goes_to_the_block_address(block_env):
    stub, _ = block_env

    report_tool_call_for_customer_b()

    assert sent(stub) == [(CUSTOMER_B_EVENTS_URL, CUSTOMER_B_KEY)]


def test_inside_metering_key_a_configure_key_and_address_go_together(block_env):
    stub, _ = block_env
    configure(metering_url="https://configured.example", api_key=CONFIGURE_KEY)

    report_tool_call_for_customer_b()

    assert sent(stub) == [(CONFIGURED_EVENTS_URL, CONFIGURE_KEY)]
