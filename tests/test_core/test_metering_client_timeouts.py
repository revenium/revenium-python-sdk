"""Metering client timeout and retry configuration (BACK-3904)."""
import logging
import socket
import threading
import time

import httpx
import pytest

from revenium_middleware._core import metering, metering_buffer
from revenium_middleware._core.metering import _build_metering_client
from revenium_middleware._core.metering_buffer import MeteringBuffer
from revenium_middleware._core.metering_submission import submit_ai_event

API_KEY = "hak_timeouts_" + "x" * 24
ENV_VARS = (metering.TIMEOUT_ENV, metering.CONNECT_TIMEOUT_ENV, metering.MAX_RETRIES_ENV)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in ENV_VARS:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def hanging_endpoint():
    """Accepts connections and never answers, like the reporter's stuck endpoint."""
    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen(16)
    held = []
    stop = threading.Event()

    def accept():
        server.settimeout(0.1)
        while not stop.is_set():
            try:
                held.append(server.accept()[0])
            except OSError:
                continue

    acceptor = threading.Thread(target=accept, daemon=True)
    acceptor.start()
    yield f"http://127.0.0.1:{server.getsockname()[1]}/meter/"
    stop.set()
    acceptor.join(5)
    for conn in held:
        conn.close()
    server.close()


def test_background_defaults_are_shorter_than_the_generated_client():
    client = _build_metering_client(API_KEY, None)

    assert client.timeout == httpx.Timeout(10.0, connect=5.0)
    assert client.max_retries == 2


def test_env_overrides_reach_the_client(monkeypatch):
    monkeypatch.setenv(metering.TIMEOUT_ENV, "2.5")
    monkeypatch.setenv(metering.CONNECT_TIMEOUT_ENV, "0.5")
    monkeypatch.setenv(metering.MAX_RETRIES_ENV, "0")

    client = _build_metering_client(API_KEY, "https://metering.test/meter/")

    assert client.timeout == httpx.Timeout(2.5, connect=0.5)
    assert client.max_retries == 0


@pytest.mark.parametrize(
    "name, raw",
    [
        (metering.TIMEOUT_ENV, "0"),
        (metering.TIMEOUT_ENV, "-1"),
        (metering.TIMEOUT_ENV, "inf"),
        (metering.TIMEOUT_ENV, "soon"),
        (metering.CONNECT_TIMEOUT_ENV, "nan"),
        (metering.MAX_RETRIES_ENV, "-1"),
        (metering.MAX_RETRIES_ENV, "1.5"),
    ],
)
def test_invalid_values_fall_back_to_defaults_with_a_warning(monkeypatch, caplog, name, raw):
    monkeypatch.setenv(name, raw)

    with caplog.at_level(logging.WARNING):
        client = _build_metering_client(API_KEY, None)

    assert client.timeout == httpx.Timeout(10.0, connect=5.0)
    assert client.max_retries == 2
    assert name in caplog.text


def test_initialize_metering_rereads_the_timeouts(monkeypatch):
    monkeypatch.setattr(metering, "client", None)
    monkeypatch.setenv(metering.TIMEOUT_ENV, "3")

    assert metering.initialize_metering(api_key=API_KEY)

    assert metering.client.timeout == httpx.Timeout(3.0, connect=5.0)


def test_a_hanging_endpoint_releases_the_caller_within_the_configured_budget(monkeypatch, hanging_endpoint):
    monkeypatch.setenv(metering.TIMEOUT_ENV, "0.2")
    monkeypatch.setenv(metering.MAX_RETRIES_ENV, "1")
    monkeypatch.setattr(metering, "client", _build_metering_client(API_KEY, hanging_endpoint))
    buffer = MeteringBuffer(flush_interval=3600, replay_fn=lambda event, timeout: None)
    monkeypatch.setattr(metering_buffer, "_buffer", buffer)

    started = time.monotonic()
    result = submit_ai_event("completion", {
        "completion_start_time": "2026-10-06T00:00:00Z", "cost_type": "AI", "input_token_count": 1,
        "is_streamed": False, "model": "gpt-4o-mini", "output_token_count": 1, "provider": "OPENAI",
        "request_duration": 1, "request_time": "2026-10-06T00:00:00Z",
        "response_time": "2026-10-06T00:00:00Z", "stop_reason": "END", "total_token_count": 2,
        "transaction_id": "txn-hanging",
    })
    elapsed = time.monotonic() - started

    assert result is None
    assert buffer.stats()["size"] == 1
    # Two attempts of 0.2s plus one sub-second backoff; the 60s default would take minutes.
    assert elapsed < 3.0
