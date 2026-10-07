"""Load against a dead metering endpoint keeps threads bounded (BACK-3904).

On demand: ``pytest -m integration tests/test_core/test_metering_load_integration.py``.
A scaled-down version of the reported load: metered calls at a fixed rate
while the endpoint either accepts connections and never answers (hanging) or
drops connection attempts (blackhole). Before BACK-3904 every call started a
thread that lived for minutes, so live threads grew with rate x time.
"""
import socket
import threading
import time
import uuid

import pytest

from revenium_middleware._core import metering, metering_buffer, metering_pool
from revenium_middleware._core.metering import _build_metering_client, run_async_in_thread
from revenium_middleware._core.metering_buffer import MeteringBuffer
from revenium_middleware._core.metering_pool import MeteringWorkerPool
from revenium_middleware._core.metering_submission import submit_ai_event

pytestmark = pytest.mark.integration

API_KEY = "hak_load_" + "x" * 24
WORKERS = 4
QUEUE_SIZE = 10
RATE_PER_SECOND = 20
SECONDS = 3.0
NON_ROUTABLE = "http://10.255.255.1:81/meter/"
# The buffer's flush thread, started by the first buffered event, and its
# overflow build thread, started by the first overflowed one.
OTHER_THREADS = 2


@pytest.fixture
def hanging_endpoint():
    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen(128)
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


@pytest.fixture
def endpoint(request, hanging_endpoint):
    return hanging_endpoint if request.param == "hanging" else NON_ROUTABLE


@pytest.fixture
def delivery(monkeypatch):
    monkeypatch.setenv(metering.TIMEOUT_ENV, "0.5")
    monkeypatch.setenv(metering.CONNECT_TIMEOUT_ENV, "0.25")
    metering.shutdown_event.clear()
    pool = MeteringWorkerPool(WORKERS, QUEUE_SIZE, metering_pool._overflow_to_buffer)
    buffer = MeteringBuffer(flush_interval=3600)
    monkeypatch.setattr(metering_pool, "_pool", pool)
    monkeypatch.setattr(metering_buffer, "_buffer", buffer)
    yield pool, buffer
    pool.stop(timeout=10)
    for event in buffer._take_overflow():
        event.payload["task"].discard()


def completion_args():
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    return {
        "completion_start_time": now, "cost_type": "AI", "input_token_count": 1, "is_streamed": False,
        "model": "gpt-4o-mini", "output_token_count": 1, "provider": "OPENAI", "request_duration": 1,
        "request_time": now, "response_time": now, "stop_reason": "END", "total_token_count": 2,
        "transaction_id": str(uuid.uuid4()),
    }


async def metering_call():
    submit_ai_event("completion", completion_args())


@pytest.mark.parametrize("endpoint", ["hanging", "blackhole"], indirect=True)
def test_live_threads_stay_at_pool_size_under_load(monkeypatch, endpoint, delivery):
    pool, buffer = delivery
    monkeypatch.setattr(metering, "client", _build_metering_client(API_KEY, endpoint))
    baseline = threading.active_count()
    peak = baseline
    slowest_call = 0.0

    started = time.monotonic()
    for n in range(int(RATE_PER_SECOND * SECONDS)):
        while time.monotonic() < started + n / RATE_PER_SECOND:
            time.sleep(0.005)
        before = time.monotonic()
        assert run_async_in_thread(metering_call()) is not None
        slowest_call = max(slowest_call, time.monotonic() - before)
        peak = max(peak, threading.active_count())

    assert peak - baseline <= WORKERS + OTHER_THREADS
    assert slowest_call < 0.05
    assert pool.pending() <= QUEUE_SIZE + WORKERS
    if endpoint != NON_ROUTABLE:
        assert buffer.stats()["total_overflowed"] > 0
