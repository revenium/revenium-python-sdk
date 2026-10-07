"""The default delivery keeps up with 30 metered calls a second (BACK-3919).

On demand: ``pytest -m integration tests/test_core/test_metering_throughput_integration.py``.
Each test drives the real worker pool, buffer and metering client, sized from
their defaults, against a local endpoint that answers every record after a
fixed delay, or stops answering for a while, at 30 calls a second, the load
of a busy LiteLLM proxy. One record per call, and two at 250 ms for a proxy
that still meters each call twice. The slow-endpoint runs take five
minutes each and the outage run about seven.
"""
import json
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Dict, List

import pytest

from revenium_middleware._core import delivery_circuit, metering, metering_buffer, metering_pool
from revenium_middleware._core.metering import _build_metering_client, run_async_in_thread
from revenium_middleware._core.metering_status import get_metering_status, reset_metering_status
from revenium_middleware._core.metering_submission import submit_ai_event

pytestmark = pytest.mark.integration

API_KEY = "hak_load_" + "x" * 24
RATE_PER_SECOND = 30
SLOW_RUN_SECONDS = 300
OUTAGE_SECONDS = 300
RECOVERY_SECONDS = 120
# A record delivered later than this was waiting in a backlog, not in flight.
MAX_DELIVERY_LAG_SECONDS = 5.0
DELIVERY_THREAD_PREFIXES = ("ReveniumMeteringWorker", "MeteringBufferFlush", "MeteringOverflowBuild", "MeteringReplay")
# The buffer's flush thread and its overflow build thread.
BUFFER_THREADS = 2


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    stub: "SlowMeteringStub"

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if not self.stub.answer():
            self.close_connection = True
            return
        self.stub.record(body["transactionId"])
        reply = json.dumps({"id": "m", "label": "m", "resourceType": "metering", "signature": "s"}).encode()
        self.send_response(201)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(reply)))
        self.end_headers()
        self.wfile.write(reply)

    def log_message(self, *args):
        pass


class _Server(ThreadingHTTPServer):
    daemon_threads = True
    request_queue_size = 256


class SlowMeteringStub:
    """A metering endpoint that answers after ``latency`` seconds, or holds requests while ``hanging``."""

    def __init__(self, latency: float):
        self.latency = latency
        self.hanging = threading.Event()
        self._stopped = threading.Event()
        self._lock = threading.Lock()
        self.arrivals: Dict[str, float] = {}
        handler = type("Handler", (_Handler,), {"stub": self})
        self._server = _Server(("127.0.0.1", 0), handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self.base_url = f"http://127.0.0.1:{self._server.server_address[1]}/meter/"

    def answer(self) -> bool:
        """Wait out the latency, or the outage; False when the request should get no answer."""
        if self.hanging.is_set():
            while self.hanging.is_set() and not self._stopped.is_set():
                time.sleep(0.05)
            return False
        time.sleep(self.latency)
        return True

    def record(self, transaction_id: str) -> None:
        with self._lock:
            self.arrivals.setdefault(transaction_id, time.monotonic())

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stopped.set()
        self._server.shutdown()
        self._server.server_close()


def completion_args(transaction_id: str) -> dict:
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    return {
        "completion_start_time": now, "cost_type": "AI", "input_token_count": 1, "is_streamed": False,
        "model": "gpt-4o-mini", "output_token_count": 1, "provider": "LITELLM", "request_duration": 1,
        "request_time": now, "response_time": now, "stop_reason": "END", "total_token_count": 2,
        "transaction_id": transaction_id,
    }


def delivery_threads() -> int:
    return sum(1 for thread in threading.enumerate() if thread.name.startswith(DELIVERY_THREAD_PREFIXES))


class Driver:
    """Meters ``RATE_PER_SECOND`` calls a second and remembers when each record was offered."""

    def __init__(self, records_per_call: int = 1):
        self.records_per_call = records_per_call
        self.offered: Dict[str, float] = {}
        self.peak_threads = 0
        self.slowest_call = 0.0

    def run(self, seconds: float) -> None:
        started = time.monotonic()
        for n in range(int(RATE_PER_SECOND * seconds)):
            while time.monotonic() < started + n / RATE_PER_SECOND:
                time.sleep(0.002)
            for _ in range(self.records_per_call):
                transaction_id = str(uuid.uuid4())
                self.offered[transaction_id] = time.monotonic()
                before = time.monotonic()
                assert run_async_in_thread(self._meter(transaction_id)) is not None
                self.slowest_call = max(self.slowest_call, time.monotonic() - before)
            if n % RATE_PER_SECOND == 0:
                self.peak_threads = max(self.peak_threads, delivery_threads())

    @staticmethod
    async def _meter(transaction_id: str) -> None:
        submit_ai_event("completion", completion_args(transaction_id))


def wait_for_delivery(stub: SlowMeteringStub, offered: List[str], timeout: float) -> List[str]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        missing = [txn for txn in offered if txn not in stub.arrivals]
        if not missing:
            return []
        time.sleep(0.5)
    return [txn for txn in offered if txn not in stub.arrivals]


@pytest.fixture
def default_delivery(monkeypatch):
    """The process-wide pool, buffer and circuit, built fresh from their defaults."""
    for name in (metering_pool.WORKERS_ENV, metering_pool.QUEUE_SIZE_ENV, metering.TIMEOUT_ENV,
                 metering.CONNECT_TIMEOUT_ENV, metering.MAX_RETRIES_ENV,
                 "REVENIUM_BUFFER_MAX_SIZE", "REVENIUM_BUFFER_FLUSH_INTERVAL"):
        monkeypatch.delenv(name, raising=False)
    metering.shutdown_event.clear()
    monkeypatch.setattr(metering_pool, "_pool", None)
    monkeypatch.setattr(metering_buffer, "_buffer", None)
    delivery_circuit.reset()
    reset_metering_status()
    yield
    if metering_pool._pool is not None:
        metering_pool._pool.stop(timeout=15)
    buffer = metering_buffer._buffer
    if buffer is not None:
        metering.shutdown_event.set()
        buffer.request_replay()
        if buffer._thread is not None:
            buffer._thread.join(timeout=15)
        for event in buffer._take_overflow():
            event.payload["task"].discard()
        metering.shutdown_event.clear()
    delivery_circuit.reset()


def connect(monkeypatch, stub: SlowMeteringStub) -> None:
    monkeypatch.setattr(metering, "client", _build_metering_client(API_KEY, stub.base_url))


@pytest.mark.parametrize("latency, records_per_call", [(0.25, 1), (0.30, 1), (0.25, 2)])
def test_defaults_deliver_every_record_promptly_from_a_slow_endpoint(
    monkeypatch, default_delivery, latency, records_per_call
):
    with SlowMeteringStub(latency) as stub:
        connect(monkeypatch, stub)
        baseline_threads = delivery_threads()
        driver = Driver(records_per_call)
        driver.run(SLOW_RUN_SECONDS)

        missing = wait_for_delivery(stub, list(driver.offered), timeout=30)

    lags = sorted(stub.arrivals[txn] - offered_at for txn, offered_at in driver.offered.items()
                  if txn in stub.arrivals)
    stats = metering_buffer.get_buffer_stats()
    assert missing == []
    assert lags[-1] < MAX_DELIVERY_LAG_SECONDS
    assert stats["total_overflowed"] == 0
    assert stats["total_evicted"] == 0
    assert get_metering_status().evicted_count == 0
    assert driver.peak_threads - baseline_threads <= metering_pool.DEFAULT_WORKERS
    assert driver.slowest_call < 0.05


def test_an_outage_is_buffered_without_holding_workers_and_replayed_after_recovery(monkeypatch, default_delivery):
    with SlowMeteringStub(0.25) as stub:
        connect(monkeypatch, stub)
        stub.hanging.set()
        outage = Driver()
        outage.run(OUTAGE_SECONDS)
        pending_as_the_outage_ends = metering_pool._pool.pending()
        stub.hanging.clear()

        recovery = Driver()
        recovery.run(RECOVERY_SECONDS)
        offered = list(outage.offered) + list(recovery.offered)
        missing = wait_for_delivery(stub, offered, timeout=180)

    stats = metering_buffer.get_buffer_stats()
    assert missing == []
    assert stats["total_evicted"] == 0
    assert get_metering_status().evicted_count == 0
    # Once the circuit opens, records go to the buffer as fast as they come
    # and only the probes still hold a worker; without it the queue is full.
    assert pending_as_the_outage_ends <= metering_pool.DEFAULT_WORKERS
    peak_threads = max(outage.peak_threads, recovery.peak_threads)
    assert peak_threads <= metering_pool.DEFAULT_WORKERS + BUFFER_THREADS + metering_buffer.REPLAY_CONCURRENCY - 1
    assert max(outage.slowest_call, recovery.slowest_call) < 0.05
