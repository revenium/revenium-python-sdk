"""A real local enforcement-rules endpoint for the non-blocking tests (BACK-3917).

The blocking defect lived below every mock the older tests installed: they
stubbed ``httpx.get`` or ``check_enforcement`` itself, so the inline refresh,
its retries and its sleeps never ran. These helpers put a real socket behind
``REVENIUM_ENFORCEMENT_BASE_URL`` instead, so the code under test makes its
real request and meets the failure for real.

Not a test module (no ``test_`` prefix), so pytest imports it rather than
collecting it.
"""
import asyncio
import json
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import List, NamedTuple, Optional

from revenium_middleware._core import enforcement

HANG = "hang"
REFUSE = "refuse"
UNAVAILABLE = "503"
FORBIDDEN = "403"
HEALTHY = "healthy"
FAILURE_MODES = (HANG, REFUSE, UNAVAILABLE, FORBIDDEN)

# Longest a hung request is held before the stub drops it. Teardown releases
# it at once; the cap only bounds a run against code that blocks on it.
_HANG_CAP_SECONDS = 3.0
# A Revenium round trip seen from a customer's proxy. The 403 stall the ticket
# measured is one round trip per request, so a stub answering in microseconds
# would hide it.
_ROUND_TRIP_SECONDS = 0.3


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


class EnforcementStub:
    """A rules endpoint that hangs, refuses, fails or answers, as asked.

    ``hits`` counts the requests that reached it. A ``refuse`` stub has no
    listener at all, so nothing can reach it and its count stays zero.
    """

    def __init__(self, mode: str, rules: Optional[list] = None):
        self.mode = mode
        self.rules = rules or []
        self.release = threading.Event()
        self._lock = threading.Lock()
        self._hits = 0
        self._server: Optional[ThreadingHTTPServer] = None
        self.port = _free_port()
        if mode != REFUSE:
            self._server = ThreadingHTTPServer(("127.0.0.1", self.port), self._handler())
            self._server.daemon_threads = True
            threading.Thread(target=self._server.serve_forever, daemon=True,
                             name="enforcement-stub").start()

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    @property
    def hits(self) -> int:
        with self._lock:
            return self._hits

    def _record_hit(self) -> None:
        with self._lock:
            self._hits += 1

    def _handler(self):
        stub = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args):
                pass

            def do_GET(self):
                stub._record_hit()
                if stub.mode == HANG:
                    stub.release.wait(_HANG_CAP_SECONDS)
                    self.close_connection = True
                    return
                if stub.mode == HEALTHY:
                    self._send(200, {"rules": stub.rules})
                    return
                stub.release.wait(_ROUND_TRIP_SECONDS)
                self._send(int(stub.mode), {"error": stub.mode})

            def _send(self, status, body):
                payload = json.dumps(body).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

        return Handler

    def close(self) -> None:
        self.release.set()
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()


def point_enforcement_at(monkeypatch, stub: EnforcementStub, poll_interval: int = 60) -> None:
    """Circuit breaker on, aimed at ``stub``, with a cache that has never loaded."""
    monkeypatch.setenv("REVENIUM_CIRCUIT_BREAKER_ENABLED", "true")
    monkeypatch.setenv("REVENIUM_METERING_API_KEY", "hak_enforcement_stub")
    monkeypatch.setenv("REVENIUM_TEAM_ID", "team-stub")
    monkeypatch.setenv("REVENIUM_ENFORCEMENT_BASE_URL", stub.base_url)
    monkeypatch.setenv("REVENIUM_CB_POLL_INTERVAL_SECONDS", str(poll_interval))
    for name in ("REVENIUM_BYPASS", "REVENIUM_CB_FAIL_MODE", "REVENIUM_CACHE_DIR"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(enforcement, "_cached_rules", [])
    monkeypatch.setattr(enforcement, "_cached_org_unit_blocks", {})
    monkeypatch.setattr(enforcement, "_cached_org_unit_block_balances", {})
    monkeypatch.setattr(enforcement, "_cached_org_unit_warnings", {})
    monkeypatch.setattr(enforcement, "_cache_timestamp", 0.0)
    monkeypatch.setattr(enforcement, "_cache_initialized", False)
    monkeypatch.setattr(enforcement, "_disk_load_attempted", True)
    monkeypatch.setattr(enforcement, "_refresh_cooldown_until", 0.0)


def shut_down(stub: EnforcementStub) -> None:
    """Stop the poller and the stub without leaving a thread to haunt the next test.

    The stop is signalled before the stub lets go of a hung request, so the
    poller sees it at its next backoff wait instead of retrying, and only then
    joined.
    """
    enforcement._stop_event.set()
    stub.close()
    enforcement.stop_polling()


def wait_until(predicate, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


class LoopReport(NamedTuple):
    """What a run did to the event loop, and what each request waited."""

    max_lag: float
    latencies: List[float]

    def p99(self) -> float:
        ordered = sorted(self.latencies)
        return ordered[min(len(ordered) - 1, int(len(ordered) * 0.99))]


async def drive_on_loop(call, requests: int, rate: float, tick: float = 0.005,
                        give_up_after: float = 1.0) -> LoopReport:
    """Run ``call()`` ``requests`` times at ``rate`` per second, watching the loop.

    A ticker sleeping ``tick`` seconds at a time measures how late each wake-up
    is; anything that holds the loop -- a blocking socket read, a ``sleep`` on
    the loop's own thread -- shows up as one late tick, however short the
    request that caused it.

    No further requests are issued once one has taken ``give_up_after``
    seconds: the run has already failed, and against code that blocks, every
    remaining request would block too.
    """
    loop = asyncio.get_running_loop()
    lags: List[float] = []
    latencies: List[float] = []
    finished = asyncio.Event()

    async def monitor():
        while not finished.is_set():
            started = loop.time()
            await asyncio.sleep(tick)
            lags.append(loop.time() - started - tick)

    async def one():
        started = time.perf_counter()
        try:
            await call()
        finally:
            latencies.append(time.perf_counter() - started)

    ticker = asyncio.create_task(monitor())
    tasks = []
    for _ in range(requests):
        if latencies and max(latencies) > give_up_after:
            break
        tasks.append(asyncio.create_task(one()))
        await asyncio.sleep(1.0 / rate)
    await asyncio.gather(*tasks, return_exceptions=True)
    finished.set()
    await ticker
    return LoopReport(max(lags, default=0.0), latencies)
