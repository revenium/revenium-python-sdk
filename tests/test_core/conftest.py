"""Shared stubs for the enforcement rule-fetch tests.

``test_enforcement_fetch_retry.py`` and ``test_enforcement_retry_after_bound.py``
both drive ``_fetch_rules`` against a scripted sequence of responses with the
backoff wait recorded rather than taken. The helpers live here so the two files
cannot drift apart -- an earlier copy of ``SequencedGet`` had silently lost the
branch that raises connection errors.
"""
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx
import pytest

from revenium_middleware._core import enforcement, tool_event_transport

URL = "https://api.test/v2/api/ai/enforcement-rules/team-1"


def make_response(status_code, headers=None, json_body=None):
    request = httpx.Request("GET", URL)
    return httpx.Response(status_code, headers=headers or {},
                          json=json_body if json_body is not None else {"rules": []},
                          request=request)


class SequencedGet:
    """httpx.get stand-in returning (or raising) a scripted sequence.

    The last outcome repeats once the sequence runs out, so a test that wants
    "429 forever" can pass one response. ``calls`` is the control-plane request
    count the tests assert on, and ``requests`` is the ``(url, kwargs)`` of
    each one for the tests that assert on the request the fetch built.
    """

    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = 0
        self.requests = []

    def __call__(self, *args, **kwargs):
        url = args[0] if args else kwargs.get("url")
        self.requests.append((url, kwargs))
        outcome = self.outcomes[min(self.calls, len(self.outcomes) - 1)]
        self.calls += 1
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    @property
    def last_url(self):
        return self.requests[-1][0]

    @property
    def last_params(self):
        return self.requests[-1][1].get("params")


def stub_get(monkeypatch, outcomes):
    stub = SequencedGet(outcomes)
    monkeypatch.setattr(enforcement.httpx, "get", stub)
    return stub


def stale_cache_timestamp():
    """A ``_cache_timestamp`` that ``_get_rules`` is guaranteed to treat as stale.

    Not ``0.0``. Staleness is ``time.monotonic() - _cache_timestamp``, and
    ``time.monotonic()`` is seconds since boot, so ``0.0`` means "as old as
    this host's uptime". On a developer laptop that is days and the cache reads
    as stale; on a freshly booted CI runner it is a few seconds, which is
    younger than ``_CACHE_TTL``, so the cache reads as **fresh** and no refresh
    happens at all -- a test expecting one then sees zero requests.

    Reads the clock through ``enforcement.time`` so a test that installed a
    ``FakeClock`` gets a timestamp stale on that clock, not on the real one.
    """
    return enforcement.time.monotonic() - enforcement._CACHE_TTL * 2


class FakeClock:
    """Monotonic clock a test can advance by hand.

    ``time.monotonic`` is the only clock ``enforcement`` reads, so replacing
    the module's ``time`` lets a test step over a Retry-After cooldown deadline
    without sleeping through it.
    """

    def __init__(self, now=1_000_000.0):
        self.now = now

    def monotonic(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


@pytest.fixture(autouse=True)
def reset_refresh_cooldown(monkeypatch):
    """Clear any Retry-After cooldown around every test in this package.

    The cooldown is module state, so a test that gives up on a 429 would
    otherwise suppress the refresh in whichever test ran next.
    """
    monkeypatch.setattr(enforcement, "_refresh_cooldown_until", 0.0)


@pytest.fixture(autouse=True)
def reset_department_warning_dedupe():
    """Forget which callers have been warned, around every test in this package.

    The warn-once set is module state that ``monkeypatch`` cannot restore
    (it is mutated in place, not rebound), so a caller warned in one test
    would otherwise silence the same caller in whichever test ran next.
    """
    enforcement._begin_cache_generation()
    yield
    enforcement._begin_cache_generation()


@pytest.fixture(autouse=True)
def reset_non_blocking_rule_warnings():
    """Forget which warn-only breaches have been logged, around every test.

    Mutated in place and deliberately kept across cache generations, so
    neither ``monkeypatch`` nor the generation bump above resets it.
    """
    enforcement._warned_non_blocking_rules.clear()
    enforcement._noted_unevaluable_filters.clear()
    yield
    enforcement._warned_non_blocking_rules.clear()
    enforcement._noted_unevaluable_filters.clear()


@pytest.fixture()
def fetch_env(monkeypatch):
    """A fetchable enforcement config with every wait recorded, never taken."""
    monkeypatch.setenv("REVENIUM_METERING_API_KEY", "hak_enforcement_test")
    monkeypatch.setenv("REVENIUM_TEAM_ID", "team-1")
    sleeps = []
    monkeypatch.setattr(enforcement, "_sleep", lambda seconds: sleeps.append(seconds), raising=False)
    return monkeypatch, sleeps


@pytest.fixture()
def department_cache(monkeypatch):
    """Circuit breaker on, the whole department payload seeded, nothing live.

    Returns a callable taking ``(rules, blocks, balances, warnings,
    candidates)`` and
    installing them as a fresh, initialized cache, so ``check_enforcement``
    evaluates them with no network and no poller thread. A map is installed
    exactly as given -- a malformed value reaches the code under test rather
    than being coerced to an empty dict -- and only an omitted (``None``) map
    becomes one. Seeding the cache by hand is a new cache generation, so the
    warn-once set is cleared and the generation bumped with it, exactly as a
    fetch or a disk load does.

    The sibling ``load_rules`` in ``test_enforcement_group_breakdown.py``
    predates ``departmentBudgetBlockBalances`` and seeds only the block map; tests
    that need the balances or the warnings take this one.
    """
    monkeypatch.setenv("REVENIUM_CIRCUIT_BREAKER_ENABLED", "true")
    monkeypatch.delenv("REVENIUM_BYPASS", raising=False)
    monkeypatch.delenv("REVENIUM_CB_FAIL_MODE", raising=False)
    monkeypatch.setattr(enforcement, "_load_cache_from_disk", lambda: None)
    monkeypatch.setattr(enforcement, "_ensure_poller_running", lambda: None)

    def load(rules, blocks=None, balances=None, warnings=None, candidates=None):
        # ``is None`` means "not supplied", never ``or {}``: a test that seeds a
        # deliberately malformed map has to see it reach the cache, and every
        # interesting malformed value ([], "", 0) is falsy. Coercing those to an
        # ordinary empty dict defanged the fail-open tests that pass them.
        monkeypatch.setattr(enforcement, "_cached_rules", list(rules))
        monkeypatch.setattr(enforcement, "_cached_org_unit_blocks",
                            blocks if blocks is not None else {})
        monkeypatch.setattr(enforcement, "_cached_org_unit_block_balances",
                            balances if balances is not None else {})
        monkeypatch.setattr(enforcement, "_cached_org_unit_warnings",
                            warnings if warnings is not None else {})
        monkeypatch.setattr(enforcement, "_cached_org_unit_candidates",
                            candidates if candidates is not None else {})
        monkeypatch.setattr(enforcement, "_cache_timestamp", enforcement.time.monotonic())
        monkeypatch.setattr(enforcement, "_cache_initialized", True)
        enforcement._begin_cache_generation()

    return load


@pytest.fixture()
def department_snapshot_dir(monkeypatch, tmp_path):
    """A snapshot directory plus a cache that has never been loaded.

    Yields the directory so a test can read (or corrupt) the two snapshot
    files the enforcement cache writes into it.
    """
    monkeypatch.setenv("REVENIUM_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("REVENIUM_TEAM_ID", "team-snapshot")
    monkeypatch.setattr(enforcement, "_disk_load_attempted", False)
    monkeypatch.setattr(enforcement, "_cached_rules", [])
    monkeypatch.setattr(enforcement, "_cached_org_unit_blocks", {})
    monkeypatch.setattr(enforcement, "_cached_org_unit_block_balances", {})
    monkeypatch.setattr(enforcement, "_cached_org_unit_warnings", {})
    monkeypatch.setattr(enforcement, "_cached_org_unit_candidates", {})
    monkeypatch.setattr(enforcement, "_cache_timestamp", 0.0)
    monkeypatch.setattr(enforcement, "_cache_initialized", False)
    return tmp_path


@pytest.fixture(autouse=True)
def fresh_tool_event_client():
    """Start and end every test with no cached tool-event client.

    The client is module state, so a stub installed or a real client built by
    one test would otherwise carry its connections into whichever test ran
    next.
    """
    tool_event_transport._client = None
    yield
    client = tool_event_transport._client
    tool_event_transport._client = None
    if isinstance(client, httpx.Client):
        client.close()


class KeepAliveMeter:
    """A local HTTP/1.1 metering endpoint that records the client port of every request.

    It answers with a Content-Length body and keeps the connection open, so a
    sender that pools connections reuses one and the recorded ports show how
    many connections the sender opened.
    """

    def __init__(self):
        ports = self.ports = []

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def do_POST(self):
                self.rfile.read(int(self.headers.get("Content-Length", 0)))
                ports.append(self.client_address[1])
                reply = b'{"resourceType":"metered-event"}'
                self.send_response(201)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(reply)))
                self.end_headers()
                self.wfile.write(reply)

            def log_message(self, *args):
                pass

        class Server(ThreadingHTTPServer):
            daemon_threads = True
            # Every metering worker may connect at once; the default backlog
            # of 5 resets the rest before they are accepted.
            request_queue_size = 128

        self._server = Server(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self.base_url = f"http://127.0.0.1:{self._server.server_address[1]}"
        self.url = f"{self.base_url}/meter/v2/tool/events"

    def start(self):
        self._thread.start()

    def stop(self):
        self._server.shutdown()
        self._server.server_close()

    def wait_for_requests(self, count, timeout=10.0):
        deadline = time.monotonic() + timeout
        while len(self.ports) < count and time.monotonic() < deadline:
            time.sleep(0.02)
        return len(self.ports) >= count


@pytest.fixture()
def keep_alive_meter():
    meter = KeepAliveMeter()
    meter.start()
    yield meter
    meter.stop()


class RecordingMeter:
    """A local metering address that records every request and answers with a fixed status.

    ``answer_with`` makes it answer requests under ``/meter/`` with another
    status and headers (a redirect, say); every other path gets 201, so a
    client that followed a same-host redirect leaves a second request behind.
    ``delay`` holds every answer back that many seconds, after the request is
    recorded, so a sender that timed out still leaves its request behind.
    """

    def __init__(self):
        requests = self.requests = []
        meter = self
        self.status = 201
        self.headers = {}
        self.delay = 0.0

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def _record_and_answer(self):
                self.rfile.read(int(self.headers.get("Content-Length", 0)))
                requests.append({"method": self.command, "path": self.path, "headers": dict(self.headers)})
                status, extra = (meter.status, meter.headers) if self.path.startswith("/meter/") else (201, {})
                time.sleep(meter.delay)
                reply = b'{"resourceType":"metered-event"}'
                try:
                    self.send_response(status)
                    for name, value in extra.items():
                        self.send_header(name, value)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(reply)))
                    self.end_headers()
                    self.wfile.write(reply)
                except (BrokenPipeError, ConnectionResetError):
                    pass

            do_POST = do_GET = _record_and_answer

            def log_message(self, *args):
                pass

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._server.daemon_threads = True
        self.port = self._server.server_address[1]
        self.base_url = f"http://127.0.0.1:{self.port}"
        self.meter_url = f"{self.base_url}/meter/"
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    def answer_with(self, status, headers=None):
        self.status = status
        self.headers = headers or {}

    def stop(self):
        self._server.shutdown()
        self._server.server_close()


@pytest.fixture()
def recording_meters():
    """Two recording metering addresses, A and B; both stop when the test ends."""
    meters = (RecordingMeter(), RecordingMeter())
    yield meters
    for meter in meters:
        meter.stop()


@pytest.fixture()
def isolated_metering(monkeypatch):
    """Real metering clients against local stubs, with no process-wide delivery state leaking in or out.

    Yields a private store-and-forward buffer. Client retries are off, so a
    failed send fails at once instead of sleeping through backoff.
    """
    from collections import OrderedDict

    from revenium_middleware._core import delivery_circuit, metering, metering_buffer
    from revenium_middleware._core.metering_status import reset_metering_status

    delivery_circuit.reset()
    reset_metering_status()
    buffer = metering_buffer.MeteringBuffer(flush_interval=3600)
    monkeypatch.setattr(metering_buffer, "_buffer", buffer)
    monkeypatch.setattr(metering, "_scoped_clients", OrderedDict())
    monkeypatch.setenv(metering.MAX_RETRIES_ENV, "0")
    yield buffer
    for client in metering._scoped_clients.values():
        client.close()
    reset_metering_status()
    delivery_circuit.reset()
