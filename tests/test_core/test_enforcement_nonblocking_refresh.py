"""The pre-call check never waits on Revenium (BACK-3917).

With the circuit breaker on, a LiteLLM proxy served 0 of 5,400 requests while
the rules endpoint hung: ``check_enforcement`` refreshed a stale cache inline,
through five attempts and their sleeps, on whatever thread called it -- the
proxy's only event loop -- and a failed refresh left the cache stale, so the
very next request did it all again.

The contract these tests pin:

* ``check_enforcement`` judges a request against the rules already cached and
  returns at once, whatever the endpoint is doing;
* the refresh runs on the poller thread, one at a time per process;
* a failed refresh is not retried for ``_refresh_failure_backoff_seconds()``,
  however many requests arrive meanwhile;
* with no rules cached yet and Revenium down, the default mode fails open
  without waiting, and ``REVENIUM_CB_FAIL_MODE=closed`` refuses without
  waiting.

The endpoint here is a real local socket (``enforcement_stub_server``), never
a stand-in for ``httpx`` or for ``check_enforcement``: the defect lived in the
request, the retries and the sleeps that those stand-ins skip.
"""
import threading
import time

import httpx
import pytest

from revenium_middleware._core import enforcement
from revenium_middleware._core.exceptions import BudgetExceededError

from .conftest import FakeClock, make_response, stale_cache_timestamp, stub_get
from .enforcement_stub_server import (
    FAILURE_MODES,
    HANG,
    HEALTHY,
    UNAVAILABLE,
    EnforcementStub,
    point_enforcement_at,
    shut_down,
    wait_until,
)

# Longest one check may take. The check itself is a few dict lookups; the
# bound only has to sit far below the smallest stall it guards against, which
# is one 0.3 s round trip to the stub.
CHECK_BOUND_SECONDS = 0.05


def _timed_check(metadata=None):
    started = time.perf_counter()
    try:
        enforcement.check_enforcement(metadata or {"organizationName": "AcmeCorp"})
    finally:
        elapsed = time.perf_counter() - started
    return elapsed


@pytest.fixture(autouse=True)
def restore_the_cache(monkeypatch):
    """Put back whatever a refresh in these tests installs, so no rule outlives its test."""
    for name in ("_cached_rules", "_cached_org_unit_blocks", "_cached_org_unit_block_balances",
                 "_cached_org_unit_warnings", "_cache_timestamp", "_cache_initialized"):
        monkeypatch.setattr(enforcement, name, getattr(enforcement, name))


@pytest.fixture
def endpoint(monkeypatch):
    """Start a stub in the requested mode and aim the circuit breaker at it."""
    stubs = []

    def start(mode, rules=None, poll_interval=60):
        stub = EnforcementStub(mode, rules)
        stubs.append(stub)
        point_enforcement_at(monkeypatch, stub, poll_interval)
        return stub

    yield start
    for stub in stubs:
        shut_down(stub)


class TestTheCheckNeverWaitsOnTheEndpoint:

    @pytest.mark.parametrize("mode", FAILURE_MODES)
    def test_thirty_cold_start_checks_each_return_at_once_and_fail_open(self, endpoint, mode):
        endpoint(mode)

        for attempt in range(30):
            elapsed = _timed_check()
            assert elapsed < CHECK_BOUND_SECONDS, (
                f"check {attempt + 1} waited {elapsed:.3f}s on a {mode} rules endpoint"
            )

    def test_a_hanging_endpoint_holds_one_request_and_no_caller(self, endpoint):
        """The ticket's live failure, against the real check: nobody waits on the hang."""
        stub = endpoint(HANG)

        callers = 50
        durations = []
        lock = threading.Lock()

        def caller():
            elapsed = _timed_check()
            with lock:
                durations.append(elapsed)

        threads = [threading.Thread(target=caller) for _ in range(callers)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=2)

        assert len(durations) == callers, "a caller was still parked on the hung refresh"
        assert max(durations) < CHECK_BOUND_SECONDS
        assert wait_until(lambda: stub.hits == 1)
        time.sleep(0.2)
        assert stub.hits == 1, "more than one refresh was in flight against the hang"
        pollers = [t for t in threading.enumerate() if t.name == "revenium-enforcement-poll"]
        assert len(pollers) == 1

    def test_cold_start_in_closed_mode_refuses_at_once_instead_of_waiting(self, endpoint, monkeypatch):
        endpoint(HANG)
        monkeypatch.setenv("REVENIUM_CB_FAIL_MODE", "closed")

        started = time.perf_counter()
        with pytest.raises(BudgetExceededError, match="uninitialized"):
            enforcement.check_enforcement({})
        assert time.perf_counter() - started < CHECK_BOUND_SECONDS


class TestRulesStillArriveAndStillBlock:
    """Moving the refresh off the caller changes when rules land, never what they block."""

    def test_a_tripped_rule_blocks_once_the_poller_has_fetched_it(self, endpoint):
        endpoint(HEALTHY, rules=[{"ruleId": 7, "name": "Team Budget", "breached": True,
                                  "threshold": 10.0, "currentValue": 11.5}])

        _timed_check()  # starts the poller; nothing is cached yet, so this passes
        assert wait_until(lambda: enforcement._cache_initialized)

        with pytest.raises(BudgetExceededError) as blocked:
            enforcement.check_enforcement({"organizationName": "AcmeCorp"})
        assert blocked.value.rule_id == 7

    def test_a_stale_cache_is_refreshed_by_the_poller_when_a_request_asks(self, endpoint, monkeypatch):
        """A poll interval past the TTL still refreshes on demand, just not on the caller."""
        stub = endpoint(HEALTHY, rules=[{"ruleId": 9, "breached": False}], poll_interval=3600)
        _timed_check()
        assert wait_until(lambda: stub.hits == 1 and enforcement._cache_initialized)

        monkeypatch.setattr(enforcement, "_cache_timestamp", stale_cache_timestamp())
        assert _timed_check() < CHECK_BOUND_SECONDS

        assert wait_until(lambda: stub.hits == 2), "a stale cache was never refreshed"


class TestAFailedRefreshBacksOff:

    def test_concurrent_requests_through_an_outage_cause_one_fetch_per_interval(
        self, endpoint, monkeypatch
    ):
        """Sixteen threads hammering the check for three intervals: about three fetches."""
        interval = 1
        monkeypatch.setattr(enforcement, "_FETCH_MAX_ATTEMPTS", 1)
        stub = endpoint(UNAVAILABLE, poll_interval=interval)

        window = 3.2
        stop = threading.Event()
        checks = []

        def hammer():
            while not stop.is_set():
                enforcement.check_enforcement({"organizationName": "AcmeCorp"})
                checks.append(1)
                time.sleep(0.002)

        threads = [threading.Thread(target=hammer) for _ in range(16)]
        for thread in threads:
            thread.start()
        time.sleep(window)
        stop.set()
        for thread in threads:
            thread.join(timeout=2)

        allowed = int(window // interval) + 1
        assert len(checks) > 1000
        assert 2 <= stub.hits <= allowed, (
            f"{stub.hits} fetches in {window}s with a {interval}s backoff"
        )

    @pytest.mark.parametrize("failure", [
        httpx.ReadTimeout("hung"),
        httpx.ConnectError("refused"),
        make_response(503),
        make_response(403),
        make_response(401),
    ], ids=["timeout", "refused", "503", "403", "401"])
    def test_every_kind_of_failure_starts_the_backoff(self, fetch_env, failure):
        monkeypatch, _sleeps = fetch_env
        clock = FakeClock()
        monkeypatch.setattr(enforcement, "time", clock)
        stub = stub_get(monkeypatch, [failure])

        enforcement._refresh_cache()
        first = stub.calls
        enforcement._refresh_cache()

        assert first >= 1
        assert stub.calls == first, "a second refresh went out inside the backoff"

        clock.advance(enforcement._refresh_failure_backoff_seconds() + 1)
        enforcement._refresh_cache()
        assert stub.calls > first, "the backoff never released the refresh"

    def test_the_backoff_is_the_poll_interval(self, monkeypatch):
        monkeypatch.setenv("REVENIUM_CB_POLL_INTERVAL_SECONDS", "45")
        assert enforcement._refresh_failure_backoff_seconds() == 45.0
        monkeypatch.delenv("REVENIUM_CB_POLL_INTERVAL_SECONDS")
        assert enforcement._refresh_failure_backoff_seconds() == float(enforcement._DEFAULT_POLL_INTERVAL)

    def test_requests_inside_the_backoff_do_not_wake_the_poller(self, fetch_env):
        monkeypatch, _sleeps = fetch_env
        clock = FakeClock()
        monkeypatch.setattr(enforcement, "time", clock)
        monkeypatch.setattr(enforcement, "_cache_timestamp", stale_cache_timestamp())
        stub_get(monkeypatch, [make_response(503)])
        enforcement._refresh_cache()
        enforcement._poll_wakeup.clear()

        for _ in range(50):
            enforcement._get_rules()
        assert not enforcement._poll_wakeup.is_set()

        clock.advance(enforcement._refresh_failure_backoff_seconds() + 1)
        enforcement._get_rules()
        assert enforcement._poll_wakeup.is_set()

    def test_a_success_after_the_backoff_serves_the_new_rules(self, fetch_env):
        monkeypatch, _sleeps = fetch_env
        clock = FakeClock()
        monkeypatch.setattr(enforcement, "time", clock)
        monkeypatch.setattr(enforcement, "_cached_rules", [{"ruleId": 1}])
        stub_get(monkeypatch, [make_response(503),
                               make_response(200, json_body={"rules": [{"ruleId": 2}]})])
        monkeypatch.setattr(enforcement, "_FETCH_MAX_ATTEMPTS", 1)

        enforcement._refresh_cache()
        assert enforcement._cached_rules == [{"ruleId": 1}], "a failure must keep the cached rules"

        clock.advance(enforcement._refresh_failure_backoff_seconds() + 1)
        enforcement._refresh_cache()
        assert enforcement._cached_rules == [{"ruleId": 2}]


class TestOneRefreshAtATime:

    def test_a_refresh_already_in_flight_is_not_joined_by_a_second(self, fetch_env):
        monkeypatch, _sleeps = fetch_env
        entered = threading.Event()
        release = threading.Event()
        calls = []

        def slow_get(*_args, **_kwargs):
            calls.append(1)
            entered.set()
            release.wait(2)
            return make_response(200)

        monkeypatch.setattr(enforcement.httpx, "get", slow_get)
        first = threading.Thread(target=enforcement._refresh_cache)
        first.start()
        assert entered.wait(2)

        enforcement._refresh_cache()
        release.set()
        first.join(2)

        assert len(calls) == 1

    def test_get_rules_never_fetches_on_the_callers_thread(self, fetch_env):
        monkeypatch, _sleeps = fetch_env
        stub = stub_get(monkeypatch, [make_response(200)])
        monkeypatch.setattr(enforcement, "_cache_timestamp", stale_cache_timestamp())

        enforcement._get_rules()

        assert stub.calls == 0
        assert enforcement._poll_wakeup.is_set(), "a stale cache did not ask the poller to refresh"

    def test_a_forked_child_is_not_left_a_refresh_lock_its_parent_held(self, fetch_env):
        monkeypatch, _sleeps = fetch_env
        held_by_the_parents_poller = threading.Lock()
        held_by_the_parents_poller.acquire()
        monkeypatch.setattr(enforcement, "_refresh_lock", held_by_the_parents_poller)
        stub = stub_get(monkeypatch, [make_response(200, json_body={"rules": [{"ruleId": 3}]})])

        enforcement._forget_parent_refresh()
        enforcement._refresh_cache()

        assert stub.calls == 1
        assert enforcement._cached_rules == [{"ruleId": 3}]


class TestThePollerKeepsItsCadenceThroughABackoff:

    def test_it_sleeps_out_a_running_cooldown_rather_than_a_whole_interval(self, monkeypatch):
        clock = FakeClock()
        monkeypatch.setattr(enforcement, "time", clock)
        monkeypatch.setattr(enforcement, "_refresh_cooldown_until", clock.monotonic() + 0.25)

        assert enforcement._next_poll_wait_seconds(60) == pytest.approx(0.25)

    def test_it_sleeps_one_interval_when_no_cooldown_is_running(self, monkeypatch):
        clock = FakeClock()
        monkeypatch.setattr(enforcement, "time", clock)
        monkeypatch.setattr(enforcement, "_refresh_cooldown_until", clock.monotonic() - 1)

        assert enforcement._next_poll_wait_seconds(60) == 60

    def test_a_long_poll_interval_is_not_cut_to_the_retry_after_ceiling(self, fetch_env):
        """A one-hour interval means one attempt an hour through an outage, not one every 5 minutes."""
        monkeypatch, _sleeps = fetch_env
        monkeypatch.setenv("REVENIUM_CB_POLL_INTERVAL_SECONDS", "3600")
        clock = FakeClock()
        monkeypatch.setattr(enforcement, "time", clock)
        stub = stub_get(monkeypatch, [make_response(503)])
        monkeypatch.setattr(enforcement, "_FETCH_MAX_ATTEMPTS", 1)

        enforcement._refresh_cache()
        assert enforcement._next_poll_wait_seconds(3600) == pytest.approx(3600)

        clock.advance(enforcement._REFRESH_COOLDOWN_CEILING_SECONDS + 1)
        enforcement._refresh_cache()
        assert stub.calls == 1, "the failure backoff was clamped to the Retry-After ceiling"

        clock.advance(3600)
        enforcement._refresh_cache()
        assert stub.calls == 2

    def test_a_retry_after_cooldown_is_still_clamped_to_the_ceiling(self, fetch_env):
        monkeypatch, _sleeps = fetch_env
        clock = FakeClock()
        monkeypatch.setattr(enforcement, "time", clock)

        remaining = enforcement._begin_refresh_cooldown(999999)

        assert remaining == pytest.approx(enforcement._REFRESH_COOLDOWN_CEILING_SECONDS)
