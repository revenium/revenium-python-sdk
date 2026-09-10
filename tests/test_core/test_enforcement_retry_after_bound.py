"""A 429 on the rule fetch must not be retried earlier than the server allowed.

The enforcement fetch used to hand ``Retry-After`` to the same constant that
caps its own exponential backoff, so a server asking for 30 s was truncated to
8 s and retried at 8 s -- the one thing a 429 asks a client not to do. The
bound is now a give-up threshold, not a ``min()`` on the wait, and it has three
parts:

* a ``Retry-After`` that fits the remaining budget is waited out in full, then
  retried;
* ``_RETRY_AFTER_GIVE_UP_SECONDS`` is a budget for the whole ``_fetch_rules``
  call, so a server repeating an in-bound interval cannot park a caller for a
  multiple of the bound;
* a ``Retry-After`` that does not fit is neither waited out nor retried. The
  fetch records a cooldown for the interval instead -- capped by a sanity
  ceiling so one absurd header cannot suppress refreshes for hours -- so the
  server's request is still honoured, and fails open on the cached rules: one
  control-plane request while throttled, not one per customer request.

The bound is 20 s rather than the metering clients' 60 s because this fetch is
synchronous: ``_get_rules`` refreshes on the caller's thread once the cache is
past ``_CACHE_TTL``, so the wait parks a customer request. See
``docs/conventions/retry-after.md`` for the numbers in every SDK.
"""
import datetime
import threading
from email.utils import format_datetime

import pytest

from revenium_middleware._core import enforcement

from .conftest import FakeClock, make_response, stale_cache_timestamp, stub_get


def throttled(retry_after):
    """A 429 asking for ``retry_after`` seconds."""
    return make_response(429, headers={"Retry-After": str(retry_after)})


class TestHonoursTheFullIntervalWithinTheBound:
    """Within the bound the server's number is used as sent, never truncated."""

    def test_retry_after_5s_waits_the_full_5s_then_retries(self, fetch_env):
        monkeypatch, sleeps = fetch_env
        stub = stub_get(monkeypatch, [
            throttled(5),
            make_response(200, json_body={"rules": [{"ruleId": 1}]}),
        ])

        fetched = enforcement._fetch_rules()

        assert sleeps == [5.0]
        assert stub.calls == 2
        assert fetched.rules == [{"ruleId": 1}]

    def test_a_wait_longer_than_the_backoff_cap_is_still_honoured_in_full(self, fetch_env):
        """The regression itself: 15 s used to be truncated to the 8 s backoff cap."""
        monkeypatch, sleeps = fetch_env
        stub = stub_get(monkeypatch, [throttled(15), make_response(200)])

        enforcement._fetch_rules()

        assert sleeps == [15.0]
        assert stub.calls == 2

    def test_the_bound_itself_is_honoured_not_given_up_on(self, fetch_env):
        monkeypatch, sleeps = fetch_env
        bound = enforcement._RETRY_AFTER_GIVE_UP_SECONDS
        stub_get(monkeypatch, [throttled(int(bound)), make_response(200)])

        enforcement._fetch_rules()

        assert sleeps == [bound]


class TestTheBoundIsABudgetForTheWholeCall:
    """The bound limits the total wait per fetch, not each response separately."""

    def test_four_in_bound_intervals_do_not_park_the_caller_for_four_bounds(self, fetch_env):
        """A per-response threshold let ``Retry-After: 20`` x4 sleep 80 s."""
        monkeypatch, sleeps = fetch_env
        bound = enforcement._RETRY_AFTER_GIVE_UP_SECONDS
        stub = stub_get(monkeypatch, [throttled(int(bound))] * 10)

        assert enforcement._fetch_rules() is None
        assert sum(sleeps) <= bound
        assert sleeps == [bound]  # the first is affordable, the second is not
        assert stub.calls == 2

    def test_the_budget_is_spent_across_attempts_then_gives_up(self, fetch_env):
        monkeypatch, sleeps = fetch_env
        stub = stub_get(monkeypatch, [throttled(8)] * 10)

        assert enforcement._fetch_rules() is None
        # 8 + 8 fits in 20; the third 8 does not fit the remaining 4, and is
        # not truncated to 4 either -- the fetch gives up instead.
        assert sleeps == [8.0, 8.0]
        assert sum(sleeps) <= enforcement._RETRY_AFTER_GIVE_UP_SECONDS
        assert stub.calls == 3

    def test_our_own_backoff_does_not_spend_the_servers_budget(self, fetch_env):
        """A 503 with no header costs us backoff, not the caller's Retry-After allowance."""
        monkeypatch, sleeps = fetch_env
        stub_get(monkeypatch, [make_response(503), make_response(503), throttled(20),
                               make_response(200)])

        enforcement._fetch_rules()

        # Two backoff waits, then the full 20 s the server asked for.
        assert sleeps[-1] == 20.0
        assert len(sleeps) == 3


class TestGivesUpBeyondTheBound:
    """Beyond the bound: no wait, no retry, previous cache left in force."""

    def test_retry_after_120s_produces_no_sleep_and_no_retry(self, fetch_env):
        monkeypatch, sleeps = fetch_env
        stub = stub_get(monkeypatch, [throttled(120)] * 10)

        fetched = enforcement._fetch_rules()

        assert sleeps == []
        assert stub.calls == 1
        assert fetched is None  # fail open: the caller keeps the previous cache

    def test_an_http_date_beyond_the_bound_also_gives_up(self, fetch_env):
        """Both RFC 9110 forms are parsed, so both reach the give-up threshold."""
        monkeypatch, sleeps = fetch_env
        when = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(seconds=300)
        stub = stub_get(monkeypatch, [
            make_response(429, headers={"Retry-After": format_datetime(when, usegmt=True)}),
        ] * 10)

        assert enforcement._fetch_rules() is None
        assert sleeps == []
        assert stub.calls == 1

    def test_giving_up_leaves_the_cached_rules_untouched(self, fetch_env):
        monkeypatch, _ = fetch_env
        stub_get(monkeypatch, [throttled(120)] * 10)
        monkeypatch.setattr(enforcement, "_cached_rules", [{"ruleId": 7777}])
        monkeypatch.setattr(enforcement, "_cached_org_unit_blocks", {"keep@me.test": 7777})

        enforcement._refresh_cache()

        assert enforcement._cached_rules == [{"ruleId": 7777}]
        assert enforcement._cached_org_unit_blocks == {"keep@me.test": 7777}


class TestGivingUpStartsACooldown:
    """Declining to wait is only half of honouring the header.

    Without a cooldown the cache is still stale on the next customer request,
    which triggers another synchronous refresh and another control-plane
    request -- one per customer request for as long as the throttling lasts.
    """

    @pytest.fixture()
    def clocked(self, fetch_env):
        monkeypatch, sleeps = fetch_env
        clock = FakeClock()
        monkeypatch.setattr(enforcement, "time", clock)
        monkeypatch.setattr(enforcement, "_cached_rules", [{"ruleId": 1, "breached": False}])
        monkeypatch.setattr(enforcement, "_cached_org_unit_blocks", {})
        monkeypatch.setattr(enforcement, "_cache_initialized", True)
        monkeypatch.setattr(enforcement, "_cache_timestamp", stale_cache_timestamp())
        return monkeypatch, sleeps, clock

    def test_fifty_customer_requests_inside_the_window_make_one_control_plane_request(self, clocked):
        monkeypatch, sleeps, clock = clocked
        stub = stub_get(monkeypatch, [throttled(120), make_response(200, json_body={"rules": []})])

        for _ in range(50):
            rules, _blocks, initialized = enforcement._get_rules()
            assert rules == [{"ruleId": 1, "breached": False}]  # cached rules serve every one
            assert initialized is True

        assert stub.calls == 1, "the cooldown did not suppress the request-driven refresh"
        assert sleeps == [], "a customer request was parked waiting on the 429"

    def test_the_next_request_after_the_deadline_refreshes(self, clocked):
        monkeypatch, sleeps, clock = clocked
        stub = stub_get(monkeypatch, [
            throttled(120),
            make_response(200, json_body={"rules": [{"ruleId": 2}]}),
        ])

        enforcement._get_rules()
        assert stub.calls == 1

        clock.advance(119)
        enforcement._get_rules()
        assert stub.calls == 1, "refreshed before the interval the server asked for"

        clock.advance(2)  # now past now+120
        rules, _blocks, _initialized = enforcement._get_rules()

        assert stub.calls == 2
        assert rules == [{"ruleId": 2}]
        assert sleeps == []

    def test_the_background_poller_also_waits_out_the_interval(self, clocked):
        """The poll interval is shorter than a long Retry-After; it must not race it."""
        monkeypatch, _sleeps, clock = clocked
        stub = stub_get(monkeypatch, [throttled(120), make_response(200)])

        enforcement._refresh_cache()
        assert stub.calls == 1

        clock.advance(60)  # one poll interval
        enforcement._refresh_cache()

        assert stub.calls == 1

    def test_an_absurd_retry_after_cannot_suppress_refreshes_for_hours(self, clocked):
        """The ceiling bounds the cooldown, and nothing waits inside the window.

        ``Retry-After: 999999`` is malformed in practice. Honouring it literally
        would leave the cached limits unrefreshed for eleven days on the
        strength of one response, so the cooldown is clamped -- which is not
        the forbidden ``min()`` on a wait: the fetch has already given up, so
        no caller is parked and no request goes out early.
        """
        monkeypatch, sleeps, clock = clocked
        ceiling = enforcement._REFRESH_COOLDOWN_CEILING_SECONDS
        stub = stub_get(monkeypatch, [
            throttled(999999),
            make_response(200, json_body={"rules": [{"ruleId": 3}]}),
        ])

        enforcement._get_rules()
        assert stub.calls == 1
        assert sleeps == []

        clock.advance(ceiling - 1)
        enforcement._get_rules()
        assert stub.calls == 1, "refreshed before the ceiling elapsed"

        clock.advance(2)
        rules, _blocks, _initialized = enforcement._get_rules()

        assert stub.calls == 2, "the ceiling did not release the cooldown"
        assert rules == [{"ruleId": 3}]
        assert sleeps == [], "the ceiling must bound a cooldown, never park a caller"

    def test_a_longer_cooldown_is_not_shortened_by_a_later_one(self, clocked):
        monkeypatch, _sleeps, clock = clocked
        monkeypatch.setattr(enforcement, "_refresh_cooldown_until", clock.monotonic() + 300)
        stub = stub_get(monkeypatch, [throttled(30), make_response(200)])

        clock.advance(60)
        enforcement._refresh_cache()

        assert stub.calls == 0, "a shorter Retry-After cut a longer cooldown short"


class TestPathShapeAndBound:
    """One assertion pinning both halves of the decision.

    Shape: SYNCHRONOUS. ``_get_rules`` calls ``_refresh_cache`` on the
    caller's thread once the cache is older than ``_CACHE_TTL``, so a
    Retry-After wait here parks a customer request that is already blocked on
    its own provider call. Bound: 20 s, deliberately shorter than the 60 s the
    metering clients use.

    If this path ever becomes background-only (the shape the Go SDK's ``poll``
    goroutine has), this is the test that has to change: a wait that costs a
    customer nothing gets the 60 s bound instead.
    """

    def test_bound_is_20s_because_the_refresh_is_synchronous_on_the_callers_thread(self, fetch_env):
        monkeypatch, _ = fetch_env

        assert enforcement._RETRY_AFTER_GIVE_UP_SECONDS == 20.0
        # A server-supplied give-up threshold and a cap on the backoff we chose
        # ourselves are two different decisions, so they stay two constants.
        assert enforcement._RETRY_AFTER_GIVE_UP_SECONDS != enforcement._FETCH_BACKOFF_CAP

        caller = threading.current_thread()
        refreshed_on = []
        monkeypatch.setattr(enforcement, "_refresh_cache",
                            lambda: refreshed_on.append(threading.current_thread()))
        monkeypatch.setattr(enforcement, "_cache_timestamp", stale_cache_timestamp())

        enforcement._get_rules()

        assert refreshed_on == [caller], "the refresh no longer runs on the caller's thread"


class TestNoCustomerRequestIsHarmedWhenTheFetchGivesUp:
    """Giving up is a fail-open, and the cached verdict stays authoritative."""

    @pytest.fixture(autouse=True)
    def seeded_cache(self, fetch_env):
        monkeypatch, _ = fetch_env
        monkeypatch.setenv("REVENIUM_CIRCUIT_BREAKER_ENABLED", "true")
        monkeypatch.delenv("REVENIUM_BYPASS", raising=False)
        monkeypatch.delenv("REVENIUM_CB_FAIL_MODE", raising=False)
        # Keep the seeded cache: no disk snapshot load, and no background
        # poller racing the assertions.
        monkeypatch.setattr(enforcement, "_disk_load_attempted", True)
        monkeypatch.setattr(enforcement, "_ensure_poller_running", lambda: None)
        monkeypatch.setattr(enforcement, "_cached_org_unit_blocks", {})
        monkeypatch.setattr(enforcement, "_cache_initialized", True)
        monkeypatch.setattr(enforcement, "_cache_timestamp", stale_cache_timestamp())
        return monkeypatch

    def test_a_stale_cache_and_a_120s_retry_after_still_lets_the_request_through(self, fetch_env):
        monkeypatch, sleeps = fetch_env
        stub = stub_get(monkeypatch, [throttled(120)] * 10)
        monkeypatch.setattr(enforcement, "_cached_rules", [{"ruleId": 1, "breached": False}])

        enforcement.check_enforcement({"organizationName": "AcmeCorp"})  # must not raise

        assert sleeps == [], "a customer request was parked waiting on a 429"
        assert stub.calls == 1, "the fetch retried past a Retry-After it could not honour"

    def test_a_tripped_cached_rule_still_blocks_after_the_fetch_gives_up(self, fetch_env):
        """Failing open means keeping the cached rules, not stopping enforcement."""
        monkeypatch, sleeps = fetch_env
        stub_get(monkeypatch, [throttled(120)] * 10)
        monkeypatch.setattr(enforcement, "_cached_rules", [{
            "ruleId": 1, "name": "Monthly cap", "breached": True, "currentValue": 120.0,
        }])

        with pytest.raises(enforcement.BudgetExceededError):
            enforcement.check_enforcement({"organizationName": "AcmeCorp"})

        assert sleeps == []

    def test_fifty_requests_while_throttled_stay_unblocked_and_unretried(self, fetch_env):
        """The cooldown must not turn into a fail-closed or a stalled caller."""
        monkeypatch, sleeps = fetch_env
        stub = stub_get(monkeypatch, [throttled(120)] * 10)
        monkeypatch.setattr(enforcement, "_cached_rules", [{"ruleId": 1, "breached": False}])

        for _ in range(50):
            enforcement.check_enforcement({"organizationName": "AcmeCorp"})  # must not raise

        assert stub.calls == 1
        assert sleeps == []
