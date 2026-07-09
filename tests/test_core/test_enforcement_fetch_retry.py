"""The enforcement rule fetch must ride out 429/5xx spikes with backoff.

A bare GET meant any 429 failed the refresh outright, leaving spend controls
on stale limits until the next poll interval. The fetch now retries
retryable failures (408/429/5xx/connection errors) with exponential backoff
capped at 8s, honors Retry-After (delta-seconds and HTTP-date), and on
exhaustion fails open -- an enforcement-refresh outage must never become a
customer traffic outage.
"""
import datetime
from email.utils import format_datetime
from types import SimpleNamespace

import httpx
import pytest

from revenium_middleware._core import enforcement

URL = "https://api.test/v2/api/ai/enforcement-rules/team-1"


def make_response(status_code, headers=None, json_body=None):
    request = httpx.Request("GET", URL)
    return httpx.Response(status_code, headers=headers or {},
                          json=json_body if json_body is not None else {"rules": []},
                          request=request)


class SequencedGet:
    """httpx.get stand-in returning (or raising) a scripted sequence."""

    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = 0

    def __call__(self, *args, **kwargs):
        outcome = self.outcomes[min(self.calls, len(self.outcomes) - 1)]
        self.calls += 1
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


@pytest.fixture()
def fetch_env(monkeypatch):
    monkeypatch.setenv("REVENIUM_METERING_API_KEY", "hak_enforcement_test")
    monkeypatch.setenv("REVENIUM_TEAM_ID", "team-1")
    sleeps = []
    monkeypatch.setattr(enforcement, "_sleep", lambda seconds: sleeps.append(seconds), raising=False)
    return monkeypatch, sleeps


def stub_get(monkeypatch, outcomes):
    stub = SequencedGet(outcomes)
    monkeypatch.setattr(enforcement.httpx, "get", stub)
    return stub


class TestRetryOnTransientFailures:
    def test_429_spike_is_ridden_out(self, fetch_env):
        monkeypatch, sleeps = fetch_env
        stub = stub_get(monkeypatch, [
            make_response(429),
            make_response(429),
            make_response(200, json_body={"rules": [{"id": "r1"}]}),
        ])

        rules = enforcement._fetch_rules()

        assert rules == [{"id": "r1"}]
        assert stub.calls == 3
        assert len(sleeps) == 2

    @pytest.mark.parametrize("status", [408, 500, 502, 503, 504])
    def test_retryable_statuses(self, fetch_env, status):
        monkeypatch, sleeps = fetch_env
        stub = stub_get(monkeypatch, [make_response(status), make_response(200)])

        rules = enforcement._fetch_rules()

        assert rules == []
        assert stub.calls == 2

    def test_connection_errors_are_retried(self, fetch_env):
        monkeypatch, sleeps = fetch_env
        stub = stub_get(monkeypatch, [
            httpx.ConnectError("refused"),
            httpx.ReadTimeout("slow"),
            make_response(200, json_body={"rules": []}),
        ])

        rules = enforcement._fetch_rules()

        assert rules == []
        assert stub.calls == 3

    def test_backoff_is_exponential_and_capped_at_8s(self, fetch_env):
        monkeypatch, sleeps = fetch_env
        stub = stub_get(monkeypatch, [make_response(503)] * 10)

        enforcement._fetch_rules()

        assert sleeps == sorted(sleeps)  # non-decreasing
        assert all(delay <= 8.0 for delay in sleeps)
        assert sleeps[0] < sleeps[-1]  # actually grows


class TestRetryAfter:
    def test_delta_seconds_honored(self, fetch_env):
        monkeypatch, sleeps = fetch_env
        stub_get(monkeypatch, [
            make_response(429, headers={"Retry-After": "3"}),
            make_response(200),
        ])

        enforcement._fetch_rules()

        assert sleeps == [3.0]

    def test_http_date_honored(self, fetch_env):
        monkeypatch, sleeps = fetch_env
        when = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(seconds=5)
        stub_get(monkeypatch, [
            make_response(429, headers={"Retry-After": format_datetime(when, usegmt=True)}),
            make_response(200),
        ])

        enforcement._fetch_rules()

        assert len(sleeps) == 1
        assert 3.0 <= sleeps[0] <= 8.0

    def test_retry_after_clamped_to_8s(self, fetch_env):
        monkeypatch, sleeps = fetch_env
        stub_get(monkeypatch, [
            make_response(429, headers={"Retry-After": "120"}),
            make_response(200),
        ])

        enforcement._fetch_rules()

        assert sleeps == [8.0]

    def test_unparseable_retry_after_falls_back_to_backoff(self, fetch_env):
        monkeypatch, sleeps = fetch_env
        stub_get(monkeypatch, [
            make_response(429, headers={"Retry-After": "not-a-date"}),
            make_response(200),
        ])

        enforcement._fetch_rules()

        assert len(sleeps) == 1
        assert 0 < sleeps[0] <= 8.0


class TestFailOpen:
    def test_permanent_failure_does_not_retry(self, fetch_env):
        monkeypatch, sleeps = fetch_env
        stub = stub_get(monkeypatch, [make_response(401)])

        rules = enforcement._fetch_rules()

        assert rules is None  # fail open: caller keeps previous cache
        assert stub.calls == 1
        assert sleeps == []

    def test_exhaustion_fails_open_without_raising(self, fetch_env):
        monkeypatch, sleeps = fetch_env
        stub = stub_get(monkeypatch, [make_response(503)] * 20)

        rules = enforcement._fetch_rules()

        assert rules is None
        assert stub.calls == enforcement._FETCH_MAX_ATTEMPTS  # rode the spike to exhaustion

    def test_shutdown_during_backoff_stops_retrying(self, fetch_env):
        """A stop event set mid-backoff must abort the remaining attempts."""
        monkeypatch, sleeps = fetch_env
        stub = stub_get(monkeypatch, [make_response(503)] * 20)
        # _sleep returns True when shutdown was signalled during the wait.
        monkeypatch.setattr(enforcement, "_sleep",
                            lambda seconds: sleeps.append(seconds) or True)

        rules = enforcement._fetch_rules()

        assert rules is None  # fail open: caller keeps previous cache
        assert stub.calls == 1  # no network calls after shutdown
        assert sleeps == [enforcement._FETCH_BACKOFF_INITIAL]

    def test_exhaustion_preserves_previous_cache(self, fetch_env):
        monkeypatch, _ = fetch_env
        stub_get(monkeypatch, [make_response(503)] * 20)
        monkeypatch.setattr(enforcement, "_cached_rules", [{"id": "keep-me"}])

        enforcement._refresh_cache()

        assert enforcement._cached_rules == [{"id": "keep-me"}]

    def test_check_enforcement_never_blocks_on_fetch_outage(self, fetch_env):
        """Card verification: enforcement degrades open, traffic flows."""
        monkeypatch, _ = fetch_env
        monkeypatch.setenv("REVENIUM_CIRCUIT_BREAKER_ENABLED", "true")
        stub_get(monkeypatch, [httpx.ConnectError("backend down")] * 20)

        enforcement.check_enforcement({"organizationName": "AcmeCorp"})  # must not raise
