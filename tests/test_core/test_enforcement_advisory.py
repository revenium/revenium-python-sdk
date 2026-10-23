"""BACK-4890: notify-only per-person budgets are read on demand, never enforced.

From API 2.25.0 the enforcement-rules read appends a team's advisory budgets
(``source`` ``ANOMALY``, ``action`` ``NOTIFY``) after its cost-control rows, but
only when the request carries ``include=advisory`` and no ``ruleId``. The SDK
asks for them from ``fetch_advisory_budgets`` alone: an anomaly's numeric id
can equal a cost control's, so the cache ``check_enforcement`` reads must keep
receiving the cost-control rows only.

These run against the real local endpoint in ``enforcement_stub_server``, so
the query string asserted on is the one that went over the wire.
"""
import pytest

from revenium_middleware._core import enforcement, fetch_advisory_budgets

from .enforcement_stub_server import (
    HEALTHY,
    NO_CONTENT,
    REFUSE,
    EnforcementStub,
    point_enforcement_at,
    shut_down,
)

COST_CONTROL = {
    "ruleId": 41,
    "name": "Monthly cap",
    "action": "BLOCK",
    "groupBy": None,
    "shadowMode": False,
    "breached": False,
    "threshold": 500.0,
}

ADVISORY = {
    "ruleId": 41,
    "name": "Per-person spend alert",
    "action": "NOTIFY",
    "source": "ANOMALY",
    "groupBy": "SUBSCRIBER",
    "shadowMode": False,
    "warnThreshold": None,
    "threshold": 20.0,
    "readingAt": "2026-10-20T12:00:00Z",
    "groupBreakdown": [
        {"groupValue": "jane@acme.test", "currentValue": 31.5, "breached": True},
    ],
}


@pytest.fixture()
def serve(monkeypatch):
    """Start a stub in the given mode with enforcement aimed at it; waits are recorded, never taken."""
    stubs = []
    sleeps = []
    monkeypatch.setattr(enforcement, "_sleep", lambda seconds: sleeps.append(seconds) or False)

    def start(mode, rules=None, advisory_rules=None):
        stub = EnforcementStub(mode, rules=rules, advisory_rules=advisory_rules)
        stubs.append(stub)
        point_enforcement_at(monkeypatch, stub)
        return stub

    yield start
    for stub in stubs:
        shut_down(stub)


class TestFetchAdvisoryBudgets:
    def test_the_request_asks_for_advisory_rows_on_the_whole_team_read(self, serve):
        stub = serve(HEALTHY, rules=[COST_CONTROL], advisory_rules=[ADVISORY])

        fetch_advisory_budgets()

        assert stub.queries == [{"include": ["advisory"]}]

    def test_only_the_anomaly_rows_of_a_mixed_payload_are_returned(self, serve):
        serve(HEALTHY, rules=[COST_CONTROL], advisory_rules=[ADVISORY])

        assert fetch_advisory_budgets() == [ADVISORY]

    def test_an_advisory_only_payload_returns_its_rows(self, serve):
        second = dict(ADVISORY, ruleId=42, readingAt=None)
        serve(HEALTHY, advisory_rules=[ADVISORY, second])

        assert fetch_advisory_budgets() == [ADVISORY, second]

    def test_no_content_is_an_empty_list(self, serve):
        serve(NO_CONTENT)

        assert fetch_advisory_budgets() == []

    def test_an_unreachable_server_is_none(self, serve):
        serve(REFUSE)

        assert fetch_advisory_budgets() is None

    def test_the_rule_cache_is_neither_read_nor_written(self, serve, monkeypatch):
        serve(HEALTHY, rules=[COST_CONTROL], advisory_rules=[ADVISORY])
        monkeypatch.setattr(enforcement, "_cached_rules", [{"ruleId": 7}])

        fetch_advisory_budgets()

        assert enforcement._cached_rules == [{"ruleId": 7}]
        assert enforcement._cache_initialized is False


class TestTheCacheStaysCostControlOnly:
    def test_a_poller_refresh_sends_no_include_and_caches_no_anomaly_row(self, serve):
        stub = serve(HEALTHY, rules=[COST_CONTROL], advisory_rules=[ADVISORY])

        enforcement._refresh_cache()

        assert stub.queries == [{}]
        assert enforcement._cached_rules == [COST_CONTROL]

    def test_advisory_rows_cannot_be_asked_for_with_a_rule_id(self, serve):
        stub = serve(HEALTHY, advisory_rules=[ADVISORY])

        with pytest.raises(ValueError):
            enforcement._fetch_rules("mN3xpQz", include_advisory=True)

        assert stub.hits == 0


class TestAnAdvisoryRowNeverBlocks:
    def test_a_breached_notify_row_lets_the_call_through(self, department_cache, caplog):
        department_cache([ADVISORY])

        enforcement.check_enforcement({"subscriber": {"email": "jane@acme.test"}})

        assert "not blocking" in caplog.text
