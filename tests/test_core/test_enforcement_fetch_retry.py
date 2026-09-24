"""The enforcement rule fetch must ride out 429/5xx spikes with backoff.

A bare GET meant any 429 failed the refresh outright, leaving spend controls
on stale limits until the next poll interval. The fetch now retries
retryable failures (408/429/5xx/connection errors) with exponential backoff
capped at 8s, honors Retry-After (delta-seconds and HTTP-date) up to its own
separate bound, and on exhaustion fails open -- an enforcement-refresh outage
must never become a customer traffic outage. The Retry-After bound itself is
covered in ``test_enforcement_retry_after_bound.py``.

A successful fetch returns the whole payload: the rules plus the server's
top-level ``orgUnitBudgetBlocks`` map (normalized subscriber email -> blocking
rule id) that decides department budgets, and the ``orgUnitBudgetBlockBalances``
map of what each of those people was judged against. Bodies that predate either
map -- a bare list, a dict without the key, an HTTP 204 -- yield empty maps,
which block nobody and report the rule's own value.
"""
import datetime
from email.utils import format_datetime
from types import SimpleNamespace

import httpx
import pytest

from revenium_middleware._core import enforcement

from .conftest import make_response, stub_get


class TestRetryOnTransientFailures:
    def test_429_spike_is_ridden_out(self, fetch_env):
        monkeypatch, sleeps = fetch_env
        stub = stub_get(monkeypatch, [
            make_response(429),
            make_response(429),
            make_response(200, json_body={"rules": [{"id": "r1"}]}),
        ])

        fetched = enforcement._fetch_rules()

        assert fetched.rules == [{"id": "r1"}]
        assert stub.calls == 3
        assert len(sleeps) == 2

    @pytest.mark.parametrize("status", [408, 500, 502, 503, 504])
    def test_retryable_statuses(self, fetch_env, status):
        monkeypatch, sleeps = fetch_env
        stub = stub_get(monkeypatch, [make_response(status), make_response(200)])

        fetched = enforcement._fetch_rules()

        assert fetched.rules == []
        assert stub.calls == 2

    def test_connection_errors_are_retried(self, fetch_env):
        monkeypatch, sleeps = fetch_env
        stub = stub_get(monkeypatch, [
            httpx.ConnectError("refused"),
            httpx.ReadTimeout("slow"),
            make_response(200, json_body={"rules": []}),
        ])

        fetched = enforcement._fetch_rules()

        assert fetched.rules == []
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

    def test_retry_after_beyond_the_bound_gives_up_instead_of_truncating(self, fetch_env):
        """This assertion used to read ``sleeps == [8.0]``.

        That was the FRONT-1682 defect written down as an expectation: the
        server's 120 s was truncated to the 8 s backoff cap and retried there.
        The bound is now a give-up threshold -- no wait, no retry, cached
        rules left in force. The full behaviour lives in
        ``test_enforcement_retry_after_bound.py``.
        """
        monkeypatch, sleeps = fetch_env
        stub = stub_get(monkeypatch, [
            make_response(429, headers={"Retry-After": "120"}),
            make_response(200),
        ])

        assert enforcement._fetch_rules() is None
        assert sleeps == []
        assert stub.calls == 1

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

        fetched = enforcement._fetch_rules()

        assert fetched is None  # fail open: caller keeps previous cache
        assert stub.calls == 1
        assert sleeps == []

    def test_exhaustion_fails_open_without_raising(self, fetch_env):
        monkeypatch, sleeps = fetch_env
        stub = stub_get(monkeypatch, [make_response(503)] * 20)

        fetched = enforcement._fetch_rules()

        assert fetched is None
        assert stub.calls == enforcement._FETCH_MAX_ATTEMPTS  # rode the spike to exhaustion

    def test_shutdown_during_backoff_stops_retrying(self, fetch_env):
        """A stop event set mid-backoff must abort the remaining attempts."""
        monkeypatch, sleeps = fetch_env
        stub = stub_get(monkeypatch, [make_response(503)] * 20)
        # _sleep returns True when shutdown was signalled during the wait.
        monkeypatch.setattr(enforcement, "_sleep",
                            lambda seconds: sleeps.append(seconds) or True)

        fetched = enforcement._fetch_rules()

        assert fetched is None  # fail open: caller keeps previous cache
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


class TestOrgUnitBudgetBlocks:
    """The department-budget map must survive the fetch and reach the cache."""

    def test_map_is_returned_alongside_the_rules(self, fetch_env):
        monkeypatch, _ = fetch_env
        stub_get(monkeypatch, [make_response(200, json_body={
            "rules": [{"ruleId": 7777, "name": "Engineering budget"}],
            "compiledAt": "2026-08-24T00:00:00Z",
            "orgUnitBudgetBlocks": {"dept-user@example.test": 7777},
        })])

        fetched = enforcement._fetch_rules()

        assert fetched.rules == [{"ruleId": 7777, "name": "Engineering budget"}]
        assert fetched.org_unit_blocks == {"dept-user@example.test": 7777}

    def test_legacy_bare_list_body_still_works(self, fetch_env):
        """A body predating the wrapper object yields rules and an empty map."""
        monkeypatch, _ = fetch_env
        stub_get(monkeypatch, [make_response(200, json_body=[{"ruleId": 7777}])])

        fetched = enforcement._fetch_rules()

        assert fetched.rules == [{"ruleId": 7777}]
        assert fetched.org_unit_blocks == {}

    def test_dict_body_without_the_key_yields_an_empty_map(self, fetch_env):
        monkeypatch, _ = fetch_env
        stub_get(monkeypatch, [make_response(200, json_body={"rules": [], "compiledAt": "x"})])

        assert enforcement._fetch_rules().org_unit_blocks == {}

    @pytest.mark.parametrize("blocks", [None, [], "nope", 7])
    def test_malformed_map_is_ignored(self, fetch_env, blocks):
        """Garbage in the map field must not become a blocking verdict."""
        monkeypatch, _ = fetch_env
        stub_get(monkeypatch, [make_response(200, json_body={
            "rules": [], "orgUnitBudgetBlocks": blocks,
        })])

        assert enforcement._fetch_rules().org_unit_blocks == {}

    def test_204_yields_empty_rules_and_empty_maps(self, fetch_env):
        monkeypatch, _ = fetch_env
        response = SimpleNamespace(status_code=204,
                                   headers={},
                                   raise_for_status=lambda: None,
                                   json=lambda: None)
        stub_get(monkeypatch, [response])

        # Rules, the department block map, the per-person balance map and the
        # warn-tier map.
        assert enforcement._fetch_rules() == ([], {}, {}, {})

    def test_refresh_caches_the_map_with_the_rules(self, fetch_env):
        monkeypatch, _ = fetch_env
        monkeypatch.delenv("REVENIUM_CACHE_DIR", raising=False)
        monkeypatch.setattr(enforcement, "_cached_rules", [])
        monkeypatch.setattr(enforcement, "_cached_org_unit_blocks", {})
        monkeypatch.setattr(enforcement, "_cache_timestamp", 0.0)
        monkeypatch.setattr(enforcement, "_cache_initialized", False)
        stub_get(monkeypatch, [make_response(200, json_body={
            "rules": [{"ruleId": 7777}],
            "orgUnitBudgetBlocks": {"dept-user@example.test": 7777},
        })])

        enforcement._refresh_cache()

        assert enforcement._cached_rules == [{"ruleId": 7777}]
        assert enforcement._cached_org_unit_blocks == {"dept-user@example.test": 7777}
        assert enforcement._cache_initialized is True

    def test_fetch_outage_preserves_the_previous_map(self, fetch_env):
        monkeypatch, _ = fetch_env
        stub_get(monkeypatch, [make_response(503)] * 20)
        monkeypatch.setattr(enforcement, "_cached_org_unit_blocks", {"keep@me.test": 1})

        enforcement._refresh_cache()

        assert enforcement._cached_org_unit_blocks == {"keep@me.test": 1}


class TestRuleIdFilter:
    """BACK-3359: the per-rule read is opt-in, and the default read is untouched.

    The server computes the three department-budget maps team-wide and hangs
    them off the whole-team read, so a refresh that narrowed to one rule would
    stop receiving them and department budgets would quietly stop blocking
    anybody (BACK-3066). The filter therefore exists only for a caller who
    asks for it by name.
    """

    def test_the_default_fetch_sends_no_rule_filter(self, fetch_env):
        monkeypatch, _ = fetch_env
        stub = stub_get(monkeypatch, [make_response(200)])

        enforcement._fetch_rules()

        assert stub.last_params is None
        assert stub.last_url.endswith("/v2/api/ai/enforcement-rules/team-1")

    def test_the_polling_refresh_never_narrows(self, fetch_env):
        """The cache the pre-call path reads must stay the team-wide payload."""
        monkeypatch, _ = fetch_env
        stub = stub_get(monkeypatch, [make_response(200, json_body={
            "rules": [{"ruleId": 1}],
            "orgUnitBudgetBlocks": {"dept-user@example.test": 1},
        })])

        enforcement._refresh_cache()

        assert stub.last_params is None
        assert enforcement._cached_org_unit_blocks == {"dept-user@example.test": 1}

    def test_a_rule_id_is_sent_as_the_ruleId_query_parameter(self, fetch_env):
        monkeypatch, _ = fetch_env
        stub = stub_get(monkeypatch, [make_response(200)])

        enforcement._fetch_rules("mN3xpQz")

        assert stub.last_params == {"ruleId": "mN3xpQz"}

    def test_fetch_enforcement_rule_returns_the_one_rule(self, fetch_env):
        monkeypatch, _ = fetch_env
        stub = stub_get(monkeypatch, [make_response(200, json_body={
            "rules": [{"ruleId": "mN3xpQz", "name": "Monthly cap"}],
        })])

        rule = enforcement.fetch_enforcement_rule("mN3xpQz")

        assert rule == {"ruleId": "mN3xpQz", "name": "Monthly cap"}
        assert stub.last_params == {"ruleId": "mN3xpQz"}

    def test_a_rule_the_team_does_not_have_is_none_not_an_error(self, fetch_env):
        """A disabled rule is never compiled, so an empty list is the ordinary answer."""
        monkeypatch, _ = fetch_env
        stub_get(monkeypatch, [make_response(200, json_body={"rules": []})])

        assert enforcement.fetch_enforcement_rule("mN3xpQz") is None

    def test_an_unreachable_server_is_none_not_an_error(self, fetch_env):
        monkeypatch, _ = fetch_env
        stub_get(monkeypatch, [make_response(503)] * 10)

        assert enforcement.fetch_enforcement_rule("mN3xpQz") is None

    def test_an_empty_rule_id_is_refused_rather_than_read_as_the_whole_team(self, fetch_env):
        monkeypatch, _ = fetch_env
        stub = stub_get(monkeypatch, [make_response(200)])

        with pytest.raises(ValueError):
            enforcement.fetch_enforcement_rule("")

        assert stub.calls == 0


class TestRuleRoster:
    """BACK-3360: who one rule covers, read on demand and never cached.

    The roster answers a person's debugging question, not the pre-call path's.
    Nothing in ``check_enforcement`` reads it, so caching it would add a second
    lifetime beside ``_cached_rules`` that could outlive the rule it describes.
    """

    def test_the_roster_is_a_sub_resource_of_the_team_read(self, fetch_env):
        monkeypatch, _ = fetch_env
        stub = stub_get(monkeypatch, [make_response(200, json_body={"rows": []})])

        enforcement.fetch_enforcement_rule_roster("mN3xpQz")

        assert stub.last_url.endswith("/v2/api/ai/enforcement-rules/team-1/roster")
        assert stub.last_params["ruleId"] == "mN3xpQz"

    def test_the_server_pages_bands_and_searches_not_the_sdk(self, fetch_env):
        monkeypatch, _ = fetch_env
        stub = stub_get(monkeypatch, [make_response(200, json_body={"rows": []})])

        enforcement.fetch_enforcement_rule_roster(
            "mN3xpQz", page=2, size=50, search="jane", band="BLOCKED")

        assert stub.last_params == {
            "ruleId": "mN3xpQz", "page": 2, "size": 50,
            "search": "jane", "band": "BLOCKED",
        }

    def test_selectors_left_out_are_not_sent(self, fetch_env):
        monkeypatch, _ = fetch_env
        stub = stub_get(monkeypatch, [make_response(200, json_body={"rows": []})])

        enforcement.fetch_enforcement_rule_roster("mN3xpQz")

        assert stub.last_params == {"ruleId": "mN3xpQz", "page": 0, "size": 25}

    def test_the_rows_and_band_counts_are_returned_as_the_server_computed_them(self, fetch_env):
        monkeypatch, _ = fetch_env
        payload = {
            "ruleId": "mN3xpQz",
            "dimension": "SUBSCRIBER",
            "threshold": 100.0,
            "total": 2,
            "blockedCount": 1,
            "warnedCount": 1,
            "underCount": 0,
            "rows": [
                {"key": "jane@acme.test", "label": "Jane Smith", "spend": 105.5,
                 "limit": 100.0, "band": "BLOCKED"},
                {"key": "bob@acme.test", "label": "Bob", "spend": 85.0,
                 "limit": 100.0, "band": "WARNED"},
            ],
        }
        stub_get(monkeypatch, [make_response(200, json_body=payload)])

        assert enforcement.fetch_enforcement_rule_roster("mN3xpQz") == payload

    def test_no_reading_yet_is_none(self, fetch_env):
        """204 here means the rule is not in the current compiled snapshot."""
        monkeypatch, _ = fetch_env
        stub_get(monkeypatch, [make_response(204)])

        assert enforcement.fetch_enforcement_rule_roster("mN3xpQz") is None

    def test_a_rule_with_no_roster_is_none(self, fetch_env):
        monkeypatch, _ = fetch_env
        stub_get(monkeypatch, [make_response(404)])

        assert enforcement.fetch_enforcement_rule_roster("mN3xpQz") is None

    def test_an_unexpected_body_shape_is_none(self, fetch_env):
        monkeypatch, _ = fetch_env
        stub_get(monkeypatch, [make_response(200, json_body=["not", "an", "object"])])

        assert enforcement.fetch_enforcement_rule_roster("mN3xpQz") is None

    def test_every_read_asks_the_server_again(self, fetch_env):
        """Not cached: a stale roster must not outlive the rule it describes."""
        monkeypatch, _ = fetch_env
        stub = stub_get(monkeypatch, [make_response(200, json_body={"rows": []})])

        enforcement.fetch_enforcement_rule_roster("mN3xpQz")
        enforcement.fetch_enforcement_rule_roster("mN3xpQz")

        assert stub.calls == 2

    def test_the_roster_never_touches_the_enforcement_cache(self, fetch_env):
        monkeypatch, _ = fetch_env
        monkeypatch.setattr(enforcement, "_cached_rules", [{"ruleId": 1}])
        monkeypatch.setattr(enforcement, "_cached_org_unit_blocks", {"keep@me.test": 1})
        stub_get(monkeypatch, [make_response(200, json_body={"rows": [{"key": "a"}]})])

        enforcement.fetch_enforcement_rule_roster("mN3xpQz")

        assert enforcement._cached_rules == [{"ruleId": 1}]
        assert enforcement._cached_org_unit_blocks == {"keep@me.test": 1}

    def test_an_empty_rule_id_is_refused(self, fetch_env):
        monkeypatch, _ = fetch_env
        stub = stub_get(monkeypatch, [make_response(200)])

        with pytest.raises(ValueError):
            enforcement.fetch_enforcement_rule_roster("")

        assert stub.calls == 0

    def test_an_unconfigured_sdk_reads_nothing(self, fetch_env):
        monkeypatch, _ = fetch_env
        monkeypatch.delenv("REVENIUM_TEAM_ID", raising=False)
        stub = stub_get(monkeypatch, [make_response(200)])

        assert enforcement.fetch_enforcement_rule_roster("mN3xpQz") is None
        assert stub.calls == 0
