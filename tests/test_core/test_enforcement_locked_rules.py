"""A scoped caller is stopped by a breached limit the server hands over as a locked rule.

BACK-5684. For a Member or Department Admin the enforcement-rules read moves
every rule whose figure is not theirs to see out of ``rules`` and into a
top-level ``lockedRules`` list (``ScopedEnforcementRules.narrowed``, hypercurrent
release/v2.27.0). A locked rule keeps its name, window, threshold, action,
``shadowMode`` and ``filters``, carries ``breached`` / ``warnBreached`` taken from
the caller's own department verdicts, and has no ``currentValue``. Until this
suite the SDK read only ``rules``, so a breached locked BLOCK rule let the paid
call through.

What is pinned here:

1. A locked rule is acted on only for a call that names no subscriber. Its
   ``breached`` is the API key owner's own standing, and its ``filters`` can be
   broader than the real rule's, because the server drops the SUBSCRIBER rows
   the caller may not see (``RuleVisibilityPolicy.disclosedFilters``). A call
   naming a subscriber, the owner included, is left to the rules and the
   department maps, which carry each in-scope person's own verdict. A blank
   subscriber names nobody; a credential alone names no subscriber.
2. For a call naming no subscriber, a breached locked BLOCK rule refuses the
   call with the rule's name, id and threshold, and ``current_value`` is
   ``None``: the server sends no spend figure and the SDK never invents one.
3. WARN_ONLY, ``shadowMode``, ``warnBreached``-only, ended-window and
   filter-mismatch entries let the call through; malformed entries are ignored.
4. A locked rule's filter rows are judged as the server judges a department
   cap that has already named its caller (``matchesCallScope``): ignoring
   case, with department and subscriber rows and a credential sent only by
   raw value counting as covering.
5. A payload without ``lockedRules``, a bare-list body and a 204 behave as
   before.
6. Locked rules are cleared with the other cached data when the cache owner
   changes, the API key the read authenticates with included, and survive a
   restart in their own snapshot file, leaving the rules snapshot the bare
   list an older SDK reads. A snapshot saved under another key is not loaded.
7. A fetched answer is installed and saved under the key its request was
   sent with, and only while that key is still the process's: an answer read
   with another key is dropped, even when the key has switched back since.
"""
import datetime
import json
import logging
import os
from pathlib import Path

import pytest

from revenium_middleware._core import enforcement
from revenium_middleware._core.exceptions import BudgetExceededError

from .conftest import make_response, stub_get

MIDDLEWARE_LOGGER = "revenium_middleware.extension"

LOCKED_RULE_ID = 4211
LOCKED_RULE_NAME = "Mobile team daily limit"
LOCKED_THRESHOLD = 750.0
MEMBER_EMAIL = "tomas@example.test"


def _iso(moment):
    return moment.strftime("%Y-%m-%dT%H:%M:%S.999999999Z")


def _now():
    return datetime.datetime.now(datetime.timezone.utc)


OPEN_WINDOW_END = _iso(_now() + datetime.timedelta(days=1))
ENDED_WINDOW_END = _iso(_now() - datetime.timedelta(days=1))


def locked_rule(**overrides):
    """A ``LockedEnforcementRule`` as the scoped read serializes it."""
    rule = {
        "ruleId": LOCKED_RULE_ID,
        "teamId": 1052,
        "name": LOCKED_RULE_NAME,
        "metricType": "TOKEN_COUNT",
        "threshold": LOCKED_THRESHOLD,
        "warnThreshold": None,
        "periodType": "DAILY",
        "windowStart": _iso(_now() - datetime.timedelta(hours=1)),
        "windowEnd": OPEN_WINDOW_END,
        "windowKey": "DAILY:2026-11-10",
        "action": "BLOCK",
        "shadowMode": False,
        "groupBy": "SUBSCRIBER",
        "filters": [],
        "departmentId": 20,
        "breached": True,
        "warnBreached": False,
        "compiledAt": _iso(_now()),
    }
    rule.update(overrides)
    return rule


def pooled_rule(**overrides):
    """A rule the scoped caller sees whole, in ``rules``."""
    rule = {
        "ruleId": 9,
        "name": "Company-wide daily limit",
        "threshold": 5000.0,
        "currentValue": 4321.99,
        "action": "BLOCK",
        "breached": False,
        "shadowMode": False,
        "windowEnd": OPEN_WINDOW_END,
    }
    rule.update(overrides)
    return rule


def provider_filter(value):
    return {"dimension": "PROVIDER", "operator": "IS", "value": value, "values": []}


def filter_row(dimension, operator, value):
    return {"dimension": dimension, "operator": operator, "value": value, "values": []}


def blocked(usage_metadata=None, **call):
    try:
        enforcement.check_enforcement(usage_metadata, **call)
    except BudgetExceededError:
        return True
    return False


def locked_rule_warnings(caplog):
    return [r for r in caplog.records
            if r.levelno == logging.WARNING and f"(id {LOCKED_RULE_ID})" in r.getMessage()]


OTHER_EMAIL = "someone.else@example.test"
DEPARTMENT_RULE_ID = 31


def department_rule(**overrides):
    """A department cap the scoped caller sees in ``rules``, decided by the maps."""
    rule = {
        "ruleId": DEPARTMENT_RULE_ID,
        "name": "Engineering monthly limit",
        "threshold": 900.0,
        "currentValue": 950.0,
        "action": "BLOCK",
        "breached": True,
        "shadowMode": False,
        "groupBy": "SUBSCRIBER",
        "departmentId": 20,
        "windowEnd": OPEN_WINDOW_END,
    }
    rule.update(overrides)
    return rule


NAMES_A_SUBSCRIBER = [
    pytest.param({"subscriber": {"email": MEMBER_EMAIL}}, id="the-key-owners-own-address"),
    pytest.param({"subscriber": {"email": OTHER_EMAIL}}, id="another-persons-address"),
    pytest.param({"subscriber_email": OTHER_EMAIL}, id="a-flat-address"),
    pytest.param({"subscriber": {"id": "sub-7"}}, id="a-subscriber-id-only"),
    pytest.param({"subscriber_id": "sub-7"}, id="a-flat-subscriber-id-only"),
    pytest.param({"subscriber": {"id": 7}}, id="an-integer-subscriber-id"),
    pytest.param({"subscriber": {"credential": {"name": "k"}}, "subscriber_email": OTHER_EMAIL},
                 id="a-flat-address-beside-a-nested-credential"),
]

NAMES_NO_SUBSCRIBER = [
    pytest.param(None, id="no-metadata"),
    pytest.param({}, id="empty-metadata"),
    pytest.param({"subscriber": {"email": "   "}}, id="a-blank-address"),
    pytest.param({"subscriber_email": " \t "}, id="a-blank-flat-address"),
    pytest.param({"subscriber": {"id": "  "}}, id="a-blank-id"),
    pytest.param({"subscriber": {"email": "", "id": ""}}, id="an-empty-address-and-id"),
    pytest.param({"subscriber": {"id": 0}}, id="a-zero-id"),
    pytest.param({"subscriber": {"credential": {"name": "prod-key"}}}, id="a-credential-only"),
    pytest.param({"agent": "support-bot"}, id="attribution-without-a-subscriber"),
]


class TestABreachedLockedBlockRuleRefusesACallNamingNoSubscriber:

    def test_it_raises_naming_the_rule_its_id_and_threshold(self, department_cache):
        department_cache([], locked_rules=[locked_rule()])

        with pytest.raises(BudgetExceededError) as raised:
            enforcement.check_enforcement()

        assert raised.value.rule_name == LOCKED_RULE_NAME
        assert raised.value.rule_id == LOCKED_RULE_ID
        assert raised.value.threshold == LOCKED_THRESHOLD

    def test_the_error_carries_no_spend_figure(self, department_cache):
        """The locked rule has no ``currentValue`` by design; the SDK must not
        report the threshold, or any other rule's figure, in its place."""
        department_cache([pooled_rule()], locked_rules=[locked_rule()])

        with pytest.raises(BudgetExceededError) as raised:
            enforcement.check_enforcement()

        assert raised.value.current_value is None

    @pytest.mark.parametrize("usage_metadata", NAMES_NO_SUBSCRIBER)
    def test_a_call_naming_no_subscriber_is_refused(self, department_cache, usage_metadata):
        """``breached`` is the key owner's own standing, and a call naming no
        subscriber is metered as theirs."""
        department_cache([], locked_rules=[locked_rule()])

        assert blocked(usage_metadata)

    @pytest.mark.parametrize("shape", [
        {"groupBy": "DEPARTMENT", "departmentId": None},
        {"groupBy": "SUBSCRIBER", "departmentId": 20},
        {"groupBy": None, "departmentId": 20},
    ])
    def test_a_department_shaped_locked_rule_is_not_skipped_like_one_in_rules(
        self, department_cache, shape
    ):
        """In ``rules`` a department rule's ``breached`` speaks for the whole
        department and is skipped; a locked rule's speaks for the key owner."""
        department_cache([], locked_rules=[locked_rule(**shape)])

        assert blocked()

    def test_an_absent_action_falls_toward_enforcing(self, department_cache):
        rule = locked_rule()
        del rule["action"]
        department_cache([], locked_rules=[rule])

        assert blocked()

    def test_a_rule_the_caller_sees_is_judged_before_the_locked_ones(self, department_cache):
        department_cache([pooled_rule(breached=True)], locked_rules=[locked_rule()])

        with pytest.raises(BudgetExceededError) as raised:
            enforcement.check_enforcement()

        assert raised.value.rule_id == 9
        assert raised.value.current_value == 4321.99


class TestACallNamingASubscriberIsLeftToTheRulesAndDepartmentMaps:
    """A locked rule's verdict is the key owner's, and its disclosed filters
    may cover people the real rule leaves out, so it never judges a call made
    for a named subscriber."""

    @pytest.mark.parametrize("usage_metadata", NAMES_A_SUBSCRIBER)
    def test_a_breached_locked_block_rule_does_not_refuse_it(self, department_cache, usage_metadata):
        department_cache([], locked_rules=[locked_rule()])

        assert not blocked(usage_metadata)

    @pytest.mark.parametrize("email", [MEMBER_EMAIL, OTHER_EMAIL])
    def test_the_department_block_map_still_refuses_that_subscriber(self, department_cache, email):
        department_cache([department_rule()], blocks={email: DEPARTMENT_RULE_ID},
                         balances={email: 975.0}, locked_rules=[locked_rule()])

        with pytest.raises(BudgetExceededError) as raised:
            enforcement.check_enforcement({"subscriber": {"email": email}})

        assert raised.value.rule_id == DEPARTMENT_RULE_ID
        assert raised.value.current_value == 975.0

    def test_the_department_candidates_still_refuse_that_subscriber(self, department_cache):
        department_cache(
            [department_rule()],
            candidates={OTHER_EMAIL: [{"ruleId": DEPARTMENT_RULE_ID, "tier": "BLOCK",
                                       "departmentId": 20, "balance": 960.0}]},
            locked_rules=[locked_rule()],
        )

        with pytest.raises(BudgetExceededError) as raised:
            enforcement.check_enforcement({"subscriber": {"email": OTHER_EMAIL}})

        assert raised.value.rule_id == DEPARTMENT_RULE_ID

    def test_a_subscriber_the_department_map_does_not_block_goes_through(self, department_cache):
        department_cache([department_rule()], blocks={MEMBER_EMAIL: DEPARTMENT_RULE_ID},
                         locked_rules=[locked_rule()])

        assert not blocked({"subscriber": {"email": OTHER_EMAIL}})

    @pytest.mark.parametrize("overrides", [
        pytest.param({"action": "WARN_ONLY"}, id="warn-only"),
        pytest.param({"breached": False, "warnBreached": True}, id="warn-breached-only"),
    ])
    def test_it_hears_no_locked_rule_warning(self, department_cache, caplog, overrides):
        department_cache([], locked_rules=[locked_rule(**overrides)])

        with caplog.at_level(logging.WARNING, logger=MIDDLEWARE_LOGGER):
            enforcement.check_enforcement({"subscriber": {"email": OTHER_EMAIL}})
            enforcement.check_enforcement({"subscriber": {"id": "sub-7"}})

        assert locked_rule_warnings(caplog) == []

    @pytest.mark.parametrize("overrides", [
        pytest.param({"action": "WARN_ONLY"}, id="warn-only"),
        pytest.param({"breached": False, "warnBreached": True}, id="warn-breached-only"),
    ])
    def test_its_call_does_not_use_up_the_warning_a_later_call_naming_nobody_hears(
        self, department_cache, caplog, overrides
    ):
        department_cache([], locked_rules=[locked_rule(**overrides)])

        with caplog.at_level(logging.WARNING, logger=MIDDLEWARE_LOGGER):
            enforcement.check_enforcement({"subscriber": {"email": MEMBER_EMAIL}})
            assert locked_rule_warnings(caplog) == []
            enforcement.check_enforcement()

        assert len(locked_rule_warnings(caplog)) == 1


class TestLockedRulesThatLetTheCallThrough:

    @pytest.mark.parametrize("overrides", [
        pytest.param({"action": "WARN_ONLY"}, id="warn-only"),
        pytest.param({"shadowMode": True}, id="shadow-mode"),
        pytest.param({"breached": False, "warnBreached": True}, id="warn-breached-only"),
        pytest.param({"breached": False, "warnBreached": False}, id="under-the-limit"),
        pytest.param({"windowEnd": ENDED_WINDOW_END}, id="ended-window"),
        pytest.param({"filters": [provider_filter("OPENAI")]}, id="filter-mismatch"),
    ])
    def test_the_call_goes_through(self, department_cache, overrides):
        department_cache([], locked_rules=[locked_rule(**overrides)])

        assert not blocked(None, provider="ANTHROPIC")

    def test_a_matching_filter_still_refuses_the_call(self, department_cache):
        department_cache([], locked_rules=[locked_rule(filters=[provider_filter("OPENAI")])])

        assert blocked(provider="OPENAI")

    def test_a_breached_warn_only_rule_is_logged_once_per_window(self, department_cache, caplog):
        department_cache([], locked_rules=[locked_rule(action="WARN_ONLY")])

        with caplog.at_level(logging.WARNING, logger=MIDDLEWARE_LOGGER):
            for _ in range(3):
                enforcement.check_enforcement()

        assert len(locked_rule_warnings(caplog)) == 1

    def test_a_warn_breached_rule_is_logged_once_per_window_and_never_raises(
        self, department_cache, caplog
    ):
        department_cache([], locked_rules=[locked_rule(breached=False, warnBreached=True)])

        with caplog.at_level(logging.WARNING, logger=MIDDLEWARE_LOGGER):
            for _ in range(3):
                enforcement.check_enforcement()

        lines = locked_rule_warnings(caplog)
        assert len(lines) == 1
        assert "warning threshold" in lines[0].getMessage()

    def test_a_rule_under_both_lines_logs_nothing(self, department_cache, caplog):
        department_cache([], locked_rules=[locked_rule(breached=False, warnBreached=False)])

        with caplog.at_level(logging.WARNING, logger=MIDDLEWARE_LOGGER):
            enforcement.check_enforcement()

        assert locked_rule_warnings(caplog) == []

    def test_a_refresh_does_not_repeat_the_warn_only_line(self, department_cache, fetch_env, caplog):
        """The once-per-window dedupe is pruned to the rules the cache serves,
        and a locked rule is one of them."""
        monkeypatch, _ = fetch_env
        monkeypatch.delenv("REVENIUM_CACHE_DIR", raising=False)
        department_cache([])
        stub_get(monkeypatch, [make_response(200, json_body={
            "rules": [], "lockedRules": [locked_rule(action="WARN_ONLY")]})])

        with caplog.at_level(logging.WARNING, logger=MIDDLEWARE_LOGGER):
            for _ in range(3):
                enforcement._refresh_cache()
                enforcement.check_enforcement()

        assert len(locked_rule_warnings(caplog)) == 1

    @pytest.mark.parametrize("junk", ["locked", None, 7, ["nested"]])
    def test_a_non_dict_entry_is_ignored(self, department_cache, junk):
        department_cache([], locked_rules=[junk])

        assert not blocked()

    def test_a_non_dict_entry_does_not_hide_a_breached_rule_after_it(self, department_cache):
        department_cache([], locked_rules=["locked", None, locked_rule()])

        assert blocked()


class TestLockedRuleFiltersAreJudgedAsTheServerJudgesADepartmentCap:
    """A locked rule's ``breached`` is a department verdict, which the server
    applies to a call through ``EnforcementFilterMatcher.matchesCallScope``."""

    def test_a_row_naming_the_call_in_another_case_covers_it(self, department_cache):
        department_cache([], locked_rules=[locked_rule(filters=[filter_row("AGENT", "IS", "Support-Bot")])])

        assert blocked({"agent": "support-bot"})

    def test_an_is_not_row_naming_the_call_in_another_case_excludes_it(self, department_cache):
        department_cache([], locked_rules=[locked_rule(filters=[filter_row("AGENT", "IS_NOT", "Support-Bot")])])

        assert not blocked({"agent": "support-bot"})

    def test_a_department_row_counts_as_covering(self, department_cache):
        department_cache([], locked_rules=[locked_rule(filters=[filter_row("DEPARTMENT", "IS", "20")])])

        assert blocked()

    def test_a_credential_row_covers_a_call_that_sends_only_the_raw_value(self, department_cache):
        """The server totals the cap under the credential's saved name, which a
        raw value cannot be matched to, so the cap keeps applying."""
        department_cache([], locked_rules=[locked_rule(filters=[filter_row("CREDENTIAL", "IS", "prod-key")])])

        assert blocked({"subscriber": {"credential": {"value": "sk-raw"}}})

    def test_a_credential_row_still_excludes_a_call_naming_another_credential(self, department_cache):
        department_cache([], locked_rules=[locked_rule(filters=[filter_row("CREDENTIAL", "IS", "prod-key")])])

        assert not blocked({"subscriber": {"credential": {"name": "dev-key"}}})

    @pytest.mark.parametrize("row", [
        pytest.param(filter_row("SUBSCRIBER", "IS", MEMBER_EMAIL), id="is-the-owner"),
        pytest.param({"dimension": "SUBSCRIBER", "operator": "IN", "value": None,
                      "values": [MEMBER_EMAIL]}, id="in-a-list-naming-the-owner"),
        pytest.param(filter_row("SUBSCRIBER", "IS", "sub-7"), id="is-an-id"),
    ])
    def test_a_disclosed_subscriber_row_covers_a_call_naming_no_subscriber(self, department_cache, row):
        """The server settled the SUBSCRIBER rows in naming the key owner under
        this rule, which is where ``breached`` came from, and its department
        matcher never judges them against the call. Judging them against a call
        that names nobody would exclude exactly the call the verdict is for."""
        department_cache([], locked_rules=[locked_rule(filters=[row])])

        assert blocked()


class TestLockedRulesComeOffTheWire:

    def test_the_list_is_read_beside_the_rules(self):
        rule = locked_rule()
        fetched = enforcement._fetched_from_payload({"rules": [pooled_rule()], "lockedRules": [rule]})

        assert fetched.locked_rules == [rule]
        assert fetched.rules == [pooled_rule()]

    @pytest.mark.parametrize("value", [None, {"ruleId": 1}, "locked", 3, True])
    def test_a_value_that_is_not_a_list_reads_as_none(self, value):
        assert enforcement._fetched_from_payload({"rules": [], "lockedRules": value}).locked_rules == []

    def test_a_payload_without_the_key_reads_as_none(self):
        assert enforcement._fetched_from_payload({"rules": [pooled_rule()]}).locked_rules == []

    def test_a_bare_list_body_reads_as_none(self):
        assert enforcement._fetched_from_payload([pooled_rule()]).locked_rules == []

    def test_a_204_reads_as_none(self):
        assert enforcement._rules_from_response(make_response(204)).locked_rules == []

    def test_a_check_reads_its_own_copy_beside_the_rules(self, department_cache):
        rule = locked_rule()
        department_cache([pooled_rule()], locked_rules=[rule])

        cached = enforcement._get_rules()
        enforcement._cached_locked_rules.clear()

        assert cached.rules == [pooled_rule()]
        assert cached.locked_rules == [rule]

    def test_a_refresh_installs_them_and_the_next_call_is_refused(self, department_cache, fetch_env):
        monkeypatch, _ = fetch_env
        monkeypatch.delenv("REVENIUM_CACHE_DIR", raising=False)
        department_cache([])
        stub_get(monkeypatch, [make_response(200, json_body={
            "rules": [pooled_rule()], "lockedRules": [locked_rule()]})])

        enforcement._refresh_cache()

        assert blocked()

    def test_a_refresh_without_them_lifts_the_previous_ones(self, department_cache, fetch_env):
        monkeypatch, _ = fetch_env
        monkeypatch.delenv("REVENIUM_CACHE_DIR", raising=False)
        department_cache([], locked_rules=[locked_rule()])
        stub_get(monkeypatch, [make_response(200, json_body={"rules": [pooled_rule()]})])

        enforcement._refresh_cache()

        assert not blocked()


class TestAPayloadWithoutLockedRulesBehavesAsBefore:

    def test_a_breached_rule_still_reports_its_own_value(self, department_cache, fetch_env):
        monkeypatch, _ = fetch_env
        monkeypatch.delenv("REVENIUM_CACHE_DIR", raising=False)
        department_cache([])
        stub_get(monkeypatch, [make_response(200, json_body={"rules": [pooled_rule(breached=True)]})])
        enforcement._refresh_cache()

        with pytest.raises(BudgetExceededError) as raised:
            enforcement.check_enforcement()

        assert raised.value.current_value == 4321.99

    def test_an_unbreached_rule_lets_the_call_through(self, department_cache, fetch_env):
        monkeypatch, _ = fetch_env
        monkeypatch.delenv("REVENIUM_CACHE_DIR", raising=False)
        department_cache([])
        stub_get(monkeypatch, [make_response(200, json_body={"rules": [pooled_rule()]})])
        enforcement._refresh_cache()

        assert not blocked()

    def test_a_204_lets_the_call_through(self, department_cache, fetch_env):
        monkeypatch, _ = fetch_env
        monkeypatch.delenv("REVENIUM_CACHE_DIR", raising=False)
        department_cache([], locked_rules=[locked_rule()])
        stub_get(monkeypatch, [make_response(204)])
        enforcement._refresh_cache()

        assert not blocked()


class TestLockedRulesFollowTheCacheOwner:

    def test_they_are_dropped_when_the_process_switches_team(self, department_cache, monkeypatch):
        monkeypatch.setenv("REVENIUM_TEAM_ID", "team-a")
        department_cache([], locked_rules=[locked_rule()])
        assert blocked()

        monkeypatch.setenv("REVENIUM_TEAM_ID", "team-b")

        assert not blocked()
        assert enforcement._cached_locked_rules == []

    def test_they_are_not_installed_from_a_fetch_for_the_team_switched_away_from(
        self, department_cache, fetch_env
    ):
        monkeypatch, _ = fetch_env
        monkeypatch.delenv("REVENIUM_CACHE_DIR", raising=False)
        department_cache([])
        enforcement.check_enforcement()

        def switch_team_mid_fetch(*args, **kwargs):
            monkeypatch.setenv("REVENIUM_TEAM_ID", "team-2")
            return make_response(200, json_body={"rules": [], "lockedRules": [locked_rule()]})

        monkeypatch.setattr(enforcement.httpx, "get", switch_team_mid_fetch)
        enforcement._refresh_cache()

        assert enforcement._cached_locked_rules == []


KEY_A = "hak_owner_a_0123456789abcdef"
KEY_B = "hak_owner_b_fedcba9876543210"


class TestLockedRulesFollowTheApiKey:
    """``breached`` is the verdict for whoever the key belongs to. The rest of
    the cache carries verdicts on rules and subscribers, which a key-only
    switch keeps (BACK-4250)."""

    def test_they_are_dropped_when_the_process_switches_to_another_key_on_the_same_team(
        self, department_cache, mock_revenium_client
    ):
        mock_revenium_client.api_key = KEY_A
        department_cache([pooled_rule()], blocks={MEMBER_EMAIL: 9}, locked_rules=[locked_rule()])
        assert blocked()

        mock_revenium_client.api_key = KEY_B

        assert not blocked()
        assert enforcement._cached_locked_rules == []

    def test_the_rules_and_department_maps_stay(self, department_cache, mock_revenium_client):
        mock_revenium_client.api_key = KEY_A
        department_cache([pooled_rule(breached=True)], blocks={MEMBER_EMAIL: 9}, locked_rules=[locked_rule()])
        blocked()

        mock_revenium_client.api_key = KEY_B

        with pytest.raises(BudgetExceededError) as raised:
            enforcement.check_enforcement()
        assert raised.value.rule_id == 9
        assert enforcement._cached_org_unit_blocks == {MEMBER_EMAIL: 9}
        assert enforcement._cache_initialized is True

    def test_the_poller_is_woken_to_fetch_the_new_keys_locked_rules(
        self, department_cache, mock_revenium_client
    ):
        mock_revenium_client.api_key = KEY_A
        department_cache([], locked_rules=[locked_rule()])
        blocked()
        enforcement._poll_wakeup.clear()

        mock_revenium_client.api_key = KEY_B
        blocked()

        assert enforcement._poll_wakeup.is_set()

    def test_the_same_key_keeps_the_cache(self, department_cache, mock_revenium_client):
        mock_revenium_client.api_key = KEY_A
        department_cache([], locked_rules=[locked_rule()])
        assert blocked()

        assert blocked()

    def test_the_switch_log_names_neither_the_key_nor_its_fingerprint(
        self, department_cache, mock_revenium_client, caplog
    ):
        mock_revenium_client.api_key = KEY_A
        department_cache([], locked_rules=[locked_rule()])
        blocked()
        mock_revenium_client.api_key = KEY_B

        with caplog.at_level(logging.DEBUG, logger=MIDDLEWARE_LOGGER):
            blocked()

        assert caplog.records
        for key in (KEY_A, KEY_B):
            assert key not in caplog.text
            assert enforcement._credential_fingerprint(key) not in caplog.text

    def test_the_team_lines_name_the_team_and_never_the_key_or_its_fingerprint(
        self, department_cache, fetch_env, mock_revenium_client, caplog
    ):
        monkeypatch, _ = fetch_env
        monkeypatch.delenv("REVENIUM_CACHE_DIR", raising=False)
        mock_revenium_client.api_key = KEY_A
        department_cache([])
        enforcement.check_enforcement()
        stub_get(monkeypatch, [make_response(500)] * enforcement._FETCH_MAX_ATTEMPTS)

        with caplog.at_level(logging.INFO, logger=MIDDLEWARE_LOGGER):
            enforcement._refresh_cache()
            monkeypatch.setenv("REVENIUM_TEAM_ID", "team-2")
            enforcement.check_enforcement()

        team_lines = [r.getMessage() for r in caplog.records
                      if "could not be refreshed" in r.getMessage() or "previous team" in r.getMessage()]
        assert len(team_lines) == 2
        assert "team-1" in team_lines[0] and "team-2" in team_lines[1]
        for line in team_lines:
            assert KEY_A not in line
            assert enforcement._credential_fingerprint(KEY_A) not in line

    def test_they_are_not_installed_from_a_fetch_made_before_the_key_changed(
        self, department_cache, fetch_env, mock_revenium_client
    ):
        monkeypatch, _ = fetch_env
        monkeypatch.delenv("REVENIUM_CACHE_DIR", raising=False)
        mock_revenium_client.api_key = KEY_A
        department_cache([])
        enforcement.check_enforcement()

        def switch_key_mid_fetch(*args, **kwargs):
            mock_revenium_client.api_key = KEY_B
            return make_response(200, json_body={"rules": [], "lockedRules": [locked_rule()]})

        monkeypatch.setattr(enforcement.httpx, "get", switch_key_mid_fetch)
        enforcement._refresh_cache()

        assert enforcement._cached_locked_rules == []
        assert not blocked()

    def test_a_key_switch_lifts_the_backoff_the_previous_key_started(
        self, department_cache, fetch_env, mock_revenium_client
    ):
        monkeypatch, _ = fetch_env
        monkeypatch.delenv("REVENIUM_CACHE_DIR", raising=False)
        mock_revenium_client.api_key = KEY_A
        department_cache([])
        enforcement.check_enforcement()
        stub_get(monkeypatch, [make_response(403)])
        enforcement._refresh_cache()
        assert enforcement._refresh_is_on_cooldown()

        mock_revenium_client.api_key = KEY_B
        enforcement.check_enforcement()

        assert not enforcement._refresh_is_on_cooldown()

    def test_the_fingerprint_is_a_truncated_digest_that_does_not_carry_the_key(self):
        fingerprint = enforcement._credential_fingerprint(KEY_A)

        assert len(fingerprint) == 16
        assert int(fingerprint, 16) >= 0
        assert fingerprint != enforcement._credential_fingerprint(KEY_B)

    def test_no_key_has_no_fingerprint(self):
        assert enforcement._credential_fingerprint("") == ""
        assert enforcement._credential_fingerprint(None) == ""


@pytest.fixture()
def snapshot_env(department_snapshot_dir, fetch_env):
    """A snapshot directory, a fetchable team and a breaker that never polls."""
    monkeypatch, _ = fetch_env
    monkeypatch.setenv("REVENIUM_TEAM_ID", "team-snapshot")
    monkeypatch.setenv("REVENIUM_CIRCUIT_BREAKER_ENABLED", "true")
    monkeypatch.delenv("REVENIUM_BYPASS", raising=False)
    monkeypatch.delenv("REVENIUM_CB_FAIL_MODE", raising=False)
    monkeypatch.setattr(enforcement, "_ensure_poller_running", lambda: None)
    return monkeypatch, department_snapshot_dir


def simulate_restart(monkeypatch):
    monkeypatch.setattr(enforcement, "_disk_load_attempted", False)
    monkeypatch.setattr(enforcement, "_cached_rules", [])
    monkeypatch.setattr(enforcement, "_cached_locked_rules", [])
    monkeypatch.setattr(enforcement, "_cache_timestamp", 0.0)
    monkeypatch.setattr(enforcement, "_cache_initialized", False)
    monkeypatch.setattr(enforcement, "_cache_owner", None)


class TestLockedRulesSurviveARestart:

    def test_a_breached_locked_rule_still_refuses_the_call_after_a_restart_in_fail_closed_mode(
        self, snapshot_env
    ):
        monkeypatch, _ = snapshot_env
        stub_get(monkeypatch, [make_response(200, json_body={
            "rules": [pooled_rule()], "lockedRules": [locked_rule()]})])
        enforcement._refresh_cache()
        simulate_restart(monkeypatch)
        monkeypatch.setenv("REVENIUM_CB_FAIL_MODE", "closed")

        with pytest.raises(BudgetExceededError) as raised:
            enforcement.check_enforcement()

        assert raised.value.rule_id == LOCKED_RULE_ID
        assert raised.value.current_value is None

    def test_the_rules_snapshot_stays_the_bare_list_an_older_sdk_reads(self, snapshot_env):
        monkeypatch, _ = snapshot_env
        rules, locked = [pooled_rule()], [locked_rule()]
        stub_get(monkeypatch, [make_response(200, json_body={"rules": rules, "lockedRules": locked})])

        enforcement._refresh_cache()

        assert json.loads(Path(enforcement._cache_file_path()).read_text(encoding="utf-8")) == rules
        envelope = json.loads(Path(enforcement._locked_rules_cache_file_path()).read_text(encoding="utf-8"))
        assert envelope["lockedRules"] == locked
        assert envelope["rules_fingerprint"] == enforcement._rules_fingerprint(rules)

    @pytest.mark.skipif(os.name == "nt", reason="POSIX file modes")
    def test_the_locked_rules_snapshot_is_owner_only(self, snapshot_env):
        enforcement._persist_cache_to_disk([], locked_rules=[locked_rule()])

        assert os.stat(enforcement._locked_rules_cache_file_path()).st_mode & 0o777 == 0o600

    def test_a_torn_snapshot_pair_drops_the_locked_rules(self, snapshot_env):
        enforcement._persist_cache_to_disk([pooled_rule()], locked_rules=[locked_rule()])
        Path(enforcement._cache_file_path()).write_text(json.dumps([pooled_rule(ruleId=10)]), encoding="utf-8")

        enforcement._load_cache_from_disk()

        assert enforcement._cached_rules == [pooled_rule(ruleId=10)]
        assert enforcement._cached_locked_rules == []

    @pytest.mark.parametrize("content", ["not json", "[]", '{"rules_fingerprint": "x"}'])
    def test_an_unreadable_locked_rules_snapshot_still_loads_the_rules(self, snapshot_env, content):
        enforcement._persist_cache_to_disk([pooled_rule()], locked_rules=[locked_rule()])
        Path(enforcement._locked_rules_cache_file_path()).write_text(content, encoding="utf-8")

        enforcement._load_cache_from_disk()

        assert enforcement._cached_rules == [pooled_rule()]
        assert enforcement._cached_locked_rules == []
        assert enforcement._cache_initialized is True

    def test_a_snapshot_written_before_locked_rules_existed_loads_none(self, snapshot_env):
        enforcement._persist_cache_to_disk([pooled_rule()])
        os.unlink(enforcement._locked_rules_cache_file_path())

        enforcement._load_cache_from_disk()

        assert enforcement._cached_rules == [pooled_rule()]
        assert enforcement._cached_locked_rules == []


def drop_from_envelope(path, key):
    envelope = json.loads(Path(path).read_text(encoding="utf-8"))
    del envelope[key]
    Path(path).write_text(json.dumps(envelope), encoding="utf-8")


class TestTheLockedRulesSnapshotIsBoundToTheApiKey:

    def test_a_restart_under_another_key_loads_the_rules_and_maps_but_not_the_locked_rules(
        self, snapshot_env, mock_revenium_client
    ):
        mock_revenium_client.api_key = KEY_A
        enforcement._persist_cache_to_disk([pooled_rule()], {MEMBER_EMAIL: 9}, locked_rules=[locked_rule()])
        mock_revenium_client.api_key = KEY_B

        enforcement._load_cache_from_disk()

        assert enforcement._cached_locked_rules == []
        assert enforcement._cached_rules == [pooled_rule()]
        assert enforcement._cached_org_unit_blocks == {MEMBER_EMAIL: 9}
        assert enforcement._cache_initialized is True

    def test_a_restart_under_another_key_does_not_block_on_the_previous_keys_verdict(
        self, snapshot_env, mock_revenium_client
    ):
        monkeypatch, _ = snapshot_env
        mock_revenium_client.api_key = KEY_A
        stub_get(monkeypatch, [make_response(200, json_body={
            "rules": [pooled_rule()], "lockedRules": [locked_rule()]})])
        enforcement._refresh_cache()
        simulate_restart(monkeypatch)
        mock_revenium_client.api_key = KEY_B
        monkeypatch.setattr(enforcement, "_refresh_cache", lambda: None)

        assert not blocked()

    def test_a_restart_under_the_same_key_loads_everything(self, snapshot_env, mock_revenium_client):
        mock_revenium_client.api_key = KEY_A
        enforcement._persist_cache_to_disk([pooled_rule()], {MEMBER_EMAIL: 9}, locked_rules=[locked_rule()])

        enforcement._load_cache_from_disk()

        assert enforcement._cached_rules == [pooled_rule()]
        assert enforcement._cached_org_unit_blocks == {MEMBER_EMAIL: 9}
        assert len(enforcement._cached_locked_rules) == 1

    def test_locked_rules_saved_without_a_key_are_not_loaded(self, snapshot_env, mock_revenium_client):
        mock_revenium_client.api_key = KEY_A
        enforcement._persist_cache_to_disk([pooled_rule()], locked_rules=[locked_rule()])
        drop_from_envelope(enforcement._locked_rules_cache_file_path(), "credential")

        enforcement._load_cache_from_disk()

        assert enforcement._cached_rules == [pooled_rule()]
        assert enforcement._cached_locked_rules == []

    def test_the_snapshot_files_keep_their_names_across_keys(self, snapshot_env, mock_revenium_client):
        """An older SDK reads the rules and map files by name; the key is bound
        inside the locked-rules envelope instead."""
        mock_revenium_client.api_key = KEY_A
        paths_under_a = (enforcement._cache_file_path(), enforcement._org_unit_blocks_cache_file_path(),
                         enforcement._locked_rules_cache_file_path())
        mock_revenium_client.api_key = KEY_B

        assert (enforcement._cache_file_path(), enforcement._org_unit_blocks_cache_file_path(),
                enforcement._locked_rules_cache_file_path()) == paths_under_a

    def test_no_snapshot_file_carries_the_key(self, snapshot_env, mock_revenium_client):
        _, directory = snapshot_env
        mock_revenium_client.api_key = KEY_A
        enforcement._persist_cache_to_disk([pooled_rule()], {MEMBER_EMAIL: 9}, locked_rules=[locked_rule()])

        for path in Path(directory).iterdir():
            assert KEY_A not in path.name
            assert KEY_A not in path.read_text(encoding="utf-8")


def locked_rules_envelope():
    return json.loads(Path(enforcement._locked_rules_cache_file_path()).read_text(encoding="utf-8"))


def answer_with(locked, rules=None):
    return make_response(200, json_body={"rules": rules or [], "lockedRules": locked})


class TestAnAnswerIsInstalledUnderTheKeyItWasReadWith:
    """A locked rule's ``breached`` is the standing of whoever owns the key the
    read was sent with, so the answer is installed and saved under that key's
    owner, taken from the same read of the key the request header was built
    from, and never under one sampled before or after the request."""

    def test_a_normal_read_installs_and_saves_under_the_key_it_was_sent_with(
        self, snapshot_env, mock_revenium_client
    ):
        monkeypatch, _ = snapshot_env
        mock_revenium_client.api_key = KEY_A
        locked = [locked_rule()]
        read = stub_get(monkeypatch, [answer_with(locked, [pooled_rule()])])

        enforcement._refresh_cache()

        assert read.requests[0][1]["headers"]["x-api-key"] == KEY_A
        assert enforcement._cache_owner.credential == enforcement._credential_fingerprint(KEY_A)
        assert enforcement._cached_locked_rules == locked
        assert locked_rules_envelope()["credential"] == enforcement._credential_fingerprint(KEY_A)
        assert locked_rules_envelope()["lockedRules"] == locked
        assert blocked()

    def test_the_snapshot_keeps_the_reading_keys_fingerprint_when_the_key_moves_on_before_it_is_written(
        self, snapshot_env, mock_revenium_client
    ):
        monkeypatch, _ = snapshot_env
        mock_revenium_client.api_key = KEY_A
        stub_get(monkeypatch, [answer_with([locked_rule()])])
        persist = enforcement._persist_cache_to_disk

        def persist_after_the_key_moved_on(*args, **kwargs):
            mock_revenium_client.api_key = KEY_B
            persist(*args, **kwargs)

        monkeypatch.setattr(enforcement, "_persist_cache_to_disk", persist_after_the_key_moved_on)

        enforcement._refresh_cache()

        assert locked_rules_envelope()["credential"] == enforcement._credential_fingerprint(KEY_A)

    def test_a_read_sent_with_one_key_is_not_installed_once_the_key_has_changed(
        self, snapshot_env, mock_revenium_client
    ):
        monkeypatch, _ = snapshot_env
        mock_revenium_client.api_key = KEY_A
        sent_with = []

        def answer_after_the_key_changed_to_b(url, **kwargs):
            sent_with.append(kwargs["headers"]["x-api-key"])
            mock_revenium_client.api_key = KEY_B
            return answer_with([locked_rule()])

        monkeypatch.setattr(enforcement.httpx, "get", answer_after_the_key_changed_to_b)

        enforcement._refresh_cache()

        assert sent_with == [KEY_A]
        assert enforcement._cached_locked_rules == []
        assert not os.path.exists(enforcement._locked_rules_cache_file_path())
        assert not blocked()

    def test_a_read_sent_with_another_key_is_neither_installed_nor_saved_after_the_key_returns(
        self, snapshot_env, mock_revenium_client
    ):
        monkeypatch, _ = snapshot_env
        mock_revenium_client.api_key = KEY_A
        stub_get(monkeypatch, [answer_with([], [pooled_rule()])])
        enforcement._refresh_cache()
        saved_under_a = locked_rules_envelope()
        sent_with = read_with_b_while_every_other_owner_read_sees_a(monkeypatch, mock_revenium_client)

        enforcement._refresh_cache()

        assert sent_with == [KEY_B]
        assert enforcement._cached_locked_rules == []
        assert enforcement._cached_rules == [pooled_rule()]
        assert locked_rules_envelope() == saved_under_a
        assert not blocked()

    def test_a_read_sent_with_another_key_is_not_adopted_by_a_cache_nobody_owns_yet(
        self, snapshot_env, mock_revenium_client
    ):
        monkeypatch, _ = snapshot_env
        mock_revenium_client.api_key = KEY_A
        sent_with = read_with_b_while_every_other_owner_read_sees_a(monkeypatch, mock_revenium_client)

        enforcement._refresh_cache()

        assert sent_with == [KEY_B]
        assert enforcement._cache_owner is None
        assert enforcement._cached_locked_rules == []
        assert not os.path.exists(enforcement._locked_rules_cache_file_path())
        assert not blocked()


def read_with_b_while_every_other_owner_read_sees_a(monkeypatch, client):
    """Make the next read go A, B, A: the key turns to B as the request is
    built and back to A before the answer, which is B's, comes back.

    Returns the keys the requests were sent with."""
    resolve_target = enforcement._fetch_target
    sent_with = []

    def build_the_request_under_b():
        client.api_key = KEY_B
        return resolve_target()

    def answer_after_the_key_returned_to_a(url, **kwargs):
        sent_with.append(kwargs["headers"]["x-api-key"])
        client.api_key = KEY_A
        return answer_with([locked_rule()], [pooled_rule(ruleId=10)])

    monkeypatch.setattr(enforcement, "_fetch_target", build_the_request_under_b)
    monkeypatch.setattr(enforcement.httpx, "get", answer_after_the_key_returned_to_a)
    return sent_with
