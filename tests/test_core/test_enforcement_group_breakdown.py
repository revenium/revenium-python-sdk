"""Subscriber-grouped rules must block on the caller's own balance.

``CompiledEnforcementRule`` carries ``groupBreakdown``, a per-group balance
list populated on API reads for subscriber-grouped rules. Before it was read,
one aggregate rule-level flag decided for every caller, so a single breaching
subscriber blocked the whole team. The evaluation now defers to the caller's
matching ``EnforcementGroupEntry`` (subscriber id first, then email) and falls
open when no group key resolves or no entry matches. Pooled rules -- null,
absent, or empty ``groupBreakdown`` -- keep the rule-level behaviour
unchanged, including across the disk-snapshot round trip.

Department (org-unit) budgets are a separate path. The server pre-computes
org-unit membership and publishes a flat ``departmentBudgetBlocks`` map of
subscriber email -> blocking rule id next to the rules; that map is the
entire verdict. The SDK never resolves the caller's org unit, so
``groupBy=DEPARTMENT`` rules are skipped by the per-rule loop.
"""
import json
import logging
import os
import time

import pytest

from revenium_middleware._core import enforcement
from revenium_middleware._core.decorators import revenium_metadata
from revenium_middleware._core.exceptions import BudgetExceededError
from revenium_middleware._core.subscriber import extract_subscriber_from_metadata

RULE_NAME = "Monthly subscriber cap"
POOLED_NAME = "Team monthly cap"
DEPARTMENT_NAME = "Engineering monthly budget"
ORG_UNIT_RULE_ID = 7777
DEPT_EMAIL = "dept-user@example.test"
OTHER_EMAIL = "other-user@example.test"
# ``windowEnd`` as the server writes it: the last nanosecond of the window, UTC.
ENDED_WINDOW = "2000-01-31T23:59:59.999999999Z"
OPEN_WINDOW = "2999-01-31T23:59:59.999999999Z"


def group_entry(group_value, current_value, breached, usage_percent=None):
    """An ``EnforcementGroupEntry`` as the API read returns it."""
    return {
        "groupValue": group_value,
        "displayName": group_value,
        "currentValue": current_value,
        "usagePercent": usage_percent,
        "breached": breached,
    }


def grouped_rule(entries, breached=False, **overrides):
    """A ``groupBy=SUBSCRIBER`` rule carrying a ``groupBreakdown``."""
    rule = {
        "ruleId": 4242,
        "name": RULE_NAME,
        "metricType": "TOTAL_COST",
        "threshold": 100.0,
        "currentValue": 250.0,
        "periodType": "MONTHLY",
        "groupBy": "SUBSCRIBER",
        "breached": breached,
        "shadowMode": False,
        "groupBreakdown": entries,
    }
    rule.update(overrides)
    return rule


def org_unit_rule(**overrides):
    """A ``groupBy=DEPARTMENT`` rule, as the department map references it."""
    rule = {
        "ruleId": ORG_UNIT_RULE_ID,
        "name": DEPARTMENT_NAME,
        "metricType": "TOTAL_COST",
        "threshold": 1000.0,
        "currentValue": 1450.0,
        "periodType": "MONTHLY",
        "groupBy": "DEPARTMENT",
        "action": "BLOCK",
        "breached": True,
        "shadowMode": False,
    }
    rule.update(overrides)
    return rule


def pooled_rule(**overrides):
    """A pooled rule: no ``groupBy``, no ``groupBreakdown`` key at all."""
    rule = {
        "ruleId": 99,
        "name": POOLED_NAME,
        "metricType": "TOTAL_COST",
        "threshold": 500.0,
        "currentValue": 640.0,
        "periodType": "MONTHLY",
        "groupBy": None,
        "breached": True,
        "shadowMode": False,
    }
    rule.update(overrides)
    return rule


def subscriber_metadata(subscriber_id=None, email=None):
    subscriber = {}
    if subscriber_id:
        subscriber["id"] = subscriber_id
    if email:
        subscriber["email"] = email
    return {"subscriber": subscriber}


def two_group_rule(breached=False):
    """sub-a is over its own cap, sub-b is nowhere near it."""
    return grouped_rule(
        [group_entry("sub-a", 140.0, True, 1.4), group_entry("sub-b", 12.5, False, 0.125)],
        breached=breached,
    )


@pytest.fixture()
def load_rules(monkeypatch):
    """Circuit breaker on, cache pre-seeded, poller and disk snapshot inert.

    Returns a callable that installs the rule list as a fresh, initialized
    cache so ``check_enforcement`` evaluates it without any network or thread.
    """
    monkeypatch.setenv("REVENIUM_CIRCUIT_BREAKER_ENABLED", "true")
    monkeypatch.delenv("REVENIUM_BYPASS", raising=False)
    monkeypatch.delenv("REVENIUM_CB_FAIL_MODE", raising=False)
    monkeypatch.setattr(enforcement, "_load_cache_from_disk", lambda: None)
    monkeypatch.setattr(enforcement, "_ensure_poller_running", lambda: None)

    def load(rules, org_unit_blocks=None):
        monkeypatch.setattr(enforcement, "_cached_rules", list(rules))
        monkeypatch.setattr(enforcement, "_cached_org_unit_blocks", org_unit_blocks or {})
        monkeypatch.setattr(enforcement, "_cache_timestamp", time.monotonic())
        monkeypatch.setattr(enforcement, "_cache_initialized", True)

    return load


@pytest.fixture()
def cold_cache(monkeypatch, tmp_path):
    """A snapshot directory plus a cache that has never been loaded."""
    monkeypatch.setenv("REVENIUM_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("REVENIUM_TEAM_ID", "team-snapshot")
    monkeypatch.setattr(enforcement, "_disk_load_attempted", False)
    monkeypatch.setattr(enforcement, "_cached_rules", [])
    monkeypatch.setattr(enforcement, "_cached_org_unit_blocks", {})
    monkeypatch.setattr(enforcement, "_cache_timestamp", 0.0)
    monkeypatch.setattr(enforcement, "_cache_initialized", False)
    return monkeypatch


class TestPerGroupEvaluation:
    def test_breaching_subscriber_blocked_while_peer_is_allowed(self, load_rules):
        """Card verification: one rule, divergent verdicts per subscriber."""
        load_rules([two_group_rule(breached=True)])

        with pytest.raises(BudgetExceededError) as excinfo:
            enforcement.check_enforcement(subscriber_metadata(subscriber_id="sub-a"))

        assert enforcement.check_enforcement(subscriber_metadata(subscriber_id="sub-b")) is None
        assert excinfo.value.rule_name == RULE_NAME
        assert excinfo.value.rule_id == 4242
        assert str(excinfo.value) == f"Request blocked by Revenium enforcement rule: {RULE_NAME}"

    def test_error_reports_the_group_balance_against_the_rule_threshold(self, load_rules):
        load_rules([two_group_rule(breached=True)])

        with pytest.raises(BudgetExceededError) as excinfo:
            enforcement.check_enforcement(subscriber_metadata(subscriber_id="sub-a"))

        assert excinfo.value.current_value == 140.0  # the caller's own balance
        assert excinfo.value.threshold == 100.0     # threshold stays rule-level

    def test_rule_level_breach_alone_does_not_block_a_clean_group(self, load_rules):
        """Chosen semantics: the caller's entry outranks the aggregate flag."""
        load_rules([grouped_rule([group_entry("sub-b", 12.5, False)], breached=True)])

        assert enforcement.check_enforcement(subscriber_metadata(subscriber_id="sub-b")) is None

    def test_breached_group_blocks_even_when_rule_level_flag_is_false(self, load_rules):
        """The other direction of the same authority: pooled total may lag."""
        load_rules([grouped_rule([group_entry("sub-a", 140.0, True)], breached=False)])

        with pytest.raises(BudgetExceededError) as excinfo:
            enforcement.check_enforcement(subscriber_metadata(subscriber_id="sub-a"))

        assert excinfo.value.current_value == 140.0

    def test_subscriber_id_is_matched_before_email(self, load_rules):
        """A caller supplying both must not be attributed to the email bucket."""
        load_rules([grouped_rule(
            [group_entry("sub-a", 5.0, False), group_entry("a@example.test", 900.0, True)],
            breached=True,
        )])

        metadata = subscriber_metadata(subscriber_id="sub-a", email="a@example.test")

        assert enforcement.check_enforcement(metadata) is None

    def test_email_matches_when_the_entry_carries_no_id(self, load_rules):
        load_rules([grouped_rule([group_entry("a@example.test", 900.0, True)], breached=True)])

        with pytest.raises(BudgetExceededError):
            enforcement.check_enforcement(subscriber_metadata(email="a@example.test"))

    def test_overlong_subscriber_id_matches_the_entry_keyed_by_its_stored_prefix(self, load_rules):
        """BACK-4358: the record stores 255 characters, so the server keys the balance by them."""
        long_id = "u" * 300
        load_rules([grouped_rule([group_entry(long_id[:255], 900.0, True)], breached=True)])

        with pytest.raises(BudgetExceededError):
            enforcement.check_enforcement(subscriber_metadata(subscriber_id=long_id))

    def test_overlong_subscriber_email_matches_the_entry_keyed_by_its_stored_prefix(self, load_rules):
        """BACK-4358: the email is cut to 255 characters before the server keys the balance."""
        long_email = "e" * 300 + "@example.test"
        load_rules([grouped_rule([group_entry(long_email[:255], 900.0, True)], breached=True)])

        with pytest.raises(BudgetExceededError):
            enforcement.check_enforcement(subscriber_metadata(email=long_email))

    def test_flat_subscriber_metadata_pattern_matches(self, load_rules):
        """The legacy flat pattern resolves through the shared extractor too."""
        load_rules([two_group_rule(breached=True)])

        with pytest.raises(BudgetExceededError):
            enforcement.check_enforcement({"subscriber_id": "sub-a"})

    def test_partial_nested_subscriber_does_not_suppress_the_flat_id(self, load_rules):
        """Mixed metadata must still match an id-keyed breached entry.

        The shared extractor's nested branch suppresses the flat pattern
        entirely, so a partial nested subscriber (email only) plus a flat
        subscriber_id would otherwise never be looked up by id — letting the
        caller slip past their own breached balance.
        """
        load_rules([grouped_rule(
            [group_entry("sub-a", 900.0, True)],
            breached=True,
        )])

        metadata = {
            "subscriber": {"email": "a@example.test"},
            "subscriber_id": "sub-a",
        }

        with pytest.raises(BudgetExceededError):
            enforcement.check_enforcement(metadata)

    @pytest.mark.parametrize("bad_id", [{"oops": 1}, ["sub-a"], 42, 0.5])
    def test_non_string_subscriber_id_fails_open_not_loud(self, load_rules, bad_id):
        """Garbage in subscriber_id must never abort the pre-provider hook."""
        load_rules([grouped_rule([group_entry("sub-a", 900.0, True)], breached=True)])

        assert enforcement.check_enforcement({"subscriber_id": bad_id}) is None

    def test_nested_id_still_wins_over_a_flat_id(self, load_rules):
        """Precedence stays id-then-email across sources, nested first."""
        load_rules([grouped_rule(
            [group_entry("nested-id", 5.0, False), group_entry("flat-id", 900.0, True)],
            breached=True,
        )])

        metadata = {
            "subscriber": {"id": "nested-id"},
            "subscriber_id": "flat-id",
        }

        assert enforcement.check_enforcement(metadata) is None

    def test_shadow_mode_grouped_rule_never_blocks(self, load_rules):
        load_rules([grouped_rule(
            [group_entry("sub-a", 140.0, True)],
            breached=True,
            shadowMode=True,
        )])

        assert enforcement.check_enforcement(subscriber_metadata(subscriber_id="sub-a")) is None

    @pytest.mark.parametrize("metadata", [
        subscriber_metadata(email="Dev@Example.com"),
        subscriber_metadata(email=" dev@example.com "),
        {"subscriber_email": "DEV@example.com"},
    ], ids=["mixed-case", "padded", "flat-upper-case"])
    def test_mixed_case_or_padded_email_matches_the_normalized_entry(self, load_rules, metadata):
        """The server keys a per-person entry by the email trimmed and lower-cased."""
        load_rules([grouped_rule([group_entry("dev@example.com", 140.0, True)], breached=True)])

        with pytest.raises(BudgetExceededError) as excinfo:
            enforcement.check_enforcement(metadata)

        assert excinfo.value.current_value == 140.0

    def test_subscriber_id_match_stays_case_sensitive(self, load_rules):
        """An id is opaque on the server: ``sub-A`` and ``sub-a`` are two people."""
        load_rules([grouped_rule([group_entry("sub-A", 140.0, True)], breached=True)])

        assert enforcement.check_enforcement(subscriber_metadata(subscriber_id="sub-a")) is None

    def test_group_breakdown_is_not_rescanned_on_every_call(self, load_rules):
        class CountingList(list):
            iterations = 0

            def __iter__(self):
                self.iterations += 1
                return super().__iter__()

        def breakdown(prefix):
            return CountingList(
                group_entry(f"{prefix}-{index}@example.com", 1.0, False) for index in range(1000)
            )

        first, second = breakdown("first"), breakdown("second")
        load_rules([
            grouped_rule(first, ruleId=1),
            grouped_rule(second, ruleId=2, name="Second per-person cap"),
        ])
        newcomer = subscriber_metadata(subscriber_id="sub-new", email="new@example.com")

        for _ in range(50):
            assert enforcement.check_enforcement(newcomer) is None

        iterations = {"first": first.iterations, "second": second.iterations}
        assert max(iterations.values()) <= 1, iterations

    def test_a_current_breakdown_index_is_kept(self, load_rules):
        rule = two_group_rule(breached=False)
        load_rules([rule])

        enforcement.check_enforcement(subscriber_metadata(subscriber_id="sub-b"))

        assert id(rule["groupBreakdown"]) in enforcement._group_indexes

    def test_a_check_spanning_a_refresh_does_not_cache_the_replaced_breakdown(self, load_rules):
        replaced = two_group_rule(breached=True)
        load_rules([replaced])
        _, departments, _ = enforcement._get_rules()
        load_rules([two_group_rule(breached=True)])
        enforcement._begin_cache_generation()

        entry = enforcement._matching_group_entry(
            replaced["groupBreakdown"], subscriber_metadata(subscriber_id="sub-a"), departments.generation)

        assert entry is replaced["groupBreakdown"][0]
        assert enforcement._group_indexes == {}

    def test_empty_breakdown_on_a_per_person_rule_blocks_nobody(self, load_rules):
        """An empty list on a breached per-person rule means the balances were
        not read: the rule-level flag only says someone is over, not who."""
        load_rules([grouped_rule([], breached=True)])

        assert enforcement.check_enforcement(subscriber_metadata(email="bob@example.com")) is None
        assert enforcement.check_enforcement(subscriber_metadata(email="carol@example.com")) is None

    def test_malformed_entries_are_skipped_not_fatal(self, load_rules):
        load_rules([grouped_rule(
            ["not-a-dict", None, group_entry("sub-a", 140.0, True)],
            breached=True,
        )])

        with pytest.raises(BudgetExceededError):
            enforcement.check_enforcement(subscriber_metadata(subscriber_id="sub-a"))

    @pytest.mark.parametrize("metadata", [
        {"subscriber": {"id": 12345}},
        {"subscriber_id": 12345},
    ], ids=["nested", "flat"])
    def test_numeric_subscriber_id_is_blocked_by_its_breached_entry(self, load_rules, metadata):
        """The server stores an integer id as its decimal text and keys the balance by it."""
        load_rules([grouped_rule([group_entry("12345", 900.0, True)], breached=True)])

        with pytest.raises(BudgetExceededError) as excinfo:
            enforcement.check_enforcement(metadata)

        assert excinfo.value.current_value == 900.0

    def test_numeric_subscriber_id_from_the_decorator_is_blocked(self, load_rules):
        load_rules([grouped_rule([group_entry("12345", 900.0, True)], breached=True)])

        @revenium_metadata(subscriber={"id": 12345, "email": "n@example.test"})
        def call_the_model():
            enforcement.check_unmetered_call(None)

        with pytest.raises(BudgetExceededError):
            call_the_model()

    def test_text_subscriber_id_is_still_blocked(self, load_rules):
        load_rules([grouped_rule([group_entry("12345", 900.0, True)], breached=True)])

        with pytest.raises(BudgetExceededError):
            enforcement.check_enforcement({"subscriber": {"id": "12345"}})

    @pytest.mark.parametrize("shape", ["nested", "flat"])
    @pytest.mark.parametrize("bad_id", [True, False, 0, 12345.0, ""],
                             ids=["true", "false", "zero", "float", "empty"])
    def test_bool_zero_float_or_empty_subscriber_id_is_not_matched(self, load_rules, bad_id, shape):
        load_rules([grouped_rule(
            [group_entry(value, 900.0, True) for value in ("True", "1", "0", "12345")],
            breached=True,
        )])
        metadata = {"subscriber": {"id": bad_id}} if shape == "nested" else {"subscriber_id": bad_id}

        assert enforcement.check_enforcement(metadata) is None

    def test_subscriber_extractor_keeps_a_numeric_id_unchanged(self):
        """The metering record sends the id as given; only the cap lookup reads it as text."""
        assert extract_subscriber_from_metadata({"subscriber": {"id": 12345}}) == {"id": 12345}
        assert extract_subscriber_from_metadata({"subscriber_id": 12345}) == {"id": 12345}

    def test_numeric_id_with_email_falls_back_to_the_email_entry(self, load_rules):
        load_rules([grouped_rule([group_entry("n@example.test", 900.0, True)], breached=True)])

        with pytest.raises(BudgetExceededError):
            enforcement.check_enforcement({"subscriber": {"id": 12345, "email": "n@example.test"}})

    def test_numeric_id_entry_wins_over_the_email_entry(self, load_rules):
        load_rules([grouped_rule(
            [group_entry("12345", 5.0, False), group_entry("n@example.test", 900.0, True)],
            breached=True,
        )])

        assert enforcement.check_enforcement(
            {"subscriber": {"id": 12345, "email": "n@example.test"}}) is None

    @pytest.mark.parametrize("flat_id", [12345, "12345"], ids=["numeric", "text"])
    def test_flat_id_is_not_tried_when_the_nested_id_has_no_entry(self, load_rules, flat_id):
        """Metering sends the nested id alone, so another person's cap must not refuse this call."""
        load_rules([grouped_rule([group_entry("12345", 900.0, True)], breached=True)])

        metadata = {"subscriber": {"id": "current-user"}, "subscriber_id": flat_id}

        assert enforcement.check_enforcement(metadata) is None


class TestUnresolvableGroupFailsOpen:
    @pytest.mark.parametrize("metadata", [
        None,
        {},
        {"organizationName": "AcmeCorp"},
        {"subscriber": {}},
    ])
    def test_caller_without_a_group_key_is_not_blocked(self, load_rules, metadata):
        """Fail-open posture: unresolvable attribution is not a blocking path."""
        load_rules([two_group_rule(breached=True)])

        assert enforcement.check_enforcement(metadata) is None

    def test_resolved_caller_with_no_matching_entry_is_not_blocked(self, load_rules):
        """No entry means the sentinel bucket may own them; do not guess it."""
        load_rules([two_group_rule(breached=True)])

        assert enforcement.check_enforcement(subscriber_metadata(subscriber_id="sub-zzz")) is None


class TestPooledRulesUnchanged:
    def test_absent_group_breakdown_blocks_on_the_rule_level_breach(self, load_rules):
        rule = pooled_rule()
        load_rules([rule])
        assert "groupBreakdown" not in rule

        with pytest.raises(BudgetExceededError) as excinfo:
            enforcement.check_enforcement(subscriber_metadata(subscriber_id="sub-a"))

        assert excinfo.value.current_value == 640.0  # the pooled balance
        assert excinfo.value.threshold == 500.0
        assert excinfo.value.rule_name == POOLED_NAME

    @pytest.mark.parametrize("breakdown", [None, []])
    def test_null_or_empty_group_breakdown_is_treated_as_pooled(self, load_rules, breakdown):
        load_rules([pooled_rule(groupBreakdown=breakdown)])

        with pytest.raises(BudgetExceededError) as excinfo:
            enforcement.check_enforcement(subscriber_metadata(subscriber_id="sub-a"))

        assert excinfo.value.current_value == 640.0

    def test_pooled_rule_blocks_a_caller_with_no_metadata_at_all(self, load_rules):
        load_rules([pooled_rule()])

        with pytest.raises(BudgetExceededError):
            enforcement.check_enforcement(None)

    def test_untripped_pooled_rule_does_not_block(self, load_rules):
        load_rules([pooled_rule(breached=False)])

        assert enforcement.check_enforcement(None) is None

    def test_legacy_blocked_flag_still_trips_a_pooled_rule(self, load_rules):
        rule = pooled_rule(blocked=True)
        rule.pop("breached")
        load_rules([rule])

        with pytest.raises(BudgetExceededError):
            enforcement.check_enforcement(None)

    def test_shadow_mode_pooled_rule_never_blocks(self, load_rules):
        load_rules([pooled_rule(shadowMode=True)])

        assert enforcement.check_enforcement(None) is None

    def test_non_dict_rules_are_skipped(self, load_rules):
        load_rules(["junk", None, pooled_rule()])

        with pytest.raises(BudgetExceededError):
            enforcement.check_enforcement(None)


class TestNonBlockingActions:
    """Only a ``BLOCK`` action refuses a call; a warn-only rule is a notice.

    The server blocks only when ``!shadowMode && action == "BLOCK"``
    (``WARN_ONLY`` and the legacy ``THROTTLE`` never block), and so do the
    Node and Go SDKs. An absent ``action`` still counts as ``BLOCK``, as on
    the department path.
    """

    @staticmethod
    def warn_only_grouped_rule(action="WARN_ONLY"):
        rule = two_group_rule(breached=True)
        rule["action"] = action
        return rule

    @pytest.mark.parametrize("action", ["WARN_ONLY", "THROTTLE", "warn_only"])
    @pytest.mark.parametrize("shape", ["pooled", "grouped"])
    def test_breached_warn_only_rule_does_not_block(self, load_rules, action, shape):
        if shape == "pooled":
            load_rules([pooled_rule(action=action)])
        else:
            load_rules([self.warn_only_grouped_rule(action)])

        assert enforcement.check_enforcement(subscriber_metadata(subscriber_id="sub-a")) is None

    def test_breached_block_rule_still_raises(self, load_rules):
        load_rules([pooled_rule(action="BLOCK")])

        with pytest.raises(BudgetExceededError) as excinfo:
            enforcement.check_enforcement(subscriber_metadata(subscriber_id="sub-a"))

        assert excinfo.value.rule_name == POOLED_NAME
        assert excinfo.value.current_value == 640.0
        assert excinfo.value.rule_id == 99

    @pytest.mark.parametrize("shape", ["pooled", "grouped"])
    def test_warn_only_rule_does_not_shadow_a_later_block_rule(self, load_rules, shape):
        if shape == "pooled":
            warn_only = pooled_rule(ruleId=1, action="WARN_ONLY")
        else:
            warn_only = self.warn_only_grouped_rule()
        load_rules([warn_only, pooled_rule(ruleId=2, name="hard", action="BLOCK")])

        with pytest.raises(BudgetExceededError) as excinfo:
            enforcement.check_enforcement(subscriber_metadata(subscriber_id="sub-a"))

        assert excinfo.value.rule_name == "hard"

    def test_warn_only_breach_is_logged_once_per_breach(self, load_rules, caplog):
        def warnings_naming_the_rule():
            return [r for r in caplog.records
                    if r.levelno == logging.WARNING and "is over its threshold" in r.getMessage()]

        window_end = "2999-01-31T23:59:59.999999999Z"
        load_rules([pooled_rule(action="WARN_ONLY", windowEnd=window_end)])
        with caplog.at_level(logging.WARNING, logger="revenium_middleware.extension"):
            enforcement.check_enforcement(None)
            enforcement.check_enforcement(None)
            enforcement._begin_cache_generation()
            load_rules([pooled_rule(action="WARN_ONLY", windowEnd=window_end)])
            enforcement.check_enforcement(None)

            assert len(warnings_naming_the_rule()) == 1

            load_rules([pooled_rule(action="WARN_ONLY", windowEnd="2999-02-28T23:59:59.999999999Z")])
            enforcement.check_enforcement(None)

        assert len(warnings_naming_the_rule()) == 2

    def test_a_refresh_drops_the_warning_record_of_a_rule_no_longer_served(self, load_rules):
        load_rules([pooled_rule(ruleId=1, action="WARN_ONLY", windowEnd=OPEN_WINDOW)])
        enforcement.check_enforcement(None)
        assert "1" in enforcement._warned_non_blocking_rules

        load_rules([pooled_rule(ruleId=2, action="BLOCK", breached=False)])
        enforcement._begin_cache_generation()

        assert enforcement._warned_non_blocking_rules == {}

    def test_a_refresh_still_serving_the_rule_keeps_it_quiet_for_the_window(self, load_rules, caplog):
        def warn_only():
            return pooled_rule(ruleId=1, action="WARN_ONLY", windowEnd=OPEN_WINDOW)

        with caplog.at_level(logging.WARNING, logger="revenium_middleware.extension"):
            load_rules([warn_only()])
            enforcement.check_enforcement(None)
            load_rules([warn_only()])
            enforcement._begin_cache_generation()
            enforcement.check_enforcement(None)

        assert enforcement._warned_non_blocking_rules == {"1": OPEN_WINDOW}
        assert len([r for r in caplog.records if "is over its threshold" in r.getMessage()]) == 1


class TestWindowHasEnded:
    @pytest.mark.parametrize("window_end, ended", [
        ("2000-01-31T23:59:59.999999999Z", True),
        ("2000-01-31T23:59:59.9Z", True),
        ("2000-01-31T23:59:59Z", True),
        ("2000-01-31T23:59:59-05:00", True),
        ("2000-01-31T23:59:59", True),
        ("2000-01-31T23:59:59.999999999Z[UTC]", True),
        (OPEN_WINDOW, False),
        (949363199, False),
        ({"epochSecond": 949363199}, False),
        ("", False),
        ("garbage", False),
        (None, False),
    ], ids=[
        "nine-digits-z", "one-digit", "no-fraction", "offset", "naive", "zone-suffix",
        "far-future", "int", "dict", "empty", "garbage", "none",
    ])
    def test_window_has_ended_parses_server_shapes(self, window_end, ended):
        assert enforcement._window_has_ended({"windowEnd": window_end}) is ended


class TestEndedWindowStopsBlocking:
    """A breach belongs to its window; once ``windowEnd`` passes it is over,
    even while the rules service cannot be reached to say so."""

    @pytest.mark.parametrize("shape", ["pooled", "grouped"])
    def test_breached_rule_whose_window_ended_does_not_block(self, load_rules, shape):
        if shape == "pooled":
            rule = pooled_rule(windowEnd=ENDED_WINDOW)
        else:
            rule = two_group_rule(breached=True)
            rule["windowEnd"] = ENDED_WINDOW
        load_rules([rule])

        assert enforcement.check_enforcement(subscriber_metadata(subscriber_id="sub-a")) is None

    @pytest.mark.parametrize("window_end", [OPEN_WINDOW, "garbage", None, "absent"])
    def test_open_window_or_unparseable_window_still_blocks(self, load_rules, window_end):
        rule = pooled_rule()
        if window_end != "absent":
            rule["windowEnd"] = window_end
        load_rules([rule])

        with pytest.raises(BudgetExceededError):
            enforcement.check_enforcement(None)


class TestLegacyCredentialFilter:
    """The ``credential`` filter is a legacy-payload shim, not group matching."""

    def test_mismatched_legacy_credential_still_skips_the_rule(self, load_rules):
        load_rules([pooled_rule(credential="cred-a")])

        assert enforcement.check_enforcement({"subscriber_credential": "cred-b"}) is None

    def test_matching_legacy_credential_still_blocks(self, load_rules):
        load_rules([pooled_rule(credential="cred-a")])

        with pytest.raises(BudgetExceededError):
            enforcement.check_enforcement({"subscriber_credential": "cred-a"})


class TestDiskSnapshotRoundTrip:
    def test_group_breakdown_survives_persist_and_reload(self, cold_cache):
        rules = [two_group_rule(breached=True)]

        enforcement._persist_cache_to_disk(rules)
        enforcement._load_cache_from_disk()

        assert enforcement._cached_rules == rules
        entries = enforcement._cached_rules[0]["groupBreakdown"]
        assert [entry["groupValue"] for entry in entries] == ["sub-a", "sub-b"]
        assert entries[0]["currentValue"] == 140.0
        assert entries[0]["breached"] is True
        assert entries[1]["breached"] is False
        assert enforcement._cache_initialized is True

    def test_reloaded_grouped_rule_still_evaluates_per_group(self, cold_cache):
        """A restart must not degrade a grouped rule to rule-level blocking."""
        monkeypatch = cold_cache
        monkeypatch.setenv("REVENIUM_CIRCUIT_BREAKER_ENABLED", "true")
        monkeypatch.delenv("REVENIUM_BYPASS", raising=False)
        monkeypatch.delenv("REVENIUM_CB_FAIL_MODE", raising=False)
        monkeypatch.setattr(enforcement, "_ensure_poller_running", lambda: None)
        enforcement._persist_cache_to_disk([two_group_rule(breached=True)])

        enforcement._load_cache_from_disk()
        # The snapshot loads deliberately stale; pin it fresh so the assertion
        # exercises the evaluation rather than a refresh.
        monkeypatch.setattr(enforcement, "_cache_timestamp", time.monotonic())

        with pytest.raises(BudgetExceededError):
            enforcement.check_enforcement(subscriber_metadata(subscriber_id="sub-a"))
        assert enforcement.check_enforcement(subscriber_metadata(subscriber_id="sub-b")) is None

    def test_snapshot_without_group_breakdown_is_not_an_error(self, cold_cache):
        """The field is API-read-only, so a snapshot legitimately omits it."""
        enforcement._persist_cache_to_disk([pooled_rule()])

        enforcement._load_cache_from_disk()

        assert "groupBreakdown" not in enforcement._cached_rules[0]
        assert enforcement._cache_initialized is True


    def test_reloaded_snapshot_from_an_ended_window_does_not_block(self, cold_cache):
        """A restart during a rules outage must not revive yesterday's breach."""
        monkeypatch = cold_cache
        monkeypatch.setenv("REVENIUM_CIRCUIT_BREAKER_ENABLED", "true")
        monkeypatch.delenv("REVENIUM_BYPASS", raising=False)
        monkeypatch.delenv("REVENIUM_CB_FAIL_MODE", raising=False)
        monkeypatch.setattr(enforcement, "_ensure_poller_running", lambda: None)
        enforcement._persist_cache_to_disk([
            pooled_rule(ruleId=1, name="ended", windowEnd=ENDED_WINDOW),
            pooled_rule(ruleId=2, name="open", windowEnd=OPEN_WINDOW),
        ])
        assert os.path.exists(enforcement._cache_file_path())

        enforcement._load_cache_from_disk()
        assert len(enforcement._cached_rules) == 2
        assert enforcement._cache_initialized is True

        with pytest.raises(BudgetExceededError) as excinfo:
            enforcement.check_enforcement(None)
        assert excinfo.value.rule_name == "open"


class TestSnapshotIsPerTeam:
    """Two teams' services on one host may share one ``REVENIUM_CACHE_DIR``.

    A restarted process must load only the snapshot saved for its own team:
    another team's rules would judge its calls, and another team's department
    map is keyed by that team's people's email addresses.
    """

    @staticmethod
    def forget_the_cache(monkeypatch):
        monkeypatch.setattr(enforcement, "_disk_load_attempted", False)
        monkeypatch.setattr(enforcement, "_cached_rules", [])
        monkeypatch.setattr(enforcement, "_cached_org_unit_blocks", {})
        monkeypatch.setattr(enforcement, "_cached_org_unit_block_balances", {})
        monkeypatch.setattr(enforcement, "_cached_org_unit_warnings", {})
        monkeypatch.setattr(enforcement, "_cache_initialized", False)

    @staticmethod
    def enforce_without_a_poller(monkeypatch):
        monkeypatch.setenv("REVENIUM_CIRCUIT_BREAKER_ENABLED", "true")
        monkeypatch.delenv("REVENIUM_BYPASS", raising=False)
        monkeypatch.delenv("REVENIUM_CB_FAIL_MODE", raising=False)
        monkeypatch.setattr(enforcement, "_ensure_poller_running", lambda: None)

    def test_second_team_never_loads_the_first_teams_snapshot(self, cold_cache):
        monkeypatch = cold_cache
        self.enforce_without_a_poller(monkeypatch)
        monkeypatch.setenv("REVENIUM_TEAM_ID", "team-a")
        enforcement._persist_cache_to_disk(
            [pooled_rule(name="Team A daily cap"), org_unit_rule()],
            {DEPT_EMAIL: ORG_UNIT_RULE_ID},
        )
        assert os.path.exists(enforcement._cache_file_path())

        self.forget_the_cache(monkeypatch)
        monkeypatch.setenv("REVENIUM_TEAM_ID", "team-b")
        enforcement._load_cache_from_disk()

        assert enforcement._cached_rules == []
        assert enforcement._cached_org_unit_blocks == {}
        assert enforcement._cached_org_unit_block_balances == {}
        assert enforcement._cached_org_unit_warnings == {}
        assert enforcement._cache_initialized is False
        assert enforcement.check_enforcement(None) is None

    def test_same_team_still_reloads_its_snapshot(self, cold_cache):
        monkeypatch = cold_cache
        self.enforce_without_a_poller(monkeypatch)
        monkeypatch.setenv("REVENIUM_TEAM_ID", "team-a")
        enforcement._persist_cache_to_disk([pooled_rule(name="Team A daily cap")])
        assert os.path.exists(enforcement._cache_file_path())

        self.forget_the_cache(monkeypatch)
        enforcement._load_cache_from_disk()

        assert enforcement._cached_rules == [pooled_rule(name="Team A daily cap")]
        assert enforcement._cache_initialized is True
        with pytest.raises(BudgetExceededError) as excinfo:
            enforcement.check_enforcement(None)
        assert excinfo.value.rule_name == "Team A daily cap"


class TestDepartmentBudgetBlocks:
    """The email map is the whole verdict for a department budget."""

    def test_mapped_caller_is_blocked_with_the_rule_details(self, load_rules):
        """Card verification: a breached department budget now blocks."""
        load_rules([org_unit_rule()], {DEPT_EMAIL: ORG_UNIT_RULE_ID})

        with pytest.raises(BudgetExceededError) as excinfo:
            enforcement.check_enforcement(subscriber_metadata(email=DEPT_EMAIL))

        assert excinfo.value.rule_name == DEPARTMENT_NAME
        assert excinfo.value.current_value == 1450.0
        assert excinfo.value.threshold == 1000.0
        assert excinfo.value.rule_id == ORG_UNIT_RULE_ID
        assert str(excinfo.value) == (
            f"Request blocked by Revenium enforcement rule: {DEPARTMENT_NAME}"
        )

    def test_a_colleague_outside_the_map_is_not_blocked(self, load_rules):
        load_rules([org_unit_rule()], {DEPT_EMAIL: ORG_UNIT_RULE_ID})

        assert enforcement.check_enforcement(subscriber_metadata(email=OTHER_EMAIL)) is None

    @pytest.mark.parametrize("metadata", [
        None,
        {},
        {"subscriber": {"id": "sub-a"}},
        {"organizationName": "AcmeCorp"},
    ])
    def test_caller_without_an_email_is_not_blocked(self, load_rules, metadata):
        """No email means no key -- and no sentinel is invented for them."""
        load_rules([org_unit_rule()], {DEPT_EMAIL: ORG_UNIT_RULE_ID})

        assert enforcement.check_enforcement(metadata) is None

    def test_flat_subscriber_email_is_looked_up_too(self, load_rules):
        load_rules([org_unit_rule()], {DEPT_EMAIL: ORG_UNIT_RULE_ID})

        with pytest.raises(BudgetExceededError):
            enforcement.check_enforcement({"subscriber_email": DEPT_EMAIL})

    def test_map_hit_for_an_unknown_rule_id_still_blocks(self, load_rules):
        """Stale or racing payload: the server already decided; enforce anyway."""
        load_rules([], {DEPT_EMAIL: ORG_UNIT_RULE_ID})

        with pytest.raises(BudgetExceededError) as excinfo:
            enforcement.check_enforcement(subscriber_metadata(email=DEPT_EMAIL))

        assert excinfo.value.rule_name == "Department budget"
        assert excinfo.value.rule_id == ORG_UNIT_RULE_ID
        assert excinfo.value.current_value is None

    def test_rules_are_resolved_by_id_not_by_name(self, load_rules):
        """Names are not unique; the map's value is an id, so ids decide."""
        namesake = org_unit_rule(ruleId=1, name=DEPARTMENT_NAME, currentValue=1.0)
        load_rules([namesake, org_unit_rule()], {DEPT_EMAIL: ORG_UNIT_RULE_ID})

        with pytest.raises(BudgetExceededError) as excinfo:
            enforcement.check_enforcement(subscriber_metadata(email=DEPT_EMAIL))

        assert excinfo.value.rule_id == ORG_UNIT_RULE_ID
        assert excinfo.value.current_value == 1450.0

    def test_shadow_mode_department_rule_does_not_block(self, load_rules):
        load_rules([org_unit_rule(shadowMode=True)], {DEPT_EMAIL: ORG_UNIT_RULE_ID})

        assert enforcement.check_enforcement(subscriber_metadata(email=DEPT_EMAIL)) is None

    def test_shadow_mode_department_rule_leaves_other_rules_evaluating(self, load_rules):
        """Observe-and-log on one rule must not disarm the rest."""
        load_rules(
            [org_unit_rule(shadowMode=True), pooled_rule()],
            {DEPT_EMAIL: ORG_UNIT_RULE_ID},
        )

        with pytest.raises(BudgetExceededError) as excinfo:
            enforcement.check_enforcement(subscriber_metadata(email=DEPT_EMAIL))

        assert excinfo.value.rule_name == POOLED_NAME

    @pytest.mark.parametrize("action", ["WARN_ONLY", "NOTIFY", "warn_only"])
    def test_non_blocking_action_does_not_raise(self, load_rules, action):
        load_rules([org_unit_rule(action=action)], {DEPT_EMAIL: ORG_UNIT_RULE_ID})

        assert enforcement.check_enforcement(subscriber_metadata(email=DEPT_EMAIL)) is None

    def test_lowercase_block_action_still_blocks(self, load_rules):
        load_rules([org_unit_rule(action="block")], {DEPT_EMAIL: ORG_UNIT_RULE_ID})

        with pytest.raises(BudgetExceededError):
            enforcement.check_enforcement(subscriber_metadata(email=DEPT_EMAIL))

    def test_absent_action_falls_toward_enforcing(self, load_rules):
        """The map is already the server's blocking decision."""
        rule = org_unit_rule()
        rule.pop("action")
        load_rules([rule], {DEPT_EMAIL: ORG_UNIT_RULE_ID})

        with pytest.raises(BudgetExceededError):
            enforcement.check_enforcement(subscriber_metadata(email=DEPT_EMAIL))

    def test_department_block_for_an_ended_window_does_not_block(self, load_rules):
        load_rules([org_unit_rule(windowEnd=ENDED_WINDOW)], {DEPT_EMAIL: ORG_UNIT_RULE_ID})

        assert enforcement.check_enforcement(subscriber_metadata(email=DEPT_EMAIL)) is None

    def test_department_block_is_independent_of_the_rule_breached_flag(self, load_rules):
        """The map, not the rule's own flag, is the verdict."""
        load_rules([org_unit_rule(breached=False)], {DEPT_EMAIL: ORG_UNIT_RULE_ID})

        with pytest.raises(BudgetExceededError):
            enforcement.check_enforcement(subscriber_metadata(email=DEPT_EMAIL))

    def test_a_department_block_carries_the_rules_deny_message(self, load_rules):
        """BACK-4407: the message the customer wrote for the department's rule reaches the caller."""
        deny = "Ask your manager before retrying. {spent} of {limit}"
        load_rules([org_unit_rule(denyMessage=deny)], {DEPT_EMAIL: ORG_UNIT_RULE_ID})

        with pytest.raises(BudgetExceededError) as excinfo:
            enforcement.check_enforcement(subscriber_metadata(email=DEPT_EMAIL))

        assert str(excinfo.value) == f"Request blocked by Revenium enforcement rule: {DEPARTMENT_NAME}. {deny}"
        assert excinfo.value.deny_message == deny

    @pytest.mark.parametrize("deny_message", [None, "", "   ", 42, ["x"], {"a": 1}],
                             ids=["none", "empty", "blank", "number", "list", "dict"])
    def test_a_department_block_with_an_unusable_deny_message_still_blocks_with_todays_text(
            self, load_rules, deny_message):
        load_rules([org_unit_rule(denyMessage=deny_message)], {DEPT_EMAIL: ORG_UNIT_RULE_ID})

        with pytest.raises(BudgetExceededError) as excinfo:
            enforcement.check_enforcement(subscriber_metadata(email=DEPT_EMAIL))

        assert str(excinfo.value) == f"Request blocked by Revenium enforcement rule: {DEPARTMENT_NAME}"
        assert getattr(excinfo.value, "deny_message", "unset") is None


class TestNestedEmailIsAuthoritative:
    """The flat subscriber_email is a fallback, never a second identity.

    Mixed metadata can name two different people; looking both up would let a
    stale or spoofed flat email block the caller against a department that is
    not theirs (hypercurrent#101 review).
    """

    def test_flat_email_is_ignored_when_a_nested_email_exists(self, load_rules):
        load_rules([org_unit_rule()], {DEPT_EMAIL: ORG_UNIT_RULE_ID})
        metadata = subscriber_metadata(email=OTHER_EMAIL)
        metadata["subscriber_email"] = DEPT_EMAIL  # blocked, but not who the call is for

        assert enforcement.check_enforcement(metadata) is None

    def test_flat_email_still_resolves_when_no_nested_email_exists(self, load_rules):
        load_rules([org_unit_rule()], {DEPT_EMAIL: ORG_UNIT_RULE_ID})

        with pytest.raises(BudgetExceededError):
            enforcement.check_enforcement({"subscriber_email": DEPT_EMAIL})


class TestDepartmentMapRobustness:
    """A malformed map must degrade to "no block", never to a raised TypeError."""

    @pytest.mark.parametrize("blocks", [
        None,
        [],
        "nope",
        {DEPT_EMAIL: "7777"},
        {DEPT_EMAIL: None},
        {DEPT_EMAIL: True},
        {DEPT_EMAIL: [7777]},
        {DEPT_EMAIL: {"ruleId": 7777}},
        {7777: 7777},
    ])
    def test_malformed_map_never_blocks_and_never_raises(self, load_rules, blocks):
        load_rules([org_unit_rule()], blocks)

        assert enforcement.check_enforcement(subscriber_metadata(email=DEPT_EMAIL)) is None

    @pytest.mark.parametrize("bad_email", [{"oops": 1}, ["a@b.test"], 42])
    def test_non_string_email_fails_open(self, load_rules, bad_email):
        load_rules([org_unit_rule()], {DEPT_EMAIL: ORG_UNIT_RULE_ID})

        assert enforcement.check_enforcement({"subscriber_email": bad_email}) is None

    def test_non_dict_rules_in_the_cache_are_skipped_during_resolution(self, load_rules):
        load_rules(["junk", None, org_unit_rule()], {DEPT_EMAIL: ORG_UNIT_RULE_ID})

        with pytest.raises(BudgetExceededError) as excinfo:
            enforcement.check_enforcement(subscriber_metadata(email=DEPT_EMAIL))

        assert excinfo.value.rule_name == DEPARTMENT_NAME


class TestOrgUnitRulesSkipThePerRuleLoop:
    """Their verdict comes only from the map, so the loop must ignore them."""

    def test_breached_org_unit_rule_without_a_map_entry_does_not_block(self, load_rules):
        load_rules([org_unit_rule(breached=True)], {})

        assert enforcement.check_enforcement(subscriber_metadata(email=DEPT_EMAIL)) is None

    @pytest.mark.parametrize("department_fields", [
        {"groupBy": "ORG_UNIT"},
        {"groupBy": None, "orgUnitId": "ou-42", "orgUnitPath": "/root/eng"},
        {"groupBy": "department"},
    ], ids=["legacy-group-by", "legacy-org-unit-id", "lower-case-group-by"])
    def test_every_department_spelling_is_skipped_by_the_loop(
        self, load_rules, department_fields
    ):
        """BACK-3540: the pre-BACK-3448 spellings still classify for one release."""
        load_rules([org_unit_rule(breached=True, **department_fields)], {})

        assert enforcement.check_enforcement(subscriber_metadata(email=OTHER_EMAIL)) is None

    def test_breached_org_unit_rule_does_not_block_an_unmapped_colleague(self, load_rules):
        load_rules([org_unit_rule()], {DEPT_EMAIL: ORG_UNIT_RULE_ID})

        assert enforcement.check_enforcement(subscriber_metadata(email=OTHER_EMAIL)) is None

    def test_breached_ancestor_cap_rule_does_not_block_the_whole_team(self, load_rules):
        # hypercurrent#101 review (cross-repo): ancestor-cap department rules
        # carry departmentId != null with a null groupBy, and rule-level breached
        # means "this department is over budget" — not "this caller is". The
        # server excludes them from applicableRules on departmentId alone; so
        # must the SDK, or one department's breach blocks every colleague.
        ancestor_cap = org_unit_rule(
            groupBy=None,
            departmentId="ou-42",
            departmentPath="/root/eng",
            breached=True,
        )
        load_rules([ancestor_cap], {DEPT_EMAIL: ORG_UNIT_RULE_ID})

        assert enforcement.check_enforcement(subscriber_metadata(email=OTHER_EMAIL)) is None

    def test_ancestor_cap_map_hit_still_blocks_the_department_member(self, load_rules):
        ancestor_cap = org_unit_rule(
            groupBy=None,
            departmentId="ou-42",
            departmentPath="/root/eng",
            breached=True,
        )
        load_rules([ancestor_cap], {DEPT_EMAIL: ORG_UNIT_RULE_ID})

        with pytest.raises(BudgetExceededError) as excinfo:
            enforcement.check_enforcement(subscriber_metadata(email=DEPT_EMAIL))
        assert DEPARTMENT_NAME in str(excinfo.value)

    def test_org_unit_rule_with_a_group_breakdown_is_still_skipped(self, load_rules):
        """No crash, and no block from the grouped path either."""
        load_rules(
            [org_unit_rule(groupBreakdown=[group_entry("sub-a", 1450.0, True)])],
            {},
        )

        assert enforcement.check_enforcement(
            subscriber_metadata(subscriber_id="sub-a", email=DEPT_EMAIL)
        ) is None


class TestSubscriberGroupedPathUnchanged:
    """Regression: the department map must not disturb subscriber grouping."""

    def test_grouped_rule_still_splits_verdicts_with_a_map_present(self, load_rules):
        load_rules([two_group_rule(breached=True)], {OTHER_EMAIL: 4242})

        with pytest.raises(BudgetExceededError) as excinfo:
            enforcement.check_enforcement(subscriber_metadata(subscriber_id="sub-a"))

        assert enforcement.check_enforcement(subscriber_metadata(subscriber_id="sub-b")) is None
        assert excinfo.value.rule_name == RULE_NAME
        assert excinfo.value.current_value == 140.0

    def test_unmapped_email_leaves_the_group_verdict_alone(self, load_rules):
        """A caller with an email that is not in the map takes the old path."""
        load_rules([two_group_rule(breached=True)], {OTHER_EMAIL: 4242})

        metadata = subscriber_metadata(subscriber_id="sub-b", email=DEPT_EMAIL)

        assert enforcement.check_enforcement(metadata) is None

    def test_pooled_rule_still_blocks_with_an_empty_map(self, load_rules):
        load_rules([pooled_rule()], {})

        with pytest.raises(BudgetExceededError) as excinfo:
            enforcement.check_enforcement(subscriber_metadata(email=DEPT_EMAIL))

        assert excinfo.value.rule_name == POOLED_NAME


class TestDepartmentMapDiskSnapshot:
    def test_map_survives_persist_and_reload(self, cold_cache):
        rules = [org_unit_rule()]
        blocks = {DEPT_EMAIL: ORG_UNIT_RULE_ID}

        enforcement._persist_cache_to_disk(rules, blocks)
        enforcement._load_cache_from_disk()

        assert enforcement._cached_rules == rules
        assert enforcement._cached_org_unit_blocks == blocks
        assert enforcement._cache_initialized is True

    def test_snapshot_written_without_a_map_loads_an_empty_one(self, cold_cache):
        enforcement._persist_cache_to_disk([pooled_rule()])
        enforcement._load_cache_from_disk()

        assert enforcement._cached_org_unit_blocks == {}
        assert enforcement._cache_initialized is True

    def test_rules_snapshot_stays_a_bare_list_an_older_sdk_can_read(self, cold_cache, tmp_path):
        # Rollback guarantee (hypercurrent#101 review): the rules file keeps the
        # exact shape every published SDK reads — a bare JSON list — so a
        # downgraded process does not discard the cache. The map rides in its
        # own file the old loader never opens.
        import json

        rules = [org_unit_rule()]
        enforcement._persist_cache_to_disk(rules, {DEPT_EMAIL: ORG_UNIT_RULE_ID})

        with open(enforcement._cache_file_path(), encoding="utf-8") as handle:
            on_disk = json.load(handle)
        assert on_disk == rules  # a bare list, not a dict wrapper

        with open(
            enforcement._org_unit_blocks_cache_file_path(), encoding="utf-8"
        ) as handle:
            envelope = json.load(handle)
        # The map rides in an envelope that names the rules snapshot it was cut
        # from; only the new loader opens this file, so the envelope carries no
        # compatibility burden.
        assert envelope["blocks"] == {DEPT_EMAIL: ORG_UNIT_RULE_ID}
        assert envelope["rules_fingerprint"] == enforcement._rules_fingerprint(rules)

    def test_torn_snapshot_pair_drops_the_map(self, cold_cache, tmp_path):
        """A crash between the two writes can leave a map beside rules it was
        not computed from; pairing them would interpret rule IDs against the
        wrong action/shadowMode. The mismatched map must be dropped."""
        import json

        enforcement._persist_cache_to_disk(
            [org_unit_rule()], {DEPT_EMAIL: ORG_UNIT_RULE_ID}
        )
        # Simulate the torn pair: the rules file is replaced (as an interrupted
        # later persist would) while the map file stays.
        older_rules = [pooled_rule()]
        with open(enforcement._cache_file_path(), "w", encoding="utf-8") as handle:
            json.dump(older_rules, handle)

        enforcement._load_cache_from_disk()

        assert enforcement._cached_rules == older_rules
        assert enforcement._cached_org_unit_blocks == {}
        assert enforcement._cache_initialized is True

    def test_legacy_flat_map_file_is_dropped_not_crashed_on(self, cold_cache, tmp_path):
        """A pre-envelope map file (flat email->ruleId dict) carries no pairing
        proof, so it is skipped; department blocks resume on the next fetch."""
        import json

        enforcement._persist_cache_to_disk([org_unit_rule()], {DEPT_EMAIL: ORG_UNIT_RULE_ID})
        with open(
            enforcement._org_unit_blocks_cache_file_path(), "w", encoding="utf-8"
        ) as handle:
            json.dump({DEPT_EMAIL: ORG_UNIT_RULE_ID}, handle)

        enforcement._load_cache_from_disk()

        assert enforcement._cached_org_unit_blocks == {}
        assert enforcement._cache_initialized is True

    @pytest.mark.skipif(os.name == "nt", reason="POSIX file modes")
    def test_department_map_file_is_owner_only(self, cold_cache, tmp_path):
        # The map is keyed by subscriber email (PII); it must not be left
        # world-readable via the process umask.
        enforcement._persist_cache_to_disk([pooled_rule()], {DEPT_EMAIL: ORG_UNIT_RULE_ID})

        mode = os.stat(enforcement._org_unit_blocks_cache_file_path()).st_mode & 0o777
        assert mode == 0o600

    def test_missing_map_file_loads_rules_with_no_department_blocks(self, cold_cache, tmp_path):
        enforcement._persist_cache_to_disk([pooled_rule()], {DEPT_EMAIL: ORG_UNIT_RULE_ID})
        os.unlink(enforcement._org_unit_blocks_cache_file_path())

        enforcement._load_cache_from_disk()

        assert enforcement._cached_rules == [pooled_rule()]
        assert enforcement._cached_org_unit_blocks == {}

    def test_legacy_bare_list_snapshot_loads_with_an_empty_map(self, cold_cache, tmp_path):
        """A snapshot written before the map existed must still load."""
        with open(enforcement._cache_file_path(), "w", encoding="utf-8") as handle:
            json.dump([pooled_rule()], handle)

        enforcement._load_cache_from_disk()

        assert enforcement._cached_rules == [pooled_rule()]
        assert enforcement._cached_org_unit_blocks == {}
        assert enforcement._cache_initialized is True

    def test_reloaded_snapshot_still_blocks_the_department(self, cold_cache):
        """A restart must not fail a breached department budget open."""
        monkeypatch = cold_cache
        monkeypatch.setenv("REVENIUM_CIRCUIT_BREAKER_ENABLED", "true")
        monkeypatch.delenv("REVENIUM_BYPASS", raising=False)
        monkeypatch.delenv("REVENIUM_CB_FAIL_MODE", raising=False)
        monkeypatch.setattr(enforcement, "_ensure_poller_running", lambda: None)
        enforcement._persist_cache_to_disk([org_unit_rule()], {DEPT_EMAIL: ORG_UNIT_RULE_ID})

        enforcement._load_cache_from_disk()
        # The snapshot loads deliberately stale; pin it fresh so the assertion
        # exercises the evaluation rather than a refresh.
        monkeypatch.setattr(enforcement, "_cache_timestamp", time.monotonic())

        with pytest.raises(BudgetExceededError) as excinfo:
            enforcement.check_enforcement(subscriber_metadata(email=DEPT_EMAIL))
        assert excinfo.value.rule_name == DEPARTMENT_NAME
        assert enforcement.check_enforcement(subscriber_metadata(email=OTHER_EMAIL)) is None
