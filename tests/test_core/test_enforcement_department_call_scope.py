"""A department cap blocks only the calls its own filters cover.

A department cost cap can carry provider, model, agent, task type, product,
organization and credential filter rows, and the server counts only the
matching spend toward it. Its verdict reached the SDK as one rule id per person
(``departmentBudgetBlocks``), which the SDK applied to every call that person
made: a breached cap on ``PROVIDER IS openai`` refused their Anthropic calls
too.

The server now also publishes ``departmentBudgetCandidates``: per person, every
department cap that could apply, best first. The SDK takes the first candidate
whose filter rows match the call, ignoring the department and subscriber rows
the server already decided, comparing without regard to letter case as the
server's total does, and counting a row it cannot judge as matching. A person
with no candidate list keeps the single-slot verdict exactly as before.

``department_call_scope_cases.json`` is the hypercurrent fixture shared with
metering and the Go SDK; every case runs here through the real response
parser and the real pre-call check.
"""
import json
import logging
from pathlib import Path

import pytest

from revenium_middleware._core import enforcement
from revenium_middleware._core.exceptions import BudgetExceededError

from .conftest import make_response, stub_get

CASES = json.loads(
    (Path(__file__).parents[1] / "fixtures" / "department_call_scope_cases.json").read_text()
)["cases"]

MIDDLEWARE_LOGGER = "revenium_middleware.extension"
EMAIL = "alice@example.com"
OPENAI_RULE_ID = 1
ANTHROPIC_RULE_ID = 2
WINDOW_STILL_OPEN = "2999-01-01T00:00:00Z"
WINDOW_ENDED = "2000-01-01T00:00:00Z"


def call_arguments(call):
    """The fixture's call as the SDK's pre-call check receives it."""
    metadata = {"subscriber": {"email": call["subscriberEmail"]}}
    for field, key in (("agent", "agent"), ("taskType", "task_type"),
                       ("productName", "product_name"), ("organizationName", "organization_name")):
        if field in call:
            metadata[key] = call[field]
    credential = {key: call[field] for field, key in (("credentialName", "name"), ("credentialValue", "value"))
                  if field in call}
    if credential:
        metadata["subscriber"]["credential"] = credential
    return metadata, {"model": call.get("model"), "provider": call.get("provider")}


def refused_by(metadata=None, **keywords):
    """The error the pre-call check raised, or ``None`` when it let the call through."""
    try:
        enforcement.check_enforcement(metadata, **keywords)
    except BudgetExceededError as error:
        return error
    return None


def department_cap(rule_id, provider=None, **overrides):
    """A breached live department cap, optionally scoped to one provider."""
    filters = [{"dimension": "DEPARTMENT", "operator": "IS", "value": "7", "values": []}]
    if provider is not None:
        filters.append({"dimension": "PROVIDER", "operator": "IS", "value": provider, "values": []})
    rule = {
        "ruleId": rule_id,
        "name": f"Department cap {rule_id}",
        "metricType": "TOTAL_COST",
        "threshold": 10.0,
        "currentValue": 40.0,
        "windowEnd": WINDOW_STILL_OPEN,
        "breached": True,
        "shadowMode": False,
        "action": "BLOCK",
        "filters": filters,
        "groupBy": "SUBSCRIBER",
        "departmentId": 7,
    }
    rule.update(overrides)
    return rule


def candidate(rule_id, tier="BLOCK", balance=12.0):
    return {"ruleId": rule_id, "tier": tier, "departmentId": 7, "balance": balance}


def alice():
    return {"subscriber": {"email": EMAIL}}


def warn_records(caplog):
    return [record for record in caplog.records if "Approaching" in record.getMessage()]


@pytest.mark.parametrize("case", CASES, ids=[case["name"] for case in CASES])
def test_shared_department_call_scope_case(case, department_cache, fetch_env):
    monkeypatch, _ = fetch_env
    monkeypatch.delenv("REVENIUM_CACHE_DIR", raising=False)
    department_cache([])
    stub_get(monkeypatch, [make_response(200, json_body=case["payload"])])
    enforcement._refresh_cache()
    metadata, keywords = call_arguments(case["call"])

    error = refused_by(metadata, **keywords)

    if case["expect"]["blocked"]:
        assert error is not None, "the call went through"
        assert error.rule_id == case["expect"]["ruleId"]
    else:
        assert error is None, f"the call was refused by rule {error.rule_id}"


class TestTheChosenCandidateDescribesTheBlock:

    def test_the_block_reports_the_candidates_balance_not_the_slots(self, department_cache):
        department_cache(
            [department_cap(OPENAI_RULE_ID, "openai"), department_cap(ANTHROPIC_RULE_ID, "anthropic")],
            {EMAIL: OPENAI_RULE_ID}, {EMAIL: 12.0}, {},
            {EMAIL: [candidate(OPENAI_RULE_ID, balance=12.0), candidate(ANTHROPIC_RULE_ID, balance=15.5)]},
        )

        error = refused_by(alice(), model="claude-sonnet-4-5", provider="anthropic")

        assert error.rule_id == ANTHROPIC_RULE_ID
        assert error.rule_name == "Department cap 2"
        assert error.current_value == 15.5
        assert error.threshold == 10.0

    def test_a_candidate_with_no_balance_reports_the_rules_own_value(self, department_cache):
        department_cache(
            [department_cap(OPENAI_RULE_ID, "openai")],
            {EMAIL: OPENAI_RULE_ID}, {EMAIL: 12.0}, {},
            {EMAIL: [candidate(OPENAI_RULE_ID, balance=None)]},
        )

        error = refused_by(alice(), model="gpt-4o", provider="openai")

        assert error.current_value == 40.0

    def test_a_balance_sent_as_a_string_is_reported_as_a_number(self, department_cache, fetch_env):
        monkeypatch, _ = fetch_env
        monkeypatch.delenv("REVENIUM_CACHE_DIR", raising=False)
        department_cache([])
        stub_get(monkeypatch, [make_response(200, json_body={
            "rules": [department_cap(OPENAI_RULE_ID, "openai")],
            "departmentBudgetBlocks": {EMAIL: OPENAI_RULE_ID},
            "departmentBudgetCandidates": {EMAIL: [candidate(OPENAI_RULE_ID, balance="13.25")]},
        })])
        enforcement._refresh_cache()

        error = refused_by(alice(), model="gpt-4o", provider="openai")

        assert error.current_value == 13.25


class TestWalkingTheCandidates:

    def test_a_candidate_whose_window_has_ended_is_skipped_for_a_later_live_one(self, department_cache):
        department_cache(
            [department_cap(OPENAI_RULE_ID, windowEnd=WINDOW_ENDED), department_cap(ANTHROPIC_RULE_ID)],
            {EMAIL: OPENAI_RULE_ID}, {}, {},
            {EMAIL: [candidate(OPENAI_RULE_ID), candidate(ANTHROPIC_RULE_ID)]},
        )

        error = refused_by(alice(), model="gpt-4o", provider="openai")

        assert error.rule_id == ANTHROPIC_RULE_ID

    def test_only_ended_candidates_let_the_call_through_without_the_slot(self, department_cache):
        """The slot names a live cap; once a list exists it is never the fallback."""
        department_cache(
            [department_cap(OPENAI_RULE_ID, windowEnd=WINDOW_ENDED), department_cap(ANTHROPIC_RULE_ID)],
            {EMAIL: ANTHROPIC_RULE_ID}, {}, {},
            {EMAIL: [candidate(OPENAI_RULE_ID)]},
        )

        assert refused_by(alice(), model="gpt-4o", provider="openai") is None

    def test_the_first_match_decides_even_when_it_does_not_block(self, department_cache):
        department_cache(
            [department_cap(OPENAI_RULE_ID, shadowMode=True), department_cap(ANTHROPIC_RULE_ID)],
            {EMAIL: ANTHROPIC_RULE_ID}, {}, {},
            {EMAIL: [candidate(OPENAI_RULE_ID), candidate(ANTHROPIC_RULE_ID)]},
        )

        assert refused_by(alice(), model="gpt-4o", provider="openai") is None

    def test_the_caller_is_found_under_the_normalized_address(self, department_cache):
        department_cache(
            [department_cap(OPENAI_RULE_ID, "openai")],
            {}, {}, {},
            {EMAIL: [candidate(OPENAI_RULE_ID)]},
        )

        error = refused_by({"subscriber": {"email": "  Alice@Example.COM "}}, model="gpt-4o", provider="openai")

        assert error.rule_id == OPENAI_RULE_ID

    def test_a_list_of_malformed_items_falls_back_to_the_single_slot(self, department_cache, fetch_env):
        monkeypatch, _ = fetch_env
        monkeypatch.delenv("REVENIUM_CACHE_DIR", raising=False)
        department_cache([])
        stub_get(monkeypatch, [make_response(200, json_body={
            "rules": [department_cap(OPENAI_RULE_ID, "openai")],
            "departmentBudgetBlocks": {EMAIL: OPENAI_RULE_ID},
            "departmentBudgetCandidates": {EMAIL: [
                "not an object",
                {"ruleId": True, "tier": "BLOCK"},
                {"ruleId": OPENAI_RULE_ID, "tier": "SOMETIMES"},
                {"tier": "BLOCK"},
            ]},
        })])
        enforcement._refresh_cache()

        error = refused_by(alice(), model="claude-sonnet-4-5", provider="anthropic")

        assert error.rule_id == OPENAI_RULE_ID

    def test_malformed_items_beside_a_good_one_are_dropped(self, department_cache, fetch_env):
        monkeypatch, _ = fetch_env
        monkeypatch.delenv("REVENIUM_CACHE_DIR", raising=False)
        department_cache([])
        stub_get(monkeypatch, [make_response(200, json_body={
            "rules": [department_cap(OPENAI_RULE_ID, "openai")],
            "departmentBudgetBlocks": {EMAIL: OPENAI_RULE_ID},
            "departmentBudgetCandidates": {EMAIL: [None, candidate(OPENAI_RULE_ID)]},
        })])
        enforcement._refresh_cache()

        assert refused_by(alice(), model="claude-sonnet-4-5", provider="anthropic") is None
        assert refused_by(alice(), model="gpt-4o", provider="openai").rule_id == OPENAI_RULE_ID

    def test_a_provider_scoped_cap_still_blocks_a_call_whose_provider_is_unknown(self, department_cache):
        department_cache(
            [department_cap(OPENAI_RULE_ID, "openai")], {}, {}, {},
            {EMAIL: [candidate(OPENAI_RULE_ID)]},
        )

        assert refused_by(alice()).rule_id == OPENAI_RULE_ID

    def test_an_un_normalized_key_on_the_wire_still_finds_the_caller(self, department_cache, fetch_env):
        monkeypatch, _ = fetch_env
        monkeypatch.delenv("REVENIUM_CACHE_DIR", raising=False)
        department_cache([])
        stub_get(monkeypatch, [make_response(200, json_body={
            "rules": [department_cap(OPENAI_RULE_ID, "openai")],
            "departmentBudgetBlocks": {EMAIL: OPENAI_RULE_ID},
            "departmentBudgetCandidates": {" Alice@Example.com ": [candidate(OPENAI_RULE_ID)]},
        })])
        enforcement._refresh_cache()

        assert refused_by(alice(), model="claude-sonnet-4-5", provider="anthropic") is None

    def test_a_credential_row_is_judged_against_the_call(self, department_cache):
        rule = department_cap(OPENAI_RULE_ID)
        rule["filters"].append({"dimension": "CREDENTIAL", "operator": "IS", "value": "Team-Key", "values": []})
        department_cache([rule], {EMAIL: OPENAI_RULE_ID}, {}, {}, {EMAIL: [candidate(OPENAI_RULE_ID)]})

        def with_credential(name):
            return {"subscriber": {"email": EMAIL, "credential": {"name": name}}}

        assert refused_by(with_credential("team-key"), model="gpt-4o", provider="openai").rule_id == OPENAI_RULE_ID
        assert refused_by(with_credential("other-key"), model="gpt-4o", provider="openai") is None


class TestOrdinaryCapsKeepTheirOwnMatching:

    def test_a_non_department_agent_cap_still_compares_case_as_given(self, department_cache):
        """Folding case is for department candidates only, whose totals are
        counted that way; the per-rule loop keeps the server's own matcher."""
        team_cap = {
            "ruleId": 50,
            "name": "Team agent cap",
            "threshold": 1.0,
            "currentValue": 2.0,
            "windowEnd": WINDOW_STILL_OPEN,
            "action": "BLOCK",
            "breached": True,
            "shadowMode": False,
            "filters": [{"dimension": "AGENT", "operator": "IS", "value": "ClaudeCode"}],
        }
        department_cache([team_cap])

        assert refused_by({"agent": "claudecode"}, model="gpt-4o", provider="openai") is None
        assert refused_by({"agent": "ClaudeCode"}, model="gpt-4o", provider="openai").rule_id == 50


class TestAWarnTierCandidate:

    def test_the_warning_names_the_candidates_rule_and_balance(self, department_cache, caplog):
        department_cache(
            [department_cap(OPENAI_RULE_ID, "openai"),
             department_cap(ANTHROPIC_RULE_ID, "anthropic", name="Claude budget")],
            {EMAIL: OPENAI_RULE_ID}, {EMAIL: 12.0}, {},
            {EMAIL: [candidate(OPENAI_RULE_ID), candidate(ANTHROPIC_RULE_ID, tier="WARN", balance=9.0)]},
        )

        with caplog.at_level(logging.WARNING, logger=MIDDLEWARE_LOGGER):
            error = refused_by(alice(), model="claude-sonnet-4-5", provider="anthropic")

        assert error is None
        assert [record.getMessage() for record in warn_records(caplog)] == [
            "Approaching the Revenium department budget enforced by rule Claude budget "
            "(9.00 of 10.00 spent); calls are blocked once the threshold is crossed",
        ]

    def test_the_warning_is_logged_once_however_many_calls_are_made(self, department_cache, caplog):
        department_cache(
            [department_cap(ANTHROPIC_RULE_ID, "anthropic")],
            {}, {}, {},
            {EMAIL: [candidate(ANTHROPIC_RULE_ID, tier="WARN", balance=9.0)]},
        )

        with caplog.at_level(logging.WARNING, logger=MIDDLEWARE_LOGGER):
            for _ in range(3):
                enforcement.check_enforcement(alice(), model="claude-sonnet-4-5", provider="anthropic")

        assert len(warn_records(caplog)) == 1

    def test_a_warn_candidate_for_another_provider_says_nothing(self, department_cache, caplog):
        """The slot map would warn this person; their only candidate does not cover the call."""
        department_cache(
            [department_cap(ANTHROPIC_RULE_ID, "anthropic")],
            {}, {}, {EMAIL: ANTHROPIC_RULE_ID},
            {EMAIL: [candidate(ANTHROPIC_RULE_ID, tier="WARN", balance=9.0)]},
        )

        with caplog.at_level(logging.WARNING, logger=MIDDLEWARE_LOGGER):
            enforcement.check_enforcement(alice(), model="gpt-4o", provider="openai")

        assert warn_records(caplog) == []


class TestTheCandidatesSurviveTheDiskSnapshot:

    @pytest.fixture
    def enforcing(self, department_snapshot_dir, monkeypatch):
        monkeypatch.setenv("REVENIUM_CIRCUIT_BREAKER_ENABLED", "true")
        monkeypatch.delenv("REVENIUM_BYPASS", raising=False)
        monkeypatch.delenv("REVENIUM_CB_FAIL_MODE", raising=False)
        monkeypatch.setattr(enforcement, "_ensure_poller_running", lambda: None)
        return department_snapshot_dir

    def test_a_restarted_process_still_scopes_the_cap_to_its_provider(self, enforcing, fetch_env):
        """The refresh writes the snapshot; a fresh process reads it back."""
        monkeypatch, _ = fetch_env
        monkeypatch.setenv("REVENIUM_TEAM_ID", "team-snapshot")
        stub_get(monkeypatch, [make_response(200, json_body={
            "rules": [department_cap(OPENAI_RULE_ID, "openai")],
            "departmentBudgetBlocks": {EMAIL: OPENAI_RULE_ID},
            "departmentBudgetBlockBalances": {EMAIL: 12.0},
            "departmentBudgetCandidates": {EMAIL: [candidate(OPENAI_RULE_ID, balance=13.5)]},
        })])
        enforcement._refresh_cache()
        monkeypatch.setattr(enforcement, "_cached_rules", [])
        monkeypatch.setattr(enforcement, "_cached_org_unit_blocks", {})
        monkeypatch.setattr(enforcement, "_cached_org_unit_block_balances", {})
        monkeypatch.setattr(enforcement, "_cached_org_unit_candidates", {})
        monkeypatch.setattr(enforcement, "_cache_initialized", False)
        monkeypatch.setattr(enforcement, "_disk_load_attempted", False)

        assert refused_by(alice(), model="claude-sonnet-4-5", provider="anthropic") is None
        error = refused_by(alice(), model="gpt-4o", provider="openai")
        assert error.rule_id == OPENAI_RULE_ID
        assert error.current_value == 13.5

    def test_a_snapshot_written_before_the_candidates_blocks_as_before(self, enforcing):
        snapshot = enforcement._org_unit_blocks_cache_file_path()
        rules = [department_cap(OPENAI_RULE_ID, "openai")]
        enforcement._persist_cache_to_disk(rules, {EMAIL: OPENAI_RULE_ID})
        with open(snapshot, encoding="utf-8") as handle:
            envelope = json.load(handle)
        envelope.pop("candidates", None)
        with open(snapshot, "w", encoding="utf-8") as handle:
            json.dump(envelope, handle)

        error = refused_by(alice(), model="claude-sonnet-4-5", provider="anthropic")

        assert error.rule_id == OPENAI_RULE_ID
