"""A worker that switches teams stops applying the previous team's caps at once (BACK-4250).

The cap cache carried no owner. After a switch from team A to team B in one
process (``REVENIUM_TEAM_ID``, the enforcement address, ``initialize_metering()``),
team B's calls were judged by team A's rules and department email list until a
fetch for team B succeeded, and forever when it failed; team B's saved snapshot
was never read. The cache is owned by (enforcement address, team id), the scope
the saved snapshot already uses, so a key-only switch keeps the team's rules.
"""
import logging
import threading

import pytest

import revenium_middleware
import revenium_middleware._core as core
from revenium_middleware._core import enforcement, metering
from revenium_middleware._core.exceptions import BudgetExceededError

from .enforcement_stub_server import (
    FORBIDDEN, HANG, HEALTHY, REFUSE, SLOW, EnforcementStub, point_enforcement_at, shut_down, wait_until,
)

TEAM_A, TEAM_B = "team-a-hashed", "team-b-hashed"
ALICE = {"subscriber": {"email": "alice@team-a.example"}}
BOB = {"subscriber": {"email": "bob@team-b.example"}}


def breached_rule(rule_id, name):
    return {
        "ruleId": rule_id,
        "name": name,
        "metricType": "TOTAL_COST",
        "threshold": 1.0,
        "currentValue": 2.5,
        "windowType": "DAILY",
        "windowEnd": "2999-01-01T00:00:00Z",
        "action": "BLOCK",
        "breached": True,
        "shadowMode": False,
    }


TEAM_A_RULE = breached_rule(4250, "Team A daily cap")
TEAM_B_RULE = breached_rule(4251, "Team B daily cap")
TEAM_A_DEPARTMENTS = {
    "departmentBudgetBlocks": {"alice@team-a.example": 4250},
    "departmentBudgetBlockBalances": {"alice@team-a.example": 9.5},
    "departmentBudgetWarnings": {"alice@team-a.example": 4250},
}


def refusal(metadata):
    """The refusal ``check_enforcement`` raises for ``metadata``, or None when the call goes through."""
    try:
        enforcement.check_enforcement(metadata, model="gpt-4o-mini", provider="OPENAI")
    except BudgetExceededError as error:
        return error
    return None


def team_a_is_in_memory():
    return (any(rule.get("name") == TEAM_A_RULE["name"] for rule in enforcement._cached_rules)
            or any("alice@team-a.example" in cached for cached in (
                enforcement._cached_org_unit_blocks, enforcement._cached_org_unit_block_balances,
                enforcement._cached_org_unit_warnings)))


@pytest.fixture
def teams(monkeypatch):
    """Team A's rules are loaded; ``teams.switch(stub_b, key)`` moves the process to team B.

    No poller runs: each test fetches with ``_refresh_cache()`` itself, so what
    the cache holds never depends on when a background fetch lands.
    """
    stubs = []
    monkeypatch.setattr(enforcement, "_ensure_poller_running", lambda: None)
    for name in ("_cached_rules", "_cached_org_unit_blocks", "_cached_org_unit_block_balances",
                 "_cached_org_unit_warnings", "_cache_timestamp", "_cache_initialized"):
        monkeypatch.setattr(enforcement, name, getattr(enforcement, name))
    monkeypatch.setattr(metering, "client", metering.client)
    monkeypatch.setattr(core, "client", core.client)

    def stub(mode, rules=None, departments=None):
        endpoint = EnforcementStub(mode, rules, departments)
        stubs.append(endpoint)
        return endpoint

    def load_team_a(mode=HEALTHY):
        endpoint = stub(mode, [TEAM_A_RULE], TEAM_A_DEPARTMENTS)
        point_enforcement_at(monkeypatch, endpoint)
        monkeypatch.setenv("REVENIUM_TEAM_ID", TEAM_A)
        if mode == HEALTHY:
            enforcement._refresh_cache()
            assert refusal(BOB).rule_name == TEAM_A_RULE["name"]
        return endpoint

    def switch(endpoint_b, api_key="hak_team_b_fake"):
        monkeypatch.setenv("REVENIUM_TEAM_ID", TEAM_B)
        monkeypatch.setenv("REVENIUM_ENFORCEMENT_BASE_URL", endpoint_b.base_url)
        metering.initialize_metering(api_key=api_key)

    yield type("Teams", (), {"stub": staticmethod(stub), "load_team_a": staticmethod(load_team_a),
                             "switch": staticmethod(switch)})
    for endpoint in stubs:
        shut_down(endpoint)


@pytest.mark.parametrize("fail_mode", ["open", "closed"])
def test_after_a_team_switch_the_next_call_is_not_refused_by_the_old_teams_rule(teams, monkeypatch, fail_mode):
    teams.load_team_a()
    monkeypatch.setenv("REVENIUM_CB_FAIL_MODE", fail_mode)
    team_b = teams.stub(HEALTHY, [])

    teams.switch(team_b)
    refused = refusal(BOB)

    if fail_mode == "open":
        assert refused is None
    else:
        # Strict mode refuses until team B's rules load, never with a team A rule.
        assert refused is None or (refused.rule_name is None and "uninitialized" in str(refused))
    assert not team_a_is_in_memory()
    enforcement._refresh_cache()
    assert refusal(BOB) is None


def test_when_team_bs_rules_are_refused_team_a_rules_are_never_applied(teams, caplog):
    teams.load_team_a()
    team_b = teams.stub(FORBIDDEN)
    caplog.set_level(logging.WARNING, logger="revenium_middleware")

    teams.switch(team_b)
    first = refusal(BOB)
    enforcement._refresh_cache()

    assert (first, refusal(BOB), refusal(ALICE)) == (None, None, None)
    assert not team_a_is_in_memory()
    assert [record for record in caplog.records
            if "could not be refreshed" in record.getMessage() and TEAM_B in record.getMessage()]


def test_team_bs_saved_snapshot_is_loaded_after_the_switch(teams, monkeypatch, tmp_path):
    team_b = teams.stub(REFUSE)
    monkeypatch.setenv("REVENIUM_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("REVENIUM_TEAM_ID", TEAM_B)
    monkeypatch.setenv("REVENIUM_ENFORCEMENT_BASE_URL", team_b.base_url)
    enforcement._persist_cache_to_disk([TEAM_B_RULE])
    teams.load_team_a()
    monkeypatch.setenv("REVENIUM_CACHE_DIR", str(tmp_path))

    teams.switch(team_b)

    refused = refusal(BOB)
    assert refused is not None and refused.rule_name == TEAM_B_RULE["name"]


def test_a_fetch_for_team_a_in_flight_at_the_switch_is_not_installed(teams):
    team_a = teams.load_team_a(mode=SLOW)
    assert refusal(BOB) is None
    fetch_for_team_a = threading.Thread(target=enforcement._refresh_cache)
    fetch_for_team_a.start()
    assert wait_until(lambda: team_a.hits == 1)
    team_b = teams.stub(HEALTHY, [])

    teams.switch(team_b)
    assert refusal(BOB) is None
    team_a.release.set()
    fetch_for_team_a.join(5)

    assert not team_a_is_in_memory()
    assert refusal(BOB) is None


def test_a_fetch_for_team_a_failing_after_the_switch_does_not_hold_off_team_bs_rules(teams, monkeypatch):
    monkeypatch.setattr(enforcement, "_FETCH_MAX_ATTEMPTS", 1)
    team_a = teams.load_team_a(mode=HANG)
    fetch_for_team_a = threading.Thread(target=enforcement._refresh_cache)
    fetch_for_team_a.start()
    assert wait_until(lambda: team_a.hits == 1)
    team_b = teams.stub(HEALTHY, [TEAM_B_RULE])

    teams.switch(team_b)
    assert refusal(BOB) is None
    team_a.release.set()
    fetch_for_team_a.join(5)
    enforcement._refresh_cache()

    refused = refusal(BOB)
    assert refused is not None and refused.rule_name == TEAM_B_RULE["name"]
    assert team_b.hits == 1


def test_reinitialising_with_the_same_credentials_keeps_the_cache(teams):
    team_a = teams.load_team_a()

    metering.initialize_metering(api_key=metering.process_client().api_key)

    refused = refusal(BOB)
    assert refused is not None and refused.rule_name == TEAM_A_RULE["name"]
    assert team_a.hits == 1


def test_switching_only_the_key_keeps_the_teams_rules(teams):
    team_a = teams.load_team_a()

    metering.initialize_metering(api_key="hak_rotated_key_fake")

    refused = refusal(BOB)
    assert refused is not None and refused.rule_name == TEAM_A_RULE["name"]
    assert team_a.hits == 1


def test_a_metering_key_scope_does_not_reset_the_cap_cache(teams):
    team_a = teams.load_team_a()

    with revenium_middleware.metering_key("hak_another_customer_fake"):
        refused = refusal(BOB)

    assert refused is not None and refused.rule_name == TEAM_A_RULE["name"]
    assert team_a.hits == 1


def load_caps_and_refuse_bob():
    enforcement._refresh_cache()
    return refusal(BOB)


def fetch_team_a_rule():
    return enforcement.fetch_enforcement_rule(str(TEAM_A_RULE["ruleId"]))


@pytest.mark.parametrize("first_read", [load_caps_and_refuse_bob, fetch_team_a_rule],
                         ids=["cap-cache-load", "fetch-enforcement-rule"])
def test_caps_first_read_inside_a_metering_key_block_use_the_process_key(teams, monkeypatch, first_read):
    """BACK-4311: caps follow the process key, never a metering_key() block's."""
    stub = teams.stub(HEALTHY, [TEAM_A_RULE], TEAM_A_DEPARTMENTS)
    point_enforcement_at(monkeypatch, stub)
    monkeypatch.setenv("REVENIUM_TEAM_ID", TEAM_A)
    metering.initialize_metering(api_key="hak_process_owner_fake")

    with revenium_middleware.metering_key("hak_another_customer_fake"):
        read = first_read()

    assert read is not None
    assert stub.api_keys == ["hak_process_owner_fake"]
