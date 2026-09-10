"""Department budgets must find the caller, and report the caller's own spend.

Two defects on the same path (BACK-3066), both made visible by the server-side
work in BACK-2978:

1. The server keys every department-budget map by the address normalized the
   way ``EmailNormalizer`` does it -- trimmed and lower-cased -- because that is
   the only identity org-unit attribution can follow. The SDK looked the caller
   up with the address exactly as the application supplied it, so a call
   carrying ``Dept-User@Example.Test`` (or the same address with a stray
   space) missed a block the server had already decided on: the paid provider
   call went out, and the block only surfaced on the *next* call's response.
2. For a per-person cap scoped to one department, the rule's own
   ``currentValue`` is the highest single balance in that department, not the
   caller's. Each named person's own balance rides beside the block map in
   ``orgUnitBudgetBlockBalances``. Populating ``BudgetExceededError`` from the
   rule handed the blocked developer the department's top spender's figure.

The server's own violation consumer reads
``orgUnitBudgetBlockBalances[email] ?: rule.currentValue`` after normalizing
the event's address once; this suite pins the SDK to the same two steps, and to
the unchanged behaviour of a payload that carries no balance map at all.
"""
import json

import pytest

from revenium_middleware._core import enforcement
from revenium_middleware._core.exceptions import BudgetExceededError

from .conftest import make_response, stub_get

DEPARTMENT_NAME = "Engineering monthly budget"
ORG_UNIT_RULE_ID = 7777
# The directory address, in the normalized form the server publishes.
DEPT_EMAIL = "dept-user@example.test"
OTHER_EMAIL = "other-user@example.test"
# What the rule itself reports: the highest single balance in the department.
DEPARTMENT_TOP_BALANCE = 1450.0
# What this caller actually spent.
OWN_BALANCE = 1120.5


def org_unit_rule(**overrides):
    """A per-person cap scoped to one department, as the map references it."""
    rule = {
        "ruleId": ORG_UNIT_RULE_ID,
        "name": DEPARTMENT_NAME,
        "metricType": "TOTAL_COST",
        "threshold": 1000.0,
        "currentValue": DEPARTMENT_TOP_BALANCE,
        "periodType": "MONTHLY",
        "groupBy": "ORG_UNIT",
        "action": "BLOCK",
        "breached": True,
        "shadowMode": False,
    }
    rule.update(overrides)
    return rule


def nested(email):
    return {"subscriber": {"email": email}}


def blocked(load, metadata, blocks=None, balances=None, rules=None):
    """Run the pre-call hook and hand back the error it raised."""
    load(
        rules if rules is not None else [org_unit_rule()],
        blocks if blocks is not None else {DEPT_EMAIL: ORG_UNIT_RULE_ID},
        balances,
    )
    with pytest.raises(BudgetExceededError) as excinfo:
        enforcement.check_enforcement(metadata)
    return excinfo.value


class TestCallerIsFoundUnderTheNormalizedAddress:
    """Card AC 1: case and surrounding whitespace must not lose the block."""

    @pytest.mark.parametrize("supplied", [
        "Dept-User@Example.Test",
        "DEPT-USER@EXAMPLE.TEST",
        "  dept-user@example.test  ",
        "\tdept-user@example.test\n",
        "  Dept-User@Example.Test ",
        DEPT_EMAIL,
    ])
    def test_block_is_found_however_the_address_was_typed(self, department_cache, supplied):
        error = blocked(department_cache, nested(supplied))

        assert error.rule_name == DEPARTMENT_NAME
        assert error.rule_id == ORG_UNIT_RULE_ID

    @pytest.mark.parametrize("supplied", [
        "Dept-User@Example.Test",
        "  dept-user@example.test  ",
    ])
    def test_flat_address_is_normalized_too(self, department_cache, supplied):
        error = blocked(department_cache, {"subscriber_email": supplied})

        assert error.rule_id == ORG_UNIT_RULE_ID

    def test_an_unmapped_colleague_is_still_not_blocked(self, department_cache):
        """Normalizing must widen nothing: only the mapped person is blocked."""
        department_cache([org_unit_rule()], {DEPT_EMAIL: ORG_UNIT_RULE_ID})

        assert enforcement.check_enforcement(nested("Other-User@Example.Test")) is None

    def test_a_whitespace_only_address_is_not_a_key(self, department_cache):
        """It normalizes to nothing, and nothing must never match a map key."""
        department_cache([org_unit_rule()], {"": ORG_UNIT_RULE_ID, "  ": ORG_UNIT_RULE_ID})

        assert enforcement.check_enforcement(nested("   ")) is None

    def test_an_un_normalized_payload_key_is_re_keyed_on_the_way_in(
        self, department_cache, fetch_env
    ):
        """Defensive, for a server that has not normalized its keys yet.

        The published contract says the keys are normalized, so this rewrites
        nothing on a current payload. It exists so that normalizing the
        caller's side cannot *stop* matching a key that used to match byte for
        byte -- which would have traded one missed block for another. Doing it
        at ingestion rather than at lookup keeps the pre-provider path a single
        dict hit for the callers who are not blocked, which is most of them.
        """
        monkeypatch, _ = fetch_env
        monkeypatch.delenv("REVENIUM_CACHE_DIR", raising=False)
        # Installs a monkeypatched (and therefore restored) cache; the refresh
        # below overwrites it with what the stubbed payload carries.
        department_cache([], {}, {})
        stub_get(monkeypatch, [make_response(200, json_body={
            "rules": [org_unit_rule()],
            "orgUnitBudgetBlocks": {" Dept-User@Example.Test ": ORG_UNIT_RULE_ID},
            "orgUnitBudgetBlockBalances": {" Dept-User@Example.Test ": OWN_BALANCE},
        })])

        enforcement._refresh_cache()

        assert enforcement._cached_org_unit_blocks == {DEPT_EMAIL: ORG_UNIT_RULE_ID}
        with pytest.raises(BudgetExceededError) as excinfo:
            enforcement.check_enforcement(nested(DEPT_EMAIL))
        assert excinfo.value.rule_id == ORG_UNIT_RULE_ID
        assert excinfo.value.current_value == OWN_BALANCE


class TestFlatAndNestedPrecedenceUnchanged:
    """Normalization must not turn the flat key into a second identity."""

    def test_a_mixed_case_nested_address_still_suppresses_the_flat_one(self, department_cache):
        department_cache([org_unit_rule()], {DEPT_EMAIL: ORG_UNIT_RULE_ID})
        metadata = nested("Other-User@Example.Test")
        metadata["subscriber_email"] = DEPT_EMAIL  # blocked, but not who the call is for

        assert enforcement.check_enforcement(metadata) is None

    def test_a_whitespace_only_nested_address_still_suppresses_the_flat_one(
        self, department_cache
    ):
        """Precedence is decided by presence, before normalization empties it."""
        department_cache([org_unit_rule()], {DEPT_EMAIL: ORG_UNIT_RULE_ID})
        metadata = nested("   ")
        metadata["subscriber_email"] = DEPT_EMAIL

        assert enforcement.check_enforcement(metadata) is None

    @pytest.mark.parametrize("bad_email", [{"oops": 1}, ["a@b.test"], 42, None])
    def test_a_non_string_nested_address_falls_through_to_the_flat_one(
        self, department_cache, bad_email
    ):
        """Unchanged: only strings are candidates, so a non-string is skipped."""
        metadata = nested(bad_email)
        metadata["subscriber_email"] = "Dept-User@Example.Test"

        assert blocked(department_cache, metadata).rule_id == ORG_UNIT_RULE_ID


class TestErrorCarriesTheCallersOwnBalance:
    """Card AC 2 and AC 3."""

    def test_own_balance_replaces_the_departments_top_spender(self, department_cache):
        error = blocked(
            department_cache, nested(DEPT_EMAIL),
            balances={DEPT_EMAIL: OWN_BALANCE, OTHER_EMAIL: DEPARTMENT_TOP_BALANCE},
        )

        assert error.current_value == OWN_BALANCE
        # The threshold stays rule-level: it is the cap this person was judged
        # against, and it is the same for everyone the rule names.
        assert error.threshold == 1000.0

    @pytest.mark.parametrize("supplied", ["Dept-User@Example.Test", " dept-user@example.test "])
    def test_own_balance_is_found_for_a_differently_typed_address(
        self, department_cache, supplied
    ):
        """One normalization serves both lookups, or the block reports the
        wrong number for exactly the callers defect 1 was about."""
        error = blocked(department_cache, nested(supplied), balances={DEPT_EMAIL: OWN_BALANCE})

        assert error.current_value == OWN_BALANCE

    @pytest.mark.parametrize("wire_value", ["1120.50", "1120.5", 1120.5])
    def test_a_bigdecimal_serialized_either_way_is_reported(self, department_cache, wire_value):
        error = blocked(department_cache, nested(DEPT_EMAIL), balances={DEPT_EMAIL: wire_value})

        assert error.current_value == OWN_BALANCE

    def test_a_payload_without_the_balance_map_behaves_exactly_as_before(self, department_cache):
        """AC 3: an older server reports the rule's own value, as it always did."""
        error = blocked(department_cache, nested(DEPT_EMAIL), balances=None)

        assert error.current_value == DEPARTMENT_TOP_BALANCE

    def test_another_persons_balance_does_not_leak_into_this_error(self, department_cache):
        error = blocked(
            department_cache, nested(DEPT_EMAIL),
            balances={OTHER_EMAIL: 42.0},
        )

        assert error.current_value == DEPARTMENT_TOP_BALANCE

    @pytest.mark.parametrize("balance", ["nope", "", None, True, [1120.5], {"amount": 1120.5}])
    def test_an_unusable_balance_falls_back_to_the_rules_value(self, department_cache, balance):
        error = blocked(department_cache, nested(DEPT_EMAIL), balances={DEPT_EMAIL: balance})

        assert error.current_value == DEPARTMENT_TOP_BALANCE

    @pytest.mark.parametrize("balances", [[], "nope", 7, {7777: 1120.5}])
    def test_a_malformed_balance_map_never_raises_and_falls_back(
        self, department_cache, balances
    ):
        error = blocked(department_cache, nested(DEPT_EMAIL), balances=balances)

        assert error.current_value == DEPARTMENT_TOP_BALANCE

    def test_a_zero_balance_is_reported_rather_than_read_as_absent(self, department_cache):
        """0.0 is falsy; it is still the number the threshold was compared to."""
        error = blocked(department_cache, nested(DEPT_EMAIL), balances={DEPT_EMAIL: 0.0})

        assert error.current_value == 0.0

    def test_own_balance_is_reported_when_the_rule_is_not_in_the_cache(self, department_cache):
        """A stale or racing payload still blocks; the balance is the caller's."""
        error = blocked(
            department_cache, nested("Dept-User@Example.Test"),
            balances={DEPT_EMAIL: OWN_BALANCE}, rules=[],
        )

        assert error.rule_name == "Department budget"
        assert error.current_value == OWN_BALANCE

    def test_no_balance_and_no_rule_reports_nothing_as_before(self, department_cache):
        error = blocked(department_cache, nested(DEPT_EMAIL), rules=[])

        assert error.current_value is None

    def test_a_warn_only_department_rule_still_does_not_block(self, department_cache):
        """The warn tier is out of scope: a balance is not a directive."""
        department_cache(
            [org_unit_rule(action="WARN_ONLY")],
            {DEPT_EMAIL: ORG_UNIT_RULE_ID},
            {DEPT_EMAIL: OWN_BALANCE},
        )

        assert enforcement.check_enforcement(nested("Dept-User@Example.Test")) is None

    def test_a_shadow_mode_department_rule_still_does_not_block(self, department_cache):
        department_cache(
            [org_unit_rule(shadowMode=True)],
            {DEPT_EMAIL: ORG_UNIT_RULE_ID},
            {DEPT_EMAIL: OWN_BALANCE},
        )

        assert enforcement.check_enforcement(nested("Dept-User@Example.Test")) is None


class TestBalanceMapComesOffTheWire:
    def test_the_map_is_parsed_beside_the_block_map(self, fetch_env):
        monkeypatch, _ = fetch_env
        stub_get(monkeypatch, [make_response(200, json_body={
            "rules": [org_unit_rule()],
            "orgUnitBudgetBlocks": {DEPT_EMAIL: ORG_UNIT_RULE_ID},
            "orgUnitBudgetBlockBalances": {DEPT_EMAIL: "1120.50"},
        })])

        fetched = enforcement._fetch_rules()

        assert fetched.org_unit_blocks == {DEPT_EMAIL: ORG_UNIT_RULE_ID}
        # Coerced once at the boundary, so nothing downstream handles a string.
        assert fetched.org_unit_block_balances == {DEPT_EMAIL: OWN_BALANCE}

    def test_a_body_without_the_key_yields_an_empty_map(self, fetch_env):
        monkeypatch, _ = fetch_env
        stub_get(monkeypatch, [make_response(200, json_body={"rules": [], "compiledAt": "x"})])

        assert enforcement._fetch_rules().org_unit_block_balances == {}

    def test_a_legacy_bare_list_body_yields_an_empty_map(self, fetch_env):
        monkeypatch, _ = fetch_env
        stub_get(monkeypatch, [make_response(200, json_body=[org_unit_rule()])])

        assert enforcement._fetch_rules().org_unit_block_balances == {}

    @pytest.mark.parametrize("balances", [None, [], "nope", 7])
    def test_a_malformed_map_is_ignored(self, fetch_env, balances):
        monkeypatch, _ = fetch_env
        stub_get(monkeypatch, [make_response(200, json_body={
            "rules": [], "orgUnitBudgetBlockBalances": balances,
        })])

        assert enforcement._fetch_rules().org_unit_block_balances == {}

    def test_unusable_entries_are_dropped_and_the_rest_kept(self, fetch_env):
        monkeypatch, _ = fetch_env
        stub_get(monkeypatch, [make_response(200, json_body={
            "rules": [],
            "orgUnitBudgetBlockBalances": {
                DEPT_EMAIL: "1120.50",
                OTHER_EMAIL: "not-a-number",
                "bool@example.test": True,
                "null@example.test": None,
            },
        })])

        assert enforcement._fetch_rules().org_unit_block_balances == {DEPT_EMAIL: OWN_BALANCE}

    def test_two_keys_normalizing_to_one_address_keep_the_first_and_do_not_raise(
        self, fetch_env
    ):
        """A collision has no ordering rule to resolve it, and dropping both
        would fail a block the server had already decided open."""
        monkeypatch, _ = fetch_env
        stub_get(monkeypatch, [make_response(200, json_body={
            "rules": [],
            "orgUnitBudgetBlocks": {
                "Dept-User@Example.Test": ORG_UNIT_RULE_ID,
                DEPT_EMAIL: 1,
            },
            "orgUnitBudgetBlockBalances": {
                "Dept-User@Example.Test": OWN_BALANCE,
                DEPT_EMAIL: 1.0,
            },
        })])

        fetched = enforcement._fetch_rules()

        assert fetched.org_unit_blocks == {DEPT_EMAIL: ORG_UNIT_RULE_ID}
        assert fetched.org_unit_block_balances == {DEPT_EMAIL: OWN_BALANCE}

    def test_a_non_string_key_is_dropped_rather_than_normalized(self):
        """JSON object keys are always strings, so this guards the re-keying
        against a map built in process, not against the wire."""
        fetched = enforcement._fetched_from_payload({
            "rules": [],
            "orgUnitBudgetBlocks": {7777: 1, DEPT_EMAIL: ORG_UNIT_RULE_ID},
            "orgUnitBudgetBlockBalances": {7777: 1.0, DEPT_EMAIL: OWN_BALANCE},
        })

        assert fetched.org_unit_blocks == {DEPT_EMAIL: ORG_UNIT_RULE_ID}
        assert fetched.org_unit_block_balances == {DEPT_EMAIL: OWN_BALANCE}

    def test_refresh_caches_the_balances_with_the_rules(self, fetch_env):
        monkeypatch, _ = fetch_env
        monkeypatch.delenv("REVENIUM_CACHE_DIR", raising=False)
        monkeypatch.setattr(enforcement, "_cached_org_unit_block_balances", {})
        monkeypatch.setattr(enforcement, "_cache_initialized", False)
        stub_get(monkeypatch, [make_response(200, json_body={
            "rules": [org_unit_rule()],
            "orgUnitBudgetBlocks": {DEPT_EMAIL: ORG_UNIT_RULE_ID},
            "orgUnitBudgetBlockBalances": {DEPT_EMAIL: OWN_BALANCE},
        })])

        enforcement._refresh_cache()

        assert enforcement._cached_org_unit_block_balances == {DEPT_EMAIL: OWN_BALANCE}

    def test_a_fetch_outage_preserves_the_previous_balances(self, fetch_env):
        monkeypatch, _ = fetch_env
        stub_get(monkeypatch, [make_response(503)] * 20)
        monkeypatch.setattr(enforcement, "_cached_org_unit_block_balances",
                            {DEPT_EMAIL: OWN_BALANCE})

        enforcement._refresh_cache()

        assert enforcement._cached_org_unit_block_balances == {DEPT_EMAIL: OWN_BALANCE}


class TestBalanceMapSurvivesTheDiskSnapshot:
    """The maps ride in one envelope, so a restart must not lose the balances."""

    def test_balances_survive_persist_and_reload(self, department_snapshot_dir):
        rules = [org_unit_rule()]

        enforcement._persist_cache_to_disk(
            rules, {DEPT_EMAIL: ORG_UNIT_RULE_ID}, {DEPT_EMAIL: OWN_BALANCE},
        )
        enforcement._load_cache_from_disk()

        assert enforcement._cached_org_unit_blocks == {DEPT_EMAIL: ORG_UNIT_RULE_ID}
        assert enforcement._cached_org_unit_block_balances == {DEPT_EMAIL: OWN_BALANCE}
        assert enforcement._cache_initialized is True

    def test_a_snapshot_written_without_balances_loads_an_empty_map(
        self, department_snapshot_dir
    ):
        enforcement._persist_cache_to_disk([org_unit_rule()], {DEPT_EMAIL: ORG_UNIT_RULE_ID})
        enforcement._load_cache_from_disk()

        assert enforcement._cached_org_unit_blocks == {DEPT_EMAIL: ORG_UNIT_RULE_ID}
        assert enforcement._cached_org_unit_block_balances == {}

    def test_a_pre_balance_envelope_still_loads_its_blocks(self, department_snapshot_dir):
        """An envelope written by the version before this one has no
        ``balances`` key at all; the blocks in it must still be honoured."""
        rules = [org_unit_rule()]
        enforcement._persist_cache_to_disk(rules, {DEPT_EMAIL: ORG_UNIT_RULE_ID})
        path = department_snapshot_dir / enforcement._ORG_UNIT_BLOCKS_CACHE_FILENAME
        envelope = json.loads(path.read_text(encoding="utf-8"))
        del envelope["balances"]
        path.write_text(json.dumps(envelope), encoding="utf-8")

        enforcement._load_cache_from_disk()

        assert enforcement._cached_org_unit_blocks == {DEPT_EMAIL: ORG_UNIT_RULE_ID}
        assert enforcement._cached_org_unit_block_balances == {}

    def test_a_torn_snapshot_pair_drops_the_balances_with_the_blocks(
        self, department_snapshot_dir
    ):
        """The two maps are one verdict; keeping either against foreign rules
        would report a number computed from a different payload."""
        enforcement._persist_cache_to_disk(
            [org_unit_rule()], {DEPT_EMAIL: ORG_UNIT_RULE_ID}, {DEPT_EMAIL: OWN_BALANCE},
        )
        rules_path = department_snapshot_dir / enforcement._RULES_CACHE_FILENAME
        rules_path.write_text(json.dumps([org_unit_rule(ruleId=1)]), encoding="utf-8")

        enforcement._load_cache_from_disk()

        assert enforcement._cached_org_unit_blocks == {}
        assert enforcement._cached_org_unit_block_balances == {}

    def test_a_malformed_balance_entry_on_disk_is_dropped_not_crashed_on(
        self, department_snapshot_dir
    ):
        rules = [org_unit_rule()]
        enforcement._persist_cache_to_disk(rules, {DEPT_EMAIL: ORG_UNIT_RULE_ID})
        path = department_snapshot_dir / enforcement._ORG_UNIT_BLOCKS_CACHE_FILENAME
        envelope = json.loads(path.read_text(encoding="utf-8"))
        envelope["balances"] = {DEPT_EMAIL: "nope", OTHER_EMAIL: "42"}
        path.write_text(json.dumps(envelope), encoding="utf-8")

        enforcement._load_cache_from_disk()

        assert enforcement._cached_org_unit_block_balances == {OTHER_EMAIL: 42.0}

    def test_un_normalized_keys_in_a_snapshot_are_re_keyed_on_load(
        self, department_snapshot_dir
    ):
        """The snapshot is a second ingestion path, so it re-keys too: an
        envelope written by an older SDK can carry un-normalized keys."""
        rules = [org_unit_rule()]
        enforcement._persist_cache_to_disk(rules, {DEPT_EMAIL: ORG_UNIT_RULE_ID})
        path = department_snapshot_dir / enforcement._ORG_UNIT_BLOCKS_CACHE_FILENAME
        envelope = json.loads(path.read_text(encoding="utf-8"))
        envelope["blocks"] = {" Dept-User@Example.Test ": ORG_UNIT_RULE_ID}
        envelope["balances"] = {"DEPT-USER@EXAMPLE.TEST": OWN_BALANCE}
        path.write_text(json.dumps(envelope), encoding="utf-8")

        enforcement._load_cache_from_disk()

        assert enforcement._cached_org_unit_blocks == {DEPT_EMAIL: ORG_UNIT_RULE_ID}
        assert enforcement._cached_org_unit_block_balances == {DEPT_EMAIL: OWN_BALANCE}

    def test_a_reloaded_snapshot_reports_the_callers_own_balance(
        self, department_snapshot_dir, monkeypatch
    ):
        """End to end across a restart: the caller is found under a differently
        typed address and sees their own number, not the department's top."""
        monkeypatch.setenv("REVENIUM_CIRCUIT_BREAKER_ENABLED", "true")
        monkeypatch.delenv("REVENIUM_BYPASS", raising=False)
        monkeypatch.delenv("REVENIUM_CB_FAIL_MODE", raising=False)
        monkeypatch.setattr(enforcement, "_ensure_poller_running", lambda: None)
        enforcement._persist_cache_to_disk(
            [org_unit_rule()], {DEPT_EMAIL: ORG_UNIT_RULE_ID}, {DEPT_EMAIL: OWN_BALANCE},
        )

        enforcement._load_cache_from_disk()
        # The snapshot loads deliberately stale; pin it fresh so the assertion
        # exercises the evaluation rather than a refresh.
        monkeypatch.setattr(enforcement, "_cache_timestamp", enforcement.time.monotonic())

        with pytest.raises(BudgetExceededError) as excinfo:
            enforcement.check_enforcement(nested("  Dept-User@Example.Test "))

        assert excinfo.value.rule_name == DEPARTMENT_NAME
        assert excinfo.value.current_value == OWN_BALANCE


def test_non_finite_balances_are_dropped_and_fall_back_to_the_rule_value():
    """"NaN" and "Infinity" parse as floats but are not dollar figures; they must
    not reach BudgetExceededError.current_value (Greptile P2 on sdk#107)."""
    from revenium_middleware._core import enforcement

    coerced = enforcement._coerce_balances({
        "nan@example.test": "NaN",
        "inf@example.test": float("inf"),
        "neg@example.test": "-Infinity",
        "ok@example.test": "12.5",
    })
    assert coerced == {"ok@example.test": 12.5}


def test_a_non_finite_balance_seeded_straight_into_the_cache_falls_back(monkeypatch):
    """The cache-side lookup applies the same finiteness rule as ingestion."""
    from revenium_middleware._core import enforcement

    monkeypatch.setattr(
        enforcement, "_cached_org_unit_block_balances",
        {"dev@example.test": float("nan"), "ops@example.test": 3.25},
    )
    nested = {"subscriber": {"email": "dev@example.test"}}
    assert enforcement._caller_block_balance(
        enforcement._cached_org_unit_block_balances, nested) is None
    ops = {"subscriber": {"email": "OPS@example.test"}}
    assert enforcement._caller_block_balance(
        enforcement._cached_org_unit_block_balances, ops) == 3.25
