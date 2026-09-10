"""A developer approaching a department budget must hear about it before the block.

BACK-3077. The server publishes ``orgUnitBudgetWarnings`` beside the block map
-- normalized subscriber email -> the rule whose warn tier that person crossed,
for a per-person cap scoped to one department, and disjoint from
``orgUnitBudgetBlocks`` because a person already blocked is not warned. Until
this suite the SDK read none of it: the first signal a developer got was the
hard block, while Slack, the webhook and the audit trail had all warned
already.

What is pinned here:

1. A warned caller is *not* blocked, and gets one log line per (caller, rule)
   per cache generation -- never one per call, which would bury the signal in
   its own noise, and never a ``BudgetExceededError``, which would turn the
   warn tier into a block.
2. A blocked caller is blocked exactly as before and is not additionally
   warned.
3. A payload with no warnings map, or a malformed one, behaves exactly as it
   did; the map rides in the same department-budget envelope as the other two
   and is bound to the same rules fingerprint on disk.
4. The evaluator is agnostic about ``metricType``: ``AIAlertMetricType`` is
   shared with AI Alerts and grows there (``QUALITY_RATE`` is evaluated from
   job outcome facts, not by enforcement), so a rule carrying a value this SDK
   has never seen is evaluated on ``breached`` / ``threshold`` /
   ``currentValue`` like any other and never raises for the metric itself.
"""
import inspect
import json
import logging

import pytest

from revenium_middleware._core import enforcement
from revenium_middleware._core.exceptions import BudgetExceededError

from .conftest import make_response, stub_get

MIDDLEWARE_LOGGER = "revenium_middleware.extension"

WARN_RULE_ID = 8888
BLOCK_RULE_ID = 7777
WARN_RULE_NAME = "Engineering monthly budget"
# The directory addresses, in the normalized form the server publishes.
WARNED_EMAIL = "warned-user@example.test"
BLOCKED_EMAIL = "blocked-user@example.test"
# What this caller has spent, and the cap they are approaching.
OWN_BALANCE = 812.5
THRESHOLD = 1000.0


def org_unit_rule(**overrides):
    """A per-person cap scoped to one department, as both maps reference it."""
    rule = {
        "ruleId": WARN_RULE_ID,
        "name": WARN_RULE_NAME,
        "metricType": "TOTAL_COST",
        "threshold": THRESHOLD,
        "currentValue": 1450.0,
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


def warn_records(caplog):
    """The warn-tier lines, isolated from any other warning on the path."""
    return [r for r in caplog.records if "Approaching" in r.getMessage()]


class TestAWarnedCallerIsSignalledWithoutBeingBlocked:
    """Card AC 1."""

    def test_a_warned_caller_is_not_blocked(self, department_cache):
        department_cache([org_unit_rule()], {}, {}, {WARNED_EMAIL: WARN_RULE_ID})

        assert enforcement.check_enforcement(nested(WARNED_EMAIL)) is None

    def test_the_warning_is_logged_once_however_many_calls_are_made(
        self, department_cache, caplog
    ):
        """One line per cache generation. A line per call would make the signal
        indistinguishable from the noise it is meant to stand out from."""
        department_cache([org_unit_rule()], {}, {}, {WARNED_EMAIL: WARN_RULE_ID})

        with caplog.at_level(logging.WARNING, logger=MIDDLEWARE_LOGGER):
            for _ in range(5):
                enforcement.check_enforcement(nested(WARNED_EMAIL))

        assert len(warn_records(caplog)) == 1

    def test_the_warning_names_the_rule(self, department_cache, caplog):
        department_cache([org_unit_rule()], {}, {}, {WARNED_EMAIL: WARN_RULE_ID})

        with caplog.at_level(logging.WARNING, logger=MIDDLEWARE_LOGGER):
            enforcement.check_enforcement(nested(WARNED_EMAIL))

        assert WARN_RULE_NAME in caplog.text

    def test_the_warning_carries_the_callers_own_balance_and_the_threshold(
        self, department_cache, caplog
    ):
        """The rule's own ``currentValue`` is the department's top spender, so
        the number the warned person is told is the one from the balance map."""
        department_cache(
            [org_unit_rule()],
            {},
            {WARNED_EMAIL: OWN_BALANCE},
            {WARNED_EMAIL: WARN_RULE_ID},
        )

        with caplog.at_level(logging.WARNING, logger=MIDDLEWARE_LOGGER):
            enforcement.check_enforcement(nested(WARNED_EMAIL))

        assert "812.50" in caplog.text
        assert "1000.00" in caplog.text
        assert "1450" not in caplog.text

    def test_the_warning_does_not_log_the_address(self, department_cache, caplog):
        """The department maps are PII; the existing collision logs omit the
        address for the same reason."""
        department_cache(
            [org_unit_rule()], {}, {WARNED_EMAIL: OWN_BALANCE},
            {WARNED_EMAIL: WARN_RULE_ID},
        )

        with caplog.at_level(logging.WARNING, logger=MIDDLEWARE_LOGGER):
            enforcement.check_enforcement(nested(WARNED_EMAIL))

        assert WARNED_EMAIL not in caplog.text

    def test_a_warning_with_no_published_balance_still_names_the_rule(
        self, department_cache, caplog
    ):
        """An ancestor cap publishes no per-person balance; the signal is still
        worth emitting without a number."""
        department_cache([org_unit_rule()], {}, {}, {WARNED_EMAIL: WARN_RULE_ID})

        with caplog.at_level(logging.WARNING, logger=MIDDLEWARE_LOGGER):
            enforcement.check_enforcement(nested(WARNED_EMAIL))

        assert WARN_RULE_NAME in caplog.text
        assert len(warn_records(caplog)) == 1

    def test_a_rule_the_cache_has_not_seen_is_named_by_its_id(
        self, department_cache, caplog
    ):
        """A stale or racing payload can warn against a rule the rules snapshot
        does not carry; the id is still more use than silence."""
        department_cache([], {}, {}, {WARNED_EMAIL: WARN_RULE_ID})

        with caplog.at_level(logging.WARNING, logger=MIDDLEWARE_LOGGER):
            enforcement.check_enforcement(nested(WARNED_EMAIL))

        assert str(WARN_RULE_ID) in caplog.text

    @pytest.mark.parametrize("supplied", [
        "Warned-User@Example.Test",
        "  warned-user@example.test  ",
        WARNED_EMAIL,
    ])
    def test_the_caller_is_found_under_the_normalized_address(
        self, department_cache, caplog, supplied
    ):
        department_cache([org_unit_rule()], {}, {}, {WARNED_EMAIL: WARN_RULE_ID})

        with caplog.at_level(logging.WARNING, logger=MIDDLEWARE_LOGGER):
            enforcement.check_enforcement(nested(supplied))

        assert len(warn_records(caplog)) == 1

    def test_an_unwarned_colleague_hears_nothing(self, department_cache, caplog):
        department_cache([org_unit_rule()], {}, {}, {WARNED_EMAIL: WARN_RULE_ID})

        with caplog.at_level(logging.WARNING, logger=MIDDLEWARE_LOGGER):
            assert enforcement.check_enforcement(nested(BLOCKED_EMAIL)) is None

        assert warn_records(caplog) == []

    def test_a_caller_with_no_address_hears_nothing(self, department_cache, caplog):
        department_cache([org_unit_rule()], {}, {}, {WARNED_EMAIL: WARN_RULE_ID})

        with caplog.at_level(logging.WARNING, logger=MIDDLEWARE_LOGGER):
            assert enforcement.check_enforcement({}) is None

        assert warn_records(caplog) == []

    def test_each_warned_caller_in_the_process_is_signalled(
        self, department_cache, caplog
    ):
        """The dedupe is per caller, not per process: a gateway meters many
        people, and one warned developer must not silence the next."""
        department_cache([org_unit_rule()], {}, {}, {
            WARNED_EMAIL: WARN_RULE_ID, BLOCKED_EMAIL: WARN_RULE_ID,
        })

        with caplog.at_level(logging.WARNING, logger=MIDDLEWARE_LOGGER):
            enforcement.check_enforcement(nested(WARNED_EMAIL))
            enforcement.check_enforcement(nested(BLOCKED_EMAIL))
            enforcement.check_enforcement(nested(WARNED_EMAIL))

        assert len(warn_records(caplog)) == 2


class TestABlockedCallerIsNotAlsoWarned:
    """Card AC 2: the block is the verdict; a warning beside it is noise."""

    def test_a_caller_in_both_maps_is_blocked(self, department_cache, caplog):
        department_cache(
            [org_unit_rule(ruleId=BLOCK_RULE_ID)],
            {BLOCKED_EMAIL: BLOCK_RULE_ID},
            {BLOCKED_EMAIL: 1450.0},
            {BLOCKED_EMAIL: BLOCK_RULE_ID},
        )

        with caplog.at_level(logging.WARNING, logger=MIDDLEWARE_LOGGER):
            with pytest.raises(BudgetExceededError) as excinfo:
                enforcement.check_enforcement(nested(BLOCKED_EMAIL))

        assert excinfo.value.rule_id == BLOCK_RULE_ID
        assert warn_records(caplog) == []

    def test_a_warning_never_raises_for_a_caller_who_is_only_warned(
        self, department_cache
    ):
        department_cache(
            [org_unit_rule()], {}, {WARNED_EMAIL: OWN_BALANCE},
            {WARNED_EMAIL: WARN_RULE_ID},
        )

        assert enforcement.check_enforcement(nested(WARNED_EMAIL)) is None

    def test_a_shadow_mode_block_still_lets_the_warning_through(
        self, department_cache, caplog
    ):
        """Shadow mode suppresses the block, not the caller's own early signal."""
        department_cache(
            [org_unit_rule(shadowMode=True)],
            {WARNED_EMAIL: WARN_RULE_ID},
            {},
            {WARNED_EMAIL: WARN_RULE_ID},
        )

        with caplog.at_level(logging.WARNING, logger=MIDDLEWARE_LOGGER):
            assert enforcement.check_enforcement(nested(WARNED_EMAIL)) is None

        assert len(warn_records(caplog)) == 1


class TestANewCacheGenerationSignalsAgain:
    """The dedupe is per cache generation, so a refreshed verdict is heard."""

    def test_a_refresh_re_emits_the_warning_once(self, department_cache, fetch_env, caplog):
        monkeypatch, _ = fetch_env
        monkeypatch.delenv("REVENIUM_CACHE_DIR", raising=False)
        department_cache([org_unit_rule()], {}, {}, {WARNED_EMAIL: WARN_RULE_ID})
        stub_get(monkeypatch, [make_response(200, json_body={
            "rules": [org_unit_rule()],
            "orgUnitBudgetWarnings": {WARNED_EMAIL: WARN_RULE_ID},
        })])

        with caplog.at_level(logging.WARNING, logger=MIDDLEWARE_LOGGER):
            enforcement.check_enforcement(nested(WARNED_EMAIL))
            enforcement.check_enforcement(nested(WARNED_EMAIL))
            enforcement._refresh_cache()
            enforcement.check_enforcement(nested(WARNED_EMAIL))
            enforcement.check_enforcement(nested(WARNED_EMAIL))

        assert len(warn_records(caplog)) == 2

    def test_a_refresh_naming_a_different_rule_names_it(
        self, department_cache, fetch_env, caplog
    ):
        monkeypatch, _ = fetch_env
        monkeypatch.delenv("REVENIUM_CACHE_DIR", raising=False)
        department_cache([org_unit_rule()], {}, {}, {WARNED_EMAIL: WARN_RULE_ID})
        stub_get(monkeypatch, [make_response(200, json_body={
            "rules": [org_unit_rule(ruleId=99, name="Platform quarterly cap")],
            "orgUnitBudgetWarnings": {WARNED_EMAIL: 99},
        })])

        with caplog.at_level(logging.WARNING, logger=MIDDLEWARE_LOGGER):
            enforcement.check_enforcement(nested(WARNED_EMAIL))
            enforcement._refresh_cache()
            enforcement.check_enforcement(nested(WARNED_EMAIL))

        assert WARN_RULE_NAME in caplog.text
        assert "Platform quarterly cap" in caplog.text

    def test_a_stale_snapshots_claim_cannot_silence_the_new_generation(
        self, department_cache, fetch_env, caplog
    ):
        """The race Greptile and Tessie both found on the first cut.

        A request can read the cache, be descheduled, and only then take its
        warn claim -- by which time a poll has installed a refreshed verdict
        and cleared the emitted set. With the generation in the key that late
        claim lands under the generation it actually read, so it neither
        silences the refreshed verdict nor escapes its own dedupe.
        """
        monkeypatch, _ = fetch_env
        monkeypatch.delenv("REVENIUM_CACHE_DIR", raising=False)
        department_cache([org_unit_rule()], {}, {}, {WARNED_EMAIL: WARN_RULE_ID})
        stub_get(monkeypatch, [make_response(200, json_body={
            "rules": [org_unit_rule()],
            "orgUnitBudgetWarnings": {WARNED_EMAIL: WARN_RULE_ID},
        })])
        # What the descheduled request read before the refresh.
        stale_rules, stale_departments, _ = enforcement._get_rules()

        with caplog.at_level(logging.WARNING, logger=MIDDLEWARE_LOGGER):
            enforcement._refresh_cache()
            # It resumes and claims now, against the maps it actually read.
            enforcement._check_org_unit_block(
                stale_rules, stale_departments, nested(WARNED_EMAIL))
            enforcement._check_org_unit_block(
                stale_rules, stale_departments, nested(WARNED_EMAIL))
            # A request on the refreshed verdict must still be warned.
            enforcement.check_enforcement(nested(WARNED_EMAIL))
            enforcement.check_enforcement(nested(WARNED_EMAIL))

        # One for the stale generation, one for the refreshed one.
        assert len(warn_records(caplog)) == 2

    def test_the_generation_travels_with_the_snapshot_the_check_reads(
        self, department_cache
    ):
        """Two snapshots either side of a replacement are different verdicts."""
        department_cache([org_unit_rule()], {}, {}, {WARNED_EMAIL: WARN_RULE_ID})
        before = enforcement._get_rules()[1].generation
        department_cache([org_unit_rule()], {}, {}, {WARNED_EMAIL: WARN_RULE_ID})
        after = enforcement._get_rules()[1].generation

        assert after > before

    def test_a_disk_load_emits_the_warning_once(
        self, department_snapshot_dir, monkeypatch, caplog
    ):
        monkeypatch.setenv("REVENIUM_CIRCUIT_BREAKER_ENABLED", "true")
        monkeypatch.delenv("REVENIUM_BYPASS", raising=False)
        monkeypatch.delenv("REVENIUM_CB_FAIL_MODE", raising=False)
        monkeypatch.setattr(enforcement, "_ensure_poller_running", lambda: None)
        enforcement._persist_cache_to_disk(
            [org_unit_rule()], {}, {WARNED_EMAIL: OWN_BALANCE},
            {WARNED_EMAIL: WARN_RULE_ID},
        )

        with caplog.at_level(logging.WARNING, logger=MIDDLEWARE_LOGGER):
            enforcement._load_cache_from_disk()
            # The snapshot loads deliberately stale; pin it fresh so the calls
            # exercise the evaluation rather than a refresh.
            monkeypatch.setattr(enforcement, "_cache_timestamp",
                                enforcement.time.monotonic())
            enforcement.check_enforcement(nested(WARNED_EMAIL))
            enforcement.check_enforcement(nested(WARNED_EMAIL))

        assert len(warn_records(caplog)) == 1
        assert "812.50" in caplog.text


class TestTheWarningsMapComesOffTheWire:
    def test_the_map_is_parsed_beside_the_other_two(self, fetch_env):
        monkeypatch, _ = fetch_env
        stub_get(monkeypatch, [make_response(200, json_body={
            "rules": [org_unit_rule()],
            "orgUnitBudgetBlocks": {BLOCKED_EMAIL: BLOCK_RULE_ID},
            "orgUnitBudgetBlockBalances": {BLOCKED_EMAIL: 1450.0},
            "orgUnitBudgetWarnings": {WARNED_EMAIL: WARN_RULE_ID},
        })])

        fetched = enforcement._fetch_rules()

        assert fetched.org_unit_blocks == {BLOCKED_EMAIL: BLOCK_RULE_ID}
        assert fetched.org_unit_warnings == {WARNED_EMAIL: WARN_RULE_ID}

    def test_an_un_normalized_key_is_re_keyed_on_the_way_in(self):
        fetched = enforcement._fetched_from_payload({
            "rules": [],
            "orgUnitBudgetWarnings": {" Warned-User@Example.Test ": WARN_RULE_ID},
        })

        assert fetched.org_unit_warnings == {WARNED_EMAIL: WARN_RULE_ID}

    def test_a_body_without_the_key_yields_an_empty_map(self, fetch_env):
        monkeypatch, _ = fetch_env
        stub_get(monkeypatch, [make_response(200, json_body={"rules": [], "compiledAt": "x"})])

        assert enforcement._fetch_rules().org_unit_warnings == {}

    def test_a_legacy_bare_list_body_yields_an_empty_map(self, fetch_env):
        monkeypatch, _ = fetch_env
        stub_get(monkeypatch, [make_response(200, json_body=[org_unit_rule()])])

        assert enforcement._fetch_rules().org_unit_warnings == {}

    def test_a_204_yields_an_empty_map(self, fetch_env):
        monkeypatch, _ = fetch_env
        stub_get(monkeypatch, [make_response(204)])

        assert enforcement._fetch_rules().org_unit_warnings == {}

    @pytest.mark.parametrize("warnings", [None, [], "nope", 7])
    def test_a_malformed_map_is_ignored(self, fetch_env, warnings):
        monkeypatch, _ = fetch_env
        stub_get(monkeypatch, [make_response(200, json_body={
            "rules": [], "orgUnitBudgetWarnings": warnings,
        })])

        assert enforcement._fetch_rules().org_unit_warnings == {}

    def test_a_non_string_key_is_dropped_rather_than_normalized(self):
        fetched = enforcement._fetched_from_payload({
            "rules": [],
            "orgUnitBudgetWarnings": {8888: 1, WARNED_EMAIL: WARN_RULE_ID},
        })

        assert fetched.org_unit_warnings == {WARNED_EMAIL: WARN_RULE_ID}

    @pytest.mark.parametrize("rule_id", ["8888", None, True, 12.5, [8888]])
    def test_a_value_that_is_not_a_rule_id_warns_nobody(
        self, department_cache, caplog, rule_id
    ):
        """Same rule as the block map: the map is documented as email -> int,
        so anything else is absent rather than a verdict."""
        department_cache([org_unit_rule()], {}, {}, {WARNED_EMAIL: rule_id})

        with caplog.at_level(logging.WARNING, logger=MIDDLEWARE_LOGGER):
            assert enforcement.check_enforcement(nested(WARNED_EMAIL)) is None

        assert warn_records(caplog) == []

    @pytest.mark.parametrize("warnings", [
        [], "nope", 7, 0.0, {8888: WARN_RULE_ID}, {WARNED_EMAIL: "notint"},
    ])
    def test_a_malformed_cached_map_never_takes_a_request_down(
        self, department_cache, caplog, warnings
    ):
        """The department check stays fail-open, warn tier included.

        The seeded value is asserted to have reached the cache unchanged: a
        fixture that coerced a falsy map to ``{}`` would leave the non-dict
        guards in ``_snapshot_department_budgets`` and ``_org_unit_map_entry``
        untested for exactly the values that break them.
        """
        department_cache([org_unit_rule()], {}, {}, warnings)
        assert enforcement._cached_org_unit_warnings == warnings

        with caplog.at_level(logging.WARNING, logger=MIDDLEWARE_LOGGER):
            assert enforcement.check_enforcement(nested(WARNED_EMAIL)) is None

        assert warn_records(caplog) == []

    def test_a_refresh_caches_the_warnings_with_the_rules(self, fetch_env):
        monkeypatch, _ = fetch_env
        monkeypatch.delenv("REVENIUM_CACHE_DIR", raising=False)
        monkeypatch.setattr(enforcement, "_cached_org_unit_warnings", {})
        monkeypatch.setattr(enforcement, "_cache_initialized", False)
        stub_get(monkeypatch, [make_response(200, json_body={
            "rules": [org_unit_rule()],
            "orgUnitBudgetWarnings": {WARNED_EMAIL: WARN_RULE_ID},
        })])

        enforcement._refresh_cache()

        assert enforcement._cached_org_unit_warnings == {WARNED_EMAIL: WARN_RULE_ID}

    def test_a_fetch_outage_preserves_the_previous_warnings(self, fetch_env):
        monkeypatch, _ = fetch_env
        stub_get(monkeypatch, [make_response(503)] * 20)
        monkeypatch.setattr(enforcement, "_cached_org_unit_warnings",
                            {WARNED_EMAIL: WARN_RULE_ID})

        enforcement._refresh_cache()

        assert enforcement._cached_org_unit_warnings == {WARNED_EMAIL: WARN_RULE_ID}


class TestTheWarningsMapSurvivesTheDiskSnapshot:
    """Card AC 3: one envelope, one fingerprint, all three maps."""

    def test_warnings_survive_persist_and_reload(self, department_snapshot_dir):
        enforcement._persist_cache_to_disk(
            [org_unit_rule()], {BLOCKED_EMAIL: BLOCK_RULE_ID},
            {BLOCKED_EMAIL: 1450.0}, {WARNED_EMAIL: WARN_RULE_ID},
        )
        enforcement._load_cache_from_disk()

        assert enforcement._cached_org_unit_blocks == {BLOCKED_EMAIL: BLOCK_RULE_ID}
        assert enforcement._cached_org_unit_warnings == {WARNED_EMAIL: WARN_RULE_ID}
        assert enforcement._cache_initialized is True

    def test_a_snapshot_written_without_warnings_loads_an_empty_map(
        self, department_snapshot_dir
    ):
        enforcement._persist_cache_to_disk([org_unit_rule()], {BLOCKED_EMAIL: BLOCK_RULE_ID})
        enforcement._load_cache_from_disk()

        assert enforcement._cached_org_unit_blocks == {BLOCKED_EMAIL: BLOCK_RULE_ID}
        assert enforcement._cached_org_unit_warnings == {}

    def test_a_pre_warnings_envelope_still_loads_its_blocks(self, department_snapshot_dir):
        """An envelope written by the version before this one has no
        ``warnings`` key at all; the blocks in it must still be honoured."""
        enforcement._persist_cache_to_disk(
            [org_unit_rule()], {BLOCKED_EMAIL: BLOCK_RULE_ID}, {BLOCKED_EMAIL: 1450.0},
        )
        path = department_snapshot_dir / enforcement._ORG_UNIT_BLOCKS_CACHE_FILENAME
        envelope = json.loads(path.read_text(encoding="utf-8"))
        envelope.pop("warnings", None)
        path.write_text(json.dumps(envelope), encoding="utf-8")

        enforcement._load_cache_from_disk()

        assert enforcement._cached_org_unit_blocks == {BLOCKED_EMAIL: BLOCK_RULE_ID}
        assert enforcement._cached_org_unit_warnings == {}

    def test_a_torn_snapshot_pair_drops_the_warnings_with_the_blocks(
        self, department_snapshot_dir
    ):
        """The maps are one verdict, and a rule id read against foreign rules
        would name somebody else's cap."""
        enforcement._persist_cache_to_disk(
            [org_unit_rule()], {}, {}, {WARNED_EMAIL: WARN_RULE_ID},
        )
        rules_path = department_snapshot_dir / enforcement._RULES_CACHE_FILENAME
        rules_path.write_text(json.dumps([org_unit_rule(ruleId=1)]), encoding="utf-8")

        enforcement._load_cache_from_disk()

        assert enforcement._cached_org_unit_warnings == {}

    def test_un_normalized_keys_in_a_snapshot_are_re_keyed_on_load(
        self, department_snapshot_dir
    ):
        enforcement._persist_cache_to_disk([org_unit_rule()], {}, {}, {})
        path = department_snapshot_dir / enforcement._ORG_UNIT_BLOCKS_CACHE_FILENAME
        envelope = json.loads(path.read_text(encoding="utf-8"))
        envelope["warnings"] = {" Warned-User@Example.Test ": WARN_RULE_ID}
        path.write_text(json.dumps(envelope), encoding="utf-8")

        enforcement._load_cache_from_disk()

        assert enforcement._cached_org_unit_warnings == {WARNED_EMAIL: WARN_RULE_ID}

    def test_a_malformed_warnings_map_on_disk_is_dropped_not_crashed_on(
        self, department_snapshot_dir
    ):
        enforcement._persist_cache_to_disk([org_unit_rule()], {BLOCKED_EMAIL: BLOCK_RULE_ID})
        path = department_snapshot_dir / enforcement._ORG_UNIT_BLOCKS_CACHE_FILENAME
        envelope = json.loads(path.read_text(encoding="utf-8"))
        envelope["warnings"] = "nope"
        path.write_text(json.dumps(envelope), encoding="utf-8")

        enforcement._load_cache_from_disk()

        assert enforcement._cached_org_unit_warnings == {}
        assert enforcement._cached_org_unit_blocks == {BLOCKED_EMAIL: BLOCK_RULE_ID}


class TestAnUnfamiliarMetricTypeIsEvaluatedLikeAnyOtherRule:
    """Card AC 4.

    ``CompiledEnforcementRule.metricType`` is ``AIAlertMetricType``, an enum
    shared with AI Alerts, so it grows without the SDK. ``QUALITY_RATE`` is
    evaluated from job outcome facts, not by the enforcement path, and reaches
    the SDK only as a value on a rule.
    """

    @staticmethod
    def quality_rate_rule(**overrides):
        """A pooled rule carrying a metric the SDK has never seen."""
        rule = {
            "ruleId": 5150,
            "name": "Quality floor",
            "metricType": "QUALITY_RATE",
            "threshold": 0.95,
            "currentValue": 0.42,
            "periodType": "MONTHLY",
            "action": "BLOCK",
            "breached": True,
            "shadowMode": False,
        }
        rule.update(overrides)
        return rule

    def test_a_breached_rule_blocks_on_breached_threshold_and_current_value(
        self, department_cache
    ):
        department_cache([self.quality_rate_rule()])

        with pytest.raises(BudgetExceededError) as excinfo:
            enforcement.check_enforcement(nested(WARNED_EMAIL))

        assert excinfo.value.rule_name == "Quality floor"
        assert excinfo.value.current_value == 0.42
        assert excinfo.value.threshold == 0.95

    def test_an_unbreached_rule_does_not_block(self, department_cache):
        department_cache([self.quality_rate_rule(breached=False)])

        assert enforcement.check_enforcement(nested(WARNED_EMAIL)) is None

    def test_the_unfamiliar_metric_itself_never_raises(self, department_cache):
        """Anything but ``BudgetExceededError`` out of the pre-call hook is the
        SDK breaking the caller's request over a value it does not need."""
        department_cache([self.quality_rate_rule(breached=False, action="WARN_ONLY")])

        assert enforcement.check_enforcement(nested(WARNED_EMAIL)) is None

    def test_behaviour_is_identical_with_the_field_removed_entirely(
        self, department_cache
    ):
        """The evaluator's verdict must not move when the field goes away --
        the field is not part of the decision."""
        with_metric = self.quality_rate_rule()
        without_metric = {k: v for k, v in with_metric.items() if k != "metricType"}

        department_cache([with_metric])
        with pytest.raises(BudgetExceededError) as first:
            enforcement.check_enforcement(nested(WARNED_EMAIL))

        department_cache([without_metric])
        with pytest.raises(BudgetExceededError) as second:
            enforcement.check_enforcement(nested(WARNED_EMAIL))

        assert (first.value.rule_name, first.value.current_value, first.value.threshold) == (
            second.value.rule_name, second.value.current_value, second.value.threshold
        )

    @pytest.mark.parametrize("metric", [
        "QUALITY_RATE", "SOMETHING_SHIPPED_NEXT_QUARTER", "", None, 42, ["TOTAL_COST"],
    ])
    def test_any_metric_value_at_all_is_evaluated_the_same(self, department_cache, metric):
        department_cache([self.quality_rate_rule(metricType=metric)])

        with pytest.raises(BudgetExceededError) as excinfo:
            enforcement.check_enforcement(nested(WARNED_EMAIL))

        assert excinfo.value.rule_id == 5150

    def test_a_department_rule_with_an_unfamiliar_metric_blocks_from_the_map(
        self, department_cache
    ):
        """The department path reads the map, not the metric, either."""
        department_cache(
            [org_unit_rule(metricType="QUALITY_RATE")],
            {BLOCKED_EMAIL: WARN_RULE_ID},
        )

        with pytest.raises(BudgetExceededError) as excinfo:
            enforcement.check_enforcement(nested(BLOCKED_EMAIL))

        assert excinfo.value.rule_id == WARN_RULE_ID

    def test_the_evaluator_never_reads_the_field(self):
        """A closed client-side enum would turn every metric the platform adds
        into a hard failure, so the field is not read at all. Pinned on reads
        of the field, not on the string, so a comment naming it is still free.
        """
        source = inspect.getsource(enforcement)
        reads = (
            'get("metricType")', "get('metricType')",
            '["metricType"]', "['metricType']", ".metricType",
        )

        assert [read for read in reads if read in source] == []
