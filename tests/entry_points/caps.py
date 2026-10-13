"""Breached rules seeded straight into the enforcement cache, and a record of every cap check made.

Nothing is fetched: the circuit breaker is switched on, the poller and the
disk snapshot are stubbed out, and the given rules become a fresh cache.
"""
from revenium_middleware._core import enforcement


def breached_rule(rule_id, *filters, action="BLOCK", shadow=False):
    rule = {
        "ruleId": rule_id,
        "name": "Spending cap",
        "metricType": "TOTAL_COST",
        "threshold": 1.0,
        "currentValue": 2.5,
        "windowType": "DAILY",
        "windowEnd": "2999-01-01T00:00:00Z",
        "action": action,
        "breached": True,
        "shadowMode": shadow,
    }
    if filters:
        rule["filters"] = list(filters)
    return rule


def scoped(dimension, operator, value):
    return {"dimension": dimension, "operator": operator, "value": value}


def seed_rules(monkeypatch, *rules):
    monkeypatch.setenv("REVENIUM_CIRCUIT_BREAKER_ENABLED", "true")
    monkeypatch.delenv("REVENIUM_BYPASS", raising=False)
    monkeypatch.delenv("REVENIUM_CB_FAIL_MODE", raising=False)
    monkeypatch.setattr(enforcement, "_load_cache_from_disk", lambda: None)
    monkeypatch.setattr(enforcement, "_ensure_poller_running", lambda: None)
    monkeypatch.setattr(enforcement, "_fetch_rules", lambda: None)
    monkeypatch.setattr(enforcement, "_cached_rules", list(rules))
    monkeypatch.setattr(enforcement, "_cached_org_unit_blocks", {})
    monkeypatch.setattr(enforcement, "_cached_org_unit_block_balances", {})
    monkeypatch.setattr(enforcement, "_cached_org_unit_warnings", {})
    monkeypatch.setattr(enforcement, "_cache_timestamp", float("inf"))
    monkeypatch.setattr(enforcement, "_cache_initialized", True)
    enforcement._begin_cache_generation()
    enforcement._warned_non_blocking_rules.clear()
    enforcement._noted_unevaluable_filters.clear()


class CapCheck:
    def __init__(self, usage_metadata, model, provider):
        self.usage_metadata = dict(usage_metadata or {})
        self.model = model
        self.provider = provider

    def __repr__(self):
        return f"CapCheck(model={self.model!r}, provider={self.provider!r}, usage_metadata={self.usage_metadata!r})"


def record_cap_checks(monkeypatch):
    """Every check that judged the call against the rules, with the labels it was given."""
    checks = []
    scope = enforcement._CallScope

    class RecordedScope(scope):
        def __init__(self, usage_metadata, model, provider):
            checks.append(CapCheck(usage_metadata, model, provider))
            super().__init__(usage_metadata, model, provider)

    monkeypatch.setattr(enforcement, "_CallScope", RecordedScope)
    return checks
