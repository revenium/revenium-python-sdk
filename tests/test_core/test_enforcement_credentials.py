"""Spending caps read their rules with the key and address metering uses, wherever they were set (BACK-4245).

The cap reader read only ``REVENIUM_METERING_API_KEY`` from the environment.
A team that set its key with ``initialize_metering(api_key=...)`` got usage
records but no caps, and with ``REVENIUM_CB_FAIL_MODE=closed`` every call was
refused for the life of the process.
"""
import logging

import pytest

import revenium_middleware._core as core
from revenium_middleware._core import enforcement, metering
from revenium_middleware._core.exceptions import BudgetExceededError

from .enforcement_stub_server import HEALTHY, EnforcementStub, point_enforcement_at, shut_down, wait_until

IN_CODE_KEY = "hak_in_code_fake"
ENV_KEY = "hak_enforcement_stub"
RULE_NAME = "Team daily cap"


def breached_rule():
    return {
        "ruleId": 4245,
        "name": RULE_NAME,
        "metricType": "TOTAL_COST",
        "threshold": 1.0,
        "currentValue": 2.5,
        "windowType": "DAILY",
        "windowEnd": "2999-01-01T00:00:00Z",
        "action": "BLOCK",
        "breached": True,
        "shadowMode": False,
    }


def refused():
    try:
        enforcement.check_enforcement({}, model="gpt-4o-mini", provider="OPENAI")
    except BudgetExceededError:
        return True
    return False


@pytest.fixture
def stub(monkeypatch):
    stubs = []
    for name in ("_cached_rules", "_cached_org_unit_blocks", "_cached_org_unit_block_balances",
                 "_cached_org_unit_warnings", "_cache_timestamp", "_cache_initialized"):
        monkeypatch.setattr(enforcement, name, getattr(enforcement, name))
    # The metering client a test builds is put back afterwards.
    monkeypatch.setattr(metering, "client", metering.client)
    monkeypatch.setattr(core, "client", core.client)

    def start(rules):
        endpoint = EnforcementStub(HEALTHY, rules)
        stubs.append(endpoint)
        point_enforcement_at(monkeypatch, endpoint)
        return endpoint

    yield start
    for endpoint in stubs:
        shut_down(endpoint)


def key_only_in_code(monkeypatch, base_url=None):
    monkeypatch.delenv("REVENIUM_METERING_API_KEY", raising=False)
    metering.initialize_metering(api_key=IN_CODE_KEY, base_url=base_url)


def test_a_key_set_in_code_fetches_rules_with_that_key_and_a_breached_cap_refuses(stub, monkeypatch):
    endpoint = stub([breached_rule()])
    key_only_in_code(monkeypatch)

    enforcement._refresh_cache()

    assert endpoint.api_keys == [IN_CODE_KEY]
    assert refused()


def test_a_key_in_code_with_fail_mode_closed_lets_calls_through_when_no_cap_exists(stub, monkeypatch):
    stub([])
    monkeypatch.setenv("REVENIUM_CB_FAIL_MODE", "closed")
    key_only_in_code(monkeypatch)

    enforcement._refresh_cache()

    assert not refused()


def test_a_base_url_set_in_code_reaches_the_cap_reader(stub, monkeypatch):
    endpoint = stub([breached_rule()])
    monkeypatch.delenv("REVENIUM_ENFORCEMENT_BASE_URL")
    monkeypatch.delenv("REVENIUM_METERING_BASE_URL", raising=False)
    key_only_in_code(monkeypatch, base_url=endpoint.base_url + "/meter/")

    enforcement._refresh_cache()

    assert endpoint.api_keys == [IN_CODE_KEY]
    assert refused()


def test_no_key_anywhere_warns_that_the_key_is_missing(stub, monkeypatch, caplog):
    endpoint = stub([breached_rule()])
    monkeypatch.delenv("REVENIUM_METERING_API_KEY", raising=False)
    monkeypatch.setattr(metering, "client", None)
    monkeypatch.setattr(enforcement, "_api_key_warned", False, raising=False)
    caplog.set_level(logging.WARNING, logger="revenium_middleware")

    for _poll in range(2):
        monkeypatch.setattr(enforcement, "_refresh_cooldown_until", 0.0)
        enforcement._refresh_cache()

    warnings = [record.getMessage() for record in caplog.records if record.levelno == logging.WARNING]
    assert [w for w in warnings if "no Revenium API key is configured" in w] == [
        "Spending caps are not enforced: no Revenium API key is configured. Set REVENIUM_METERING_API_KEY "
        "or call revenium_middleware.initialize_metering(api_key=...)."
    ]
    assert not [w for w in warnings if "could not be refreshed" in w]
    assert endpoint.hits == 0


def test_no_key_in_fail_closed_mode_warns_that_every_call_is_refused(stub, monkeypatch, caplog):
    stub([breached_rule()])
    monkeypatch.setenv("REVENIUM_CB_FAIL_MODE", "closed")
    monkeypatch.delenv("REVENIUM_METERING_API_KEY", raising=False)
    monkeypatch.setattr(metering, "client", None)
    monkeypatch.setattr(enforcement, "_api_key_warned", False, raising=False)
    caplog.set_level(logging.WARNING, logger="revenium_middleware")

    enforcement._refresh_cache()

    warnings = [record.getMessage() for record in caplog.records if record.levelno == logging.WARNING]
    assert [w for w in warnings if "no Revenium API key is configured" in w] == [
        "Every AI call is refused (REVENIUM_CB_FAIL_MODE=closed): no Revenium API key is configured. "
        "Set REVENIUM_METERING_API_KEY or call revenium_middleware.initialize_metering(api_key=...)."
    ]


def test_a_key_in_the_environment_still_reaches_the_cap_reader(stub, monkeypatch):
    endpoint = stub([breached_rule()])
    monkeypatch.setattr(metering, "client", None)

    enforcement._refresh_cache()

    assert endpoint.api_keys == [ENV_KEY]
    assert refused()


@pytest.mark.parametrize("suffix", ["\n", " "], ids=["newline", "space"])
def test_a_padded_key_in_the_environment_reaches_the_cap_reader_trimmed(stub, monkeypatch, suffix):
    endpoint = stub([breached_rule()])
    monkeypatch.setenv("REVENIUM_METERING_API_KEY", ENV_KEY + suffix)
    monkeypatch.setattr(metering, "client", None)
    monkeypatch.setattr(metering, "_last_failed_key", None)
    monkeypatch.setattr(enforcement, "_FETCH_MAX_ATTEMPTS", 1)

    enforcement._refresh_cache()

    assert endpoint.api_keys == [ENV_KEY]
    assert refused()


def test_a_padded_key_set_in_code_reaches_the_cap_reader_trimmed(stub, monkeypatch):
    endpoint = stub([breached_rule()])
    monkeypatch.delenv("REVENIUM_METERING_API_KEY", raising=False)
    monkeypatch.setattr(enforcement, "_FETCH_MAX_ATTEMPTS", 1)
    metering.initialize_metering(api_key=IN_CODE_KEY + "\n")

    enforcement._refresh_cache()

    assert endpoint.api_keys == [IN_CODE_KEY]
    assert refused()


def read_rules_on_a_first_call():
    enforcement.check_enforcement({}, model="gpt-4o-mini", provider="OPENAI")
    wait_until(lambda: enforcement._cache_initialized)


@pytest.mark.parametrize("drive", ["refresh", "first-call"])
@pytest.mark.parametrize("value", ["true ", "true\r"], ids=["space", "carriage-return"])
def test_a_padded_circuit_breaker_flag_still_fetches_rules_and_refuses(stub, monkeypatch, value, drive):
    endpoint = stub([breached_rule()])
    monkeypatch.setenv("REVENIUM_CIRCUIT_BREAKER_ENABLED", value)

    if drive == "refresh":
        enforcement._refresh_cache()
    else:
        read_rules_on_a_first_call()

    assert endpoint.hits >= 1
    assert refused()
