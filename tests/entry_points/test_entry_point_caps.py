"""A breached spending cap stops every metered entry point before the provider is contacted (BACK-4190).

Rows come from manifest.yaml, as in the execution matrix. Each executable row
runs against its stubbed provider with one breached rule seeded in the
enforcement cache and the circuit breaker on. A BLOCK cap must refuse the call
with ``BudgetExceededError`` before any request reaches the stub and before any
metering payload is recorded; a WARN_ONLY cap must let it through with one
warning; a shadow cap must let it through silently. The scoped caps pin the
labels each path hands the check: the provider label and the model the row's
metering record carries.
"""
import logging
import pathlib

import pytest
import yaml

from entry_points import calls
from entry_points.caps import breached_rule, scoped, seed_rules
from entry_points.instrumented import import_all_middleware, selected_environment_name
from entry_points.stub_requests import count_provider_requests

import_all_middleware()

from revenium_middleware._core import enforcement  # noqa: E402
from revenium_middleware._core.exceptions import BudgetExceededError  # noqa: E402

MANIFEST = yaml.safe_load((pathlib.Path(__file__).parent / "manifest.yaml").read_text())
PROVIDERS = MANIFEST["providers"]
ENVIRONMENT = selected_environment_name()
ROWS = [row for row in MANIFEST["rows"]
        if row["status"] in ("metered", "gap") and row.get("env", "default") == ENVIRONMENT]
WARNING_TEXT = "is over its threshold"
OTHER_MODEL = "some-other-model"


def _call_id(row):
    return row.get("call", row["id"])


def _model(row):
    return row.get("model", PROVIDERS[row["provider"]]["model"])


def _provider_label(row):
    return PROVIDERS[row["provider"]]["expected_provider"]


def _param(row):
    marks = []
    if row.get("isolate"):
        marks.append(pytest.mark.skip(reason=(
            f"runs in a subprocess importing only {', '.join(row['isolate'])}, which the seeded rule cache "
            "cannot reach; tests/test_core/test_enforcement_provider_paths.py checks this path instead")))
    return pytest.param(row, id=row["id"], marks=marks)


ROW_PARAMS = [_param(row) for row in ROWS]

# litellm.batch_completion runs each item on its own thread and returns an
# item's exception in the result list instead of raising it, so the refusal
# cannot surface to the caller here.
ITEM_ERRORS_RETURNED = {"litellm.batch_completion.sync"}


def _refusal_param(row):
    if row["id"] not in ITEM_ERRORS_RETURNED:
        return _param(row)
    return pytest.param(row, id=row["id"], marks=[pytest.mark.skip(
        reason="litellm.batch_completion returns each item's BudgetExceededError in its result list")])


REFUSAL_PARAMS = [_refusal_param(row) for row in ROWS]


def _first_row_per_provider():
    seen = {}
    for row in ROWS:
        if not row.get("isolate"):
            seen.setdefault(row["provider"], row)
    return [pytest.param(row, id=row["id"]) for row in seen.values()]


def _rule(row, *filters, action="BLOCK", shadow=False):
    return breached_rule(f"caps-{row['id']}", *filters, action=action, shadow=shadow)


@pytest.fixture
def seed(monkeypatch):
    """The circuit breaker on and ``rules`` installed as a fresh cache, nothing live."""
    monkeypatch.delenv("REVENIUM_SELECTIVE_METERING", raising=False)
    yield lambda *rules: seed_rules(monkeypatch, *rules)
    enforcement._warned_non_blocking_rules.clear()
    enforcement._noted_unevaluable_filters.clear()


@pytest.fixture
def provider_requests(monkeypatch):
    return count_provider_requests(monkeypatch)


def _run(row, recording_client):
    error = None
    try:
        calls.run_call(_call_id(row))
    except Exception as exc:  # noqa: BLE001
        error = exc
    calls.wait_for_metering()
    return error, calls.recorded_payloads(recording_client)


def _assert_refused(row, recording_client, provider_requests):
    error, payloads = _run(row, recording_client)
    assert isinstance(error, BudgetExceededError), f"expected BudgetExceededError, got {error!r}"
    assert sum(provider_requests.values()) == 0, f"the provider stub was contacted: {dict(provider_requests)}"
    assert payloads == []


def _assert_let_through(row, recording_client):
    error, payloads = _run(row, recording_client)
    assert error is None, f"the call should have gone through, got {error!r}"
    assert len(payloads) == 1, payloads


def _threshold_warnings(caplog):
    return [record for record in caplog.records if WARNING_TEXT in record.getMessage()]


@pytest.mark.parametrize("row", REFUSAL_PARAMS)
def test_a_breached_block_cap_refuses_every_entry_point_before_the_provider(
        row, seed, provider_requests, mock_revenium_client):
    seed(_rule(row))
    _assert_refused(row, mock_revenium_client, provider_requests)


@pytest.mark.parametrize("row", ROW_PARAMS)
def test_a_warn_only_cap_lets_every_entry_point_through_with_one_warning(row, seed, mock_revenium_client, caplog):
    seed(_rule(row, action="WARN_ONLY"))
    with caplog.at_level(logging.WARNING, logger="revenium_middleware.extension"):
        _assert_let_through(row, mock_revenium_client)
    assert len(_threshold_warnings(caplog)) == 1


@pytest.mark.parametrize("row", _first_row_per_provider())
def test_a_shadow_cap_lets_every_entry_point_through(row, seed, mock_revenium_client, caplog):
    seed(_rule(row, shadow=True))
    with caplog.at_level(logging.WARNING, logger="revenium_middleware.extension"):
        _assert_let_through(row, mock_revenium_client)
    assert _threshold_warnings(caplog) == []


@pytest.mark.parametrize("row", REFUSAL_PARAMS)
def test_a_cap_scoped_to_this_provider_and_model_refuses(row, seed, provider_requests, mock_revenium_client):
    seed(_rule(row, scoped("PROVIDER", "IS", _provider_label(row)), scoped("MODEL", "IS", _model(row))))
    _assert_refused(row, mock_revenium_client, provider_requests)


@pytest.mark.parametrize("row", ROW_PARAMS)
def test_a_cap_scoped_away_from_this_provider_lets_the_call_through(row, seed, mock_revenium_client):
    seed(_rule(row, scoped("PROVIDER", "IS_NOT", _provider_label(row))))
    _assert_let_through(row, mock_revenium_client)


@pytest.mark.parametrize("row", ROW_PARAMS)
def test_a_cap_scoped_to_another_model_lets_the_call_through(row, seed, mock_revenium_client):
    seed(_rule(row, scoped("MODEL", "IS", OTHER_MODEL)))
    _assert_let_through(row, mock_revenium_client)


def test_litellm_batch_completion_returns_each_refusal_in_its_result_list(seed, mock_revenium_client, monkeypatch):
    """Pins LiteLLM's documented behaviour that the README describes: a refused item comes back as an entry, not a raise."""
    litellm = pytest.importorskip("litellm")

    row = next(r for r in MANIFEST["rows"] if r["id"] == "litellm.batch_completion.sync")
    seed(_rule(row))
    reached_litellm = []
    monkeypatch.setattr(litellm, "input_callback", [lambda *args, **kwargs: reached_litellm.append(args)])
    results = litellm.batch_completion(
        model=calls.LITELLM_MODEL,
        messages=[calls.USER_MESSAGES, calls.USER_MESSAGES],
        mock_response="hello world",
    )
    assert len(results) == 2
    assert all(isinstance(entry, BudgetExceededError) for entry in results), results
    assert reached_litellm == [], "litellm began a request for a refused item"
    calls.wait_for_metering()
    assert calls.recorded_payloads(mock_revenium_client) == []
