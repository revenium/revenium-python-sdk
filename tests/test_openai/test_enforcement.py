"""
Unit tests for the cost-controls circuit breaker.

The enforcement engine lives in ``revenium_middleware._core.enforcement`` and
is wired into the OpenAI middleware via ``check_enforcement(usage_metadata)``.
These tests exercise the engine's decision logic in isolation — fetch is
stubbed so no network calls are made.
"""

from __future__ import annotations

import importlib

import pytest

from revenium_middleware._core import enforcement
from revenium_middleware._core.exceptions import BudgetExceededError


@pytest.fixture(autouse=True)
def _reset_enforcement_state(monkeypatch):
    """Each test starts with a fresh in-memory cache and no env vars set."""
    for var in (
        "REVENIUM_CIRCUIT_BREAKER_ENABLED",
        "REVENIUM_BYPASS",
        "REVENIUM_CB_FAIL_MODE",
        "REVENIUM_CB_POLL_INTERVAL_SECONDS",
        "REVENIUM_CACHE_DIR",
        "REVENIUM_TEAM_ID",
        "REVENIUM_METERING_API_KEY",
        "REVENIUM_ENFORCEMENT_BASE_URL",
    ):
        monkeypatch.delenv(var, raising=False)

    importlib.reload(enforcement)
    yield
    enforcement.stop_polling()


def _seed_rules(monkeypatch, rules: list, *, ttl_age: float = 0.0, initialized: bool = True) -> None:
    """Inject rules directly into the in-memory cache, bypassing the poller.

    ``initialized=True`` (default) models a successful fetch having populated
    the cache. Pass ``initialized=False`` to model the boot state where no
    fetch has yet completed (relevant to fail-closed semantics).
    """
    monkeypatch.setattr(enforcement, "_cached_rules", list(rules))
    monkeypatch.setattr(enforcement, "_cache_timestamp", float("inf") if ttl_age == 0.0 else 0.0)
    monkeypatch.setattr(enforcement, "_cache_initialized", initialized)
    # Block both the poller and any synchronous refresh from clobbering us.
    monkeypatch.setattr(enforcement, "_ensure_poller_running", lambda: None)
    monkeypatch.setattr(enforcement, "_fetch_rules", lambda: None)


def test_disabled_is_noop(monkeypatch):
    """No env vars set → check_enforcement returns immediately, never fetches."""
    called = []
    monkeypatch.setattr(enforcement, "_ensure_poller_running", lambda: called.append("poll"))
    enforcement.check_enforcement({})
    assert called == []


def test_bypass_short_circuits(monkeypatch):
    """REVENIUM_BYPASS=true wins even when the breaker is enabled."""
    monkeypatch.setenv("REVENIUM_CIRCUIT_BREAKER_ENABLED", "true")
    monkeypatch.setenv("REVENIUM_BYPASS", "true")
    _seed_rules(monkeypatch, [{"name": "would-block", "breached": True}])
    enforcement.check_enforcement({})


def test_breached_rule_raises_with_structured_fields(monkeypatch):
    """A breached rule raises with all 6 fields populated from the server payload."""
    monkeypatch.setenv("REVENIUM_CIRCUIT_BREAKER_ENABLED", "true")
    _seed_rules(
        monkeypatch,
        [
            {
                "ruleId": 152,
                "name": "monthly-gpt4-cap",
                "threshold": 0.01,
                "currentValue": 2934,
                "breached": True,
                "shadowMode": False,
                "resetsAt": "2026-06-01T00:00:00Z",
            }
        ],
    )

    with pytest.raises(BudgetExceededError) as exc_info:
        enforcement.check_enforcement({})

    err = exc_info.value
    assert err.message == "Request blocked by Revenium enforcement rule: monthly-gpt4-cap"
    assert err.rule_name == "monthly-gpt4-cap"
    assert err.current_value == 2934.0
    assert err.threshold == 0.01
    assert err.resets_at == "2026-06-01T00:00:00Z"
    assert err.rule_id == 152


def test_legacy_blocked_flag_still_honored(monkeypatch):
    """Pre-BACK-1133 servers used ``blocked`` instead of ``breached``."""
    monkeypatch.setenv("REVENIUM_CIRCUIT_BREAKER_ENABLED", "true")
    _seed_rules(monkeypatch, [{"name": "legacy-rule", "blocked": True}])
    with pytest.raises(BudgetExceededError) as exc_info:
        enforcement.check_enforcement({})
    assert exc_info.value.rule_name == "legacy-rule"


def test_shadow_mode_does_not_raise(monkeypatch):
    """``shadowMode: true`` rules are observe-and-log; they must not block."""
    monkeypatch.setenv("REVENIUM_CIRCUIT_BREAKER_ENABLED", "true")
    _seed_rules(
        monkeypatch,
        [{"name": "shadow", "breached": True, "shadowMode": True}],
    )
    enforcement.check_enforcement({})


def test_credential_scoped_rule_skipped_for_other_credential(monkeypatch):
    """A rule scoped to credential X must not block calls from credential Y."""
    monkeypatch.setenv("REVENIUM_CIRCUIT_BREAKER_ENABLED", "true")
    _seed_rules(
        monkeypatch,
        [{"name": "alice-only", "breached": True, "credential": "alice-key"}],
    )
    enforcement.check_enforcement({"subscriber_credential": "bob-key"})


def test_credential_scoped_rule_blocks_matching_credential(monkeypatch):
    monkeypatch.setenv("REVENIUM_CIRCUIT_BREAKER_ENABLED", "true")
    _seed_rules(
        monkeypatch,
        [{"name": "alice-only", "breached": True, "credential": "alice-key"}],
    )
    with pytest.raises(BudgetExceededError):
        enforcement.check_enforcement({"subscriber_credential": "alice-key"})


def test_fail_closed_with_uninitialized_cache_raises(monkeypatch):
    """``REVENIUM_CB_FAIL_MODE=closed`` must refuse when the cache has never loaded."""
    monkeypatch.setenv("REVENIUM_CIRCUIT_BREAKER_ENABLED", "true")
    monkeypatch.setenv("REVENIUM_CB_FAIL_MODE", "closed")
    _seed_rules(monkeypatch, [], initialized=False)
    with pytest.raises(BudgetExceededError):
        enforcement.check_enforcement({})


def test_fail_closed_with_initialized_empty_cache_passes(monkeypatch):
    """A successful fetch that returns ``[]`` (HTTP 204 = no rules apply) must
    pass through even in fail-closed mode — the server explicitly said the
    request is permitted, so blocking it would be a false positive."""
    monkeypatch.setenv("REVENIUM_CIRCUIT_BREAKER_ENABLED", "true")
    monkeypatch.setenv("REVENIUM_CB_FAIL_MODE", "closed")
    _seed_rules(monkeypatch, [], initialized=True)
    enforcement.check_enforcement({})


def test_fail_open_with_empty_cache_passes(monkeypatch):
    """Default fail-mode is open — empty cache means pass through."""
    monkeypatch.setenv("REVENIUM_CIRCUIT_BREAKER_ENABLED", "true")
    _seed_rules(monkeypatch, [], initialized=False)
    enforcement.check_enforcement({})


def test_204_response_clears_cache(monkeypatch):
    """A 204 from the enforcement API is 'no rules configured' → cache empty list."""
    monkeypatch.setenv("REVENIUM_CIRCUIT_BREAKER_ENABLED", "true")
    monkeypatch.setenv("REVENIUM_METERING_API_KEY", "hak_test")
    monkeypatch.setenv("REVENIUM_TEAM_ID", "team-xyz")

    class _Resp:
        status_code = 204

        def raise_for_status(self):
            pass

        def json(self):
            return None

    monkeypatch.setattr(enforcement.httpx, "get", lambda *a, **kw: _Resp())
    # (rules, orgUnitBudgetBlocks): no rules, and no department blocks either.
    assert enforcement._fetch_rules() == ([], {})


def test_poll_interval_env_var_parsed(monkeypatch):
    """Custom poll interval honored; invalid values fall back to default."""
    monkeypatch.setenv("REVENIUM_CB_POLL_INTERVAL_SECONDS", "5")
    assert enforcement._poll_interval_seconds() == 5

    monkeypatch.setenv("REVENIUM_CB_POLL_INTERVAL_SECONDS", "garbage")
    assert enforcement._poll_interval_seconds() == 60


def test_cache_dir_persists_and_loads(monkeypatch, tmp_path):
    """When ``REVENIUM_CACHE_DIR`` is set, rules survive a process restart."""
    monkeypatch.setenv("REVENIUM_CACHE_DIR", str(tmp_path))

    enforcement._persist_cache_to_disk([{"name": "persisted", "breached": True}])

    importlib.reload(enforcement)
    monkeypatch.setenv("REVENIUM_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("REVENIUM_CIRCUIT_BREAKER_ENABLED", "true")
    monkeypatch.setattr(enforcement, "_ensure_poller_running", lambda: None)
    monkeypatch.setattr(enforcement, "_fetch_rules", lambda: None)

    with pytest.raises(BudgetExceededError) as exc_info:
        enforcement.check_enforcement({})
    assert exc_info.value.rule_name == "persisted"


# ---------------------------------------------------------------------------
# Wrapper-propagation tests
#
# The OpenAI wrappers must let ``BudgetExceededError`` escape so callers
# can ``except`` it. ``handle_exception_safely`` lives next to them and treats
# every other ``ReveniumMiddlewareError`` subclass as graceful-degradation —
# so a regression in either the dedicated re-raise guard inside that decorator
# or the import identity of the exception class would silently swallow
# enforcement at runtime, defeating the circuit breaker.
# ---------------------------------------------------------------------------


@pytest.fixture
def _patch_check_enforcement_raises(monkeypatch):
    """Force ``check_enforcement`` (as imported by the wrappers) to raise."""
    from revenium_middleware.openai import middleware as openai_middleware

    def _raise(_metadata):
        raise BudgetExceededError(
            message="budget breached",
            rule_name="propagation-test",
        )

    monkeypatch.setattr(openai_middleware, "check_enforcement", _raise)
    return _raise


class _StubInstance:
    """Minimal stand-in for an OpenAI client resource instance."""
    _client = None


def _stub_wrapped(*_args, **_kwargs):  # pragma: no cover — must never run
    raise AssertionError("wrapped() must not be invoked when enforcement raises")


def test_handle_exception_safely_does_not_swallow_cost_limit():
    """Direct decorator test — guards against an import-identity regression."""
    from revenium_middleware.openai.exceptions import handle_exception_safely

    @handle_exception_safely
    def fn():
        raise BudgetExceededError(message="x", rule_name="r")

    with pytest.raises(BudgetExceededError):
        fn()


def test_create_wrapper_propagates_cost_limit_exception(_patch_check_enforcement_raises):
    from revenium_middleware.openai.middleware import create_wrapper

    with pytest.raises(BudgetExceededError):
        create_wrapper(_stub_wrapped, _StubInstance(), (), {"usage_metadata": {}})


def test_embeddings_create_wrapper_propagates_cost_limit_exception(_patch_check_enforcement_raises):
    from revenium_middleware.openai.middleware import embeddings_create_wrapper

    with pytest.raises(BudgetExceededError):
        embeddings_create_wrapper(_stub_wrapped, _StubInstance(), (), {"usage_metadata": {}})


def test_responses_create_wrapper_propagates_cost_limit_exception(_patch_check_enforcement_raises):
    from revenium_middleware.openai.middleware import responses_create_wrapper

    with pytest.raises(BudgetExceededError):
        responses_create_wrapper(_stub_wrapped, _StubInstance(), (), {"usage_metadata": {}})


def test_async_create_wrapper_propagates_cost_limit_exception(_patch_check_enforcement_raises):
    """Async chat.completions wrapper raises synchronously before returning a coroutine."""
    from revenium_middleware.openai.middleware import async_create_wrapper

    with pytest.raises(BudgetExceededError):
        async_create_wrapper(_stub_wrapped, _StubInstance(), (), {"usage_metadata": {}})


def test_async_embeddings_create_wrapper_propagates_cost_limit_exception(_patch_check_enforcement_raises):
    from revenium_middleware.openai.middleware import async_embeddings_create_wrapper

    with pytest.raises(BudgetExceededError):
        async_embeddings_create_wrapper(_stub_wrapped, _StubInstance(), (), {"usage_metadata": {}})
