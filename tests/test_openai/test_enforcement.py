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
    # (rules, departmentBudgetBlocks, departmentBudgetBlockBalances,
    # departmentBudgetWarnings, departmentBudgetCandidates): no rules, no
    # department blocks, no per-person balances, nobody warned and no
    # candidates.
    assert enforcement._fetch_rules() == ([], {}, {}, {}, {})


def test_poll_interval_env_var_parsed(monkeypatch):
    """Custom poll interval honored; invalid values fall back to default."""
    monkeypatch.setenv("REVENIUM_CB_POLL_INTERVAL_SECONDS", "5")
    assert enforcement._poll_interval_seconds() == 5

    monkeypatch.setenv("REVENIUM_CB_POLL_INTERVAL_SECONDS", "garbage")
    assert enforcement._poll_interval_seconds() == 60


def test_cache_dir_persists_and_loads(monkeypatch, tmp_path):
    """When ``REVENIUM_CACHE_DIR`` is set, rules survive a process restart."""
    monkeypatch.setenv("REVENIUM_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("REVENIUM_TEAM_ID", "team-persisted")

    enforcement._persist_cache_to_disk([{"name": "persisted", "breached": True}])

    importlib.reload(enforcement)
    monkeypatch.setenv("REVENIUM_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("REVENIUM_TEAM_ID", "team-persisted")
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

    def _raise(*_args, **_kwargs):
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


# ---------------------------------------------------------------------------
# Selective metering pass-through (BACK-3991): a call outside @revenium_meter
# is not metered, but it still drops the Revenium-only ``usage_metadata``
# keyword before the provider sees it and still runs the spending-cap check.
# ---------------------------------------------------------------------------

SUBSCRIBER = {"email": "dev@example.test"}

_SYNC_WRAPPERS = (
    "create_wrapper",
    "embeddings_create_wrapper",
    "responses_create_wrapper",
    "chat_parse_wrapper",
    "responses_parse_wrapper",
)
_ASYNC_WRAPPERS = (
    "async_create_wrapper",
    "async_embeddings_create_wrapper",
    "async_responses_create_wrapper",
    "async_chat_parse_wrapper",
    "async_responses_parse_wrapper",
)
_ALL_WRAPPERS = _SYNC_WRAPPERS + _ASYNC_WRAPPERS


@pytest.fixture
def _selective_breaker_on(monkeypatch):
    monkeypatch.setenv("REVENIUM_SELECTIVE_METERING", "true")
    monkeypatch.setenv("REVENIUM_CIRCUIT_BREAKER_ENABLED", "true")


def _wrapper(name):
    from revenium_middleware.openai import middleware as openai_middleware
    return getattr(openai_middleware, name)


def _provider_must_not_be_reached(*_args, **_kwargs):
    pytest.fail("provider reached")


class _RecordingProvider:
    def __init__(self, is_async):
        self.kwargs = []
        self._is_async = is_async

    def __call__(self, *_args, **kwargs):
        self.kwargs.append(kwargs)
        if not self._is_async:
            return "provider response"

        async def respond():
            return "provider response"
        return respond()


def _call_through(name, provider, kwargs):
    import asyncio
    result = _wrapper(name)(provider, _StubInstance(), (), kwargs)
    if name in _ASYNC_WRAPPERS:
        result = asyncio.run(result)
    return result


@pytest.fixture
def _recorded_checks(monkeypatch):
    from revenium_middleware.openai import middleware as openai_middleware
    checks = []

    def record(usage_metadata=None, **_keywords):
        checks.append(usage_metadata)

    monkeypatch.setattr(openai_middleware, "check_enforcement", record)
    return checks


@pytest.mark.parametrize("name", _ALL_WRAPPERS)
def test_selective_pass_through_still_enforces_caps(name, _selective_breaker_on, _patch_check_enforcement_raises):
    """The cap check raises on the call itself, before the provider, for the
    async wrappers too (nothing is awaited here)."""
    with pytest.raises(BudgetExceededError):
        _wrapper(name)(_provider_must_not_be_reached, _StubInstance(), (),
                       {"model": "m", "usage_metadata": {"subscriber": SUBSCRIBER}})


@pytest.mark.parametrize("usage_metadata", [{"subscriber": SUBSCRIBER}, None], ids=["subscriber", "none"])
@pytest.mark.parametrize("name", _ALL_WRAPPERS)
def test_selective_pass_through_drops_usage_metadata(name, usage_metadata, _selective_breaker_on, _recorded_checks):
    provider = _RecordingProvider(is_async=name in _ASYNC_WRAPPERS)

    result = _call_through(name, provider, {"model": "m", "usage_metadata": usage_metadata})

    assert result == "provider response"
    assert provider.kwargs == [{"model": "m"}]
    expected_subscriber = SUBSCRIBER if usage_metadata else None
    assert [check.get("subscriber") for check in _recorded_checks] == [expected_subscriber]


def test_selective_pass_through_with_the_breaker_off_drops_usage_metadata_without_a_check(
        monkeypatch, _recorded_checks):
    monkeypatch.setenv("REVENIUM_SELECTIVE_METERING", "true")
    provider = _RecordingProvider(is_async=False)

    _call_through("create_wrapper", provider, {"model": "m", "usage_metadata": {"subscriber": SUBSCRIBER}})

    assert provider.kwargs == [{"model": "m"}]
    assert _recorded_checks == []


def test_selective_pass_through_checks_the_langchain_caller_without_taking_its_attribution(
        _selective_breaker_on, _recorded_checks):
    from revenium_middleware._core.call_ownership import OPENAI, publish_callback_metadata
    publication = publish_callback_metadata({"subscriber": SUBSCRIBER}, {OPENAI})
    provider = _RecordingProvider(is_async=False)

    _call_through("create_wrapper", provider, {"model": "m"})

    assert [check.get("subscriber") for check in _recorded_checks] == [SUBSCRIBER]
    assert publication.closed is False
    publication.withdraw()


# ---------------------------------------------------------------------------
# Scoped caps (BACK-4176): the check is told the request model and the
# provider label the metering record will carry, so a cap filtered to one
# model or provider can be judged before the provider is called.
# ---------------------------------------------------------------------------


class _ProviderReached(Exception):
    pass


def _provider_reached(*_args, **_kwargs):
    raise _ProviderReached()


@pytest.fixture
def _cap_check_keywords(monkeypatch):
    """Record the keywords each cap check receives, then refuse the call."""
    from revenium_middleware.openai import middleware as openai_middleware
    seen = []

    def record(usage_metadata=None, **keywords):
        seen.append(keywords)
        raise BudgetExceededError(message="recorded", rule_name="wiring")

    monkeypatch.setenv("REVENIUM_CIRCUIT_BREAKER_ENABLED", "true")
    monkeypatch.setattr(openai_middleware, "check_enforcement", record)
    return seen


class _Client:
    def __init__(self, base_url):
        self.base_url = base_url


class AzureOpenAI(_Client):
    """Named like the SDK's Azure client, which is how the provider is detected."""


class _InstanceOf:
    def __init__(self, client):
        self._client = client


@pytest.mark.parametrize("name", [
    "create_wrapper",
    "embeddings_create_wrapper",
    "responses_create_wrapper",
    "async_create_wrapper",
    "async_embeddings_create_wrapper",
])
def test_wrappers_pass_the_request_model_and_provider_to_the_cap_check(name, _cap_check_keywords):
    with pytest.raises(BudgetExceededError):
        _wrapper(name)(_provider_must_not_be_reached, _StubInstance(), (),
                       {"model": "gpt-4o-mini", "usage_metadata": {}})

    assert _cap_check_keywords == [{"model": "gpt-4o-mini", "provider": "OPENAI"}]


def test_an_azure_client_is_checked_under_the_azure_provider_label(_cap_check_keywords):
    instance = _InstanceOf(AzureOpenAI("https://example.openai.azure.com/"))

    with pytest.raises(BudgetExceededError):
        _wrapper("create_wrapper")(_provider_must_not_be_reached, instance, (),
                                   {"model": "my-deployment", "usage_metadata": {}})

    assert _cap_check_keywords == [{"model": "my-deployment", "provider": "Azure"}]


@pytest.fixture
def _without_the_perplexity_middleware(monkeypatch):
    """The OpenAI parse wrap meters a Perplexity-bound parse only when the
    Perplexity middleware, which other tests import, is not loaded."""
    from revenium_middleware.openai import middleware as openai_middleware
    monkeypatch.setattr(openai_middleware, "_perplexity_meters_parse", lambda instance: False)


@pytest.mark.usefixtures("_without_the_perplexity_middleware")
def test_a_perplexity_bound_parse_is_checked_under_the_perplexity_provider_label(_cap_check_keywords):
    instance = _InstanceOf(_Client("https://api.perplexity.ai"))

    with pytest.raises(BudgetExceededError):
        _wrapper("chat_parse_wrapper")(_provider_must_not_be_reached, instance, (),
                                       {"model": "sonar", "usage_metadata": {}})

    assert _cap_check_keywords == [{"model": "sonar", "provider": "PERPLEXITY"}]


@pytest.mark.parametrize("name", _ALL_WRAPPERS)
def test_unmetered_calls_pass_the_request_model_to_the_cap_check(name, _selective_breaker_on, _cap_check_keywords):
    with pytest.raises(BudgetExceededError):
        _wrapper(name)(_provider_must_not_be_reached, _StubInstance(), (),
                       {"model": "gpt-4o-mini", "usage_metadata": {}})

    assert _cap_check_keywords == [{"model": "gpt-4o-mini", "provider": "OPENAI"}]


@pytest.mark.usefixtures("_without_the_perplexity_middleware")
def test_an_unmetered_perplexity_bound_parse_is_checked_under_the_perplexity_provider_label(
        _selective_breaker_on, _cap_check_keywords):
    instance = _InstanceOf(_Client("https://api.perplexity.ai"))

    with pytest.raises(BudgetExceededError):
        _wrapper("chat_parse_wrapper")(_provider_must_not_be_reached, instance, (),
                                       {"model": "sonar", "usage_metadata": {}})

    assert _cap_check_keywords == [{"model": "sonar", "provider": "PERPLEXITY"}]


_PERPLEXITY_CHAT_WRAPS = (
    "perplexity:openai.resources.chat.completions.Completions.create",
    "perplexity:openai.resources.chat.completions.AsyncCompletions.create",
)


def _set_perplexity_wraps(monkeypatch, installed):
    """Say whether the Perplexity middleware wrapped the chat slot, whatever this run imported."""
    from revenium_middleware._core import patch_registry
    patched = set(patch_registry._patched) - set(_PERPLEXITY_CHAT_WRAPS)
    if installed:
        patched |= set(_PERPLEXITY_CHAT_WRAPS)
    monkeypatch.setattr(patch_registry, "_patched", patched)


@pytest.mark.parametrize("name", ["create_wrapper", "async_create_wrapper"])
def test_a_perplexity_bound_chat_call_is_checked_when_the_perplexity_wrap_is_absent(
        name, monkeypatch, _cap_check_keywords):
    """BACK-4252: with only the OpenAI middleware imported nothing else checks
    this call, so the OpenAI wrap must."""
    _set_perplexity_wraps(monkeypatch, installed=False)
    instance = _InstanceOf(_Client("https://api.perplexity.ai"))

    with pytest.raises(BudgetExceededError):
        _wrapper(name)(_provider_must_not_be_reached, instance, (), {"model": "sonar", "usage_metadata": {}})

    assert _cap_check_keywords == [{"model": "sonar", "provider": "PERPLEXITY"}]


@pytest.mark.parametrize("name", ["create_wrapper", "async_create_wrapper"])
def test_a_perplexity_bound_call_with_its_agent_in_extra_body_is_held_by_that_agents_cap(
        name, monkeypatch, _selective_breaker_on):
    """BACK-4252: attribution given only in extra_body must reach the cap check."""
    _set_perplexity_wraps(monkeypatch, installed=False)
    _seed_rules(monkeypatch, [{
        "ruleId": 4252, "name": "agent cap", "action": "BLOCK", "breached": True, "shadowMode": False,
        "filters": [{"dimension": "AGENT", "operator": "IS", "value": "limited-agent"}],
    }])
    instance = _InstanceOf(_Client("https://api.perplexity.ai"))

    with pytest.raises(BudgetExceededError):
        _wrapper(name)(_provider_must_not_be_reached, instance, (),
                       {"model": "sonar", "extra_body": {"usage_metadata": {"agent": "limited-agent"}}})


@pytest.mark.parametrize("name", ["create_wrapper", "async_create_wrapper"])
def test_a_refused_perplexity_bound_call_leaves_the_callers_extra_body_so_a_retry_is_refused_again(
        name, monkeypatch, _selective_breaker_on):
    """BACK-4252: a retry with the same extra_body must still carry its agent."""
    _set_perplexity_wraps(monkeypatch, installed=False)
    _seed_rules(monkeypatch, [{
        "ruleId": 4252, "name": "agent cap", "action": "BLOCK", "breached": True, "shadowMode": False,
        "filters": [{"dimension": "AGENT", "operator": "IS", "value": "limited-agent"}],
    }])
    instance = _InstanceOf(_Client("https://api.perplexity.ai"))
    extra_body = {"usage_metadata": {"agent": "limited-agent"}}

    for _ in range(2):
        with pytest.raises(BudgetExceededError):
            _wrapper(name)(_provider_must_not_be_reached, instance, (), {"model": "sonar", "extra_body": extra_body})

    assert extra_body == {"usage_metadata": {"agent": "limited-agent"}}


@pytest.mark.parametrize("name", ["create_wrapper", "async_create_wrapper"])
def test_an_allowed_perplexity_bound_call_does_not_send_extra_body_usage_metadata_on(
        name, monkeypatch, _selective_breaker_on):
    import asyncio

    _set_perplexity_wraps(monkeypatch, installed=False)
    _seed_rules(monkeypatch, [])
    instance = _InstanceOf(_Client("https://api.perplexity.ai"))
    provider = _RecordingProvider(is_async=name in _ASYNC_WRAPPERS)

    result = _wrapper(name)(provider, instance, (), {
        "model": "sonar", "extra_body": {"usage_metadata": {"agent": "limited-agent"}, "search_mode": "web"}})
    if name in _ASYNC_WRAPPERS:
        result = asyncio.run(result)

    assert result == "provider response"
    assert provider.kwargs == [{"model": "sonar", "extra_body": {"search_mode": "web"}}]


@pytest.mark.parametrize("name", ["create_wrapper", "async_create_wrapper"])
def test_a_perplexity_bound_chat_call_is_left_to_the_perplexity_wrap_when_it_is_installed(
        name, monkeypatch, _cap_check_keywords):
    import asyncio

    _set_perplexity_wraps(monkeypatch, installed=True)
    instance = _InstanceOf(_Client("https://api.perplexity.ai"))
    provider = _RecordingProvider(is_async=name in _ASYNC_WRAPPERS)

    result = _wrapper(name)(provider, instance, (), {"model": "sonar"})
    if name in _ASYNC_WRAPPERS:
        result = asyncio.run(result)

    assert result == "provider response"
    assert provider.kwargs == [{"model": "sonar"}]
    assert _cap_check_keywords == []


@pytest.mark.parametrize("name", _ALL_WRAPPERS)
def test_an_unmetered_azure_deployment_is_held_by_a_breached_model_cap(name, monkeypatch, _selective_breaker_on):
    _seed_rules(monkeypatch, [{
        "ruleId": 4176, "name": "gpt-4o cap", "action": "BLOCK", "breached": True, "shadowMode": False,
        "filters": [{"dimension": "MODEL", "operator": "IS", "value": "gpt-4o"}],
    }])
    instance = _InstanceOf(AzureOpenAI("https://example.openai.azure.com/"))

    with pytest.raises(BudgetExceededError):
        _wrapper(name)(_provider_must_not_be_reached, instance, (), {"model": "prod-4o", "usage_metadata": {}})


@pytest.mark.parametrize("name", _ALL_WRAPPERS)
def test_a_breached_cap_on_another_provider_lets_an_unmetered_openai_call_through(
        name, monkeypatch, _selective_breaker_on):
    _seed_rules(monkeypatch, [{
        "ruleId": 4176, "name": "anthropic cap", "action": "BLOCK", "breached": True, "shadowMode": False,
        "filters": [{"dimension": "PROVIDER", "operator": "IS", "value": "anthropic"}],
    }])
    provider = _RecordingProvider(is_async=name in _ASYNC_WRAPPERS)

    assert _call_through(name, provider, {"model": "gpt-4o-mini", "usage_metadata": {}}) == "provider response"
    assert provider.kwargs == [{"model": "gpt-4o-mini"}]


def test_a_model_filtered_cap_lets_another_model_through_the_chat_wrapper(monkeypatch):
    from revenium_middleware.openai.middleware import create_wrapper

    monkeypatch.setenv("REVENIUM_CIRCUIT_BREAKER_ENABLED", "true")
    _seed_rules(monkeypatch, [{
        "ruleId": 4176, "name": "gpt-4o-mini cap", "action": "BLOCK", "breached": True, "shadowMode": False,
        "filters": [{"dimension": "MODEL", "operator": "IS", "value": "gpt-4o-mini"}],
    }])

    with pytest.raises(_ProviderReached):
        create_wrapper(_provider_reached, _StubInstance(), (), {"model": "gpt-4.1", "usage_metadata": {}})
    with pytest.raises(BudgetExceededError):
        create_wrapper(_provider_reached, _StubInstance(), (), {"model": "gpt-4o-mini", "usage_metadata": {}})


# ---------------------------------------------------------------------------
# BACK-4239: a call refused by a spending cap is never recorded as a failed
# call, whether the cap check or the provider call raised it, and however
# deeply the provider call wrapped it.
# ---------------------------------------------------------------------------


def _wrapped_budget_refusal(*_args, **_kwargs):
    try:
        raise BudgetExceededError(message="budget breached", rule_name="inner-cap")
    except BudgetExceededError as refusal:
        try:
            raise RuntimeError("transport wrapped it") from refusal
        except RuntimeError:
            raise ValueError("client wrapped it again")


@pytest.fixture
def sent_records(monkeypatch):
    from unittest.mock import MagicMock
    from revenium_middleware.openai import middleware as openai_middleware

    sent = []
    monkeypatch.setattr(openai_middleware, "submit_ai_event",
                        lambda operation, args, **kwargs: sent.append(args) or MagicMock())
    monkeypatch.setattr(openai_middleware, "run_async_in_thread", lambda coroutine: __import__("asyncio").run(coroutine))
    monkeypatch.setattr(openai_middleware, "current_client", lambda: object())
    return sent


def test_budget_exceeded_sends_no_error_record(_patch_check_enforcement_raises, sent_records):
    from revenium_middleware.openai.middleware import create_wrapper

    with pytest.raises(BudgetExceededError):
        create_wrapper(_stub_wrapped, _StubInstance(), (), {"model": "gpt-4o-mini", "usage_metadata": {}})
    assert sent_records == []


def test_budget_refusal_wrapped_by_the_provider_call_sends_no_error_record(sent_records):
    from revenium_middleware.openai.middleware import create_wrapper

    with pytest.raises(ValueError, match="client wrapped it again"):
        create_wrapper(_wrapped_budget_refusal, _StubInstance(), (), {"model": "gpt-4o-mini", "usage_metadata": {}})
    assert sent_records == []


def test_async_anthropic_budget_refusal_sends_no_error_record(monkeypatch):
    import asyncio
    from unittest.mock import MagicMock
    pytest.importorskip("anthropic")
    from revenium_middleware.anthropic import middleware as anthropic_middleware

    sent = []
    monkeypatch.setattr(anthropic_middleware, "submit_ai_event", lambda operation, args, **kwargs: sent.append(args))
    monkeypatch.setattr(anthropic_middleware, "_get_thread_safe_client", lambda: MagicMock())
    monkeypatch.setattr(anthropic_middleware, "_safe_run_async_in_thread",
                        lambda coro_func, *a, **k: asyncio.run(coro_func(*a, **k)))

    def refuse(*_args, **_kwargs):
        raise BudgetExceededError(message="budget breached", rule_name="anthropic-cap")

    monkeypatch.setattr(anthropic_middleware, "_check_caps", refuse)

    async def must_not_run(*_args, **_kwargs):  # pragma: no cover
        raise AssertionError("the provider must not be reached")

    with pytest.raises(BudgetExceededError):
        asyncio.run(anthropic_middleware.async_create_wrapper(
            must_not_run, None, (), {"model": "claude-haiku-4-5", "max_tokens": 8, "messages": []}))
    assert sent == []


def test_litellm_budget_refusal_rewrapped_by_litellm_sends_no_error_record(monkeypatch):
    pytest.importorskip("litellm")
    from revenium_middleware.litellm.client import middleware as litellm_middleware

    sent = []
    monkeypatch.setattr(litellm_middleware, "submit_ai_event", lambda operation, args, **kwargs: sent.append(args))
    monkeypatch.setattr(litellm_middleware, "run_async_in_thread", lambda coroutine: __import__("asyncio").run(coroutine))

    with pytest.raises(ValueError, match="client wrapped it again"):
        litellm_middleware.completion_wrapper(
            _wrapped_budget_refusal, None, (), {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "hi"}]})
    assert sent == []


@pytest.fixture
def anthropic_records(monkeypatch):
    import asyncio
    import threading
    from unittest.mock import MagicMock
    pytest.importorskip("anthropic")
    from revenium_middleware.anthropic import middleware as anthropic_middleware

    def run_in_its_own_thread(coro_func, *args, **kwargs):
        thread = threading.Thread(target=lambda: asyncio.run(coro_func(*args, **kwargs)))
        thread.start()
        thread.join()

    sent = []
    monkeypatch.setattr(anthropic_middleware, "submit_ai_event", lambda operation, args, **kwargs: sent.append(args))
    monkeypatch.setattr(anthropic_middleware, "_get_thread_safe_client", lambda: MagicMock())
    monkeypatch.setattr(anthropic_middleware, "_safe_run_async_in_thread", run_in_its_own_thread)
    return sent


_ANTHROPIC_REQUEST = {"model": "claude-haiku-4-5", "max_tokens": 8, "messages": []}


def test_anthropic_budget_refusal_wrapped_by_the_provider_call_sends_no_error_record(anthropic_records):
    from revenium_middleware.anthropic import middleware as anthropic_middleware

    with pytest.raises(ValueError, match="client wrapped it again"):
        anthropic_middleware.create_wrapper(_wrapped_budget_refusal, None, (), dict(_ANTHROPIC_REQUEST))
    assert anthropic_records == []


def test_async_anthropic_budget_refusal_wrapped_by_the_provider_call_sends_no_error_record(anthropic_records):
    import asyncio
    from revenium_middleware.anthropic import middleware as anthropic_middleware

    async def wrapped(*args, **kwargs):
        _wrapped_budget_refusal(*args, **kwargs)

    with pytest.raises(ValueError, match="client wrapped it again"):
        asyncio.run(anthropic_middleware.async_create_wrapper(wrapped, None, (), dict(_ANTHROPIC_REQUEST)))
    assert anthropic_records == []


# A caller that falls back to another call while handling a cap refusal of its
# own: the fallback's provider error implicitly chains back to that refusal.

class _RateLimited(Exception):
    pass


def _rate_limited_inside_the_sdk(*_args, **_kwargs):
    try:
        raise LookupError("HTTP 429")
    except LookupError:
        raise _RateLimited("rate limited") from None


def _fallback_after_a_cap_refusal(call):
    try:
        raise BudgetExceededError(message="budget breached", rule_name="primary-model-cap")
    except BudgetExceededError:
        return call()


def test_a_fallback_call_made_while_handling_a_cap_refusal_records_its_own_failure(sent_records):
    from revenium_middleware.openai.middleware import create_wrapper

    with pytest.raises(_RateLimited):
        _fallback_after_a_cap_refusal(lambda: create_wrapper(
            _rate_limited_inside_the_sdk, _StubInstance(), (), {"model": "gpt-4o-mini", "usage_metadata": {}}))
    assert [record["stop_reason"] for record in sent_records] == ["ERROR"]


def test_an_anthropic_fallback_call_made_while_handling_a_cap_refusal_records_its_own_failure(anthropic_records):
    from revenium_middleware.anthropic import middleware as anthropic_middleware

    with pytest.raises(_RateLimited):
        _fallback_after_a_cap_refusal(lambda: anthropic_middleware.create_wrapper(
            _rate_limited_inside_the_sdk, None, (), dict(_ANTHROPIC_REQUEST)))
    assert [record["stop_reason"] for record in anthropic_records] == ["ERROR"]


def test_an_async_anthropic_fallback_call_made_while_handling_a_cap_refusal_records_its_own_failure(
        anthropic_records):
    import asyncio
    from revenium_middleware.anthropic import middleware as anthropic_middleware

    async def wrapped(*args, **kwargs):
        _rate_limited_inside_the_sdk(*args, **kwargs)

    async def caller():
        try:
            raise BudgetExceededError(message="budget breached", rule_name="primary-model-cap")
        except BudgetExceededError:
            await anthropic_middleware.async_create_wrapper(wrapped, None, (), dict(_ANTHROPIC_REQUEST))

    with pytest.raises(_RateLimited):
        asyncio.run(caller())
    assert [record["stop_reason"] for record in anthropic_records] == ["ERROR"]


def test_a_litellm_fallback_call_made_while_handling_a_cap_refusal_records_its_own_failure(monkeypatch):
    pytest.importorskip("litellm")
    from revenium_middleware.litellm.client import middleware as litellm_middleware

    sent = []
    monkeypatch.setattr(litellm_middleware, "submit_ai_event", lambda operation, args, **kwargs: sent.append(args))
    monkeypatch.setattr(litellm_middleware, "run_async_in_thread", lambda coroutine: __import__("asyncio").run(coroutine))

    with pytest.raises(_RateLimited):
        _fallback_after_a_cap_refusal(lambda: litellm_middleware.completion_wrapper(
            _rate_limited_inside_the_sdk, None, (),
            {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "hi"}]}))
    assert [record["stop_reason"] for record in sent] == ["ERROR"]


def _breached_rule(**overrides):
    return {"ruleId": 152, "name": "monthly-gpt4-cap", "threshold": 0.01, "currentValue": 2934,
            "breached": True, "shadowMode": False, **overrides}


@pytest.mark.parametrize("deny_message", [
    "Ask the user before retrying.",
    "x" * 2000,
    "{0.__class__}",
    "{",
    "Ask first. {spent} of {limit}",
], ids=["plain", "longest-allowed", "attribute-field", "lone-brace", "placeholders"])
def test_a_breached_rule_with_a_deny_message_raises_with_it(monkeypatch, deny_message):
    """BACK-4407: the message the customer wrote for the rule reaches the caller, placeholders as written."""
    monkeypatch.setenv("REVENIUM_CIRCUIT_BREAKER_ENABLED", "true")
    _seed_rules(monkeypatch, [_breached_rule(denyMessage=deny_message)])

    with pytest.raises(BudgetExceededError) as exc_info:
        enforcement.check_enforcement({})

    err = exc_info.value
    assert err.message == f"Request blocked by Revenium enforcement rule: monthly-gpt4-cap. {deny_message}"
    assert str(err) == err.message
    assert err.deny_message == deny_message
    assert err.rule_name == "monthly-gpt4-cap"


@pytest.mark.parametrize("deny_message", [None, "", "   ", 42, ["x"], {"a": 1}],
                         ids=["none", "empty", "blank", "number", "list", "dict"])
def test_a_rule_with_a_blank_or_non_string_deny_message_keeps_todays_text(monkeypatch, deny_message):
    monkeypatch.setenv("REVENIUM_CIRCUIT_BREAKER_ENABLED", "true")
    _seed_rules(monkeypatch, [_breached_rule(denyMessage=deny_message)])

    with pytest.raises(BudgetExceededError) as exc_info:
        enforcement.check_enforcement({})

    err = exc_info.value
    assert err.message == "Request blocked by Revenium enforcement rule: monthly-gpt4-cap"
    assert getattr(err, "deny_message", "unset") is None
