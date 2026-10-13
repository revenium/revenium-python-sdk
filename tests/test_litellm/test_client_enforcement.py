"""Spending caps on LiteLLM client calls.

The client wrapper runs the pre-call check with the caller's own metadata and
raises ``BudgetExceededError`` before LiteLLM is entered, so a per-person cap
blocks its developer, and a team cap reaches the caller as the documented
budget error rather than a LiteLLM provider error that retry and fallback
logic treat as an outage (BACK-4028). A call outside @revenium_meter under
selective metering is not metered but still runs the check (BACK-3991).
Requests the LiteLLM proxy serves are left to the proxy guardrail.
"""
import asyncio
import contextvars
import importlib
import threading
import time
from http.server import ThreadingHTTPServer

import pytest

pytest.importorskip("litellm")

import litellm  # noqa: E402

import revenium_middleware.openai  # noqa: E402,F401
from revenium_middleware import revenium_metadata  # noqa: E402
from revenium_middleware._core import enforcement  # noqa: E402
from revenium_middleware._core.exceptions import BudgetExceededError  # noqa: E402
from revenium_middleware.litellm.client import middleware as mw  # noqa: E402

from . import fake_openai_upstream  # noqa: E402

MODEL = "gpt-4o-mini"
MESSAGES = [{"role": "user", "content": "hi"}]
SUBSCRIBER = {"email": "dev@example.test"}

ENTRY_POINTS = {
    "completion": (mw.completion_wrapper, False),
    "acompletion": (mw.acompletion_wrapper, True),
    "embedding": (mw.embedding_wrapper, False),
    "aembedding": (mw.aembedding_wrapper, True),
}


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
        "REVENIUM_SELECTIVE_METERING",
    ):
        monkeypatch.delenv(var, raising=False)

    importlib.reload(enforcement)
    yield
    enforcement.stop_polling()


@pytest.fixture
def breaker_on(monkeypatch):
    monkeypatch.setenv("REVENIUM_CIRCUIT_BREAKER_ENABLED", "true")


@pytest.fixture
def check_raises(monkeypatch):
    checked = []

    def check(usage_metadata, **_keywords):
        checked.append(usage_metadata)
        raise BudgetExceededError(message="budget breached", rule_name="client-test")

    monkeypatch.setattr(mw, "check_enforcement", check, raising=False)
    return checked


def provider_must_not_be_reached(*_args, **_kwargs):
    pytest.fail("provider reached")


async def async_provider_must_not_be_reached(*_args, **_kwargs):
    pytest.fail("provider reached")


def start_entry_point(entry_point, kwargs):
    """Start a call through the wrapper with a provider that fails the test if
    reached, and return what finishes it. An async entry point must not raise
    until it is awaited, so its call happens here, outside the caller's
    ``pytest.raises``."""
    wrapper, is_async = ENTRY_POINTS[entry_point]
    if not is_async:
        return lambda: wrapper(provider_must_not_be_reached, None, (), kwargs)
    pending = wrapper(async_provider_must_not_be_reached, None, (), kwargs)
    return lambda: asyncio.run(pending)


@pytest.mark.parametrize("entry_point", list(ENTRY_POINTS))
def test_selective_pass_through_still_enforces_caps(entry_point, breaker_on, check_raises, monkeypatch):
    monkeypatch.setenv("REVENIUM_SELECTIVE_METERING", "true")

    finish = start_entry_point(entry_point, {"model": MODEL, "messages": MESSAGES,
                                             "usage_metadata": {"subscriber": SUBSCRIBER}})

    with pytest.raises(BudgetExceededError):
        finish()

    assert [checked.get("subscriber") for checked in check_raises] == [SUBSCRIBER]


# ---- rules served from the enforcement cache --------------------------------

OVER_CAP_EMAIL = "over-cap@example.test"
UNDER_CAP_EMAIL = "under-cap@example.test"


def per_person_rule():
    """A ``groupBy=SUBSCRIBER`` rule on which only OVER_CAP_EMAIL is breached."""
    return {
        "ruleId": 4242,
        "name": "Per-developer cap",
        "metricType": "TOTAL_COST",
        "threshold": 100.0,
        "currentValue": 250.0,
        "periodType": "MONTHLY",
        "groupBy": "SUBSCRIBER",
        "breached": False,
        "shadowMode": False,
        "groupBreakdown": [
            {"groupValue": OVER_CAP_EMAIL, "displayName": OVER_CAP_EMAIL, "currentValue": 140.0,
             "usagePercent": 1.4, "breached": True},
            {"groupValue": UNDER_CAP_EMAIL, "displayName": UNDER_CAP_EMAIL, "currentValue": 12.5,
             "usagePercent": 0.125, "breached": False},
        ],
    }


def pooled_rule():
    return {
        "ruleId": 99,
        "name": "Team monthly cap",
        "metricType": "TOTAL_COST",
        "threshold": 500.0,
        "currentValue": 640.0,
        "periodType": "MONTHLY",
        "groupBy": None,
        "breached": True,
        "shadowMode": False,
    }


@pytest.fixture
def load_rules(monkeypatch, breaker_on):
    """Install a rule list as a fresh, initialized cache, with no poller,
    disk snapshot or fetch."""
    monkeypatch.setattr(enforcement, "_load_cache_from_disk", lambda: None)
    monkeypatch.setattr(enforcement, "_ensure_poller_running", lambda: None)
    monkeypatch.setattr(enforcement, "_fetch_rules", lambda: None)

    def load(rules):
        monkeypatch.setattr(enforcement, "_cached_rules", list(rules))
        monkeypatch.setattr(enforcement, "_cached_org_unit_blocks", {})
        monkeypatch.setattr(enforcement, "_cache_timestamp", time.monotonic())
        monkeypatch.setattr(enforcement, "_cache_initialized", True)

    return load


def mocked_completion(email):
    return litellm.completion(model=MODEL, messages=MESSAGES, mock_response="hi",
                              usage_metadata={"subscriber": {"email": email}})


def test_per_person_cap_blocks_the_caller_before_litellm_runs(load_rules):
    load_rules([per_person_rule()])

    with pytest.raises(BudgetExceededError) as blocked:
        mocked_completion(OVER_CAP_EMAIL)

    assert type(blocked.value) is BudgetExceededError
    assert mocked_completion(UNDER_CAP_EMAIL).choices[0].message.content == "hi"


def test_team_cap_raises_budget_exceeded_error_not_a_litellm_error(load_rules):
    load_rules([pooled_rule()])

    with pytest.raises(BudgetExceededError) as blocked:
        mocked_completion(UNDER_CAP_EMAIL)

    assert type(blocked.value) is BudgetExceededError
    assert blocked.value.rule_name == "Team monthly cap"


def test_revenium_metadata_identity_is_checked(load_rules):
    load_rules([per_person_rule()])

    @revenium_metadata(subscriber={"email": OVER_CAP_EMAIL})
    def call_as_the_over_cap_developer():
        return litellm.completion(model=MODEL, messages=MESSAGES, mock_response="hi")

    with pytest.raises(BudgetExceededError) as blocked:
        call_as_the_over_cap_developer()

    assert type(blocked.value) is BudgetExceededError


@pytest.mark.parametrize("entry_point", list(ENTRY_POINTS))
def test_every_metered_entry_point_checks_before_the_provider(entry_point, breaker_on, check_raises):
    finish = start_entry_point(entry_point, {"model": MODEL, "messages": MESSAGES,
                                             "usage_metadata": {"subscriber": SUBSCRIBER}})

    with pytest.raises(BudgetExceededError):
        finish()

    assert [checked.get("subscriber") for checked in check_raises] == [SUBSCRIBER]


def test_guardrail_owned_request_is_left_to_the_guardrail(breaker_on, check_raises, monkeypatch):
    monkeypatch.setattr(mw, "proxy_meters_request", lambda _kwargs: True)

    response = litellm.completion(model=MODEL, messages=MESSAGES, mock_response="hi")

    assert response.choices[0].message.content == "hi"
    assert check_raises == []


def test_proxy_served_request_is_left_to_the_guardrail(breaker_on, check_raises):
    response = litellm.completion(model=MODEL, messages=MESSAGES, mock_response="hi", proxy_server_request={})

    assert response.choices[0].message.content == "hi"
    assert check_raises == []


def test_a_context_copied_inside_the_checked_call_is_checked_once_the_call_returns(load_rules):
    """An async LiteLLM success callback runs in a task that copies the
    call's context, so it must not inherit the outer check after the call."""
    load_rules([pooled_rule()])

    with enforcement.checked_by_outer_wrapper():
        enforcement.check_enforcement({})
        callback_context = contextvars.copy_context()

    with pytest.raises(BudgetExceededError):
        callback_context.run(enforcement.check_enforcement, {})


# ---- real LiteLLM over the real OpenAI SDK, both middlewares loaded ---------

class _CountingHandler(fake_openai_upstream._Handler):
    requests = 0

    def do_POST(self):
        type(self).requests += 1
        super().do_POST()


@pytest.fixture
def upstream():
    _CountingHandler.requests = 0
    server = ThreadingHTTPServer(("127.0.0.1", 0), _CountingHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield fake_openai_upstream.api_base(server)
    server.shutdown()
    server.server_close()


def openai_sdk_completion(api_base, email):
    return litellm.completion(model="openai/gpt-4o-mini", messages=MESSAGES, api_base=api_base, api_key="sk-test",
                              usage_metadata={"subscriber": {"email": email}})


@pytest.mark.parametrize("rule, email", [
    pytest.param(per_person_rule(), OVER_CAP_EMAIL, id="per_person"),
    pytest.param(pooled_rule(), UNDER_CAP_EMAIL, id="pooled"),
])
def test_a_blocked_call_through_the_openai_sdk_is_a_budget_error_and_never_sent(rule, email, upstream, load_rules):
    load_rules([rule])

    with pytest.raises(Exception) as blocked:
        openai_sdk_completion(upstream, email)

    assert type(blocked.value) is BudgetExceededError
    assert _CountingHandler.requests == 0


def test_an_allowed_call_through_the_openai_sdk_reaches_the_upstream(upstream, load_rules):
    load_rules([per_person_rule()])

    response = openai_sdk_completion(upstream, UNDER_CAP_EMAIL)

    assert response.choices[0].message.content == "hi"
    assert _CountingHandler.requests == 1


def test_router_retries_do_not_resend_a_blocked_call(upstream, load_rules):
    load_rules([pooled_rule()])
    router = litellm.Router(
        model_list=[{"model_name": "team-model",
                     "litellm_params": {"model": "openai/gpt-4o-mini", "api_base": upstream, "api_key": "sk-test"}}],
        num_retries=2,
    )

    with pytest.raises(Exception) as blocked:
        router.completion(model="team-model", messages=MESSAGES,
                          usage_metadata={"subscriber": {"email": UNDER_CAP_EMAIL}})

    assert type(blocked.value) is BudgetExceededError
    assert _CountingHandler.requests == 0


@pytest.mark.parametrize("is_async", [False, True], ids=["completion", "acompletion"])
@pytest.mark.parametrize("selective", [False, True], ids=["metered", "selective_unmetered"])
def test_the_call_identity_wins_over_the_decorator_through_the_openai_sdk(selective, is_async, upstream, load_rules,
                                                                          monkeypatch):
    if selective:
        monkeypatch.setenv("REVENIUM_SELECTIVE_METERING", "true")
    load_rules([per_person_rule()])

    @revenium_metadata(subscriber={"email": OVER_CAP_EMAIL})
    def call_as_the_under_cap_developer():
        if not is_async:
            return openai_sdk_completion(upstream, UNDER_CAP_EMAIL)
        return asyncio.run(litellm.acompletion(model="openai/gpt-4o-mini", messages=MESSAGES, api_base=upstream,
                                               api_key="sk-test",
                                               usage_metadata={"subscriber": {"email": UNDER_CAP_EMAIL}}))

    assert call_as_the_under_cap_developer().choices[0].message.content == "hi"
    assert _CountingHandler.requests == 1


def completion_from_langchain_settings(api_base, email):
    """A call whose caller is named only in LangChain's per-call settings,
    which the OpenAI wrapper reads from the ``config`` of a calling frame."""
    config = {"metadata": {"usage_metadata": {"subscriber": {"email": email}}}}  # noqa: F841
    return litellm.completion(model="openai/gpt-4o-mini", messages=MESSAGES, api_base=api_base, api_key="sk-test")


@pytest.mark.parametrize("selective", [False, True], ids=["metered", "selective_unmetered"])
def test_a_caller_named_only_in_langchain_settings_is_blocked_through_the_openai_sdk(selective, upstream, load_rules,
                                                                                     monkeypatch):
    if selective:
        monkeypatch.setenv("REVENIUM_SELECTIVE_METERING", "true")
    load_rules([per_person_rule()])

    with pytest.raises(Exception) as blocked:
        completion_from_langchain_settings(upstream, OVER_CAP_EMAIL)

    assert type(blocked.value) is BudgetExceededError
    assert _CountingHandler.requests == 0
    assert completion_from_langchain_settings(upstream, UNDER_CAP_EMAIL).choices[0].message.content == "hi"
    assert _CountingHandler.requests == 1


def test_a_proxy_served_call_keeps_the_openai_check(upstream, load_rules):
    """The client wrapper leaves a proxy-served call's cap check to the proxy,
    so without a Revenium guardrail the OpenAI wrapper still checks it."""
    load_rules([pooled_rule()])

    with pytest.raises(Exception, match="Team monthly cap"):
        litellm.completion(model="openai/gpt-4o-mini", messages=MESSAGES, api_base=upstream, api_key="sk-test",
                           proxy_server_request={})

    assert _CountingHandler.requests == 0


# ---- scoped caps (BACK-4176) -------------------------------------------------

@pytest.fixture
def check_keywords(monkeypatch):
    """Record the keywords each cap check receives, then refuse the call."""
    seen = []

    def check(usage_metadata=None, **keywords):
        seen.append(keywords)
        raise BudgetExceededError(message="recorded", rule_name="client-wiring")

    monkeypatch.setattr(mw, "check_enforcement", check, raising=False)
    return seen


@pytest.mark.parametrize("selective", [False, True], ids=["metered", "unmetered"])
@pytest.mark.parametrize("entry_point", list(ENTRY_POINTS))
def test_litellm_calls_are_checked_with_their_model_and_the_litellm_provider(
        entry_point, selective, breaker_on, check_keywords, monkeypatch):
    if selective:
        monkeypatch.setenv("REVENIUM_SELECTIVE_METERING", "true")

    finish = start_entry_point(entry_point, {"model": "openai/gpt-4o-mini", "messages": MESSAGES,
                                             "usage_metadata": {"subscriber": SUBSCRIBER}})

    with pytest.raises(BudgetExceededError):
        finish()

    assert check_keywords == [{"model": "openai/gpt-4o-mini", "provider": "LITELLM"}]


def provider_taking_a_positional_model(model, messages=None, **_kwargs):
    pytest.fail("provider reached")


@pytest.mark.parametrize("selective", [False, True], ids=["metered", "unmetered"])
def test_a_positional_model_reaches_the_cap_check(selective, breaker_on, check_keywords, monkeypatch):
    if selective:
        monkeypatch.setenv("REVENIUM_SELECTIVE_METERING", "true")

    with pytest.raises(BudgetExceededError):
        mw.completion_wrapper(provider_taking_a_positional_model, None, ("gpt-4o-mini",),
                              {"messages": MESSAGES})

    assert check_keywords == [{"model": "gpt-4o-mini", "provider": "LITELLM"}]


def test_an_agent_filtered_cap_lets_another_agent_through_litellm(load_rules):
    load_rules([{
        "ruleId": 4176, "name": "agent-a cap", "action": "BLOCK", "breached": True, "shadowMode": False,
        "filters": [{"dimension": "AGENT", "operator": "IS", "value": "agent-a"}],
    }])

    response = litellm.completion(model=MODEL, messages=MESSAGES, mock_response="hi",
                                  usage_metadata={"agent": "agent-b"})
    assert response.choices[0].message.content == "hi"

    with pytest.raises(BudgetExceededError):
        litellm.completion(model=MODEL, messages=MESSAGES, mock_response="hi",
                           usage_metadata={"agent": "agent-a"})


@pytest.mark.parametrize("requested, vendor, capped, refused", [
    pytest.param("gemini-flash-latest", "gemini", "gemini-3.8-flash", True, id="gemini-alias"),
    pytest.param("grok-code-fast-1", "xai", "grok-build-0.1", True, id="xai-short-name"),
    pytest.param("gemini-flash-latest", "gemini", "gemini-flash-latest", True, id="gemini-exact-name"),
    pytest.param("grok-code-fast-1", "ollama", "grok-build-0.1", False, id="other-vendor-not-refused"),
    pytest.param("gemini-flash-latest", "gemini", "gpt-4o-mini", False, id="other-model-not-refused"),
])
def test_a_bare_model_with_custom_llm_provider_is_judged_as_that_vendor_serves_it(
        requested, vendor, capped, refused, load_rules):
    """LiteLLM routes a bare model name by ``custom_llm_provider``, so a breached
    cap on the model the vendor serves stops the call before the provider is
    reached, and a model the vendor does not serve is left alone (BACK-4351,
    BACK-4352)."""
    load_rules([{
        "ruleId": 4351, "name": "served model cap", "action": "BLOCK", "breached": True, "shadowMode": False,
        "filters": [{"dimension": "MODEL", "operator": "IS", "value": capped}],
    }])

    def call():
        return litellm.completion(model=requested, custom_llm_provider=vendor, messages=MESSAGES,
                                  mock_response="hi")

    if refused:
        with pytest.raises(BudgetExceededError):
            call()
    else:
        assert call().choices[0].message.content == "hi"
