"""A breached cap applies only to the calls inside its filters (BACK-4176).

The server compiles each cap's scope into ``CompiledEnforcementRule.filters``
(dimension, operator, value, values) and counts only the requests that match
every entry. The SDK used to ignore the list, so a cap scoped to one agent,
model or provider, once breached, refused every call the team made.

These tests pin the SDK to the server's matcher, and to the places where the
SDK deliberately keeps a cap applying because it cannot judge the filter: a
model or provider it was not told, an unknown dimension or operator, and a
malformed entry. A filter is never allowed to weaken a cap.
"""
import logging

import pytest

from revenium_middleware._core import enforcement
from revenium_middleware._core.exceptions import BudgetExceededError

from .enforcement_stub_server import HEALTHY, EnforcementStub, point_enforcement_at, shut_down, wait_until

RULE_ID = 4176
RULE_NAME = "Scoped cap"
WARNING_TEXT = "is over its threshold"
UNEVALUABLE_TEXT = "cannot evaluate"
_ABSENT = object()


def f(dimension, operator, value=None, values=None):
    entry = {"dimension": dimension, "operator": operator, "value": value}
    if values is not None:
        entry["values"] = values
    return entry


def filtered_rule(filters=_ABSENT, action="BLOCK", **overrides):
    """A breached pooled rule whose window is still open."""
    rule = {
        "ruleId": RULE_ID,
        "name": RULE_NAME,
        "metricType": "TOTAL_COST",
        "threshold": 1.0,
        "currentValue": 2.5,
        "windowType": "DAILY",
        "windowEnd": "2999-01-01T00:00:00Z",
        "action": action,
        "breached": True,
        "shadowMode": False,
    }
    if filters is not _ABSENT:
        rule["filters"] = filters
    rule.update(overrides)
    return rule


def blocked(metadata=None, **keywords):
    """Run the pre-call check and say whether it refused the call."""
    try:
        enforcement.check_enforcement(metadata, **keywords)
    except BudgetExceededError as error:
        assert error.rule_id == RULE_ID
        return True
    return False


def agent(name):
    return {"agent": name}


@pytest.fixture
def cap(department_cache):
    """Install one rule as the whole cache."""
    def install(rule):
        department_cache([rule])
    return install


# ---- AC1: an agent filter --------------------------------------------------

def test_agent_filtered_block_refuses_the_matching_agent_and_lets_another_through(cap):
    cap(filtered_rule([f("AGENT", "IS", "agent-a")]))

    assert blocked(agent("agent-a"))
    assert not blocked(agent("agent-b"))


# ---- AC2: model and provider filters ---------------------------------------

def test_model_filtered_block_applies_only_to_the_filtered_model(cap):
    cap(filtered_rule([f("MODEL", "IS", "gpt-4o-mini")]))

    assert blocked({}, model="gpt-4o-mini", provider="OPENAI")
    assert not blocked({}, model="gpt-4.1", provider="OPENAI")


def test_provider_filtered_block_applies_only_to_the_filtered_provider(cap):
    cap(filtered_rule([f("PROVIDER", "IS", "openai")]))

    assert blocked({}, model="gpt-4o-mini", provider="OPENAI")
    assert not blocked({}, model="gpt-4o-mini", provider="LITELLM")


def test_a_dated_model_filter_holds_the_undated_request_model(cap):
    """OpenAI meters ``gpt-4o-mini-2024-07-18`` for a ``gpt-4o-mini`` request,
    so a cap authored from the dashboard's dated name covers that request."""
    cap(filtered_rule([f("MODEL", "IS", "gpt-4o-mini-2024-07-18")]))

    assert blocked({}, model="gpt-4o-mini", provider="OPENAI")
    assert not blocked({}, model="gpt-4o", provider="OPENAI")


def test_a_provider_prefix_on_the_request_model_is_ignored(cap):
    cap(filtered_rule([f("MODEL", "IS", "gpt-4o-mini")]))

    assert blocked({}, model="openai/gpt-4o-mini", provider="LITELLM")
    assert not blocked({}, model="openai/gpt-4.1", provider="LITELLM")


def test_a_latest_alias_is_held_to_a_filter_naming_it(cap):
    cap(filtered_rule([f("MODEL", "IS", "chatgpt-4o-latest")]))

    assert blocked({}, model="chatgpt-4o-latest", provider="OPENAI")
    assert not blocked({}, model="gpt-4o", provider="OPENAI")


@pytest.mark.parametrize("dated, alias, provider", [
    ("claude-3-5-sonnet-20241022", "claude-3-5-sonnet-latest", "ANTHROPIC"),
    ("claude-3-7-sonnet-20250219", "claude-3-7-sonnet-latest", "ANTHROPIC"),
])
def test_a_dated_model_filter_holds_its_latest_alias(cap, dated, alias, provider):
    """The provider meters a ``-latest`` request as the dated model it serves,
    so a cap authored from that dated name covers the alias (BACK-4248)."""
    cap(filtered_rule([f("MODEL", "IS", dated)]))

    assert blocked({}, model=alias, provider=provider)


def test_a_dated_model_filter_does_not_hold_another_models_latest_alias(cap):
    cap(filtered_rule([f("MODEL", "IS", "claude-3-5-haiku-20241022")]))

    assert not blocked({}, model="claude-3-5-sonnet-latest", provider="ANTHROPIC")


def test_a_router_model_name_with_a_slash_is_held_to_a_filter_naming_it_whole(cap):
    cap(filtered_rule([f("MODEL", "IS", "meta-llama/llama-3-8b")]))

    assert blocked({}, model="meta-llama/llama-3-8b", provider="OPENAI")
    assert not blocked({}, model="meta-llama/llama-3-70b", provider="OPENAI")


def test_is_not_an_undated_model_still_holds_that_request(cap):
    """The server compares the dated name it meters, which is never equal to
    the undated filter value, so it counts the call toward the cap."""
    cap(filtered_rule([f("MODEL", "IS_NOT", "gpt-4o-mini")]))

    assert blocked({}, model="gpt-4o-mini", provider="OPENAI")


def test_is_not_a_dated_model_exempts_that_exact_dated_request(cap):
    cap(filtered_rule([f("MODEL", "IS_NOT", "gpt-4o-mini-2024-07-18")]))

    assert not blocked({}, model="gpt-4o-mini-2024-07-18", provider="OPENAI")
    assert blocked({}, model="gpt-4o", provider="OPENAI")


def test_is_not_a_dated_model_exempts_that_request_behind_a_provider_prefix(cap):
    cap(filtered_rule([f("MODEL", "IS_NOT", "gpt-4o-mini-2024-07-18")]))

    assert not blocked({}, model="openai/gpt-4o-mini-2024-07-18", provider="LITELLM")
    assert blocked({}, model="openai/gpt-4o-2024-08-06", provider="LITELLM")


BEDROCK_HAIKU = enforcement.ModelKnownByAliases("anthropic.claude-haiku-4-5-20251001-v1:0", "claude-haiku-4-5-20251001")


def test_is_not_the_claude_name_still_holds_a_bedrock_request_recorded_under_its_bedrock_id(cap):
    """The server compares the Bedrock id the record carries, never the Claude
    name the SDK also knows the model by (BACK-4281)."""
    cap(filtered_rule([f("MODEL", "IS_NOT", "claude-haiku-4-5-20251001")]))

    assert blocked({}, model=BEDROCK_HAIKU, provider="AWS")


def test_is_not_the_recorded_bedrock_id_exempts_that_bedrock_request(cap):
    cap(filtered_rule([f("MODEL", "IS_NOT", "anthropic.claude-haiku-4-5-20251001-v1:0")]))

    assert not blocked({}, model=BEDROCK_HAIKU, provider="AWS")


def test_a_dash_dated_model_filter_holds_the_vertex_at_spelling(cap):
    """Vertex AI writes a Claude snapshot as ``claude-haiku-4-5@20251001`` (BACK-4307)."""
    cap(filtered_rule([f("MODEL", "IS", "claude-haiku-4-5-20251001")]))

    assert blocked({}, model="claude-haiku-4-5@20251001", provider="ANTHROPIC")
    assert blocked({}, model="vertex_ai/claude-haiku-4-5@20251001", provider="LITELLM")
    assert not blocked({}, model="claude-sonnet-4-5@20250929", provider="ANTHROPIC")


def test_a_vertex_at_spelled_filter_holds_both_spellings(cap):
    cap(filtered_rule([f("MODEL", "IS", "claude-haiku-4-5@20251001")]))

    assert blocked({}, model="claude-haiku-4-5@20251001", provider="ANTHROPIC")
    assert blocked({}, model="claude-haiku-4-5-20251001", provider="ANTHROPIC")
    assert not blocked({}, model="claude-sonnet-4-5-20250929", provider="ANTHROPIC")


def test_is_not_the_dash_spelling_still_holds_a_vertex_at_request(cap):
    cap(filtered_rule([f("MODEL", "IS_NOT", "claude-haiku-4-5-20251001")]))

    assert blocked({}, model="claude-haiku-4-5@20251001", provider="ANTHROPIC")


def test_an_ends_with_at_date_filter_does_not_hold_another_family_dated_that_day(cap):
    cap(filtered_rule([f("MODEL", "ENDS_WITH", "@20251001")]))

    assert blocked({}, model="claude-haiku-4-5@20251001", provider="ANTHROPIC")
    assert not blocked({}, model="claude-sonnet-4-5-20251001", provider="ANTHROPIC")


@pytest.mark.parametrize("operator, value", [
    ("IS", "gpt-4o-mini-2024-07-18"),
    ("CONTAINS", "gpt-4o-mini-2024-07-18"),
    ("CONTAINS", "mini-2024-07-18"),
    ("STARTS_WITH", "gpt-4o-mini-2024"),
    ("ENDS_WITH", "gpt-4o-mini-2024-07-18"),
    ("ENDS_WITH", "mini-20240718"),
])
def test_a_dated_filter_holds_the_same_undated_model(cap, operator, value):
    cap(filtered_rule([f("MODEL", operator, value)]))

    assert blocked({}, model="gpt-4o-mini", provider="OPENAI")


def test_a_dated_in_filter_holds_the_same_undated_model(cap):
    cap(filtered_rule([f("MODEL", "IN", values=["gpt-4.1", "gpt-4o-mini-2024-07-18"])]))

    assert blocked({}, model="gpt-4o-mini", provider="OPENAI")
    assert not blocked({}, model="gpt-4o", provider="OPENAI")


@pytest.mark.parametrize("operator, value, model", [
    ("CONTAINS", "gpt-4o-mini-2024-07-18", "gpt-4.1"),
    ("ENDS_WITH", "claude-3-5-sonnet-20241022", "gpt-4o"),
    ("STARTS_WITH", "gpt-4o-mini", "gpt-4o"),
    ("STARTS_WITH", "gpt-4o-mini-2024-07-18", "gpt-4o"),
    ("IS", "gpt-4o-mini-2024-07-18", "gpt-4o-mini-2024-08-01"),
    ("ENDS_WITH", "-2024-07-18", "gpt-4o-mini-2024-08-01"),
])
def test_a_dated_filter_does_not_hold_another_model(cap, operator, value, model):
    cap(filtered_rule([f("MODEL", operator, value)]))

    assert not blocked({}, model=model, provider="OPENAI")


def test_an_azure_deployment_name_never_rules_out_a_model_filter(cap):
    cap(filtered_rule([f("MODEL", "IS", "gpt-4o"), f("AGENT", "IS", "agent-a")]))

    assert blocked(agent("agent-a"), model="my-deployment", provider="Azure")
    assert not blocked(agent("agent-b"), model="my-deployment", provider="Azure")


def test_a_litellm_azure_deployment_never_rules_out_a_model_filter(cap):
    cap(filtered_rule([f("MODEL", "IS", "gpt-4o"), f("AGENT", "IS", "agent-a")]))

    assert blocked(agent("agent-a"), model="azure/my-deployment", provider="LITELLM")
    assert not blocked(agent("agent-b"), model="azure/my-deployment", provider="LITELLM")


@pytest.mark.parametrize("rule_filter, alias, provider", [
    (f("MODEL", "IS", "gemini-3.8-flash"), "gemini-flash-latest", "GOOGLE"),
    (f("MODEL", "IS", "gemini-3.5-flash-lite"), "gemini-flash-lite-latest", "GOOGLE"),
    (f("MODEL", "IS", "gemini-3.8-pro"), "gemini-pro-latest", "GOOGLE"),
    (f("MODEL", "IS", "gemini-3.8-flash"), "models/gemini-flash-latest", "GOOGLE"),
    (f("MODEL", "IS", "gemini-3.8-flash"), "publishers/google/models/gemini-flash-latest", "GOOGLE"),
    (f("MODEL", "IS", "gemini-3.8-flash"), "gemini/gemini-flash-latest", "LITELLM"),
    (f("MODEL", "IN", values=["gpt-4o", "gemini-3.8-flash"]), "gemini-flash-latest", "GOOGLE"),
    (f("MODEL", "IS", "gemini-2.5-flash-preview-09-2025"), "gemini-flash-latest", "GOOGLE"),
    (f("MODEL", "IS", "gemini-1.5-flash-002"), "gemini-flash-latest", "GOOGLE"),
])
def test_a_gemini_model_filter_holds_the_latest_alias_of_its_tier(cap, rule_filter, alias, provider):
    """Google meters a ``gemini-<tier>-latest`` request as the newest model of
    that tier, which moves with each release, so a cap on any version of the
    tier holds the alias (BACK-4351)."""
    cap(filtered_rule([rule_filter]))

    assert blocked({}, model=alias, provider=provider)


@pytest.mark.parametrize("rule_filter, model", [
    (f("MODEL", "IS", "gemini-3.5-flash-lite"), "gemini-flash-latest"),
    (f("MODEL", "IS", "gemini-3.8-flash"), "gemini-flash-lite-latest"),
    (f("MODEL", "IS", "gemini-3.8-flash"), "gemini-pro-latest"),
    (f("MODEL", "IS", "gemini-2.5-flash-preview-tts"), "gemini-flash-latest"),
    (f("MODEL", "IS", "gemini-2.5-flash-preview-native-audio-dialog"), "gemini-flash-latest"),
    (f("MODEL", "IS", "gemini-2.5-flash-preview-image-generation"), "gemini-flash-latest"),
    (f("MODEL", "IS", "gemini-2.5-flash-image"), "gemini-flash-latest"),
    (f("MODEL", "IS", "gemini-2.5-flash-image-preview"), "gemini-flash-latest"),
    (f("MODEL", "IS", "gemini-2.0-flash-exp"), "gemini-flash-latest"),
    (f("MODEL", "IS", "gemini-3.8-flash"), "chatgpt-4o-latest"),
    (f("MODEL", "IS", "gemini-3.8-flash"), "gpt-5-chat-latest"),
    (f("MODEL", "IS", "gemini-3.8-flash"), "claude-sonnet-latest"),
    (f("MODEL", "IS", "gemini-3.8-flash"), "grok-4.20-reasoning-latest"),
    (f("MODEL", "IS", "claude-opus-4-8"), "gemini-flash-latest"),
])
def test_a_gemini_model_filter_does_not_hold_another_tiers_latest_alias(cap, rule_filter, model):
    cap(filtered_rule([rule_filter]))

    assert not blocked({}, model=model, provider="GOOGLE")


def test_a_gemini_model_filter_still_holds_the_model_named_directly(cap):
    cap(filtered_rule([f("MODEL", "IS", "gemini-3.8-flash")]))
    assert blocked({}, model="gemini-3.8-flash", provider="GOOGLE")

    cap(filtered_rule([f("MODEL", "IS_NOT", "gemini-3.8-flash")]))
    assert blocked({}, model="gemini-flash-latest", provider="GOOGLE")


@pytest.mark.parametrize("operator, value", [("STARTS_WITH", "gemini-3"), ("CONTAINS", "3.8-flash")])
def test_a_substring_gemini_model_filter_does_not_hold_a_latest_alias(cap, operator, value):
    """A documented limit (README, cost controls filters): the model Google
    serves for the alias is unknown before the call, so only a filter naming a
    whole Gemini model can be judged against it (BACK-4351)."""
    cap(filtered_rule([f("MODEL", operator, value)]))

    assert not blocked({}, model="gemini-flash-latest", provider="GOOGLE")


XAI_PUBLISHED_ALIASES = [
    ("grok-4.20-0309-non-reasoning", "grok-4.20-non-reasoning"),
    ("grok-4.20-0309-non-reasoning", "grok-4.20-non-reasoning-latest"),
    ("grok-4.20-0309-non-reasoning", "grok-4.20-beta-non-reasoning"),
    ("grok-4.20-0309-non-reasoning", "grok-4.20-beta-latest-non-reasoning"),
    ("grok-4.20-0309-non-reasoning", "grok-4.20-experimental-beta-0304-non-reasoning"),
    ("grok-4.20-0309-non-reasoning", "grok-4.20-experimental-beta-non-reasoning-latest"),
    ("grok-4.20-0309-non-reasoning", "grok-4.20-beta-0309-non-reasoning"),
    ("grok-4.20-0309-non-reasoning", "grok-4.20-non-reasoning-gv2"),
    ("grok-4.20-0309-reasoning", "grok-4.20-reasoning-latest"),
    ("grok-4.20-0309-reasoning", "grok-4.20"),
    ("grok-4.20-0309-reasoning", "grok-4.20-reasoning"),
    ("grok-4.20-0309-reasoning", "grok-4.20-0309"),
    ("grok-4.20-0309-reasoning", "grok-4.20-beta-0309-reasoning"),
    ("grok-4.20-0309-reasoning", "grok-4.20-beta"),
    ("grok-4.20-0309-reasoning", "grok-4.20-beta-0309"),
    ("grok-4.20-0309-reasoning", "grok-4.20-beta-latest"),
    ("grok-4.20-0309-reasoning", "grok-4.20-beta-latest-reasoning"),
    ("grok-4.20-0309-reasoning", "grok-4.20-beta-reasoning"),
    ("grok-4.20-0309-reasoning", "grok-4.20-experimental-beta-0304-reasoning"),
    ("grok-4.20-0309-reasoning", "grok-4.20-experimental-beta-0304"),
    ("grok-4.20-0309-reasoning", "grok-4.20-experimental-beta-reasoning-latest"),
    ("grok-4.20-0309-reasoning", "grok-4.20-experimental-beta-latest"),
    ("grok-4.20-0309-reasoning", "grok-4.20-reasoning-gv2"),
    ("grok-4.20-multi-agent-0309", "grok-4.20-multi-agent"),
    ("grok-4.20-multi-agent-0309", "grok-4.20-multi-agent-latest"),
    ("grok-4.20-multi-agent-0309", "grok-4.20-multi-agent-beta-latest"),
    ("grok-4.20-multi-agent-0309", "grok-4.20-multi-agent-experimental-beta-0304"),
    ("grok-4.20-multi-agent-0309", "grok-4.20-multi-agent-experimental-beta-latest"),
    ("grok-4.20-multi-agent-0309", "grok-4.20-multi-agent-beta-0309"),
    ("grok-4.3", "grok-4.3-latest"),
    ("grok-4.5", "grok-4.5-latest"),
    ("grok-4.5", "grok-build-latest"),
    ("grok-build-0.1", "grok-code-fast-1"),
    ("grok-build-0.1", "grok-code-fast"),
    ("grok-build-0.1", "grok-code-fast-1-0825"),
]


@pytest.mark.parametrize("served, alias, provider", [
    *[(served, alias, "XAI") for served, alias in XAI_PUBLISHED_ALIASES],
    ("grok-4.20-0309-reasoning", "xai/grok-4.20", "LITELLM"),
])
def test_an_xai_model_filter_holds_every_short_name_xai_publishes_for_it(cap, served, alias, provider):
    """xAI answers and meters a short name as the full model id it lists the
    name under, so a cap on that id holds every name xAI publishes for it
    (BACK-4352). The list is xAI's GET /v1/language-models of 2026-10-11."""
    cap(filtered_rule([f("MODEL", "IS", served)]))

    assert blocked({}, model=alias, provider=provider)


@pytest.mark.parametrize("capped, model", [
    ("grok-4.20-0309-non-reasoning", "grok-4.20"),
    ("grok-build-0.1", "grok-4.20"),
    ("grok-4.20-0309-reasoning", "grok-4.20-non-reasoning"),
    ("grok-4.20-0309-reasoning", "grok-code-fast-1"),
    ("grok-build-0.1", "grok-build-latest"),
    ("grok-4.20-multi-agent-0309", "grok-4.20"),
])
def test_an_xai_model_filter_does_not_hold_another_grok_models_short_name(cap, capped, model):
    cap(filtered_rule([f("MODEL", "IS", capped)]))

    assert not blocked({}, model=model, provider="XAI")


def test_an_xai_model_filter_still_holds_the_full_model_name(cap):
    cap(filtered_rule([f("MODEL", "IS", "grok-4.20-0309-reasoning")]))

    assert blocked({}, model="grok-4.20-0309-reasoning", provider="XAI")


@pytest.mark.parametrize("capped, model, provider", [
    ("grok-build-0.1", "grok-code-fast-1", "OLLAMA"),
    ("grok-4.20-0309-reasoning", "grok-4.20", "OLLAMA"),
    ("grok-4.20-0309-reasoning", "openrouter/x-ai/grok-4.20", "LITELLM"),
    ("grok-4.20-0309-reasoning", "ollama/grok-4.20", "LITELLM"),
    ("gemini-3.8-flash", "gemini-flash-latest", "OLLAMA"),
    ("gemini-3.8-flash", "gemini-flash-latest", "OPENAI"),
    ("gemini-3.8-flash", "ollama/gemini-flash-latest", "LITELLM"),
])
def test_a_vendor_alias_is_held_only_on_a_call_to_that_vendor(cap, capped, model, provider):
    """A vendor's short name means that vendor's model only when the call goes
    to that vendor: Ollama records its own model name and resolves no xAI or
    Google alias, so a local model that shares the name is not the capped one
    (BACK-4351, BACK-4352)."""
    cap(filtered_rule([f("MODEL", "IS", capped)]))

    assert not blocked({}, model=model, provider=provider)


@pytest.mark.parametrize("capped, model", [
    ("grok-build-0.1", "grok-code-fast-1"),
    ("gemini-3.8-flash", "gemini-flash-latest"),
])
def test_a_vendor_alias_is_held_when_the_provider_is_unknown(cap, capped, model):
    """A call whose provider the SDK was not told keeps the cap applying."""
    cap(filtered_rule([f("MODEL", "IS", capped)]))

    assert blocked({}, model=model, provider=None)


@pytest.mark.parametrize("capped, model", [
    ("grok-build-0.1", "grok-code-fast-1"),
    ("gemini-3.8-flash", "gemini-flash-latest"),
])
def test_a_vendor_alias_is_held_on_a_bare_litellm_model(cap, capped, model):
    """A LiteLLM call naming a bare model may still go to the vendor, which
    LiteLLM picks from the name or ``custom_llm_provider``, so the cap keeps
    applying unless a route names another vendor (BACK-4351, BACK-4352)."""
    cap(filtered_rule([f("MODEL", "IS", capped)]))

    assert blocked({}, model=model, provider="LITELLM")


# ---- AC3: several filters, IN and the other operators ----------------------

def test_a_rule_with_several_filters_applies_only_when_every_filter_matches(cap):
    cap(filtered_rule([f("AGENT", "IS", "agent-a"), f("MODEL", "IS", "gpt-4o-mini")]))

    assert blocked(agent("agent-a"), model="gpt-4o-mini", provider="OPENAI")
    assert not blocked(agent("agent-a"), model="gpt-4.1", provider="OPENAI")
    assert not blocked(agent("agent-b"), model="gpt-4o-mini", provider="OPENAI")


def test_in_filter_reads_values_not_the_joined_value(cap):
    """The server also writes the comma-joined list into ``value`` for IN."""
    cap(filtered_rule([f("AGENT", "IN", "agent-a,agent-c", ["agent-a", "agent-c"])]))

    assert blocked(agent("agent-c"))
    assert not blocked(agent("agent-b"))
    assert not blocked(agent("agent-a,agent-c"))


@pytest.mark.parametrize("operator, value, inside, outside", [
    ("IS_NOT", "agent-a", "agent-b", "agent-a"),
    ("CONTAINS", "-research-", "team-research-bot", "team-Research-bot"),
    ("STARTS_WITH", "team-", "team-bot", "Team-bot"),
    ("ENDS_WITH", "-bot", "team-bot", "team-BOT"),
])
def test_scalar_operators_follow_the_server(cap, operator, value, inside, outside):
    cap(filtered_rule([f("AGENT", operator, value)]))

    assert blocked(agent(inside))
    assert not blocked(agent(outside))


def test_subscriber_filter_matches_email_canonically_and_id_exactly(cap):
    cap(filtered_rule([f("SUBSCRIBER", "IS", "dev@example.com")]))
    assert blocked({"subscriber": {"email": " Dev@Example.COM "}})
    assert not blocked({"subscriber": {"email": "other@example.com"}})

    cap(filtered_rule([f("SUBSCRIBER", "IS", "sub-1")]))
    assert blocked({"subscriber": {"id": "sub-1"}})
    assert not blocked({"subscriber": {"id": "SUB-1"}})


def test_subscriber_substring_operators_ignore_case_on_the_email(cap):
    cap(filtered_rule([f("SUBSCRIBER", "ENDS_WITH", "@Example.com")]))

    assert blocked({"subscriber": {"email": "dev@EXAMPLE.COM"}})
    assert not blocked({"subscriber": {"email": "dev@elsewhere.com"}})


def test_credential_filter_reads_the_credential_name_before_its_value(cap):
    cap(filtered_rule([f("CREDENTIAL", "IS", "team-key")]))

    assert blocked({"subscriber": {"id": "s", "credential": {"name": "team-key", "value": "v"}}})
    assert not blocked({"subscriber": {"id": "s", "credential": {"name": "other-key", "value": "team-key"}}})


@pytest.mark.parametrize("key", ["credential_alias", "credentialAlias"])
def test_a_credential_alias_in_the_metadata_is_the_credential_name(cap, monkeypatch, key):
    """Metering sends the alias as ``subscriber.credential.name``."""
    monkeypatch.delenv("REVENIUM_CREDENTIAL_ALIAS", raising=False)
    cap(filtered_rule([f("CREDENTIAL", "IS", "prod-key")]))

    assert blocked({key: "prod-key"})
    assert not blocked({key: "staging-key"})


def test_the_credential_alias_environment_variable_is_the_credential_name(cap, monkeypatch):
    cap(filtered_rule([f("CREDENTIAL", "IS", "prod-key")]))

    monkeypatch.setenv("REVENIUM_CREDENTIAL_ALIAS", "prod-key")
    assert blocked({})
    monkeypatch.setenv("REVENIUM_CREDENTIAL_ALIAS", "staging-key")
    assert not blocked({})


def test_a_named_credential_wins_over_the_alias(cap, monkeypatch):
    monkeypatch.setenv("REVENIUM_CREDENTIAL_ALIAS", "prod-key")
    cap(filtered_rule([f("CREDENTIAL", "IS", "prod-key")]))

    assert not blocked({"credential_alias": "prod-key", "subscriber": {"credential": {"name": "other-key"}}})


def test_the_alias_is_not_the_credential_of_a_named_subscriber(cap, monkeypatch):
    monkeypatch.setenv("REVENIUM_CREDENTIAL_ALIAS", "prod-key")
    cap(filtered_rule([f("CREDENTIAL", "IS", "prod-key")]))

    assert not blocked({"subscriber": {"id": "s"}})


@pytest.mark.parametrize("dimension, key", [
    ("TASK_TYPE", "task_type"),
    ("TASK_TYPE", "taskType"),
    ("PRODUCT", "product_name"),
    ("PRODUCT", "productName"),
    ("ORGANIZATION", "organization_name"),
    ("ORGANIZATION", "organizationName"),
])
def test_task_type_product_and_organization_filters_read_the_metered_fields(cap, dimension, key):
    cap(filtered_rule([f(dimension, "IS", "wanted")]))

    assert blocked({key: "wanted"})
    assert not blocked({key: "other"})


# ---- AC4: no filters --------------------------------------------------------

@pytest.mark.parametrize("filters", [_ABSENT, [], None], ids=["missing", "empty", "null"])
def test_a_rule_without_filters_blocks_every_call_as_before(cap, filters):
    """Regression guard: passes before and after the fix."""
    cap(filtered_rule(filters))

    assert blocked(agent("agent-a"))
    assert blocked(agent("agent-b"))
    assert blocked({})
    assert blocked(None)


# ---- AC5: warn-only ---------------------------------------------------------

def threshold_warnings(caplog):
    return [record for record in caplog.records if WARNING_TEXT in record.getMessage()]


def test_a_filtered_warn_only_rule_logs_nothing_for_a_non_matching_call(cap, caplog):
    cap(filtered_rule([f("AGENT", "IS", "agent-a")], action="WARN_ONLY"))

    with caplog.at_level(logging.WARNING, logger="revenium_middleware.extension"):
        # The non-matching calls go first: the warning is logged once per
        # window, so an agent-a call first would hide a wrong verdict for them.
        for _ in range(3):
            assert not blocked(agent("agent-b"))
        assert threshold_warnings(caplog) == []

        assert not blocked(agent("agent-a"))
    assert len(threshold_warnings(caplog)) == 1


# ---- AC7: a filter the SDK cannot judge never weakens the cap --------------

@pytest.mark.parametrize("dimension", ["WORKSPACE", "API_KEY", "DEPARTMENT", "ORG_UNIT", "FUTURE_DIM"])
def test_an_unknown_dimension_keeps_the_rule_applying(cap, dimension):
    cap(filtered_rule([f(dimension, "IS", "unknowable"), f("AGENT", "IS", "agent-a")]))

    assert blocked(agent("agent-a"))
    assert not blocked(agent("agent-b"))


def test_a_department_filter_on_a_per_person_rule_keeps_the_rule_applying(cap):
    """A per-person cap scoped to one department, sent without ``departmentId``."""
    cap(filtered_rule(
        [f("DEPARTMENT", "IS", "dept-1"), f("AGENT", "IS", "agent-a")],
        groupBy="SUBSCRIBER",
        breached=False,
        groupBreakdown=[{"groupValue": "sub-a", "currentValue": 9.0, "breached": True}],
    ))

    assert blocked({"agent": "agent-a", "subscriber": {"id": "sub-a"}})
    assert not blocked({"agent": "agent-b", "subscriber": {"id": "sub-a"}})


def test_an_unknown_operator_keeps_the_rule_applying(cap):
    cap(filtered_rule([f("AGENT", "REGEX", "x"), f("MODEL", "IS", "gpt-4o-mini")]))

    assert blocked(agent("y"), model="gpt-4o-mini", provider="OPENAI")
    assert not blocked(agent("y"), model="gpt-4.1", provider="OPENAI")


def test_a_call_missing_a_metadata_value_is_outside_an_is_filter(cap):
    """The server never counts an untagged call toward an agent's cap."""
    cap(filtered_rule([f("AGENT", "IS", "agent-a")]))

    assert not blocked({})
    assert not blocked(None)
    assert blocked(agent("agent-a"))


def test_a_call_missing_a_metadata_value_is_inside_an_is_not_filter(cap):
    cap(filtered_rule([f("AGENT", "IS_NOT", "agent-a")]))

    assert blocked({})
    assert not blocked(agent("agent-a"))


def test_a_non_string_call_value_keeps_the_rule_applying(cap):
    cap(filtered_rule([f("AGENT", "IS", "agent-a")]))

    assert blocked({"agent": 7})
    assert not blocked(agent("agent-b"))


def test_model_and_provider_not_passed_keep_the_rule_applying(cap):
    cap(filtered_rule([f("MODEL", "IS", "gpt-4o-mini"), f("PROVIDER", "IS", "openai"),
                       f("AGENT", "IS", "agent-a")]))

    assert blocked(agent("agent-a"))
    assert not blocked(agent("agent-b"))


@pytest.mark.parametrize("unusable", [f("MODEL", "IS", None), f("MODEL", "IN", None, [])],
                         ids=["scalar-without-value", "in-without-values"])
def test_filters_without_a_usable_value_constrain_nothing(cap, unusable):
    cap(filtered_rule([unusable, f("AGENT", "IS", "agent-a")]))

    assert blocked(agent("agent-a"), model="gpt-4.1", provider="OPENAI")
    assert not blocked(agent("agent-b"), model="gpt-4.1", provider="OPENAI")


@pytest.mark.parametrize("filters", ["x", {}], ids=["string", "mapping"])
def test_filters_that_are_not_a_list_keep_the_rule_applying(cap, filters):
    """Regression guard: a malformed list never crashes and never exempts a call."""
    cap(filtered_rule(filters))

    assert blocked(agent("agent-b"))


@pytest.mark.parametrize("malformed", [
    "AGENT IS agent-b",
    None,
    {"dimension": "AGENT", "operator": "IS", "value": 5},
    {"dimension": None, "operator": "IS", "value": "agent-b"},
    {"dimension": "AGENT", "operator": None, "value": "agent-b"},
    {"dimension": "AGENT", "operator": "IN", "values": "agent-b"},
], ids=["string-entry", "null-entry", "number-value", "no-dimension", "no-operator", "values-not-a-list"])
def test_malformed_filters_fail_toward_enforcing(cap, malformed):
    cap(filtered_rule([malformed, f("AGENT", "IS", "agent-a")]))

    assert blocked(agent("agent-a"))
    assert not blocked(agent("agent-b"))


def test_an_unevaluable_filter_logs_one_debug_line_per_rule(cap, caplog):
    cap(filtered_rule([f("WORKSPACE", "IS", "workspace-secret"), f("AGENT", "IS", "agent-a")]))

    with caplog.at_level(logging.DEBUG, logger="revenium_middleware.extension"):
        for _ in range(3):
            assert blocked({"agent": "agent-a", "subscriber": {"email": "person@example.com"}})

    lines = [record.getMessage() for record in caplog.records if UNEVALUABLE_TEXT in record.getMessage()]
    assert len(lines) == 1
    assert "WORKSPACE" in lines[0]
    for private in ("workspace-secret", "agent-a", "person@example.com"):
        assert private not in lines[0]


def test_a_refresh_forgets_the_unevaluable_filters_of_a_rule_no_longer_served(cap, monkeypatch):
    cap(filtered_rule([f("WORKSPACE", "IS", "w")]))
    blocked({})
    assert enforcement._noted_unevaluable_filters

    monkeypatch.setattr(enforcement, "_cached_rules", [filtered_rule(ruleId=RULE_ID + 1)])
    enforcement._begin_cache_generation()

    assert not enforcement._noted_unevaluable_filters


def test_a_refresh_still_serving_the_rule_keeps_its_unevaluable_filters_noted(cap):
    cap(filtered_rule([f("WORKSPACE", "IS", "w")]))
    blocked({})
    noted = set(enforcement._noted_unevaluable_filters)

    enforcement._begin_cache_generation()

    assert enforcement._noted_unevaluable_filters == noted


# ---- interactions with the rest of the rule loop ---------------------------

def test_per_person_rule_with_an_agent_filter_blocks_only_the_breached_person_on_the_matching_agent(cap):
    cap(filtered_rule(
        [f("AGENT", "IS", "agent-a")],
        groupBy="SUBSCRIBER",
        breached=False,
        groupBreakdown=[
            {"groupValue": "sub-a", "currentValue": 9.0, "breached": True},
            {"groupValue": "sub-b", "currentValue": 0.1, "breached": False},
        ],
    ))

    assert blocked({"agent": "agent-a", "subscriber": {"id": "sub-a"}})
    assert not blocked({"agent": "agent-b", "subscriber": {"id": "sub-a"}})
    assert not blocked({"agent": "agent-a", "subscriber": {"id": "sub-b"}})


def test_legacy_credential_mismatch_still_skips_a_filtered_rule(cap):
    """Regression guard: the legacy ``credential`` shim is unchanged."""
    cap(filtered_rule([f("AGENT", "IS", "agent-a")], credential="x"))

    assert not blocked({"agent": "agent-a", "subscriber_credential": "y"})


def test_a_department_rule_with_filters_is_still_decided_by_the_department_map(department_cache):
    """Regression guard: department rules stay decided by the server's map."""
    email = "dept-user@example.test"
    rule = filtered_rule([f("AGENT", "IS", "agent-a")], departmentId="dept-1")
    department_cache([rule], {email: RULE_ID})

    assert blocked({"agent": "agent-b", "subscriber": {"email": email}})


def test_filters_survive_the_disk_snapshot_round_trip(department_snapshot_dir, monkeypatch):
    monkeypatch.setenv("REVENIUM_CIRCUIT_BREAKER_ENABLED", "true")
    monkeypatch.delenv("REVENIUM_BYPASS", raising=False)
    monkeypatch.delenv("REVENIUM_CB_FAIL_MODE", raising=False)
    monkeypatch.setattr(enforcement, "_ensure_poller_running", lambda: None)
    enforcement._persist_cache_to_disk([filtered_rule([f("AGENT", "IS", "agent-a")])])

    assert not blocked(agent("agent-b"))
    assert blocked(agent("agent-a"))


# ---- AC6, unit half: a filtered rule served by a real endpoint --------------

@pytest.fixture
def restore_the_cache(monkeypatch):
    """Put back whatever the poller installs, so no rule outlives its test."""
    for name in ("_cached_rules", "_cached_org_unit_blocks", "_cached_org_unit_block_balances",
                 "_cached_org_unit_warnings", "_cached_org_unit_candidates", "_cache_timestamp",
                 "_cache_initialized"):
        monkeypatch.setattr(enforcement, name, getattr(enforcement, name))


@pytest.fixture
def endpoint(monkeypatch, restore_the_cache):
    stubs = []

    def start(rules):
        stub = EnforcementStub(HEALTHY, rules)
        stubs.append(stub)
        point_enforcement_at(monkeypatch, stub)
        return stub

    yield start
    for stub in stubs:
        shut_down(stub)


def test_a_filtered_rule_served_by_the_endpoint_blocks_only_its_agent(endpoint):
    endpoint([filtered_rule([f("AGENT", "IS", "agent-a")])])

    enforcement.check_enforcement(agent("agent-b"))  # starts the poller; nothing is cached yet
    assert wait_until(lambda: enforcement._cache_initialized)

    assert blocked(agent("agent-a"))
    assert not blocked(agent("agent-b"))


# ---- provider aliases: Claude served through Foundry is Anthropic spend ----

SERVER_PROVIDER_ALIASES = {
    "FOUNDRY": {"FOUNDRY", "ANTHROPIC"},
    "ClaudeCode": {"ClaudeCode", "ANTHROPIC"},
    "ClaudeCowork": {"ClaudeCowork", "ANTHROPIC"},
    "CodexCli": {"CodexCli", "OPENAI"},
    "GeminiCli": {"GeminiCli", "GOOGLE"},
}
CLAUDE = "claude-haiku-4-5"


def provider_blocked(provider):
    return blocked({}, model=CLAUDE, provider=provider)


@pytest.mark.parametrize("label", ["Foundry", "FOUNDRY", "foundry"])
def test_a_provider_is_anthropic_cap_holds_a_foundry_call(cap, label):
    cap(filtered_rule([f("PROVIDER", "IS", "ANTHROPIC")]))

    assert provider_blocked(label)


@pytest.mark.parametrize("values", [["ANTHROPIC"], ["OPENAI", "anthropic"]])
def test_a_provider_in_anthropic_cap_holds_a_foundry_call(cap, values):
    cap(filtered_rule([f("PROVIDER", "IN", ",".join(values), values)]))

    assert provider_blocked("Foundry")


def test_a_provider_is_not_anthropic_cap_lets_a_foundry_call_through(cap):
    cap(filtered_rule([f("PROVIDER", "IS_NOT", "ANTHROPIC")]))

    assert not provider_blocked("Foundry")


def test_a_provider_foundry_cap_stays_narrow(cap):
    """Anthropic spend is not Foundry spend: the alias runs one way only."""
    for rule_filter in (f("PROVIDER", "IS", "FOUNDRY"), f("PROVIDER", "IN", "FOUNDRY", ["FOUNDRY"])):
        cap(filtered_rule([rule_filter]))
        assert provider_blocked("Foundry")
        assert not provider_blocked("ANTHROPIC")

    cap(filtered_rule([f("PROVIDER", "IS_NOT", "FOUNDRY")]))
    assert provider_blocked("ANTHROPIC")
    assert not provider_blocked("Foundry")


@pytest.mark.parametrize("rule_filter, refused", [
    (f("PROVIDER", "IS", "ANTHROPIC"), True),
    (f("PROVIDER", "IN", "ANTHROPIC", ["ANTHROPIC"]), True),
    (f("PROVIDER", "IS_NOT", "ANTHROPIC"), False),
], ids=["is", "in", "is_not"])
def test_direct_anthropic_calls_are_judged_as_before_under_anthropic_provider_caps(cap, rule_filter, refused):
    cap(filtered_rule([rule_filter]))

    assert provider_blocked("ANTHROPIC") is refused


def test_provider_substring_filters_ignore_the_alias(cap):
    """An alias says who the provider is, not which characters its label has."""
    cap(filtered_rule([f("PROVIDER", "CONTAINS", "anthrop")]))
    assert not provider_blocked("Foundry")

    cap(filtered_rule([f("PROVIDER", "STARTS_WITH", "found")]))
    assert provider_blocked("Foundry")


@pytest.mark.parametrize("value, label, refused", [
    ("ANTHROPIC", "AWS", False),
    ("OPENAI", "Azure", False),
    ("GOOGLE", "Google", True),
])
def test_hosting_labels_are_not_aliased(cap, value, label, refused):
    cap(filtered_rule([f("PROVIDER", "IS", value)]))

    assert provider_blocked(label) is refused


def test_provider_aliases_match_the_servers_provider_vocabulary():
    """The SDK's copy of the server's alias table, from hypercurrent develop
    843ba8a9f6 ``commons/.../ProviderVocabulary.kt`` and
    ``CodingAssistantProviderIds.kt``. This repo cannot read the server, so
    the test catches the SDK drifting from that copy, not the server moving."""
    server = {key.upper(): {value.upper() for value in bucket} for key, bucket in SERVER_PROVIDER_ALIASES.items()}

    assert enforcement._PROVIDER_ALIAS_BUCKETS == server


ALIAS_PAIRS = [
    pytest.param(key, target, id=f"{key}->{target}")
    for key, bucket in SERVER_PROVIDER_ALIASES.items() for target in sorted(bucket) if target != key
]


@pytest.mark.parametrize("label, target", ALIAS_PAIRS)
def test_every_server_alias_is_honoured_one_way(cap, label, target):
    cap(filtered_rule([f("PROVIDER", "IS", target)]))
    assert provider_blocked(label)
    cap(filtered_rule([f("PROVIDER", "IN", target, [target])]))
    assert provider_blocked(label)
    cap(filtered_rule([f("PROVIDER", "IS_NOT", target)]))
    assert not provider_blocked(label)

    cap(filtered_rule([f("PROVIDER", "IS", label)]))
    assert not provider_blocked(target)


def test_the_provider_alias_applies_to_the_provider_dimension_only(cap):
    cap(filtered_rule([f("AGENT", "IS", "ANTHROPIC")]))
    assert not blocked(agent("Foundry"))

    cap(filtered_rule([f("AGENT", "IS_NOT", "ANTHROPIC")]))
    assert blocked(agent("Foundry"))
