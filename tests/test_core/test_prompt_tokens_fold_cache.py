"""Which LiteLLM upstreams fold the cache into ``prompt_tokens`` (BACK-4162).

LiteLLM's Bedrock routes (Converse, invoke, mantle) and its Vertex AI Claude
route build ``prompt_tokens`` as uncached input plus cache read plus cache
write, exactly as its direct Anthropic route does. Left gross, the cache is
billed twice: once at the input rate and again at the cache rates. Gemini on
Vertex AI is not Claude, so it is left as LiteLLM built it. On Bedrock the
conversion LiteLLM itself picked for the call decides, not the model name, so
an application inference profile ARN and Nova are netted like Claude.

The model that decides is read only from the fields LiteLLM fills (``model``
and ``litellm_model_name``), never from the caller's request metadata.
"""
import importlib.util

import pytest

from revenium_middleware._core import cache_tokens
from revenium_middleware._core.cache_tokens import prompt_tokens_fold_cache

BEDROCK_ARN = "arn:aws:bedrock:us-east-1:123456789012:application-inference-profile/abc123"
BEDROCK_CLAUDE = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
VERTEX_CLAUDE = "claude-haiku-4-5@20251001"

requires_litellm = pytest.mark.skipif(
    importlib.util.find_spec("litellm") is None, reason="asks the real LiteLLM routing")


@pytest.mark.parametrize("sources", [
    pytest.param(({"custom_llm_provider": "bedrock", "model": BEDROCK_CLAUDE},), id="bedrock-converse"),
    pytest.param(({"custom_llm_provider": "bedrock",
                   "litellm_model_name": "bedrock/invoke/anthropic.claude-3-5-haiku-20241022-v1:0"},),
                 id="bedrock-invoke-litellm-model-name"),
    pytest.param(({"custom_llm_provider": "vertex_ai", "model": VERTEX_CLAUDE},), id="vertex-claude"),
    pytest.param(({"custom_llm_provider": "BEDROCK", "model": "US.ANTHROPIC.CLAUDE-HAIKU-4-5"},), id="upper-case"),
    pytest.param(({"custom_llm_provider": "bedrock"}, {"litellm_model_name": "bedrock/" + BEDROCK_CLAUDE}),
                 id="model-on-a-later-source"),
    pytest.param(({"model": BEDROCK_CLAUDE}, {"custom_llm_provider": "bedrock"}),
                 id="provider-on-a-later-source"),
    pytest.param((None, {"custom_llm_provider": "vertex_ai"}, {"model": VERTEX_CLAUDE}),
                 id="missing-source-skipped"),
])
def test_claude_on_bedrock_and_vertex_folds(sources):
    assert prompt_tokens_fold_cache(*sources) is True


@pytest.mark.parametrize("sources", [
    pytest.param(({"custom_llm_provider": "vertex_ai", "model": "gemini-2.5-flash-lite"},), id="vertex-gemini"),
    pytest.param(({"custom_llm_provider": "bedrock"},), id="bedrock-without-a-model"),
    pytest.param(({"custom_llm_provider": "openai", "model": "claude-proxy-alias"},), id="openai-named-claude"),
])
def test_non_claude_models_on_the_same_providers_stay_gross(sources):
    assert prompt_tokens_fold_cache(*sources) is False


@pytest.mark.parametrize("model", [
    pytest.param(BEDROCK_ARN, id="bare-arn"),
    pytest.param("bedrock/" + BEDROCK_ARN, id="bedrock-prefixed-arn"),
    pytest.param("converse/" + BEDROCK_ARN, id="converse-arn"),
    pytest.param("us.amazon.nova-micro-v1:0", id="nova-converse"),
    pytest.param("converse/amazon.nova-pro-v1:0", id="nova-explicit-converse"),
    pytest.param("invoke/us.amazon.nova-pro-v1:0", id="nova-invoke"),
    pytest.param("arn:aws:bedrock:us-east-1:123456789012:custom-model/amazon.nova-micro-v1:0/xyz", id="nova-custom-model-arn"),
    pytest.param("invoke/anthropic/" + BEDROCK_ARN, id="invoke-anthropic-arn"),
])
@requires_litellm
def test_bedrock_usage_litellm_folds_is_netted_whatever_the_model(model):
    assert prompt_tokens_fold_cache({"custom_llm_provider": "bedrock", "model": model}) is True
    assert prompt_tokens_fold_cache({"custom_llm_provider": "bedrock", "litellm_model_name": model}) is True


@pytest.mark.parametrize("model, expected", [
    pytest.param("invoke/us.meta.llama3-3-70b-instruct-v1:0", False, id="llama-invoke"),
    pytest.param("chat_completions/openai.gpt-oss-120b-1:0", False, id="chat-completions"),
    pytest.param("openai/openai.gpt-oss-120b-1:0", False, id="openai-route"),
    pytest.param("mantle/openai.gpt-5.5", False, id="mantle"),
    pytest.param("invoke/" + BEDROCK_ARN, False, id="invoke-application-profile-arn"),
    pytest.param(BEDROCK_ARN.upper(), False, id="upper-case-arn"),
    pytest.param("invoke/anthropic.claude-3-5-haiku-20241022-v1:0", True, id="invoke-claude"),
])
def test_bedrock_routes_whose_usage_litellm_does_not_fold_keep_the_claude_rule(model, expected):
    assert prompt_tokens_fold_cache({"custom_llm_provider": "bedrock", "model": model}) is expected


@pytest.fixture
def without_litellm(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def refuse(name, *args, **kwargs):
        if name.startswith("litellm"):
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", refuse)


@pytest.fixture
def reset_bedrock_routing_warning(monkeypatch):
    monkeypatch.setattr(cache_tokens, "_BEDROCK_ROUTING_WARNED", False)


def test_a_litellm_routing_failure_keeps_todays_claude_rule_and_warns(
        without_litellm, reset_bedrock_routing_warning, caplog):
    caplog.set_level("WARNING", logger="revenium_middleware")

    assert prompt_tokens_fold_cache(
        {"custom_llm_provider": "bedrock", "model": BEDROCK_CLAUDE}) is True
    assert not caplog.records

    assert prompt_tokens_fold_cache(
        {"custom_llm_provider": "bedrock", "model": "chat_completions/openai.gpt-oss-120b-1:0"}) is False
    assert prompt_tokens_fold_cache({"custom_llm_provider": "bedrock", "model": BEDROCK_ARN}) is False

    warnings = [r for r in caplog.records if r.levelname == "WARNING"]
    assert len(warnings) == 1
    assert "LiteLLM Bedrock routing unavailable" in warnings[0].getMessage()


@requires_litellm
def test_a_failing_invoke_provider_lookup_keeps_the_anthropic_window_and_the_claude_rule(
        monkeypatch, reset_bedrock_routing_warning):
    from litellm.llms.bedrock.base_aws_llm import BaseAWSLLM

    def raises(model):
        raise AttributeError("get_bedrock_invoke_provider")

    monkeypatch.setattr(BaseAWSLLM, "get_bedrock_invoke_provider", staticmethod(raises))
    router = "invoke/anthropic.claude-3-5-haiku-20241022-v1:0"

    assert cache_tokens.routes_via_bedrock_converse(
        {"custom_llm_provider": "bedrock", "model": router}) is False
    assert prompt_tokens_fold_cache({"custom_llm_provider": "bedrock", "model": router}) is True
    assert prompt_tokens_fold_cache(
        {"custom_llm_provider": "bedrock", "model": "invoke/us.amazon.nova-pro-v1:0"}) is False


@pytest.mark.parametrize("sources", [
    pytest.param((), id="no-sources"),
    pytest.param(({"model": BEDROCK_CLAUDE},), id="claude-model-without-provider"),
    pytest.param(({"litellm_model_name": "bedrock/" + BEDROCK_CLAUDE}, None), id="model-name-without-provider"),
])
def test_unknown_provider_or_claude_model_without_provider_stays_gross(sources):
    assert prompt_tokens_fold_cache(*sources) is False


@pytest.mark.parametrize("provider", ["anthropic", "anthropic_text"])
def test_direct_anthropic_still_folds_without_a_model(provider):
    assert prompt_tokens_fold_cache({"custom_llm_provider": provider}) is True


def test_claude_in_nested_metadata_does_not_decide_the_model():
    """``metadata.model_group`` is the alias the caller picked, not the deployment."""
    assert prompt_tokens_fold_cache(
        {"custom_llm_provider": "vertex_ai", "model": "gemini-2.5-flash-lite",
         "metadata": {"model_group": "claude-haiku"}},
        {"metadata": {"model_group": "claude-haiku", "model": BEDROCK_CLAUDE}},
    ) is False


def test_the_first_model_named_decides():
    """A Gemini deployment is not turned into Claude by a later source's name."""
    assert prompt_tokens_fold_cache(
        {"custom_llm_provider": "vertex_ai", "model": "gemini-2.5-flash-lite"},
        {"litellm_model_name": "vertex_ai/" + VERTEX_CLAUDE},
    ) is False


@pytest.fixture
def litellm_version(monkeypatch):
    def pin(version):
        monkeypatch.setattr(cache_tokens, "_installed_litellm_version", lambda: version, raising=False)

    return pin


VERSION_CASES = [
    ("1.65.0", True, True),
    ("1.66.0", False, False),
    ("1.70.0", False, False),
    ("1.77.2", False, False),
    ("1.77.2.post1", False, False),
    ("1.77.3", True, False),
    ("1.77.9", True, False),
    ("1.78.0", True, True),
    ("1.104.2", True, True),
    ("1.80.0rc1", True, True),
    (None, True, True),
    ("garbage", True, True),
    ("1.77", True, True),
]


@pytest.mark.parametrize("version, anthropic_style_folds, converse_folds", VERSION_CASES)
@pytest.mark.parametrize("converse", [False, True], ids=["anthropic-style", "converse"])
def test_litellm_folds_the_cache_write_outside_its_read_only_window(
        litellm_version, version, anthropic_style_folds, converse_folds, converse):
    """LiteLLM added only the cache read to ``prompt_tokens`` from 1.66.0, until 1.77.3
    on the Anthropic-style routes and until 1.78.0 on Bedrock Converse."""
    litellm_version(version)

    expected = converse_folds if converse else anthropic_style_folds
    assert cache_tokens.litellm_folds_cache_write(converse=converse) is expected


@pytest.mark.parametrize("sources", [
    pytest.param(({"custom_llm_provider": "bedrock", "model": BEDROCK_CLAUDE},), id="bedrock-claude-default"),
    pytest.param(({"custom_llm_provider": "bedrock", "litellm_model_name": "bedrock/" + BEDROCK_CLAUDE},),
                 id="bedrock-claude-litellm-model-name"),
    pytest.param(({"custom_llm_provider": "bedrock", "model": "converse/" + BEDROCK_CLAUDE},),
                 id="bedrock-explicit-converse"),
])
def test_bedrock_claude_routes_via_converse(sources):
    assert cache_tokens.routes_via_bedrock_converse(*sources) is True


@pytest.mark.parametrize("sources", [
    pytest.param(({"custom_llm_provider": "bedrock",
                   "model": "invoke/anthropic.claude-3-5-haiku-20241022-v1:0"},), id="bedrock-invoke"),
    pytest.param(({"custom_llm_provider": "bedrock",
                   "litellm_model_name": "bedrock/invoke/anthropic.claude-3-5-haiku-20241022-v1:0"},),
                 id="bedrock-invoke-litellm-model-name"),
    pytest.param(({"custom_llm_provider": "vertex_ai", "model": VERTEX_CLAUDE},), id="vertex-claude"),
    pytest.param(({"custom_llm_provider": "anthropic", "model": "claude-haiku-4-5"},), id="anthropic"),
    pytest.param(({"call_type": "anthropic_messages"}, {"custom_llm_provider": "bedrock", "model": BEDROCK_CLAUDE}),
                 id="anthropic-messages-call-type-first"),
    pytest.param((), id="no-sources"),
])
def test_other_routes_are_not_converse(sources):
    assert cache_tokens.routes_via_bedrock_converse(*sources) is False


def test_converse_is_assumed_unless_invoke_is_named_when_litellm_cannot_say(
        without_litellm, reset_bedrock_routing_warning):
    assert cache_tokens.routes_via_bedrock_converse(
        {"custom_llm_provider": "bedrock", "model": BEDROCK_CLAUDE}) is True
    assert cache_tokens.routes_via_bedrock_converse(
        {"custom_llm_provider": "bedrock", "model": BEDROCK_ARN}) is True
    assert cache_tokens.routes_via_bedrock_converse(
        {"custom_llm_provider": "bedrock", "model": "invoke/anthropic.claude-3-5-haiku-20241022-v1:0"}) is False
    assert cache_tokens.routes_via_bedrock_converse(
        {"custom_llm_provider": "bedrock", "model": "invoke/us.amazon.nova-pro-v1:0"}) is False


@pytest.mark.parametrize("version, sources, expected", [
    ("1.104.2", ({"custom_llm_provider": "bedrock", "model": BEDROCK_CLAUDE},), 4700),
    ("1.70.0", ({"custom_llm_provider": "anthropic"},), 0),
    ("1.70.0", ({"custom_llm_provider": "bedrock", "model": BEDROCK_CLAUDE},), 0),
    ("1.77.3", ({"custom_llm_provider": "anthropic"},), 4700),
    ("1.77.3", ({"custom_llm_provider": "vertex_ai", "model": VERTEX_CLAUDE},), 4700),
    ("1.77.3", ({"custom_llm_provider": "bedrock",
                 "model": "invoke/anthropic.claude-3-5-haiku-20241022-v1:0"},), 4700),
    ("1.77.3", ({"custom_llm_provider": "bedrock", "model": BEDROCK_CLAUDE},), 0),
    pytest.param("1.77.3", ({"custom_llm_provider": "bedrock", "model": "invoke/us.amazon.nova-pro-v1:0"},), 0,
                 marks=requires_litellm),
    ("1.77.3", ({"custom_llm_provider": "bedrock", "model": "invoke/anthropic/" + BEDROCK_ARN},), 4700),
    ("1.104.2", ({"custom_llm_provider": "bedrock", "model": BEDROCK_ARN},), 4700),
])
def test_only_a_write_litellm_folded_is_taken_out(litellm_version, version, sources, expected):
    litellm_version(version)

    assert cache_tokens.cache_creation_inside_litellm_prompt(4700, *sources) == expected
