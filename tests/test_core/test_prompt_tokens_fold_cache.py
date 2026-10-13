"""Which LiteLLM upstreams fold the cache into ``prompt_tokens`` (BACK-4162).

LiteLLM's Bedrock routes (Converse, invoke, mantle) and its Vertex AI Claude
route build ``prompt_tokens`` as uncached input plus cache read plus cache
write, exactly as its direct Anthropic route does. Left gross, the cache is
billed twice: once at the input rate and again at the cache rates. Gemini on
Vertex AI and Nova on Bedrock are not Claude, so this fix leaves them as
LiteLLM built them.

The model that decides is read only from the fields LiteLLM fills (``model``
and ``litellm_model_name``), never from the caller's request metadata.
"""
import pytest

from revenium_middleware._core import cache_tokens
from revenium_middleware._core.cache_tokens import prompt_tokens_fold_cache

BEDROCK_CLAUDE = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
VERTEX_CLAUDE = "claude-haiku-4-5@20251001"


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
    pytest.param(({"custom_llm_provider": "bedrock", "model": "us.amazon.nova-micro-v1:0"},), id="bedrock-nova"),
    pytest.param(({"custom_llm_provider": "bedrock",
                   "model": "arn:aws:bedrock:us-east-1:123456789012:application-inference-profile/abc123"},),
                 id="bedrock-inference-profile-arn"),
    pytest.param(({"custom_llm_provider": "bedrock"},), id="bedrock-without-a-model"),
    pytest.param(({"custom_llm_provider": "openai", "model": "claude-proxy-alias"},), id="openai-named-claude"),
])
def test_non_claude_models_on_the_same_providers_stay_gross(sources):
    assert prompt_tokens_fold_cache(*sources) is False


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
        {"custom_llm_provider": "bedrock", "model": "us.amazon.nova-micro-v1:0",
         "metadata": {"model_group": "claude-haiku"}},
        {"metadata": {"model_group": "claude-haiku", "model": BEDROCK_CLAUDE}},
    ) is False


def test_the_first_model_named_decides():
    """A Nova deployment is not turned into Claude by a later source's name."""
    assert prompt_tokens_fold_cache(
        {"custom_llm_provider": "bedrock", "model": "us.amazon.nova-micro-v1:0"},
        {"litellm_model_name": "bedrock/" + BEDROCK_CLAUDE},
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


def test_converse_is_assumed_unless_invoke_is_named_when_litellm_cannot_say(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def without_litellm(name, *args, **kwargs):
        if name.startswith("litellm"):
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_litellm)

    assert cache_tokens.routes_via_bedrock_converse(
        {"custom_llm_provider": "bedrock", "model": BEDROCK_CLAUDE}) is True
    assert cache_tokens.routes_via_bedrock_converse(
        {"custom_llm_provider": "bedrock", "model": "invoke/anthropic.claude-3-5-haiku-20241022-v1:0"}) is False


@pytest.mark.parametrize("version, sources, expected", [
    ("1.104.2", ({"custom_llm_provider": "bedrock", "model": BEDROCK_CLAUDE},), 4700),
    ("1.70.0", ({"custom_llm_provider": "anthropic"},), 0),
    ("1.70.0", ({"custom_llm_provider": "bedrock", "model": BEDROCK_CLAUDE},), 0),
    ("1.77.3", ({"custom_llm_provider": "anthropic"},), 4700),
    ("1.77.3", ({"custom_llm_provider": "vertex_ai", "model": VERTEX_CLAUDE},), 4700),
    ("1.77.3", ({"custom_llm_provider": "bedrock",
                 "model": "invoke/anthropic.claude-3-5-haiku-20241022-v1:0"},), 4700),
    ("1.77.3", ({"custom_llm_provider": "bedrock", "model": BEDROCK_CLAUDE},), 0),
])
def test_only_a_write_litellm_folded_is_taken_out(litellm_version, version, sources, expected):
    litellm_version(version)

    assert cache_tokens.cache_creation_inside_litellm_prompt(4700, *sources) == expected
