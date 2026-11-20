"""Every hosting provider LiteLLM knows has a recorded decision for Claude's cache tokens (BACK-6110).

``bedrock_mantle`` was a LiteLLM provider for months before the netting rule
listed it, and Claude calls through it were billed for their cached tokens
twice. This runs in the job that installs the newest LiteLLM, so a provider
name added upstream fails here until someone decides whether it folds.
"""
import pytest

pytest.importorskip("litellm")

from revenium_middleware._core.cache_tokens import prompt_tokens_fold_cache  # noqa: E402

# Whether a Claude call on each of LiteLLM's Bedrock-, Vertex AI- and
# Azure-flavoured providers is netted here. A provider LiteLLM adds later is in
# neither table, which fails the test below until someone decides.
PROVIDERS_NETTED_FOR_CLAUDE = {"bedrock", "bedrock_mantle", "vertex_ai"}
PROVIDERS_NOT_NETTED_FOR_CLAUDE = {
    "vertex_ai_beta": "LiteLLM dispatches it to its Gemini handler only",
    "azure": "Azure OpenAI deployments; no Claude",
    "azure_text": "Azure OpenAI completions; no Claude",
    "azure_speech": "speech only",
    "azure_ai": "folds by code reading, stored row not yet checked (BACK-5800)",
}
HOSTING_FLAVOURS = ("bedrock", "vertex", "azure")


def test_every_hosting_provider_litellm_knows_has_a_decision_for_claude():
    from litellm.types.utils import LlmProviders

    hosting = {p.value for p in LlmProviders if any(flavour in p.value for flavour in HOSTING_FLAVOURS)}

    assert hosting - PROVIDERS_NETTED_FOR_CLAUDE - set(PROVIDERS_NOT_NETTED_FOR_CLAUDE) == set()
    for provider in hosting:
        folds = prompt_tokens_fold_cache({"custom_llm_provider": provider, "model": "claude-haiku-4-5"})
        assert folds is (provider in PROVIDERS_NETTED_FOR_CLAUDE), provider
