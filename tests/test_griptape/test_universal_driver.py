"""Provider detection and delegation of the universal drivers."""
import pytest

pytest.importorskip("griptape")

from revenium_middleware.griptape.universal_driver import (
    ReveniumDriver,
    ReveniumEmbeddingDriver,
)


def _bare(cls):
    """Instance without running __init__, to unit-test detection helpers."""
    return cls.__new__(cls)


@pytest.mark.parametrize(
    "model,provider",
    [
        ("gpt-4", "openai"),
        ("gpt-4o-mini", "openai"),
        ("claude-3-5-sonnet-20241022", "anthropic"),
        ("claude-sonnet-4-5-20250929", "anthropic"),
        ("claude-opus-4-1", "anthropic"),
        ("llama3.2", "ollama"),
        ("mistral", "ollama"),
        ("gemini-1.5-flash", "litellm"),
        ("command-r", "litellm"),  # unknown → litellm fallback
        # Provider-qualified (LiteLLM convention) names route to litellm even
        # when they contain direct-provider substrings like "gpt-" or "claude".
        ("azure/gpt-4", "litellm"),
        ("bedrock/anthropic.claude-3-sonnet", "litellm"),
        ("openrouter/deepseek/deepseek-chat", "litellm"),
    ],
)
def test_detect_provider_from_model(model, provider):
    assert _bare(ReveniumDriver)._detect_provider_from_model(model) == provider


def test_requires_model_or_base_driver():
    with pytest.raises(ValueError):
        ReveniumDriver()


def test_detect_provider_from_driver_class_name():
    class FakeOpenAiChatPromptDriver:
        pass

    class FakeAnthropicPromptDriver:
        pass

    d = _bare(ReveniumDriver)
    assert d._detect_provider_from_driver(FakeOpenAiChatPromptDriver()) == "openai"
    assert d._detect_provider_from_driver(FakeAnthropicPromptDriver()) == "anthropic"


def test_wrap_existing_unknown_driver_raises():
    class SomeRandomPromptDriver:
        pass

    d = _bare(ReveniumDriver)
    d.usage_metadata = {}
    with pytest.raises(ValueError):
        d._wrap_existing_driver(SomeRandomPromptDriver())


def test_getattr_delegates_to_wrapped_driver():
    d = _bare(ReveniumDriver)
    d._wrapped_driver = type("W", (), {"model": "gpt-4", "ping": lambda self: "pong"})()
    assert d.model == "gpt-4"
    assert d.ping() == "pong"


def test_getattr_without_wrapped_driver_raises():
    d = _bare(ReveniumDriver)
    d._wrapped_driver = None
    with pytest.raises(AttributeError):
        _ = d.anything


@pytest.mark.parametrize(
    "model,provider",
    [
        ("text-embedding-3-large", "openai"),
        ("voyage-2", "voyageai"),
        ("embed-english-v3.0", "cohere"),
        ("nomic-embed-text", "ollama"),
        ("something-unknown", "openai"),  # unknown → openai fallback
    ],
)
def test_detect_embedding_provider_from_model(model, provider):
    assert (
        _bare(ReveniumEmbeddingDriver)._detect_provider_from_embedding_model(model)
        == provider
    )


def test_create_embedding_driver_unsupported_provider_raises():
    d = _bare(ReveniumEmbeddingDriver)
    d.usage_metadata = {}
    with pytest.raises(ValueError):
        d._create_embedding_driver_for_provider("cohere", "embed-english-v3.0")


def test_wrap_existing_unsupported_embedding_driver_raises():
    class CohereEmbeddingDriver:
        pass

    d = _bare(ReveniumEmbeddingDriver)
    d.usage_metadata = {}
    with pytest.raises(ValueError):
        d._wrap_existing_embedding_driver(CohereEmbeddingDriver())


def test_agent_accepts_revenium_driver():
    from griptape.structures import Agent
    from revenium_middleware.griptape import ReveniumDriver

    driver = ReveniumDriver(model="gpt-4o-mini", api_key="test-key")
    agent = Agent(prompt_driver=driver)
    # The agent must actually hold the Revenium wrapper, not a substitute.
    assert agent.prompt_driver is driver
    assert isinstance(agent.prompt_driver, ReveniumDriver)


def test_wrap_existing_openai_driver_preserves_base_url_and_organization():
    from griptape.drivers.prompt.openai_chat_prompt_driver import OpenAiChatPromptDriver

    base_driver = OpenAiChatPromptDriver(
        model="gpt-4o-mini",
        api_key="test-key",
        base_url="https://gateway.internal/v1",
        organization="org-123",
    )
    d = ReveniumDriver(base_driver=base_driver)
    assert d.wrapped_driver.base_url == "https://gateway.internal/v1"
    assert d.wrapped_driver.organization == "org-123"


def test_wrap_existing_anthropic_driver_preserves_top_p():
    from griptape.drivers.prompt.anthropic import AnthropicPromptDriver

    base_driver = AnthropicPromptDriver(
        model="claude-3-5-sonnet-20241022",
        api_key="test-key",
        top_p=0.9,
    )
    d = ReveniumDriver(base_driver=base_driver)
    assert d.wrapped_driver.top_p == 0.9
    assert d.wrapped_driver.api_key == "test-key"


def test_env_proxy_url_forces_litellm_provider(monkeypatch):
    import revenium_middleware.griptape.litellm_driver as litellm_driver

    monkeypatch.setenv("LITELLM_PROXY_URL", "https://proxy.internal/chat/completions")
    # litellm itself may not be installed in the test env; the driver only
    # gates its __init__ on this flag, so force it on for construction.
    monkeypatch.setattr(litellm_driver, "LITELLM_AVAILABLE", True)

    d = ReveniumDriver(model="gpt-4")
    assert d.provider == "litellm"


def test_wrap_existing_unknown_embedding_driver_raises():
    # Unrecognized class names detect as the "unknown" provider, so wrapping
    # must raise rather than silently treating the driver as OpenAI or handing
    # back the original unmetered driver.
    class SomeRandomEmbeddingDriver:
        model = "text-embedding-3-small"

    d = _bare(ReveniumEmbeddingDriver)
    d.usage_metadata = {}
    with pytest.raises(ValueError, match="Cannot wrap SomeRandomEmbeddingDriver"):
        d._wrap_existing_embedding_driver(SomeRandomEmbeddingDriver())
