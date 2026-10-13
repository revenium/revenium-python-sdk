"""usage_metadata injection in provider-specific drivers."""
import pytest

pytest.importorskip("griptape")

from griptape.common.prompt_stack.prompt_stack import PromptStack


def _prompt_stack():
    ps = PromptStack()
    ps.add_user_message("hello")
    return ps


def test_anthropic_driver_injects_usage_metadata(monkeypatch):
    import revenium_middleware.griptape.anthropic_driver as mod

    monkeypatch.setattr(mod, "MIDDLEWARE_AVAILABLE", True)
    driver = mod.ReveniumAnthropicDriver(
        model="claude-3-5-sonnet-20241022",
        api_key="test-key",
        usage_metadata={"trace_id": "t-1", "task_type": "unit-test"},
    )
    params = driver._base_params(_prompt_stack())
    assert params["usage_metadata"] == {"trace_id": "t-1", "task_type": "unit-test"}


def test_anthropic_driver_strips_auth_keys(monkeypatch):
    import revenium_middleware.griptape.anthropic_driver as mod

    monkeypatch.setattr(mod, "MIDDLEWARE_AVAILABLE", True)
    driver = mod.ReveniumAnthropicDriver(
        model="claude-3-5-sonnet-20241022",
        api_key="test-key",
        usage_metadata={"trace_id": "t", "revenium_api_key": "secret"},
    )
    params = driver._base_params(_prompt_stack())
    # Auth credentials are stripped; only clean business metadata is injected.
    assert params["usage_metadata"] == {"trace_id": "t"}


def test_anthropic_driver_skips_injection_without_middleware(monkeypatch):
    import revenium_middleware.griptape.anthropic_driver as mod

    monkeypatch.setattr(mod, "MIDDLEWARE_AVAILABLE", False)
    driver = mod.ReveniumAnthropicDriver(
        model="claude-3-5-sonnet-20241022",
        api_key="test-key",
        usage_metadata={"trace_id": "t-1"},
    )
    params = driver._base_params(_prompt_stack())
    assert "usage_metadata" not in params


def test_anthropic_driver_no_metadata_no_injection(monkeypatch):
    import revenium_middleware.griptape.anthropic_driver as mod

    monkeypatch.setattr(mod, "MIDDLEWARE_AVAILABLE", True)
    driver = mod.ReveniumAnthropicDriver(
        model="claude-3-5-sonnet-20241022", api_key="test-key"
    )
    params = driver._base_params(_prompt_stack())
    assert "usage_metadata" not in params


def test_openai_driver_stores_usage_metadata():
    from revenium_middleware.griptape.openai_driver import ReveniumOpenAiDriver

    driver = ReveniumOpenAiDriver(
        model="gpt-4o-mini", api_key="test-key", usage_metadata={"trace_id": "t-2"}
    )
    assert driver.usage_metadata == {"trace_id": "t-2"}


def test_openai_driver_skips_injection_without_middleware(monkeypatch):
    import revenium_middleware.griptape.openai_driver as mod

    monkeypatch.setattr(mod, "REVENIUM_AVAILABLE", False)
    driver = mod.ReveniumOpenAiDriver(
        model="gpt-4o-mini", api_key="test-key", usage_metadata={"trace_id": "t"}
    )
    params = driver._base_params(_prompt_stack())
    assert "usage_metadata" not in params


def test_openai_driver_injects_with_middleware(monkeypatch):
    import revenium_middleware.griptape.openai_driver as mod

    monkeypatch.setattr(mod, "REVENIUM_AVAILABLE", True)
    driver = mod.ReveniumOpenAiDriver(
        model="gpt-4o-mini",
        api_key="test-key",
        usage_metadata={"trace_id": "t", "revenium_api_key": "secret"},
    )
    params = driver._base_params(_prompt_stack())
    # Auth credentials are stripped; only clean business metadata is injected.
    assert params["usage_metadata"] == {"trace_id": "t"}


def test_openai_embedding_driver_skips_injection_without_middleware(monkeypatch):
    import revenium_middleware.griptape.openai_embedding_driver as mod

    monkeypatch.setattr(mod, "REVENIUM_AVAILABLE", False)
    driver = mod.ReveniumOpenAiEmbeddingDriver(
        model="text-embedding-3-small",
        api_key="test-key",
        usage_metadata={"trace_id": "t"},
    )
    params = driver._params("hello")
    assert "usage_metadata" not in params


def test_openai_embedding_driver_injects_with_middleware(monkeypatch):
    import revenium_middleware.griptape.openai_embedding_driver as mod

    monkeypatch.setattr(mod, "REVENIUM_AVAILABLE", True)
    driver = mod.ReveniumOpenAiEmbeddingDriver(
        model="text-embedding-3-small",
        api_key="test-key",
        usage_metadata={"trace_id": "t", "revenium_api_key": "secret"},
    )
    params = driver._params("hello")
    # Auth credentials are stripped; only clean business metadata is injected.
    assert params["usage_metadata"] == {"trace_id": "t"}


def _revenium_and_parent_drivers():
    """(Revenium class, its Griptape parent class, settings), one per Revenium driver class."""
    from griptape.drivers.embedding.openai import OpenAiEmbeddingDriver
    from griptape.drivers.prompt.anthropic import AnthropicPromptDriver
    from griptape.drivers.prompt.azure_openai_chat_prompt_driver import AzureOpenAiChatPromptDriver
    from griptape.drivers.prompt.ollama import OllamaPromptDriver
    from griptape.drivers.prompt.openai_chat_prompt_driver import OpenAiChatPromptDriver

    from revenium_middleware.griptape import (
        ReveniumAnthropicDriver,
        ReveniumLiteLLMDriver,
        ReveniumOllamaDriver,
        ReveniumOpenAiDriver,
        ReveniumOpenAiEmbeddingDriver,
    )
    from revenium_middleware.griptape.openai_driver import ReveniumAzureOpenAiDriver

    azure = dict(model="gpt-4o", azure_deployment="dep1", azure_endpoint="https://myres.openai.azure.com",
                 api_key="test-key")
    return [
        (ReveniumOpenAiDriver, OpenAiChatPromptDriver, dict(model="gpt-4o-mini", api_key="test-key")),
        (ReveniumAzureOpenAiDriver, AzureOpenAiChatPromptDriver, azure),
        (ReveniumAnthropicDriver, AnthropicPromptDriver, dict(model="claude-3-5-haiku-latest", api_key="test-key")),
        (ReveniumOllamaDriver, OllamaPromptDriver, dict(model="llama3.2")),
        (ReveniumLiteLLMDriver, None, dict(model="gemini/gemini-1.5-flash")),
        (ReveniumOpenAiEmbeddingDriver, OpenAiEmbeddingDriver, dict(model="text-embedding-3-small", api_key="test-key")),
    ]


def _parent_default_exclusions(parent_class, settings):
    if parent_class is None:
        import attrs
        from griptape.drivers.prompt.base_prompt_driver import BasePromptDriver

        return attrs.fields(BasePromptDriver).ignored_exception_types.default.factory()
    return parent_class(**settings).ignored_exception_types


def test_a_refused_prompt_call_is_attempted_once_and_never_retried(
        breached_cap, stub_provider, attempts, retries, prompt_driver):
    from revenium_middleware import BudgetExceededError

    with pytest.raises(BudgetExceededError):
        prompt_driver(stub_provider).run(_prompt_stack())

    assert attempts["prompt"] == 1
    assert retries == []
    assert stub_provider.hits == 0


def test_a_refused_embedding_call_is_attempted_once_and_never_retried(
        breached_cap, stub_provider, attempts, retries, embedding_driver):
    from revenium_middleware import BudgetExceededError

    with pytest.raises(BudgetExceededError):
        embedding_driver(stub_provider).embed("hello")

    assert attempts["embedding"] == 1
    assert retries == []
    assert stub_provider.hits == 0


@pytest.mark.parametrize("index", range(6), ids=[
    "openai", "azure-openai", "anthropic", "ollama", "litellm", "openai-embedding"])
def test_every_revenium_driver_treats_the_cap_error_as_final(index):
    import attrs

    from revenium_middleware import BudgetExceededError

    revenium_class, parent_class, settings = _revenium_and_parent_drivers()[index]

    driver = revenium_class(ignored_exception_types=(KeyError,), **settings)
    assert driver.ignored_exception_types == (KeyError, BudgetExceededError)
    assert attrs.evolve(driver).ignored_exception_types.count(BudgetExceededError) == 1

    defaults = revenium_class(**settings).ignored_exception_types
    assert defaults == (*_parent_default_exclusions(parent_class, settings), BudgetExceededError)


def test_a_provider_error_is_still_retried(failing_provider, attempts, retries, prompt_driver):
    from griptape.artifacts import ErrorArtifact
    from griptape.structures import Agent

    driver = prompt_driver(failing_provider)
    agent = Agent(prompt_driver=driver)
    agent.run("hi")

    assert isinstance(agent.output, ErrorArtifact)
    assert attempts["prompt"] == driver.max_attempts == 2
    assert len(retries) == 2


def test_an_agent_refusal_is_not_retried(breached_cap, stub_provider, attempts, retries, prompt_driver):
    from griptape.artifacts import ErrorArtifact
    from griptape.structures import Agent

    from revenium_middleware import BudgetExceededError

    agent = Agent(prompt_driver=prompt_driver(stub_provider))
    agent.run("hi")

    assert isinstance(agent.output, ErrorArtifact)
    assert isinstance(agent.output.exception, BudgetExceededError)
    assert attempts["prompt"] == 1
    assert retries == []
    assert stub_provider.hits == 0
