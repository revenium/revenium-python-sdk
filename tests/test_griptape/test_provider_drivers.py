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
