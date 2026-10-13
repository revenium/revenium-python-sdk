"""Public surface of revenium_middleware.griptape."""
import pytest

pytest.importorskip("griptape")

EXPECTED_NAMES = [
    "ReveniumDriver",
    "ReveniumEmbeddingDriver",
    "ReveniumOpenAiDriver",
    "ReveniumOpenAiEmbeddingDriver",
    "ReveniumAnthropicDriver",
    "ReveniumOllamaDriver",
    "ReveniumLiteLLMDriver",
    "enable_budget_errors",
    "disable_budget_errors",
    "raise_if_budget_exceeded",
]


def test_public_names_importable():
    import revenium_middleware.griptape as gt

    for name in EXPECTED_NAMES:
        assert hasattr(gt, name), f"missing export: {name}"


def test_all_matches_expected():
    import revenium_middleware.griptape as gt

    assert sorted(gt.__all__) == sorted(EXPECTED_NAMES)
