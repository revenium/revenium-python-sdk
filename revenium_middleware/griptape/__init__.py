"""
Revenium metering drivers for the Griptape framework.

Universal support for LLM providers through the unified Revenium SDK.
Install with: pip install "revenium-python-sdk[griptape]"

Supported providers:
- Tier 1 (direct): OpenAI, Anthropic, Ollama
- Tier 2 (via LiteLLM): Google/Gemini, Cohere, Azure OpenAI, Bedrock, and 100+ others
"""
try:
    import griptape  # noqa: F401
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "The griptape package is required for revenium_middleware.griptape. "
        "Install it with: pip install 'revenium-python-sdk[griptape]' "
        "(requires Python 3.10+)."
    ) from exc

from .universal_driver import ReveniumDriver, ReveniumEmbeddingDriver
from .openai_driver import ReveniumOpenAiDriver
from .openai_embedding_driver import ReveniumOpenAiEmbeddingDriver
from .anthropic_driver import ReveniumAnthropicDriver
from .ollama_driver import ReveniumOllamaDriver
from .litellm_driver import ReveniumLiteLLMDriver

__all__ = [
    "ReveniumDriver",
    "ReveniumEmbeddingDriver",
    "ReveniumOpenAiDriver",
    "ReveniumOpenAiEmbeddingDriver",
    "ReveniumAnthropicDriver",
    "ReveniumOllamaDriver",
    "ReveniumLiteLLMDriver",
]
