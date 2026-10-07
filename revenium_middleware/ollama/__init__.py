"""
Revenium Middleware for Ollama Python SDK.

When you install and import this library, it will automatically hook chat,
generate, embed and the legacy embeddings call on ollama.Client and
ollama.AsyncClient using wrapt, which also covers the module-level ollama.chat,
ollama.generate, ollama.embed and ollama.embeddings, and meter token usage after
each request.
"""

import logging

logger = logging.getLogger(__name__)

# Conditionally import middleware (requires ollama SDK)
try:
    import ollama as _ollama  # noqa: F401
    from .middleware import chat_wrapper, generate_wrapper, embed_wrapper, embeddings_wrapper
except ImportError as e:
    from revenium_middleware._core.load_diagnostics import log_middleware_load_failure
    log_middleware_load_failure("Ollama", e, required_packages=("ollama",))
    chat_wrapper = None  # type: ignore
    generate_wrapper = None  # type: ignore
    embed_wrapper = None  # type: ignore
    embeddings_wrapper = None  # type: ignore

__all__ = [
    "chat_wrapper",
    "generate_wrapper",
    "embed_wrapper",
    "embeddings_wrapper",
]
