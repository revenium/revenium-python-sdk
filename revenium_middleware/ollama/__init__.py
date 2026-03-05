"""
Revenium Middleware for Ollama Python SDK.

When you install and import this library, it will automatically hook
ollama.chat, ollama.generate, and ollama.embed using wrapt, and log token usage after
each request. You can customize or extend this logging logic later
to add user or organization metadata for metering purposes.
"""

import logging

logger = logging.getLogger(__name__)

# Conditionally import middleware (requires ollama SDK)
try:
    import ollama as _ollama  # noqa: F401
    from .middleware import chat_wrapper, generate_wrapper, embed_wrapper
except ImportError:
    logger.debug("Ollama SDK (ollama) not available, middleware not loaded")
    chat_wrapper = None  # type: ignore
    generate_wrapper = None  # type: ignore
    embed_wrapper = None  # type: ignore

__all__ = [
    "chat_wrapper",
    "generate_wrapper",
    "embed_wrapper",
]
