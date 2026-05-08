"""
Revenium middleware for OpenAI and Azure OpenAI.

Automatically hooks openai.ChatCompletion.create using wrapt and logs
token usage after each request. Supports OpenAI, Azure OpenAI, and
LangChain integrations.

Usage:
    import revenium_middleware.openai  # auto-instruments OpenAI SDK
"""
import logging

from revenium_middleware._core.exceptions import ReveniumCostLimitExceeded

logger = logging.getLogger(__name__)

# Conditionally import middleware (requires wrapt + openai SDK)
try:
    import wrapt  # noqa: F401
    from .middleware import create_wrapper
except ImportError:
    logger.debug("OpenAI middleware dependencies not available, middleware not loaded")
    create_wrapper = None  # type: ignore

__all__ = ["create_wrapper", "ReveniumCostLimitExceeded"]
