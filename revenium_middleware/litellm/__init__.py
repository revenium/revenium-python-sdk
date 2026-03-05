"""
Revenium Middleware for LiteLLM.

Provides both client-side (wrapt-based) and proxy-side (custom logger) middleware
for metering LiteLLM usage.
"""

import logging

logger = logging.getLogger(__name__)

# Conditionally import subpackages
try:
    import litellm as _litellm  # noqa: F401
    from . import client
    from . import proxy
except ImportError:
    logger.debug("LiteLLM SDK (litellm) not available, middleware not loaded")
    client = None  # type: ignore
    proxy = None  # type: ignore

__all__ = [
    "client",
    "proxy",
]
