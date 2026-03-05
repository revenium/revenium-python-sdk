"""
Revenium LiteLLM Proxy Middleware

When you install and import this library, it will automatically hook
LiteLLM proxy requests using a custom logger, and log token usage after
each request. You can customize or extend this logging logic later
to add user or organization metadata for metering purposes.
"""

import logging

logger = logging.getLogger(__name__)

try:
    import litellm  # noqa: F401
    from .middleware import MiddlewareHandler, proxy_handler_instance
except ImportError:
    logger.debug("LiteLLM SDK (litellm) not available, proxy middleware not loaded")
    MiddlewareHandler = None  # type: ignore
    proxy_handler_instance = None  # type: ignore

__all__ = [
    "MiddlewareHandler",
    "proxy_handler_instance",
]
