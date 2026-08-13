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
except ImportError as e:
    from revenium_middleware._core.load_diagnostics import log_middleware_load_failure
    log_middleware_load_failure("LiteLLM proxy", e, required_packages=("litellm",))
    MiddlewareHandler = None  # type: ignore
    proxy_handler_instance = None  # type: ignore

__all__ = [
    "MiddlewareHandler",
    "proxy_handler_instance",
]
