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
except ImportError as e:
    from revenium_middleware._core.load_diagnostics import log_middleware_load_failure
    log_middleware_load_failure("LiteLLM", e, required_packages=("litellm",))
    client = None  # type: ignore
    proxy = None  # type: ignore

__all__ = [
    "client",
    "proxy",
]
