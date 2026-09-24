"""
Revenium LiteLLM Proxy integration.

``ReveniumGuardrail`` is the supported integration: a LiteLLM
``CustomGuardrail`` that enforces the caller's budget before the proxied call
and meters usage after it (success, failure and streamed responses alike).
Configure it under ``litellm_settings.guardrails`` in the proxy's
``config.yaml``.

``MiddlewareHandler`` / ``proxy_handler_instance`` -- the ``CustomLogger``
callback -- is deprecated. It meters but never enforces, and enabling both
meters every call twice. See the README's "Migrating from the callback".

``ReveniumGuardrail`` is ``None`` when the proxy extra is not installed
(``pip install "revenium-python-sdk[litellm-proxy]"``) or on Python < 3.10,
which ``litellm[proxy]`` does not support.
"""

import logging

logger = logging.getLogger(__name__)

try:
    import litellm  # noqa: F401
    from .middleware import MiddlewareHandler
except ImportError as e:
    from revenium_middleware._core.load_diagnostics import log_middleware_load_failure
    log_middleware_load_failure("LiteLLM proxy", e, required_packages=("litellm",))
    MiddlewareHandler = None  # type: ignore
    proxy_handler_instance = None  # type: ignore

try:
    from .guardrail import ReveniumGuardrail
except ImportError as e:
    # Missing fastapi (client-only install) or Python < 3.10. Neither is an
    # error for a caller who only wants the client middleware, so this is a
    # debug line rather than the load-failure diagnostic.
    logger.debug("Revenium LiteLLM guardrail unavailable: %s", e)
    ReveniumGuardrail = None  # type: ignore


def __getattr__(name):
    """Resolve ``proxy_handler_instance`` lazily.

    Building the deprecated handler emits a ``DeprecationWarning``, and
    importing this package must not emit one at a caller who only uses the
    guardrail. The attribute is created the first time something actually asks
    for it -- which, for a LiteLLM proxy, is when it resolves the callback
    string in ``litellm_settings``.
    """
    if name == "proxy_handler_instance":
        from . import middleware
        return middleware.proxy_handler_instance
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ReveniumGuardrail",
    "MiddlewareHandler",
    "proxy_handler_instance",
]
