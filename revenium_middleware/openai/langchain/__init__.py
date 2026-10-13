"""
LangChain integration for Revenium middleware OpenAI.

This module provides zero-touch integration with LangChain applications through
callback handlers that automatically track usage and costs.

Usage:
    from revenium_middleware.openai.langchain import wrap
    from langchain_openai import ChatOpenAI
    
    llm = wrap(ChatOpenAI(model="gpt-4o-mini"))
    response = llm.invoke("Hello Revenium!")

Installation:
    pip install revenium-python-sdk[openai,langchain]
"""

import logging
import threading
from typing import Any, TYPE_CHECKING

# Only import types during type checking to avoid runtime dependencies
if TYPE_CHECKING:
    try:
        # LangChain 1.0+ uses langchain_core
        from langchain_core.language_models import BaseLanguageModel  # noqa: F401
        from langchain_core.embeddings import Embeddings  # noqa: F401
    except ImportError:
        # LangChain 0.x uses langchain.schema and langchain.embeddings.base
        from langchain.schema import BaseLanguageModel  # noqa: F401
        from langchain.embeddings.base import Embeddings  # noqa: F401

# Import our enhanced dependency checking utilities
from ._utils import require_langchain_or_raise, is_langchain_available  # noqa: F401
from ._abandoned_runs import install_abandoned_run_cleanup
from ._perplexity_cost_capture import install_perplexity_cost_capture
from ._text_completion_usage import install_text_completion_usage_capture

install_perplexity_cost_capture()
install_text_completion_usage_capture()
install_abandoned_run_cleanup()

logger = logging.getLogger("revenium_middleware.langchain")

_WARNED_NO_CALLBACKS_CLASSES = set()
_WARNED_NO_CALLBACKS_LOCK = threading.Lock()


def _warn_no_callbacks_field_once(llm) -> None:
    from langchain_core.embeddings import Embeddings as LangChainEmbeddings

    model_class = type(llm)
    with _WARNED_NO_CALLBACKS_LOCK:
        if model_class in _WARNED_NO_CALLBACKS_CLASSES:
            return
        _WARNED_NO_CALLBACKS_CLASSES.add(model_class)
    if isinstance(llm, LangChainEmbeddings):
        logger.warning(
            "wrap() and attach_to() cannot attach Revenium tracking to %s: LangChain embeddings models run no "
            "callbacks, so the usage_metadata given to wrap() is not applied. OpenAI and Azure OpenAI embeddings "
            "are still recorded by the OpenAI middleware; tag them with @revenium_metadata. Embeddings that "
            "reach a vendor client with its own Revenium middleware, such as Ollama, are recorded by that "
            "middleware. Embeddings from other vendors are not metered.",
            model_class.__name__,
        )
    else:
        logger.warning(
            "wrap() and attach_to() cannot attach Revenium tracking to %s: it has no callbacks field, so "
            "nothing LangChain will call was attached. Wrap the chat model or LLM inside it, or pass the handler in "
            "config={'callbacks': [handler]}.",
            model_class.__name__,
        )


# Lazy loading implementation using __getattr__
def __getattr__(name: str) -> Any:
    """
    Implement lazy loading for LangChain-dependent functionality.

    This allows the module to be imported without LangChain installed,
    but raises clear errors when trying to use LangChain-specific features.
    """
    if name in ("wrap", "attach_to"):
        require_langchain_or_raise(f"the {name}() function")
        # Import the unified handler and create wrapper functions
        from .unified_handler import UnifiedReveniumCallbackHandler  # noqa: F401

        def wrap(llm, **kwargs):
            """Wrap a LangChain LLM with Revenium usage tracking."""
            handler = UnifiedReveniumCallbackHandler(**kwargs)

            # Attach handler to LLM, replacing any Revenium handler attached before
            if hasattr(llm, 'callbacks'):
                if llm.callbacks is None:
                    existing = []
                elif isinstance(llm.callbacks, list):
                    existing = llm.callbacks
                else:
                    # Convert to list if it's something else
                    existing = [llm.callbacks]
                llm.callbacks = [
                    c for c in existing if not isinstance(c, UnifiedReveniumCallbackHandler)
                ] + [handler]
            else:
                _warn_no_callbacks_field_once(llm)
                # Try to add callbacks attribute
                try:
                    llm.callbacks = [handler]
                except Exception:
                    # Some models don't support callbacks
                    pass

            return llm

        def attach_to(llm, **kwargs):
            """Attach Revenium usage tracking to an existing LangChain LLM in-place."""
            return wrap(llm, **kwargs)

        # Cache the functions in the module namespace
        if name == "wrap":
            globals()[name] = wrap
            return wrap
        else:  # attach_to
            globals()[name] = attach_to
            return attach_to

    elif name in ("ReveniumCallbackHandler", "UnifiedReveniumCallbackHandler"):
        require_langchain_or_raise(f"the {name} class")
        from .unified_handler import UnifiedReveniumCallbackHandler  # noqa: F401

        # Both names point to the same unified handler
        globals()["ReveniumCallbackHandler"] = UnifiedReveniumCallbackHandler
        globals()["UnifiedReveniumCallbackHandler"] = UnifiedReveniumCallbackHandler

        return UnifiedReveniumCallbackHandler

    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


# Convenience function for checking availability
def check_availability() -> bool:
    """
    Check if LangChain is available for use with this integration.

    Returns:
        True if LangChain is available, False otherwise
    """
    return is_langchain_available()


# Define __all__ for explicit exports
__all__ = [
    "wrap",
    "attach_to",
    "ReveniumCallbackHandler",
    "UnifiedReveniumCallbackHandler",
    "check_availability",
]
