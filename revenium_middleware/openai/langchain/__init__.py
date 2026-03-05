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

            # Attach handler to LLM
            if hasattr(llm, 'callbacks'):
                if llm.callbacks is None:
                    llm.callbacks = [handler]
                elif isinstance(llm.callbacks, list):
                    llm.callbacks.append(handler)
                else:
                    # Convert to list if it's something else
                    llm.callbacks = [llm.callbacks, handler]
            else:
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
