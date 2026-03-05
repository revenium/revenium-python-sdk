"""
Context tracking for selective metering and metadata injection with decorators.

This module provides thread-safe and async-safe context tracking to determine
whether code is currently executing inside a decorated function that should be metered,
and to store metadata that should be injected into API calls.
"""

import contextvars
from typing import Optional, Dict, Any

# Context variable to track if we're inside a decorated function
_decorated_function_context: contextvars.ContextVar[bool] = contextvars.ContextVar(
    'revenium_decorated_function', default=False
)

# Context variable to store metadata from the current decorated function
_function_metadata_context: contextvars.ContextVar[Optional[Dict[str, Any]]] = contextvars.ContextVar(
    'revenium_function_metadata', default=None
)

# Context variable to store injected metadata from @revenium_metadata decorator
_injected_metadata_context: contextvars.ContextVar[Optional[Dict[str, Any]]] = contextvars.ContextVar(
    'revenium_injected_metadata', default=None
)


def is_inside_decorated_function() -> bool:
    """
    Check if code is currently executing inside a decorated function.
    
    Returns:
        True if inside a decorated function, False otherwise
    """
    return _decorated_function_context.get()


def get_function_metadata() -> Optional[Dict[str, Any]]:
    """
    Get metadata from the current decorated function context.
    
    Returns:
        Dictionary of metadata or None if not in decorated function
    """
    return _function_metadata_context.get()


def set_decorated_context(is_decorated: bool, metadata: Optional[Dict[str, Any]] = None) -> None:
    """
    Set the decorated function context.
    
    Args:
        is_decorated: Whether we're inside a decorated function
        metadata: Optional metadata from the decorator
    """
    _decorated_function_context.set(is_decorated)
    _function_metadata_context.set(metadata)


def clear_decorated_context() -> None:
    """Clear the decorated function context."""
    _decorated_function_context.set(False)
    _function_metadata_context.set(None)


def get_injected_metadata() -> Optional[Dict[str, Any]]:
    """
    Get metadata from the current @revenium_metadata decorator context.

    Returns:
        Dictionary of injected metadata or None if not in decorated function
    """
    return _injected_metadata_context.get()


def set_injected_metadata(metadata: Optional[Dict[str, Any]]) -> None:
    """
    Set the injected metadata context.

    Args:
        metadata: Dictionary of metadata to inject into API calls
    """
    _injected_metadata_context.set(metadata)


def clear_injected_metadata() -> None:
    """Clear the injected metadata context."""
    _injected_metadata_context.set(None)


def merge_metadata(api_metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Merge injected metadata with API-level metadata.

    API-level metadata takes precedence over injected metadata.

    Args:
        api_metadata: Metadata passed directly to the API call

    Returns:
        Merged metadata dictionary with API-level metadata taking precedence
    """
    injected = get_injected_metadata() or {}
    api = api_metadata or {}

    # Start with injected metadata, then override with API-level metadata
    merged = {**injected, **api}
    return merged

