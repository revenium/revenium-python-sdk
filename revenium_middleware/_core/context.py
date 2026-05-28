"""
Context tracking for selective metering and metadata injection with decorators.

This module provides thread-safe and async-safe context tracking to determine
whether code is currently executing inside a decorated function that should be metered,
and to store metadata that should be injected into API calls.
"""

import contextlib
import contextvars
from typing import Optional, Dict, Any, Iterator

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

# Context variable to store an Idempotency-Key override for the current scope
_idempotency_key_context: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    'revenium_idempotency_key', default=None
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


def get_idempotency_key() -> Optional[str]:
    """Return the Idempotency-Key override set on the current context, or None."""
    return _idempotency_key_context.get()


def set_idempotency_key(key: Optional[str]) -> contextvars.Token:
    """Set the Idempotency-Key override on the current context. Returns a Token for reset().

    Note: passing ``None`` is indistinguishable from "no override set" to consumers — both
    cause the wrapper to fall through to UUID v4 generation. Use this function with a real
    string for explicit overrides; rely on the ``idempotency_key()`` context manager for
    scoped overrides that auto-reset on exit.

    Raises:
        ValueError: if ``key`` is the empty string. Mirrors the guard on the
            ``idempotency_key()`` CM so empty strings can never reach the wrapper
            via any entry point.
    """
    if key == "":
        raise ValueError(
            "idempotency_key must be a non-empty string; "
            "the backend rejects empty keys"
        )
    return _idempotency_key_context.set(key)


@contextlib.contextmanager
def idempotency_key(key: str) -> Iterator[None]:
    """Override the auto-generated Idempotency-Key for AI metering calls within the block.

    Args:
        key: The explicit key to use. Must be a non-empty string. Forwarded
            verbatim to the backend, which enforces the 1-255 printable ASCII
            constraint.

    Raises:
        ValueError: if ``key`` is the empty string. The backend would reject
            empty keys with ``400 invalid_idempotency_key``, which the
            provider middleware then swallows — failing fast at the CM
            surfaces the bug at its source instead.

    Example:
        from revenium_middleware import idempotency_key

        with idempotency_key(f"order-{order_id}"):
            response = openai.chat.completions.create(...)
    """
    if key == "":
        raise ValueError(
            "idempotency_key must be a non-empty string; "
            "the backend rejects empty keys and the SDK's middleware swallows the error"
        )
    token = _idempotency_key_context.set(key)
    try:
        yield
    finally:
        _idempotency_key_context.reset(token)

