"""
Context tracking for selective metering and metadata injection with decorators.

This module provides thread-safe and async-safe context tracking to determine
whether code is currently executing inside a decorated function that should be metered,
and to store metadata that should be injected into API calls.
"""

import contextlib
import contextvars
import re
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

# Context variable holding agentic job fields (wire-name keys) for the current scope.
# This is the seam the public JobContext (BACK-777 Phase 2) builds on.
_agentic_job_context: contextvars.ContextVar[Optional[Dict[str, Any]]] = contextvars.ContextVar(
    'revenium_agentic_job', default=None
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


_CAMEL_BOUNDARY = re.compile(r"(?<!^)(?=[A-Z])")


def _canonical_metadata_key(key: Any) -> Any:
    """Return the snake_case spelling of a metadata key.

    Every aliased metadata field in this SDK is accepted as snake_case and as
    its exact camelCase conversion (``agent_version`` / ``agentVersion``,
    ``ticket_id`` / ``ticketId``), so a key is mapped to its snake_case form
    and two keys are the same field only when those forms match. This is
    deliberately narrower than stripping underscores and lower-casing:
    arbitrary custom keys such as ``foo_bar`` and ``foobar`` are distinct
    fields and must not collide. Non-string keys are returned unchanged and
    can only ever collide with themselves.
    """
    if not isinstance(key, str):
        return key
    return _CAMEL_BOUNDARY.sub("_", key).lower()


def merge_metadata(api_metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Merge injected metadata with API-level metadata.

    API-level metadata takes precedence over injected metadata, including
    when the two sides spell the same field differently.

    Args:
        api_metadata: Metadata passed directly to the API call

    Returns:
        Merged metadata dictionary with API-level metadata taking precedence
    """
    injected = get_injected_metadata() or {}
    api = api_metadata or {}

    if injected and api:
        # Merging by literal key alone keeps a scoped ``agent_version``
        # alongside a direct ``agentVersion``. The alias precedence applied
        # downstream then picks whichever spelling it looks for first, which
        # can resolve the scoped value and invert the precedence documented
        # above. Drop scoped keys the direct call already supplies under any
        # spelling, so the direct value is the only one left to resolve.
        api_keys = {_canonical_metadata_key(key) for key in api}
        injected = {
            key: value
            for key, value in injected.items()
            if _canonical_metadata_key(key) not in api_keys
        }

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


def get_agentic_job_fields() -> Optional[Dict[str, Any]]:
    """Return the agentic job fields set on the current context, or None.

    Keys are wire names (``agenticJobId``, ``agenticJobName``, ``agenticJobType``,
    ``agenticJobVersion``) — the same keys ``extract_agentic_job_fields`` emits,
    so the fallback merge in ``_core/fields.py`` is key-aligned.
    """
    return _agentic_job_context.get()


def set_agentic_job_fields(
    job_id: Optional[str] = None,
    name: Optional[str] = None,
    type: Optional[str] = None,
    version: Optional[str] = None,
) -> contextvars.Token:
    """Set agentic job fields on the current context. Returns a Token for reset().

    Maps snake-case arguments to wire-name keys, omitting Nones. Callers own the
    token and must reset it (``_agentic_job_context.reset(token)``) when the scope
    ends — the JobContext context manager in Phase 2 does this on exit.

    Raises:
        ValueError: if no field is provided.
    """
    fields: Dict[str, Any] = {}
    if job_id is not None:
        fields["agenticJobId"] = job_id
    if name is not None:
        fields["agenticJobName"] = name
    if type is not None:
        fields["agenticJobType"] = type
    if version is not None:
        fields["agenticJobVersion"] = version
    if not fields:
        raise ValueError("set_agentic_job_fields requires at least one field")
    return _agentic_job_context.set(fields)
