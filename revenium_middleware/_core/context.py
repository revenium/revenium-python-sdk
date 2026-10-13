"""
Context tracking for selective metering and metadata injection with decorators.

This module provides thread-safe and async-safe context tracking to determine
whether code is currently executing inside a decorated function that should be metered,
and to store metadata that should be injected into API calls.
"""

import contextlib
import contextvars
import functools
import hashlib
import logging
import os
import re
import threading
from typing import Optional, Dict, Any, Callable, Iterator, TypeVar, Union

logger = logging.getLogger("revenium_middleware")

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

# The backend rejects longer keys with 400 invalid_idempotency_key.
_MAX_IDEMPOTENCY_KEY_LENGTH = 255

IDEMPOTENCY_KEY_RULE = (
    "must be 1 to 255 printable ASCII characters, with no line breaks, accented or non-Latin letters, "
    "or leading or trailing spaces"
)


def _is_sendable_idempotency_key(key: object) -> bool:
    """Whether ``key`` meets the backend's printable-ASCII rule and can go in an HTTP header unchanged."""
    return (
        isinstance(key, str)
        and 1 <= len(key) <= _MAX_IDEMPOTENCY_KEY_LENGTH
        and all(" " <= character <= "~" for character in key)
        and key == key.strip()
    )


def _refuse_unsendable_idempotency_key(key: object, empty_key_message: str) -> None:
    if key == "":
        raise ValueError(empty_key_message)
    if _is_sendable_idempotency_key(key):
        return
    shown = repr(key[:64]) if isinstance(key, str) else f"a {type(key).__name__}"
    raise ValueError(f"idempotency_key {IDEMPOTENCY_KEY_RULE}; got {shown}")


_scope_lock = threading.Lock()


class _IdempotencyScope:
    """One idempotency block: the caller's key and how many keys it has issued.

    Mutable and shared on purpose: the metering worker submits each event from
    a copy of the caller's context, and every copy must draw from the same
    count, or two calls in one block would get the same key.
    """

    def __init__(self, key: str) -> None:
        self.key = key
        self._issued = 0

    def next_key(self) -> str:
        with _scope_lock:
            self._issued += 1
            sequence = self._issued
        if sequence == 1:
            return self.key
        derived = f"{self.key}:{sequence}"
        if len(derived) <= _MAX_IDEMPOTENCY_KEY_LENGTH:
            return derived
        return f"{hashlib.sha256(self.key.encode()).hexdigest()}:{sequence}"


def _free_scope_lock_in_child() -> None:
    """Give a forked child a free scope lock.

    A metering worker thread holding the lock at fork time does not exist in
    the child, so the inherited lock would never be released.
    """
    global _scope_lock
    _scope_lock = threading.Lock()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_free_scope_lock_in_child)


# Context variable holding the idempotency block for the current scope. A plain
# string is tolerated for callers that set the variable directly.
_idempotency_key_context: contextvars.ContextVar[Optional[Union[_IdempotencyScope, str]]] = contextvars.ContextVar(
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


@contextlib.contextmanager
def decorated_scope(metadata: Optional[Dict[str, Any]]) -> Iterator[None]:
    """Mark the enclosed code as inside a decorated function, then restore
    whatever scope was in force before, so an enclosing decorated function
    keeps its own scope after a nested one returns."""
    decorated_token = _decorated_function_context.set(True)
    metadata_token = _function_metadata_context.set(metadata)
    try:
        yield
    finally:
        _function_metadata_context.reset(metadata_token)
        _decorated_function_context.reset(decorated_token)


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


@contextlib.contextmanager
def injected_metadata_scope(metadata: Optional[Dict[str, Any]]) -> Iterator[None]:
    """Lay ``metadata`` over the injected metadata already in force for the
    enclosed code, then restore it, so a nested decorator adds to the outer
    one's tags and the outer function keeps them after the inner returns."""
    if not metadata:
        yield
        return
    token = _injected_metadata_context.set(overlay_metadata(get_injected_metadata(), metadata))
    try:
        yield
    finally:
        _injected_metadata_context.reset(token)


@contextlib.contextmanager
def replaced_injected_metadata_scope(metadata: Optional[Dict[str, Any]]) -> Iterator[None]:
    """Install exactly ``metadata`` as the injected metadata for the enclosed
    code, then restore whatever was in force before."""
    token = _injected_metadata_context.set(metadata)
    try:
        yield
    finally:
        _injected_metadata_context.reset(token)


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
    when the two sides spell the same field differently. An API-level value
    of None or "" does not replace an injected value that is set.

    Args:
        api_metadata: Metadata passed directly to the API call

    Returns:
        Merged metadata dictionary with API-level metadata taking precedence
    """
    return overlay_metadata(get_injected_metadata(), api_metadata)


def is_blank_metadata_value(value: Any) -> bool:
    """True for the two values that never make a meaningful tag: None and "".

    Zero, False and empty containers are real values."""
    return value is None or (isinstance(value, str) and value == "")


# The record reads these legacy id names as the name field when the name is
# absent, so a blank name must not hide a value set under the id name.
_BLANK_OVERRIDE_FIELD_ALIASES = {"organization_id": "organization_name", "product_id": "product_name"}


def _blank_override_field(key: Any) -> Any:
    canonical = _canonical_metadata_key(key)
    return _BLANK_OVERRIDE_FIELD_ALIASES.get(canonical, canonical)


def _set_fields(metadata: Dict[str, Any]) -> set:
    return {_blank_override_field(key) for key, value in metadata.items() if not is_blank_metadata_value(value)}


def _without_blanks_over_set_values(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    set_fields = _set_fields(base)
    set_fields |= _set_fields(override)
    return {
        key: value for key, value in override.items()
        if not (is_blank_metadata_value(value) and _blank_override_field(key) in set_fields)
    }


def overlay_metadata(base: Optional[Dict[str, Any]], override: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """``base`` with every field of ``override`` taking precedence, under either spelling.

    An override of None or "" does not replace a value ``base`` sets for the
    same field; with nothing set underneath, it passes through unchanged."""
    base = base or {}
    override = override or {}

    if base and override:
        override = _without_blanks_over_set_values(base, override)
        # Merging by literal key alone keeps a scoped ``agent_version``
        # alongside a direct ``agentVersion``. The alias precedence applied
        # downstream then picks whichever spelling it looks for first, which
        # can resolve the base value and invert the precedence. Drop base
        # keys the override already supplies under any spelling, so the
        # override is the only one left to resolve.
        override_keys = {_canonical_metadata_key(key) for key in override}
        override_fields = _set_fields(override)
        base = {
            key: value
            for key, value in base.items()
            if _canonical_metadata_key(key) not in override_keys and _blank_override_field(key) not in override_fields
        }

    return snapshot_metadata({**base, **override})


def snapshot_metadata(value: Any) -> Any:
    """Return ``value`` with every dict, list and tuple in it copied, so the
    caller changing its own containers after a call cannot change that call's
    record, which is built later on the metering thread.

    Any other value is kept as the same object. Never raises: a container
    that cannot be read, or nests too deep to copy, is kept as passed.
    """
    return _snapshot(value, {})


def _snapshot(value: Any, copies: Dict[int, Any]) -> Any:
    if not isinstance(value, (dict, list)) and type(value) is not tuple:
        return value
    if id(value) in copies:
        return copies[id(value)]
    try:
        return _copy_container(value, copies)
    except Exception as exc:  # noqa: BLE001 - metering must never break callers
        copies.pop(id(value), None)
        logger.debug("Kept a metadata value uncopied: %s", type(exc).__name__)
        return value


def _copy_container(value: Any, copies: Dict[int, Any]) -> Any:
    if isinstance(value, dict):
        copied_dict: Dict[Any, Any] = {}
        copies[id(value)] = copied_dict
        for key, item in list(value.items()):
            copied_dict[key] = _snapshot(item, copies)
        return copied_dict
    if isinstance(value, list):
        copied_list: list = []
        copies[id(value)] = copied_list
        copied_list.extend([_snapshot(item, copies) for item in value])
        return copied_list
    copied_tuple = tuple([_snapshot(item, copies) for item in value])
    copies[id(value)] = copied_tuple
    return copied_tuple


def get_idempotency_key() -> Optional[str]:
    """Return the Idempotency-Key override set on the current context, or None."""
    scope = _idempotency_key_context.get()
    if isinstance(scope, _IdempotencyScope):
        return scope.key
    return scope


def next_idempotency_key() -> Optional[str]:
    """Return the key for the next metered call in the current idempotency block, or None outside one.

    The first call gets the block's key unchanged and call N gets ``key:N``, so
    every call in the block is recorded and rerunning the block maps its Nth
    call to the same key. When ``key:N`` would be longer than the backend
    accepts, the key's SHA-256 hex digest stands in for it.
    """
    scope = _idempotency_key_context.get()
    if isinstance(scope, _IdempotencyScope):
        return scope.next_key()
    return scope


def set_idempotency_key(key: Optional[str]) -> contextvars.Token:
    """Set the Idempotency-Key override on the current context. Returns a Token for reset().

    Note: passing ``None`` is indistinguishable from "no override set" to consumers — both
    cause the wrapper to fall through to UUID v4 generation. Use this function with a real
    string for explicit overrides; rely on the ``idempotency_key()`` context manager for
    scoped overrides that auto-reset on exit.

    Raises:
        ValueError: if ``key`` is the empty string, or is not 1 to 255 printable
            ASCII characters without leading or trailing spaces. Mirrors the
            guard on the ``idempotency_key()`` CM so a key Revenium cannot
            accept never reaches the wrapper via any entry point.
    """
    if key is not None:
        _refuse_unsendable_idempotency_key(
            key,
            "idempotency_key must be a non-empty string; "
            "the backend rejects empty keys",
        )
    return _idempotency_key_context.set(None if key is None else _IdempotencyScope(key))


@contextlib.contextmanager
def idempotency_key(key: str) -> Iterator[None]:
    """Override the auto-generated Idempotency-Key for AI metering calls within the block.

    Args:
        key: The explicit key to use: 1 to 255 printable ASCII characters,
            with no leading or trailing spaces. Sent unchanged.

    Raises:
        ValueError: if ``key`` breaks that rule, before anything in the block
            runs. The backend would reject such a key, or the HTTP client could
            not send it, and the provider middleware would swallow the error, so
            the usage record would be lost; failing fast at the CM surfaces the
            bug at its source instead.

    Example:
        from revenium_middleware import idempotency_key

        with idempotency_key(f"order-{order_id}"):
            response = openai.chat.completions.create(...)
    """
    _refuse_unsendable_idempotency_key(
        key,
        "idempotency_key must be a non-empty string; "
        "the backend rejects empty keys and the SDK's middleware swallows the error",
    )
    token = _idempotency_key_context.set(_IdempotencyScope(key))
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


T = TypeVar("T")


def carry_context(fn: Callable[..., T]) -> Callable[..., T]:
    """Return ``fn`` bound to the caller's context, for running on a worker thread.

    ``ThreadPoolExecutor``, ``threading.Thread`` and ``loop.run_in_executor``
    start their target in an empty context, so AI calls made there are outside
    the ``@revenium_meter`` function that started them: skipped under selective
    metering, and recorded without the ``@revenium_metadata`` tags otherwise.
    The returned callable runs ``fn`` with everything in force where
    ``carry_context`` was called: the metering mark, the tags, and any
    ``metering_key()`` or ``idempotency_key()`` block.

    Call it inside the decorated function, each time you hand work to a thread,
    for example ``pool.map(carry_context(summarize), docs)``. It is not a
    decorator: a wrap made at import time or outside the decorated function
    carries nothing. Each call runs in its own copy of the captured context, so
    workers never see each other's changes. Wrap a plain function; a generator
    function's body runs in the context of whoever iterates it. Wrapping an
    ``async def`` carries nothing, because its coroutine runs later on the event
    loop, outside the copied context: wrap the sync function the thread runs
    instead, or copy the context around the awaited call.
    """
    captured = contextvars.copy_context()

    @functools.wraps(fn)
    def run_in_captured_context(*args: Any, **kwargs: Any) -> T:
        return captured.copy().run(fn, *args, **kwargs)

    return run_in_captured_context
