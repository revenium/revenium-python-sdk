"""Metering error visibility: status counters and subscriber callbacks.

Metering runs in background threads and must never raise into the customer's
code path, so failures were historically visible only as log lines. This
module gives customers two ways to learn about metering failures
programmatically:

- ``on_metering_error(callback)`` — subscribe to failures as they happen.
- ``get_metering_status()`` — poll a snapshot of success/error counters and
  the most recent error.

Both are re-exported from the top-level ``revenium_middleware`` package.
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, List, Optional

logger = logging.getLogger("revenium_middleware")


@dataclass(frozen=True)
class MeteringErrorEvent:
    """A single metering delivery failure, passed to error callbacks."""

    error: BaseException
    operation: Optional[str]
    timestamp: datetime


@dataclass(frozen=True)
class MeteringStatus:
    """Point-in-time snapshot of metering delivery health."""

    error_count: int
    success_count: int
    last_error: Optional[BaseException]
    last_error_at: Optional[datetime]


_lock = threading.Lock()
_error_count = 0
_success_count = 0
_last_error: Optional[BaseException] = None
_last_error_at: Optional[datetime] = None
_error_callbacks: List[Callable[[MeteringErrorEvent], None]] = []


def get_metering_status() -> MeteringStatus:
    """Return a snapshot of metering success/error counters."""
    with _lock:
        return MeteringStatus(
            error_count=_error_count,
            success_count=_success_count,
            last_error=_last_error,
            last_error_at=_last_error_at,
        )


def on_metering_error(
    callback: Callable[[MeteringErrorEvent], None],
) -> Callable[[MeteringErrorEvent], None]:
    """Subscribe to metering failures.

    The callback receives a :class:`MeteringErrorEvent` for every recorded
    failure. Callbacks run on the (background) thread that detected the
    failure; exceptions they raise are logged and suppressed so they can
    never disrupt the customer's AI calls. Returns the callback, so it can
    be used as a decorator.
    """
    with _lock:
        _error_callbacks.append(callback)
    return callback


def remove_metering_error_callback(
    callback: Callable[[MeteringErrorEvent], None],
) -> None:
    """Unsubscribe a callback previously registered via ``on_metering_error``."""
    with _lock:
        try:
            _error_callbacks.remove(callback)
        except ValueError:
            pass


def reset_metering_status() -> None:
    """Reset counters, last error, and registered callbacks."""
    global _error_count, _success_count, _last_error, _last_error_at
    with _lock:
        _error_count = 0
        _success_count = 0
        _last_error = None
        _last_error_at = None
        _error_callbacks.clear()


def record_metering_success() -> None:
    """Record one successfully delivered metering event."""
    global _success_count
    with _lock:
        _success_count += 1


_SENSITIVE_HEADERS = ("x-api-key", "authorization")


def _redact_sensitive_headers(error: BaseException) -> None:
    """Redact auth headers on any HTTP request/response attached to ``error``.

    HTTP client exceptions (e.g. ``APIStatusError``) expose the live
    ``httpx.Request``/``httpx.Response`` as public attributes, and httpx does
    not redact the ``x-api-key`` header the SDK authenticates with. Overwrite
    ``x-api-key`` and ``authorization`` (case-insensitive, only if already
    present) in place before the error reaches callbacks or ``last_error``.
    Never raises, and each target is attempted independently: httpx's
    ``Response.request`` property raises ``RuntimeError`` (not
    ``AttributeError``) when no request was attached, and a failure on one
    lookup must not defeat redaction of the objects that are reachable.
    """
    targets = []
    for owner, attr in ((error, "request"), (error, "response")):
        try:
            targets.append(getattr(owner, attr, None))
        except Exception:
            targets.append(None)
    try:
        targets.append(getattr(targets[1], "request", None))
    except Exception:
        pass
    for obj in targets:
        try:
            headers = getattr(obj, "headers", None)
            if headers is None:
                continue
            for name in _SENSITIVE_HEADERS:
                if name in headers:
                    headers[name] = "[REDACTED]"
        except Exception:
            continue


def record_metering_error(
    error: BaseException, operation: Optional[str] = None
) -> None:
    """Record a metering delivery failure and notify subscribers.

    Auth headers on any HTTP request/response carried by ``error`` are
    redacted in place before the error is stored or fanned out. Never
    raises: subscriber exceptions are logged and suppressed so error
    reporting cannot disrupt the customer's code path.
    """
    global _error_count, _last_error, _last_error_at
    _redact_sensitive_headers(error)
    event = MeteringErrorEvent(
        error=error, operation=operation, timestamp=datetime.now(timezone.utc)
    )
    with _lock:
        _error_count += 1
        _last_error = error
        _last_error_at = event.timestamp
        callbacks = list(_error_callbacks)
    for callback in callbacks:
        try:
            callback(event)
        except Exception:
            logger.warning(
                "Metering error callback %r raised; ignoring", callback, exc_info=True
            )
