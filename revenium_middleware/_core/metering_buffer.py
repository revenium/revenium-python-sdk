"""Store-and-forward buffer for metering events that exhaust retries.

Events that fail with a retryable error after the HTTP client's own retries
are buffered here instead of being discarded, and replayed by a background
daemon thread when the backend becomes reachable again. Memory-only, bounded,
with FIFO eviction and a 24h event TTL aligned with the backend's
Idempotency-Key window.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from collections import deque
from typing import Any, Callable, Deque, Dict, Optional

import httpx

logger = logging.getLogger("revenium_middleware")

DEFAULT_MAX_SIZE = 1000
DEFAULT_FLUSH_INTERVAL = 30.0
DEFAULT_MAX_AGE_SECONDS = 24 * 60 * 60
REPLAY_TIMEOUT_SECONDS = 10.0

# 409 is deliberately absent (diverging from the vendored client's blanket
# retry): the backend's idempotency_key_mismatch 409 is permanent and would
# otherwise circle in the buffer until TTL, while the retryable
# idempotency_key_in_progress 409 carries Retry-After and is caught by the
# header rule below. Matches the Go and Node SDKs.
_RETRYABLE_STATUS_CODES = frozenset({408, 429})


class BufferedEvent:
    """One undelivered metering event plus everything needed to replay it."""

    __slots__ = ("kind", "payload", "enqueued_at")

    def __init__(self, kind: str, payload: Dict[str, Any], enqueued_at: float):
        self.kind = kind  # "ai" | "tool"
        self.payload = payload
        self.enqueued_at = enqueued_at


def _status_is_retryable(status_code: int, headers: Any) -> bool:
    """Mirror the vendored HTTP client's _should_retry status semantics."""
    should_retry_header = ""
    if headers is not None:
        should_retry_header = headers.get("x-should-retry", "")
    if should_retry_header == "true":
        return True
    if should_retry_header == "false":
        return False
    # A Retry-After header is an explicit backend invitation to retry,
    # whatever the status code (e.g. 409 idempotency_key_in_progress).
    if headers is not None and headers.get("retry-after"):
        return True
    return status_code in _RETRYABLE_STATUS_CODES or status_code >= 500


def is_retryable_failure(exc: BaseException) -> bool:
    """Whether a metering delivery failure is worth buffering for replay.

    Transient failures (connection/timeout errors, 408/409/429, 5xx, or an
    explicit ``x-should-retry: true``) qualify. Permanent failures -- other
    4xx such as 401/403/404/422 -- must never be buffered.
    """
    # Vendored metering-client exceptions (AI events).
    from revenium_middleware._metering._exceptions import (
        APIConnectionError,
        APIStatusError,
    )

    if isinstance(exc, APIConnectionError):
        return True
    if isinstance(exc, APIStatusError):
        response = getattr(exc, "response", None)
        if response is None:
            return False
        return _status_is_retryable(response.status_code, response.headers)

    # Raw httpx failures (tool events).
    if isinstance(exc, httpx.HTTPStatusError):
        return _status_is_retryable(exc.response.status_code, exc.response.headers)
    if isinstance(exc, (httpx.TimeoutException, httpx.TransportError)):
        return True

    return False


def _replay_ai_event(payload: Dict[str, Any], timeout_seconds: float) -> None:
    """Re-submit a buffered AI event through the metering client."""
    from revenium_middleware._core.metering import get_client

    client = get_client()
    if client is None:
        # Backend credentials unavailable right now: treat as transient so
        # the event stays buffered for the next flush cycle.
        raise httpx.ConnectError("metering client not configured")

    method = getattr(client.ai, f"create_{payload['operation']}")
    kwargs = dict(payload["args"])
    # Replay owns the timeout: the flush budget must win over any frozen
    # caller-supplied value (which applied to the original call, not replay).
    kwargs["timeout"] = timeout_seconds
    method(**kwargs)


def _replay_tool_event(payload: Dict[str, Any], timeout_seconds: float) -> None:
    """Re-POST a buffered tool event with its original payload."""
    # Replay after an outage should honor credential rotation, matching the
    # AI path (which resolves the current client at replay time). Fall back
    # to the endpoint frozen at dispatch when nothing is configured now.
    from revenium_middleware._metering.decorator import _resolve_endpoint

    url, key = _resolve_endpoint()
    if url is None or key is None:
        url, key = payload["url"], payload["key"]

    with httpx.Client(timeout=timeout_seconds) as client:
        response = client.post(
            url,
            headers={
                "x-api-key": key,
                "Content-Type": "application/json",
                "Idempotency-Key": payload["event_payload"]["transactionId"],
            },
            json=payload["event_payload"],
        )
        response.raise_for_status()


def _default_replay(event: BufferedEvent, timeout_seconds: float) -> None:
    if event.kind == "ai":
        _replay_ai_event(event.payload, timeout_seconds)
    elif event.kind == "tool":
        _replay_tool_event(event.payload, timeout_seconds)
    else:  # unknown kinds are a programming error; treat as permanent
        raise ValueError(f"Unknown buffered event kind: {event.kind!r}")


class MeteringBuffer:
    """Thread-safe bounded FIFO buffer with a periodic replay thread."""

    def __init__(
        self,
        max_size: int = DEFAULT_MAX_SIZE,
        flush_interval: float = DEFAULT_FLUSH_INTERVAL,
        max_age_seconds: float = DEFAULT_MAX_AGE_SECONDS,
        replay_fn: Optional[Callable[[BufferedEvent, float], None]] = None,
        now_fn: Callable[[], float] = time.time,
    ):
        self._max_size = max_size
        self._flush_interval = flush_interval
        self._max_age_seconds = max_age_seconds
        self._replay_fn = replay_fn or _default_replay
        self._now_fn = now_fn

        self._events: Deque[BufferedEvent] = deque()
        self._lock = threading.Lock()
        self._flush_lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._was_full = False

        self._total_buffered = 0
        self._total_replayed = 0
        self._total_evicted = 0
        self._total_expired = 0
        self._total_discarded = 0

    def push(self, kind: str, payload: Dict[str, Any]) -> None:
        """Buffer one undelivered event, evicting the oldest when full."""
        with self._lock:
            if len(self._events) >= self._max_size:
                self._events.popleft()
                self._total_evicted += 1
                if not self._was_full:
                    logger.warning(
                        "Metering buffer full (%d events); evicting oldest events",
                        self._max_size,
                    )
                    self._was_full = True
            self._events.append(BufferedEvent(kind, payload, self._now_fn()))
            self._total_buffered += 1
            depth = len(self._events)
        logger.debug("Buffered undelivered %s metering event (buffer depth: %d)", kind, depth)
        self._ensure_flush_thread()

    def flush(self, deadline_seconds: Optional[float] = None) -> Dict[str, int]:
        """Replay buffered events oldest-first.

        Stops at the first retryable failure (backend still unreachable), on
        the deadline, or when the buffer is drained. Permanent failures and
        expired events are discarded. Serialized: concurrent flushes queue up.
        """
        sent = expired = discarded = 0
        started = time.monotonic()

        with self._flush_lock:
            while True:
                if deadline_seconds is not None and time.monotonic() - started >= deadline_seconds:
                    break

                with self._lock:
                    if not self._events:
                        break
                    event = self._events[0]
                    if self._now_fn() - event.enqueued_at > self._max_age_seconds:
                        self._events.popleft()
                        self._total_expired += 1
                        expired += 1
                        continue

                # Cap each replay call so a single slow network call cannot
                # blow through the flush deadline (e.g. the shutdown budget).
                if deadline_seconds is None:
                    per_call_timeout = REPLAY_TIMEOUT_SECONDS
                else:
                    elapsed = time.monotonic() - started
                    # The remaining budget strictly bounds the call: a tiny
                    # timeout just fails fast, which is better than letting a
                    # slow call overrun the deadline.
                    per_call_timeout = min(REPLAY_TIMEOUT_SECONDS, deadline_seconds - elapsed)

                try:
                    self._replay_fn(event, per_call_timeout)
                except Exception as exc:  # noqa: BLE001 - classified below
                    if is_retryable_failure(exc):
                        logger.debug("Buffer flush stopped; backend still unreachable: %s", exc)
                        break
                    # Only count the discard if the event is still at the
                    # front; a concurrent push at capacity may have evicted
                    # (and counted) it already.
                    with self._lock:
                        if self._events and self._events[0] is event:
                            self._events.popleft()
                            self._total_discarded += 1
                            discarded += 1
                    logger.debug("Discarded buffered event after permanent failure: %s", exc)
                    continue

                # Same identity guard: only count the replay if we actually
                # popped this event (not concurrently evicted-and-counted).
                with self._lock:
                    if self._events and self._events[0] is event:
                        self._events.popleft()
                        self._total_replayed += 1
                        if len(self._events) < self._max_size:
                            self._was_full = False
                        sent += 1

        remaining = self.stats()["size"]
        if sent or expired or discarded:
            logger.debug(
                "Buffer flush: %d replayed, %d expired, %d discarded, %d remaining",
                sent, expired, discarded, remaining,
            )
        return {"sent": sent, "expired": expired, "discarded": discarded, "remaining": remaining}

    def stats(self) -> Dict[str, int]:
        with self._lock:
            return {
                "size": len(self._events),
                "max_size": self._max_size,
                "total_buffered": self._total_buffered,
                "total_replayed": self._total_replayed,
                "total_evicted": self._total_evicted,
                "total_expired": self._total_expired,
                "total_discarded": self._total_discarded,
            }

    def _ensure_flush_thread(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._thread = threading.Thread(
                target=self._run, name="MeteringBufferFlush", daemon=True
            )
            self._thread.start()

    def _run(self) -> None:
        from revenium_middleware._core.metering import shutdown_event

        while not shutdown_event.wait(self._flush_interval):
            try:
                self.flush()
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("Metering buffer flush error: %s", exc)
        # Final drain happens in handle_exit(), which owns the shutdown budget.


_buffer: Optional[MeteringBuffer] = None
_buffer_init_lock = threading.Lock()


def _read_env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning("Invalid %s=%r; using default %s", name, raw, default)
        return default


def get_buffer() -> MeteringBuffer:
    """Return the process-wide buffer singleton (created on first use)."""
    global _buffer
    if _buffer is None:
        with _buffer_init_lock:
            if _buffer is None:
                _buffer = MeteringBuffer(
                    max_size=int(_read_env_float("REVENIUM_BUFFER_MAX_SIZE", DEFAULT_MAX_SIZE)),
                    flush_interval=_read_env_float(
                        "REVENIUM_BUFFER_FLUSH_INTERVAL", DEFAULT_FLUSH_INTERVAL
                    ),
                )
    return _buffer


def get_buffer_stats() -> Dict[str, int]:
    """Snapshot of the buffer's counters for programmatic observability."""
    return get_buffer().stats()
