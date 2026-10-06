"""Store-and-forward buffer for metering events that exhaust retries.

Events that fail with a retryable error after the HTTP client's own retries
are buffered here instead of being discarded, and replayed by a background
daemon thread when the backend becomes reachable again. Memory-only, bounded,
with FIFO eviction and a 24h event TTL aligned with the backend's
Idempotency-Key window.

It also holds metering tasks the delivery queue had no room for: each flush
first runs them with delivery deferred, so they build and buffer their event
instead of sending it, and the event then replays like any other.
"""
from __future__ import annotations

import contextlib
import contextvars
import logging
import threading
import time
from collections import deque
from typing import Any, Callable, Deque, Dict, Iterator, List, Optional, Protocol, Set, Tuple

import httpx

from revenium_middleware._core.config import read_env_number
from revenium_middleware._core.metering_status import (
    record_metering_error,
    record_metering_success,
)

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


OVERFLOW_KIND = "overflow"

_deferred_since: contextvars.ContextVar[Optional[float]] = contextvars.ContextVar(
    "revenium_delivery_deferred_since", default=None
)


@contextlib.contextmanager
def delivery_deferred_to_buffer(enqueued_at: float) -> Iterator[None]:
    """Make metering submissions in this context buffer their event instead of sending it.

    ``enqueued_at`` is when the overflowed task entered the buffer; the event it
    builds keeps that age, so the buffer's TTL still counts from the overflow.
    """
    token = _deferred_since.set(enqueued_at)
    try:
        yield
    finally:
        _deferred_since.reset(token)


def is_delivery_deferred_to_buffer() -> bool:
    return _deferred_since.get() is not None


def buffer_deferred_event(kind: str, payload: Dict[str, Any]) -> None:
    """Buffer the event an overflowed task just built, in place of sending it."""
    enqueued_at = _deferred_since.get()
    if enqueued_at is None:
        raise RuntimeError("buffer_deferred_event called outside delivery_deferred_to_buffer")
    get_buffer().readmit(kind, payload, enqueued_at)


class OverflowTask(Protocol):
    """A metering task that overflowed the delivery queue before building its payload."""

    def materialize(self, enqueued_at: float) -> None:
        """Build the event and buffer it (under ``delivery_deferred_to_buffer(enqueued_at)``)."""

    def discard(self) -> None:
        """Drop the task without running it."""


class BufferedEvent:
    """One undelivered metering event plus everything needed to replay it."""

    __slots__ = ("kind", "payload", "enqueued_at")

    def __init__(self, kind: str, payload: Dict[str, Any], enqueued_at: float):
        self.kind = kind  # "ai" | "tool" | OVERFLOW_KIND
        self.payload = payload
        self.enqueued_at = enqueued_at


def _discard_if_overflow(event: BufferedEvent) -> None:
    if event.kind == OVERFLOW_KIND:
        event.payload["task"].discard()


def _discard_overflow(events: List[BufferedEvent]) -> None:
    for event in events:
        _discard_if_overflow(event)


class _OverflowBuild:
    """Builds overflow tasks in order on one thread; anyone can stop it and take back the unstarted ones."""

    def __init__(self, overflow: List[BufferedEvent]):
        self._overflow = overflow
        self._lock = threading.Lock()
        self._next = 0
        self._built = 0
        self._stopped = False
        self._finished = threading.Event()

    def run(self) -> None:
        try:
            while True:
                with self._lock:
                    if self._stopped or self._next == len(self._overflow):
                        return
                    event = self._overflow[self._next]
                    self._next += 1
                event.payload["task"].materialize(event.enqueued_at)
                with self._lock:
                    self._built += 1
        finally:
            self._finished.set()

    def wait(self, timeout: Optional[float]) -> None:
        self._finished.wait(timeout)

    def reclaim(self) -> List[BufferedEvent]:
        """Stop starting tasks and hand back the unstarted ones; later calls return nothing."""
        with self._lock:
            self._stopped = True
            unstarted = self._overflow[self._next:]
            self._overflow = self._overflow[:self._next]
            return unstarted

    def unbuilt(self) -> int:
        """Tasks this build took on and has not finished, excluding any reclaimed."""
        with self._lock:
            return len(self._overflow) - self._built


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

    # The buffer's own replay cycle is the retry loop; client-side retries
    # would multiply the per-call timeout and overrun a flush deadline.
    method = getattr(client.with_options(max_retries=0).ai, f"create_{payload['operation']}")
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
        self._active_builds: Set[_OverflowBuild] = set()
        self._was_full = False

        self._total_buffered = 0
        self._total_replayed = 0
        self._total_evicted = 0
        self._total_expired = 0
        self._total_discarded = 0
        self._total_overflowed = 0

    def push_overflow(self, task: OverflowTask) -> None:
        """Hold a task the delivery queue had no room for until a flush builds it."""
        with self._lock:
            self._total_overflowed += 1
        self.push(OVERFLOW_KIND, {"task": task})

    def push(self, kind: str, payload: Dict[str, Any]) -> None:
        """Buffer one undelivered event, evicting the oldest when full."""
        self._append(kind, payload, self._now_fn(), counted=False)

    def readmit(self, kind: str, payload: Dict[str, Any], enqueued_at: float) -> None:
        """Buffer the event an overflow task built; it was counted, and aged, from its overflow."""
        self._append(kind, payload, enqueued_at, counted=True)

    def _append(self, kind: str, payload: Dict[str, Any], enqueued_at: float, counted: bool) -> None:
        with self._lock:
            evicted = self._evict_down_to(self._max_size - 1)
            self._events.append(BufferedEvent(kind, payload, enqueued_at))
            if not counted:
                self._total_buffered += 1
            depth = len(self._events)
        _discard_overflow(evicted)
        logger.debug("Buffered undelivered %s metering event (buffer depth: %d)", kind, depth)
        self._ensure_flush_thread()

    def flush(self, deadline_seconds: Optional[float] = None) -> Dict[str, int]:
        """Replay buffered events oldest-first.

        Stops at the first retryable failure (backend still unreachable), on
        the deadline, or when the buffer is drained. Permanent failures and
        expired events are discarded. Serialized: concurrent flushes queue up,
        and one with a deadline stops waiting for the flush in progress when the
        deadline passes, replaying nothing.
        """
        sent = expired = discarded = 0
        started = time.monotonic()

        def remaining_budget() -> Optional[float]:
            if deadline_seconds is None:
                return None
            return max(0.0, deadline_seconds - (time.monotonic() - started))

        self.materialize_overflow(remaining_budget())
        late_overflow: List[BufferedEvent] = []
        # Status recording is deferred until _flush_lock is released:
        # record_metering_error() runs subscriber callbacks synchronously, and
        # a callback that calls back into this buffer would self-deadlock on
        # the non-reentrant lock.
        deferred_outcomes: List[Optional[Tuple[BaseException, str]]] = []

        if deadline_seconds is None:
            acquired = self._flush_lock.acquire()
        else:
            budget_left = deadline_seconds - (time.monotonic() - started)
            acquired = budget_left > 0 and self._flush_lock.acquire(timeout=budget_left)
        if not acquired:
            logger.debug("Buffer flush skipped: another flush was still running at the %ss deadline",
                         deadline_seconds)
            return {"sent": 0, "expired": 0, "discarded": 0, "remaining": self.stats()["size"]}
        try:
            while True:
                if deadline_seconds is not None and time.monotonic() - started >= deadline_seconds:
                    break

                expired_kind: Optional[str] = None
                with self._lock:
                    if not self._events:
                        break
                    event = self._events[0]
                    if self._now_fn() - event.enqueued_at > self._max_age_seconds:
                        self._events.popleft()
                        self._total_expired += 1
                        expired += 1
                        expired_kind = event.kind
                    elif event.kind == OVERFLOW_KIND:
                        late_overflow.append(self._events.popleft())
                        continue
                if expired_kind is not None:
                    _discard_if_overflow(event)
                    # Expiry is a terminal failure with no delivery exception
                    # in hand; synthesize one so status counters and
                    # on_metering_error subscribers still get the signal.
                    deferred_outcomes.append((
                        TimeoutError(
                            f"buffered {expired_kind} metering event expired after "
                            f"{self._max_age_seconds:.0f}s without successful replay"
                        ),
                        expired_kind,
                    ))
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
                    discarded_here = False
                    with self._lock:
                        if self._events and self._events[0] is event:
                            self._events.popleft()
                            self._total_discarded += 1
                            discarded += 1
                            discarded_here = True
                    if discarded_here:
                        # Terminal failure: the event is gone for good, so
                        # surface it to status counters and subscribers.
                        deferred_outcomes.append((exc, event.kind))
                    logger.debug("Discarded buffered event after permanent failure: %s", exc)
                    continue

                # Same identity guard: only count the replay if we actually
                # popped this event (not concurrently evicted-and-counted).
                replayed_here = False
                with self._lock:
                    if self._events and self._events[0] is event:
                        self._events.popleft()
                        self._total_replayed += 1
                        if len(self._events) < self._max_size:
                            self._was_full = False
                        sent += 1
                        replayed_here = True
                if replayed_here:
                    deferred_outcomes.append(None)
        finally:
            self._flush_lock.release()

        for outcome in deferred_outcomes:
            if outcome is None:
                record_metering_success()
            else:
                error, kind = outcome
                record_metering_error(error, operation=kind)
        # Built outside _flush_lock for the same reason status is recorded
        # there: a task that fails reports to subscriber callbacks
        # synchronously. Building only pushes the event back as a regular
        # one, so it is neither a replay nor a delivery success.
        self._build_overflow(late_overflow, remaining_budget())

        remaining = self.stats()["size"]
        if sent or expired or discarded:
            logger.debug(
                "Buffer flush: %d replayed, %d expired, %d discarded, %d remaining",
                sent, expired, discarded, remaining,
            )
        return {"sent": sent, "expired": expired, "discarded": discarded, "remaining": remaining}

    def _evict_down_to(self, size: int) -> List[BufferedEvent]:
        """Evict oldest-first until at most ``size`` events remain; caller holds ``_lock``."""
        evicted: List[BufferedEvent] = []
        while len(self._events) > size:
            evicted.append(self._events.popleft())
            self._total_evicted += 1
            if not self._was_full:
                logger.warning(
                    "Metering buffer full (%d events); evicting oldest events",
                    self._max_size,
                )
                self._was_full = True
        return evicted

    def materialize_overflow(self, deadline_seconds: Optional[float] = None) -> int:
        """Build every waiting overflow task's event now, oldest first; return how many are left unbuilt.

        Tasks the deadline leaves unbuilt go back to the buffer.
        """
        return self._build_overflow(self._take_overflow(), deadline_seconds)

    def build_all_overflow(self, deadline_seconds: float) -> int:
        """Like ``materialize_overflow``, but also takes over tasks another flush is still building.

        Their unstarted tasks are built here; the ones they are building are
        waited for with what is left of the deadline. Returns how many are left unbuilt.
        """
        started = time.monotonic()
        with self._lock:
            in_progress = list(self._active_builds)
        reclaimed = [event for build in in_progress for event in build.reclaim()]
        unbuilt = self._build_overflow(reclaimed + self._take_overflow(), deadline_seconds)
        for build in in_progress:
            build.wait(max(0.0, deadline_seconds - (time.monotonic() - started)))
            unbuilt += build.unbuilt()
        return unbuilt

    def _build_overflow(self, overflow: List[BufferedEvent], deadline_seconds: Optional[float]) -> int:
        """Build ``overflow`` in order, waiting at most ``deadline_seconds``; return how many are unbuilt.

        Tasks not started by then go back to the buffer. The one being built
        when the wait ends finishes on the build thread and buffers its event then.
        """
        if not overflow:
            return 0
        build = _OverflowBuild(overflow)
        with self._lock:
            self._active_builds.add(build)
        # A dedicated thread, because a task's coroutine can block past the
        # deadline and a task builds on an event loop of its own, which cannot
        # run on a thread already running one (an async host shutting down).
        builder = threading.Thread(target=self._run_build, args=(build,), name="MeteringOverflowBuild", daemon=True)
        try:
            builder.start()
        except RuntimeError as exc:
            self._forget_build(build)
            logger.warning("Could not start a thread to build %d overflowed metering event(s) (%s); "
                           "they stay buffered unbuilt", len(overflow), exc)
            self._return_overflow(overflow)
            return len(overflow)
        build.wait(deadline_seconds)
        unstarted = build.reclaim()
        self._return_overflow(unstarted)
        return len(unstarted) + build.unbuilt()

    def _run_build(self, build: _OverflowBuild) -> None:
        try:
            build.run()
        finally:
            self._forget_build(build)

    def _forget_build(self, build: _OverflowBuild) -> None:
        with self._lock:
            self._active_builds.discard(build)

    def _return_overflow(self, overflow: List[BufferedEvent]) -> None:
        with self._lock:
            self._events.extendleft(reversed(overflow))
            evicted = self._evict_down_to(self._max_size)
        _discard_overflow(evicted)

    def _take_overflow(self) -> List[BufferedEvent]:
        with self._lock:
            overflow = [event for event in self._events if event.kind == OVERFLOW_KIND]
            if overflow:
                self._events = deque(event for event in self._events if event.kind != OVERFLOW_KIND)
        return overflow

    def stats(self) -> Dict[str, int]:
        """Snapshot of the counters.

        ``total_buffered`` counts each event once, when it enters the buffer:
        an overflowed task counts when it overflows (also in
        ``total_overflowed``), not again when its event is built.
        """
        with self._lock:
            return {
                "size": len(self._events),
                "max_size": self._max_size,
                "total_buffered": self._total_buffered,
                "total_replayed": self._total_replayed,
                "total_evicted": self._total_evicted,
                "total_expired": self._total_expired,
                "total_discarded": self._total_discarded,
                "total_overflowed": self._total_overflowed,
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


def _whole_number(raw: str) -> int:
    """Accept "500", "500.0" and "1e3" alike, as this variable always has."""
    return int(float(raw))


def get_buffer() -> MeteringBuffer:
    """Return the process-wide buffer singleton (created on first use)."""
    global _buffer
    if _buffer is None:
        with _buffer_init_lock:
            if _buffer is None:
                _buffer = MeteringBuffer(
                    max_size=read_env_number(
                        "REVENIUM_BUFFER_MAX_SIZE", DEFAULT_MAX_SIZE, _whole_number, lambda size: size >= 1
                    ),
                    flush_interval=read_env_number("REVENIUM_BUFFER_FLUSH_INTERVAL", DEFAULT_FLUSH_INTERVAL, float),
                )
    return _buffer


def get_buffer_stats() -> Dict[str, int]:
    """Snapshot of the buffer's counters for programmatic observability."""
    return get_buffer().stats()
