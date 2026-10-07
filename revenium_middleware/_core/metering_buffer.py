"""Store-and-forward buffer for metering events that exhaust retries.

Events that fail with a retryable error after the HTTP client's own retries
are buffered here instead of being discarded, and replayed by a background
daemon thread when the backend becomes reachable again. Memory-only, bounded,
with FIFO eviction and a 24h event TTL aligned with the backend's
Idempotency-Key window.

It also holds metering tasks the delivery queue had no room for: each flush
first runs them with delivery deferred, so they build and buffer their event
instead of sending it, and the event then replays like any other. AI records
the delivery circuit holds back during an outage arrive here the same way, and
the first delivery that succeeds afterwards wakes the flush thread to replay them.
"""
from __future__ import annotations

import asyncio
import contextlib
import contextvars
import logging
import queue
import threading
import time
from collections import deque
from typing import Any, Callable, Deque, Dict, Iterator, List, NamedTuple, Optional, Protocol, Set, Tuple

import httpx

from revenium_middleware._core.config import read_env_number
from revenium_middleware._core.delivery_circuit import get_circuit
from revenium_middleware._core.metering_status import (
    record_metering_error,
    record_metering_eviction,
    record_metering_success,
)

logger = logging.getLogger("revenium_middleware")

# Eleven minutes of records at 30 calls a second (5.5 at two records a call),
# about 45 MB of typical LiteLLM records at the measured 2.3 KB each.
DEFAULT_MAX_SIZE = 20_000
DEFAULT_FLUSH_INTERVAL = 30.0
DEFAULT_MAX_AGE_SECONDS = 24 * 60 * 60
# httpx rejects a timeout of zero or less with a ValueError, which the replay
# would classify as a permanent failure and discard; a spent budget must
# instead fail as a timeout, which is retryable and leaves the event buffered.
MIN_REPLAY_TIMEOUT_SECONDS = 0.001
# Replays in flight at once while a backlog drains: at a 300 ms round trip
# that replays about 50 records a second next to the live traffic.
REPLAY_CONCURRENCY = 16

# 409 is deliberately absent (diverging from the vendored client's blanket
# retry): the backend's idempotency_key_mismatch 409 is permanent and would
# otherwise circle in the buffer until TTL, while the retryable
# idempotency_key_in_progress 409 carries Retry-After and is caught by the
# header rule below. Matches the Go and Node SDKs.
_RETRYABLE_STATUS_CODES = frozenset({408, 429})


AI_KIND = "ai"
TOOL_KIND = "tool"
OVERFLOW_KIND = "overflow"


class _Deferral(NamedTuple):
    enqueued_at: float
    counted: bool


_deferral: contextvars.ContextVar[Optional[_Deferral]] = contextvars.ContextVar(
    "revenium_delivery_deferral", default=None
)


@contextlib.contextmanager
def delivery_deferred_to_buffer(enqueued_at: float, counted: bool = True) -> Iterator[None]:
    """Make metering submissions in this context buffer their event instead of sending it.

    ``enqueued_at`` is when the task entered the buffer; the event it builds
    keeps that age, so the buffer's TTL still counts from then. ``counted`` says
    the buffer already counted the task (an overflowed one); a record the
    delivery circuit held back is counted when its event is buffered.
    """
    token = _deferral.set(_Deferral(enqueued_at, counted))
    try:
        yield
    finally:
        _deferral.reset(token)


def is_delivery_deferred_to_buffer() -> bool:
    return _deferral.get() is not None


def buffer_deferred_event(kind: str, payload: Dict[str, Any]) -> None:
    """Buffer the event a deferred task just built, in place of sending it."""
    deferral = _deferral.get()
    if deferral is None:
        raise RuntimeError("buffer_deferred_event called outside delivery_deferred_to_buffer")
    get_buffer()._append(kind, payload, deferral.enqueued_at, counted=deferral.counted)


class BuildDeadlineExceeded(Exception):
    """An overflow task's build ran out of time before it buffered its event, which is lost."""


class OverflowTask(Protocol):
    """A metering task that overflowed the delivery queue before building its payload.

    ``blocks_synchronously``: building it runs synchronous code that a build
    timeout cannot cut short.
    """

    blocks_synchronously: bool

    def materialize(self, enqueued_at: float, timeout: Optional[float] = None) -> None:
        """Build the event and buffer it (under ``delivery_deferred_to_buffer(enqueued_at)``).

        With ``timeout``, give up on a build still awaiting after that many
        seconds and raise ``BuildDeadlineExceeded``; the task is settled either way.
        """

    def discard(self) -> None:
        """Drop the task without running it."""


class BufferedEvent:
    """One undelivered metering event plus everything needed to replay it."""

    __slots__ = ("kind", "payload", "enqueued_at")

    def __init__(self, kind: str, payload: Dict[str, Any], enqueued_at: float):
        self.kind = kind  # AI_KIND | TOOL_KIND | OVERFLOW_KIND
        self.payload = payload
        self.enqueued_at = enqueued_at


def _discard_if_overflow(event: BufferedEvent) -> None:
    if event.kind == OVERFLOW_KIND:
        event.payload["task"].discard()


def _settle_evicted(events: List[BufferedEvent]) -> None:
    """Count evicted events in the metering status and drop their overflow tasks.

    Called only after the buffer lock is released: a thread inside
    ``get_metering_status`` holds the status lock and can be interrupted by a
    finalizer that meters into the buffer, so taking the status lock under the
    buffer lock could leave the two threads waiting for each other.
    """
    if events:
        record_metering_eviction(len(events))
    _discard_overflow(events)


def _discard_overflow(events: List[BufferedEvent]) -> None:
    for event in events:
        _discard_if_overflow(event)


class _OverflowBuild:
    """Builds overflow tasks in order on one thread; anyone can stop it and take back the unstarted ones."""

    def __init__(self, overflow: List[BufferedEvent]):
        self._overflow = overflow
        # Reentrant: the opt-in SIGTERM handler can run the exit drain, which
        # reclaims and counts builds, on a main thread inside one of these.
        self._lock = threading.RLock()
        self._next = 0
        self._built = 0
        self._cut_short = 0
        self._stopped = False
        self._finished = threading.Event()

    def run(self, until: Optional[float] = None) -> None:
        """Build every task; with ``until``, a ``time.monotonic()`` value, start none after it and bound each by it."""
        try:
            while True:
                with self._lock:
                    if self._stopped or self._next == len(self._overflow):
                        return
                    timeout = None if until is None else until - time.monotonic()
                    if timeout is not None and timeout <= 0:
                        return
                    event = self._overflow[self._next]
                    self._next += 1
                cut_short = self._materialize(event, timeout)
                with self._lock:
                    self._built += 1
                    self._cut_short += cut_short
        finally:
            self._finished.set()

    @staticmethod
    def _materialize(event: BufferedEvent, timeout: Optional[float]) -> int:
        """Build one task; 1 when the deadline cut it short, else 0."""
        try:
            event.payload["task"].materialize(event.enqueued_at, timeout)
        except BuildDeadlineExceeded:
            return 1
        return 0

    def wait(self, timeout: Optional[float]) -> None:
        self._finished.wait(timeout)

    def add(self, event: BufferedEvent) -> None:
        """Take on one more task, built after the ones already taken unless the build was stopped."""
        with self._lock:
            self._overflow.append(event)

    def reclaim(self) -> List[BufferedEvent]:
        """Stop starting tasks and hand back the unstarted ones; later calls return only tasks added since."""
        with self._lock:
            self._stopped = True
            unstarted = self._overflow[self._next:]
            self._overflow = self._overflow[:self._next]
            return unstarted

    def set_aside_blocking(self) -> List[BufferedEvent]:
        """Hand back the unstarted tasks that block synchronously; the others stay in this build."""
        with self._lock:
            unstarted = self._overflow[self._next:]
            blocking = [event for event in unstarted if event.payload["task"].blocks_synchronously]
            if blocking:
                kept = [event for event in unstarted if not event.payload["task"].blocks_synchronously]
                self._overflow = self._overflow[:self._next] + kept
            return blocking

    def unbuilt(self) -> int:
        """Tasks this build took on and has not finished, excluding any reclaimed."""
        with self._lock:
            return len(self._overflow) - self._built

    def cut_short(self) -> int:
        """Tasks the deadline stopped before they buffered their event."""
        with self._lock:
            return self._cut_short


class _BuildWorker:
    """One long-lived daemon thread that runs overflow builds in turn.

    It starts with the first overflowed task, so the exit drain never has to
    start a thread to build: an interpreter shutting down may refuse to
    (Python 3.12.0 to 3.12.2 do, from ``atexit``), as may a process at its
    thread limit.
    """

    def __init__(self) -> None:
        self._jobs: "queue.SimpleQueue[Callable[[], None]]" = queue.SimpleQueue()
        self._lock = threading.RLock()
        self._thread: Optional[threading.Thread] = None
        self._starting = False
        self._unfinished_jobs = 0

    def ensure_started(self) -> bool:
        """Start the thread unless it is running or starting; False when no thread can be started now.

        A call re-entered while this thread is starting the build thread (a
        finalizer that meters) counts on that thread; should it fail to
        start, the caller's build wait is what bounds the queued job.
        """
        with self._lock:
            if self._starting or (self._thread is not None and self._thread.is_alive()):
                return True
            self._starting = True
            try:
                return self._start()
            finally:
                self._starting = False

    def _start(self) -> bool:
        thread = threading.Thread(target=self._work, name="MeteringOverflowBuild", daemon=True)
        try:
            thread.start()
        except RuntimeError as exc:
            logger.debug("Could not start the overflow build thread: %s", exc)
            return False
        self._thread = thread
        return True

    def submit(self, job: Callable[[], None]) -> bool:
        """Queue ``job`` behind the ones already queued; False when there is no thread to run it."""
        if not self.ensure_started():
            return False
        with self._lock:
            self._unfinished_jobs += 1
        self._jobs.put(job)
        return True

    def is_busy(self) -> bool:
        """Whether a job is running or queued, so a new one would wait for it."""
        with self._lock:
            return self._unfinished_jobs > 0

    def is_current_thread(self) -> bool:
        return self._thread is threading.current_thread()

    def _work(self) -> None:
        while True:
            job = self._jobs.get()
            try:
                job()
            except BaseException as exc:  # noqa: BLE001 - the builds queued behind this one still need the thread
                logger.warning("Overflow build failed: %s", exc)
            finally:
                del job
                with self._lock:
                    self._unfinished_jobs -= 1


def _event_loop_running_here() -> bool:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return False
    return True


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
    if event.kind == AI_KIND:
        _replay_ai_event(event.payload, timeout_seconds)
    elif event.kind == TOOL_KIND:
        _replay_tool_event(event.payload, timeout_seconds)
    else:  # unknown kinds are a programming error; treat as permanent
        raise ValueError(f"Unknown buffered event kind: {event.kind!r}")


class _ReplayOutcome(NamedTuple):
    sent: int
    ai_sent: int
    discarded: int
    retry: List[BufferedEvent]
    status: List[Optional[Tuple[BaseException, str]]]


def _replay_concurrently(
    replay_fn: Callable[[BufferedEvent, float], None], batch: List[BufferedEvent], timeout_seconds: float
) -> List[Optional[Exception]]:
    """Replay ``batch`` with one thread per event after the first, which runs here; each event's error or None."""
    errors: List[Optional[Exception]] = [None] * len(batch)

    def replay(index: int) -> None:
        try:
            replay_fn(batch[index], timeout_seconds)
        except Exception as exc:  # noqa: BLE001 - classified by the flush that waits on the batch
            errors[index] = exc

    threads = []
    for index in range(1, len(batch)):
        thread = threading.Thread(target=replay, args=(index,), name="MeteringReplay", daemon=True)
        try:
            thread.start()
        except RuntimeError:
            replay(index)
            continue
        threads.append(thread)
    replay(0)
    for thread in threads:
        thread.join()
    return errors


def _sort_outcomes(batch: List[BufferedEvent], errors: List[Optional[Exception]]) -> _ReplayOutcome:
    sent = ai_sent = discarded = 0
    retry: List[BufferedEvent] = []
    status: List[Optional[Tuple[BaseException, str]]] = []
    for event, error in zip(batch, errors):
        if error is None:
            sent += 1
            ai_sent += event.kind == AI_KIND
            status.append(None)
        elif is_retryable_failure(error):
            logger.debug("Buffer flush stopped replaying %s events; their endpoint is still unreachable: %s",
                         event.kind, error)
            retry.append(event)
        else:
            discarded += 1
            status.append((error, event.kind))
            logger.debug("Discarded buffered event after permanent failure: %s", error)
    return _ReplayOutcome(sent, ai_sent, discarded, retry, status)


class _BatchLimits(NamedTuple):
    """How many events one replay batch takes, and how many of those may be AI events."""

    size: int
    ai: int


def _expiry_error(kind: str, max_age_seconds: float) -> TimeoutError:
    """Expiry is a terminal failure with no delivery exception in hand; this one carries the signal."""
    return TimeoutError(
        f"buffered {kind} metering event expired after {max_age_seconds:.0f}s without successful replay"
    )


class MeteringBuffer:
    """Thread-safe bounded FIFO buffer with a periodic replay thread.

    Pushes may be re-entered on the same thread, so its locks are reentrant:
    stream wrappers meter from finalizers, which the garbage collector runs
    on whichever thread it interrupts, including one inside a push.
    """

    def __init__(
        self,
        max_size: int = DEFAULT_MAX_SIZE,
        flush_interval: float = DEFAULT_FLUSH_INTERVAL,
        max_age_seconds: float = DEFAULT_MAX_AGE_SECONDS,
        replay_fn: Optional[Callable[[BufferedEvent, float], None]] = None,
        now_fn: Callable[[], float] = time.time,
        replay_concurrency: Optional[int] = None,
        replay_timeout_seconds: Optional[float] = None,
    ):
        """Without ``replay_concurrency``, replay ``REPLAY_CONCURRENCY`` events at once with the default replay, else 1.

        ``replay_timeout_seconds`` caps one replay attempt; by default, ``REVENIUM_METERING_TIMEOUT_SECONDS``.
        """
        if replay_timeout_seconds is None:
            from revenium_middleware._core.metering import metering_client_timeout

            replay_timeout_seconds = metering_client_timeout().read
        self._replay_timeout_seconds = replay_timeout_seconds
        self._max_size = max_size
        self._flush_interval = flush_interval
        self._max_age_seconds = max_age_seconds
        self._replay_fn = replay_fn or _default_replay
        if replay_concurrency is None:
            replay_concurrency = REPLAY_CONCURRENCY if replay_fn is None else 1
        self._replay_concurrency = replay_concurrency
        self._now_fn = now_fn

        self._events: Deque[BufferedEvent] = deque()
        # Records of a kind whose endpoint failed during the running flush,
        # ahead of everything in _events; the flush returns them when it ends.
        self._parked: Deque[BufferedEvent] = deque()
        self._lock = threading.RLock()
        self._flush_lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._active_builds: Set[_OverflowBuild] = set()
        self._exit_build_started = False
        self._build_worker = _BuildWorker()
        self._was_full = False
        self._evicted_unreported = 0
        self._replays_in_flight = 0
        self._replay_requested = threading.Event()

        self._total_buffered = 0
        self._total_replayed = 0
        self._total_evicted = 0
        self._total_expired = 0
        self._total_discarded = 0
        self._total_overflowed = 0
        self._total_cut_short = 0

    def push_overflow(self, task: OverflowTask) -> None:
        """Hold a task the delivery queue had no room for until a flush builds it."""
        with self._lock:
            self._total_overflowed += 1
        self._build_worker.ensure_started()
        self.push(OVERFLOW_KIND, {"task": task})

    def push(self, kind: str, payload: Dict[str, Any]) -> None:
        """Buffer one undelivered event, evicting the oldest when full."""
        self._append(kind, payload, self._now_fn(), counted=False)

    def _append(self, kind: str, payload: Dict[str, Any], enqueued_at: float, counted: bool) -> None:
        with self._lock:
            evicted = self._evict_down_to(self._max_size - 1)
            self._events.append(BufferedEvent(kind, payload, enqueued_at))
            if not counted:
                self._total_buffered += 1
            depth = self._held()
        _settle_evicted(evicted)
        logger.debug("Buffered undelivered %s metering event (buffer depth: %d)", kind, depth)
        self._ensure_flush_thread()

    def flush(self, deadline_seconds: Optional[float] = None) -> Dict[str, int]:
        """Replay buffered events oldest-first within each kind.

        AI and tool events go to endpoints that can fail apart, so a retryable
        failure (endpoint still unreachable) stops replaying that kind only:
        its events are parked, each visited once, and go back to the front in
        order when the flush ends, while the other kind's events behind them
        are still replayed. Stops when every kind has failed, on the deadline,
        or when the buffer is drained. Permanent failures and expired events
        are discarded. Without a deadline, up to ``replay_concurrency`` events
        are replayed at once, no more than one of them an AI event while the
        delivery circuit is open; with a deadline (the exit drain), one at a
        time. Serialized: concurrent flushes queue up, and one with a deadline
        stops waiting for the flush in progress when the deadline passes,
        replaying nothing.
        """
        sent = expired = discarded = 0
        started = time.monotonic()

        def remaining_budget() -> Optional[float]:
            if deadline_seconds is None:
                return None
            return max(0.0, deadline_seconds - (time.monotonic() - started))

        self.materialize_overflow(remaining_budget())
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
            return {"sent": 0, "expired": 0, "discarded": 0, "remaining": self.undelivered()}
        with self._lock:
            late_overflow = self._register_build([])
        unreachable_kinds: Set[str] = set()
        try:
            while deadline_seconds is None or time.monotonic() - started < deadline_seconds:
                batch, expired_events = self._take_replay_batch(
                    self._batch_limits(deadline_seconds), unreachable_kinds, late_overflow
                )
                deferred_outcomes.extend(self._expire(expired_events))
                expired += len(expired_events)
                if not batch:
                    break

                errors = _replay_concurrently(self._replay_fn, batch, self._replay_timeout(deadline_seconds, started))
                outcome = _sort_outcomes(batch, errors)
                self._settle_replays(outcome)
                sent += outcome.sent
                discarded += outcome.discarded
                deferred_outcomes.extend(outcome.status)
                unreachable_kinds.update(event.kind for event in outcome.retry)
        finally:
            self._unpark()
            self._flush_lock.release()

        for status in deferred_outcomes:
            if status is None:
                record_metering_success()
            else:
                error, kind = status
                record_metering_error(error, operation=kind)
        # Built outside _flush_lock for the same reason status is recorded
        # there: a task that fails reports to subscriber callbacks
        # synchronously. Building only pushes the event back as a regular
        # one, so it is neither a replay nor a delivery success.
        self._build_unless_exiting(late_overflow, remaining_budget())

        remaining = self.undelivered()
        if sent or expired or discarded:
            logger.debug(
                "Buffer flush: %d replayed, %d expired, %d discarded, %d remaining",
                sent, expired, discarded, remaining,
            )
        return {"sent": sent, "expired": expired, "discarded": discarded, "remaining": remaining}

    def _batch_limits(self, deadline_seconds: Optional[float]) -> _BatchLimits:
        if deadline_seconds is not None:
            return _BatchLimits(size=1, ai=1)
        if get_circuit().is_open():
            return _BatchLimits(size=self._replay_concurrency, ai=1)
        return _BatchLimits(size=self._replay_concurrency, ai=self._replay_concurrency)

    def _replay_timeout(self, deadline_seconds: Optional[float], started: float) -> float:
        """Cap each replay call so a single slow network call cannot blow through the flush deadline.

        The remaining budget strictly bounds the call: a tiny timeout just
        fails fast, which is better than letting a slow call overrun the deadline.
        Never below ``MIN_REPLAY_TIMEOUT_SECONDS``, even once the budget is spent.
        """
        if deadline_seconds is None:
            return self._replay_timeout_seconds
        remaining = deadline_seconds - (time.monotonic() - started)
        return max(MIN_REPLAY_TIMEOUT_SECONDS, min(self._replay_timeout_seconds, remaining))

    def _take_replay_batch(
        self, limits: _BatchLimits, unreachable_kinds: Set[str], late_overflow: _OverflowBuild
    ) -> Tuple[List[BufferedEvent], List[BufferedEvent]]:
        """Take the oldest replayable events ``limits`` allows, plus the expired ones met on the way.

        Events of ``unreachable_kinds`` are parked; an AI event beyond
        ``limits.ai`` ends the batch in front of it. Either way no event is
        visited twice in one flush. Overflowed tasks met on the way join
        ``late_overflow``, a registered build the exit drain can take over.
        Taken events are out of the buffer while they replay, so a push at
        capacity cannot evict one that is being delivered, and counted in
        ``undelivered`` until ``_settle_replays`` accounts for them.
        """
        batch: List[BufferedEvent] = []
        expired: List[BufferedEvent] = []
        ai_taken = 0
        with self._lock:
            while self._events and len(batch) < limits.size:
                front = self._events[0]
                if front.kind == AI_KIND and AI_KIND not in unreachable_kinds and ai_taken == limits.ai:
                    break
                event = self._events.popleft()
                if self._now_fn() - event.enqueued_at > self._max_age_seconds:
                    self._total_expired += 1
                    expired.append(event)
                elif event.kind == OVERFLOW_KIND:
                    late_overflow.add(event)
                elif event.kind in unreachable_kinds:
                    self._parked.append(event)
                else:
                    batch.append(event)
                    ai_taken += event.kind == AI_KIND
            self._replays_in_flight += len(batch)
        return batch, expired

    def _expire(self, expired: List[BufferedEvent]) -> List[Optional[Tuple[BaseException, str]]]:
        """Drop ``expired`` events already taken from the buffer; return the errors status should record."""
        for event in expired:
            _discard_if_overflow(event)
        return [(_expiry_error(event.kind, self._max_age_seconds), event.kind) for event in expired]

    def _settle_replays(self, outcome: _ReplayOutcome) -> None:
        """Count a batch's outcome, park its retryable failures and report AI ones to the circuit.

        A failure parks its kind for the rest of the flush, and no event of
        that kind is parked before it, so parking keeps each kind in order.
        """
        with self._lock:
            self._replays_in_flight -= outcome.sent + outcome.discarded + len(outcome.retry)
            self._total_replayed += outcome.sent
            self._total_discarded += outcome.discarded
            self._parked.extend(outcome.retry)
            evicted = self._evict_down_to(self._max_size)
            if outcome.sent and self._held() < self._max_size:
                self._was_full = False
        _settle_evicted(evicted)
        # A batch that delivered any AI event shows the endpoint answering; the
        # records that failed in it go back to the front and are retried.
        if outcome.ai_sent:
            get_circuit().record_success()
        elif any(event.kind == AI_KIND for event in outcome.retry):
            get_circuit().record_failure()

    def _unpark(self) -> None:
        """Put the events the flush parked back in front, in the order they were parked."""
        with self._lock:
            self._events.extendleft(reversed(self._parked))
            self._parked.clear()

    def _held(self) -> int:
        """Events in the buffer, parked ones included; caller holds ``_lock``."""
        return len(self._events) + len(self._parked)

    def _evict_down_to(self, size: int) -> List[BufferedEvent]:
        """Evict oldest-first, parked events before the rest, until at most ``size`` remain; caller holds ``_lock``.

        The caller hands the result to ``_settle_evicted`` once it has released
        the lock.
        """
        evicted: List[BufferedEvent] = []
        while self._held() > size:
            evicted.append((self._parked or self._events).popleft())
            self._total_evicted += 1
            self._evicted_unreported += 1
            if not self._was_full:
                logger.warning(
                    "Metering buffer full (%d events); evicting oldest events, which are lost",
                    self._max_size,
                )
                self._was_full = True
        return evicted

    def undelivered(self) -> int:
        """Events buffered plus those a flush has taken out to replay and not yet accounted for."""
        with self._lock:
            return self._held() + self._replays_in_flight

    def report_evictions(self) -> None:
        """Log how many events were evicted since the last report, if any."""
        with self._lock:
            evicted, self._evicted_unreported = self._evicted_unreported, 0
            total = self._total_evicted
        if evicted:
            logger.warning(
                "Metering buffer evicted %d undelivered event(s) since its last report because it was full "
                "(%d in total); their usage is lost",
                evicted, total,
            )

    def request_replay(self) -> None:
        """Wake the flush thread to replay now rather than at its next interval."""
        self._replay_requested.set()

    def materialize_overflow(self, deadline_seconds: Optional[float] = None) -> int:
        """Build every waiting overflow task's event now, oldest first; return how many are left unbuilt.

        Tasks the deadline leaves unbuilt go back to the buffer, as do all of
        them once the exit drain has started building (see ``build_all_overflow``).
        """
        with self._lock:
            build = self._register_build(self._take_overflow())
        return self._build_unless_exiting(build, deadline_seconds)

    def build_all_overflow(self, deadline_seconds: float) -> int:
        """Like ``materialize_overflow``, but also takes over tasks another flush is still building.

        Their unstarted tasks are built here, including those a flush has
        taken out of the buffer and not started; the ones they are building
        are waited for with what is left of the deadline. Returns how many are left unbuilt.

        From this call on, no other build starts: the exit drain sets
        ``shutdown_event`` next, and a task built after that skips itself
        without buffering an event, so the exit warning could not count it.
        """
        started = time.monotonic()
        with self._lock:
            self._exit_build_started = True
            in_progress = list(self._active_builds)
            reclaimed = [event for build in in_progress for event in build.reclaim()]
            building = [build for build in in_progress if build.unbuilt()]
            takeover = self._register_build(reclaimed + self._take_overflow())
        # A build blocked in its first task holds the build thread; queued
        # behind it, the takeover would build nothing before the deadline.
        unbuilt = self._build_overflow(takeover, deadline_seconds, own_thread_if_worker_busy=True)
        for build in building:
            build.wait(max(0.0, deadline_seconds - (time.monotonic() - started)))
            unbuilt += build.unbuilt()
        return unbuilt

    def _register_build(self, overflow: List[BufferedEvent]) -> _OverflowBuild:
        """A build of ``overflow`` that ``build_all_overflow`` can take over; caller holds ``_lock``."""
        build = _OverflowBuild(overflow)
        self._active_builds.add(build)
        return build

    def _build_unless_exiting(self, build: _OverflowBuild, deadline_seconds: Optional[float]) -> int:
        """Run ``build``, or return its tasks to the buffer unbuilt once the exit drain has started building."""
        with self._lock:
            exiting = self._exit_build_started
            if exiting:
                self._active_builds.discard(build)
                unstarted = build.reclaim()
        if not exiting:
            return self._build_overflow(build, deadline_seconds)
        self._return_to_front(unstarted)
        return len(unstarted)

    def _build_overflow(
        self, build: _OverflowBuild, deadline_seconds: Optional[float], own_thread_if_worker_busy: bool = False
    ) -> int:
        """Run a registered build, waiting at most ``deadline_seconds``; return how many tasks are unbuilt.

        Tasks not started by then go back to the buffer. The one being built
        when the wait ends finishes on its thread and buffers its event then.
        With ``own_thread_if_worker_busy``, a build thread busy with another
        build is passed over for a thread started for this one.
        """
        if not build.unbuilt():
            self._forget_build(build)
            return 0
        worker = self._build_worker
        if worker.is_current_thread():
            return self._build_on_this_thread(build, deadline_seconds)
        # A task builds on an event loop of its own, which cannot run on a
        # thread already running one (an async host shutting down), and it can
        # block past the deadline, synchronously where wait_for cannot cut it
        # short; a thread other than the caller's absorbs both.
        if own_thread_if_worker_busy and worker.is_busy():
            handed_off = self._start_build_thread(build)
        else:
            handed_off = worker.submit(lambda: self._run_build(build))
        if not handed_off:
            return self._build_on_this_thread(build, deadline_seconds)
        build.wait(deadline_seconds)
        return self._return_unstarted(build)

    def _start_build_thread(self, build: _OverflowBuild) -> bool:
        thread = threading.Thread(target=self._run_build, args=(build,), name="MeteringOverflowTakeover", daemon=True)
        try:
            thread.start()
        except RuntimeError as exc:
            logger.debug("Could not start a thread for the overflow takeover: %s", exc)
            return False
        return True

    def _build_on_this_thread(self, build: _OverflowBuild, deadline_seconds: Optional[float]) -> int:
        """Build here, by the deadline if there is one.

        With a deadline, tasks that block synchronously stay buffered unbuilt;
        on a thread running an event loop, every task does.
        """
        if _event_loop_running_here():
            self._forget_build(build)
            unstarted = build.reclaim()
            logger.warning("No thread could build %d overflowed metering event(s) and this one runs an event loop; "
                           "they stay buffered unbuilt", len(unstarted))
            self._return_to_front(unstarted)
            return len(unstarted)
        if deadline_seconds is None:
            return self._build_here_until(build, None)
        until = time.monotonic() + deadline_seconds
        blocking = build.set_aside_blocking()
        self._return_to_front(blocking)
        if blocking:
            logger.warning("No thread could build %d overflowed metering event(s) that run synchronous code, which "
                           "the %.3fs deadline could not cut short; they stay buffered unbuilt",
                           len(blocking), deadline_seconds)
        return self._build_here_until(build, until) + len(blocking)

    def _build_here_until(self, build: _OverflowBuild, until: Optional[float]) -> int:
        try:
            build.run(until)
        finally:
            self._forget_build(build)
        return self._return_unstarted(build)

    def _return_unstarted(self, build: _OverflowBuild) -> int:
        unstarted = build.reclaim()
        self._return_to_front(unstarted)
        return len(unstarted) + build.unbuilt()

    def _run_build(self, build: _OverflowBuild) -> None:
        try:
            build.run()
        finally:
            self._forget_build(build)

    def overflow_in_build(self) -> int:
        """Overflowed tasks a build has taken out of the buffer and not finished."""
        with self._lock:
            builds = list(self._active_builds)
        return sum(build.unbuilt() for build in builds)

    def overflow_lost_to_deadlines(self) -> int:
        """Overflowed tasks whose build ran out of time before buffering their event, since this buffer was created."""
        with self._lock:
            return self._total_cut_short + sum(build.cut_short() for build in self._active_builds)

    def _forget_build(self, build: _OverflowBuild) -> None:
        with self._lock:
            if build in self._active_builds:
                self._active_builds.discard(build)
                self._total_cut_short += build.cut_short()

    def _return_to_front(self, events: List[BufferedEvent]) -> None:
        if not events:
            return
        with self._lock:
            self._events.extendleft(reversed(events))
            evicted = self._evict_down_to(self._max_size)
        _settle_evicted(evicted)

    def _take_overflow(self) -> List[BufferedEvent]:
        """Remove and return the waiting overflow tasks; caller holds ``_lock``."""
        overflow = [event for event in self._events if event.kind == OVERFLOW_KIND]
        if overflow:
            self._events = deque(event for event in self._events if event.kind != OVERFLOW_KIND)
        return overflow

    def stats(self) -> Dict[str, int]:
        """Snapshot of the counters.

        ``size`` counts every event not yet delivered, including those a flush
        is replaying right now. ``total_buffered`` counts each event once, when
        it enters the buffer: an overflowed task counts when it overflows (also
        in ``total_overflowed``), not again when its event is built.
        """
        with self._lock:
            return {
                "size": self._held() + self._replays_in_flight,
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
            thread = threading.Thread(target=self._run, name="MeteringBufferFlush", daemon=True)
            try:
                thread.start()
            except RuntimeError as exc:
                logger.debug("Could not start the buffer flush thread; the exit drain still flushes: %s", exc)
                return
            self._thread = thread

    def _run(self) -> None:
        from revenium_middleware._core.metering import shutdown_event

        while True:
            self._replay_requested.wait(self._flush_interval)
            self._replay_requested.clear()
            # Final drain happens in handle_exit(), which owns the shutdown budget.
            if shutdown_event.is_set():
                return
            try:
                self.flush()
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("Metering buffer flush error: %s", exc)
            self.report_evictions()


_buffer: Optional[MeteringBuffer] = None
# Reentrant for the same reason as the buffer's own locks.
_buffer_init_lock = threading.RLock()


def _is_usable_flush_interval(seconds: float) -> bool:
    """Above zero, so the flush thread pauses between replays, and a wait ``threading.Event`` accepts."""
    return 0 < seconds <= threading.TIMEOUT_MAX


def _whole_number(raw: str) -> int:
    """Accept "500", "500.0" and "1e3" alike, as this variable always has."""
    return int(float(raw))


def get_buffer() -> MeteringBuffer:
    """Return the process-wide buffer singleton (created on first use).

    A call re-entered while this thread builds the buffer (a finalizer that
    meters) publishes its own first; the outer call keeps that one and drops
    its own, which holds no events and no threads yet.
    """
    global _buffer
    if _buffer is None:
        with _buffer_init_lock:
            if _buffer is None:
                built = MeteringBuffer(
                    max_size=read_env_number(
                        "REVENIUM_BUFFER_MAX_SIZE", DEFAULT_MAX_SIZE, _whole_number, lambda size: size >= 1
                    ),
                    flush_interval=read_env_number(
                        "REVENIUM_BUFFER_FLUSH_INTERVAL", DEFAULT_FLUSH_INTERVAL, float, _is_usable_flush_interval
                    ),
                )
                if _buffer is None:
                    _buffer = built
    return _buffer


def request_replay() -> None:
    """Wake the buffer's flush thread, if there is a buffer, to replay what it holds now."""
    if _buffer is not None:
        _buffer.request_replay()


def get_buffer_stats() -> Dict[str, int]:
    """Snapshot of the buffer's counters for programmatic observability."""
    return get_buffer().stats()
