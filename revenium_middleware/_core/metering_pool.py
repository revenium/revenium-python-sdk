"""Bounded background delivery of metering events.

Every metered call hands its metering coroutine to ``run_async_in_thread``,
which queues it here. A fixed pool of daemon worker threads drains the queue,
each worker running coroutines on an event loop it owns for its whole life.
When the queue is full the task goes to the store-and-forward buffer instead,
so a slow or unreachable metering endpoint can never grow the thread count,
block the caller, or hold more than ``queue size + buffer size`` events.
While the delivery circuit is open the workers build each record into the
buffer without sending it, so an outage does not hold them.
"""
from __future__ import annotations

import asyncio
import collections
import contextvars
import itertools
import logging
import queue
import threading
import time
from typing import Any, Callable, Coroutine, Deque, List, Optional, Tuple

from revenium_middleware._core.config import read_env_number
from revenium_middleware._core.delivery_circuit import get_circuit
from revenium_middleware._core.metering_buffer import BuildDeadlineExceeded, delivery_deferred_to_buffer, get_buffer
from revenium_middleware._core.metering_status import record_metering_error

logger = logging.getLogger("revenium_middleware")

WORKERS_ENV = "REVENIUM_METERING_WORKERS"
QUEUE_SIZE_ENV = "REVENIUM_METERING_QUEUE_SIZE"
# Each worker sends one record at a time, so 32 keep up with 30 records a
# second until a round trip takes about a second: 3.5 times the 300 ms at
# which the BACK-3919 sweep lost records with 8. Measured cost of the 32 idle
# threads together: about 1.4 MB of RSS and 96 file descriptors (their event loops).
DEFAULT_WORKERS = 32
DEFAULT_QUEUE_SIZE = 1000
# How often an idle worker wakes to check whether it was stopped, so stopping
# never depends on there being queue capacity for a wake-up item.
STOP_CHECK_SECONDS = 0.25

_task_ids = itertools.count(1)


def _never() -> bool:
    return False


class MeteringTask:
    """One metering coroutine accepted for background delivery.

    Joinable like the per-call thread it replaces, so callers that wait for
    delivery (tests, scripts) keep calling ``join``/``is_alive``.
    """

    def __init__(
        self,
        coro: Coroutine[Any, Any, Any],
        ctx: contextvars.Context,
        is_shutting_down: Callable[[], bool] = _never,
    ):
        self._coro = coro
        self._ctx = ctx
        self._is_shutting_down = is_shutting_down
        self._done = threading.Event()
        self._settle_lock = threading.Lock()
        self._settled_callbacks: List[Callable[[], None]] = []
        self.error: Optional[BaseException] = None
        self.name = f"MeteringTask-{next(_task_ids)}"

    def on_settled(self, callback: Callable[[], None]) -> None:
        """Call ``callback`` once the task has run, been built into a buffered event, or been discarded."""
        with self._settle_lock:
            if not self._done.is_set():
                self._settled_callbacks.append(callback)
                return
        callback()

    def join(self, timeout: Optional[float] = None) -> None:
        self._done.wait(timeout)

    def is_alive(self) -> bool:
        return not self._done.is_set()

    def run(self, loop: asyncio.AbstractEventLoop, send: bool = True) -> None:
        """Deliver the event on ``loop`` inside the caller's captured context; unless ``send``, buffer it unsent."""
        try:
            if send:
                self._ctx.run(loop.run_until_complete, self._coro)
            else:
                self._ctx.run(_run_deferred, loop, self._coro, time.time(), False)
        except Exception as exc:  # noqa: BLE001 - background task boundary: recorded and logged below
            self._record_failure(exc)
        finally:
            self._settle()

    def materialize(self, enqueued_at: float, timeout: Optional[float] = None) -> None:
        """Run the coroutine so it buffers its payload, aged from ``enqueued_at``, instead of sending it.

        With ``timeout``, cancel it if it is still awaiting after that many
        seconds and raise ``BuildDeadlineExceeded``, once the task is settled.
        """
        loop = asyncio.new_event_loop()
        started = time.monotonic()
        cut_short = False
        try:
            self._ctx.run(_run_deferred, loop, asyncio.wait_for(self._coro, timeout), enqueued_at)
        except asyncio.TimeoutError as exc:
            cut_short = timeout is not None and time.monotonic() - started >= timeout
            if not cut_short:
                self._record_failure(exc)
        except Exception as exc:  # noqa: BLE001 - background task boundary: recorded and logged below
            self._record_failure(exc)
        finally:
            loop.close()
            self._settle()
        if cut_short:
            raise BuildDeadlineExceeded(f"{self.name} was still building after {timeout:.3f}s")

    def discard(self) -> None:
        self._coro.close()
        self._settle()

    def _settle(self) -> None:
        with self._settle_lock:
            if self._done.is_set():
                return
            self._done.set()
            callbacks, self._settled_callbacks = self._settled_callbacks, []
        for callback in callbacks:
            callback()

    def _record_failure(self, exc: Exception) -> None:
        if self._is_shutting_down():
            logger.debug("Exception ignored in metering thread %s during shutdown: %s", self.name, exc)
            return
        self.error = exc
        record_metering_error(exc)
        logger.error("Error in metering thread %s: %s", self.name, exc, exc_info=True)


def _run_deferred(
    loop: asyncio.AbstractEventLoop, coro: Coroutine[Any, Any, Any], enqueued_at: float, counted: bool = True
) -> None:
    with delivery_deferred_to_buffer(enqueued_at, counted):
        loop.run_until_complete(coro)


class _QueueSlots:
    """Room left in the delivery queue, taken and given back without a lock.

    ``deque.pop`` and ``deque.append`` are atomic, so a garbage-collector
    finalizer that meters a stream while its thread is inside ``submit``
    takes a slot without waiting on a lock its own thread holds.
    """

    def __init__(self, size: int):
        self.size = size
        self._free: Deque[None] = collections.deque([None] * size)

    def take(self) -> bool:
        try:
            self._free.pop()
        except IndexError:
            return False
        return True

    def give_back(self) -> None:
        self._free.append(None)

    def all_free(self) -> bool:
        return len(self._free) == self.size


class MeteringWorkerPool:
    """A bounded queue drained by a fixed number of daemon worker threads.

    ``submit`` may be re-entered on the same thread: metering runs from
    finalizers, which the garbage collector calls wherever it happens to run.
    """

    def __init__(
        self,
        workers: int,
        queue_size: int,
        overflow: Callable[[MeteringTask], None],
        is_shutting_down: Callable[[], bool] = _never,
    ):
        self._worker_count = workers
        self._is_shutting_down = is_shutting_down
        self._queue: "queue.SimpleQueue[MeteringTask]" = queue.SimpleQueue()
        self._slots = _QueueSlots(queue_size)
        self._overflow = overflow
        self._workers: List[Tuple[threading.Thread, threading.Event]] = []
        self._workers_lock = threading.RLock()
        self._ensuring_workers = False
        self._worker_ids = itertools.count(1)
        self._idle = threading.Condition()
        self._unfinished = 0
        self._overflow_unbuilt = 0
        self._overflowing = False
        self._overflowed_this_episode = 0

    @property
    def worker_count(self) -> int:
        return self._worker_count

    @property
    def queue_size(self) -> int:
        return self._slots.size

    def new_task(self, coro: Coroutine[Any, Any, Any], ctx: contextvars.Context) -> MeteringTask:
        """A task for ``coro`` that, like this pool, knows when the process is shutting down."""
        return MeteringTask(coro, ctx, self._is_shutting_down)

    def submit(self, task: MeteringTask) -> None:
        """Queue ``task`` without blocking; a full queue sends it to the overflow."""
        self._ensure_workers()
        if self._slots.take():
            with self._idle:
                self._unfinished += 1
            self._queue.put(task)
            return
        with self._idle:
            first_overflow = not self._overflowing
            self._overflowing = True
            self._overflowed_this_episode += 1
            self._overflow_unbuilt += 1
        task.on_settled(self._mark_overflow_built)
        if first_overflow:
            logger.warning(
                "Metering queue full (%d events); routing new events to the "
                "store-and-forward buffer until it drains",
                self.queue_size,
            )
        self._overflow(task)

    def pending(self) -> int:
        """Events queued or being delivered right now."""
        with self._idle:
            return self._unfinished

    def overflow_unbuilt(self) -> int:
        """Overflowed events whose task the buffer has not built yet."""
        with self._idle:
            return self._overflow_unbuilt

    def wait_until_idle(self, timeout: Optional[float]) -> bool:
        """Block until every accepted event is delivered or built into a buffered record; False on timeout."""
        with self._idle:
            return self._idle.wait_for(lambda: self._unfinished == 0 and self._overflow_unbuilt == 0, timeout)

    def wait_until_queue_drained(self, timeout: Optional[float]) -> bool:
        """Block until every queued event is delivered, ignoring overflow; False on timeout."""
        with self._idle:
            return self._idle.wait_for(lambda: self._unfinished == 0, timeout)

    def live_workers(self) -> int:
        with self._workers_lock:
            return sum(1 for worker, _ in self._workers if worker.is_alive())

    def stop(self, timeout: Optional[float] = None) -> None:
        """Discard queued events, let each worker finish its current one and exit.

        Waits at most ``timeout`` in all. A worker still busy after that stays
        counted against the pool size until it exits; a later submit then
        starts its replacement.
        """
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._workers_lock:
            workers = list(self._workers)
        for _, stop_requested in workers:
            stop_requested.set()
        self._discard_queued()
        for worker, _ in workers:
            worker.join(None if deadline is None else max(0.0, deadline - time.monotonic()))
        with self._workers_lock:
            self._workers = [(worker, stop) for worker, stop in self._workers if worker.is_alive()]

    def _discard_queued(self) -> None:
        while True:
            try:
                task = self._queue.get_nowait()
            except queue.Empty:
                return
            self._slots.give_back()
            task.discard()
            self._mark_finished()

    def _mark_finished(self) -> None:
        with self._idle:
            self._unfinished -= 1
            self._notify_if_idle()

    def _end_overflow_episode(self) -> None:
        with self._idle:
            if not self._overflowing:
                return
            self._overflowing = False
            overflowed, self._overflowed_this_episode = self._overflowed_this_episode, 0
        logger.warning(
            "Metering queue drained; %d event(s) went to the store-and-forward buffer while it was full",
            overflowed,
        )

    def _mark_overflow_built(self) -> None:
        with self._idle:
            self._overflow_unbuilt -= 1
            self._notify_if_idle()

    def _notify_if_idle(self) -> None:
        if self._unfinished == 0 or self._overflow_unbuilt == 0:
            self._idle.notify_all()

    def _ensure_workers(self) -> None:
        with self._workers_lock:
            if self._ensuring_workers:
                return
            self._ensuring_workers = True
            try:
                self._start_missing_workers()
            finally:
                self._ensuring_workers = False

    def _start_missing_workers(self) -> None:
        self._workers = [(worker, stop) for worker, stop in self._workers if worker.is_alive()]
        while len(self._workers) < self._worker_count:
            stop_requested = threading.Event()
            worker = threading.Thread(
                target=self._work,
                args=(stop_requested,),
                name=f"ReveniumMeteringWorker-{next(self._worker_ids)}",
                daemon=True,
            )
            worker.start()
            self._workers.append((worker, stop_requested))

    def _work(self, stop_requested: threading.Event) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            while not stop_requested.is_set():
                try:
                    task = self._queue.get(timeout=STOP_CHECK_SECONDS)
                except queue.Empty:
                    continue
                self._slots.give_back()
                if self._overflowing and self._slots.all_free():
                    self._end_overflow_episode()
                try:
                    task.run(loop, send=get_circuit().allows_send())
                finally:
                    self._mark_finished()
        finally:
            loop.close()


_pool: Optional[MeteringWorkerPool] = None
# Reentrant for the same reason as the pool's own locks.
_pool_lock = threading.RLock()


def _is_positive(value: int) -> bool:
    return value >= 1


def _overflow_to_buffer(task: MeteringTask) -> None:
    get_buffer().push_overflow(task)


def get_pool(is_shutting_down: Callable[[], bool]) -> MeteringWorkerPool:
    """Return the process-wide pool, sized from the environment on first use.

    ``is_shutting_down`` is only used when this call creates the pool. A
    finalizer that meters while this thread is building the pool re-enters
    here and publishes a pool of its own first; the outer call then keeps that
    one and drops its own, which has no workers or events yet.
    """
    global _pool
    if _pool is None:
        with _pool_lock:
            if _pool is None:
                built = MeteringWorkerPool(
                    workers=read_env_number(WORKERS_ENV, DEFAULT_WORKERS, int, _is_positive),
                    queue_size=read_env_number(QUEUE_SIZE_ENV, DEFAULT_QUEUE_SIZE, int, _is_positive),
                    overflow=_overflow_to_buffer,
                    is_shutting_down=is_shutting_down,
                )
                if _pool is None:
                    _pool = built
    return _pool


def wait_until_idle(timeout: Optional[float]) -> bool:
    """Wait for queued metering events to be delivered; True when none are left."""
    if _pool is None:
        return True
    return _pool.wait_until_idle(timeout)


def drain(deadline_seconds: float) -> int:
    """Give queued events up to ``deadline_seconds`` to deliver; return how many are left.

    Overflowed events are not waited for: the buffer builds them, not the workers.
    """
    if _pool is None or _pool.wait_until_queue_drained(deadline_seconds):
        return 0
    return _pool.pending()
