"""Bounded metering delivery: fixed worker pool, bounded queue, overflow to the buffer (BACK-3904)."""
import asyncio
import contextvars
import datetime
import gc
import inspect
import logging
import subprocess
import sys
import textwrap
import threading
import time
import weakref

import pytest

from revenium_middleware._core import metering, metering_buffer, metering_pool
from revenium_middleware._core.config import Config
from revenium_middleware._core.metering_buffer import MeteringBuffer, delivery_deferred_to_buffer
from revenium_middleware._core.metering_pool import MeteringTask, MeteringWorkerPool
from revenium_middleware._core.metering_submission import submit_ai_event
from revenium_middleware._metering._exceptions import APIConnectionError

COMPLETION_ARGS = {"model": "gpt-4o-mini", "input_token_count": 3, "output_token_count": 2}


def task_of(coro):
    return MeteringTask(coro, contextvars.copy_context())


@pytest.fixture
def make_pool():
    pools = []

    def make(workers=2, queue_size=10, overflow=None):
        pool = MeteringWorkerPool(workers, queue_size, overflow or (lambda task: None))
        pools.append(pool)
        return pool

    yield make
    for pool in pools:
        pool.stop(timeout=5)


@pytest.fixture
def gate():
    """An event every blocking task waits on; released at teardown so no worker stays stuck."""
    event = threading.Event()
    yield event
    event.set()


@pytest.fixture
def buffer(monkeypatch):
    buf = MeteringBuffer(flush_interval=3600, replay_fn=lambda event, timeout: None)
    monkeypatch.setattr(metering_buffer, "_buffer", buf)
    return buf


@pytest.fixture
def no_shutdown():
    was_set = metering.shutdown_event.is_set()
    metering.shutdown_event.clear()
    yield
    if was_set:
        metering.shutdown_event.set()
    else:
        metering.shutdown_event.clear()


async def wait_on(gate):
    gate.wait(10)


def fill_queue(pool, gate):
    """Block every worker on ``gate`` and fill the queue behind them."""
    for _ in range(pool.worker_count):
        pool.submit(task_of(wait_on(gate)))
    deadline = time.monotonic() + 5
    while pool._queue.qsize() and time.monotonic() < deadline:
        time.sleep(0.01)
    for _ in range(pool.queue_size):
        pool.submit(task_of(wait_on(gate)))


def wait_until(condition, timeout=5.0):
    deadline = time.monotonic() + timeout
    while not condition() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert condition()


def refuse_new_threads(thread):
    raise RuntimeError("can't create new thread at interpreter shutdown")


def connection_error():
    import httpx

    return APIConnectionError(request=httpx.Request("POST", "http://metering.test/v2/ai/completions"))


class TestWorkerPool:
    def test_thread_count_stays_at_pool_size_whatever_the_backlog(self, make_pool, gate):
        before = threading.active_count()
        pool = make_pool(workers=2, queue_size=100)

        for _ in range(50):
            pool.submit(task_of(wait_on(gate)))

        assert threading.active_count() - before == 2
        assert pool.pending() == 50

    def test_events_reuse_the_same_workers(self, make_pool):
        pool = make_pool(workers=3, queue_size=100)
        seen = set()

        async def record_thread():
            seen.add(threading.current_thread().name)

        for _ in range(40):
            pool.submit(task_of(record_thread()))

        assert pool.wait_until_idle(5)
        assert 1 <= len(seen) <= 3

    def test_full_queue_goes_to_overflow_without_blocking_or_new_threads(self, make_pool, gate, caplog):
        overflowed = []
        pool = make_pool(workers=1, queue_size=2, overflow=overflowed.append)
        pool.submit(task_of(wait_on(gate)))
        deadline = time.monotonic() + 5
        while pool._queue.qsize() and time.monotonic() < deadline:
            time.sleep(0.01)
        pool.submit(task_of(wait_on(gate)))
        pool.submit(task_of(wait_on(gate)))
        threads_before = threading.active_count()

        started = time.monotonic()
        with caplog.at_level(logging.WARNING, logger="revenium_middleware"):
            extra = [task_of(wait_on(gate)) for _ in range(5)]
            for task in extra:
                pool.submit(task)

        assert time.monotonic() - started < 0.5
        assert overflowed == extra
        assert threading.active_count() == threads_before
        assert sum("Metering queue full" in r.getMessage() for r in caplog.records) == 1
        for task in extra:
            task.discard()

    def test_the_queue_full_warning_logs_once_per_overflow_episode(self, make_pool, gate, caplog):
        overflowed = []
        pool = make_pool(workers=1, queue_size=2, overflow=overflowed.append)
        gates = [threading.Event() for _ in range(4)]
        pool.submit(task_of(wait_on(gates[0])))
        wait_until(lambda: pool._queue.qsize() == 0)
        pool.submit(task_of(wait_on(gates[1])))
        pool.submit(task_of(wait_on(gates[2])))

        def full_warnings():
            return sum("Metering queue full" in record.getMessage() for record in caplog.records)

        with caplog.at_level(logging.WARNING, logger="revenium_middleware"):
            pool.submit(task_of(asyncio.sleep(0)))
            gates[0].set()
            wait_until(lambda: pool._queue.qsize() == 1)
            pool.submit(task_of(wait_on(gates[3])))
            pool.submit(task_of(asyncio.sleep(0)))
            assert full_warnings() == 1

            for released in gates:
                released.set()
            assert pool.wait_until_queue_drained(5)
            drained = [record.getMessage() for record in caplog.records
                       if "Metering queue drained" in record.getMessage()]
            assert len(drained) == 1 and "2 event(s)" in drained[0]

            fill_queue(pool, gate)
            assert pool._queue.qsize() == pool.queue_size
            pool.submit(task_of(asyncio.sleep(0)))
            assert full_warnings() == 2
            assert len(overflowed) == 3
        for task in overflowed:
            task.discard()

    def test_task_handle_joins_on_delivery(self, make_pool):
        pool = make_pool()
        task = task_of(asyncio.sleep(0.05))
        assert task.is_alive()

        pool.submit(task)
        task.join(timeout=5)

        assert not task.is_alive()

    def test_failing_event_is_recorded_on_the_handle_and_the_worker_survives(self, make_pool, no_shutdown):
        pool = make_pool(workers=1)

        async def fail():
            raise ValueError("boom")

        failing = task_of(fail())
        pool.submit(failing)
        failing.join(5)
        after = task_of(asyncio.sleep(0))
        pool.submit(after)
        after.join(5)

        assert isinstance(failing.error, ValueError)
        assert not after.is_alive()

    def test_a_failure_while_shutting_down_is_not_recorded(self, make_pool, caplog):
        pool = MeteringWorkerPool(1, 10, lambda task: None, is_shutting_down=lambda: True)

        async def fail():
            raise ValueError("boom")

        task = pool.new_task(fail(), contextvars.copy_context())
        try:
            with caplog.at_level(logging.ERROR, logger="revenium_middleware"):
                pool.submit(task)
                task.join(5)
        finally:
            pool.stop(timeout=5)

        assert task.error is None
        assert "Error in metering thread" not in caplog.text

    def test_the_pool_imports_without_the_metering_module(self):
        probe = textwrap.dedent('''
            import importlib.util, sys, types
            from pathlib import Path

            root = Path(importlib.util.find_spec("revenium_middleware").submodule_search_locations[0])
            for name, path in (("revenium_middleware", root), ("revenium_middleware._core", root / "_core")):
                package = types.ModuleType(name)
                package.__path__ = [str(path)]
                sys.modules[name] = package

            import revenium_middleware._core.metering_pool  # noqa: F401
            print("revenium_middleware._core.metering" in sys.modules)
        ''')
        completed = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, timeout=120)

        assert completed.returncode == 0, completed.stderr[-2000:]
        assert completed.stdout.strip().splitlines()[-1] == "False"

    @pytest.mark.filterwarnings("ignore::pytest.PytestUnhandledThreadExceptionWarning")
    def test_a_worker_killed_by_a_base_exception_is_replaced(self, make_pool):
        pool = make_pool(workers=2)

        async def exit_thread():
            raise SystemExit

        killer = task_of(exit_thread())
        pool.submit(killer)
        killer.join(5)
        deadline = time.monotonic() + 5
        while pool.live_workers() == 2 and time.monotonic() < deadline:
            time.sleep(0.01)
        assert pool.live_workers() == 1

        pool.submit(task_of(asyncio.sleep(0)))

        assert pool.live_workers() == 2

    def test_wait_until_idle_times_out_while_an_event_is_in_flight(self, make_pool, gate):
        pool = make_pool(workers=1)
        pool.submit(task_of(wait_on(gate)))

        assert pool.wait_until_idle(0.05) is False
        gate.set()
        assert pool.wait_until_idle(5) is True

    def test_wait_until_idle_waits_for_an_overflowed_event_to_be_built(self, make_pool, buffer, gate, no_shutdown):
        pool = make_pool(workers=1, queue_size=1, overflow=metering_pool._overflow_to_buffer)
        fill_queue(pool, gate)

        async def metering_call():
            submit_ai_event("completion", dict(COMPLETION_ARGS))

        pool.submit(task_of(metering_call()))
        gate.set()

        assert pool.wait_until_queue_drained(5) is True
        assert pool.overflow_unbuilt() == 1
        assert pool.wait_until_idle(0.05) is False

        buffer.materialize_overflow()

        assert pool.wait_until_idle(5) is True
        assert buffer._events[-1].kind == "ai"

    def test_discarding_an_overflowed_event_also_settles_it(self, make_pool, gate):
        overflowed = []
        pool = make_pool(workers=1, queue_size=1, overflow=overflowed.append)
        fill_queue(pool, gate)
        pool.submit(task_of(asyncio.sleep(0)))
        gate.set()
        assert pool.wait_until_idle(0.05) is False

        overflowed[0].discard()

        assert pool.wait_until_idle(5) is True

    def test_timed_stop_returns_within_its_timeout_when_the_queue_is_full(self, make_pool, gate):
        pool = make_pool(workers=1, queue_size=2)
        fill_queue(pool, gate)

        started = time.monotonic()
        pool.stop(timeout=0.2)

        assert time.monotonic() - started < 1.0
        assert pool.live_workers() == 1, "a worker still busy stays counted until it exits"
        gate.set()

    def test_stop_wakes_every_idle_worker_even_with_more_workers_than_queue_slots(self, make_pool):
        pool = make_pool(workers=8, queue_size=2)
        pool.submit(task_of(asyncio.sleep(0)))
        assert pool.wait_until_idle(5)
        workers = [worker for worker, _ in pool._workers]
        assert len(workers) == 8

        started = time.monotonic()
        pool.stop(timeout=2)

        assert time.monotonic() - started < 2
        assert not any(worker.is_alive() for worker in workers)
        assert pool.live_workers() == 0

    def test_a_submit_after_a_timed_stop_never_exceeds_the_worker_bound(self, make_pool, gate):
        pool = make_pool(workers=2, queue_size=4)
        fill_queue(pool, gate)
        pool.stop(timeout=0.1)
        before = threading.active_count()

        pool.submit(task_of(asyncio.sleep(0)))

        assert threading.active_count() == before
        gate.set()


class TestPoolConfiguration:
    @pytest.fixture(autouse=True)
    def fresh_pool(self, monkeypatch):
        monkeypatch.setattr(metering_pool, "_pool", None)
        yield
        if metering_pool._pool is not None:
            metering_pool._pool.stop(timeout=5)

    def test_defaults(self, monkeypatch):
        monkeypatch.delenv(metering_pool.WORKERS_ENV, raising=False)
        monkeypatch.delenv(metering_pool.QUEUE_SIZE_ENV, raising=False)

        pool = metering_pool.get_pool(lambda: False)

        assert pool.worker_count == 32
        assert pool.queue_size == 1000

    def test_env_overrides(self, monkeypatch):
        monkeypatch.setenv(metering_pool.WORKERS_ENV, "7")
        monkeypatch.setenv(metering_pool.QUEUE_SIZE_ENV, "25")

        pool = metering_pool.get_pool(lambda: False)

        assert pool.worker_count == 7
        assert pool.queue_size == 25

    @pytest.mark.parametrize("raw", ["0", "-3", "four", "2.5"])
    def test_invalid_values_fall_back_to_defaults_with_a_warning(self, monkeypatch, caplog, raw):
        monkeypatch.setenv(metering_pool.WORKERS_ENV, raw)
        monkeypatch.setenv(metering_pool.QUEUE_SIZE_ENV, raw)

        with caplog.at_level(logging.WARNING):
            pool = metering_pool.get_pool(lambda: False)

        assert pool.worker_count == metering_pool.DEFAULT_WORKERS
        assert pool.queue_size == metering_pool.DEFAULT_QUEUE_SIZE
        assert metering_pool.WORKERS_ENV in caplog.text


class TestOverflowThroughTheBuffer:
    def test_overflowed_event_is_built_once_and_delivered_once_with_a_frozen_key(
        self, buffer, mock_revenium_client, no_shutdown
    ):
        delivered = []
        attempts = {"count": 0}

        def replay(event, timeout):
            attempts["count"] += 1
            if attempts["count"] == 1:
                raise connection_error()
            delivered.append(event.payload)

        buffer._replay_fn = replay

        async def metering_call():
            submit_ai_event("completion", dict(COMPLETION_ARGS))

        task = task_of(metering_call())
        buffer.push_overflow(task)

        buffer.flush()
        assert not task.is_alive()
        assert buffer.stats()["size"] == 1
        frozen = buffer._events[0].payload["args"]["extra_headers"]["Idempotency-Key"]

        buffer.flush()
        buffer.flush()

        assert mock_revenium_client.ai.create_completion.call_count == 0
        assert len(delivered) == 1
        assert delivered[0]["operation"] == "completion"
        assert delivered[0]["args"]["extra_headers"]["Idempotency-Key"] == frozen
        assert buffer.stats()["total_overflowed"] == 1
        assert buffer.stats()["total_buffered"] == 1
        assert buffer.stats()["total_replayed"] == 1

    def test_overflowed_event_keeps_the_callers_idempotency_key(self, buffer, no_shutdown):
        from revenium_middleware import idempotency_key

        async def metering_call():
            submit_ai_event("completion", dict(COMPLETION_ARGS))

        with idempotency_key("caller-key"):
            task = task_of(metering_call())
        buffer.push_overflow(task)

        buffer.materialize_overflow()

        assert buffer._events[0].payload["args"]["extra_headers"]["Idempotency-Key"] == "caller-key"

    def test_a_built_event_keeps_the_age_of_its_overflow_and_expires_on_time(self, monkeypatch, no_shutdown):
        clock = {"now": 1_000_000.0}
        replayed = []
        buf = MeteringBuffer(flush_interval=3600, max_age_seconds=60, now_fn=lambda: clock["now"],
                             replay_fn=lambda event, timeout: replayed.append(event))
        monkeypatch.setattr(metering_buffer, "_buffer", buf)

        async def metering_call():
            submit_ai_event("completion", dict(COMPLETION_ARGS))

        buf.push_overflow(task_of(metering_call()))
        clock["now"] += 61

        result = buf.flush()

        assert replayed == []
        assert result["expired"] == 1
        assert buf.stats()["size"] == 0

    def test_unbuilt_overflow_returned_after_a_deadline_respects_capacity(self, gate, monkeypatch):
        buf = MeteringBuffer(max_size=3, flush_interval=3600, replay_fn=lambda event, timeout: None)
        filled = threading.Event()

        class FillsTheBufferWhileItBuilds:
            discarded = False

            def materialize(self, enqueued_at, timeout=None):
                for n in range(3):
                    buf.push("ai", {"operation": "completion", "args": {"seq": n}})
                filled.set()
                gate.wait(10)

            def discard(self):
                self.discarded = True

        class Leftover(FillsTheBufferWhileItBuilds):
            def materialize(self, enqueued_at, timeout=None):
                raise AssertionError("the deadline passed before this task")

        leftover = Leftover()
        buf.push_overflow(FillsTheBufferWhileItBuilds())
        buf.push_overflow(leftover)
        evicted_before = buf.stats()["total_evicted"]
        # The deadline passes once the first task has filled the buffer and is
        # still building, whenever the build thread gets to it.
        monkeypatch.setattr(metering_buffer._OverflowBuild, "wait", lambda build, timeout: filled.wait(10))

        unbuilt = buf.materialize_overflow(deadline_seconds=0.05)

        assert unbuilt == 2, "the task still building when the wait ends is not built yet"
        assert buf.stats()["size"] == 3
        assert buf.stats()["total_evicted"] == evicted_before + 1
        assert leftover.discarded
        assert [event.kind for event in buf._events] == ["ai", "ai", "ai"]

    def test_a_build_that_overruns_its_deadline_returns_on_time_and_keeps_the_rest(self, buffer, no_shutdown):
        async def slow_call():
            time.sleep(0.6)
            submit_ai_event("completion", dict(COMPLETION_ARGS))

        async def next_call():
            submit_ai_event("completion", dict(COMPLETION_ARGS))

        slow, waiting = task_of(slow_call()), task_of(next_call())
        buffer.push_overflow(slow)
        buffer.push_overflow(waiting)

        started = time.monotonic()
        unbuilt = buffer.materialize_overflow(deadline_seconds=0.1)

        assert time.monotonic() - started < 0.4
        assert unbuilt == 2
        assert [event.payload.get("task") for event in buffer._events] == [waiting]
        assert waiting.is_alive(), "a task the build did not reach is not settled"
        slow.join(5)
        assert [event.kind for event in buffer._events] == ["overflow", "ai"]
        waiting.discard()

    def test_a_flush_sends_the_events_of_tasks_that_overflowed_before_it(self, buffer, no_shutdown):
        replayed = []
        buffer._replay_fn = lambda event, timeout: replayed.append(event.payload["args"]["model"])
        buffer.push("ai", {"operation": "completion", "args": {"model": "buffered"}})

        async def metering_call():
            submit_ai_event("completion", {**COMPLETION_ARGS, "model": "overflowed"})

        buffer.push_overflow(task_of(metering_call()))

        result = buffer.flush()

        assert sorted(replayed) == ["buffered", "overflowed"]
        assert result["remaining"] == 0

    def test_flush_passes_its_remaining_budget_to_the_build(self, buffer, no_shutdown):
        async def slow_call():
            time.sleep(0.6)

        slow = task_of(slow_call())
        buffer.push_overflow(slow)

        started = time.monotonic()
        buffer.flush(deadline_seconds=0.1)

        assert time.monotonic() - started < 0.4
        slow.join(5)

    def test_builds_run_on_the_thread_started_with_the_first_overflow(self, buffer, monkeypatch, no_shutdown):
        build_threads = []

        async def metering_call():
            build_threads.append(threading.current_thread())

        first, second = task_of(metering_call()), task_of(metering_call())
        buffer.push_overflow(first)
        monkeypatch.setattr(threading.Thread, "start", refuse_new_threads)

        assert buffer.materialize_overflow(deadline_seconds=1.0) == 0
        buffer.push_overflow(second)
        assert buffer.materialize_overflow(deadline_seconds=1.0) == 0

        assert not first.is_alive() and not second.is_alive()
        assert len(build_threads) == 2 and build_threads[0] is build_threads[1]
        assert build_threads[0].name == "MeteringOverflowBuild"

    def test_with_no_thread_to_build_on_the_tasks_build_on_the_calling_thread(self, buffer, monkeypatch, no_shutdown):
        monkeypatch.setattr(threading.Thread, "start", refuse_new_threads)
        build_threads = []

        async def metering_call():
            build_threads.append(threading.current_thread())

        tasks = [task_of(metering_call()), task_of(metering_call())]
        for task in tasks:
            buffer.push_overflow(task)

        assert buffer.materialize_overflow(deadline_seconds=1.0) == 0

        assert build_threads == [threading.current_thread()] * 2
        assert not any(task.is_alive() for task in tasks)
        assert buffer.stats()["size"] == 0

    def test_a_build_on_the_calling_thread_keeps_to_its_deadline(self, buffer, monkeypatch, no_shutdown):
        monkeypatch.setattr(threading.Thread, "start", refuse_new_threads)

        async def awaits_past_the_deadline():
            await asyncio.sleep(5)

        slow, waiting = task_of(awaits_past_the_deadline()), task_of(asyncio.sleep(0))
        buffer.push_overflow(slow)
        buffer.push_overflow(waiting)

        started = time.monotonic()
        unbuilt = buffer.materialize_overflow(deadline_seconds=0.2)

        assert time.monotonic() - started < 0.6
        assert not slow.is_alive(), "the awaiting task was cut short at the deadline"
        assert unbuilt == 1
        assert [event.payload["task"] for event in buffer._events] == [waiting]
        waiting.discard()

    def test_a_build_the_timeout_cuts_short_is_settled_and_reported_as_cut_short_not_as_a_failure(
        self, buffer, no_shutdown
    ):
        from revenium_middleware._core.metering_buffer import BuildDeadlineExceeded
        from revenium_middleware._core.metering_status import reset_metering_status

        reset_metering_status()
        cut_short = task_of(asyncio.sleep(5))
        raises_its_own_timeout = task_of(asyncio.wait_for(asyncio.sleep(5), 0.01))

        with pytest.raises(BuildDeadlineExceeded):
            cut_short.materialize(time.time(), timeout=0.05)
        raises_its_own_timeout.materialize(time.time(), timeout=2.0)

        assert not cut_short.is_alive() and cut_short.error is None
        assert isinstance(raises_its_own_timeout.error, asyncio.TimeoutError)
        reset_metering_status()

    def test_with_no_thread_and_an_event_loop_running_the_tasks_stay_buffered_unbuilt(
        self, buffer, monkeypatch, caplog
    ):
        monkeypatch.setattr(threading.Thread, "start", refuse_new_threads)
        ran = []

        async def metering_call():
            ran.append(True)

        tasks = [task_of(metering_call()), task_of(metering_call())]
        for task in tasks:
            buffer.push_overflow(task)

        async def async_host():
            return buffer.materialize_overflow(deadline_seconds=1.0)

        with caplog.at_level(logging.WARNING, logger="revenium_middleware"):
            unbuilt = asyncio.run(async_host())
        monkeypatch.undo()

        assert unbuilt == 2
        assert ran == []
        assert [event.payload["task"] for event in buffer._events] == tasks
        assert all(task.is_alive() for task in tasks)
        assert sum("No thread could build 2 overflowed" in r.getMessage() for r in caplog.records) == 1
        for task in tasks:
            task.discard()

    def test_a_build_started_from_the_build_thread_runs_there_instead_of_waiting_on_itself(self, buffer, no_shutdown):
        from revenium_middleware._core.metering_status import on_metering_error, reset_metering_status

        reset_metering_status()
        nested = task_of(asyncio.sleep(0))

        def flush_from_the_error_callback(error):
            buffer.push_overflow(nested)
            buffer.materialize_overflow()

        on_metering_error(flush_from_the_error_callback)

        async def failing_call():
            raise RuntimeError("build failed")

        buffer.push_overflow(task_of(failing_call()))
        try:
            flush = threading.Thread(target=buffer.materialize_overflow, daemon=True)
            flush.start()
            flush.join(5)

            assert not flush.is_alive(), "the build thread waited on a build queued behind itself"
            assert not nested.is_alive()
        finally:
            reset_metering_status()

    def test_evicting_an_overflowed_task_closes_its_coroutine(self, monkeypatch):
        buf = MeteringBuffer(max_size=1, flush_interval=3600, replay_fn=lambda event, timeout: None)

        async def never_run():
            raise AssertionError("an evicted task must not run")

        coro = never_run()
        task = task_of(coro)
        buf.push_overflow(task)
        buf.push("ai", {"operation": "completion", "args": {}})

        assert not task.is_alive()
        assert inspect.getcoroutinestate(coro) == inspect.CORO_CLOSED

    def test_the_pool_hands_overflow_to_the_process_buffer(self, buffer, monkeypatch, gate):
        pool = MeteringWorkerPool(1, 1, metering_pool._overflow_to_buffer)
        try:
            pool.submit(task_of(wait_on(gate)))
            deadline = time.monotonic() + 5
            while pool._queue.qsize() and time.monotonic() < deadline:
                time.sleep(0.01)
            pool.submit(task_of(wait_on(gate)))
            overflowed = task_of(asyncio.sleep(0))

            pool.submit(overflowed)

            assert buffer.stats()["total_overflowed"] == 1
            assert buffer._events[0].payload["task"] is overflowed
            overflowed.discard()
        finally:
            gate.set()
            pool.stop(timeout=5)


class AbandonedStream:
    """A stream wrapper in a reference cycle that meters itself when the garbage collector finalizes it."""

    def __init__(self, pool, metered):
        self._pool = pool
        self._metered = metered
        self._cycle = self

    def __del__(self):
        task = task_of(asyncio.sleep(0))
        self._pool.submit(task)
        self._metered.append(task)


def collect_an_abandoned_stream_once(pool, metered):
    """Run the cyclic garbage collector on this thread the first time it is called, as an allocation can."""
    if not metered:
        AbandonedStream(pool, metered)
        gc.collect()


UNPATCHED_THREAD_START = threading.Thread.start


def run_with_deadlock_check(target, *args):
    runner = threading.Thread(target=target, args=args, daemon=True)
    UNPATCHED_THREAD_START(runner)
    runner.join(5)
    return not runner.is_alive()


class TestReentrantSubmit:
    """Stream wrappers meter from finalizers, which run on whichever thread the garbage collector interrupts."""

    def test_a_finalizer_that_meters_while_its_thread_starts_a_worker_does_not_deadlock(self, make_pool, monkeypatch):
        pool = make_pool(workers=1, queue_size=2)
        metered = []

        def start_after_collecting(thread):
            collect_an_abandoned_stream_once(pool, metered)
            UNPATCHED_THREAD_START(thread)

        monkeypatch.setattr(threading.Thread, "start", start_after_collecting)
        outer = task_of(asyncio.sleep(0))

        assert run_with_deadlock_check(pool.submit, outer), "submit deadlocked on its own worker lock"
        monkeypatch.undo()
        outer.join(5)
        metered[0].join(5)
        assert not outer.is_alive() and not metered[0].is_alive()
        assert pool.live_workers() == 1

    def test_a_finalizer_that_meters_while_its_thread_holds_the_buffer_lock_does_not_deadlock(
        self, make_pool, gate, monkeypatch
    ):
        buf = MeteringBuffer(max_size=1, flush_interval=3600, replay_fn=lambda event, timeout: None)
        monkeypatch.setattr(metering_buffer, "_buffer", buf)
        pool = make_pool(workers=1, queue_size=1, overflow=metering_pool._overflow_to_buffer)
        fill_queue(pool, gate)
        metered = []

        class CollectsGarbageOnBufferFull(logging.Handler):
            def emit(self, record):
                if record.getMessage().startswith("Metering buffer full"):
                    collect_an_abandoned_stream_once(pool, metered)

        handler = CollectsGarbageOnBufferFull()
        logger = logging.getLogger("revenium_middleware")
        logger.addHandler(handler)
        try:
            buf.push("ai", {"seq": 0})

            assert run_with_deadlock_check(buf.push, "ai", {"seq": 1}), "the buffer deadlocked on its own lock"
        finally:
            logger.removeHandler(handler)
        assert len(metered) == 1 and not metered[0].is_alive(), "the finalizer's event was overflowed and evicted"
        assert buf.stats()["size"] == 1


class MetersWhenCollected:
    """An object in a reference cycle that runs ``meter`` when the garbage collector finalizes it."""

    def __init__(self, meter):
        self._meter = meter
        self._cycle = self

    def __del__(self):
        self._meter()


def collect_while_constructing(cls, meter):
    """A subclass of ``cls`` whose first construction runs a finalizer that calls ``meter``, as an allocation can."""
    triggered = []

    class Collecting(cls):
        def __init__(self, *args, **kwargs):
            if not triggered:
                triggered.append(True)
                MetersWhenCollected(meter)
                gc.collect()
            super().__init__(*args, **kwargs)

    return Collecting


class TestReentrantInitialization:
    """A finalizer that meters while the first pool or buffer is being built must land in the one that is kept."""

    def test_a_finalizer_that_meters_while_the_pool_is_built_uses_the_pool_that_is_kept(
        self, buffer, monkeypatch, no_shutdown
    ):
        monkeypatch.setattr(metering_pool, "_pool", None)
        metered = []
        monkeypatch.setattr(metering_pool, "MeteringWorkerPool", collect_while_constructing(
            MeteringWorkerPool, lambda: metered.append(metering.run_async_in_thread(asyncio.sleep(0)))))

        try:
            pool = metering_pool.get_pool(lambda: False)

            assert len(metered) == 1
            metered[0].join(5)
            assert not metered[0].is_alive()
            assert metering_pool._pool is pool
            assert pool.live_workers() == pool.worker_count
        finally:
            if metering_pool._pool is not None:
                metering_pool._pool.stop(timeout=5)

    def test_a_finalizer_that_buffers_while_the_buffer_is_built_keeps_its_event(self, monkeypatch):
        monkeypatch.setattr(metering_buffer, "_buffer", None)
        monkeypatch.setattr(metering_buffer, "MeteringBuffer", collect_while_constructing(
            MeteringBuffer, lambda: metering_buffer.get_buffer().push("ai", {"seq": "from a finalizer"})))

        buf = metering_buffer.get_buffer()

        assert [event.payload["seq"] for event in buf._events] == ["from a finalizer"]
        assert metering_buffer.get_buffer() is buf


class TestBuildThread:
    def test_a_finalizer_that_overflows_while_the_build_thread_starts_does_not_start_a_second_one(
        self, buffer, monkeypatch, no_shutdown
    ):
        build_threads = set()
        started = []

        async def metering_call():
            build_threads.add(threading.current_thread())

        nested = task_of(metering_call())

        class AbandonedStreamThatOverflows:
            def __init__(self):
                self._cycle = self

            def __del__(self):
                buffer.push_overflow(nested)

        def start_after_collecting(thread):
            if thread.name == "MeteringOverflowBuild":
                started.append(thread)
                if len(started) == 1:
                    AbandonedStreamThatOverflows()
                    gc.collect()
            UNPATCHED_THREAD_START(thread)

        monkeypatch.setattr(threading.Thread, "start", start_after_collecting)
        outer = task_of(metering_call())
        buffer.push_overflow(outer)
        monkeypatch.undo()

        assert len(started) == 1, "a second build thread started"
        assert buffer.materialize_overflow(deadline_seconds=5.0) == 0
        assert not outer.is_alive() and not nested.is_alive()
        assert build_threads == {started[0]}

    def test_a_finished_build_is_released_by_the_build_thread(self, buffer, no_shutdown):
        class Task:
            def materialize(self, enqueued_at, timeout=None):
                pass

            def discard(self):
                pass

        task = Task()
        released = weakref.ref(task)
        buffer.push_overflow(task)
        del task

        assert buffer.materialize_overflow(deadline_seconds=5.0) == 0
        deadline = time.monotonic() + 5
        while released() is not None and time.monotonic() < deadline:
            gc.collect()
            time.sleep(0.01)
        assert released() is None, "the build thread still holds the finished build"


class TestDeferredDelivery:
    def test_submit_ai_event_buffers_instead_of_sending(self, buffer, mock_revenium_client):
        with delivery_deferred_to_buffer(enqueued_at=time.time()):
            result = submit_ai_event("completion", dict(COMPLETION_ARGS))

        assert result is None
        mock_revenium_client.ai.create_completion.assert_not_called()
        assert buffer._events[0].kind == "ai"
        assert buffer._events[0].payload["args"]["model"] == "gpt-4o-mini"

    def test_tool_event_buffers_instead_of_posting(self, buffer, monkeypatch):
        import httpx

        from revenium_middleware._metering.context import ReveniumContext
        from revenium_middleware._metering.decorator import _send_tool_event_async

        def no_network(*args, **kwargs):
            raise AssertionError("deferred delivery must not open a connection")

        monkeypatch.setattr(httpx, "AsyncClient", no_network)

        async def send():
            with delivery_deferred_to_buffer(enqueued_at=time.time()):
                await _send_tool_event_async(
                    "http://metering.test/", "hak_key", tool_id="search", operation="scrape",
                    duration_ms=5, success=True, error_message=None, usage_metadata=None,
                    context=ReveniumContext(),
                    occurred_at=datetime.datetime.now(datetime.timezone.utc),
                )

        asyncio.run(send())

        assert buffer._events[0].kind == "tool"
        assert buffer._events[0].payload["event_payload"]["toolId"] == "search"


class TestShutdownDrain:
    @pytest.fixture
    def global_pool(self, monkeypatch):
        pool = MeteringWorkerPool(1, 100, lambda task: None)
        monkeypatch.setattr(metering_pool, "_pool", pool)
        yield pool
        pool.stop(timeout=5)

    def test_queued_events_are_delivered_before_the_shutdown_flag_is_raised(
        self, global_pool, no_shutdown, monkeypatch
    ):
        monkeypatch.setattr(metering_buffer, "_buffer", None)
        delivered = []

        async def integration_style_call(n):
            if metering.shutdown_event.is_set():
                return
            await asyncio.sleep(0.01)
            delivered.append(n)

        for n in range(20):
            metering.run_async_in_thread(integration_style_call(n))

        metering.handle_exit()

        assert delivered == list(range(20))

    def test_drain_is_bounded_and_reports_what_is_left(self, global_pool, gate, no_shutdown, monkeypatch, caplog):
        monkeypatch.setattr(metering_buffer, "_buffer", None)
        monkeypatch.setenv(Config.ENV_REVENIUM_SHUTDOWN_TIMEOUT_SECONDS, "0.2")
        for _ in range(3):
            metering.run_async_in_thread(wait_on(gate))

        started = time.monotonic()
        with caplog.at_level(logging.WARNING, logger="revenium_middleware"):
            metering.handle_exit()

        assert time.monotonic() - started < 0.7
        assert "3 metering event(s) still queued, in flight, unbuilt or unsent after the 0.2s shutdown budget" in caplog.text

    def test_overflowed_events_are_built_before_the_shutdown_flag_and_sent_by_the_final_flush(
        self, gate, no_shutdown, monkeypatch, mock_revenium_client
    ):
        pool = MeteringWorkerPool(1, 1, metering_pool._overflow_to_buffer)
        monkeypatch.setattr(metering_pool, "_pool", pool)
        replayed = []
        buf = MeteringBuffer(flush_interval=3600, replay_fn=lambda event, timeout: replayed.append(event))
        monkeypatch.setattr(metering_buffer, "_buffer", buf)
        try:
            fill_queue(pool, gate)

            async def integration_style_call():
                if metering.shutdown_event.is_set():
                    return
                submit_ai_event("completion", dict(COMPLETION_ARGS))

            metering.run_async_in_thread(integration_style_call())
            assert buf.stats()["total_overflowed"] == 1
            threading.Timer(0.1, gate.set).start()

            metering.handle_exit()

            assert [event.kind for event in replayed] == ["ai"]
            assert replayed[0].payload["args"]["model"] == "gpt-4o-mini"
            mock_revenium_client.ai.create_completion.assert_not_called()
        finally:
            gate.set()
            pool.stop(timeout=5)

    def test_overflowed_events_are_built_when_shutdown_runs_inside_an_event_loop(
        self, no_shutdown, monkeypatch, mock_revenium_client
    ):
        monkeypatch.setattr(metering_pool, "_pool", None)
        replayed = []
        buf = MeteringBuffer(flush_interval=3600, replay_fn=lambda event, timeout: replayed.append(event))
        monkeypatch.setattr(metering_buffer, "_buffer", buf)

        async def integration_style_call():
            if metering.shutdown_event.is_set():
                return
            submit_ai_event("completion", dict(COMPLETION_ARGS))

        overflowed = task_of(integration_style_call())
        buf.push_overflow(overflowed)

        async def async_host_shutting_down():
            metering.handle_exit()

        asyncio.run(async_host_shutting_down())

        assert overflowed.error is None
        assert [event.kind for event in replayed] == ["ai"]
        assert replayed[0].payload["args"]["model"] == "gpt-4o-mini"

    def test_a_slow_overflow_build_cannot_hold_up_shutdown(self, no_shutdown, monkeypatch):
        monkeypatch.setattr(metering_pool, "_pool", None)
        monkeypatch.setenv(Config.ENV_REVENIUM_SHUTDOWN_TIMEOUT_SECONDS, "0.2")
        buf = MeteringBuffer(flush_interval=3600, replay_fn=lambda event, timeout: None)
        monkeypatch.setattr(metering_buffer, "_buffer", buf)

        async def slow_call():
            time.sleep(2)

        slow, waiting = task_of(slow_call()), task_of(asyncio.sleep(0))
        buf.push_overflow(slow)
        buf.push_overflow(waiting)

        started = time.monotonic()
        metering.handle_exit()

        # The build and the final flush share one 0.2s budget.
        assert time.monotonic() - started < 0.6
        assert slow.is_alive()
        assert buf.stats()["total_evicted"] == 0
        waiting.discard()

    def test_shutdown_takes_over_tasks_a_running_flush_has_not_built_yet(
        self, no_shutdown, monkeypatch, mock_revenium_client
    ):
        monkeypatch.setattr(metering_pool, "_pool", None)
        monkeypatch.setenv(Config.ENV_REVENIUM_SHUTDOWN_TIMEOUT_SECONDS, "2")
        replayed = []
        buf = MeteringBuffer(flush_interval=3600, replay_fn=lambda event, timeout: replayed.append(event))
        monkeypatch.setattr(metering_buffer, "_buffer", buf)
        first_started = threading.Event()

        async def integration_style_call(model, block=False):
            if metering.shutdown_event.is_set():
                return
            if block:
                first_started.set()
                time.sleep(0.3)
            submit_ai_event("completion", {**COMPLETION_ARGS, "model": model})

        buf.push_overflow(task_of(integration_style_call("slow", block=True)))
        waiting = task_of(integration_style_call("waiting"))
        buf.push_overflow(waiting)
        periodic_flush = threading.Thread(target=buf.flush, daemon=True)
        periodic_flush.start()
        assert first_started.wait(5)

        metering.handle_exit()
        periodic_flush.join(5)

        assert not waiting.is_alive()
        assert sorted(event.payload["args"]["model"] for event in replayed) == ["slow", "waiting"]

    @pytest.mark.parametrize("new_threads", ["allowed", "refused"])
    def test_shutdown_builds_reclaimed_tasks_while_the_build_thread_is_stuck_on_another(
        self, buffer, gate, no_shutdown, monkeypatch, new_threads
    ):
        stuck_started = threading.Event()
        built_on = []

        async def stuck_call():
            stuck_started.set()
            gate.wait(10)

        async def ready_call():
            built_on.append(threading.current_thread())
            submit_ai_event("completion", {**COMPLETION_ARGS, "model": "ready"})

        stuck, ready = task_of(stuck_call()), task_of(ready_call())
        buffer.push_overflow(stuck)
        buffer.push_overflow(ready)
        periodic_build = threading.Thread(target=buffer.materialize_overflow, daemon=True)
        periodic_build.start()
        assert stuck_started.wait(5)
        if new_threads == "refused":
            monkeypatch.setattr(threading.Thread, "start", refuse_new_threads)

        unbuilt = buffer.build_all_overflow(deadline_seconds=0.5)

        assert not ready.is_alive(), "the reclaimed task waited behind the stuck build"
        assert (built_on == [threading.current_thread()]) is (new_threads == "refused")
        assert [event.payload["args"]["model"] for event in buffer._events] == ["ready"]
        assert unbuilt == 1
        gate.set()
        periodic_build.join(5)

    def test_drain_without_a_pool_is_a_no_op(self, monkeypatch):
        monkeypatch.setattr(metering_pool, "_pool", None)

        assert metering_pool.drain(0.01) == 0
        assert metering_pool.wait_until_idle(0.01) is True
