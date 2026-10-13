import asyncio
import os
import signal
import threading
import time

import pytest

from revenium_middleware import idempotency_key
from revenium_middleware._core import context as idempotency_context
from revenium_middleware._core.context import (
    _free_scope_lock_in_child,
    _IdempotencyScope,
    get_idempotency_key,
    next_idempotency_key,
    set_idempotency_key,
)


@pytest.fixture(autouse=True)
def _reset_idempotency_key_context():
    """Ensure each test starts with a clean contextvar regardless of execution order."""
    from revenium_middleware._core.context import _idempotency_key_context
    token = _idempotency_key_context.set(None)
    try:
        yield
    finally:
        _idempotency_key_context.reset(token)


def test_get_returns_none_by_default():
    assert get_idempotency_key() is None


def test_context_manager_sets_and_resets():
    assert get_idempotency_key() is None
    with idempotency_key("scoped-value"):
        assert get_idempotency_key() == "scoped-value"
    assert get_idempotency_key() is None


def test_nested_context_managers():
    with idempotency_key("outer"):
        assert get_idempotency_key() == "outer"
        with idempotency_key("inner"):
            assert get_idempotency_key() == "inner"
        assert get_idempotency_key() == "outer"
    assert get_idempotency_key() is None


def test_empty_string_key_raises_value_error():
    """Empty string is a likely caller bug; raise ValueError instead of silently breaking metering."""
    with pytest.raises(ValueError, match="must be a non-empty string"):
        with idempotency_key(""):
            pass  # pragma: no cover - the with-statement raises before this runs


def test_isolated_across_threads():
    """New threads start with an empty context; they do not inherit the spawning thread's contextvar values."""
    main_seen = []
    worker_seen = []

    def worker():
        time.sleep(0.05)
        worker_seen.append(get_idempotency_key())

    with idempotency_key("main-thread-value"):
        thread = threading.Thread(target=worker)
        thread.start()
        thread.join()
        main_seen.append(get_idempotency_key())

    assert main_seen == ["main-thread-value"]
    assert worker_seen == [None]


def test_isolated_across_asyncio_tasks():
    """asyncio tasks copy context on creation; sibling tasks don't see each other's overrides."""

    async def task_with_cm():
        with idempotency_key("task-a-value"):
            await asyncio.sleep(0.01)
            return get_idempotency_key()

    async def task_without_cm():
        await asyncio.sleep(0.01)
        return get_idempotency_key()

    async def runner():
        return await asyncio.gather(task_with_cm(), task_without_cm())

    a, b = asyncio.run(runner())
    assert a == "task-a-value"
    assert b is None


def test_set_idempotency_key_returns_token_for_manual_reset():
    """set_idempotency_key returns a Token compatible with ContextVar.reset()."""
    token = set_idempotency_key("manual")
    assert get_idempotency_key() == "manual"
    from revenium_middleware._core.context import _idempotency_key_context
    _idempotency_key_context.reset(token)
    assert get_idempotency_key() is None


def test_set_idempotency_key_empty_string_raises():
    """Empty string via set_idempotency_key must raise — closes the contextvar bypass."""
    with pytest.raises(ValueError, match="must be a non-empty string"):
        set_idempotency_key("")


UNSENDABLE_KEYS = [
    "commande-caf\u00e9-1002",
    "order-\u6771\u4eac-1003",
    "order-1006\n",
    " order-1007",
    "order-1008 ",
    "order\t1",
    "order\x7f",
    "x" * 256,
    123,
]


@pytest.mark.parametrize("key", UNSENDABLE_KEYS)
def test_a_key_that_is_not_printable_ascii_is_refused_when_the_block_opens(key):
    block_ran = []
    with pytest.raises(ValueError, match="1 to 255 printable ASCII"):
        with idempotency_key(key):
            block_ran.append(True)  # pragma: no cover - the with-statement raises before this runs
    assert block_ran == []
    assert get_idempotency_key() is None


@pytest.mark.parametrize("key", ["commande-caf\u00e9-1002", "order-\u6771\u4eac-1003", "order-1006\n", 123])
def test_set_idempotency_key_refuses_a_key_that_is_not_printable_ascii(key):
    with pytest.raises(ValueError, match="1 to 255 printable ASCII"):
        set_idempotency_key(key)
    assert get_idempotency_key() is None


def test_a_255_character_ascii_key_is_accepted():
    key = "k" * 255
    with idempotency_key(key):
        assert get_idempotency_key() == key


def test_set_idempotency_key_none_still_clears():
    set_idempotency_key("order-1001")
    set_idempotency_key(None)
    assert get_idempotency_key() is None


def test_get_idempotency_key_returns_the_callers_key_after_events_are_sent():
    from unittest.mock import MagicMock, patch

    from revenium_middleware._core.metering_submission import submit_ai_event

    with patch("revenium_middleware._core.metering_submission.current_client", lambda: MagicMock()):
        with idempotency_key("order-42"):
            submit_ai_event("completion", {"x": 1})
            submit_ai_event("completion", {"x": 1})
            assert get_idempotency_key() == "order-42"


def test_after_fork_hook_frees_the_scope_lock_held_by_a_thread_that_did_not_survive():
    held = idempotency_context._scope_lock
    held.acquire()
    try:
        _free_scope_lock_in_child()

        drawn = []

        def draw():
            with idempotency_key("order-42"):
                drawn.append(next_idempotency_key())

        thread = threading.Thread(target=draw, daemon=True)
        thread.start()
        thread.join(timeout=2)

        assert drawn == ["order-42"]
    finally:
        held.release()


@pytest.mark.skipif(not hasattr(os, "fork"), reason="requires os.fork")
def test_forked_child_can_draw_keys_while_the_scope_lock_was_held_at_fork():
    scope = _IdempotencyScope("order-42")
    held = idempotency_context._scope_lock
    held.acquire()

    pid = os.fork()
    if pid == 0:
        try:
            scope.next_key()
            scope.next_key()
            os._exit(0)
        except BaseException:
            os._exit(1)

    try:
        status = None
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            finished_pid, status = os.waitpid(pid, os.WNOHANG)
            if finished_pid:
                break
            time.sleep(0.05)
        else:
            os.kill(pid, signal.SIGKILL)
            _, status = os.waitpid(pid, 0)
    finally:
        held.release()

    assert os.WIFEXITED(status) and os.WEXITSTATUS(status) == 0
