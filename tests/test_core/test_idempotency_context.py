import asyncio
import threading
import time

import pytest

from revenium_middleware import idempotency_key
from revenium_middleware._core.context import (
    get_idempotency_key,
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
