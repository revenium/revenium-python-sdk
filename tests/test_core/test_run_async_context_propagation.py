"""Verify run_async_in_thread propagates contextvars to the worker thread."""
import asyncio
import time

import pytest

from revenium_middleware import idempotency_key
from revenium_middleware._core.context import get_idempotency_key
from revenium_middleware._core.metering import (
    active_threads,
    run_async_in_thread,
    shutdown_event,
)


@pytest.fixture
def reset_state():
    """Reset module-level state between tests."""
    was_set = shutdown_event.is_set()
    active_threads.clear()
    shutdown_event.clear()
    yield
    if was_set:
        shutdown_event.set()


def test_idempotency_context_propagates_into_metering_thread(reset_state):
    """The CM's value must be visible inside the dispatched coroutine."""
    seen = []

    async def capture():
        seen.append(get_idempotency_key())

    with idempotency_key("user-set-key"):
        thread = run_async_in_thread(capture())
        thread.join(timeout=2.0)

    assert seen == ["user-set-key"]


def test_no_context_means_none_in_thread(reset_state):
    """Without the CM active, the worker thread sees None as expected."""
    seen = []

    async def capture():
        seen.append(get_idempotency_key())

    thread = run_async_in_thread(capture())
    thread.join(timeout=2.0)

    assert seen == [None]


def test_nested_cms_propagate_innermost_value(reset_state):
    """Nested CMs forward only the innermost active value."""
    seen = []

    async def capture():
        seen.append(get_idempotency_key())

    with idempotency_key("outer"):
        with idempotency_key("inner"):
            thread = run_async_in_thread(capture())
            thread.join(timeout=2.0)

    assert seen == ["inner"]
