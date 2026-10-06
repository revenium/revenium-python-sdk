import os
import time
import logging
import asyncio
import threading
import contextvars
import atexit
import math
from typing import Literal, Awaitable, Any, Dict, Optional, Callable

import httpx

from revenium_middleware._metering import ReveniumMetering
from revenium_middleware._core.config import Config, read_env_number, validate_api_key
from revenium_middleware._core import metering_buffer, metering_pool
from revenium_middleware._core.metering_pool import MeteringTask
from revenium_middleware._core.shutdown_signals import install_requested_signal_handlers

# Get the logger that was configured in __init__.py
logger = logging.getLogger("revenium_middleware")

# Define a StopReason literal type for strict typing of stop_reason
StopReason = Literal["END", "END_SEQUENCE", "TIMEOUT", "TOKEN_LIMIT", "COST_LIMIT", "COMPLETION_LIMIT", "ERROR"]

TIMEOUT_ENV = "REVENIUM_METERING_TIMEOUT_SECONDS"
CONNECT_TIMEOUT_ENV = "REVENIUM_METERING_CONNECT_TIMEOUT_SECONDS"
MAX_RETRIES_ENV = "REVENIUM_METERING_MAX_RETRIES"
# Metering runs off the request path, so a dead endpoint should release a
# worker in seconds rather than the generated client's 60s default.
DEFAULT_TIMEOUT_SECONDS = 10.0
DEFAULT_CONNECT_TIMEOUT_SECONDS = 5.0
DEFAULT_MAX_RETRIES = 2


def _is_positive_finite(value: float) -> bool:
    return value > 0 and math.isfinite(value)


def metering_client_timeout() -> httpx.Timeout:
    """Read/write/pool timeout and connect timeout for the metering client, from the environment."""
    read = read_env_number(TIMEOUT_ENV, DEFAULT_TIMEOUT_SECONDS, float, _is_positive_finite)
    connect = read_env_number(CONNECT_TIMEOUT_ENV, DEFAULT_CONNECT_TIMEOUT_SECONDS, float, _is_positive_finite)
    return httpx.Timeout(read, connect=connect)


def metering_client_max_retries() -> int:
    return read_env_number(MAX_RETRIES_ENV, DEFAULT_MAX_RETRIES, int, lambda retries: retries >= 0)


def _new_client(api_key: str, base_url: Optional[str]) -> ReveniumMetering:
    options: Dict[str, Any] = {"timeout": metering_client_timeout(), "max_retries": metering_client_max_retries()}
    if base_url is not None:
        options["base_url"] = base_url
    return ReveniumMetering(api_key=api_key, **options)


def _build_metering_client(
    api_key: Optional[str],
    base_url_raw: Optional[str],
) -> Optional[ReveniumMetering]:
    """Construct a ReveniumMetering client from raw env-var inputs.

    Returns ``None`` when the API key is missing/empty (logs an error --
    metering silently disabled is the failure mode this SDK must surface).
    Raises ``ValueError`` when the API key is present but malformed.
    Falls back to the library default when ``base_url_raw`` is set but not
    a valid http(s) URL.
    """
    validated_base_url: Optional[str] = None
    if base_url_raw is not None:
        stripped = base_url_raw.strip()
        if stripped.startswith("http://") or stripped.startswith("https://"):
            validated_base_url = stripped
        else:
            logger.warning(
                "REVENIUM_METERING_BASE_URL=%r is invalid (must start with http:// or https://); "
                "falling back to default endpoint",
                base_url_raw,
            )

    if not api_key:
        logger.error(
            "REVENIUM_METERING_API_KEY environment variable is not set. "
            "Metering is disabled and no usage data will be sent to Revenium."
        )
        return None

    validate_api_key(api_key)
    if validated_base_url is not None:
        return _new_client(api_key, validated_base_url)
    if base_url_raw is not None:
        return _new_client(api_key, "https://api.revenium.ai/meter/")
    return _new_client(api_key, None)


api_key = os.environ.get("REVENIUM_METERING_API_KEY")
_base_url_raw = os.environ.get("REVENIUM_METERING_BASE_URL")
client = _build_metering_client(api_key, _base_url_raw)

# Remembers a malformed env key so the lazy path doesn't re-raise per call.
_last_failed_key: Optional[str] = None


def _sync_client_reexports(new_client: Optional[ReveniumMetering]) -> None:
    """Point the well-known re-export attributes at ``new_client``.

    Takes the client explicitly (rather than reading the module-level global)
    so concurrent rebuilds cannot interleave and leave the re-exports pointing
    at a different instance than the caller just assigned. Dynamic readers
    (``revenium_middleware.client``, ``revenium_middleware._core.client``)
    observe the new client. Modules that bound the name via
    ``from revenium_middleware import client`` at import time keep their
    snapshot -- code paths that matter resolve through ``get_client()``
    instead.
    """
    import sys
    for module_name in ("revenium_middleware", "revenium_middleware._core"):
        module = sys.modules.get(module_name)
        if module is not None and hasattr(module, "client"):
            module.client = new_client


def initialize_metering(api_key: Optional[str] = None, base_url: Optional[str] = None) -> bool:
    """(Re)build the metering client from explicit values or the environment.

    Call this when credentials only become available after the first
    ``revenium_middleware`` import (vault/parameter-store bootstraps, framework
    settings hooks) or to reconfigure at runtime. Explicit arguments take
    precedence over ``REVENIUM_METERING_API_KEY`` / ``REVENIUM_METERING_BASE_URL``.

    Returns True when metering is enabled after the call; a missing/empty key
    logs an ERROR and leaves metering disabled. Raises ``ValueError`` for a
    malformed API key -- explicit configuration fails loudly, unlike the lazy
    ``get_client()`` path, which logs a warning once for a malformed env key
    and leaves metering disabled.
    """
    global client
    key = api_key if api_key is not None else os.environ.get("REVENIUM_METERING_API_KEY")
    url = base_url if base_url is not None else os.environ.get("REVENIUM_METERING_BASE_URL")
    new_client = _build_metering_client(key, url)
    client = new_client
    _sync_client_reexports(new_client)
    return new_client is not None


def get_client() -> Optional[ReveniumMetering]:
    """Return the metering client, retrying the environment when unset.

    Covers env vars populated after import: as soon as
    ``REVENIUM_METERING_API_KEY`` appears, the next metering event builds the
    client instead of silently no-opping forever.

    A missing/unset key is logged at ERROR level by the build path; a
    malformed env key is logged once as a warning and metering stays
    disabled, whereas the explicit ``initialize_metering()`` raises
    ``ValueError`` so programmatic misconfiguration fails loudly.
    """
    global _last_failed_key
    if client is None:
        env_key = os.environ.get("REVENIUM_METERING_API_KEY")
        if env_key and env_key != _last_failed_key:
            try:
                initialize_metering()
            except ValueError as e:
                logger.warning("Deferred metering initialization failed: %s", e)
                _last_failed_key = env_key
    return client

shutdown_event = threading.Event()

DEFAULT_SHUTDOWN_TIMEOUT_SECONDS = 5.0


def _is_valid_budget(budget: float) -> bool:
    return math.isfinite(budget) and budget >= 0


def shutdown_budget_seconds() -> float:
    """Total seconds the exit drain may spend, from ``REVENIUM_SHUTDOWN_TIMEOUT_SECONDS``."""
    return read_env_number(
        Config.ENV_REVENIUM_SHUTDOWN_TIMEOUT_SECONDS, DEFAULT_SHUTDOWN_TIMEOUT_SECONDS, float, _is_valid_budget
    )


def handle_exit() -> None:
    """Deliver queued metering, build overflowed events and flush the buffer, within one shared budget."""
    if shutdown_event.is_set():
        return

    logger.debug("Shutdown initiated, waiting for metering calls to complete...")
    budget = shutdown_budget_seconds()
    deadline = time.monotonic() + budget

    # Both run before shutdown_event is set: every integration's metering
    # coroutine returns early once it sees the event, so a queued event, or
    # an overflowed one the buffer has not built yet, would be dropped.
    undelivered = _drain_worker_queue(deadline) + _build_overflow(deadline)
    shutdown_event.set()
    # Last, so events that exhausted retries, including those of the queue
    # drain above, get a final attempt.
    _drain_buffer(deadline)
    if undelivered:
        logger.warning(
            "%d metering event(s) still queued, in flight or unbuilt after the %ss shutdown budget (%s); "
            "their usage may not be delivered.",
            undelivered, budget, Config.ENV_REVENIUM_SHUTDOWN_TIMEOUT_SECONDS,
        )
    logger.debug("Shutdown complete")


def _seconds_left(deadline: float) -> float:
    return max(0.0, deadline - time.monotonic())


def _drain_worker_queue(deadline: float) -> int:
    """Let the worker pool deliver its queued events until ``deadline``; return how many are left."""
    return metering_pool.drain(_seconds_left(deadline))


def _build_overflow(deadline: float) -> int:
    """Build the buffer's overflowed tasks until ``deadline``; return how many are left unbuilt."""
    try:
        if metering_buffer._buffer is not None:
            return metering_buffer._buffer.build_all_overflow(deadline_seconds=_seconds_left(deadline))
    except Exception as e:
        logger.debug("Building overflowed metering events during shutdown failed: %s", e)
    return 0


def _drain_buffer(deadline: float) -> None:
    try:
        if metering_buffer._buffer is not None:
            metering_buffer._buffer.flush(deadline_seconds=_seconds_left(deadline))
    except Exception as e:
        logger.debug("Metering buffer drain during shutdown failed: %s", e)


atexit.register(handle_exit)
install_requested_signal_handlers(handle_exit)


async def _run_sync_callable(func: Callable[[], Any]) -> Any:
    if shutdown_event.is_set():
        logger.debug("Skipping sync function execution due to shutdown.")
        return None
    try:
        return func()
    except Exception as e:
        logger.warning(f"Exception in wrapped sync function: {e}", exc_info=True)
        raise


def run_async_in_thread(coroutine_or_func) -> Optional[MeteringTask]:
    """Queue a metering coroutine (or sync callable) for background delivery.

    Returns immediately. A fixed pool of daemon workers delivers queued
    events; when the queue is full the event goes to the store-and-forward
    buffer instead, so no call ever blocks on metering or starts a thread.

    Args:
        coroutine_or_func: Either an awaitable coroutine or a regular function

    Returns:
        Optional[MeteringTask]: A handle whose ``join``/``is_alive`` follow the
        event's delivery, or None if shutdown initiated or the input is invalid.
    """
    if shutdown_event.is_set():
        logger.warning("Not queueing metering event during shutdown")
        return None

    if asyncio.iscoroutine(coroutine_or_func):
        coro = coroutine_or_func
    elif callable(coroutine_or_func):
        coro = _run_sync_callable(coroutine_or_func)
    else:
        logger.error(
            "Invalid type passed to run_async_in_thread: %s. Expected coroutine or callable.",
            type(coroutine_or_func),
        )
        return None

    # Capture the current context so contextvars (e.g. idempotency_key) propagate
    # into the worker thread, which otherwise runs in its own context.
    pool = metering_pool.get_pool(shutdown_event.is_set)
    task = pool.new_task(coro, contextvars.copy_context())
    try:
        pool.submit(task)
    except RuntimeError as e:
        # Raised when a worker cannot start, e.g. during interpreter shutdown.
        logger.error(f"Failed to queue metering event {task.name}: {e}", exc_info=True)
        task.discard()
        return None
    logger.debug(f"Queued metering event {task.name}.")
    return task
