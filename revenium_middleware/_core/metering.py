import os
import sys
import time
import logging
import asyncio
import threading
import contextlib
import contextvars
import atexit
import math
import re
import http.cookiejar
import weakref
from collections import OrderedDict
from typing import Literal, Awaitable, Any, Dict, Iterator, Optional, Callable, Tuple

import httpx

from revenium_middleware._metering import ReveniumMetering
from revenium_middleware._metering._base_client import DefaultHttpxClient
from revenium_middleware._core.config import Config, read_env_number, validate_api_key
from revenium_middleware._core.log_sanitize import redact_error_text
from revenium_middleware._core import metering_buffer, metering_pool
from revenium_middleware._core.metering_pool import MeteringTask
from revenium_middleware._core.shutdown_signals import install_requested_signal_handlers, terminate_if_deferred

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


def _new_client(
    api_key: str, base_url: Optional[str], http_client: Optional[httpx.Client] = None
) -> ReveniumMetering:
    options: Dict[str, Any] = {"timeout": metering_client_timeout(), "max_retries": metering_client_max_retries()}
    if base_url is not None:
        options["base_url"] = base_url
    if http_client is not None:
        options["http_client"] = http_client
    new_client = ReveniumMetering(api_key=api_key, **options)
    _creating_pid[new_client] = os.getpid()
    return new_client


# The process that built each client, so a forked child can tell the clients
# it inherited, whichever block, record or cache still holds them.
_creating_pid: "weakref.WeakKeyDictionary[ReveniumMetering, int]" = weakref.WeakKeyDictionary()


DEFAULT_METERING_BASE_URL = "https://api.revenium.ai/meter/"
_MAX_LOGGED_ADDRESS_CHARS = 16
_LEADING_USERINFO_PATTERN = re.compile(r"^[^/\s@]*@")


def _without_schemeless_userinfo(address: str) -> str:
    """Mask the user info of an address that has no scheme, which redact_error_text only finds after ``://``."""
    if "://" in address:
        return address
    return _LEADING_USERINFO_PATTERN.sub("[REDACTED]@", address)


def _valid_base_url(base_url_raw: Optional[str]) -> Optional[str]:
    """Return ``base_url_raw`` stripped when it is an http(s) URL, else None."""
    if base_url_raw is None:
        return None
    stripped = base_url_raw.strip()
    if stripped.startswith("http://") or stripped.startswith("https://"):
        return stripped
    return None


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
    api_key = api_key.strip() if api_key else api_key
    validated_base_url = _valid_base_url(base_url_raw)
    if base_url_raw is not None and validated_base_url is None:
        logger.warning(
            "REVENIUM_METERING_BASE_URL=%r is invalid (must start with http:// or https://); "
            "falling back to default endpoint",
            redact_error_text(_without_schemeless_userinfo(base_url_raw.strip()), _MAX_LOGGED_ADDRESS_CHARS),
        )

    if not api_key:
        logger.error(
            "REVENIUM_METERING_API_KEY environment variable is not set or is blank. "
            "Metering is disabled and no usage data will be sent to Revenium. "
            "Set it, or call revenium_middleware.initialize_metering(api_key=...) in code; "
            "configure() sets the key for tool events only."
        )
        return None

    validate_api_key(api_key)
    if validated_base_url is not None:
        return _new_client(api_key, validated_base_url)
    if base_url_raw is not None:
        return _new_client(api_key, DEFAULT_METERING_BASE_URL)
    return _new_client(api_key, None)


# Remembers a malformed env key so the lazy path doesn't re-raise per call.
_last_failed_key: Optional[str] = None

api_key = (os.environ.get("REVENIUM_METERING_API_KEY") or "").strip()
_base_url_raw = os.environ.get("REVENIUM_METERING_BASE_URL")
try:
    client = _build_metering_client(api_key, _base_url_raw)
except ValueError as e:
    client = None
    _last_failed_key = api_key
    logger.warning(
        "REVENIUM_METERING_API_KEY is malformed (%s); metering is disabled until a valid key is set",
        e,
    )

# Set in a forked child: the key and address of the client it inherited, so
# its first record builds its own client with them, once.
_client_settings_from_parent: Optional[Tuple[str, str]] = None
_client_rebuild_lock = threading.Lock()


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
    global client, _client_settings_from_parent
    _client_settings_from_parent = None
    key = api_key if api_key is not None else os.environ.get("REVENIUM_METERING_API_KEY")
    url = base_url if base_url is not None else os.environ.get("REVENIUM_METERING_BASE_URL")
    new_client = _build_metering_client(key, url)
    client = new_client
    _sync_client_reexports(new_client)
    return new_client is not None


def get_client() -> Optional[ReveniumMetering]:
    """Return the metering client of the enclosing ``metering_key()`` block, else ``process_client()``."""
    scoped = _scoped_client.get()
    return _own_client(scoped) if scoped is not None else process_client()


def process_client() -> Optional[ReveniumMetering]:
    """Return the process-wide metering client, retrying the environment when unset.

    Covers env vars populated after import: as soon as
    ``REVENIUM_METERING_API_KEY`` appears, the next metering event builds the
    client instead of silently no-opping forever.

    A missing/unset key is logged at ERROR level by the build path; a
    malformed env key is logged once as a warning and metering stays
    disabled, whereas the explicit ``initialize_metering()`` raises
    ``ValueError`` so programmatic misconfiguration fails loudly.
    """
    global _last_failed_key
    if client is None and _client_settings_from_parent is not None:
        _rebuild_client_inherited_at_fork()
    if client is None:
        env_key = (os.environ.get("REVENIUM_METERING_API_KEY") or "").strip()
        if env_key and env_key != _last_failed_key:
            try:
                initialize_metering()
            except ValueError as e:
                logger.warning("Deferred metering initialization failed: %s", e)
                _last_failed_key = env_key
    return client


def _rebuild_client_inherited_at_fork() -> None:
    global client, _client_settings_from_parent
    with _client_rebuild_lock:
        settings = _client_settings_from_parent
        if client is not None or settings is None:
            return
        _client_settings_from_parent = None
        new_client = _new_client(*settings)
        client = new_client
        _sync_client_reexports(new_client)


# The client current when a metering event was queued, or when its call
# started for a call that pinned it with recorded_under(), carried in the
# event's context so a worker that runs it later, or a buffer that builds it
# after it overflowed, sends it under the key it was made with.
_UNPINNED = object()
_enqueued_client: "contextvars.ContextVar[Any]" = contextvars.ContextVar(
    "revenium_enqueued_client", default=_UNPINNED
)


def current_client() -> Optional[ReveniumMetering]:
    """Return the client pinned when this event was queued (None if metering was off), else ``get_client()``."""
    pinned = _enqueued_client.get()
    if pinned is _UNPINNED:
        return get_client()
    return None if pinned is None else _own_client(pinned)


@contextlib.contextmanager
def recorded_under(metering_client: Optional[ReveniumMetering]) -> Iterator[None]:
    """Queue the records of this block under ``metering_client``, the client current when their call started.

    None is a valid pin: metering was off when the call started, so its record is not sent.
    """
    token = _enqueued_client.set(metering_client)
    try:
        yield
    finally:
        _enqueued_client.reset(token)


_scoped_client: "contextvars.ContextVar[Optional[ReveniumMetering]]" = contextvars.ContextVar(
    "revenium_scoped_client", default=None
)

# Every scoped client sends through one shared connection pool, and its key
# goes out per request, so an evicted client holds no sockets of its own.
MAX_SCOPED_CLIENTS = 256
_scoped_clients: "OrderedDict[Tuple[str, str], ReveniumMetering]" = OrderedDict()
_scoped_clients_lock = threading.Lock()
_scoped_http_client: Optional[httpx.Client] = None


class _SharedHttpClient(DefaultHttpxClient):
    """The connection pool every ``metering_key()`` client sends through.

    ``ReveniumMetering.close()`` and its ``with`` block close the HTTP client
    they send through; for this pool that would stop delivery under every
    customer key, so closing it does nothing.
    """

    def close(self) -> None:
        pass

    def __enter__(self) -> "_SharedHttpClient":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        pass


def _cookie_jar_that_stores_nothing() -> http.cookiejar.CookieJar:
    """One customer's response must not set a cookie that is sent with another customer's records."""
    return http.cookiejar.CookieJar(policy=http.cookiejar.DefaultCookiePolicy(allowed_domains=[]))


def _shared_http_client() -> httpx.Client:
    """Build the shared pool on first use; the caller holds ``_scoped_clients_lock``."""
    global _scoped_http_client
    if _scoped_http_client is None:
        _scoped_http_client = _SharedHttpClient(
            timeout=metering_client_timeout(), cookies=_cookie_jar_that_stores_nothing()
        )
    return _scoped_http_client


# Every client metering_key() built, including ones evicted from the cache
# while a streamed call that started in the block still holds them.
_scoped_client_objects: "weakref.WeakSet[ReveniumMetering]" = weakref.WeakSet()


def is_scoped_client(metering_client: Optional[ReveniumMetering]) -> bool:
    """Whether ``metering_client`` was built by ``metering_key()``."""
    return metering_client is not None and metering_client in _scoped_client_objects


def _scoped_base_url(base_url: Optional[str]) -> str:
    if base_url is None:
        process = process_client()
        if process is not None:
            return str(process.base_url)
        base_url = os.environ.get("REVENIUM_METERING_BASE_URL")
        if base_url is None:
            return DEFAULT_METERING_BASE_URL
    validated = _valid_base_url(base_url)
    if validated is None:
        logger.warning(
            "metering_key() base URL is invalid (must start with http:// or https://); "
            "falling back to default endpoint"
        )
        return DEFAULT_METERING_BASE_URL
    return validated


def _own_client(metering_client: ReveniumMetering) -> ReveniumMetering:
    """``metering_client``, or, in a forked child that inherited it inside a block, a client of its own with its key."""
    if _creating_pid.get(metering_client, os.getpid()) == os.getpid():
        return metering_client
    _held_from_parent.setdefault(id(metering_client), metering_client)
    return _scoped_client_for(metering_client.api_key, str(metering_client.base_url))


def _scoped_client_for(api_key: str, base_url: str) -> ReveniumMetering:
    cache_key = (api_key, base_url)
    with _scoped_clients_lock:
        scoped = _scoped_clients.get(cache_key)
        if scoped is None:
            scoped = _new_client(api_key, base_url, http_client=_shared_http_client())
            _scoped_client_objects.add(scoped)
            _scoped_clients[cache_key] = scoped
            while len(_scoped_clients) > MAX_SCOPED_CLIENTS:
                _scoped_clients.popitem(last=False)
        else:
            _scoped_clients.move_to_end(cache_key)
        return scoped


@contextlib.contextmanager
def metering_key(api_key: str, base_url: Optional[str] = None) -> Iterator[None]:
    """Send the usage records and tool events of this block under ``api_key``.

    Scoped to the current thread or asyncio task: a thread started inside the
    block does not inherit it, an asyncio task created inside it does. Other
    threads and tasks keep their own key, so a service can meter several of its
    customers at once. ``base_url`` defaults to the address of the process-wide
    client (``initialize_metering()`` or ``REVENIUM_METERING_BASE_URL``). A key
    set with ``configure(api_key=...)`` still wins for tool events and goes
    with ``configure()``'s address or the process-wide address, never ``base_url``.

    Spending caps keep following the process-wide key and ``REVENIUM_TEAM_ID``.
    Outcomes and jobs are not affected; they need a write key (``rev_sk_``).

    Raises:
        ValueError: if ``api_key`` is missing, empty or malformed.
    """
    api_key = api_key.strip() if api_key else api_key
    if not api_key:
        raise ValueError("metering_key() needs a Revenium API key; it was missing or empty")
    validate_api_key(api_key)
    token = _scoped_client.set(_scoped_client_for(api_key, _scoped_base_url(base_url)))
    try:
        yield
    finally:
        _scoped_client.reset(token)

shutdown_event = threading.Event()

DEFAULT_SHUTDOWN_TIMEOUT_SECONDS = 5.0


def _is_valid_budget(budget: float) -> bool:
    return math.isfinite(budget) and budget >= 0


def shutdown_budget_seconds() -> float:
    """Total seconds the exit drain may spend, from ``REVENIUM_SHUTDOWN_TIMEOUT_SECONDS``."""
    return read_env_number(
        Config.ENV_REVENIUM_SHUTDOWN_TIMEOUT_SECONDS, DEFAULT_SHUTDOWN_TIMEOUT_SECONDS, float, _is_valid_budget
    )


class _ExitDrain:
    """The one exit drain: its budget and deadline, the thread running it, and when it is over."""

    def __init__(self, budget: float, deadline: Optional[float] = None):
        self.budget = budget
        self.deadline = time.monotonic() + budget if deadline is None else deadline
        self.thread = threading.current_thread()
        self.over = threading.Event()


# Reentrant because the opt-in SIGTERM handler runs on the main thread,
# possibly inside this lock.
_exit_lock = threading.RLock()
_running_exit_drain: Optional[_ExitDrain] = None
# Set by a spawned child's exit flush, whose budget the interpreter's exit drain then shares.
_child_exit_deadline: Optional[float] = None


def handle_exit() -> None:
    """Deliver queued metering, build overflowed events and flush the buffer, within one shared budget.

    Runs once. A call made while the drain runs on another thread waits for
    it, until its deadline; one made on the drain's own thread returns at once.
    """
    drain_or_await_exit()


def drain_or_await_exit() -> bool:
    """Run the exit drain, or wait for the one running; False when it runs underneath this call, on this thread.

    That is the opt-in SIGTERM handler interrupting the drain: it must not
    end the process before the interrupted drain finishes.
    """
    drain, started_here = _start_or_join_exit_drain()
    if drain is None:
        return True
    if not started_here:
        if drain.thread is threading.current_thread():
            return False
        drain.over.wait(_seconds_left(drain.deadline))
        return True
    try:
        _drain_within(drain.budget, drain.deadline)
    finally:
        drain.over.set()
        _end_exit_drain()
        terminate_if_deferred()
    return True


def _start_or_join_exit_drain() -> Tuple[Optional[_ExitDrain], bool]:
    global _running_exit_drain
    with _exit_lock:
        if _running_exit_drain is not None:
            return _running_exit_drain, False
        if shutdown_event.is_set():
            return None, False
        _running_exit_drain = _ExitDrain(shutdown_budget_seconds(), _child_exit_deadline)
        return _running_exit_drain, True


def _end_exit_drain() -> None:
    global _running_exit_drain
    with _exit_lock:
        _running_exit_drain = None


def _drain_within(budget: float, deadline: float) -> None:
    logger.debug("Shutdown initiated, waiting for metering calls to complete...")
    # Both run before shutdown_event is set: the provider integrations'
    # metering coroutines return early once they see it, so a queued event,
    # or an overflowed one the buffer has not built yet, would be dropped.
    # The LiteLLM guardrail and proxy callbacks and tool events do not check
    # it and are sent whenever they run.
    queued = _drain_worker_queue(deadline)
    lost_before = _overflow_lost_to_deadlines()
    _build_overflow(deadline)
    shutdown_event.set()
    # Last, so events that exhausted retries, including those of the queue
    # drain above, get a final attempt.
    undelivered = queued + _drain_buffer(deadline) + _overflow_lost_to_deadlines() - lost_before
    _warn_if_undelivered(undelivered, budget)
    logger.debug("Shutdown complete")


def _warn_if_undelivered(undelivered: int, budget: float) -> None:
    if undelivered:
        logger.warning(
            "%d metering event(s) still queued, in flight, unbuilt or unsent after the %ss shutdown budget (%s); "
            "their usage may not be delivered.",
            undelivered, budget, Config.ENV_REVENIUM_SHUTDOWN_TIMEOUT_SECONDS,
        )


def _seconds_left(deadline: float) -> float:
    return max(0.0, deadline - time.monotonic())


def _drain_worker_queue(deadline: float) -> int:
    """Let the worker pool deliver its queued events until ``deadline``; return how many are left."""
    return metering_pool.drain(_seconds_left(deadline))


def _build_overflow(deadline: float) -> None:
    """Build the buffer's overflowed tasks until ``deadline``; the final flush counts any left unbuilt."""
    try:
        if metering_buffer._buffer is not None:
            metering_buffer._buffer.build_all_overflow(deadline_seconds=_seconds_left(deadline))
    except Exception as e:
        logger.debug("Building overflowed metering events during shutdown failed: %s", e)


def _overflow_lost_to_deadlines() -> int:
    buffer = metering_buffer._buffer
    return 0 if buffer is None else buffer.overflow_lost_to_deadlines()


def _drain_buffer(deadline: float) -> int:
    """Flush the buffer until ``deadline``; return how many events it still holds or is still building."""
    buffer = metering_buffer._buffer
    if buffer is None:
        return 0
    try:
        remaining = buffer.flush(deadline_seconds=_seconds_left(deadline))["remaining"]
    except Exception as e:
        logger.debug("Metering buffer drain during shutdown failed: %s", e)
        remaining = buffer.undelivered()
    return remaining + buffer.overflow_in_build()


def flush_metering(timeout_seconds: Optional[float] = None) -> int:
    """Send the usage records queued or held so far, waiting at most ``timeout_seconds``; return how many are left.

    Call it at the end of a job in a process that ends without a normal
    interpreter exit, such as a raw ``os.fork()`` worker or an RQ work horse,
    which end through ``os._exit`` and so skip the exit flush. Metering
    carries on afterwards. ``timeout_seconds`` defaults to
    ``REVENIUM_SHUTDOWN_TIMEOUT_SECONDS`` (5 seconds). Returns 0 when every
    record was sent, and 0 once the exit flush has run.

    Raises:
        ValueError: if ``timeout_seconds`` is negative or not finite.
    """
    if timeout_seconds is None:
        budget = shutdown_budget_seconds()
    elif _is_valid_budget(timeout_seconds):
        budget = timeout_seconds
    else:
        raise ValueError(f"flush_metering() needs a timeout of zero or more seconds; got {timeout_seconds!r}")
    return _flush_until(time.monotonic() + budget)


def _flush_until(deadline: float) -> int:
    if shutdown_event.is_set():
        return 0
    lost_before = _overflow_lost_to_deadlines()
    left = _drain_worker_queue(deadline) + _drain_buffer(deadline)
    return left + _overflow_lost_to_deadlines() - lost_before


def _is_spawned_child() -> bool:
    """Whether this child runs under ``multiprocessing.spawn``, which ends it through ``sys.exit`` and so runs atexit.

    Fork and forkserver children end through ``os._exit`` instead. A forkserver child also runs ``spawn._main``,
    so only ``spawn_main``, the entry point that a spawned child alone passes through, tells them apart.
    """
    spawn = sys.modules.get("multiprocessing.spawn")
    if spawn is None:
        return False
    frame: Any = sys._getframe()
    while frame is not None:
        if frame.f_code is spawn.spawn_main.__code__:
            return True
        frame = frame.f_back
    return False


def _flush_at_child_exit() -> None:
    global _child_exit_deadline
    budget = shutdown_budget_seconds()
    deadline = time.monotonic() + budget
    if _is_spawned_child():
        _child_exit_deadline = deadline
        _flush_until(deadline)
    else:
        _warn_if_undelivered(_flush_until(deadline), budget)


_child_exit_flush_armed = False
_child_exit_flush_lock = threading.Lock()


def _arm_child_exit_flush() -> None:
    """In a multiprocessing child, flush when its target returns, which ends the process without atexit."""
    global _child_exit_flush_armed
    if _child_exit_flush_armed:
        return
    with _child_exit_flush_lock:
        if _child_exit_flush_armed:
            return
        _child_exit_flush_armed = True
        parent_process = getattr(sys.modules.get("multiprocessing"), "parent_process", None)
        if parent_process is None or parent_process() is None:
            return
        import multiprocessing.util

        # Registered here, in the child, and not from an at-fork hook: the
        # child's bootstrap clears the finalizers registered before it runs.
        multiprocessing.util.Finalize(None, _flush_at_child_exit, exitpriority=0)


atexit.register(handle_exit)
install_requested_signal_handlers(drain_or_await_exit)

# Inherited clients stay referenced in a forked child: collecting one closes
# its connection pool, which takes a lock a parent thread may have held at the fork.
_held_from_parent: Dict[int, Any] = {}


def _forget_parent_clients() -> None:
    """Give a forked child its own metering clients and exit drain, never the parent's sockets or held locks."""
    global client, _scoped_clients, _scoped_clients_lock, _scoped_http_client, _client_settings_from_parent
    global _client_rebuild_lock, _exit_lock, _running_exit_drain, _child_exit_deadline
    global _child_exit_flush_armed, _child_exit_flush_lock
    inherited = list(_scoped_clients.values())
    if client is not None:
        inherited.append(client)
        _client_settings_from_parent = (client.api_key, str(client.base_url))
    if _scoped_http_client is not None:
        inherited.append(_scoped_http_client)
    _held_from_parent.update((id(each), each) for each in inherited)
    client = None
    _scoped_clients = OrderedDict()
    _scoped_clients_lock = threading.Lock()
    _scoped_http_client = None
    _client_rebuild_lock = threading.Lock()
    _exit_lock = threading.RLock()
    _running_exit_drain = None
    _child_exit_deadline = None
    _child_exit_flush_armed = False
    _child_exit_flush_lock = threading.Lock()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_forget_parent_clients)


async def _run_sync_callable(func: Callable[[], Any]) -> Any:
    if shutdown_event.is_set():
        logger.debug("Skipping sync function execution due to shutdown.")
        return None
    try:
        return func()
    except Exception as e:
        logger.warning(f"Exception in wrapped sync function: {e}", exc_info=True)
        raise


def run_async_in_thread(coroutine_or_func, *, gated_by_circuit: bool = True) -> Optional[MeteringTask]:
    """Queue a metering coroutine (or sync callable) for background delivery.

    Returns immediately. A fixed pool of daemon workers delivers queued
    events; when the queue is full the event goes to the store-and-forward
    buffer instead, so no call ever blocks on metering or starts a thread.

    Args:
        coroutine_or_func: Either an awaitable coroutine or a regular function
        gated_by_circuit: False for an event the AI delivery circuit must not
            hold back, such as a tool event, which may go to another endpoint

    Returns:
        Optional[MeteringTask]: A handle whose ``join``/``is_alive`` follow the
        event's delivery, or None if shutdown initiated or the input is invalid.
    """
    if shutdown_event.is_set():
        logger.warning("Not queueing metering event during shutdown")
        return None
    _arm_child_exit_flush()

    if asyncio.iscoroutine(coroutine_or_func):
        coro, blocks_synchronously = coroutine_or_func, False
    elif callable(coroutine_or_func):
        coro, blocks_synchronously = _run_sync_callable(coroutine_or_func), True
    else:
        logger.error(
            "Invalid type passed to run_async_in_thread: %s. Expected coroutine or callable.",
            type(coroutine_or_func),
        )
        return None

    # Capture the current context so contextvars (e.g. idempotency_key) propagate
    # into the worker thread, which otherwise runs in its own context.
    ctx = contextvars.copy_context()
    ctx.run(_enqueued_client.set, current_client())
    pool = metering_pool.get_pool(shutdown_event.is_set)
    task = pool.new_task(
        coro, ctx,
        blocks_synchronously=blocks_synchronously, gated_by_circuit=gated_by_circuit,
    )
    try:
        pool.submit(task)
    except RuntimeError as e:
        # Raised when a worker cannot start, e.g. during interpreter shutdown.
        logger.error(f"Failed to queue metering event {task.name}: {e}", exc_info=True)
        task.discard()
        return None
    logger.debug(f"Queued metering event {task.name}.")
    return task
