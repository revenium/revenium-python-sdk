import os
import time
import logging
import asyncio
import threading
import contextvars
import atexit
import signal
from typing import Literal, Awaitable, Any, Optional, Callable
from revenium_middleware._metering import ReveniumMetering
from revenium_middleware._core.config import validate_api_key

# Get the logger that was configured in __init__.py
logger = logging.getLogger("revenium_middleware")

# Define a StopReason literal type for strict typing of stop_reason
StopReason = Literal["END", "END_SEQUENCE", "TIMEOUT", "TOKEN_LIMIT", "COST_LIMIT", "COMPLETION_LIMIT", "ERROR"]

def _build_metering_client(
    api_key: Optional[str],
    base_url_raw: Optional[str],
) -> Optional[ReveniumMetering]:
    """Construct a ReveniumMetering client from raw env-var inputs.

    Returns ``None`` when the API key is missing/empty (logs a warning).
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
        logger.warning("Metering API key not set -- metering is disabled")
        return None

    validate_api_key(api_key)
    if validated_base_url is not None:
        return ReveniumMetering(api_key=api_key, base_url=validated_base_url)
    if base_url_raw is not None:
        return ReveniumMetering(api_key=api_key, base_url="https://api.revenium.ai/meter/")
    return ReveniumMetering(api_key=api_key)


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

    Returns True when metering is enabled after the call. Raises ``ValueError``
    for a malformed API key -- explicit configuration fails loudly, unlike the
    lazy ``get_client()`` path, which logs a warning and leaves metering
    disabled (best-effort by design).
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

    Best-effort by design: a malformed env key is logged once and metering
    stays disabled, whereas the explicit ``initialize_metering()`` raises
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

_threads_lock = threading.Lock()
active_threads = []
shutdown_event = threading.Event()

def handle_exit(signum=None, frame=None):
    # Check if shutdown is already initiated to prevent redundant logging/actions
    if shutdown_event.is_set():
        return

    logger.debug("Shutdown initiated, waiting for metering calls to complete...")
    shutdown_event.set()

    # Give threads a chance to notice the shutdown event
    # Use a small delay, but avoid blocking excessively if called from signal handler
    try:
        time.sleep(0.1)
    except InterruptedError:
        # Handle potential interruption if called from a signal handler during sleep
        logger.debug("Sleep interrupted during shutdown.")
        # Ensure the event is still set
        shutdown_event.set()


    # Drain the store-and-forward buffer (bounded) before joining metering
    # threads, so events that already exhausted retries get a final attempt.
    try:
        from revenium_middleware._core import metering_buffer

        if metering_buffer._buffer is not None:
            metering_buffer._buffer.flush(deadline_seconds=5.0)
    except Exception as e:
        logger.debug("Metering buffer drain during shutdown failed: %s", e)

    with _threads_lock:
        threads_to_join = list(active_threads)
    for thread in threads_to_join:
        if thread.is_alive():
            logger.debug(f"Waiting for metering thread {thread.name} to finish...")
            thread.join(timeout=5.0)
            if thread.is_alive():
                logger.warning(f"Metering thread {thread.name} did not complete in time.")
            else:
                logger.debug(f"Metering thread {thread.name} finished.")
        with _threads_lock:
            if thread in active_threads:
                active_threads.remove(thread)


    logger.debug("Shutdown complete")
    
    # If called from a signal handler, exit the program
    if signum is not None:
        logger.debug(f"Exiting due to signal {signum}")
        os._exit(0)

# Always register atexit handler, works in any thread
atexit.register(handle_exit)

# Only register signal handlers if in the main thread
if threading.current_thread() is threading.main_thread():
    try:
        signal.signal(signal.SIGINT, handle_exit)
        signal.signal(signal.SIGTERM, handle_exit)
        logger.debug("SIGINT and SIGTERM handlers registered.")
    except ValueError as e:
        # This can happen in environments where signal handling is restricted (e.g., mod_wsgi)
        logger.warning(f"Could not register signal handlers: {e}. Shutdown will rely on atexit.")
else:
    logger.debug("Not running in main thread, skipping signal handler registration. Shutdown will rely on atexit.")


class MeteringThread(threading.Thread):
    def __init__(self, coro, *args, ctx: Optional[contextvars.Context] = None, **kwargs):
        # Default to non-daemon threads so atexit handlers wait for them
        daemon = kwargs.pop('daemon', False)
        super().__init__(*args, **kwargs)
        self.coro = coro
        self.ctx = ctx
        self.daemon = daemon # Store daemon status
        self.error = None
        self.loop = None
        # Assign a more descriptive name if not provided
        if self.name is None:
             self.name = f"MeteringThread-{id(self)}"


    def run(self):
        # Check shutdown event *before* starting the loop
        if shutdown_event.is_set():
            logger.debug(f"Metering thread {self.name} not starting due to shutdown.")
            with _threads_lock:
                if self in active_threads:
                    active_threads.remove(self)
            return

        try:
            # Create and set a new event loop for this thread
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
            logger.debug(f"Metering thread {self.name} started with loop {id(self.loop)}")
            try:
                # Run the coroutine until it completes, inside the captured context if available
                if self.ctx is not None:
                    self.ctx.run(self.loop.run_until_complete, self.coro)
                else:
                    self.loop.run_until_complete(self.coro)
            finally:
                # Ensure async generators are properly shut down
                logger.debug(f"Shutting down async generators for loop {id(self.loop)} in thread {self.name}")
                self.loop.run_until_complete(self.loop.shutdown_asyncgens())
                # Close the event loop
                logger.debug(f"Closing event loop {id(self.loop)} in thread {self.name}")
                self.loop.close()
                logger.debug(f"Event loop {id(self.loop)} closed in thread {self.name}")
        except Exception as e:
            # Log errors unless it's during shutdown
            if not shutdown_event.is_set():
                self.error = e
                # Use exc_info=True to include traceback in the log
                logger.warning(f"Error in metering thread {self.name}: {str(e)}", exc_info=True)
            else:
                logger.debug(f"Exception ignored in metering thread {self.name} during shutdown: {str(e)}")
        finally:
            with _threads_lock:
                if self in active_threads:
                    active_threads.remove(self)
                    logger.debug(f"Removed thread {self.name} from active list.")


def run_async_in_thread(coroutine_or_func):
    """
    Helper function to run an async coroutine or a regular function in a background thread
    with better handling of interpreter shutdown.

    Args:
        coroutine_or_func: Either an awaitable coroutine or a regular function

    Returns:
        Optional[threading.Thread]: The thread running the task, or None if shutdown initiated.
    """
    if shutdown_event.is_set():
        logger.warning("Not starting new metering thread during shutdown")
        return None

    # Check if we received a coroutine or a regular function
    if asyncio.iscoroutine(coroutine_or_func):
        # It's a coroutine, use it directly
        coro = coroutine_or_func
    elif callable(coroutine_or_func):
         # It's a callable (sync function), wrap it
         async def wrapper():
             # Check shutdown again before potentially long-running sync call
             if shutdown_event.is_set():
                 logger.debug("Skipping sync function execution due to shutdown.")
                 return None # Or raise an exception? Returning None seems safer.
             try:
                 return coroutine_or_func()
             except Exception as e:
                 logger.warning(f"Exception in wrapped sync function: {e}", exc_info=True)
                 # Propagate or handle error as needed
                 raise # Re-raise the exception to be caught by MeteringThread run method
         coro = wrapper()
    else:
         # If it's neither a coroutine nor callable, it's likely an error or unexpected input
         logger.error(f"Invalid type passed to run_async_in_thread: {type(coroutine_or_func)}. Expected coroutine or callable.")
         # Decide how to handle this: return None, raise TypeError, etc.
         # Returning None might be safest to avoid crashing the caller unexpectedly.
         return None


    # Capture the current context so contextvars (e.g. idempotency_key) propagate
    # into the worker thread, which otherwise starts with an empty context.
    ctx = contextvars.copy_context()

    # Create and start the thread
    # Pass daemon=False explicitly if that's the desired default
    thread = MeteringThread(coro, ctx=ctx, daemon=False)
    with _threads_lock:
        active_threads.append(thread)
    logger.debug(f"Starting and adding thread {thread.name} to active list.")
    try:
        thread.start()
    except RuntimeError as e:
        logger.error(f"Failed to start thread {thread.name}: {e}", exc_info=True)
        # Clean up: remove the thread we failed to start
        with _threads_lock:
            if thread in active_threads:
                active_threads.remove(thread)
        return None

    return thread
