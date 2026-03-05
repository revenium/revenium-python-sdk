import os
import time
import logging
import asyncio
import threading
import atexit
import signal
from typing import Literal, Awaitable, Any, Optional, Callable
from revenium_metering import ReveniumMetering

# Get the logger that was configured in __init__.py
logger = logging.getLogger("revenium_middleware")

# Define a StopReason literal type for strict typing of stop_reason
StopReason = Literal["END", "END_SEQUENCE", "TIMEOUT", "TOKEN_LIMIT", "COST_LIMIT", "COMPLETION_LIMIT", "ERROR"]

api_key = os.environ.get("REVENIUM_METERING_API_KEY") or "DUMMY_API_KEY"
client = ReveniumMetering(api_key=api_key)

# Keep track of active metering threads
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


    # Iterate over a copy of the list to avoid modification issues during iteration
    threads_to_join = list(active_threads)
    for thread in threads_to_join:
        if thread.is_alive():
            logger.debug(f"Waiting for metering thread {thread.name} to finish...")
            thread.join(timeout=5.0) # Wait up to 5 seconds for the thread
            if thread.is_alive():
                logger.warning(f"Metering thread {thread.name} did not complete in time.")
            else:
                logger.debug(f"Metering thread {thread.name} finished.")
        # Clean up thread reference if it's already finished or after joining
        if thread in active_threads:
             try:
                 active_threads.remove(thread)
             except ValueError:
                 # Thread might have been removed by itself in the finally block
                 pass


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
    def __init__(self, coro, *args, **kwargs):
        # Default to non-daemon threads so atexit handlers wait for them
        daemon = kwargs.pop('daemon', False)
        super().__init__(*args, **kwargs)
        self.coro = coro
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
            # Ensure thread is removed from active_threads if it was added
            if self in active_threads:
                try:
                    active_threads.remove(self)
                except ValueError:
                    pass # Should not happen if logic is correct, but handle defensively
            return

        try:
            # Create and set a new event loop for this thread
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
            logger.debug(f"Metering thread {self.name} started with loop {id(self.loop)}")
            try:
                # Run the coroutine until it completes
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
            # Ensure the thread is removed from the active list upon completion or error
            if self in active_threads:
                try:
                    active_threads.remove(self)
                    logger.debug(f"Removed thread {self.name} from active list.")
                except ValueError:
                     # Can happen if handle_exit removes it first during shutdown join timeout
                     logger.debug(f"Thread {self.name} already removed from active list.")
            else:
                 # This case might indicate the thread wasn't properly added or removed elsewhere
                 logger.warning(f"Thread {self.name} finished but was not found in active_threads list.")


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


    # Create and start the thread
    # Pass daemon=False explicitly if that's the desired default
    thread = MeteringThread(coro, daemon=False)
    active_threads.append(thread)
    logger.debug(f"Starting and adding thread {thread.name} to active list (now {len(active_threads)} threads).")
    try:
        thread.start()
    except RuntimeError as e:
        logger.error(f"Failed to start thread {thread.name}: {e}", exc_info=True)
        # Clean up: remove the thread we failed to start
        if thread in active_threads:
            try:
                active_threads.remove(thread)
            except ValueError:
                pass # Should be there, but handle defensively
        return None # Indicate failure to start

    return thread
