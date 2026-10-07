"""Opt-in SIGTERM hook that drains metering before a default termination.

The SDK leaves the host's signal handling alone unless
``REVENIUM_INSTALL_SIGNAL_HANDLERS`` is set. Shutdown otherwise relies on
``atexit``, which runs whenever the interpreter exits normally, including
after ``KeyboardInterrupt`` (Python's default SIGINT behaviour) and after a
host's own SIGTERM handler exits through ``sys.exit``.

The exit ``atexit`` cannot see is SIGTERM with the default disposition: the
kernel ends the process immediately. With the flag set, that case drains
first and then terminates through the same default disposition; a SIGTERM
that arrives while the drain runs lets it finish first, waiting for a drain on
another thread and, when it interrupts the drain on the main thread, ending
the process once that drain returns. A handler
installed before this import is chained to unchanged and without a drain.
When it exits normally the ``atexit`` drain runs; when it restores the
default disposition and re-raises SIGTERM, as uvicorn does after its own
graceful shutdown, the process ends before ``atexit`` and nothing is drained
(the LiteLLM proxy case, BACK-3910).
"""

import logging
import signal
import threading
from types import FrameType
from typing import Callable, Optional, Union

from revenium_middleware._core.config import Config, env_flag_enabled

logger = logging.getLogger(__name__)

# Returns False when the drain is running underneath the handler, on its
# thread: the process must end only once that drain has finished.
Drain = Callable[[], bool]
SignalHandler = Union[Callable[[int, Optional[FrameType]], object], int, signal.Handlers, None]

_deferred_signal: Optional[int] = None


def install_requested_signal_handlers(drain: Drain) -> None:
    """Install the chaining SIGTERM handler when the operator opted in."""
    flag = Config.ENV_REVENIUM_INSTALL_SIGNAL_HANDLERS
    if not env_flag_enabled(flag):
        return
    if threading.current_thread() is not threading.main_thread():
        logger.warning("%s is set, but revenium_middleware was first imported off the main thread, "
                       "where signal handlers cannot be installed; shutdown relies on atexit.", flag)
        return

    previous = signal.getsignal(signal.SIGTERM)
    if previous is signal.SIG_IGN or previous is None:
        logger.debug("SIGTERM is ignored or handled outside Python; leaving it unchanged.")
        return
    try:
        signal.signal(signal.SIGTERM, chain_sigterm(previous, drain))
    except ValueError as e:
        logger.warning("Could not install the SIGTERM handler requested by %s: %s. "
                       "Shutdown relies on atexit.", flag, e)
        return
    logger.debug("SIGTERM handler installed; it chains to %r.", previous)


def chain_sigterm(previous: SignalHandler, drain: Drain) -> Callable[[int, Optional[FrameType]], None]:
    """Build a handler that defers to ``previous``, draining first only on a default termination."""

    def handle_sigterm(signum: int, frame: Optional[FrameType]) -> None:
        global _deferred_signal
        if callable(previous):
            previous(signum, frame)
            return
        if drain():
            _terminate(signum)
        else:
            _deferred_signal = signum

    return handle_sigterm


def terminate_if_deferred() -> None:
    """End the process with the signal a handler deferred until the drain it interrupted finished, if any."""
    if _deferred_signal is not None:
        _terminate(_deferred_signal)


def _terminate(signum: int) -> None:
    signal.signal(signum, signal.SIG_DFL)
    signal.raise_signal(signum)
