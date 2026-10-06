"""Opt-in SIGTERM hook that drains metering before a default termination.

The SDK leaves the host's signal handling alone unless
``REVENIUM_INSTALL_SIGNAL_HANDLERS`` is set. Shutdown otherwise relies on
``atexit``, which runs whenever the interpreter exits normally, including
after ``KeyboardInterrupt`` (Python's default SIGINT behaviour) and after a
host's own SIGTERM handler exits through ``sys.exit``.

The exit ``atexit`` cannot see is SIGTERM with the default disposition: the
kernel ends the process immediately. With the flag set, that case drains
first and then terminates through the same default disposition. A handler
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

Drain = Callable[[], None]
SignalHandler = Union[Callable[[int, Optional[FrameType]], object], int, signal.Handlers, None]


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
        if callable(previous):
            previous(signum, frame)
            return
        drain()
        signal.signal(signum, signal.SIG_DFL)
        signal.raise_signal(signum)

    return handle_sigterm
