"""Visibility for AI calls that selective metering leaves unmetered."""

import logging
import os
import threading

from .config import is_selective_metering_enabled
from .context import is_inside_decorated_function

logger = logging.getLogger("revenium_middleware")

_unmetered_call_reported = False
_report_lock = threading.Lock()


def reported_if_unmetered(skipped: bool) -> bool:
    """Return ``skipped``, logging once per process the first time it is True."""
    global _unmetered_call_reported
    if skipped and not _unmetered_call_reported:
        with _report_lock:
            if _unmetered_call_reported:
                return skipped
            _unmetered_call_reported = True
        logger.info(
            "Selective metering is on, so an AI call made outside a @revenium_meter function was not metered. "
            "A call on a worker thread is outside the function unless the thread's target is wrapped with "
            "revenium_middleware.carry_context. This message is logged once."
        )
    return skipped


def selectively_unmetered() -> bool:
    """Whether selective metering leaves the current call unmetered, logging the first time it does."""
    return reported_if_unmetered(is_selective_metering_enabled() and not is_inside_decorated_function())


def _reset_in_forked_child() -> None:
    global _unmetered_call_reported, _report_lock
    _unmetered_call_reported = False
    _report_lock = threading.Lock()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reset_in_forked_child)
