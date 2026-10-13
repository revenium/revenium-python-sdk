"""The context a metered call started in, kept for the work that finishes it."""
import contextvars
from typing import Any, Callable, Optional, TypeVar

from revenium_middleware._core.metering import recorded_under
from revenium_middleware._metering import ReveniumMetering

T = TypeVar("T")


class CallContext:
    """Runs a call's finishing work in the context the call started in.

    A stream's record is built when the stream ends, which may be in another
    request's job block, after the call's own block has closed, or on the
    thread that runs the garbage collector. Built here, it carries the job
    and idempotency key of the block that made the call.
    """

    def __init__(self) -> None:
        self._context = contextvars.copy_context()

    def run(self, fn: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        # A fresh copy each time: one Context cannot be entered twice at once,
        # and a finalizer may run nested inside, or alongside, another.
        return self._context.copy().run(fn, *args, **kwargs)

    def run_recorded_under(self, metering_client: Optional[ReveniumMetering],
                           fn: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """``run``, with the records ``fn`` queues sent under ``metering_client``."""
        return self.run(_recorded_under, metering_client, fn, args, kwargs)


def _recorded_under(metering_client: Optional[ReveniumMetering], fn: Callable[..., T],
                    args: tuple, kwargs: dict) -> T:
    with recorded_under(metering_client):
        return fn(*args, **kwargs)
