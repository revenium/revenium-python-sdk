"""Send the SDK's last metering records inside an ASGI server's own shutdown.

uvicorn, and so the LiteLLM proxy, ends a process stopped by SIGTERM by
re-raising the signal right after the app's lifespan shutdown, before
``atexit`` runs, so the SDK's exit drain never runs there. LiteLLM gives a
guardrail or callback no shutdown hook of its own, so the drain runs from an
ASGI middleware around the proxy's app (``revenium_middleware.litellm.proxy.asgi``).

Deliberately free of LiteLLM imports and Python 3.8-compatible, like ``_metering_owner``.
"""

import asyncio
import logging
import threading
from typing import Any, Awaitable, Callable, Dict

from revenium_middleware._core.config import Config
from revenium_middleware._core.metering import handle_exit, shutdown_budget_seconds

logger = logging.getLogger("revenium_middleware.extension")

Scope = Dict[str, Any]
Message = Dict[str, Any]
Receive = Callable[[], Awaitable[Message]]
Send = Callable[[Message], Awaitable[None]]
ASGIApp = Callable[[Scope, Receive, Send], Awaitable[None]]

_LIFESPAN_SHUTDOWN_OVER = frozenset({"lifespan.shutdown.complete", "lifespan.shutdown.failed"})
# The drain checks its deadline between steps, so it may end a moment after it.
_DRAIN_OVERRUN_GRACE_SECONDS = 0.5


class DrainMeteringOnShutdown:
    """ASGI middleware that runs the SDK's exit drain once the wrapped app has finished its lifespan shutdown.

    The drain starts after the app's own shutdown (LiteLLM's spend-log flush
    among it) and before the server is told the shutdown is over, and takes at
    most ``REVENIUM_SHUTDOWN_TIMEOUT_SECONDS`` (5 seconds by default). It is the
    process's exit drain: it runs once, metering stops after it, and the
    ``atexit`` drain then has nothing left to do. HTTP and websocket traffic
    passes straight through.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "lifespan":
            await self.app(scope, receive, send)
            return

        async def send_after_draining(message: Message) -> None:
            if message["type"] in _LIFESPAN_SHUTDOWN_OVER:
                await _run_exit_drain()
            await send(message)

        await self.app(scope, receive, send_after_draining)


async def _run_exit_drain() -> None:
    """Run the exit drain and wait for it, without blocking the event loop, for at most its budget.

    Metering is off for the rest of the process once this returns: it is the
    exit drain, not ``flush_metering()``. The drain gets a daemon thread of its
    own rather than the loop's default executor, whose threads the host may
    have left busy. A drain still running when its budget ends, held by an
    ``on_metering_error`` subscriber for instance, is left behind on that
    thread so the server's shutdown can go on.
    """
    budget = shutdown_budget_seconds()
    loop = asyncio.get_running_loop()
    drained = loop.create_future()

    def drain() -> None:
        try:
            handle_exit()
        finally:
            _settle_from_thread(loop, drained)

    threading.Thread(target=drain, name="revenium-shutdown-drain", daemon=True).start()
    try:
        await asyncio.wait_for(drained, budget + _DRAIN_OVERRUN_GRACE_SECONDS)
    except asyncio.TimeoutError:
        logger.warning(
            "The metering exit drain was still running when its %ss shutdown budget (%s) ended; "
            "the shutdown goes on without waiting for it.",
            budget, Config.ENV_REVENIUM_SHUTDOWN_TIMEOUT_SECONDS,
        )


def _settle_from_thread(loop: asyncio.AbstractEventLoop, drained: "asyncio.Future[None]") -> None:
    try:
        loop.call_soon_threadsafe(_settle, drained)
    except RuntimeError:
        # The server closed the loop after the wait gave up on this drain.
        pass


def _settle(drained: "asyncio.Future[None]") -> None:
    if not drained.done():
        drained.set_result(None)


__all__ = ["DrainMeteringOnShutdown"]
