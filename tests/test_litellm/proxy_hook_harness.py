"""Shared harness for driving revenium_middleware.litellm.proxy.middleware hooks offline.

MiddlewareHandler is a litellm CustomLogger: LiteLLM calls its
async_log_success_event / async_log_failure_event with a (kwargs, response_obj,
start_time, end_time) tuple. These helpers let a unit test play LiteLLM's part
without a proxy, a provider, or a network call, and read back exactly what the
handler handed to the metering client.

Not a test module (no test_ prefix) so pytest imports it rather than collecting it.
"""
import asyncio
from types import SimpleNamespace


def run_inline(coro):
    """Execute the metering coroutine synchronously so asserts see the call."""
    asyncio.run(coro)
    return SimpleNamespace(name="inline-metering")


def run_hook(coro):
    """Drive MiddlewareHandler's async_log_*_event coroutine to completion.

    Those methods have no internal `await` points -- they build the payload
    synchronously and hand a nested coroutine to (mocked) run_async_in_thread
    without awaiting it. So a plain `send(None)` runs the whole body in one
    step, without asyncio.run()/get_event_loop() marking a loop as "running".
    That matters here because run_inline (above) calls asyncio.run() itself to
    execute the metering coroutine; nesting two real asyncio.run() calls would
    raise "cannot be called from a running event loop".
    """
    try:
        coro.send(None)
    except StopIteration:
        pass


class SubscriptableResponse:
    """Stand-in for LiteLLM's ModelResponse: `response_obj["usage"]` plus `.id`."""

    def __init__(self, response_id, usage):
        self.id = response_id
        self._usage = usage

    def __getitem__(self, key):
        if key == "usage":
            return self._usage
        raise KeyError(key)


def make_success_response(usage, response_id="txn-proxy-cache-mapping"):
    return SubscriptableResponse(response_id, usage)


def base_kwargs(model="gpt-4o-mini"):
    return {
        "model": model,
        "litellm_params": {"metadata": {"headers": {}}},
    }


def submitted_args(mock_submit):
    """The metering payload from the single expected submission.

    Asserting the call count here is deliberate: every caller then proves
    "exactly one submission", so a hook that silently submitted nothing (the
    failure mode BACK-2405 is about) cannot pass by producing no payload to
    inspect.
    """
    assert mock_submit.call_count == 1
    return mock_submit.call_args[0][1]
