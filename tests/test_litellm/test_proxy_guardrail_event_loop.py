"""The guardrail's pre-call hook never stalls the proxy's event loop (BACK-3917).

LiteLLM awaits ``async_pre_call_hook`` directly on the proxy's one event loop,
and the hook calls the synchronous ``check_enforcement``. When that check
refreshed its rules inline, a hanging Revenium froze the whole proxy for about
58 s per request and a refused or 503 endpoint for about 7.8 s; a 403 cost one
round trip of frozen loop on every request. In a live run at 30 req/s the proxy
served 0 of 5,400 requests.

These tests drive the real ``ReveniumGuardrail`` hook at the ticket's 30 req/s
against a real local rules endpoint in each failure mode, with a ticker on the
same loop measuring how late it wakes. Nothing on the enforcement path is
mocked.
"""
import asyncio

import pytest

pytest.importorskip("litellm")
pytest.importorskip("fastapi")

from fastapi import HTTPException  # noqa: E402

from revenium_middleware._core import enforcement  # noqa: E402
from revenium_middleware.litellm.proxy.guardrail import ReveniumGuardrail  # noqa: E402

from test_core.enforcement_stub_server import (  # noqa: E402
    FAILURE_MODES,
    FORBIDDEN,
    HEALTHY,
    EnforcementStub,
    drive_on_loop,
    point_enforcement_at,
    shut_down,
    wait_until,
)
from .proxy_hook_harness import guardrail_data, make_key_dict  # noqa: E402

RATE = 30
REQUESTS = 45
# Longest the loop may be held at once. A pre-call check is microseconds of
# dict lookups; the smallest stall this guards against is one 0.3 s round trip
# to the stub, and 0.1 s leaves room for a GC pause on a loaded CI runner.
MAX_LOOP_LAG_SECONDS = 0.1
# The ticket's acceptance bound on p99 against the healthy control.
P99_SLACK_SECONDS = 0.5


@pytest.fixture
def guardrail():
    from revenium_middleware.litellm.proxy import _metering_owner

    _metering_owner.reset_metering_owner()
    instance = ReveniumGuardrail(guardrail_name="revenium")
    yield instance
    _metering_owner.reset_metering_owner()


@pytest.fixture
def endpoint(monkeypatch):
    stubs = []

    def start(mode, rules=None):
        stub = EnforcementStub(mode, rules)
        stubs.append(stub)
        point_enforcement_at(monkeypatch, stub)
        return stub

    yield start
    for stub in stubs:
        shut_down(stub)


def _hook_call(guardrail):
    async def call():
        return await guardrail.async_pre_call_hook(
            user_api_key_dict=make_key_dict(user_email="dev@example.com"),
            cache=None,
            data=guardrail_data(),
            call_type="completion",
        )

    return call


class TestTheLoopKeepsServing:

    @pytest.mark.asyncio
    @pytest.mark.parametrize("mode", FAILURE_MODES)
    async def test_the_hook_never_holds_the_loop_while_revenium_is_down(
        self, guardrail, endpoint, mode
    ):
        endpoint(mode)
        served = []

        async def call():
            served.append(await _hook_call(guardrail)())

        report = await drive_on_loop(call, REQUESTS, RATE)

        assert report.max_lag < MAX_LOOP_LAG_SECONDS, (
            f"the event loop was held {report.max_lag:.3f}s with the rules endpoint {mode}"
        )
        assert len(served) == REQUESTS, "a request was refused or lost while failing open"
        assert report.p99() < P99_SLACK_SECONDS

    @pytest.mark.asyncio
    async def test_a_failing_endpoint_sees_one_request_not_one_per_customer_request(
        self, guardrail, endpoint
    ):
        stub = endpoint(FORBIDDEN)

        await drive_on_loop(_hook_call(guardrail), REQUESTS, RATE)

        assert stub.hits == 1, f"{stub.hits} rule fetches for {REQUESTS} customer requests"


class TestEnforcementStillBlocks:

    @pytest.mark.asyncio
    async def test_a_tripped_rule_fetched_in_the_background_is_a_429(self, guardrail, endpoint):
        endpoint(HEALTHY, rules=[{"ruleId": 7, "name": "Team Budget", "breached": True,
                                  "threshold": 10.0, "currentValue": 11.5}])

        assert await _hook_call(guardrail)() is not None  # cold start: nothing cached yet
        loaded = await asyncio.to_thread(wait_until, lambda: enforcement._cache_initialized)
        assert loaded

        with pytest.raises(HTTPException) as blocked:
            await _hook_call(guardrail)()
        assert blocked.value.status_code == 429
