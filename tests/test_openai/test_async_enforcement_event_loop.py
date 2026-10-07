"""Async OpenAI calls never stall the caller's event loop on enforcement (BACK-3917).

The async wrappers (``async_create_wrapper``, ``async_responses_create_wrapper``,
the async parse wrappers and ``async_embeddings_create_wrapper``) run the
synchronous ``check_enforcement`` when the call is made, on the caller's event
loop, before they hand back the coroutine. The LiteLLM sweep flagged them as the
same defect as the proxy guardrail; driving ``AsyncOpenAI`` against a hanging
rules endpoint confirms it: each call froze the loop for the whole inline
refresh. The fix is the same one -- the check never refreshes on its caller.
"""
from unittest.mock import patch

import httpx
import pytest

from revenium_middleware.openai import middleware as mw

from test_core.enforcement_stub_server import (
    FAILURE_MODES,
    EnforcementStub,
    drive_on_loop,
    point_enforcement_at,
    shut_down,
)

from .test_openai_sdk_entry_points import CHAT_REQUEST, RESPONSES_REQUEST, FakeOpenAI

MAX_LOOP_LAG_SECONDS = 0.1


class FakeOpenAIWithEmbeddings(FakeOpenAI):
    def __call__(self, request):
        if request.url.path.endswith("/embeddings"):
            return httpx.Response(200, json={
                "object": "list", "model": "text-embedding-3-small",
                "data": [{"object": "embedding", "index": 0, "embedding": [0.1]}],
                "usage": {"prompt_tokens": 1, "total_tokens": 1},
            })
        return super().__call__(request)


@pytest.fixture
def endpoint(monkeypatch):
    stubs = []

    def start(mode):
        stub = EnforcementStub(mode)
        stubs.append(stub)
        point_enforcement_at(monkeypatch, stub)
        return stub

    yield start
    for stub in stubs:
        shut_down(stub)


@pytest.fixture(autouse=True)
def no_metering_dispatch():
    with patch.object(mw, "run_async_in_thread", side_effect=lambda coro: coro.close()):
        yield


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", FAILURE_MODES)
@pytest.mark.parametrize("surface", ["chat", "responses", "embeddings"])
async def test_async_calls_never_hold_the_loop_while_revenium_is_down(endpoint, mode, surface):
    endpoint(mode)
    client = FakeOpenAIWithEmbeddings().async_client()
    calls = {
        "chat": lambda: client.chat.completions.create(**CHAT_REQUEST),
        "responses": lambda: client.responses.create(**RESPONSES_REQUEST),
        "embeddings": lambda: client.embeddings.create(model="text-embedding-3-small", input="hi"),
    }

    errors = []

    async def call():
        try:
            await calls[surface]()
        except Exception as error:  # collected so the assertion below names it
            errors.append(error)

    report = await drive_on_loop(call, requests=15, rate=30)

    assert errors == []
    assert report.max_lag < MAX_LOOP_LAG_SECONDS, (
        f"async {surface} held the loop {report.max_lag:.3f}s with the rules endpoint {mode}"
    )
