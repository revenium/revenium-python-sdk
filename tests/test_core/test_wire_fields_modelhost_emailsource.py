"""modelHost and subscriberEmailSource must reach the wire under their
camelCase aliases, and must stay off it when unset (BACK-2732).

Both fields are optional passthroughs the metering API accepts (hypercurrent
BACK-2669). Two things can silently break and neither is visible in the
payload dict callers build:

* the camelCase alias -- ``PropertyInfo(alias=...)`` in
  ``ai_create_completion_params`` is what turns ``model_host`` into
  ``modelHost``; a missing annotation ships the snake_case key instead.
* the NotGiven-vs-None distinction -- ``_transform_typeddict`` in
  ``revenium_middleware/_metering/_utils/_transform.py`` drops values that
  fail ``is_given``, so an unpassed field disappears while an explicit
  ``None`` is "given" and serializes as ``null``.

So these tests assert on the serialized httpx request body, over a real
client and a stub transport: mocking ``_post`` would hide exactly the
serialization step under test. Both the sync and the async resource maintain
their own ``create_completion`` forwarding, so both are exercised -- PR #86
hit a real divergence there with the flat attribution fields.
"""

import asyncio
import json
import threading
from contextlib import contextmanager
from unittest.mock import patch

import httpx
import pytest

from revenium_middleware._metering import AsyncReveniumMetering, ReveniumMetering

# (python keyword, wire alias, a representative value)
FIELDS = [
    ("model_host", "modelHost", "bedrock"),
    ("subscriber_email_source", "subscriberEmailSource", "cli-flag"),
]

REQUIRED_KWARGS = {
    "completion_start_time": "2026-08-24T00:00:00Z",
    "cost_type": "AI",
    "input_token_count": 10,
    "is_streamed": False,
    "model": "gpt-test",
    "output_token_count": 5,
    "provider": "OPENAI",
    "request_duration": 100,
    "request_time": "2026-08-24T00:00:00Z",
    "response_time": "2026-08-24T00:00:01Z",
    "stop_reason": "END",
    "total_token_count": 15,
    "transaction_id": "txn-wire-back-2732",
}


class WireRecorder:
    """Collects the JSON bodies of the metering requests actually sent."""

    def __init__(self):
        self.bodies = []

    @property
    def body(self):
        completions = [b for b in self.bodies if "transactionId" in b]
        assert completions, "no completion request reached the wire"
        return completions[-1]


@contextmanager
def wire_client():
    """Yield a recorder plus a real sync client wired to a stub transport."""
    recorder = WireRecorder()
    client = ReveniumMetering(
        api_key="wire-test-key",
        base_url="http://metering.invalid/meter/",
        max_retries=0,
    )

    def send(request, **kwargs):
        recorder.bodies.append(json.loads(request.read() or b"{}"))
        return httpx.Response(200, json={"status": "ok"}, request=request)

    with patch.object(client._client, "send", side_effect=send):
        yield recorder, client


@contextmanager
def async_wire_client():
    """Yield a recorder plus a real async client wired to a stub transport."""
    recorder = WireRecorder()
    client = AsyncReveniumMetering(
        api_key="wire-test-key",
        base_url="http://metering.invalid/meter/",
        max_retries=0,
    )

    async def send(request, **kwargs):
        recorder.bodies.append(json.loads(request.read() or b"{}"))
        return httpx.Response(200, json={"status": "ok"}, request=request)

    with patch.object(client._client, "send", side_effect=send):
        yield recorder, client


def run_inline(coro):
    """Drive a coroutine to completion on a dedicated thread."""
    thread = threading.Thread(target=lambda: asyncio.run(coro))
    thread.start()
    thread.join(timeout=10)


@pytest.mark.parametrize("field, alias, value", FIELDS)
class TestSyncTypedClientContract:
    """ReveniumMetering.ai.create_completion."""

    @staticmethod
    def _body(**field_kwargs):
        with wire_client() as (recorder, client):
            client.ai.create_completion(**field_kwargs, **REQUIRED_KWARGS)
        return recorder.body

    def test_supplied_value_reaches_the_wire_under_the_alias(self, field, alias, value):
        assert self._body(**{field: value})[alias] == value

    def test_an_unpassed_field_is_absent_from_the_request_body(self, field, alias, value):
        body = self._body()

        assert alias not in body
        assert field not in body

    def test_an_explicit_none_is_serialized_as_null(self, field, alias, value):
        """The NotGiven-vs-None trap: the client drops only NotGiven."""
        body = self._body(**{field: None})

        assert alias in body
        assert body[alias] is None


@pytest.mark.parametrize("field, alias, value", FIELDS)
class TestAsyncTypedClientContract:
    """AsyncReveniumMetering keeps its own create_completion forwarding."""

    @staticmethod
    def _body(**field_kwargs):
        with async_wire_client() as (recorder, client):
            run_inline(client.ai.create_completion(**field_kwargs, **REQUIRED_KWARGS))
        return recorder.body

    def test_supplied_value_reaches_the_wire_under_the_alias(self, field, alias, value):
        assert self._body(**{field: value})[alias] == value

    def test_an_unpassed_field_is_absent_from_the_request_body(self, field, alias, value):
        body = self._body()

        assert alias not in body
        assert field not in body

    def test_an_explicit_none_is_serialized_as_null(self, field, alias, value):
        """The NotGiven-vs-None trap: the client drops only NotGiven."""
        body = self._body(**{field: None})

        assert alias in body
        assert body[alias] is None
