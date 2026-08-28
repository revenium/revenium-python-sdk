"""An unset reasoning effort must never reach the wire as ``null`` (BACK-2710).

The omission contract lives one layer below the payload dicts these tests
drive. ``create_completion`` drops ``NotGiven`` values during serialization
(``_transform_typeddict`` in ``revenium_middleware/_metering/_utils/_transform.py``
skips on ``is_given``), but an explicit ``None`` is "given" and survives all the
way into the JSON body. A payload that reads ``.get("effort")`` off the sparse
resolver result therefore looks empty in the payload dict while putting
``"effort": null`` on the request.

So these tests assert on the serialized httpx request body rather than on the
payload dict: the payload dict is exactly where the difference is invisible.
"""

import asyncio
import json
import threading
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import httpx
import pytest

from revenium_middleware import shutdown_event
from revenium_middleware._metering import AsyncReveniumMetering, ReveniumMetering
from revenium_middleware.anthropic import middleware as anthropic_middleware


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
    """Yield a recorder plus a real metering client wired to a stub transport.

    A real client is the point: mocking ``_post`` or ``submit_ai_event`` hides
    the serialization step where ``None`` becomes ``null``.
    """
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
def installed_wire_client():
    """As ``wire_client``, but also installed as the middleware's metering client."""
    with wire_client() as (recorder, client):
        with patch("revenium_middleware._core.metering.client", client), patch(
            "revenium_middleware.client", client
        ), patch.object(
            anthropic_middleware, "_get_thread_safe_client", return_value=client
        ):
            yield recorder


def run_metering_synchronously(coro_func, *args, **kwargs):
    """Run the metering coroutine to completion on a dedicated thread."""
    thread = threading.Thread(target=lambda: asyncio.run(coro_func(*args, **kwargs)))
    thread.start()
    thread.join(timeout=10)
    return MagicMock()


def run_inline(coro):
    """Drive a coroutine handed to ``run_async_in_thread`` to completion here."""
    thread = threading.Thread(target=lambda: asyncio.run(coro))
    thread.start()
    thread.join(timeout=10)


@pytest.fixture(autouse=True)
def reset_shutdown_state():
    shutdown_event.clear()
    yield
    shutdown_event.clear()


def anthropic_message():
    return SimpleNamespace(
        id="msg_wire_effort",
        model="claude-sonnet-4-6",
        stop_reason="end_turn",
        usage=SimpleNamespace(
            input_tokens=100,
            output_tokens=50,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
        ),
    )


def anthropic_request_kwargs(usage_metadata):
    return {
        "messages": [{"role": "user", "content": "Hello"}],
        "model": "claude-sonnet-4-6",
        "usage_metadata": usage_metadata,
    }


def anthropic_create_wrapper():
    """The plain middleware function, across wrapt versions."""
    wrapper = anthropic_middleware.create_wrapper
    return getattr(wrapper, "_self_wrapper", wrapper)


class TestTypedClientContract:
    """The layer every provider path funnels through."""

    _KWARGS = {
        "completion_start_time": "2026-08-23T00:00:00Z",
        "cost_type": "AI",
        "input_token_count": 10,
        "is_streamed": False,
        "model": "gpt-test",
        "output_token_count": 5,
        "provider": "OPENAI",
        "request_duration": 100,
        "request_time": "2026-08-23T00:00:00Z",
        "response_time": "2026-08-23T00:00:01Z",
        "stop_reason": "END",
        "total_token_count": 15,
        "transaction_id": "txn-wire-effort",
    }

    def test_unpassed_effort_is_absent_from_the_request_body(self):
        with wire_client() as (recorder, client):
            client.ai.create_completion(**self._KWARGS)

        assert "effort" not in recorder.body

    def test_an_explicit_none_is_serialized_as_null(self):
        """The trap every provider path has to avoid.

        The generated client only drops NotGiven, so ``effort=None`` is a
        request to send null -- which is why the middleware paths spread the
        sparse resolver result instead of reading a key off it.
        """
        with wire_client() as (recorder, client):
            client.ai.create_completion(effort=None, **self._KWARGS)

        assert recorder.body["effort"] is None

    def test_effort_reaches_the_wire_verbatim(self):
        with wire_client() as (recorder, client):
            client.ai.create_completion(effort="xhigh", **self._KWARGS)

        assert recorder.body["effort"] == "xhigh"


class TestAsyncTypedClientContract:
    """AsyncReveniumMetering maintains its own create_completion forwarding.

    A sync-only wire test lets the async client silently diverge (PR #86 hit
    exactly this with the flat attribution fields), so the same three
    contract cases run against the async client over a real transport.
    """

    _KWARGS = TestTypedClientContract._KWARGS

    @contextmanager
    def _async_wire_client(self):
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

    def _create(self, **effort_kwargs):
        with self._async_wire_client() as (recorder, client):
            run_inline(client.ai.create_completion(**effort_kwargs, **self._KWARGS))
        return recorder.body

    def test_unpassed_effort_is_absent_from_the_request_body(self):
        assert "effort" not in self._create()

    def test_an_explicit_none_is_serialized_as_null(self):
        assert self._create(effort=None)["effort"] is None

    def test_effort_reaches_the_wire_verbatim(self):
        assert self._create(effort="xhigh")["effort"] == "xhigh"


class TestAnthropicNativePath:
    def test_absent_effort_is_omitted_from_the_wire(self):
        with installed_wire_client() as recorder, patch.object(
            anthropic_middleware,
            "_safe_run_async_in_thread",
            side_effect=run_metering_synchronously,
        ):
            anthropic_create_wrapper()(
                MagicMock(return_value=anthropic_message()),
                None,
                (),
                anthropic_request_kwargs({}),
            )

        assert "effort" not in recorder.body

    def test_supplied_effort_reaches_the_wire(self):
        with installed_wire_client() as recorder, patch.object(
            anthropic_middleware,
            "_safe_run_async_in_thread",
            side_effect=run_metering_synchronously,
        ):
            anthropic_create_wrapper()(
                MagicMock(return_value=anthropic_message()),
                None,
                (),
                anthropic_request_kwargs({"effort": "high"}),
            )

        assert recorder.body["effort"] == "high"


class TestBedrockTransportPath:
    @staticmethod
    def _emit(usage_metadata):
        import datetime

        import revenium_middleware
        from revenium_middleware.anthropic import bedrock_transport as bt

        token = anthropic_middleware.usage_context.set(usage_metadata)
        try:
            with installed_wire_client() as recorder, patch.object(
                revenium_middleware, "run_async_in_thread", side_effect=run_inline
            ):
                bt._emit_completion(
                    model_id="anthropic.claude-3-5-sonnet-20241022-v2:0",
                    input_tokens=18,
                    output_tokens=5,
                    cache_read_tokens=0,
                    cache_creation_tokens=0,
                    stop_reason="end_turn",
                    is_streamed=False,
                    region="us-east-1",
                    transaction_id="bedrock-wire-effort",
                    request_time_dt=datetime.datetime.now(datetime.timezone.utc),
                )
        finally:
            anthropic_middleware.usage_context.reset(token)
        return recorder.body

    def test_absent_effort_is_omitted_from_the_wire(self):
        assert "effort" not in self._emit({})

    def test_supplied_effort_reaches_the_wire(self):
        assert self._emit({"effort": "xhigh"})["effort"] == "xhigh"


class TestBedrockStreamAdapterPath:
    @staticmethod
    def _emit(usage_metadata):
        from revenium_middleware.anthropic.bedrock_adapter import BedrockStreamWrapper

        wrapper = BedrockStreamWrapper(
            model="claude-3-5-sonnet-20241022",
            payload={},
            region="us-east-1",
            usage_metadata=usage_metadata,
        )
        wrapper.response_id = "bedrock-stream-wire-effort"
        wrapper.response_time = "2026-08-23T12:00:01Z"
        wrapper.final_message = SimpleNamespace(
            model="claude-3-5-sonnet-20241022",
            usage=SimpleNamespace(input_tokens=100, output_tokens=50),
        )

        with installed_wire_client() as recorder, patch.object(
            anthropic_middleware,
            "_safe_run_async_in_thread",
            side_effect=run_metering_synchronously,
        ):
            wrapper._send_metering_data(1000.0)
        return recorder.body

    def test_absent_effort_is_omitted_from_the_wire(self):
        assert "effort" not in self._emit({})

    def test_supplied_effort_reaches_the_wire(self):
        assert self._emit({"effort": "medium"})["effort"] == "medium"


class TestAgenticOutcomePath:
    @staticmethod
    def _emit(extra_payload):
        from revenium_middleware.agentic_outcomes import (
            AgenticOutcomeClient,
            AgenticOutcomeSettings,
        )

        payload = {
            "completionStartTime": "2026-08-23T00:00:00Z",
            "costType": "AI",
            "inputTokenCount": 10,
            "isStreamed": False,
            "model": "gpt-5",
            "outputTokenCount": 5,
            "provider": "OPENAI",
            "requestDuration": 100,
            "requestTime": "2026-08-23T00:00:00Z",
            "responseTime": "2026-08-23T00:00:01Z",
            "stopReason": "END",
            "totalTokenCount": 15,
            "transactionId": "agentic-wire-effort",
            **extra_payload,
        }
        with wire_client() as (recorder, metering_client):
            client = AgenticOutcomeClient(
                AgenticOutcomeSettings(api_key="wire-test-key"),
                metering_client=metering_client,
            )
            client.emit_completion(payload)
        return recorder.body

    def test_absent_effort_is_omitted_from_the_wire(self):
        assert "effort" not in self._emit({})

    def test_explicit_none_effort_is_omitted_from_the_wire(self):
        assert "effort" not in self._emit({"effort": None})

    def test_supplied_effort_reaches_the_wire(self):
        assert self._emit({"effort": "hyper_9"})["effort"] == "hyper_9"
