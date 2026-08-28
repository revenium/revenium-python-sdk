"""Reasoning effort forwarding on the Anthropic completion paths (BACK-2710).

The native Anthropic middleware resolves per-call metadata once, in
``_extract_trace_fields``, and every completion payload site reads the result
from that dict -- so the resolver and the payload wiring are asserted
separately, and a payload site that forgot the key would be caught by the
sweep at the bottom.
"""
import asyncio
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from revenium_middleware import shutdown_event
from revenium_middleware.anthropic import middleware as anthropic_middleware
from revenium_middleware.anthropic.middleware import _extract_trace_fields


def _middleware_fn(obj):
    """Return the plain middleware function across wrapt versions."""
    return getattr(obj, "_self_wrapper", obj)


create_wrapper = _middleware_fn(anthropic_middleware.create_wrapper)
async_create_wrapper = _middleware_fn(anthropic_middleware.async_create_wrapper)


def run_metering_synchronously(coro_func, *args, **kwargs):
    """Run the metering coroutine to completion on a dedicated thread."""
    thread = threading.Thread(target=lambda: asyncio.run(coro_func(*args, **kwargs)))
    thread.start()
    thread.join(timeout=10)
    return MagicMock()


def message():
    return SimpleNamespace(
        id="msg_effort",
        model="claude-sonnet-4-6",
        stop_reason="end_turn",
        usage=SimpleNamespace(
            input_tokens=100,
            output_tokens=50,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
        ),
    )


def request_kwargs(usage_metadata):
    return {
        "messages": [{"role": "user", "content": "Hello"}],
        "model": "claude-sonnet-4-6",
        "usage_metadata": usage_metadata,
    }


@pytest.fixture(autouse=True)
def reset_shutdown_state():
    shutdown_event.clear()
    yield
    shutdown_event.clear()


class TestTraceFieldResolution:
    def test_effort_resolved_from_usage_metadata(self):
        assert _extract_trace_fields({"effort": "high"})["effort"] == "high"

    def test_unrecognised_level_resolved_unchanged(self):
        assert _extract_trace_fields({"effort": "hyper_9"})["effort"] == "hyper_9"

    def test_absent_effort_resolves_to_none(self):
        """The resolver reports "no level"; the payload sites drop the key."""
        assert _extract_trace_fields({})["effort"] is None

    def test_absent_effort_contributes_no_payload_key(self):
        """None must not survive to create_completion, which would send null."""
        from revenium_middleware.anthropic.middleware import _effort_payload

        assert _effort_payload(_extract_trace_fields({})) == {}

    def test_resolved_effort_contributes_the_payload_key(self):
        from revenium_middleware.anthropic.middleware import _effort_payload

        assert _effort_payload(_extract_trace_fields({"effort": "high"})) == {
            "effort": "high"
        }


@patch("revenium_middleware.anthropic.middleware._get_thread_safe_client", return_value=MagicMock())
@patch("revenium_middleware.anthropic.middleware.submit_ai_event", return_value=MagicMock(status_code=201))
@patch("revenium_middleware.anthropic.middleware._safe_run_async_in_thread", side_effect=run_metering_synchronously)
class TestSyncCreate:
    def test_effort_is_forwarded_on_the_payload(self, mock_thread, mock_submit, mock_client):
        create_wrapper(MagicMock(return_value=message()), None, (), request_kwargs({"effort": "xhigh"}))

        assert mock_submit.call_args[0][1]["effort"] == "xhigh"

    def test_unset_effort_is_omitted_from_the_payload(self, mock_thread, mock_submit, mock_client):
        """Omitted, not null: an explicit None would be serialized on the wire."""
        create_wrapper(MagicMock(return_value=message()), None, (), request_kwargs({}))

        assert "effort" not in mock_submit.call_args[0][1]


@patch("revenium_middleware.anthropic.middleware._get_thread_safe_client", return_value=MagicMock())
@patch("revenium_middleware.anthropic.middleware.submit_ai_event", return_value=MagicMock(status_code=201))
@patch("revenium_middleware.anthropic.middleware._safe_run_async_in_thread", side_effect=run_metering_synchronously)
class TestAsyncCreate:
    @staticmethod
    def _call(usage_metadata):
        async def fake_create(**kwargs):
            return message()

        async def scenario():
            return await async_create_wrapper(
                fake_create, None, (), request_kwargs(usage_metadata)
            )

        return asyncio.run(scenario())

    def test_effort_is_forwarded_on_the_payload(self, mock_thread, mock_submit, mock_client):
        self._call({"effort": "medium"})

        assert mock_submit.call_args[0][1]["effort"] == "medium"


def test_every_completion_payload_site_carries_effort():
    """Guard for the five completion payload sites in this module.

    The payloads are inline dict literals, so a new one is easy to add
    without the field. Each site reads ``trace_fields``, so the count of
    ticket_id reads (the nearest per-call metadata field) must match the
    count of effort reads.
    """
    import inspect

    source = inspect.getsource(anthropic_middleware)
    assert source.count("**_effort_payload(trace_fields),") == source.count(
        "trace_fields.get('ticket_id')"
    )


def test_the_effort_name_is_a_real_metering_parameter():
    """A typo here would ship a key the API silently ignores."""
    from revenium_middleware._metering._utils import maybe_transform
    from revenium_middleware._metering.types import ai_create_completion_params

    assert maybe_transform(
        {"effort": "high"}, ai_create_completion_params.AICreateCompletionParams
    ) == {"effort": "high"}


class TestBedrockStreamAdapter:
    """The Bedrock stream wrapper builds its own completion payload."""

    @staticmethod
    def _payload(usage_metadata):
        from types import SimpleNamespace as NS

        from revenium_middleware.anthropic.bedrock_adapter import BedrockStreamWrapper

        wrapper = BedrockStreamWrapper(
            model="claude-3-5-sonnet-20241022",
            payload={},
            region="us-east-1",
            usage_metadata=usage_metadata,
        )
        wrapper.response_id = "bedrock-stream-effort"
        wrapper.response_time = "2026-08-23T12:00:01Z"
        wrapper.final_message = NS(
            model="claude-3-5-sonnet-20241022",
            usage=NS(input_tokens=100, output_tokens=50),
        )

        payloads = []
        with patch(
            "revenium_middleware.anthropic.middleware._get_thread_safe_client",
            return_value=MagicMock(),
        ), patch(
            "revenium_middleware.anthropic.middleware._safe_run_async_in_thread",
            side_effect=run_metering_synchronously,
        ), patch(
            "revenium_middleware._core.submit_ai_event",
            side_effect=lambda op, args: payloads.append(args),
        ):
            wrapper._send_metering_data(1000.0)

        assert payloads, "metering call never reached submit_ai_event"
        return payloads[0]

    def test_effort_is_forwarded_on_the_payload(self):
        assert self._payload({"effort": "high"})["effort"] == "high"

    def test_unset_effort_is_omitted_from_the_payload(self):
        assert "effort" not in self._payload({})
