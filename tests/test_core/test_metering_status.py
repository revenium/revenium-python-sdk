"""Tests for metering error visibility (BACK-778).

Covers the metering status/callback surface (`_core.metering_status`), its
wiring into `submit_ai_event` and `MeteringThread`, and the ERROR-level log
for a missing API key.
"""

import asyncio
import logging
import time

import httpx
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from revenium_middleware._core.metering import _build_metering_client, run_async_in_thread
from revenium_middleware._core.metering_status import (
    MeteringErrorEvent,
    get_metering_status,
    on_metering_error,
    record_metering_error,
    record_metering_success,
    remove_metering_error_callback,
    reset_metering_status,
)
from revenium_middleware._core.metering_submission import submit_ai_event
from revenium_middleware._metering._exceptions import APIStatusError


@pytest.fixture(autouse=True)
def clean_status():
    """Isolate metering status state between tests."""
    reset_metering_status()
    yield
    reset_metering_status()


def _api_status_error(status_code: int) -> APIStatusError:
    request = httpx.Request("POST", "https://api.revenium.ai/meter/v2/ai/completions")
    response = httpx.Response(status_code, request=request)
    return APIStatusError(f"HTTP {status_code}", response=response, body=None)


@pytest.fixture
def mock_client():
    mock = MagicMock()
    mock.ai.create_completion = MagicMock(return_value="completion-result")
    with patch("revenium_middleware._core.metering_submission.get_client", lambda: mock):
        yield mock


# ---------------------------------------------------------------------------
# Status surface
# ---------------------------------------------------------------------------

def test_initial_status_is_clean():
    status = get_metering_status()
    assert status.error_count == 0
    assert status.success_count == 0
    assert status.last_error is None
    assert status.last_error_at is None


def test_record_error_updates_status():
    exc = _api_status_error(401)
    record_metering_error(exc, operation="completion")
    status = get_metering_status()
    assert status.error_count == 1
    assert status.last_error is exc
    assert status.last_error_at is not None


def test_record_success_increments_success_count():
    record_metering_success()
    record_metering_success()
    status = get_metering_status()
    assert status.success_count == 2
    assert status.error_count == 0


def test_reset_clears_counts():
    record_metering_error(ValueError("boom"))
    record_metering_success()
    reset_metering_status()
    status = get_metering_status()
    assert status.error_count == 0
    assert status.success_count == 0
    assert status.last_error is None


# ---------------------------------------------------------------------------
# Callback surface
# ---------------------------------------------------------------------------

def test_on_metering_error_callback_receives_event():
    events = []
    on_metering_error(events.append)
    exc = _api_status_error(500)
    record_metering_error(exc, operation="completion")
    assert len(events) == 1
    event = events[0]
    assert isinstance(event, MeteringErrorEvent)
    assert event.error is exc
    assert event.operation == "completion"


def test_callback_exception_does_not_propagate_and_others_still_run():
    events = []

    def bad_callback(event):
        raise RuntimeError("customer callback bug")

    on_metering_error(bad_callback)
    on_metering_error(events.append)
    record_metering_error(ValueError("boom"))
    assert len(events) == 1
    assert get_metering_status().error_count == 1


def test_remove_callback_stops_delivery():
    events = []
    on_metering_error(events.append)
    remove_metering_error_callback(events.append)
    record_metering_error(ValueError("boom"))
    assert events == []


# ---------------------------------------------------------------------------
# submit_ai_event wiring
# ---------------------------------------------------------------------------

def test_submit_success_records_success(mock_client):
    submit_ai_event("completion", {"x": 1})
    status = get_metering_status()
    assert status.success_count == 1
    assert status.error_count == 0


def test_submit_permanent_failure_records_error_logs_error_and_reraises(mock_client, caplog):
    exc = _api_status_error(401)
    mock_client.ai.create_completion.side_effect = exc
    with caplog.at_level(logging.ERROR, logger="revenium_middleware"):
        with pytest.raises(APIStatusError):
            submit_ai_event("completion", {"x": 1})
    status = get_metering_status()
    assert status.error_count == 1
    assert status.last_error is exc
    error_records = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert any("metering" in r.getMessage().lower() for r in error_records)


def test_submit_retryable_failure_buffers_records_error_and_logs_error(mock_client, caplog):
    exc = _api_status_error(500)
    mock_client.ai.create_completion.side_effect = exc
    fake_buffer = MagicMock()
    with patch(
        "revenium_middleware._core.metering_buffer.get_buffer", return_value=fake_buffer
    ):
        with caplog.at_level(logging.ERROR, logger="revenium_middleware"):
            result = submit_ai_event("completion", {"x": 1})
    assert result is None
    fake_buffer.push.assert_called_once()
    status = get_metering_status()
    assert status.error_count == 1
    error_records = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert any("metering" in r.getMessage().lower() for r in error_records)


def test_submit_failure_invokes_subscribed_callback(mock_client):
    events = []
    on_metering_error(events.append)
    exc = _api_status_error(401)
    mock_client.ai.create_completion.side_effect = exc
    with pytest.raises(APIStatusError):
        submit_ai_event("completion", {"x": 1})
    assert len(events) == 1
    assert events[0].error is exc
    assert events[0].operation == "completion"


# ---------------------------------------------------------------------------
# Tool event wiring
# ---------------------------------------------------------------------------

def test_tool_event_success_records_success():
    from revenium_middleware._metering.context import ReveniumContext
    from revenium_middleware._metering.decorator import _send_tool_event_async

    response = MagicMock()
    response.raise_for_status = MagicMock()
    client = MagicMock()
    client.post = AsyncMock(return_value=response)
    async_client = MagicMock()
    async_client.__aenter__ = AsyncMock(return_value=client)
    async_client.__aexit__ = AsyncMock(return_value=False)

    with patch(
        "revenium_middleware._metering.decorator.httpx.AsyncClient",
        return_value=async_client,
    ):
        asyncio.run(
            _send_tool_event_async(
                url="https://api.revenium.ai/meter/v2/tool/events",
                key="hak_test_key",
                tool_id="test-tool",
                operation="execute",
                duration_ms=42,
                success=True,
                error_message=None,
                usage_metadata=None,
                context=ReveniumContext(),
            )
        )

    status = get_metering_status()
    assert status.success_count == 1
    assert status.error_count == 0


# ---------------------------------------------------------------------------
# Missing API key at initialization
# ---------------------------------------------------------------------------

def test_missing_api_key_logs_error(caplog):
    with caplog.at_level(logging.ERROR, logger="revenium_middleware"):
        assert _build_metering_client(None, None) is None
    error_records = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert any("REVENIUM_METERING_API_KEY" in r.getMessage() for r in error_records)


# ---------------------------------------------------------------------------
# Sensitive header redaction
# ---------------------------------------------------------------------------

def test_record_error_redacts_api_key_header():
    request = httpx.Request(
        "POST",
        "https://api.revenium.ai/meter/v2/ai/completions",
        headers={"x-api-key": "hak_secret123"},
    )
    response = httpx.Response(401, request=request)
    exc = APIStatusError("HTTP 401", response=response, body=None)
    events = []
    on_metering_error(events.append)
    record_metering_error(exc, operation="completion")
    assert len(events) == 1
    assert events[0].error.request.headers["x-api-key"] == "[REDACTED]"
    assert (
        get_metering_status().last_error.request.headers["x-api-key"] == "[REDACTED]"
    )


def test_redaction_survives_response_without_attached_request():
    """httpx.Response.request raises RuntimeError when unset; that must not
    defeat redaction of the exception's own request/response objects."""
    class TransportError(Exception):
        pass

    exc = TransportError("boom")
    exc.request = httpx.Request(
        "POST",
        "https://api.revenium.ai/meter/v2/ai/completions",
        headers={"x-api-key": "hak_secret123"},
    )
    exc.response = httpx.Response(500)  # no request attached: .request raises
    record_metering_error(exc)
    assert exc.request.headers["x-api-key"] == "[REDACTED]"


def test_redaction_never_raises_on_odd_exceptions():
    class ExplodingError(Exception):
        @property
        def request(self):
            raise RuntimeError("no request for you")

    record_metering_error(ValueError("no request attr"))
    record_metering_error(ExplodingError("boom"))
    assert get_metering_status().error_count == 2


# ---------------------------------------------------------------------------
# Background metering thread
# ---------------------------------------------------------------------------

def test_metering_thread_failure_records_error_and_logs_error(caplog):
    async def failing_metering_call():
        raise RuntimeError("simulated metering failure")

    with caplog.at_level(logging.ERROR, logger="revenium_middleware"):
        thread = run_async_in_thread(failing_metering_call())
        assert thread is not None
        thread.join(timeout=5)
        assert not thread.is_alive()

    status = get_metering_status()
    assert status.error_count == 1
    assert isinstance(status.last_error, RuntimeError)
    error_records = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert any("metering" in r.getMessage().lower() for r in error_records)
