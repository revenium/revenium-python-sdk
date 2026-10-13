"""A record the metering API refuses with a 400 must fail visibly (BACK-2710).

A value that breaks a field rule the SDK knows (an ``effort`` over 16
characters, or with a character outside ^[A-Za-z0-9_-]+$) is left off the
record before it is sent (BACK-4339), so it no longer costs the record. Any
other 400 still loses the whole completion event, and that must never be
absorbed quietly: the customer would see "metering worked" while the row never
landed.

These tests pin the three ways such a failure surfaces:

* it is classified permanent, so the store-and-forward buffer does not
  swallow it as a "transient" event awaiting replay;
* ``submit_ai_event`` re-raises, so the provider metering path logs it;
* the failure lands in the metering status counters and the error callbacks,
  the two programmatic channels customers poll or subscribe to.
"""
import logging
from unittest.mock import MagicMock, patch

import httpx
import pytest

from revenium_middleware._core import metering_buffer, metering_submission
from revenium_middleware._core.metering_buffer import (
    MeteringBuffer,
    is_retryable_failure,
)
from revenium_middleware._core.metering_status import (
    get_metering_status,
    on_metering_error,
    reset_metering_status,
)
from revenium_middleware._metering._exceptions import APIStatusError

# Values outside the backend rule: the pattern forbids spaces, and the length
# cap is 16 characters.
MALFORMED_EFFORTS = ["much too high", "a" * 17]


def permanent_rejection_error():
    """A 400 the metering API returns for a record it refuses."""
    request = httpx.Request("POST", "https://api.test/meter/v2/ai/completions")
    response = httpx.Response(400, request=request)
    return APIStatusError(
        "Error code: 400 - record refused",
        response=response,
        body=None,
    )


@pytest.fixture
def isolated_buffer(monkeypatch):
    """An isolated buffer singleton so a stray push is visible in stats()."""
    buf = MeteringBuffer(max_size=100, flush_interval=9999.0)
    monkeypatch.setattr(metering_buffer, "_buffer", buf)
    return buf


@pytest.fixture
def rejecting_client(monkeypatch):
    client = MagicMock()
    client.ai.create_completion.side_effect = permanent_rejection_error()
    monkeypatch.setattr(metering_submission, "current_client", lambda: client)
    return client


@pytest.fixture(autouse=True)
def clean_status():
    reset_metering_status()
    yield
    reset_metering_status()


@pytest.mark.parametrize("effort", MALFORMED_EFFORTS)
def test_effort_is_extracted_verbatim(effort):
    """Extraction has no allow-list; the send guard decides what reaches the API."""
    from revenium_middleware._core.fields import extract_effort_field

    assert extract_effort_field({"effort": effort}) == {"effort": effort}


def test_a_400_from_the_metering_api_is_not_retryable():
    assert is_retryable_failure(permanent_rejection_error()) is False


def test_a_refused_record_raises_and_is_not_buffered(isolated_buffer, rejecting_client):
    with pytest.raises(APIStatusError):
        metering_submission.submit_ai_event(
            "completion", {"model": "gpt-test", "effort": "high"}
        )

    assert isolated_buffer.stats()["size"] == 0


def test_a_refused_record_is_recorded_in_the_metering_status(
    isolated_buffer, rejecting_client
):
    with pytest.raises(APIStatusError):
        metering_submission.submit_ai_event(
            "completion", {"model": "gpt-test", "effort": "high"}
        )

    status = get_metering_status()
    assert status.error_count == 1
    assert status.success_count == 0
    assert isinstance(status.last_error, APIStatusError)


def test_a_refused_record_notifies_error_callbacks(isolated_buffer, rejecting_client):
    seen = []
    on_metering_error(seen.append)

    with pytest.raises(APIStatusError):
        metering_submission.submit_ai_event(
            "completion", {"model": "gpt-test", "effort": "high"}
        )

    assert len(seen) == 1
    assert seen[0].operation == "completion"


def test_a_refused_record_is_logged_as_a_permanent_failure(
    isolated_buffer, rejecting_client, caplog
):
    with caplog.at_level(logging.ERROR, logger="revenium_middleware"):
        with pytest.raises(APIStatusError):
            metering_submission.submit_ai_event(
                "completion", {"model": "gpt-test", "effort": "high"}
            )

    assert "failed permanently" in caplog.text


def test_openai_path_logs_a_refused_record_instead_of_dropping_it(caplog):
    """End of the chain: the provider path surfaces the failure in the log.

    Metering runs on a background thread and must not raise into customer
    code, so a loud ERROR plus the status counters above is the SDK's
    error-surfacing convention -- what must not happen is silence.
    """
    import asyncio

    from revenium_middleware.openai.middleware import log_token_usage

    with patch(
        "revenium_middleware.openai.middleware.current_client", lambda: object()
    ), patch(
        "revenium_middleware.openai.middleware.submit_ai_event",
        side_effect=permanent_rejection_error(),
    ):
        with caplog.at_level(logging.ERROR):
            asyncio.run(
                log_token_usage(
                    response_id="completion-refused",
                    model="gpt-4o-mini",
                    prompt_tokens=10,
                    completion_tokens=5,
                    total_tokens=15,
                    cached_tokens=0,
                    stop_reason="END",
                    request_time="2026-08-23T12:00:00Z",
                    response_time="2026-08-23T12:00:01Z",
                    request_duration=1000,
                    usage_metadata={"effort": "high"},
                )
            )

    assert "REVENIUM FAILURE" in caplog.text
