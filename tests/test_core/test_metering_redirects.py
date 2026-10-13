"""The usage-record client never follows a redirect, so the Revenium key and the record stay at the address configured (BACK-4360)."""
import asyncio
import logging
import uuid

import pytest

from revenium_middleware._core import metering, metering_pool
from revenium_middleware._core.delivery_circuit import get_circuit
from revenium_middleware._core.metering_status import get_metering_status
from revenium_middleware._core.metering_submission import submit_ai_event
from revenium_middleware._metering import AsyncReveniumMetering
from revenium_middleware._metering._base_client import _DefaultAsyncHttpxClient, _DefaultHttpxClient
from revenium_middleware._metering._exceptions import APIStatusError
from revenium_middleware.agentic_outcomes import AgenticOutcomeClient, AgenticOutcomeSettings

pytestmark = pytest.mark.usefixtures("isolated_metering")

PROCESS_KEY = "hak_fake_redirect_probe_process_key"
SCOPED_KEY = "hak_fake_redirect_probe_scoped_key"
WRITE_KEY = "rev_sk_fake_redirect_probe_write_key"
REDIRECTS = [301, 302, 303, 307, 308]

COMPLETION_ARGS = {
    "completion_start_time": "2026-10-11T00:00:00Z",
    "cost_type": "AI",
    "input_token_count": 3,
    "is_streamed": False,
    "model": "gpt-4o-mini",
    "output_token_count": 2,
    "provider": "OPENAI",
    "request_duration": 10,
    "request_time": "2026-10-11T00:00:00Z",
    "response_time": "2026-10-11T00:00:01Z",
    "stop_reason": "END",
    "total_token_count": 5,
    "transaction_id": "txn-redirect-probe",
}

COMPLETION_PAYLOAD = {
    "completionStartTime": "2026-10-11T00:00:00Z",
    "costType": "AI",
    "inputTokenCount": 3,
    "isStreamed": False,
    "model": "gpt-4o-mini",
    "outputTokenCount": 2,
    "provider": "OPENAI",
    "requestDuration": 10,
    "requestTime": "2026-10-11T00:00:00Z",
    "responseTime": "2026-10-11T00:00:01Z",
    "stopReason": "END",
    "totalTokenCount": 5,
    "transactionId": "txn-redirect-probe",
}


@pytest.fixture
def meters(recording_meters):
    """Stub A redirects every usage record to stub B on another host name; B accepts everything."""
    a, b = recording_meters
    a.answer_with(307, {"Location": other_host_location(b)})
    return a, b


def other_host_location(b):
    return f"http://localhost:{b.port}/meter/v2/ai/completions"


@pytest.fixture
def process_client_at_a(meters, monkeypatch):
    a, _ = meters
    client = metering._new_client(PROCESS_KEY, a.meter_url)
    monkeypatch.setattr(metering, "client", client)
    yield client
    client.close()


def submit_record():
    return submit_ai_event("completion", dict(COMPLETION_ARGS))


def warnings_in(caplog):
    return [record for record in caplog.records if record.levelno >= logging.WARNING]


@pytest.mark.parametrize("status", REDIRECTS)
def test_a_redirected_usage_record_is_not_sent_to_the_other_host(meters, process_client_at_a, status):
    a, b = meters
    a.answer_with(status, {"Location": other_host_location(b)})

    with pytest.raises(APIStatusError):
        submit_record()

    assert len(a.requests) == 1
    assert b.requests == []


def test_a_redirect_without_a_location_is_not_followed_and_is_logged(meters, process_client_at_a, caplog):
    a, b = meters
    a.answer_with(307)

    with caplog.at_level(logging.WARNING, logger="revenium_middleware"):
        with pytest.raises(APIStatusError):
            submit_record()

    assert len(a.requests) == 1
    assert b.requests == []
    warnings = warnings_in(caplog)
    assert len(warnings) == 1
    assert "no Location header" in warnings[0].getMessage()


def test_a_same_host_redirect_is_not_followed(meters, process_client_at_a):
    a, _ = meters
    a.answer_with(307, {"Location": f"{a.base_url}/other"})

    with pytest.raises(APIStatusError):
        submit_record()

    assert len(a.requests) == 1


def test_a_redirect_counts_as_failed_and_is_not_buffered_or_held(meters, process_client_at_a, isolated_metering):
    for _ in range(3):
        with pytest.raises(APIStatusError):
            submit_record()

    status = get_metering_status()
    assert status.success_count == 0
    assert status.error_count == 3
    assert isolated_metering.stats()["size"] == 0
    assert get_circuit().is_open() is False


def test_a_redirect_with_retry_after_is_still_dropped(meters, process_client_at_a, isolated_metering):
    a, b = meters
    a.answer_with(307, {"Location": other_host_location(b), "Retry-After": "1"})

    with pytest.raises(APIStatusError):
        submit_record()

    assert len(a.requests) == 1
    assert b.requests == []
    assert isolated_metering.stats()["size"] == 0
    assert get_circuit().is_open() is False


@pytest.mark.parametrize("location", [True, False])
def test_a_redirect_asking_to_be_retried_is_sent_once(meters, monkeypatch, location):
    a, b = meters
    headers = {"x-should-retry": "true"}
    if location:
        headers["Location"] = other_host_location(b)
    a.answer_with(307, headers)
    monkeypatch.setenv(metering.MAX_RETRIES_ENV, str(metering.DEFAULT_MAX_RETRIES))
    client = metering._new_client(PROCESS_KEY, a.meter_url)
    monkeypatch.setattr(metering, "client", client)

    try:
        with pytest.raises(APIStatusError):
            submit_record()
    finally:
        client.close()

    assert len(a.requests) == 1
    assert b.requests == []


def test_the_async_metering_client_does_not_retry_a_redirect(meters):
    a, b = meters
    a.answer_with(307, {"Location": other_host_location(b), "x-should-retry": "true"})

    async def send():
        client = AsyncReveniumMetering(api_key=PROCESS_KEY, base_url=a.meter_url, max_retries=2)
        try:
            await client.ai.create_completion(**COMPLETION_ARGS)
        finally:
            await client.close()

    with pytest.raises(APIStatusError):
        asyncio.run(send())

    assert len(a.requests) == 1
    assert b.requests == []


def test_a_redirect_is_logged_once_with_its_address(meters, process_client_at_a, caplog):
    _, b = meters

    with caplog.at_level(logging.DEBUG, logger="revenium_middleware"):
        with pytest.raises(APIStatusError):
            submit_record()

    warnings = warnings_in(caplog)
    assert len(warnings) == 1
    assert warnings[0].name == "revenium_middleware"
    assert f"localhost:{b.port}" in warnings[0].getMessage()
    messages = [record.getMessage() for record in caplog.records]
    assert not [message for message in messages if "failed permanently" in message]
    assert not [message for message in messages if PROCESS_KEY in message]


def test_a_redirect_inside_metering_key_does_not_send_the_scoped_key(meters, process_client_at_a):
    a, b = meters

    with metering.metering_key(SCOPED_KEY, base_url=a.meter_url):
        with pytest.raises(APIStatusError):
            submit_record()

    assert len(a.requests) == 1
    assert a.requests[0]["headers"].get("x-api-key") == SCOPED_KEY
    assert b.requests == []


def test_emit_completion_does_not_follow_a_redirect(meters, caplog):
    a, b = meters
    settings = AgenticOutcomeSettings(api_key=WRITE_KEY, meter_base_url=a.base_url)

    with AgenticOutcomeClient(settings) as outcomes:
        with caplog.at_level(logging.WARNING, logger="revenium_middleware"):
            outcomes.emit_completion(dict(COMPLETION_PAYLOAD))
            assert metering_pool.wait_until_idle(10)

    assert len(a.requests) == 1
    assert b.requests == []
    warnings = warnings_in(caplog)
    assert len(warnings) == 1
    assert f"localhost:{b.port}" in warnings[0].getMessage()


def test_the_async_metering_client_does_not_follow_a_redirect(meters):
    a, b = meters

    async def send():
        client = AsyncReveniumMetering(api_key=PROCESS_KEY, base_url=a.meter_url, max_retries=0)
        try:
            await client.ai.create_completion(**COMPLETION_ARGS)
        finally:
            await client.close()

    with pytest.raises(APIStatusError):
        asyncio.run(send())

    assert len(a.requests) == 1
    assert b.requests == []


def test_the_default_http_clients_do_not_follow_redirects():
    sync_client = _DefaultHttpxClient()
    async_client = _DefaultAsyncHttpxClient()
    try:
        assert sync_client.follow_redirects is False
        assert async_client.follow_redirects is False
    finally:
        sync_client.close()
        asyncio.run(async_client.aclose())


def test_a_replayed_record_that_meets_a_redirect_is_dropped_with_one_warning(meters, isolated_metering, caplog):
    a, b = meters
    isolated_metering.push("ai", {
        "operation": "completion",
        "args": {**COMPLETION_ARGS, "extra_headers": {"Idempotency-Key": str(uuid.uuid4())}},
        "api_key": PROCESS_KEY,
        "base_url": a.meter_url,
    })

    with caplog.at_level(logging.WARNING, logger="revenium_middleware"):
        result = isolated_metering.flush()

    assert result["discarded"] == 1
    assert len(a.requests) == 1
    assert b.requests == []
    warnings = warnings_in(caplog)
    assert len(warnings) == 1
    assert f"localhost:{b.port}" in warnings[0].getMessage()
