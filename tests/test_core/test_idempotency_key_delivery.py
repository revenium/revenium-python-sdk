"""A usage record whose idempotency key cannot go in an HTTP header is still sent, and never pauses other records (BACK-4366)."""
import uuid

import pytest

from revenium_middleware._core import metering
from revenium_middleware._core.context import _idempotency_key_context
from revenium_middleware._core.delivery_circuit import get_circuit
from revenium_middleware._core.metering_submission import submit_ai_event

pytestmark = pytest.mark.usefixtures("isolated_metering")

PROCESS_KEY = "hak_fake_idempotency_probe_process_key"

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
    "transaction_id": "txn-idempotency-probe",
}


def is_uuid4(value):
    try:
        return uuid.UUID(value).version == 4
    except (TypeError, ValueError):
        return False


@pytest.fixture
def meter(recording_meters, monkeypatch):
    stub, _ = recording_meters
    client = metering._new_client(PROCESS_KEY, stub.meter_url)
    monkeypatch.setattr(metering, "client", client)
    yield stub
    client.close()


def test_three_unsendable_keys_do_not_open_the_breaker_and_a_good_record_is_sent_at_once(meter, isolated_metering):
    token = _idempotency_key_context.set("order-1006\n")
    try:
        for _ in range(3):
            submit_ai_event("completion", dict(COMPLETION_ARGS))
    finally:
        _idempotency_key_context.reset(token)

    assert get_circuit().is_open() is False
    assert get_circuit().allows_send() is True
    assert isolated_metering.stats()["size"] == 0
    sent_keys = [request["headers"].get("Idempotency-Key") for request in meter.requests]
    assert len(sent_keys) == 3
    assert all(is_uuid4(key) for key in sent_keys)
    assert not [key for key in sent_keys if key is not None and "order-1006" in key]

    submit_ai_event("completion", dict(COMPLETION_ARGS))

    assert len(meter.requests) == 4
