import uuid
from unittest.mock import MagicMock, patch

import pytest

from revenium_middleware import idempotency_key
from revenium_middleware._core import submit_ai_event


@pytest.fixture
def mock_metering_client():
    """Patch the client used by the wrapper so we capture the kwargs."""
    mock = MagicMock()
    mock.ai.create_completion = MagicMock(return_value=MagicMock(id="metered-id-1"))
    with patch("revenium_middleware._core.metering_submission.get_client", lambda: mock):
        yield mock


def test_default_idempotency_key_is_uuid_v4(mock_metering_client):
    """Default path: header is an auto-generated UUID v4."""
    submit_ai_event("completion", {"transaction_id": "t1"})
    kwargs = mock_metering_client.ai.create_completion.call_args.kwargs
    key = kwargs["extra_headers"]["Idempotency-Key"]
    assert uuid.UUID(key).version == 4


def test_override_via_context_manager_reaches_client(mock_metering_client):
    """User-facing CM: header equals the override value."""
    with idempotency_key("order-abc-123"):
        submit_ai_event("completion", {"transaction_id": "t2"})
    kwargs = mock_metering_client.ai.create_completion.call_args.kwargs
    assert kwargs["extra_headers"]["Idempotency-Key"] == "order-abc-123"
