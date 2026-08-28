"""Coding assistant attribution forwarding on the OpenAI completion path.

The usage_metadata coding assistant field must reach the metering client as
the typed coding_assistant_account_uuid keyword argument (not via extra_body).
"""
import asyncio
from types import SimpleNamespace
from unittest.mock import patch

from revenium_middleware.openai.middleware import log_token_usage


def _log(usage_metadata):
    asyncio.run(
        log_token_usage(
            response_id="completion-coding-assistant-test",
            model="gpt-4o-mini",
            prompt_tokens=100,
            completion_tokens=25,
            total_tokens=125,
            cached_tokens=0,
            stop_reason="END",
            request_time="2026-08-12T12:00:00Z",
            response_time="2026-08-12T12:00:01Z",
            request_duration=1000,
            usage_metadata=usage_metadata,
        )
    )


@patch("revenium_middleware.openai.middleware.get_client", lambda: object())
@patch("revenium_middleware.openai.middleware.submit_ai_event")
def test_usage_metadata_coding_assistant_uuid_forwarded_as_typed_kwarg(mock_submit):
    mock_submit.return_value = SimpleNamespace(id="completion-coding-assistant-test")

    _log({"coding_assistant_account_uuid": "acct-uuid-42"})

    payload = mock_submit.call_args[0][1]
    assert payload["coding_assistant_account_uuid"] == "acct-uuid-42"
    extra_body = payload.get("extra_body") or {}
    assert "codingAssistantAccountUuid" not in extra_body


@patch("revenium_middleware.openai.middleware.get_client", lambda: object())
@patch("revenium_middleware.openai.middleware.submit_ai_event")
def test_unset_coding_assistant_uuid_omitted_from_payload(mock_submit):
    mock_submit.return_value = SimpleNamespace(id="completion-coding-assistant-test")

    _log({"trace_id": "no-coding-assistant-field"})

    payload = mock_submit.call_args[0][1]
    assert "coding_assistant_account_uuid" not in payload
