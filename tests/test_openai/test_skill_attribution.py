"""Skill attribution forwarding on the OpenAI completion path.

usage_metadata skill fields must reach the metering client as the typed
skill_* keyword arguments (not via extra_body).
"""
import asyncio
import os
from types import SimpleNamespace
from unittest.mock import patch

from revenium_middleware.openai.middleware import log_token_usage


def _log(usage_metadata):
    asyncio.run(
        log_token_usage(
            response_id="completion-skill-test",
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
def test_usage_metadata_skill_fields_forwarded_as_typed_kwargs(mock_submit):
    mock_submit.return_value = SimpleNamespace(id="completion-skill-test")

    _log({"skill_name": "summarize-docs", "skill_source": "marketplace"})

    payload = mock_submit.call_args[0][1]
    assert payload["skill_name"] == "summarize-docs"
    assert payload["skill_source"] == "marketplace"
    extra_body = payload.get("extra_body") or {}
    assert "skillName" not in extra_body


@patch("revenium_middleware.openai.middleware.get_client", lambda: object())
@patch("revenium_middleware.openai.middleware.submit_ai_event")
def test_unset_skill_fields_omitted_from_payload(mock_submit):
    mock_submit.return_value = SimpleNamespace(id="completion-skill-test")

    # extract_skill_fields falls back to REVENIUM_SKILL_* env vars; clear the
    # environment so a var set in the developer's shell can't fail this test.
    with patch.dict(os.environ, {}, clear=True):
        _log({"trace_id": "no-skill-fields"})

    payload = mock_submit.call_args[0][1]
    assert "skill_name" not in payload
    assert "skill_source" not in payload
