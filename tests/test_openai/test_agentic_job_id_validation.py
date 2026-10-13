"""BACK-4020: an agentic job id AI ingest refuses must not cost the completion.

The id is left off the metered call, which is still submitted; the other job
fields stay.
"""
import asyncio
import os
from types import SimpleNamespace
from unittest.mock import patch

from revenium_middleware.openai.middleware import log_token_usage


def _log(usage_metadata):
    asyncio.run(
        log_token_usage(
            response_id="completion-job-id-test",
            model="gpt-4o-mini",
            prompt_tokens=100,
            completion_tokens=25,
            total_tokens=125,
            cached_tokens=0,
            stop_reason="END",
            request_time="2026-10-08T12:00:00Z",
            response_time="2026-10-08T12:00:01Z",
            request_duration=1000,
            usage_metadata=usage_metadata,
        )
    )


@patch("revenium_middleware.openai.middleware.current_client", lambda: object())
@patch("revenium_middleware.openai.middleware.submit_ai_event")
def test_refused_job_id_is_left_off_a_completion_that_is_still_metered(mock_submit):
    mock_submit.return_value = SimpleNamespace(id="completion-job-id-test")

    with patch.dict(os.environ, {}, clear=True):
        _log({"agentic_job_id": "order:42", "agentic_job_name": "n"})

    assert mock_submit.call_count == 1
    extra_body = mock_submit.call_args[0][1].get("extra_body") or {}
    assert "agenticJobId" not in extra_body
    assert extra_body.get("agenticJobName") == "n"


@patch("revenium_middleware.openai.middleware.current_client", lambda: object())
@patch("revenium_middleware.openai.middleware.submit_ai_event")
def test_valid_job_id_reaches_the_completion(mock_submit):
    mock_submit.return_value = SimpleNamespace(id="completion-job-id-test")

    with patch.dict(os.environ, {}, clear=True):
        _log({"agentic_job_id": "job-1"})

    assert mock_submit.call_count == 1
    extra_body = mock_submit.call_args[0][1].get("extra_body") or {}
    assert extra_body.get("agenticJobId") == "job-1"
