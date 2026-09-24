"""Reasoning effort forwarding on the Ollama completion path (BACK-2710).

Uses a real metering client (only the HTTP layer mocked) so a kwarg the typed
methods do not accept fails here instead of raising a swallowed TypeError in
production.
"""
import datetime
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from revenium_middleware.ollama.middleware import handle_response


def _response():
    return SimpleNamespace(
        model="llama3.2",
        prompt_eval_count=10,
        eval_count=5,
        done_reason="stop",
    )


def _wire_body(usage_metadata):
    from revenium_middleware._metering import ReveniumMetering

    real_client = ReveniumMetering(api_key="test-key")
    mock_post = MagicMock(return_value=type("R", (), {"id": "evt-1"})())
    with patch.object(real_client.ai, "_post", mock_post):
        with patch("revenium_middleware._core.metering.client", real_client):
            handle_response(
                response=_response(),
                request_time_dt=datetime.datetime.now(datetime.timezone.utc),
                usage_metadata=usage_metadata,
                is_streaming=False,
                transaction_id="ollama-effort-test",
                endpoint="chat",
                request_kwargs={"model": "llama3.2"},
            )
            deadline = time.monotonic() + 5.0
            while not mock_post.called and time.monotonic() < deadline:
                time.sleep(0.01)
    assert mock_post.called, "metering call never reached the HTTP layer"
    return mock_post.call_args.kwargs["body"]


def test_effort_reaches_the_wire():
    assert _wire_body({"effort": "high"})["effort"] == "high"


def test_unrecognised_level_reaches_the_wire_unchanged():
    assert _wire_body({"effort": "hyper_9"})["effort"] == "hyper_9"


def test_unset_effort_omitted_from_the_wire():
    assert "effort" not in _wire_body({"trace_id": "no-effort-field"})


PROMPT_CONTEXT_METADATA = {
    "prompt_id": "prompt-42",
    "promptLength": 1834,
    "query_source": "repl_main_thread",
    "speed": "fast",
    "subagentType": "Explore",
}

PROMPT_CONTEXT_WIRE = {
    "promptId": "prompt-42",
    "promptLength": 1834,
    "querySource": "repl_main_thread",
    "speed": "fast",
    "subagentType": "Explore",
}


def _assert_prompt_context(body):
    for wire_name, value in PROMPT_CONTEXT_WIRE.items():
        assert body[wire_name] == value, wire_name


def _assert_no_prompt_context(body):
    for wire_name in PROMPT_CONTEXT_WIRE:
        assert wire_name not in body, wire_name


def test_prompt_context_fields_reach_the_wire():
    _assert_prompt_context(_wire_body(PROMPT_CONTEXT_METADATA))


def test_unset_prompt_context_fields_omitted_from_the_wire():
    _assert_no_prompt_context(_wire_body({"trace_id": "no-prompt-context"}))
