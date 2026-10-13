"""LiteLLM client calls to Claude bill each advisor and compaction pass at its own model (BACK-4221).

Anthropic bills an advisor pass at the advisor model's rate and a compaction
pass outside the top-level usage. LiteLLM sums every pass into
``prompt_tokens`` and ``completion_tokens`` and keeps the passes on
``usage.iterations``. The main record must carry only the executor's own
tokens, and each pass must be its own record, as the direct Anthropic
integration sends them.

Every call here is a real ``litellm.completion`` or ``litellm.acompletion``
against a local Anthropic-shaped upstream, so the usage is LiteLLM's own.
A streamed answer carries ``iterations`` on ``message_delta`` only, as the live
API does.
"""
import asyncio
import contextvars
import datetime
import threading
from types import SimpleNamespace
from unittest.mock import patch

import pytest

pytest.importorskip("litellm")

import litellm  # noqa: E402

from revenium_middleware import idempotency_key  # noqa: E402
from revenium_middleware.litellm.client import middleware as mw  # noqa: E402

from . import fake_anthropic_upstream  # noqa: E402

ANTHROPIC_MODEL = "anthropic/claude-haiku-4-5"
ADVISOR_MODEL = "claude-sonnet-5-5"
ADVISOR_TOOL = {"type": "advisor_20260301", "name": "advisor", "model": ADVISOR_MODEL}
MESSAGES = [{"role": "system", "content": "SECRET SYSTEM"}, {"role": "user", "content": "Say ok"}]


def message_pass(input_tokens, output_tokens, **cache):
    return {"type": "message", "input_tokens": input_tokens, "output_tokens": output_tokens, **cache}


def advisor_pass(input_tokens, output_tokens, **cache):
    return {"type": "advisor_message", "model": ADVISOR_MODEL,
            "input_tokens": input_tokens, "output_tokens": output_tokens, **cache}


REPLAY_ITERATIONS = [message_pass(1021, 26), advisor_pass(2603, 202), message_pass(1150, 4)]


def executor_usage(iterations, **cache):
    """Anthropic's top level: the executor's own passes only."""
    executor = [entry for entry in iterations if entry["type"] not in ("advisor_message", "compaction")]
    usage = {
        "input_tokens": sum(entry["input_tokens"] for entry in executor),
        "output_tokens": sum(entry["output_tokens"] for entry in executor),
        **cache,
    }
    if iterations is not None:
        usage["iterations"] = iterations
    return usage


def run_in_caller_context(coro):
    """Run the metering coroutine as the production worker does: in a copy of the caller's context."""
    context = contextvars.copy_context()
    thread = threading.Thread(target=context.run, args=(asyncio.run, coro))
    thread.start()
    thread.join()
    return thread


@pytest.fixture
def submissions():
    """Every submit_ai_event call as (payload, idempotency_key)."""
    recorded = []

    def record(operation, payload, idempotency_key=None):
        recorded.append((payload, idempotency_key))

    with patch.object(mw, "run_async_in_thread", side_effect=run_in_caller_context), \
            patch.object(mw, "submit_ai_event", side_effect=record):
        yield recorded


@pytest.fixture
def anthropic_upstream():
    servers = []

    def start(usage):
        start_usage = {name: value for name, value in usage.items() if name != "iterations"}
        server = fake_anthropic_upstream.start_upstream(usage, start_usage=start_usage)
        servers.append(server)
        return fake_anthropic_upstream.api_base(server)

    yield start
    fake_anthropic_upstream.stop_upstreams(servers)


def call_kwargs(api_base, **extra):
    return {"model": ANTHROPIC_MODEL, "messages": MESSAGES, "api_base": api_base, "api_key": "sk-ant-fake",
            "tools": [ADVISOR_TOOL], **extra}


def sync_call(api_base):
    return litellm.completion(**call_kwargs(api_base))


def sync_stream(api_base):
    return list(litellm.completion(**call_kwargs(api_base, stream=True)))


def async_call(api_base):
    return asyncio.run(litellm.acompletion(**call_kwargs(api_base)))


def async_stream(api_base):
    async def drain():
        stream = await litellm.acompletion(**call_kwargs(api_base, stream=True))
        return [chunk async for chunk in stream]

    return asyncio.run(drain())


CALL_STYLES = {
    "completion": sync_call,
    "completion-stream": sync_stream,
    "acompletion": async_call,
    "acompletion-stream": async_stream,
}


def executor_model(style):
    """The main record's model: a stream's chunks name the requested model, a response the upstream's."""
    return ANTHROPIC_MODEL.split("/", 1)[1] if style.endswith("-stream") else fake_anthropic_upstream.UPSTREAM_MODEL


def main_and_passes(submissions, expected_passes):
    rows = [payload for payload, _ in submissions]
    assert len(rows) == 1 + expected_passes, rows
    return rows[0], rows[1:]


@pytest.mark.parametrize("style", sorted(CALL_STYLES))
def test_advisor_pass_is_recorded_under_the_advisor_model(style, submissions, anthropic_upstream):
    CALL_STYLES[style](anthropic_upstream(executor_usage(REPLAY_ITERATIONS)))

    main, (advisor,) = main_and_passes(submissions, 1)
    assert main["model"] == executor_model(style)
    assert (main["input_token_count"], main["output_token_count"], main["total_token_count"]) == (2171, 30, 2201)
    assert advisor["model"] == ADVISOR_MODEL
    assert (advisor["input_token_count"], advisor["output_token_count"]) == (2603, 202)
    assert advisor["operation_subtype"] == "advisor_message"
    assert advisor["transaction_id"] == main["transaction_id"] + "-iteration-1"
    assert advisor["parent_transaction_id"] == main["transaction_id"]
    assert advisor["provider"] == "LITELLM"
    assert advisor["is_streamed"] == style.endswith("-stream")


@pytest.mark.parametrize("style", ["completion", "completion-stream"])
def test_advisor_record_carries_no_captured_prompt_with_capture_on(style, submissions, anthropic_upstream,
                                                                   monkeypatch):
    """BACK-4231: the advisor record must not store the prompt and reply the main record holds."""
    monkeypatch.setenv("REVENIUM_CAPTURE_PROMPTS", "true")
    CALL_STYLES[style](anthropic_upstream(executor_usage(REPLAY_ITERATIONS)))

    main, (advisor,) = main_and_passes(submissions, 1)
    for field in ("system_prompt", "input_messages", "output_response"):
        assert main[field] is not None, field
        assert advisor[field] is None, field


def test_pass_records_derive_their_idempotency_key_from_the_main_record(submissions, anthropic_upstream):
    with idempotency_key("advisor-call"):
        sync_call(anthropic_upstream(executor_usage(REPLAY_ITERATIONS)))

    assert [key for _, key in submissions] == ["advisor-call", "advisor-call-iteration-1"]


ADVISOR_WRITE_BREAKDOWNS = {
    "advisor-write-detailed": {"cache_creation": {"ephemeral_5m_input_tokens": 9, "ephemeral_1h_input_tokens": 0}},
    "advisor-write-undetailed": {},
}
ADVISOR_WRITE_SPLITS = {"advisor-write-detailed": (9, 0), "advisor-write-undetailed": (None, None)}


@pytest.mark.parametrize("case", sorted(ADVISOR_WRITE_BREAKDOWNS))
@pytest.mark.parametrize("style", sorted(CALL_STYLES))
def test_advisor_cache_tokens_leave_the_main_record(style, case, submissions, anthropic_upstream):
    """LiteLLM folds an undetailed advisor write into the five-minute bucket; the main record must not."""
    one_hour_write = {"cache_creation": {"ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 50}}
    iterations = [
        message_pass(1021, 26, cache_read_input_tokens=100, cache_creation_input_tokens=50, **one_hour_write),
        advisor_pass(2603, 202, cache_read_input_tokens=7, cache_creation_input_tokens=9,
                     **ADVISOR_WRITE_BREAKDOWNS[case]),
        message_pass(1150, 4),
    ]
    CALL_STYLES[style](anthropic_upstream(executor_usage(
        iterations, cache_read_input_tokens=100, cache_creation_input_tokens=50, **one_hour_write)))

    main, (advisor,) = main_and_passes(submissions, 1)
    assert main["input_token_count"] == 2171
    assert main["cache_read_token_count"] == 100
    assert main["cache_creation_token_count"] == 50
    assert main["cache_creation1h_token_count"] == 50
    assert main["cache_creation5m_token_count"] == 0
    assert main["cache_creation5m_token_count"] + main["cache_creation1h_token_count"] == \
        main["cache_creation_token_count"]
    assert advisor["cache_read_token_count"] == 7
    assert advisor["cache_creation_token_count"] == 9
    assert (advisor.get("cache_creation5m_token_count"), advisor.get("cache_creation1h_token_count")) == \
        ADVISOR_WRITE_SPLITS[case]


@pytest.mark.parametrize("style", sorted(CALL_STYLES))
def test_compaction_pass_is_recorded_under_the_response_model(style, submissions, anthropic_upstream):
    compaction = {"type": "compaction", "input_tokens": 1800, "output_tokens": 350}
    CALL_STYLES[style](anthropic_upstream(executor_usage([compaction, message_pass(900, 20)])))

    main, (pass_record,) = main_and_passes(submissions, 1)
    assert (main["input_token_count"], main["output_token_count"]) == (900, 20)
    assert pass_record["model"] == main["model"] == executor_model(style)
    assert pass_record["operation_subtype"] == "compaction"
    assert (pass_record["input_token_count"], pass_record["output_token_count"]) == (1800, 350)
    assert pass_record["transaction_id"] == main["transaction_id"] + "-iteration-0"


NO_BILLED_PASSES = {
    "absent": None,
    "empty": [],
    "message-only": [message_pass(2171, 30)],
    "with-fallback-message": [message_pass(1000, 10), {"type": "fallback_message", "input_tokens": 1171,
                                                       "output_tokens": 20}],
}


@pytest.mark.parametrize("case", sorted(NO_BILLED_PASSES))
@pytest.mark.parametrize("style", sorted(CALL_STYLES))
def test_a_call_without_billed_passes_sends_one_unchanged_record(style, case, submissions, anthropic_upstream):
    usage = {"input_tokens": 2171, "output_tokens": 30}
    if NO_BILLED_PASSES[case] is not None:
        usage["iterations"] = NO_BILLED_PASSES[case]
    CALL_STYLES[style](anthropic_upstream(usage))

    assert len(submissions) == 1, submissions
    row, key = submissions[0]
    assert (row["input_token_count"], row["output_token_count"], row["total_token_count"]) == (2171, 30, 2201)
    assert key is None


@pytest.mark.parametrize("prompt_tokens, completion_tokens", [(5000, 232), (4774, 31), (2000, 30)])
def test_unrecognised_totals_send_one_unchanged_record(prompt_tokens, completion_tokens, submissions):
    """Totals that are neither every pass nor the executor's alone are sent as reported, with no pass records."""
    usage = SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
                            iterations=REPLAY_ITERATIONS)
    response = litellm.ModelResponse(model="claude-haiku-4-5")

    mw.handle_response(response, datetime.datetime.now(datetime.timezone.utc), {}, False,
                       usage=usage, upstream_provider="anthropic")

    assert len(submissions) == 1, submissions
    row, key = submissions[0]
    assert (row["input_token_count"], row["output_token_count"]) == (prompt_tokens, completion_tokens)
    assert key is None


def test_executor_only_totals_keep_their_counts_and_still_send_the_pass(submissions):
    """A LiteLLM that reports Anthropic's executor-only top level is not netted a second time."""
    usage = SimpleNamespace(prompt_tokens=2171, completion_tokens=30, iterations=REPLAY_ITERATIONS)
    response = litellm.ModelResponse(model="claude-haiku-4-5")

    mw.handle_response(response, datetime.datetime.now(datetime.timezone.utc), {}, False,
                       usage=usage, upstream_provider="anthropic")

    main, (advisor,) = main_and_passes(submissions, 1)
    assert (main["input_token_count"], main["output_token_count"]) == (2171, 30)
    assert advisor["model"] == ADVISOR_MODEL
    assert (advisor["input_token_count"], advisor["output_token_count"]) == (2603, 202)
