"""OpenAI cache writes are recorded as cache-creation tokens and taken out of
the input count, once, on every OpenAI and Azure OpenAI entry point
(BACK-3998).

GPT-5.6 and later report the write in ``prompt_tokens_details`` (Chat) or
``input_tokens_details`` (Responses) as ``cache_write_tokens``, inside the
prompt count, and bill it at 1.25x the input rate. Cache reads stay gross in
the input count: the platform removes them itself for OpenAI-shaped rows.

The fixture is the verified probe call: prompt 4,583, of which 4,580 written
to the cache, 5 completion tokens.
"""
from types import SimpleNamespace

import pytest

from revenium_middleware.openai import middleware as mw

from . import test_openai_sdk_entry_points as entry_points
from .test_openai_sdk_entry_points import (  # noqa: F401  payloads is a fixture
    CHAT_REQUEST,
    NON_STREAMED,
    STREAMED_CHAT,
    STREAMED_RESPONSES,
    FakeOpenAI,
    async_chat_create,
    async_responses_create,
    payloads,
    sync_chat_create,
    sync_chat_create_stream,
    sync_responses_create,
    sync_responses_create_stream,
)
from . import responses_stream_events
from .responses_stream_events import MODEL

PROMPT = 4583
WRITTEN = 4580
COMPLETION = 5
TOTAL = PROMPT + COMPLETION


def chat_usage(prompt, completion, details):
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion,
        "prompt_tokens_details": details,
    }


def responses_usage(prompt, completion, cached, written):
    return {
        "input_tokens": prompt,
        "input_tokens_details": {"cached_tokens": cached, "cache_write_tokens": written},
        "output_tokens": completion,
        "output_tokens_details": {"reasoning_tokens": 0},
        "total_tokens": prompt + completion,
    }


WRITING_CHAT_USAGE = chat_usage(PROMPT, COMPLETION, {"cached_tokens": 0, "cache_write_tokens": WRITTEN})
WRITING_RESPONSES_USAGE = responses_usage(PROMPT, COMPLETION, 0, WRITTEN)


@pytest.fixture
def served_usage(monkeypatch):
    """Make the fake OpenAI server report the given chat and Responses usage."""
    def serve(chat=WRITING_CHAT_USAGE, responses=WRITING_RESPONSES_USAGE):
        monkeypatch.setattr(entry_points, "CHAT_USAGE", chat)
        monkeypatch.setattr(entry_points, "RESPONSES_USAGE", responses)
    return serve


def the_one_record(recorded):
    assert len(recorded) == 1
    return recorded[0]


def assert_write_taken_out_of_input(record):
    assert record["input_token_count"] == PROMPT - WRITTEN
    assert record["cache_creation_token_count"] == WRITTEN
    assert record["cache_read_token_count"] == 0
    assert record["total_token_count"] == TOTAL


def test_chat_completion_takes_cache_writes_out_of_input(payloads, served_usage):
    served_usage()

    sync_chat_create(FakeOpenAI(), None)

    assert_write_taken_out_of_input(the_one_record(payloads))


def test_responses_call_takes_cache_writes_out_of_input(payloads, served_usage):
    served_usage()

    sync_responses_create(FakeOpenAI(), None)

    assert_write_taken_out_of_input(the_one_record(payloads))


@pytest.mark.parametrize("call", [c for c in NON_STREAMED if c not in (sync_chat_create, sync_responses_create)],
                         ids=lambda c: c.__name__)
def test_every_async_non_streamed_entry_point_takes_cache_writes_out_of_input(call, payloads, served_usage):
    served_usage()

    call(FakeOpenAI(), None)

    assert_write_taken_out_of_input(the_one_record(payloads))


# STREAMED_CHAT and STREAMED_RESPONSES include every raw and streaming-response form.
@pytest.mark.parametrize("call", [*STREAMED_CHAT, *STREAMED_RESPONSES], ids=lambda c: c.__name__)
def test_every_streamed_entry_point_takes_cache_writes_out_of_input(call, payloads, served_usage):
    served_usage()

    call(FakeOpenAI(), None)

    record = the_one_record(payloads)
    assert_write_taken_out_of_input(record)
    assert record["is_streamed"] is True


def test_raw_response_non_streamed_call_takes_cache_writes_out_of_input(payloads, served_usage):
    served_usage()

    FakeOpenAI().sync_client().chat.completions.with_raw_response.create(**CHAT_REQUEST).parse()

    assert_write_taken_out_of_input(the_one_record(payloads))


def _azure_chat(fake):
    return [fake.azure_client().chat.completions.create(**CHAT_REQUEST)]


def _azure_chat_stream(fake):
    return list(fake.azure_client().chat.completions.create(**CHAT_REQUEST, stream=True))


def _azure_responses(fake):
    return [fake.azure_client().responses.create(model=MODEL, input="hi")]


@pytest.mark.parametrize("call", [_azure_chat, _azure_chat_stream, _azure_responses],
                         ids=["chat", "chat_stream", "responses"])
def test_azure_takes_cache_writes_out_of_input(call, payloads, served_usage):
    served_usage()

    call(FakeOpenAI(azure=True))

    record = the_one_record(payloads)
    assert record["provider"] == "Azure"
    assert_write_taken_out_of_input(record)


READ_AND_WRITE_PROMPT = 6000
READ = 4000
READ_AND_WRITE_WRITTEN = 1500


@pytest.mark.parametrize("call", [sync_chat_create, sync_chat_create_stream, sync_responses_create],
                         ids=lambda c: c.__name__)
def test_reads_and_writes_in_one_call_subtract_only_the_write(call, payloads, served_usage):
    """The platform takes reads out of input itself; doing it here too would
    net them out twice and price the input leg at nothing."""
    served_usage(
        chat=chat_usage(READ_AND_WRITE_PROMPT, 10,
                        {"cached_tokens": READ, "cache_write_tokens": READ_AND_WRITE_WRITTEN}),
        responses=responses_usage(READ_AND_WRITE_PROMPT, 10, READ, READ_AND_WRITE_WRITTEN),
    )

    call(FakeOpenAI(), None)

    record = the_one_record(payloads)
    assert record["input_token_count"] == READ_AND_WRITE_PROMPT - READ_AND_WRITE_WRITTEN
    assert record["cache_read_token_count"] == READ
    assert record["cache_creation_token_count"] == READ_AND_WRITE_WRITTEN
    assert record["total_token_count"] == READ_AND_WRITE_PROMPT + 10


ANTHROPIC_SHAPED_READS = {
    "flat_read": {"cache_read_input_tokens": READ},
    "input_token_details_read": {"input_token_details": {"cache_read": READ}},
}


def _usage_with_a_gateway_read(read_shape, extra=None):
    """A gateway that reports the read in an Anthropic-shaped field and the
    write only in the OpenAI field."""
    extra = {**ANTHROPIC_SHAPED_READS[read_shape], **(extra or {})}
    return {
        "chat": {**chat_usage(READ_AND_WRITE_PROMPT, 10,
                              {"cached_tokens": 0, "cache_write_tokens": READ_AND_WRITE_WRITTEN}), **extra},
        "responses": {**responses_usage(READ_AND_WRITE_PROMPT, 10, 0, READ_AND_WRITE_WRITTEN), **extra},
    }


@pytest.mark.parametrize("read_shape", sorted(ANTHROPIC_SHAPED_READS))
@pytest.mark.parametrize("call", [sync_chat_create, sync_chat_create_stream, sync_responses_create,
                                  sync_responses_create_stream, async_chat_create, async_responses_create],
                         ids=lambda c: c.__name__)
def test_a_write_beside_an_anthropic_shaped_read_is_recorded_as_cache_creation(call, read_shape, payloads,
                                                                                served_usage):
    """BACK-4209: the write was taken out of input and recorded nowhere."""
    served_usage(**_usage_with_a_gateway_read(read_shape))

    call(FakeOpenAI(), None)

    record = the_one_record(payloads)
    assert record["input_token_count"] == READ_AND_WRITE_PROMPT - READ_AND_WRITE_WRITTEN
    assert record["cache_creation_token_count"] == READ_AND_WRITE_WRITTEN
    assert record["cache_read_token_count"] == READ
    assert record["total_token_count"] == READ_AND_WRITE_PROMPT + 10


def test_an_anthropic_shaped_write_is_not_doubled_by_the_openai_mirror(payloads, served_usage):
    """Regression guard: a gateway that mirrors the write into both shapes
    reports it once, and the input keeps it, as before."""
    served_usage(**_usage_with_a_gateway_read(
        "flat_read", extra={"cache_creation_input_tokens": READ_AND_WRITE_WRITTEN}))

    sync_chat_create(FakeOpenAI(), None)

    record = the_one_record(payloads)
    assert record["input_token_count"] == READ_AND_WRITE_PROMPT
    assert record["cache_creation_token_count"] == READ_AND_WRITE_WRITTEN
    assert record["cache_read_token_count"] == READ


def test_a_call_without_cache_writes_is_recorded_as_before(payloads):
    """Regression guard: passes before and after the fix."""
    sync_chat_create(FakeOpenAI(), None)

    record = the_one_record(payloads)
    assert record["input_token_count"] == 11
    assert record["cache_read_token_count"] == 3
    assert record["cache_creation_token_count"] == 0


@pytest.mark.parametrize("details", [{"cached_tokens": 3}, {"cached_tokens": 3, "cache_write_tokens": None}],
                         ids=["absent", "null"])
def test_absent_or_null_cache_write_count_counts_as_zero(details, payloads, served_usage):
    """Regression guard: pre-5.6 models report no write (P0: None on Chat)."""
    served_usage(chat=chat_usage(11, 7, details))

    sync_chat_create(FakeOpenAI(), None)

    record = the_one_record(payloads)
    assert record["input_token_count"] == 11
    assert record["cache_creation_token_count"] == 0


def test_cache_writes_above_the_prompt_clamp_input_at_zero():
    response = SimpleNamespace(
        id="chatcmpl-clamp",
        model="gpt-5.6-luna",
        usage=SimpleNamespace(
            prompt_tokens=100,
            completion_tokens=5,
            total_tokens=105,
            prompt_tokens_details=SimpleNamespace(cached_tokens=0, cache_write_tokens=150),
        ),
        choices=[SimpleNamespace(finish_reason="stop")],
        system_fingerprint=None,
    )

    usage_data, _ = mw.extract_usage_data(
        response, mw.OperationType.CHAT, "2026-10-08T12:00:00Z", "2026-10-08T12:00:01Z", 1000
    )

    assert usage_data["input_token_count"] == 0
    assert usage_data["cache_creation_token_count"] == 150


# A LiteLLM gateway serving Claude: Anthropic reported input 10, cache read
# 2000, cache write 500, output 20, and LiteLLM folded the cache into the
# prompt count. Copied verbatim from the bodies a LiteLLM proxy built (BACK-4240).
LITELLM_CLAUDE_CHAT_USAGE = {
    "cache_creation_input_tokens": 500, "cache_read_input_tokens": 2000, "completion_tokens": 20,
    "completion_tokens_details": {"reasoning_tokens": 0, "text_tokens": 20}, "prompt_tokens": 2510,
    "prompt_tokens_details": {"cache_creation_tokens": 500, "cache_write_tokens": 500, "cached_tokens": 2000,
                              "text_tokens": 10},
    "total_tokens": 2530,
}
LITELLM_CLAUDE_RESPONSES_USAGE = {
    "input_tokens": 2510, "input_tokens_details": {"cache_write_tokens": 500, "cached_tokens": 2000, "text_tokens": 10},
    "output_tokens": 20, "output_tokens_details": {"reasoning_tokens": 0, "text_tokens": 20}, "total_tokens": 2530,
}
GATEWAY_CLAUDE_MODEL = "claude-sonnet-4-5"


@pytest.fixture
def served_model(monkeypatch):
    """Make the fake server name the given model in its chat and Responses bodies."""
    def serve(model):
        monkeypatch.setattr(entry_points, "MODEL", model)
        monkeypatch.setattr(responses_stream_events, "MODEL", model)
    return serve


@pytest.mark.parametrize("call", [*NON_STREAMED, *STREAMED_CHAT, *STREAMED_RESPONSES], ids=lambda c: c.__name__)
def test_gateway_claude_record_counts_only_uncached_input(call, payloads, served_usage, served_model):
    served_usage(chat=LITELLM_CLAUDE_CHAT_USAGE, responses=LITELLM_CLAUDE_RESPONSES_USAGE)
    served_model(GATEWAY_CLAUDE_MODEL)

    call(FakeOpenAI(), None)

    record = the_one_record(payloads)
    assert record["model"] == GATEWAY_CLAUDE_MODEL
    assert record["input_token_count"] == 10
    assert record["cache_read_token_count"] == 2000
    assert record["cache_creation_token_count"] == 500
    assert record["output_token_count"] == 20
    assert record["total_token_count"] == 2530



# What e371ea7 recorded for the gateway bodies: chat kept the whole prompt,
# Responses took out only the OpenAI-style write.
INPUT_RECORDED_BEFORE = {sync_chat_create: 2510, sync_chat_create_stream: 2510, async_chat_create: 2510,
                         sync_responses_create: 2010, sync_responses_create_stream: 2010,
                         async_responses_create: 2010}
INPUT_RECORDED_BEFORE_CASES = sorted(INPUT_RECORDED_BEFORE, key=lambda c: c.__name__)


def _record_for_gateway_usage_served_as(model, call, payloads, served_usage, served_model):
    served_usage(chat=LITELLM_CLAUDE_CHAT_USAGE, responses=LITELLM_CLAUDE_RESPONSES_USAGE)
    served_model(model)

    call(FakeOpenAI(), None)

    return the_one_record(payloads)


@pytest.mark.parametrize("call", INPUT_RECORDED_BEFORE_CASES, ids=lambda c: c.__name__)
def test_gateway_shaped_usage_for_an_openai_model_is_recorded_as_before(call, payloads, served_usage,
                                                                         served_model):
    """Regression guard: only a model named Claude has its input netted here;
    the platform nets an OpenAI model's reads itself."""
    record = _record_for_gateway_usage_served_as(MODEL, call, payloads, served_usage, served_model)

    assert record["input_token_count"] == INPUT_RECORDED_BEFORE[call]
    assert record["cache_read_token_count"] == 2000
    assert record["cache_creation_token_count"] == 500


@pytest.mark.parametrize("call", INPUT_RECORDED_BEFORE_CASES, ids=lambda c: c.__name__)
def test_gateway_claude_under_an_alias_is_recorded_as_before(call, payloads, served_usage, served_model):
    """Known residual: a gateway that answers with the caller's alias for
    Claude cannot be told apart from any other model, so its input stays as
    before."""
    record = _record_for_gateway_usage_served_as("my-sonnet", call, payloads, served_usage, served_model)

    assert record["input_token_count"] == INPUT_RECORDED_BEFORE[call]
