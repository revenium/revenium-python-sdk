"""The same metadata reaches the record on every provider the SDK wraps.

Each row is one stubbed call through a real wrapped client (the entry-point
matrix stubs), recorded at the metering client. Vertex rows live in
tests/test_google/test_vertex_ai_async_and_chat.py, which runs where vertexai
is installed.
"""
import asyncio
import datetime
import logging
from types import SimpleNamespace

import pytest

import revenium_middleware.anthropic  # noqa: F401  installs the Anthropic patches
import revenium_middleware.fal  # noqa: F401  installs the fal patches
import revenium_middleware.google  # noqa: F401  installs the google-genai patches
import revenium_middleware.litellm  # noqa: F401  installs the LiteLLM client patches
import revenium_middleware.ollama  # noqa: F401  installs the Ollama patches
import revenium_middleware.openai  # noqa: F401  installs the OpenAI patches
import revenium_middleware.perplexity  # noqa: F401  installs the Perplexity patches
from revenium_middleware._core.call_ownership import (
    GOOGLE, LITELLM, _published_in_context, publish_callback_metadata, reset_claimed_response_ids,
)
from revenium_middleware._core.decorators import revenium_metadata
from revenium_middleware._core.trace_fields import get_retry_number
from revenium_middleware.anthropic import bedrock_transport
from revenium_middleware.anthropic import middleware as anthropic_middleware
from revenium_middleware.anthropic.bedrock_adapter import BedrockStreamWrapper
from revenium_middleware.fal._metering import handle_metering as fal_handle_metering
from entry_points import calls
from test_anthropic.test_bedrock_transport import (
    CLAUDE_PROFILE, converse_response, invoke_model_response, make_instance,
)

OPERATIONS = ("completion", "image", "video", "audio")


@pytest.fixture(autouse=True)
def _isolated(monkeypatch):
    monkeypatch.delenv("REVENIUM_SELECTIVE_METERING", raising=False)
    monkeypatch.setenv("REVENIUM_BEDROCK_TRANSPORT", "1")
    reset_claimed_response_ids()
    publications = _published_in_context.set(())
    yield
    _published_in_context.reset(publications)
    reset_claimed_response_ids()


@pytest.fixture
def records(mock_revenium_client):
    def recorded():
        calls.wait_for_metering()
        return [entry.kwargs for operation in OPERATIONS
                for entry in getattr(mock_revenium_client.ai, f"create_{operation}").call_args_list]

    return recorded


def _with(usage_metadata):
    return {"usage_metadata": usage_metadata} if usage_metadata is not None else {}


def _openai(usage_metadata):
    calls._openai().chat.completions.create(**calls._chat_kwargs(), **_with(usage_metadata))


def _openai_stream(usage_metadata):
    for _chunk in calls._openai().chat.completions.create(**calls._chat_kwargs(), stream=True,
                                                          **_with(usage_metadata)):
        pass


def _openai_responses(usage_metadata):
    calls._openai().responses.create(model=calls.OPENAI_MODEL, input="hi", **_with(usage_metadata))


def _anthropic(usage_metadata):
    calls._anthropic().messages.create(**calls._ANTHROPIC_KWARGS, **_with(usage_metadata))


def _anthropic_stream(usage_metadata):
    for _event in calls._anthropic().messages.create(**calls._ANTHROPIC_KWARGS, stream=True,
                                                     **_with(usage_metadata)):
        pass


def _anthropic_stream_helper(usage_metadata):
    calls._drain_sync(calls._anthropic().messages.stream(**calls._ANTHROPIC_KWARGS, **_with(usage_metadata)))


def _anthropic_beta(usage_metadata):
    calls._anthropic().beta.messages.create(**calls._ANTHROPIC_KWARGS, **_with(usage_metadata))


def _gemini(usage_metadata):
    calls._genai().models.generate_content(model=calls.GEMINI_MODEL, contents="hi", **_with(usage_metadata))


def _gemini_async(usage_metadata):
    asyncio.run(calls._genai().aio.models.generate_content(
        model=calls.GEMINI_MODEL, contents="hi", **_with(usage_metadata)))


def _gemini_stream(usage_metadata):
    for _chunk in calls._genai().models.generate_content_stream(
            model=calls.GEMINI_MODEL, contents="hi", **_with(usage_metadata)):
        pass


def _gemini_embed(usage_metadata):
    calls._genai().models.embed_content(model=calls.GEMINI_EMBED_MODEL, contents="hi", **_with(usage_metadata))


def _litellm(usage_metadata):
    import litellm
    litellm.completion(**calls._litellm_kwargs(), **_with(usage_metadata))


def _litellm_stream(usage_metadata):
    import litellm
    for _chunk in litellm.completion(**calls._litellm_kwargs(), stream=True, **_with(usage_metadata)):
        pass


def _ollama(usage_metadata):
    calls._ollama_client().chat(model=calls.OLLAMA_MODEL, messages=calls.USER_MESSAGES, **_with(usage_metadata))


def _perplexity_native(usage_metadata):
    calls._perplexity_native().chat.completions.create(
        model=calls.PERPLEXITY_MODEL, messages=calls.USER_MESSAGES, **_with(usage_metadata))


def _perplexity_compatible(usage_metadata):
    client = calls._openai(base_url=calls.PERPLEXITY_BASE_URL, model=calls.PERPLEXITY_MODEL,
                           usage=calls._PERPLEXITY_USAGE)
    client.chat.completions.create(**calls._chat_kwargs(calls.PERPLEXITY_MODEL), **_with(usage_metadata))


def _fal_image(usage_metadata):
    import fal_client
    with calls._fal_key(), calls._stubbed_http(fal_client.sync_client, calls._fal_sync_http()):
        fal_client.run(calls.FAL_APP, arguments=calls._FAL_ARGUMENTS, **_with(usage_metadata))


def _bedrock_transport(usage_metadata):
    token = anthropic_middleware.usage_context.set(usage_metadata or {})
    try:
        bedrock_transport._make_api_call_wrapper(
            lambda *args, **kwargs: converse_response(), make_instance(),
            ("Converse", {"modelId": CLAUDE_PROFILE}), {})
    finally:
        anthropic_middleware.usage_context.reset(token)


def _bedrock_transport_invoke(usage_metadata):
    token = anthropic_middleware.usage_context.set(usage_metadata or {})
    try:
        bedrock_transport._make_api_call_wrapper(
            lambda *args, **kwargs: invoke_model_response(), make_instance(),
            ("InvokeModel", {"modelId": CLAUDE_PROFILE}), {})
    finally:
        anthropic_middleware.usage_context.reset(token)


def _gemini_image(usage_metadata):
    calls._genai_vertex_express().models.generate_images(
        model=calls.IMAGEN_MODEL, prompt="a cat", **_with(usage_metadata))


def _fal_completion(usage_metadata):
    fal_handle_metering(
        "openrouter/router", {"prompt": "hi"},
        {"output": "hi", "usage": {"prompt_tokens": calls.INPUT_TOKENS, "completion_tokens": calls.OUTPUT_TOKENS}},
        datetime.datetime.now(datetime.timezone.utc), usage_metadata or {}, "fal-completion-1")


def _bedrock_stream(usage_metadata):
    stream = BedrockStreamWrapper(model=CLAUDE_PROFILE, payload={}, region="us-east-1",
                                  usage_metadata=usage_metadata)
    stream.response_id = "bedrock-stream-1"
    stream.response_time = stream.request_time
    stream.final_message = SimpleNamespace(
        model=CLAUDE_PROFILE,
        usage=SimpleNamespace(input_tokens=calls.INPUT_TOKENS, output_tokens=calls.OUTPUT_TOKENS))
    stream._send_metering_data(1.0)


DECORATOR_ROWS = [
    pytest.param(_openai, id="openai"),
    pytest.param(_anthropic, id="anthropic"),
    pytest.param(_gemini, id="gemini"),
    pytest.param(_gemini_async, id="gemini-async"),
    pytest.param(_gemini_stream, id="gemini-stream"),
    pytest.param(_gemini_embed, id="gemini-embed"),
    pytest.param(_litellm, id="litellm"),
    pytest.param(_litellm_stream, id="litellm-stream"),
    pytest.param(_ollama, id="ollama"),
    pytest.param(_perplexity_native, id="perplexity-native"),
    pytest.param(_perplexity_compatible, id="perplexity-compatible"),
    pytest.param(_fal_image, id="fal"),
    pytest.param(_bedrock_transport, id="bedrock-transport"),
]


@pytest.mark.parametrize("make_call", DECORATOR_ROWS)
def test_revenium_metadata_reaches_every_provider_record(records, make_call):
    @revenium_metadata(organizationName="DecOrg", trace_id="dec-trace")
    def decorated():
        make_call(None)

    decorated()

    [record] = records()
    assert record.get("organization_name") == "DecOrg"
    assert record.get("trace_id") == "dec-trace"


@pytest.mark.parametrize("make_call, provider", [
    pytest.param(_gemini, GOOGLE, id="gemini"),
    pytest.param(_litellm, LITELLM, id="litellm"),
])
@pytest.mark.parametrize("call_metadata, expected", [
    pytest.param(None, "Cb", id="callback-beats-decorator"),
    pytest.param({"organizationName": "Call"}, "Call", id="call-beats-callback"),
])
def test_decorator_sits_under_the_callback_and_the_call(records, make_call, provider, call_metadata, expected):
    @revenium_metadata(organizationName="Dec")
    def decorated():
        publish_callback_metadata({"organizationName": "Cb"}, providers=frozenset({provider}))
        make_call(call_metadata)

    decorated()

    [record] = records()
    assert record.get("organization_name") == expected


PER_CALL_TRACE = {
    "environment": "call-env",
    "region": "call-region",
    "credential_alias": "call-alias",
    "trace_type": "call-type",
    "trace_name": "Call Trace",
    "parent_transaction_id": "call-parent",
    "transaction_name": "Generate Response",
    "retry_number": 3,
}
PER_CALL_SKILL = {"skill_name": "md-skill", "skill_source": "plugin"}

COMPLETION_ROWS = [
    pytest.param(_openai, id="openai"),
    pytest.param(_anthropic, id="anthropic"),
    pytest.param(_gemini, id="gemini"),
    pytest.param(_litellm, id="litellm"),
    pytest.param(_ollama, id="ollama"),
    pytest.param(_perplexity_native, id="perplexity-native"),
    pytest.param(_perplexity_compatible, id="perplexity-compatible"),
    pytest.param(_fal_completion, id="fal-completion"),
    pytest.param(_bedrock_transport, id="bedrock-transport"),
    pytest.param(_bedrock_stream, id="bedrock-stream"),
]

MEDIA_ROWS = [
    pytest.param(_gemini_image, id="gemini-image"),
    pytest.param(_fal_image, id="fal-image"),
]


@pytest.fixture
def environment_defaults(monkeypatch):
    for name, value in {
        "REVENIUM_ENVIRONMENT": "env-env",
        "REVENIUM_REGION": "env-region",
        "REVENIUM_CREDENTIAL_ALIAS": "env-alias",
        "REVENIUM_TRACE_TYPE": "env-type",
        "REVENIUM_TRACE_NAME": "Env Trace",
        "REVENIUM_PARENT_TRANSACTION_ID": "env-parent",
        "REVENIUM_TRANSACTION_NAME": "deployment-default",
        "REVENIUM_RETRY_NUMBER": "1",
        "REVENIUM_SKILL_NAME": "env-skill",
    }.items():
        monkeypatch.setenv(name, value)


def _fields(record, names):
    return {name: record.get(name) for name in names}


@pytest.mark.parametrize("make_call", COMPLETION_ROWS)
def test_per_call_trace_and_skill_fields_win_on_every_provider(records, environment_defaults, make_call):
    make_call({**PER_CALL_TRACE, **PER_CALL_SKILL})

    [record] = records()
    assert _fields(record, PER_CALL_TRACE) == PER_CALL_TRACE
    assert _fields(record, PER_CALL_SKILL) == PER_CALL_SKILL


@pytest.mark.parametrize("make_call", MEDIA_ROWS)
def test_per_call_trace_fields_win_on_media_records(records, environment_defaults, make_call):
    make_call(dict(PER_CALL_TRACE))

    [record] = records()
    assert _fields(record, PER_CALL_TRACE) == PER_CALL_TRACE


@pytest.mark.parametrize("usage_metadata, expected", [
    pytest.param({"retry_number": 0}, 0, id="int-zero"),
    pytest.param({"retry_number": "0"}, 0, id="digit-string-zero"),
    pytest.param({"retryNumber": 2}, 2, id="camel-case"),
    pytest.param({"retry_number": True}, 3, id="bool-falls-back"),
    pytest.param({"retry_number": "two"}, 3, id="unparseable-falls-back"),
    pytest.param({}, 3, id="absent"),
])
def test_per_call_retry_number_beats_the_environment(monkeypatch, usage_metadata, expected):
    monkeypatch.setenv("REVENIUM_RETRY_NUMBER", "3")
    assert get_retry_number(usage_metadata) == expected


@pytest.mark.parametrize("make_call", [
    pytest.param(_gemini, id="gemini"),
    pytest.param(_gemini_image, id="gemini-image"),
    pytest.param(_litellm, id="litellm"),
    pytest.param(_bedrock_transport, id="bedrock-transport"),
])
def test_a_per_call_retry_number_of_zero_reaches_the_record(records, monkeypatch, make_call):
    monkeypatch.setenv("REVENIUM_RETRY_NUMBER", "3")

    make_call({"retry_number": 0})

    [record] = records()
    assert record["retry_number"] == 0


SCORE_ZERO_FIXED_ROWS = [
    pytest.param(_openai, id="openai"),
    pytest.param(_gemini, id="gemini"),
    pytest.param(_perplexity_native, id="perplexity-native"),
    pytest.param(_perplexity_compatible, id="perplexity-compatible"),
]


@pytest.mark.parametrize("make_call", SCORE_ZERO_FIXED_ROWS + [
    pytest.param(_anthropic, id="anthropic"),
    pytest.param(_litellm, id="litellm"),
    pytest.param(_ollama, id="ollama"),
])
def test_quality_score_of_zero_reaches_every_provider_record(records, make_call):
    make_call({"response_quality_score": 0.0})

    [record] = records()
    assert "response_quality_score" in record
    assert record["response_quality_score"] == 0.0


@pytest.mark.parametrize("make_call", SCORE_ZERO_FIXED_ROWS)
def test_missing_quality_score_is_still_omitted(records, make_call):
    make_call({"trace_id": "no-score"})

    [record] = records()
    assert "response_quality_score" not in record


RETRY_NUMBER_ROWS = [
    pytest.param(_openai, id="openai"),
    pytest.param(_openai_stream, id="openai-stream"),
    pytest.param(_openai_responses, id="openai-responses"),
    pytest.param(_anthropic, id="anthropic"),
    pytest.param(_anthropic_stream, id="anthropic-stream"),
    pytest.param(_ollama, id="ollama"),
    pytest.param(_bedrock_transport, id="bedrock-transport"),
]


def _warnings_containing(caplog, text):
    return [record for record in caplog.records
            if record.levelno == logging.WARNING and text in record.getMessage()]


@pytest.mark.parametrize("make_call", RETRY_NUMBER_ROWS)
@pytest.mark.parametrize("usage_metadata, expected, warned", [
    pytest.param({"retry_number": "abc"}, 3, True, id="non-numeric-falls-back"),
    pytest.param({"retryNumber": "abc"}, 3, True, id="non-numeric-camel-falls-back"),
    pytest.param({"retry_number": True}, 3, True, id="bool-falls-back"),
    pytest.param({"retry_number": "5"}, 5, False, id="numeric-string-is-a-number"),
    pytest.param({"retry_number": None}, 3, False, id="explicit-none-falls-back"),
    pytest.param({"retry_number": 0}, 0, False, id="zero-is-kept"),
    pytest.param({}, 3, False, id="absent"),
])
def test_a_per_call_retry_number_is_checked_on_every_path(records, monkeypatch, caplog, make_call,
                                                          usage_metadata, expected, warned):
    monkeypatch.setenv("REVENIUM_RETRY_NUMBER", "3")

    with caplog.at_level(logging.WARNING):
        make_call(usage_metadata)

    [record] = records()
    assert record["retry_number"] == expected
    assert type(record["retry_number"]) is int
    assert len(_warnings_containing(caplog, "Invalid retry number")) == (1 if warned else 0)


TRACE_CHECK_ROWS = [
    pytest.param(_anthropic, id="anthropic"),
    pytest.param(_anthropic_stream, id="anthropic-stream"),
    pytest.param(_anthropic_stream_helper, id="anthropic-stream-helper"),
    pytest.param(_anthropic_beta, id="anthropic-beta"),
    pytest.param(_bedrock_transport, id="bedrock-transport"),
    pytest.param(_bedrock_transport_invoke, id="bedrock-transport-invoke"),
    pytest.param(_openai, id="openai"),
    pytest.param(_bedrock_stream, id="bedrock-stream"),
]


@pytest.mark.parametrize("make_call", TRACE_CHECK_ROWS)
@pytest.mark.parametrize("usage_metadata, expected, invalid_type_warnings", [
    pytest.param({"trace_type": "bad type!", "trace_name": "n" * 300},
                 {"trace_type": None, "trace_name": "n" * 256}, 1, id="invalid-type-long-name"),
    pytest.param({"traceType": "bad type!", "traceName": "n" * 300},
                 {"trace_type": None, "trace_name": "n" * 256}, 1, id="camel-invalid-type-long-name"),
    pytest.param({"trace_type": 5, "trace_name": 7},
                 {"trace_type": None, "trace_name": None}, 0, id="non-string-values"),
    pytest.param({"trace_type": "call-type", "trace_name": "Call Trace"},
                 {"trace_type": "call-type", "trace_name": "Call Trace"}, 0, id="valid-values-win"),
])
def test_a_bad_per_call_trace_type_is_left_off_and_a_long_trace_name_is_cut_on_every_path(
        records, environment_defaults, caplog, make_call, usage_metadata, expected, invalid_type_warnings):
    with caplog.at_level(logging.WARNING):
        make_call(usage_metadata)

    [record] = records()
    assert _fields(record, expected) == expected
    assert len(_warnings_containing(caplog, "traceType contains invalid characters")) == invalid_type_warnings


BLANK_SKILL_AND_JOB_ROWS = COMPLETION_ROWS + [
    pytest.param(_anthropic_stream, id="anthropic-stream"),
    pytest.param(_anthropic_beta, id="anthropic-beta"),
    pytest.param(_bedrock_transport_invoke, id="bedrock-transport-invoke"),
]


@pytest.mark.parametrize("make_call", BLANK_SKILL_AND_JOB_ROWS)
def test_a_blank_per_call_skill_or_job_value_keeps_the_environment_value(records, monkeypatch, make_call):
    monkeypatch.setenv("REVENIUM_SKILL_NAME", "env-skill")
    monkeypatch.setenv("REVENIUM_AGENTIC_JOB_ID", "env-job-1")

    make_call({"skill_name": "", "agentic_job_id": ""})

    [record] = records()
    assert record["skill_name"] == "env-skill"
    assert (record.get("extra_body") or {}).get("agenticJobId") == "env-job-1"
