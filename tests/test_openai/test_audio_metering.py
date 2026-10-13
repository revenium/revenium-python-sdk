"""OpenAI speech, transcription and translation calls each send one audio record (BACK-4026).

The real ``openai`` client runs over a stubbed transport (httpx or httpx2,
whichever the installed openai sends through); no call leaves the process. The
record is read where it reaches the metering client, ``client.ai.create_audio``.
"""
import asyncio
import json

import pytest
from openai import AsyncOpenAI, AzureOpenAI, OpenAI

import revenium_middleware.openai  # noqa: F401  (registers the wraps)
from revenium_middleware import JobContext, revenium_meter
from revenium_middleware._core import enforcement
from revenium_middleware._core.exceptions import BudgetExceededError
from revenium_middleware._core.metering_pool import wait_until_idle
from revenium_middleware.openai import middleware as mw

from .openai_transport import answer_every_transport, http_library_of

BASE_URL = "https://api.openai.test/v1"
AUDIO_FILE = ("clip.mp3", b"ID3-not-really-audio")
SPEECH_BYTES = bytes(range(256)) * 4
SPEECH_INPUT = "Hello from the audio metering test."
DURATION_SECONDS = 3.2
VERBOSE_DURATION_SECONDS = 7.5
TOKEN_USAGE = {"type": "tokens", "input_tokens": 20, "output_tokens": 5, "total_tokens": 25,
               "input_token_details": {"audio_tokens": 18, "text_tokens": 2}}
STREAM_EVENTS = [
    {"type": "transcript.text.delta", "delta": "hel"},
    {"type": "transcript.text.delta", "delta": "lo"},
    {"type": "transcript.text.done", "text": "hello", "usage": TOKEN_USAGE},
]


def _sse(events):
    return "".join(f"data: {json.dumps(event)}\n\n" for event in events).encode()


class FakeOpenAIAudio:
    """Answers the three audio endpoints; ``reply`` sets the transcription or translation body."""

    def __init__(self):
        self.requests = []
        self.reply = ("json", {"text": "hello", "usage": {"type": "duration", "seconds": DURATION_SECONDS}})

    def __call__(self, request):
        http = http_library_of(request)
        request.read()
        self.requests.append(request)
        if request.url.path.endswith("/audio/speech"):
            return http.Response(200, content=SPEECH_BYTES, headers={"content-type": "audio/mpeg"})
        kind, body = self.reply
        if kind == "text":
            return http.Response(200, content=body.encode(), headers={"content-type": "text/plain"})
        if kind == "sse":
            return http.Response(200, content=_sse(body), headers={"content-type": "text/event-stream"})
        return http.Response(200, json=body)


@pytest.fixture
def fake(monkeypatch):
    stub = FakeOpenAIAudio()
    answer_every_transport(monkeypatch, stub)
    return stub


def sync_client():
    return OpenAI(api_key="sk-test", base_url=BASE_URL, max_retries=0)


def async_client():
    return AsyncOpenAI(api_key="sk-test", base_url=BASE_URL, max_retries=0)


def audio_records(revenium_client):
    wait_until_idle(10)
    revenium_client.ai.create_completion.assert_not_called()
    records = [recorded.kwargs for recorded in revenium_client.ai.create_audio.call_args_list]
    for record in records:
        assert all(value is not None for value in record.values()), record
    return records


def assert_stored_at_zero_seconds(record):
    """Revenium drops an audio record with no quantity; zero seconds stores it at no cost."""
    assert record["duration_seconds"] == 0.0
    assert "billing_unit" not in record


def one_record(revenium_client):
    records = audio_records(revenium_client)
    assert len(records) == 1, records
    return records[0]


def assert_audio_record(record, model, subtype, provider="OPENAI"):
    assert record["model"] == model
    assert record["provider"] == provider
    assert record["operation_type"] == "AUDIO"
    assert record["operation_subtype"] == subtype
    assert record["middleware_source"] == "PYTHON"
    assert record["transaction_id"]


def test_transcription_call_sends_one_audio_record(fake, mock_revenium_client):
    transcription = sync_client().audio.transcriptions.create(model="whisper-1", file=AUDIO_FILE)

    assert transcription.text == "hello"
    record = one_record(mock_revenium_client)
    assert_audio_record(record, "whisper-1", "transcription")
    assert record["duration_seconds"] == DURATION_SECONDS
    assert "billing_unit" not in record
    assert record["response_format"] == "json"


def test_translation_call_sends_one_audio_record(fake, mock_revenium_client):
    fake.reply = ("json", {"text": "hello"})

    translation = sync_client().audio.translations.create(model="whisper-1", file=AUDIO_FILE)

    assert translation.text == "hello"
    record = one_record(mock_revenium_client)
    assert_audio_record(record, "whisper-1", "translation")
    assert record["target_language"] == "en"
    assert_stored_at_zero_seconds(record)


def test_speech_call_sends_one_audio_record(fake, mock_revenium_client):
    speech = sync_client().audio.speech.create(model="tts-1", voice="alloy", input=SPEECH_INPUT, speed=1.25)

    assert speech.content == SPEECH_BYTES
    record = one_record(mock_revenium_client)
    assert_audio_record(record, "tts-1", "tts")
    assert record["character_count"] == len(SPEECH_INPUT)
    assert record["billing_unit"] == "PER_CHARACTER"
    assert record["voice"] == "alloy"
    assert record["speed"] == 1.25
    assert record["response_format"] == "mp3"


def test_async_transcription_translation_and_speech_each_send_one_audio_record(fake, mock_revenium_client):
    async def go():
        client = async_client()
        transcription = await client.audio.transcriptions.create(model="whisper-1", file=AUDIO_FILE)
        translation = await client.audio.translations.create(model="whisper-1", file=AUDIO_FILE)
        speech = await client.audio.speech.create(model="tts-1", voice="alloy", input=SPEECH_INPUT)
        return transcription, translation, speech

    transcription, translation, speech = asyncio.run(go())

    assert transcription.text == translation.text == "hello"
    assert speech.content == SPEECH_BYTES
    records = audio_records(mock_revenium_client)
    assert sorted(record["operation_subtype"] for record in records) == ["transcription", "translation", "tts"]
    by_subtype = {record["operation_subtype"]: record for record in records}
    assert by_subtype["transcription"]["duration_seconds"] == DURATION_SECONDS
    assert by_subtype["translation"]["target_language"] == "en"
    assert by_subtype["tts"]["character_count"] == len(SPEECH_INPUT)
    assert len({record["transaction_id"] for record in records}) == 3


def test_verbose_json_sends_the_billed_seconds_not_the_audio_length(fake, mock_revenium_client):
    fake.reply = ("json", {"text": "hello", "language": "english", "duration": VERBOSE_DURATION_SECONDS,
                           "usage": {"type": "duration", "seconds": DURATION_SECONDS}})

    sync_client().audio.transcriptions.create(model="whisper-1", file=AUDIO_FILE, response_format="verbose_json")

    record = one_record(mock_revenium_client)
    assert record["duration_seconds"] == DURATION_SECONDS
    assert record["language"] == "english"
    assert record["response_format"] == "verbose_json"
    assert "billing_unit" not in record


def test_async_verbose_json_sends_the_billed_seconds_not_the_audio_length(fake, mock_revenium_client):
    fake.reply = ("json", {"text": "hello", "duration": VERBOSE_DURATION_SECONDS,
                           "usage": {"type": "duration", "seconds": DURATION_SECONDS}})

    asyncio.run(async_client().audio.transcriptions.create(model="whisper-1", file=AUDIO_FILE,
                                                           response_format="verbose_json"))

    assert one_record(mock_revenium_client)["duration_seconds"] == DURATION_SECONDS


def test_verbose_json_translation_without_usage_sends_the_audio_length(fake, mock_revenium_client):
    fake.reply = ("json", {"text": "hello", "language": "english", "duration": VERBOSE_DURATION_SECONDS})

    sync_client().audio.translations.create(model="whisper-1", file=AUDIO_FILE, response_format="verbose_json")

    record = one_record(mock_revenium_client)
    assert record["duration_seconds"] == VERBOSE_DURATION_SECONDS
    assert "billing_unit" not in record


def test_token_billed_transcription_sends_its_tokens(fake, mock_revenium_client):
    fake.reply = ("json", {"text": "hello", "usage": TOKEN_USAGE})

    sync_client().audio.transcriptions.create(model="gpt-4o-mini-transcribe", file=AUDIO_FILE)

    record = one_record(mock_revenium_client)
    assert_audio_record(record, "gpt-4o-mini-transcribe", "transcription")
    assert record["input_token_count"] == 20
    assert record["output_token_count"] == 5
    assert record["input_audio_token_count"] == 18
    assert "output_audio_token_count" not in record
    assert record["billing_unit"] == "PER_TOKEN"
    assert "duration_seconds" not in record


def test_text_format_transcription_sends_one_record_without_duration(fake, mock_revenium_client):
    fake.reply = ("text", "hello\n")

    text = sync_client().audio.transcriptions.create(model="whisper-1", file=AUDIO_FILE, response_format="text")

    assert text == "hello\n"
    record = one_record(mock_revenium_client)
    assert_audio_record(record, "whisper-1", "transcription")
    assert record["response_format"] == "text"
    assert_stored_at_zero_seconds(record)


@pytest.mark.parametrize("response_format", ["srt", "vtt"])
def test_subtitle_format_transcription_is_stored_at_zero_seconds(fake, mock_revenium_client, response_format):
    fake.reply = ("text", "1\n00:00:00,000 --> 00:00:01,000\nhello\n")

    sync_client().audio.transcriptions.create(model="whisper-1", file=AUDIO_FILE, response_format=response_format)

    record = one_record(mock_revenium_client)
    assert record["response_format"] == response_format
    assert_stored_at_zero_seconds(record)


def test_async_text_transcription_and_json_translation_are_stored_at_zero_seconds(fake, mock_revenium_client):
    async def go():
        client = async_client()
        fake.reply = ("text", "hello\n")
        await client.audio.transcriptions.create(model="whisper-1", file=AUDIO_FILE, response_format="text")
        fake.reply = ("json", {"text": "hello"})
        await client.audio.translations.create(model="whisper-1", file=AUDIO_FILE)

    asyncio.run(go())

    records = audio_records(mock_revenium_client)
    assert sorted(record["operation_subtype"] for record in records) == ["transcription", "translation"]
    for record in records:
        assert_stored_at_zero_seconds(record)


def test_streaming_response_transcription_body_is_left_unread_and_stored_at_zero_seconds(
        fake, mock_revenium_client):
    with sync_client().audio.transcriptions.with_streaming_response.create(
            model="whisper-1", file=AUDIO_FILE) as response:
        body = response.json()

    assert body["text"] == "hello"
    assert_stored_at_zero_seconds(one_record(mock_revenium_client))


def test_async_streaming_response_translation_body_is_left_unread_and_stored_at_zero_seconds(
        fake, mock_revenium_client):
    fake.reply = ("json", {"text": "hello"})

    async def go():
        async with async_client().audio.translations.with_streaming_response.create(
                model="whisper-1", file=AUDIO_FILE) as response:
            return await response.json()

    assert asyncio.run(go())["text"] == "hello"
    assert_stored_at_zero_seconds(one_record(mock_revenium_client))


def test_streamed_transcription_sends_one_record_after_the_stream_ends(fake, mock_revenium_client):
    fake.reply = ("sse", STREAM_EVENTS)

    stream = sync_client().audio.transcriptions.create(model="gpt-4o-mini-transcribe", file=AUDIO_FILE,
                                                       stream=True)
    events = iter(stream)
    first = next(events)
    assert audio_records(mock_revenium_client) == []
    rest = list(events)

    assert [event.type for event in [first, *rest]] == [event["type"] for event in STREAM_EVENTS]
    record = one_record(mock_revenium_client)
    assert_audio_record(record, "gpt-4o-mini-transcribe", "transcription")
    assert record["input_token_count"] == 20
    assert record["input_audio_token_count"] == 18
    assert record["billing_unit"] == "PER_TOKEN"


def test_async_streamed_transcription_sends_one_record_after_the_stream_ends(fake, mock_revenium_client):
    fake.reply = ("sse", STREAM_EVENTS)

    async def go():
        stream = await async_client().audio.transcriptions.create(model="gpt-4o-mini-transcribe",
                                                                  file=AUDIO_FILE, stream=True)
        async with stream:
            return [event.type async for event in stream]

    assert asyncio.run(go()) == [event["type"] for event in STREAM_EVENTS]
    record = one_record(mock_revenium_client)
    assert record["output_token_count"] == 5


def test_raw_response_transcription_sends_its_duration(fake, mock_revenium_client):
    raw = sync_client().audio.transcriptions.with_raw_response.create(model="whisper-1", file=AUDIO_FILE)

    assert raw.parse().text == "hello"
    record = one_record(mock_revenium_client)
    assert record["duration_seconds"] == DURATION_SECONDS


def test_raw_response_streamed_transcription_events_reach_the_caller(fake, mock_revenium_client):
    fake.reply = ("sse", STREAM_EVENTS)

    raw = sync_client().audio.transcriptions.with_raw_response.create(model="gpt-4o-mini-transcribe",
                                                                      file=AUDIO_FILE, stream=True)

    assert [event.type for event in raw.parse()] == [event["type"] for event in STREAM_EVENTS]
    record = one_record(mock_revenium_client)
    assert_audio_record(record, "gpt-4o-mini-transcribe", "transcription")
    assert_stored_at_zero_seconds(record)


def test_raw_response_speech_is_returned_unchanged_and_metered_once(fake, mock_revenium_client):
    raw = sync_client().audio.speech.with_raw_response.create(model="tts-1", voice="alloy", input=SPEECH_INPUT)

    assert raw.content == SPEECH_BYTES
    assert raw.parse().content == SPEECH_BYTES
    record = one_record(mock_revenium_client)
    assert_audio_record(record, "tts-1", "tts")
    assert record["character_count"] == len(SPEECH_INPUT)


def test_streaming_response_speech_body_is_still_readable_and_metered_once(fake, mock_revenium_client):
    with sync_client().audio.speech.with_streaming_response.create(
            model="tts-1", voice="alloy", input=SPEECH_INPUT) as response:
        body = b"".join(response.iter_bytes())

    assert body == SPEECH_BYTES
    record = one_record(mock_revenium_client)
    assert_audio_record(record, "tts-1", "tts")


def test_usage_metadata_is_removed_from_the_request_and_attributed(fake, mock_revenium_client):
    usage_metadata = {"subscriber": {"id": "sub-audio"}, "organization_name": "org-audio",
                      "trace_id": "trace-audio", "task_type": "meeting-notes"}

    sync_client().audio.transcriptions.create(model="whisper-1", file=AUDIO_FILE, usage_metadata=usage_metadata)
    sync_client().audio.speech.create(model="tts-1", voice="alloy", input=SPEECH_INPUT,
                                      usage_metadata=usage_metadata)

    assert all(b"usage_metadata" not in request.content for request in fake.requests)
    assert all(b"sub-audio" not in request.content for request in fake.requests)
    records = audio_records(mock_revenium_client)
    assert len(records) == 2
    for record in records:
        assert record["subscriber"]["id"] == "sub-audio"
        assert record["organization_name"] == "org-audio"
        assert record["trace_id"] == "trace-audio"
        assert record["task_type"] == "meeting-notes"


def test_selective_metering_leaves_audio_unmetered(fake, mock_revenium_client, monkeypatch):
    monkeypatch.setattr(mw, "is_selective_metering_enabled", lambda: True)

    sync_client().audio.transcriptions.create(model="whisper-1", file=AUDIO_FILE)
    assert audio_records(mock_revenium_client) == []

    @revenium_meter()
    def transcribe():
        return sync_client().audio.transcriptions.create(model="whisper-1", file=AUDIO_FILE)

    transcribe()
    assert_audio_record(one_record(mock_revenium_client), "whisper-1", "transcription")


def test_azure_client_audio_call_is_labelled_azure(fake, mock_revenium_client):
    client = AzureOpenAI(api_key="azure-test", azure_endpoint="https://revenium-test.openai.azure.com",
                         api_version="2024-10-21", max_retries=0)

    client.audio.transcriptions.create(model="whisper-1", file=AUDIO_FILE)

    record = one_record(mock_revenium_client)
    assert record["provider"] == "Azure"
    assert record["operation_subtype"] == "transcription"


def test_budget_exceeded_blocks_the_audio_call_before_the_provider(fake, mock_revenium_client, monkeypatch):
    monkeypatch.setenv("REVENIUM_CIRCUIT_BREAKER_ENABLED", "true")
    monkeypatch.delenv("REVENIUM_BYPASS", raising=False)
    monkeypatch.setattr(enforcement, "_cached_rules", [{"name": "audio-cap", "breached": True}])
    monkeypatch.setattr(enforcement, "_cache_timestamp", float("inf"))
    monkeypatch.setattr(enforcement, "_cache_initialized", True)
    monkeypatch.setattr(enforcement, "_ensure_poller_running", lambda: None)
    monkeypatch.setattr(enforcement, "_fetch_rules", lambda: None)

    with pytest.raises(BudgetExceededError):
        sync_client().audio.speech.create(model="tts-1", voice="alloy", input=SPEECH_INPUT)

    assert fake.requests == []
    assert audio_records(mock_revenium_client) == []


def test_litellm_transcription_through_openai_sends_one_record(fake, mock_revenium_client, monkeypatch):
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    litellm = pytest.importorskip("litellm")
    import revenium_middleware.litellm  # noqa: F401  (client mode)

    response = litellm.transcription(model="whisper-1", file=AUDIO_FILE, client=sync_client(), api_key="sk-test")

    assert response.text == "hello"
    record = one_record(mock_revenium_client)
    assert_audio_record(record, "whisper-1", "transcription")
    assert record["duration_seconds"] == DURATION_SECONDS


def azure_client():
    return AzureOpenAI(api_key="azure-test", azure_endpoint="https://revenium-test.openai.azure.com",
                       api_version="2024-10-21", max_retries=0)


@pytest.fixture
def clean_azure_model_cache():
    from revenium_middleware.openai import azure_model_resolver

    azure_model_resolver._azure_model_cache.clear()
    yield
    azure_model_resolver._azure_model_cache.clear()


@pytest.mark.parametrize("deployment,model", [
    ("prod-gpt-4o-mini-transcribe", "gpt-4o-mini-transcribe"),
    ("my-gpt-4o-transcribe", "gpt-4o-transcribe"),
    ("prod-gpt-4o-transcribe-diarize", "gpt-4o-transcribe-diarize"),
])
def test_azure_transcription_deployment_resolves_to_the_audio_model(
        fake, mock_revenium_client, clean_azure_model_cache, deployment, model):
    azure_client().audio.transcriptions.create(model=deployment, file=AUDIO_FILE)

    assert_audio_record(one_record(mock_revenium_client), model, "transcription", provider="Azure")


def test_azure_speech_deployment_resolves_to_the_audio_model(fake, mock_revenium_client, clean_azure_model_cache):
    azure_client().audio.speech.create(model="prod-gpt-4o-mini-tts", voice="alloy", input=SPEECH_INPUT)

    assert_audio_record(one_record(mock_revenium_client), "gpt-4o-mini-tts", "tts", provider="Azure")


def test_azure_chat_deployment_still_resolves_to_the_chat_model(clean_azure_model_cache):
    from revenium_middleware.openai.azure_model_resolver import resolve_azure_model_name

    assert resolve_azure_model_name("prod-gpt-4o-mini", None, {}) == "gpt-4o-mini"
    assert resolve_azure_model_name("prod-gpt-4o", None, {}) == "gpt-4o"


JOB_FIELDS = {"agenticJobId": "job-stream-1", "agenticJobName": "Stream Job"}


def test_streamed_transcription_keeps_the_job_active_when_the_call_began(fake, mock_revenium_client):
    fake.reply = ("sse", STREAM_EVENTS)

    with JobContext(job_id="job-stream-1", name="Stream Job"):
        stream = sync_client().audio.transcriptions.create(model="gpt-4o-mini-transcribe", file=AUDIO_FILE,
                                                           stream=True)
    with JobContext(job_id="job-other"):
        list(stream)

    extra_body = one_record(mock_revenium_client)["extra_body"]
    assert {key: extra_body[key] for key in JOB_FIELDS} == JOB_FIELDS
    assert "job-other" not in json.dumps(extra_body)


def test_async_streamed_transcription_keeps_the_job_active_when_the_call_began(fake, mock_revenium_client):
    fake.reply = ("sse", STREAM_EVENTS)

    async def go():
        with JobContext(job_id="job-stream-1", name="Stream Job"):
            stream = await async_client().audio.transcriptions.create(
                model="gpt-4o-mini-transcribe", file=AUDIO_FILE, stream=True)
        with JobContext(job_id="job-other"):
            return [event async for event in stream]

    asyncio.run(go())
    extra_body = one_record(mock_revenium_client)["extra_body"]
    assert {key: extra_body[key] for key in JOB_FIELDS} == JOB_FIELDS


class CloseSpy:
    def __init__(self, stream):
        self.close_calls = 0
        original = stream.close

        def close():
            self.close_calls += 1
            return original()
        stream.close = close


class AsyncCloseSpy:
    def __init__(self, stream):
        self.close_calls = 0
        original = stream.close

        async def close():
            self.close_calls += 1
            return await original()
        stream.close = close


def open_stream(fake):
    fake.reply = ("sse", STREAM_EVENTS)
    stream = sync_client().audio.transcriptions.create(model="gpt-4o-mini-transcribe", file=AUDIO_FILE,
                                                       stream=True)
    return stream, CloseSpy(stream._stream)


def test_closing_a_transcription_stream_early_sends_one_record_and_closes_the_provider(fake, mock_revenium_client):
    stream, spy = open_stream(fake)
    next(stream)

    stream.close()

    assert spy.close_calls == 1
    record = one_record(mock_revenium_client)
    assert_audio_record(record, "gpt-4o-mini-transcribe", "transcription")
    assert_stored_at_zero_seconds(record)


def test_closing_a_transcription_stream_twice_sends_one_record_and_closes_once(fake, mock_revenium_client):
    stream, spy = open_stream(fake)
    next(stream)

    stream.close()
    stream.close()

    assert spy.close_calls == 1
    one_record(mock_revenium_client)


def test_leaving_a_transcription_stream_context_early_sends_one_record_and_closes_the_provider(
        fake, mock_revenium_client):
    stream, spy = open_stream(fake)

    with stream:
        next(stream)

    assert spy.close_calls == 1
    one_record(mock_revenium_client)


def test_async_closing_a_transcription_stream_early_sends_one_record_and_closes_the_provider(
        fake, mock_revenium_client):
    fake.reply = ("sse", STREAM_EVENTS)

    async def go():
        stream = await async_client().audio.transcriptions.create(model="gpt-4o-mini-transcribe",
                                                                  file=AUDIO_FILE, stream=True)
        spy = AsyncCloseSpy(stream._stream)
        await stream.__anext__()
        await stream.close()
        return spy

    assert asyncio.run(go()).close_calls == 1
    assert_stored_at_zero_seconds(one_record(mock_revenium_client))


def test_async_closing_a_transcription_stream_twice_sends_one_record_and_closes_once(fake, mock_revenium_client):
    fake.reply = ("sse", STREAM_EVENTS)

    async def go():
        stream = await async_client().audio.transcriptions.create(model="gpt-4o-mini-transcribe",
                                                                  file=AUDIO_FILE, stream=True)
        spy = AsyncCloseSpy(stream._stream)
        await stream.__anext__()
        await stream.close()
        await stream.close()
        return spy

    assert asyncio.run(go()).close_calls == 1
    one_record(mock_revenium_client)


def test_async_leaving_a_transcription_stream_context_early_sends_one_record_and_closes_the_provider(
        fake, mock_revenium_client):
    fake.reply = ("sse", STREAM_EVENTS)

    async def go():
        stream = await async_client().audio.transcriptions.create(model="gpt-4o-mini-transcribe",
                                                                  file=AUDIO_FILE, stream=True)
        spy = AsyncCloseSpy(stream._stream)
        async with stream:
            await stream.__anext__()
        return spy

    assert asyncio.run(go()).close_calls == 1
    one_record(mock_revenium_client)


def test_audio_calls_pass_their_model_to_the_cap_check(fake, mock_revenium_client, monkeypatch):
    """A cap filtered to one model or provider is judged on the audio call's own (BACK-4176)."""
    from revenium_middleware.openai import audio

    seen = []

    def record(usage_metadata=None, **keywords):
        seen.append(keywords)
        raise BudgetExceededError(message="recorded", rule_name="audio-wiring")

    monkeypatch.setenv("REVENIUM_CIRCUIT_BREAKER_ENABLED", "true")
    monkeypatch.setattr(audio, "check_enforcement", record)

    with pytest.raises(BudgetExceededError):
        sync_client().audio.speech.create(model="tts-1", voice="alloy", input=SPEECH_INPUT)

    assert seen == [{"model": "tts-1", "provider": "OPENAI"}]
    assert fake.requests == []


def test_an_unmetered_azure_audio_deployment_is_held_by_a_breached_model_cap(fake, mock_revenium_client, monkeypatch):
    monkeypatch.setattr(mw, "is_selective_metering_enabled", lambda: True)
    monkeypatch.setenv("REVENIUM_CIRCUIT_BREAKER_ENABLED", "true")
    monkeypatch.delenv("REVENIUM_BYPASS", raising=False)
    monkeypatch.setattr(enforcement, "_cached_rules", [{
        "name": "whisper cap", "action": "BLOCK", "breached": True, "shadowMode": False,
        "filters": [{"dimension": "MODEL", "operator": "IS", "value": "whisper-1"}],
    }])
    monkeypatch.setattr(enforcement, "_cache_timestamp", float("inf"))
    monkeypatch.setattr(enforcement, "_cache_initialized", True)
    monkeypatch.setattr(enforcement, "_ensure_poller_running", lambda: None)
    monkeypatch.setattr(enforcement, "_fetch_rules", lambda: None)
    client = AzureOpenAI(api_key="azure-test", azure_endpoint="https://revenium-test.openai.azure.com",
                         api_version="2024-10-21", max_retries=0)

    with pytest.raises(BudgetExceededError):
        client.audio.transcriptions.create(model="prod-whisper", file=AUDIO_FILE)

    assert fake.requests == []
    assert audio_records(mock_revenium_client) == []
