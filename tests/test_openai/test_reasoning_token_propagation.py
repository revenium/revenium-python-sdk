import asyncio
import datetime
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from revenium_middleware import shutdown_event
from revenium_middleware.openai.middleware import (
    OperationType,
    _extract_reasoning_token_count,
    _wrap_async_stream,
    create_metering_call,
    extract_usage_data,
    handle_streaming_response,
    handle_streaming_responses,
    log_token_usage,
)

from .responses_stream_events import MODEL, typed_events, usage_payload


def _chat_response(reasoning_tokens=None, details_as_dict=False):
    """Chat Completions response; reasoning detail block omitted when None."""
    usage = SimpleNamespace(
        prompt_tokens=100,
        completion_tokens=25,
        total_tokens=125,
        prompt_tokens_details=SimpleNamespace(cached_tokens=0),
    )
    if reasoning_tokens is not None:
        if details_as_dict:
            usage.completion_tokens_details = {"reasoning_tokens": reasoning_tokens}
        else:
            usage.completion_tokens_details = SimpleNamespace(
                reasoning_tokens=reasoning_tokens
            )
    return SimpleNamespace(
        id="chatcmpl-reasoning-test",
        model="o4-mini",
        usage=usage,
        choices=[SimpleNamespace(finish_reason="stop")],
        system_fingerprint=None,
    )


def _responses_response(reasoning_tokens=None):
    """Responses API response; reasoning lives under output_tokens_details."""
    usage = SimpleNamespace(
        input_tokens=80,
        output_tokens=20,
        total_tokens=100,
        input_tokens_details=SimpleNamespace(cached_tokens=0),
    )
    if reasoning_tokens is not None:
        usage.output_tokens_details = SimpleNamespace(
            reasoning_tokens=reasoning_tokens
        )
    return SimpleNamespace(
        id="resp-reasoning-test",
        model="o4-mini",
        usage=usage,
        system_fingerprint=None,
    )


def _run_coro_in_thread(coro):
    error = []

    def target():
        try:
            asyncio.run(coro)
        except Exception as exc:
            error.append(exc)

    thread = threading.Thread(target=target)
    thread.start()
    thread.join()
    if error:
        raise error[0]
    return MagicMock()


@pytest.fixture(autouse=True)
def reset_shutdown_state():
    shutdown_event.clear()
    yield
    shutdown_event.clear()


def test_extract_usage_data_maps_openai_reasoning_tokens():
    usage_data, _ = extract_usage_data(
        _chat_response(reasoning_tokens=64),
        OperationType.CHAT,
        "2026-06-03T12:00:00Z",
        "2026-06-03T12:00:01Z",
        1000,
    )

    assert usage_data["reasoning_token_count"] == 64


def test_extract_usage_data_maps_dict_shaped_reasoning_token_details():
    # LangChain-normalized usage exposes the detail block as a plain dict.
    usage_data, _ = extract_usage_data(
        _chat_response(reasoning_tokens=48, details_as_dict=True),
        OperationType.CHAT,
        "2026-06-03T12:00:00Z",
        "2026-06-03T12:00:01Z",
        1000,
    )

    assert usage_data["reasoning_token_count"] == 48


def test_extract_usage_data_defaults_reasoning_tokens_to_zero_when_absent():
    usage_data, _ = extract_usage_data(
        _chat_response(),
        OperationType.CHAT,
        "2026-06-03T12:00:00Z",
        "2026-06-03T12:00:01Z",
        1000,
    )

    assert usage_data["reasoning_token_count"] == 0


def test_extract_usage_data_maps_responses_api_reasoning_tokens():
    usage_data, _ = extract_usage_data(
        _responses_response(reasoning_tokens=52),
        OperationType.CHAT,
        "2026-06-03T12:00:00Z",
        "2026-06-03T12:00:01Z",
        1000,
    )

    assert usage_data["reasoning_token_count"] == 52


def test_extract_usage_data_ignores_non_scalar_reasoning_token_counts():
    usage_data, _ = extract_usage_data(
        _chat_response(reasoning_tokens=MagicMock()),
        OperationType.CHAT,
        "2026-06-03T12:00:00Z",
        "2026-06-03T12:00:01Z",
        1000,
    )

    assert usage_data["reasoning_token_count"] == 0


def test_extract_reasoning_token_count_reads_dict_and_attribute_usage_alike():
    assert _extract_reasoning_token_count(
        {"completion_tokens_details": {"reasoning_tokens": 17}}
    ) == 17
    assert _extract_reasoning_token_count(
        SimpleNamespace(
            completion_tokens_details=SimpleNamespace(reasoning_tokens=17)
        )
    ) == 17
    assert _extract_reasoning_token_count({"output_tokens_details": {"reasoning_tokens": 17}}) == 17
    assert _extract_reasoning_token_count(None) == 0
    assert _extract_reasoning_token_count({}) == 0


@patch("revenium_middleware.openai.middleware.current_client", lambda: object())
@patch("revenium_middleware.openai.middleware.submit_ai_event")
@patch("revenium_middleware.openai.middleware.run_async_in_thread")
def test_create_metering_call_forwards_reasoning_token_count(
    mock_run_async, mock_submit
):
    mock_submit.return_value = SimpleNamespace(id="completion-reasoning-test")
    mock_run_async.side_effect = _run_coro_in_thread

    create_metering_call(
        _chat_response(reasoning_tokens=128),
        OperationType.CHAT,
        datetime.datetime.now(datetime.timezone.utc),
        {"trace_id": "reasoning-forward-test"},
    )

    payload = mock_submit.call_args[0][1]
    assert payload["reasoning_token_count"] == 128


@patch("revenium_middleware.openai.middleware.current_client", lambda: object())
@patch("revenium_middleware.openai.middleware.submit_ai_event")
@patch("revenium_middleware.openai.middleware.run_async_in_thread")
def test_create_metering_call_meters_zero_reasoning_when_provider_reports_none(
    mock_run_async, mock_submit
):
    mock_submit.return_value = SimpleNamespace(id="completion-no-reasoning-test")
    mock_run_async.side_effect = _run_coro_in_thread

    create_metering_call(
        _chat_response(),
        OperationType.CHAT,
        datetime.datetime.now(datetime.timezone.utc),
        {"trace_id": "reasoning-absent-test"},
    )

    payload = mock_submit.call_args[0][1]
    assert payload["reasoning_token_count"] == 0


@patch("revenium_middleware.openai.middleware.current_client", lambda: object())
@patch("revenium_middleware.openai.middleware.submit_ai_event")
def test_log_token_usage_without_reasoning_argument_still_sends_zero(mock_submit):
    # Callers that do not know the figure keep the field on the wire as 0
    # rather than dropping it from the payload.
    mock_submit.return_value = SimpleNamespace(id="completion-legacy-reasoning-test")

    asyncio.run(
        log_token_usage(
            response_id="completion-legacy-reasoning-test",
            model="o4-mini",
            prompt_tokens=100,
            completion_tokens=25,
            total_tokens=125,
            cached_tokens=0,
            stop_reason="END",
            request_time="2026-06-03T12:00:00Z",
            response_time="2026-06-03T12:00:01Z",
            request_duration=1000,
            usage_metadata={"trace_id": "legacy-reasoning-test"},
        )
    )

    payload = mock_submit.call_args[0][1]
    assert payload["reasoning_token_count"] == 0


@patch("revenium_middleware.openai.middleware.current_client", lambda: object())
@patch("revenium_middleware.openai.middleware.submit_ai_event")
@patch("revenium_middleware.openai.middleware.run_async_in_thread")
def test_sync_streaming_chat_forwards_reasoning_token_count(
    mock_run_async, mock_submit
):
    mock_submit.return_value = SimpleNamespace(id="completion-stream-reasoning-test")
    mock_run_async.side_effect = _run_coro_in_thread

    final_chunk = _chat_response(reasoning_tokens=96)
    final_chunk.choices = []

    wrapped_stream = handle_streaming_response(
        [final_chunk],
        datetime.datetime.now(datetime.timezone.utc),
        {"trace_id": "stream-reasoning-test"},
    )

    # The middleware-injected usage chunk is hidden from the caller by default;
    # its data still reaches metering below.
    assert list(wrapped_stream) == []
    payload = mock_submit.call_args[0][1]
    assert payload["is_streamed"] is True
    assert payload["reasoning_token_count"] == 96


@patch("revenium_middleware.openai.middleware.current_client", lambda: object())
@patch("revenium_middleware.openai.middleware.submit_ai_event")
@patch("revenium_middleware.openai.middleware.run_async_in_thread")
def test_sync_streaming_chat_meters_zero_reasoning_when_absent(
    mock_run_async, mock_submit
):
    mock_submit.return_value = SimpleNamespace(id="completion-stream-no-reasoning")
    mock_run_async.side_effect = _run_coro_in_thread

    final_chunk = _chat_response()
    final_chunk.choices = []

    wrapped_stream = handle_streaming_response(
        [final_chunk],
        datetime.datetime.now(datetime.timezone.utc),
        {"trace_id": "stream-reasoning-absent-test"},
    )

    assert list(wrapped_stream) == []
    payload = mock_submit.call_args[0][1]
    assert payload["is_streamed"] is True
    assert payload["reasoning_token_count"] == 0


@patch("revenium_middleware.openai.middleware.current_client", lambda: object())
@patch("revenium_middleware.openai.middleware.submit_ai_event")
@patch("revenium_middleware.openai.middleware.run_async_in_thread")
def test_async_streaming_chat_forwards_reasoning_token_count(
    mock_run_async, mock_submit
):
    mock_submit.return_value = SimpleNamespace(
        id="completion-async-stream-reasoning-test"
    )
    mock_run_async.side_effect = _run_coro_in_thread

    final_chunk = _chat_response(reasoning_tokens=112)
    final_chunk.choices = []

    async def stream():
        yield final_chunk

    async def consume():
        wrapped_stream = _wrap_async_stream(
            stream(),
            datetime.datetime.now(datetime.timezone.utc),
            {"trace_id": "async-stream-reasoning-test"},
        )
        chunks = []
        async for chunk in wrapped_stream:
            chunks.append(chunk)
        return chunks

    # The middleware-injected usage chunk is hidden from the caller by default;
    # its data still reaches metering below.
    assert asyncio.run(consume()) == []
    payload = mock_submit.call_args[0][1]
    assert payload["is_streamed"] is True
    assert payload["reasoning_token_count"] == 112


@patch("revenium_middleware.openai.middleware.current_client", lambda: object())
@patch("revenium_middleware.openai.middleware.submit_ai_event")
@patch("revenium_middleware.openai.middleware.run_async_in_thread")
def test_streaming_responses_api_forwards_reasoning_token_count(
    mock_run_async, mock_submit
):
    mock_submit.return_value = SimpleNamespace(
        id="completion-responses-stream-reasoning-test"
    )
    mock_run_async.side_effect = _run_coro_in_thread

    events = typed_events(usage=usage_payload(reasoning_tokens=72))

    wrapped_stream = handle_streaming_responses(
        events,
        datetime.datetime.now(datetime.timezone.utc),
        {"trace_id": "responses-stream-reasoning-test"},
    )

    assert list(wrapped_stream) == events
    assert mock_submit.call_count == 1
    payload = mock_submit.call_args[0][1]
    assert payload["is_streamed"] is True
    assert payload["model"] == MODEL
    assert payload["reasoning_token_count"] == 72


@patch("revenium_middleware.openai.middleware.current_client", lambda: object())
@patch("revenium_middleware.openai.middleware.submit_ai_event")
@patch("revenium_middleware.openai.middleware.run_async_in_thread")
def test_canonical_openai_usage_vector_maps_reasoning_and_cache_fields(
    mock_run_async, mock_submit
):
    # Pinned cross-SDK vector: the Node middleware maps the same provider
    # fields to reasoningTokenCount / cacheReadTokenCount, so these three
    # emitted values must stay identical for identical provider input.
    mock_submit.return_value = SimpleNamespace(id="completion-canonical-vector-test")
    mock_run_async.side_effect = _run_coro_in_thread

    response = SimpleNamespace(
        id="chatcmpl-canonical-vector",
        model="o4-mini",
        usage=SimpleNamespace(
            prompt_tokens=1200,
            completion_tokens=450,
            total_tokens=1650,
            prompt_tokens_details=SimpleNamespace(cached_tokens=1024),
            completion_tokens_details=SimpleNamespace(reasoning_tokens=384),
        ),
        choices=[SimpleNamespace(finish_reason="stop")],
        system_fingerprint=None,
    )

    create_metering_call(
        response,
        OperationType.CHAT,
        datetime.datetime.now(datetime.timezone.utc),
        {"trace_id": "canonical-vector-test"},
    )

    payload = mock_submit.call_args[0][1]
    assert payload["reasoning_token_count"] == 384
    assert payload["cache_read_token_count"] == 1024
    assert payload["cache_creation_token_count"] == 0


# ---- xAI (BACK-4238) ------------------------------------------------------
# xAI reports completion_tokens as the answer alone and bills the reasoning
# tokens as output beside it; total_tokens is prompt + completion + reasoning.

from httpx import AsyncClient, Client, MockTransport  # noqa: E402
from openai import AsyncOpenAI, OpenAI  # noqa: E402

from . import test_openai_sdk_entry_points as entry_points  # noqa: E402
from .test_openai_sdk_entry_points import (  # noqa: E402,F401  payloads is a fixture
    CHAT_REQUEST,
    NON_STREAMED,
    STREAMED_CHAT,
    STREAMED_RESPONSES,
    FakeOpenAI,
    payloads,
    sync_chat_create,
    sync_chat_create_stream,
    sync_responses_create,
)

XAI_BASE_URL = "https://api.x.ai/v1"


def xai_chat_usage(prompt, completion, total, reasoning=None, cached=0, ticks=None):
    usage = {
        "prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": total,
        "prompt_tokens_details": {"text_tokens": prompt, "audio_tokens": 0, "image_tokens": 0, "cached_tokens": cached},
        "num_sources_used": 0,
    }
    if reasoning is not None:
        usage["completion_tokens_details"] = {"reasoning_tokens": reasoning, "audio_tokens": 0,
                                              "accepted_prediction_tokens": 0, "rejected_prediction_tokens": 0}
    if ticks is not None:
        usage["cost_in_usd_ticks"] = ticks
    return usage


def xai_responses_usage(prompt, completion, total, reasoning=0, cached=0, ticks=None):
    usage = {
        "input_tokens": prompt, "input_tokens_details": {"cached_tokens": cached},
        "output_tokens": completion, "output_tokens_details": {"reasoning_tokens": reasoning},
        "total_tokens": total,
    }
    if ticks is not None:
        usage["cost_in_usd_ticks"] = ticks
    return usage


# The verifier's live call: grok-3-mini requested, served by a reasoning model.
LIVE_XAI_CHAT_USAGE = xai_chat_usage(204, 1, 330, reasoning=125, cached=128, ticks=4356000)
LIVE_XAI_RESPONSES_USAGE = xai_responses_usage(204, 1, 330, reasoning=125, cached=128, ticks=4356000)
LIVE_XAI_COST_USD = 0.0004356


class FakeXAI(FakeOpenAI):
    def sync_client(self):
        return OpenAI(api_key="xai-test", base_url=XAI_BASE_URL,
                      http_client=Client(transport=MockTransport(self)))

    def async_client(self):
        return AsyncOpenAI(api_key="xai-test", base_url=XAI_BASE_URL,
                           http_client=AsyncClient(transport=MockTransport(self._async_handler())))


@pytest.fixture
def served_xai_usage(monkeypatch):
    def serve(chat=LIVE_XAI_CHAT_USAGE, responses=LIVE_XAI_RESPONSES_USAGE):
        monkeypatch.setattr(entry_points, "CHAT_USAGE", chat)
        monkeypatch.setattr(entry_points, "RESPONSES_USAGE", responses)
    return serve


def the_one_record(recorded):
    assert len(recorded) == 1
    return recorded[0]


EVERY_ENTRY_POINT = [*NON_STREAMED, *STREAMED_CHAT, *STREAMED_RESPONSES]


@pytest.mark.parametrize("call", EVERY_ENTRY_POINT, ids=lambda c: c.__name__)
def test_xai_reasoning_tokens_are_billed_as_output_on_every_entry_point(call, payloads, served_xai_usage):
    served_xai_usage()

    call(FakeXAI(), None)

    record = the_one_record(payloads)
    assert record["provider"] == "XAI"
    assert record["output_token_count"] == 126
    assert record["reasoning_token_count"] == 125
    assert record["input_token_count"] == 204
    assert record["total_token_count"] == 330
    assert record["cache_read_token_count"] == 128


@pytest.mark.parametrize("call", EVERY_ENTRY_POINT, ids=lambda c: c.__name__)
def test_xai_reported_cost_is_sent_as_the_total_cost_on_every_entry_point(call, payloads, served_xai_usage):
    served_xai_usage()

    call(FakeXAI(), None)

    record = the_one_record(payloads)
    assert record.get("total_cost") == pytest.approx(LIVE_XAI_COST_USD)
    assert "input_token_cost" not in record
    assert "output_token_cost" not in record


def test_xai_raw_response_call_carries_the_reasoning_and_the_cost(payloads, served_xai_usage):
    served_xai_usage()

    FakeXAI().sync_client().chat.completions.with_raw_response.create(**CHAT_REQUEST).parse()

    record = the_one_record(payloads)
    assert record["output_token_count"] == 126
    assert record.get("total_cost") == pytest.approx(LIVE_XAI_COST_USD)


def test_xai_replay_body_bills_every_reasoning_token(payloads, served_xai_usage):
    served_xai_usage(chat=xai_chat_usage(202, 1, 401, reasoning=198, ticks=6156000))

    sync_chat_create(FakeXAI(), None)

    record = the_one_record(payloads)
    assert record["output_token_count"] == 199
    assert record.get("total_cost") == pytest.approx(0.0006156)


def test_xai_completion_that_already_includes_reasoning_is_not_doubled(payloads, served_xai_usage):
    served_xai_usage(chat=xai_chat_usage(204, 126, 330, reasoning=125, ticks=4356000))

    sync_chat_create(FakeXAI(), None)

    assert the_one_record(payloads)["output_token_count"] == 126


def test_non_reasoning_grok_keeps_its_output_count(payloads, served_xai_usage):
    served_xai_usage(chat=xai_chat_usage(193, 12, 205, ticks=1068500))

    sync_chat_create(FakeXAI(), None)

    record = the_one_record(payloads)
    assert record["output_token_count"] == 12
    assert record["reasoning_token_count"] == 0


@pytest.mark.parametrize("call", [sync_chat_create, sync_chat_create_stream, sync_responses_create], ids=lambda c: c.__name__)
def test_xai_without_reported_cost_sends_no_total_cost(call, payloads, served_xai_usage):
    served_xai_usage(chat=xai_chat_usage(204, 1, 330, reasoning=125),
                     responses=xai_responses_usage(204, 1, 330, reasoning=125))

    call(FakeXAI(), None)

    assert "total_cost" not in the_one_record(payloads)


@pytest.mark.parametrize("call", EVERY_ENTRY_POINT, ids=lambda c: c.__name__)
def test_openai_usage_shaped_like_xai_is_recorded_as_before(call, payloads, served_xai_usage):
    served_xai_usage()

    call(FakeOpenAI(), None)

    record = the_one_record(payloads)
    assert record["provider"] == "OPENAI"
    assert record["output_token_count"] == 1
    assert "total_cost" not in record


@pytest.mark.parametrize("total", [None, 204 + 1 + 7], ids=["no_total", "total_matching_neither"])
def test_xai_total_matching_neither_identity_adds_nothing(total, payloads, served_xai_usage):
    served_xai_usage(chat=xai_chat_usage(204, 1, total, reasoning=125))

    sync_chat_create(FakeXAI(), None)

    assert the_one_record(payloads)["output_token_count"] == 1


def test_an_interrupted_xai_stream_is_recorded_with_zero_tokens_and_no_cost(payloads):
    """The usage an async chat stream cut before its usage chunk is metered with."""
    zero_usage = type('Usage', (), {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0})()
    response = type('Response', (), {
        'id': 'chatcmpl-cut', 'model': 'grok-4.3', 'usage': zero_usage, 'system_fingerprint': None,
        'choices': [SimpleNamespace(finish_reason=None)],
    })()

    create_metering_call(response, OperationType.CHAT, datetime.datetime.now(datetime.timezone.utc), {},
                         client_instance=FakeXAI().sync_client(), is_streamed=True)

    record = the_one_record(payloads)
    assert record["provider"] == "XAI"
    assert record["output_token_count"] == 0
    assert "total_cost" not in record
