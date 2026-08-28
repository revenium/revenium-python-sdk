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


@patch("revenium_middleware.openai.middleware.get_client", lambda: object())
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


@patch("revenium_middleware.openai.middleware.get_client", lambda: object())
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


@patch("revenium_middleware.openai.middleware.get_client", lambda: object())
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


@patch("revenium_middleware.openai.middleware.get_client", lambda: object())
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


@patch("revenium_middleware.openai.middleware.get_client", lambda: object())
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


@patch("revenium_middleware.openai.middleware.get_client", lambda: object())
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


@patch("revenium_middleware.openai.middleware.get_client", lambda: object())
@patch("revenium_middleware.openai.middleware.submit_ai_event")
@patch("revenium_middleware.openai.middleware.run_async_in_thread")
def test_streaming_responses_api_forwards_reasoning_token_count(
    mock_run_async, mock_submit
):
    mock_submit.return_value = SimpleNamespace(
        id="completion-responses-stream-reasoning-test"
    )
    mock_run_async.side_effect = _run_coro_in_thread

    final_chunk = _responses_response(reasoning_tokens=72)

    wrapped_stream = handle_streaming_responses(
        [final_chunk],
        datetime.datetime.now(datetime.timezone.utc),
        {"trace_id": "responses-stream-reasoning-test"},
    )

    assert list(wrapped_stream) == [final_chunk]
    payload = mock_submit.call_args[0][1]
    assert payload["is_streamed"] is True
    assert payload["reasoning_token_count"] == 72


@patch("revenium_middleware.openai.middleware.get_client", lambda: object())
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
