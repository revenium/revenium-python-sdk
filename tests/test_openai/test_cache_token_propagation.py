import asyncio
import datetime
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from revenium_middleware import shutdown_event
from revenium_middleware.openai.middleware import (
    OperationType,
    _wrap_async_stream,
    create_metering_call,
    extract_usage_data,
    handle_streaming_response,
    handle_streaming_responses,
    log_token_usage,
)


def _chat_response(cached_tokens=0):
    return SimpleNamespace(
        id="chatcmpl-cache-test",
        model="gpt-4o-mini",
        usage=SimpleNamespace(
            prompt_tokens=100,
            completion_tokens=25,
            total_tokens=125,
            prompt_tokens_details=SimpleNamespace(cached_tokens=cached_tokens),
        ),
        choices=[SimpleNamespace(finish_reason="stop")],
        system_fingerprint=None,
    )


def _responses_response(cached_tokens=0):
    return SimpleNamespace(
        id="resp-cache-test",
        model="gpt-4o-mini",
        usage=SimpleNamespace(
            input_tokens=80,
            output_tokens=20,
            total_tokens=100,
            input_tokens_details=SimpleNamespace(cached_tokens=cached_tokens),
        ),
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


def test_extract_usage_data_maps_openai_cached_tokens_to_cache_read():
    usage_data, _ = extract_usage_data(
        _chat_response(cached_tokens=33),
        OperationType.CHAT,
        "2026-06-03T12:00:00Z",
        "2026-06-03T12:00:01Z",
        1000,
    )

    assert usage_data["cache_creation_token_count"] == 0
    assert usage_data["cache_read_token_count"] == 33


def test_extract_usage_data_maps_responses_cached_tokens_to_cache_read():
    usage_data, _ = extract_usage_data(
        _responses_response(cached_tokens=44),
        OperationType.CHAT,
        "2026-06-03T12:00:00Z",
        "2026-06-03T12:00:01Z",
        1000,
    )

    assert usage_data["cache_creation_token_count"] == 0
    assert usage_data["cache_read_token_count"] == 44


def test_extract_usage_data_ignores_non_scalar_cache_token_counts():
    usage_data, _ = extract_usage_data(
        _chat_response(cached_tokens=MagicMock()),
        OperationType.CHAT,
        "2026-06-03T12:00:00Z",
        "2026-06-03T12:00:01Z",
        1000,
    )

    assert usage_data["cache_creation_token_count"] == 0
    assert usage_data["cache_read_token_count"] == 0


def test_extract_usage_data_preserves_anthropic_shaped_langchain_cache_details():
    response = SimpleNamespace(
        id="langchain-anthropic-cache-test",
        model="claude-3-5-sonnet-20241022",
        usage=SimpleNamespace(
            input_tokens=120,
            output_tokens=30,
            total_tokens=150,
            input_token_details=SimpleNamespace(
                cache_creation=11,
                cache_read=22,
            ),
        ),
        choices=[SimpleNamespace(finish_reason="stop")],
        system_fingerprint=None,
    )

    usage_data, _ = extract_usage_data(
        response,
        OperationType.CHAT,
        "2026-06-03T12:00:00Z",
        "2026-06-03T12:00:01Z",
        1000,
    )

    assert usage_data["cache_creation_token_count"] == 11
    assert usage_data["cache_read_token_count"] == 22


def test_extract_usage_data_does_not_sum_mutually_exclusive_cache_details():
    response = _chat_response(cached_tokens=33)
    response.usage.input_tokens_details = SimpleNamespace(cached_tokens=44)
    response.usage.input_token_details = SimpleNamespace(
        cache_creation=11,
        cache_read=22,
    )

    usage_data, _ = extract_usage_data(
        response,
        OperationType.CHAT,
        "2026-06-03T12:00:00Z",
        "2026-06-03T12:00:01Z",
        1000,
    )

    assert usage_data["cache_creation_token_count"] == 11
    assert usage_data["cache_read_token_count"] == 22


@patch("revenium_middleware.openai.middleware.get_client", lambda: object())
@patch("revenium_middleware.openai.middleware.submit_ai_event")
@patch("revenium_middleware.openai.middleware.run_async_in_thread")
def test_create_metering_call_forwards_cache_creation_and_read_counts(
    mock_run_async, mock_submit
):
    mock_submit.return_value = SimpleNamespace(id="completion-cache-test")
    mock_run_async.side_effect = _run_coro_in_thread

    create_metering_call(
        _chat_response(cached_tokens=55),
        OperationType.CHAT,
        datetime.datetime.now(datetime.timezone.utc),
        {"trace_id": "cache-forward-test"},
    )

    payload = mock_submit.call_args[0][1]
    assert payload["cache_creation_token_count"] == 0
    assert payload["cache_read_token_count"] == 55


@patch("revenium_middleware.openai.middleware.get_client", lambda: object())
@patch("revenium_middleware.openai.middleware.submit_ai_event")
def test_log_token_usage_legacy_cached_tokens_fall_back_to_cache_reads(mock_submit):
    mock_submit.return_value = SimpleNamespace(id="completion-legacy-cache-test")

    asyncio.run(
        log_token_usage(
            response_id="completion-legacy-cache-test",
            model="gpt-4o-mini",
            prompt_tokens=100,
            completion_tokens=25,
            total_tokens=125,
            cached_tokens=42,
            stop_reason="END",
            request_time="2026-06-03T12:00:00Z",
            response_time="2026-06-03T12:00:01Z",
            request_duration=1000,
            usage_metadata={"trace_id": "legacy-cache-test"},
        )
    )

    payload = mock_submit.call_args[0][1]
    assert payload["cache_creation_token_count"] == 0
    assert payload["cache_read_token_count"] == 42


@patch("revenium_middleware.openai.middleware.get_client", lambda: object())
@patch("revenium_middleware.openai.middleware.submit_ai_event")
def test_log_token_usage_explicit_zero_cache_read_overrides_legacy_cached_tokens(
    mock_submit,
):
    mock_submit.return_value = SimpleNamespace(id="completion-explicit-cache-test")

    asyncio.run(
        log_token_usage(
            response_id="completion-explicit-cache-test",
            model="gpt-4o-mini",
            prompt_tokens=100,
            completion_tokens=25,
            total_tokens=125,
            cached_tokens=42,
            stop_reason="END",
            request_time="2026-06-03T12:00:00Z",
            response_time="2026-06-03T12:00:01Z",
            request_duration=1000,
            usage_metadata={"trace_id": "explicit-cache-test"},
            cache_read_token_count=0,
        )
    )

    payload = mock_submit.call_args[0][1]
    assert payload["cache_creation_token_count"] == 0
    assert payload["cache_read_token_count"] == 0


@patch("revenium_middleware.openai.middleware.get_client", lambda: object())
@patch("revenium_middleware.openai.middleware.submit_ai_event")
@patch("revenium_middleware.openai.middleware.run_async_in_thread")
def test_sync_streaming_chat_forwards_openai_cached_tokens_as_reads(
    mock_run_async, mock_submit
):
    mock_submit.return_value = SimpleNamespace(id="completion-stream-cache-test")
    mock_run_async.side_effect = _run_coro_in_thread

    final_chunk = _chat_response(cached_tokens=66)
    final_chunk.choices = []

    wrapped_stream = handle_streaming_response(
        [final_chunk],
        datetime.datetime.now(datetime.timezone.utc),
        {"trace_id": "stream-cache-test"},
    )

    # The middleware-injected usage chunk is hidden from the caller by default;
    # its data still reaches metering below.
    assert list(wrapped_stream) == []
    payload = mock_submit.call_args[0][1]
    assert payload["is_streamed"] is True
    assert payload["cache_creation_token_count"] == 0
    assert payload["cache_read_token_count"] == 66


@patch("revenium_middleware.openai.middleware.get_client", lambda: object())
@patch("revenium_middleware.openai.middleware.submit_ai_event")
@patch("revenium_middleware.openai.middleware.run_async_in_thread")
def test_sync_streaming_chat_preserves_anthropic_shaped_cache_details(
    mock_run_async, mock_submit
):
    mock_submit.return_value = SimpleNamespace(id="completion-stream-cache-test")
    mock_run_async.side_effect = _run_coro_in_thread

    final_chunk = _chat_response(cached_tokens=0)
    final_chunk.choices = []
    final_chunk.usage.input_token_details = SimpleNamespace(
        cache_creation=12,
        cache_read=34,
    )

    wrapped_stream = handle_streaming_response(
        [final_chunk],
        datetime.datetime.now(datetime.timezone.utc),
        {"trace_id": "stream-anthropic-cache-test"},
    )

    # The middleware-injected usage chunk is hidden from the caller by default;
    # its data still reaches metering below.
    assert list(wrapped_stream) == []
    payload = mock_submit.call_args[0][1]
    assert payload["is_streamed"] is True
    assert payload["cache_creation_token_count"] == 12
    assert payload["cache_read_token_count"] == 34


@patch("revenium_middleware.openai.middleware.get_client", lambda: object())
@patch("revenium_middleware.openai.middleware.submit_ai_event")
@patch("revenium_middleware.openai.middleware.run_async_in_thread")
def test_streaming_responses_api_forwards_cached_tokens_as_reads(
    mock_run_async, mock_submit
):
    mock_submit.return_value = SimpleNamespace(id="completion-responses-stream-test")
    mock_run_async.side_effect = _run_coro_in_thread

    final_chunk = _responses_response(cached_tokens=77)

    wrapped_stream = handle_streaming_responses(
        [final_chunk],
        datetime.datetime.now(datetime.timezone.utc),
        {"trace_id": "responses-stream-cache-test"},
    )

    assert list(wrapped_stream) == [final_chunk]
    payload = mock_submit.call_args[0][1]
    assert payload["is_streamed"] is True
    assert payload["cache_creation_token_count"] == 0
    assert payload["cache_read_token_count"] == 77


@patch("revenium_middleware.openai.middleware.get_client", lambda: object())
@patch("revenium_middleware.openai.middleware.submit_ai_event")
@patch("revenium_middleware.openai.middleware.run_async_in_thread")
def test_async_streaming_chat_preserves_openai_cached_tokens_as_reads(
    mock_run_async, mock_submit
):
    mock_submit.return_value = SimpleNamespace(id="completion-async-stream-cache-test")
    mock_run_async.side_effect = _run_coro_in_thread

    final_chunk = _chat_response(cached_tokens=88)
    final_chunk.choices = []

    async def stream():
        yield final_chunk

    async def consume():
        wrapped_stream = _wrap_async_stream(
            stream(),
            datetime.datetime.now(datetime.timezone.utc),
            {"trace_id": "async-stream-cache-test"},
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
    assert payload["cache_creation_token_count"] == 0
    assert payload["cache_read_token_count"] == 88


@patch("revenium_middleware.openai.middleware.create_metering_call")
def test_async_streaming_chat_without_final_usage_meters_zero_counts(mock_create_metering):
    chunk = SimpleNamespace(
        id="chatcmpl-no-usage",
        model="gpt-4o-mini",
        choices=[
            SimpleNamespace(
                finish_reason=None,
                delta=SimpleNamespace(content="hello"),
            )
        ],
    )

    async def stream():
        yield chunk

    async def consume():
        wrapped_stream = _wrap_async_stream(
            stream(),
            datetime.datetime.now(datetime.timezone.utc),
            {"trace_id": "async-stream-no-usage-test"},
        )
        chunks = []
        async for stream_chunk in wrapped_stream:
            chunks.append(stream_chunk)
        return chunks

    assert asyncio.run(consume()) == [chunk]
    # A stream that ends without a usage chunk is still metered (zero token
    # counts) so the transaction is not silently invisible.
    mock_create_metering.assert_called_once()
    response_stub = mock_create_metering.call_args[0][0]
    assert response_stub.usage.prompt_tokens == 0
    assert response_stub.usage.completion_tokens == 0
    assert mock_create_metering.call_args.kwargs["is_streamed"] is True


def test_langchain_handler_extracts_usage_from_generation_message():
    try:
        from revenium_middleware.openai.langchain.unified_handler import (
            UnifiedReveniumCallbackHandler,
        )
    except ImportError:
        pytest.skip("LangChain not installed")

    handler = UnifiedReveniumCallbackHandler()
    response = SimpleNamespace(
        generations=[
            [
                SimpleNamespace(
                    message=SimpleNamespace(
                        usage_metadata={
                            "input_tokens": 10,
                            "output_tokens": 5,
                            "total_tokens": 15,
                        }
                    )
                )
            ]
        ]
    )

    assert handler._extract_usage_from_response(response) == {
        "input_tokens": 10,
        "output_tokens": 5,
        "total_tokens": 15,
    }


def test_langchain_mock_response_preserves_openai_and_anthropic_cache_details():
    try:
        from revenium_middleware.openai.langchain.unified_handler import (
            UnifiedReveniumCallbackHandler,
        )
    except ImportError:
        pytest.skip("LangChain not installed")

    handler = UnifiedReveniumCallbackHandler()
    run_info = {
        "serialized": {"model": "gpt-4o-mini"},
        "start_time": datetime.datetime.now(datetime.timezone.utc).timestamp(),
    }

    openai_mock_response = handler._create_mock_response(
        {
            "prompt_tokens": 90,
            "completion_tokens": 10,
            "total_tokens": 100,
            "prompt_tokens_details": {"cached_tokens": 31},
        },
        run_info,
    )
    openai_usage_data, _ = extract_usage_data(
        openai_mock_response,
        OperationType.CHAT,
        "2026-06-03T12:00:00Z",
        "2026-06-03T12:00:01Z",
        1000,
    )
    assert openai_usage_data["cache_creation_token_count"] == 0
    assert openai_usage_data["cache_read_token_count"] == 31

    anthropic_mock_response = handler._create_mock_response(
        {
            "input_tokens": 90,
            "output_tokens": 10,
            "total_tokens": 100,
            "input_token_details": {"cache_creation": 12, "cache_read": 34},
        },
        run_info,
    )
    anthropic_usage_data, _ = extract_usage_data(
        anthropic_mock_response,
        OperationType.CHAT,
        "2026-06-03T12:00:00Z",
        "2026-06-03T12:00:01Z",
        1000,
    )
    assert anthropic_usage_data["cache_creation_token_count"] == 12
    assert anthropic_usage_data["cache_read_token_count"] == 34


def test_extract_usage_data_maps_anthropic_raw_input_token_keys():
    # A live ChatAnthropic LLMResult exposes the raw `response_metadata.usage`
    # block (flat `cache_creation_input_tokens` / `cache_read_input_tokens`),
    # not the normalized `input_token_details` shape. This is the shape that is
    # actually extracted first by the handler.
    response = SimpleNamespace(
        id="langchain-anthropic-raw-cache-test",
        model="claude-sonnet-4-6",
        usage=SimpleNamespace(
            input_tokens=3,
            output_tokens=21,
            total_tokens=24,
            cache_creation_input_tokens=14,
            cache_read_input_tokens=9522,
        ),
        choices=[SimpleNamespace(finish_reason="stop")],
        system_fingerprint=None,
    )

    usage_data, _ = extract_usage_data(
        response,
        OperationType.CHAT,
        "2026-06-03T12:00:00Z",
        "2026-06-03T12:00:01Z",
        1000,
    )

    assert usage_data["cache_creation_token_count"] == 14
    assert usage_data["cache_read_token_count"] == 9522


def test_langchain_mock_response_preserves_anthropic_raw_usage_shape():
    try:
        from revenium_middleware.openai.langchain.unified_handler import (
            UnifiedReveniumCallbackHandler,
        )
    except ImportError:
        pytest.skip("LangChain not installed")

    handler = UnifiedReveniumCallbackHandler()
    run_info = {
        "serialized": {"model": "claude-sonnet-4-6"},
        "start_time": datetime.datetime.now(datetime.timezone.utc).timestamp(),
    }

    # Exact shape captured from a live ChatAnthropic `response_metadata.usage`
    # on a cache-read call.
    mock_response = handler._create_mock_response(
        {
            "input_tokens": 3,
            "output_tokens": 21,
            "total_tokens": 24,
            "cache_creation": {
                "ephemeral_1h_input_tokens": 0,
                "ephemeral_5m_input_tokens": 14,
            },
            "cache_creation_input_tokens": 14,
            "cache_read_input_tokens": 9522,
        },
        run_info,
    )
    usage_data, _ = extract_usage_data(
        mock_response,
        OperationType.CHAT,
        "2026-06-03T12:00:00Z",
        "2026-06-03T12:00:01Z",
        1000,
    )
    assert usage_data["cache_creation_token_count"] == 14
    assert usage_data["cache_read_token_count"] == 9522
