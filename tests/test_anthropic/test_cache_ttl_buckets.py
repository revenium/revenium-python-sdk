"""Native Anthropic must forward the per-TTL cache-creation breakdown.

Anthropic reports the split under the nested ``usage.cache_creation`` object
(``ephemeral_5m_input_tokens`` / ``ephemeral_1h_input_tokens``). Every native
metering path forwards it as ``cache_creation5m_token_count`` /
``cache_creation1h_token_count`` alongside the unchanged aggregate
``cache_creation_token_count``, and omits a bucket the response does not carry
so the backend keeps pricing the flat fallback from the aggregate.
"""
import asyncio
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from revenium_middleware import shutdown_event
from revenium_middleware._core.cache_tokens import extract_cache_creation_ttl_counts
from revenium_middleware.anthropic import middleware as anthropic_middleware
from revenium_middleware.anthropic.stream_create import StreamUsageState


def _middleware_fn(obj):
    """Return the plain middleware function across wrapt versions.

    Depending on the wrapt version, @wrapt.patch_function_wrapper leaves the
    module-level name as either the patched FunctionWrapper (call it directly
    and you invoke the real Anthropic SDK method; the middleware function
    lives at ._self_wrapper) or the plain wrapper function itself.
    """
    return getattr(obj, "_self_wrapper", obj)


create_wrapper = _middleware_fn(anthropic_middleware.create_wrapper)
async_create_wrapper = _middleware_fn(anthropic_middleware.async_create_wrapper)
stream_wrapper = _middleware_fn(anthropic_middleware.stream_wrapper)


def run_metering_synchronously(coro_func, *args, **kwargs):
    """Run the metering coroutine to completion on a dedicated thread.

    Mirrors production (_safe_run_async_in_thread dispatches to a thread) and
    works whether or not the caller is already inside an event loop.
    """
    thread = threading.Thread(target=lambda: asyncio.run(coro_func(*args, **kwargs)))
    thread.start()
    thread.join(timeout=10)
    return MagicMock()


def usage(cache_creation="omitted"):
    """Anthropic usage block; `cache_creation` defaults to the pre-beta shape.

    SimpleNamespace is deliberate: an attribute-style test double that
    auto-creates missing attributes (MagicMock) cannot express "the response
    did not carry a TTL split".
    """
    fields = {
        "input_tokens": 100,
        "output_tokens": 50,
        "cache_creation_input_tokens": 300,
        "cache_read_input_tokens": 7,
    }
    if cache_creation != "omitted":
        fields["cache_creation"] = cache_creation
    return SimpleNamespace(**fields)


def message(cache_creation="omitted"):
    return SimpleNamespace(
        id="msg_ttl_buckets",
        model="claude-sonnet-4-6",
        stop_reason="end_turn",
        usage=usage(cache_creation),
    )


def ttl_split(ephemeral_5m=200, ephemeral_1h=100):
    return SimpleNamespace(
        ephemeral_5m_input_tokens=ephemeral_5m,
        ephemeral_1h_input_tokens=ephemeral_1h,
    )


def request_kwargs():
    return {
        "messages": [{"role": "user", "content": "Hello"}],
        "model": "claude-sonnet-4-6",
        "usage_metadata": {"trace_id": "trace-ttl-buckets"},
    }


def stream_events(cache_creation="omitted"):
    """Minimal raw-stream event sequence as emitted by the Anthropic SDK."""
    return [
        SimpleNamespace(
            type="message_start",
            message=SimpleNamespace(
                id="msg_ttl_buckets",
                model="claude-sonnet-4-6",
                usage=usage(cache_creation),
            ),
        ),
        SimpleNamespace(
            type="message_delta",
            delta=SimpleNamespace(stop_reason="end_turn"),
            usage=SimpleNamespace(output_tokens=50),
        ),
        SimpleNamespace(type="message_stop"),
    ]


class FakeStream:
    """Stands in for anthropic.Stream: iterable, context manager, closeable."""

    def __init__(self, events):
        self._events = list(events)

    def __iter__(self):
        return iter(self._events)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False

    def close(self):
        pass


class FakeAsyncStream:
    """Stands in for anthropic.AsyncStream."""

    def __init__(self, events):
        self._events = list(events)

    def __aiter__(self):
        self._it = iter(self._events)
        return self

    async def __anext__(self):
        try:
            return next(self._it)
        except StopIteration:
            raise StopAsyncIteration

    async def aclose(self):
        pass


class FakeMessageStream:
    """Stands in for the client.messages.stream() context-manager helper."""

    def __init__(self, final_message):
        self._final_message = final_message

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False

    def get_final_message(self):
        return self._final_message


@pytest.fixture(autouse=True)
def reset_shutdown_state():
    shutdown_event.clear()
    yield
    shutdown_event.clear()


def test_extraction_returns_both_buckets_when_the_response_carries_the_split():
    assert extract_cache_creation_ttl_counts(usage(ttl_split())) == {
        "cache_creation5m_token_count": 200,
        "cache_creation1h_token_count": 100,
    }


def test_extraction_returns_nothing_when_the_nested_object_is_absent():
    assert extract_cache_creation_ttl_counts(usage()) == {}


def test_extraction_returns_nothing_when_the_nested_object_is_none():
    assert extract_cache_creation_ttl_counts(usage(None)) == {}


def test_extraction_omits_none_valued_nested_fields():
    assert extract_cache_creation_ttl_counts(usage(ttl_split(None, None))) == {}


def test_extraction_omits_only_the_none_valued_bucket():
    # A partial split must not be padded out with an invented zero for the
    # bucket the provider stayed silent about.
    assert extract_cache_creation_ttl_counts(usage(ttl_split(200, None))) == {
        "cache_creation5m_token_count": 200,
    }


def test_extraction_forwards_a_provider_reported_zero():
    # An explicit 0 from the provider is real data: the write happened entirely
    # in the other TTL bucket.
    assert extract_cache_creation_ttl_counts(usage(ttl_split(200, 0))) == {
        "cache_creation5m_token_count": 200,
        "cache_creation1h_token_count": 0,
    }


def test_extraction_reads_dict_shaped_usage():
    dict_usage = {
        "cache_creation_input_tokens": 300,
        "cache_creation": {
            "ephemeral_5m_input_tokens": 200,
            "ephemeral_1h_input_tokens": 100,
        },
    }

    assert extract_cache_creation_ttl_counts(dict_usage) == {
        "cache_creation5m_token_count": 200,
        "cache_creation1h_token_count": 100,
    }


def test_extraction_ignores_non_numeric_bucket_values():
    assert extract_cache_creation_ttl_counts(usage(MagicMock())) == {}


def test_bucket_parameter_names_map_to_the_metering_wire_aliases():
    """A typo in either parameter name would ship snake_case to the API.

    The generated params TypedDict is the only place the wire aliases are
    defined, so transforming through it proves the emitted keys are real
    metering parameters rather than silently-ignored extras.
    """
    from revenium_middleware._metering._utils import maybe_transform
    from revenium_middleware._metering.types import ai_create_completion_params

    body = maybe_transform(
        extract_cache_creation_ttl_counts(usage(ttl_split())),
        ai_create_completion_params.AICreateCompletionParams,
    )

    assert body == {
        "cacheCreation5mTokenCount": 200,
        "cacheCreation1hTokenCount": 100,
    }


@patch("revenium_middleware.anthropic.middleware._get_thread_safe_client", return_value=MagicMock())
@patch("revenium_middleware.anthropic.middleware.submit_ai_event", return_value=MagicMock(status_code=201))
@patch("revenium_middleware.anthropic.middleware._safe_run_async_in_thread", side_effect=run_metering_synchronously)
class TestSyncCreate:
    def test_split_is_forwarded_alongside_the_aggregate(self, mock_thread, mock_submit, mock_client):
        mock_wrapped = MagicMock(return_value=message(ttl_split()))

        create_wrapper(mock_wrapped, None, (), request_kwargs())

        payload = mock_submit.call_args[0][1]
        assert payload["cache_creation_token_count"] == 300
        assert payload["cache_creation5m_token_count"] == 200
        assert payload["cache_creation1h_token_count"] == 100

    def test_missing_split_leaves_only_the_aggregate(self, mock_thread, mock_submit, mock_client):
        mock_wrapped = MagicMock(return_value=message())

        create_wrapper(mock_wrapped, None, (), request_kwargs())

        payload = mock_submit.call_args[0][1]
        assert payload["cache_creation_token_count"] == 300
        assert "cache_creation5m_token_count" not in payload
        assert "cache_creation1h_token_count" not in payload

    def test_none_valued_split_leaves_only_the_aggregate(self, mock_thread, mock_submit, mock_client):
        mock_wrapped = MagicMock(return_value=message(ttl_split(None, None)))

        create_wrapper(mock_wrapped, None, (), request_kwargs())

        payload = mock_submit.call_args[0][1]
        assert payload["cache_creation_token_count"] == 300
        assert "cache_creation5m_token_count" not in payload
        assert "cache_creation1h_token_count" not in payload


@patch("revenium_middleware.anthropic.middleware._get_thread_safe_client", return_value=MagicMock())
@patch("revenium_middleware.anthropic.middleware.submit_ai_event", return_value=MagicMock(status_code=201))
@patch("revenium_middleware.anthropic.middleware._safe_run_async_in_thread", side_effect=run_metering_synchronously)
class TestAsyncCreate:
    @staticmethod
    def _call(response):
        async def fake_create(**kwargs):
            return response

        async def scenario():
            return await async_create_wrapper(fake_create, None, (), request_kwargs())

        return asyncio.run(scenario())

    def test_split_is_forwarded_alongside_the_aggregate(self, mock_thread, mock_submit, mock_client):
        self._call(message(ttl_split()))

        payload = mock_submit.call_args[0][1]
        assert payload["cache_creation_token_count"] == 300
        assert payload["cache_creation5m_token_count"] == 200
        assert payload["cache_creation1h_token_count"] == 100

    def test_missing_split_leaves_only_the_aggregate(self, mock_thread, mock_submit, mock_client):
        self._call(message())

        payload = mock_submit.call_args[0][1]
        assert payload["cache_creation_token_count"] == 300
        assert "cache_creation5m_token_count" not in payload
        assert "cache_creation1h_token_count" not in payload

    def test_none_valued_split_leaves_only_the_aggregate(self, mock_thread, mock_submit, mock_client):
        self._call(message(ttl_split(None, None)))

        payload = mock_submit.call_args[0][1]
        assert payload["cache_creation_token_count"] == 300
        assert "cache_creation5m_token_count" not in payload
        assert "cache_creation1h_token_count" not in payload


@patch("revenium_middleware.anthropic.middleware._get_thread_safe_client", return_value=MagicMock())
@patch("revenium_middleware.anthropic.middleware.submit_ai_event", return_value=MagicMock(status_code=201))
@patch("revenium_middleware.anthropic.middleware._safe_run_async_in_thread", side_effect=run_metering_synchronously)
class TestRawStream:
    @staticmethod
    def _consume(cache_creation="omitted"):
        kwargs = request_kwargs()
        kwargs["stream"] = True
        mock_wrapped = MagicMock(return_value=FakeStream(stream_events(cache_creation)))

        for _ in create_wrapper(mock_wrapped, None, (), kwargs):
            pass

    def test_split_is_forwarded_alongside_the_aggregate(self, mock_thread, mock_submit, mock_client):
        self._consume(ttl_split())

        payload = mock_submit.call_args[0][1]
        assert payload["is_streamed"] is True
        assert payload["cache_creation_token_count"] == 300
        assert payload["cache_creation5m_token_count"] == 200
        assert payload["cache_creation1h_token_count"] == 100

    def test_missing_split_leaves_only_the_aggregate(self, mock_thread, mock_submit, mock_client):
        self._consume()

        payload = mock_submit.call_args[0][1]
        assert payload["cache_creation_token_count"] == 300
        assert "cache_creation5m_token_count" not in payload
        assert "cache_creation1h_token_count" not in payload

    def test_none_valued_split_leaves_only_the_aggregate(self, mock_thread, mock_submit, mock_client):
        self._consume(ttl_split(None, None))

        payload = mock_submit.call_args[0][1]
        assert payload["cache_creation_token_count"] == 300
        assert "cache_creation5m_token_count" not in payload
        assert "cache_creation1h_token_count" not in payload

    def test_a_stream_that_never_starts_carries_no_buckets(self, mock_thread, mock_submit, mock_client):
        state = StreamUsageState()

        assert state.cache_creation_ttl_counts == {}

    def test_async_stream_forwards_the_split(self, mock_thread, mock_submit, mock_client):
        kwargs = request_kwargs()
        kwargs["stream"] = True

        async def fake_create(**call_kwargs):
            return FakeAsyncStream(stream_events(ttl_split()))

        async def scenario():
            stream = await async_create_wrapper(fake_create, None, (), kwargs)
            async for _ in stream:
                pass

        asyncio.run(scenario())

        payload = mock_submit.call_args[0][1]
        assert payload["is_streamed"] is True
        assert payload["cache_creation_token_count"] == 300
        assert payload["cache_creation5m_token_count"] == 200
        assert payload["cache_creation1h_token_count"] == 100

    def test_async_stream_without_a_split_leaves_only_the_aggregate(self, mock_thread, mock_submit, mock_client):
        kwargs = request_kwargs()
        kwargs["stream"] = True

        async def fake_create(**call_kwargs):
            return FakeAsyncStream(stream_events())

        async def scenario():
            stream = await async_create_wrapper(fake_create, None, (), kwargs)
            async for _ in stream:
                pass

        asyncio.run(scenario())

        payload = mock_submit.call_args[0][1]
        assert payload["cache_creation_token_count"] == 300
        assert "cache_creation5m_token_count" not in payload
        assert "cache_creation1h_token_count" not in payload


@patch("revenium_middleware.anthropic.middleware._get_thread_safe_client", return_value=MagicMock())
@patch("revenium_middleware.anthropic.middleware.submit_ai_event", return_value=MagicMock(status_code=201))
@patch("revenium_middleware.anthropic.middleware._safe_run_async_in_thread", side_effect=run_metering_synchronously)
class TestMessagesStream:
    @staticmethod
    def _consume(cache_creation="omitted"):
        mock_wrapped = MagicMock(return_value=FakeMessageStream(message(cache_creation)))

        with stream_wrapper(mock_wrapped, None, (), request_kwargs()):
            pass

    def test_split_is_forwarded_alongside_the_aggregate(self, mock_thread, mock_submit, mock_client):
        self._consume(ttl_split())

        payload = mock_submit.call_args[0][1]
        assert payload["is_streamed"] is True
        assert payload["cache_creation_token_count"] == 300
        assert payload["cache_creation5m_token_count"] == 200
        assert payload["cache_creation1h_token_count"] == 100

    def test_missing_split_leaves_only_the_aggregate(self, mock_thread, mock_submit, mock_client):
        self._consume()

        payload = mock_submit.call_args[0][1]
        assert payload["cache_creation_token_count"] == 300
        assert "cache_creation5m_token_count" not in payload
        assert "cache_creation1h_token_count" not in payload

    def test_none_valued_split_leaves_only_the_aggregate(self, mock_thread, mock_submit, mock_client):
        self._consume(ttl_split(None, None))

        payload = mock_submit.call_args[0][1]
        assert payload["cache_creation_token_count"] == 300
        assert "cache_creation5m_token_count" not in payload
        assert "cache_creation1h_token_count" not in payload
