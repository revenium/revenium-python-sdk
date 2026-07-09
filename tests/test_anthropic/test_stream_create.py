"""client.messages.create(stream=True) must not crash and must meter.

The raw-stream form returns anthropic.Stream/AsyncStream (no .usage/.id), which
the create wrappers previously dereferenced unconditionally.
"""
import asyncio
import datetime
import gc
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from revenium_middleware.anthropic import middleware as anthropic_middleware
from revenium_middleware.anthropic.provider import Provider
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


def make_events():
    """Minimal raw-stream event sequence as emitted by the Anthropic SDK."""
    return [
        SimpleNamespace(
            type="message_start",
            message=SimpleNamespace(
                id="msg_stream_1",
                model="claude-sonnet-5",
                usage=SimpleNamespace(
                    input_tokens=12,
                    output_tokens=1,
                    cache_creation_input_tokens=3,
                    cache_read_input_tokens=2,
                ),
            ),
        ),
        SimpleNamespace(
            type="content_block_delta",
            delta=SimpleNamespace(type="text_delta", text="Hello"),
        ),
        SimpleNamespace(
            type="message_delta",
            delta=SimpleNamespace(stop_reason="end_turn"),
            usage=SimpleNamespace(output_tokens=9),
        ),
        SimpleNamespace(type="message_stop"),
    ]


class FakeStream:
    """Stands in for anthropic.Stream: iterable, context manager, closeable.

    Deliberately has no .usage or .id attribute -- exactly like the real Stream.
    """

    def __init__(self, events):
        self._events = list(events)
        self.closed = False

    def __iter__(self):
        return iter(self._events)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def close(self):
        self.closed = True


class FakeAsyncStream:
    """Stands in for anthropic.AsyncStream. No .usage or .id attribute."""

    def __init__(self, events):
        self._events = list(events)
        self.closed = False

    def __aiter__(self):
        self._it = iter(self._events)
        return self

    async def __anext__(self):
        try:
            return next(self._it)
        except StopIteration:
            raise StopAsyncIteration

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.aclose()
        return False

    async def aclose(self):
        self.closed = True


def stream_kwargs():
    return {
        "messages": [{"role": "user", "content": "Hello"}],
        "model": "claude-sonnet-5",
        "max_tokens": 64,
        "stream": True,
        "usage_metadata": {"trace_id": "trace-stream", "organizationName": "AcmeCorp"},
    }


def run_metering_synchronously(coro_func, *args, **kwargs):
    """Run the metering coroutine to completion on a dedicated thread.

    Mirrors production (_safe_run_async_in_thread dispatches to a thread) and
    works whether or not the caller is already inside an event loop.
    """
    import threading

    thread = threading.Thread(target=lambda: asyncio.run(coro_func(*args, **kwargs)))
    thread.start()
    thread.join(timeout=10)
    return MagicMock()


@patch("revenium_middleware.anthropic.middleware._get_thread_safe_client", return_value=MagicMock())
@patch("revenium_middleware.anthropic.middleware.submit_ai_event", return_value=MagicMock(status_code=201))
@patch("revenium_middleware.anthropic.middleware._safe_run_async_in_thread", side_effect=run_metering_synchronously)
class TestSyncStreamTrue:
    def test_events_pass_through_unmodified(self, mock_thread, mock_submit, mock_client):
        events = make_events()
        mock_wrapped = MagicMock(return_value=FakeStream(events))

        result = create_wrapper(mock_wrapped, MagicMock(), (), stream_kwargs())

        assert list(result) == events

    def test_meters_once_with_stream_usage(self, mock_thread, mock_submit, mock_client):
        mock_wrapped = MagicMock(return_value=FakeStream(make_events()))

        result = create_wrapper(mock_wrapped, MagicMock(), (), stream_kwargs())
        for _ in result:
            pass

        assert mock_submit.call_count == 1
        payload = mock_submit.call_args[0][1]
        assert payload["is_streamed"] is True
        assert payload["input_token_count"] == 12
        assert payload["output_token_count"] == 9
        assert payload["cache_creation_token_count"] == 3
        assert payload["cache_read_token_count"] == 2
        assert payload["total_token_count"] == 21
        assert payload["stop_reason"] == "END"
        assert payload["transaction_id"] == "msg_stream_1"
        assert payload["model"] == "claude-sonnet-5"

    def test_context_manager_form_meters_once(self, mock_thread, mock_submit, mock_client):
        fake = FakeStream(make_events())
        mock_wrapped = MagicMock(return_value=fake)

        with create_wrapper(mock_wrapped, MagicMock(), (), stream_kwargs()) as stream:
            for _ in stream:
                pass

        assert fake.closed is True
        assert mock_submit.call_count == 1

    def test_early_break_still_meters_once(self, mock_thread, mock_submit, mock_client):
        mock_wrapped = MagicMock(return_value=FakeStream(make_events()))

        result = create_wrapper(mock_wrapped, MagicMock(), (), stream_kwargs())
        seen = 0
        for _ in result:
            seen += 1
            if seen == 2:
                break
        del result
        gc.collect()

        assert mock_submit.call_count == 1
        payload = mock_submit.call_args[0][1]
        # Only message_start had arrived; partial usage is better than none.
        assert payload["is_streamed"] is True
        assert payload["input_token_count"] == 12

    def test_non_streaming_path_unchanged(self, mock_thread, mock_submit, mock_client):
        response = MagicMock()
        response.id = "msg_plain"
        response.usage.input_tokens = 5
        response.usage.output_tokens = 7
        response.usage.cache_creation_input_tokens = 0
        response.usage.cache_read_input_tokens = 0
        response.stop_reason = "end_turn"
        response.model = "claude-sonnet-5"
        kwargs = stream_kwargs()
        del kwargs["stream"]
        mock_wrapped = MagicMock(return_value=response)

        result = create_wrapper(mock_wrapped, MagicMock(), (), kwargs)

        assert result is response
        assert mock_submit.call_count == 1
        assert mock_submit.call_args[0][1]["is_streamed"] is False


@patch("revenium_middleware.anthropic.middleware._get_thread_safe_client", return_value=MagicMock())
@patch("revenium_middleware.anthropic.middleware.submit_ai_event", return_value=MagicMock(status_code=201))
@patch("revenium_middleware.anthropic.middleware._safe_run_async_in_thread", side_effect=run_metering_synchronously)
class TestAsyncStreamTrue:
    def test_events_pass_through_unmodified(self, mock_thread, mock_submit, mock_client):
        events = make_events()

        async def fake_create(**kwargs):
            return FakeAsyncStream(events)

        async def scenario():
            stream = await async_create_wrapper(fake_create, MagicMock(), (), stream_kwargs())
            return [event async for event in stream]

        received = asyncio.run(scenario())
        assert received == events

    def test_meters_once_with_stream_usage(self, mock_thread, mock_submit, mock_client):
        async def fake_create(**kwargs):
            return FakeAsyncStream(make_events())

        async def scenario():
            stream = await async_create_wrapper(fake_create, MagicMock(), (), stream_kwargs())
            async for _ in stream:
                pass

        asyncio.run(scenario())

        assert mock_submit.call_count == 1
        payload = mock_submit.call_args[0][1]
        assert payload["is_streamed"] is True
        assert payload["input_token_count"] == 12
        assert payload["output_token_count"] == 9
        assert payload["stop_reason"] == "END"
        assert payload["transaction_id"] == "msg_stream_1"

    def test_context_manager_form_meters_once(self, mock_thread, mock_submit, mock_client):
        """__aexit__ must meter even when the stream was not exhausted.

        Consuming only 2 of 4 events means __aiter__'s finally has NOT fired
        by the time the with-block exits (the abandoned inner generator is only
        closed later, by the loop's asyncgen finalizer). So the call_count == 1
        assertion immediately after the with-block can only be satisfied by
        __aexit__'s finalize.
        """
        fake = FakeAsyncStream(make_events())

        async def fake_create(**kwargs):
            return fake

        async def scenario():
            stream = await async_create_wrapper(fake_create, MagicMock(), (), stream_kwargs())
            async with stream as s:
                seen = 0
                async for _ in s:
                    seen += 1
                    if seen == 2:
                        break

            # Immediately after __aexit__: the wrapper is still referenced and
            # the loop has not run the asyncgen finalizer for the abandoned
            # inner generator -- only __aexit__ can have metered.
            assert fake.closed is True
            assert mock_submit.call_count == 1

            # Now let GC and the loop finalize the abandoned inner generator;
            # its finally is idempotent, so the count must stay at 1.
            gc.collect()
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            assert mock_submit.call_count == 1
            return stream

        wrapper = asyncio.run(scenario())

        assert mock_submit.call_count == 1
        # Wrapper alive through every assertion above, so __del__ never ran.
        assert wrapper is not None

    def test_early_break_still_meters_once(self, mock_thread, mock_submit, mock_client):
        """Breaking out of iteration must meter via __aiter__'s finally.

        The wrapper stays referenced the whole time (no __del__) and no
        context manager is used (no __aexit__): after the break, only the
        loop's asyncgen finalizer closing the abandoned __aiter__ generator
        can fire the metering.
        """
        async def fake_create(**kwargs):
            return FakeAsyncStream(make_events())

        async def scenario():
            stream = await async_create_wrapper(fake_create, MagicMock(), (), stream_kwargs())
            seen = 0
            async for _ in stream:
                seen += 1
                if seen == 2:
                    break

            # Drive the loop so its asyncgen finalizer closes the abandoned
            # __aiter__ generator, firing the finally-block metering while the
            # wrapper itself is still alive.
            gc.collect()
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            assert mock_submit.call_count == 1
            return stream

        wrapper = asyncio.run(scenario())

        assert mock_submit.call_count == 1
        payload = mock_submit.call_args[0][1]
        # Only message_start had arrived; partial usage is better than none.
        assert payload["is_streamed"] is True
        assert payload["input_token_count"] == 12
        # Wrapper alive through every assertion above, so __del__ never ran.
        assert wrapper is not None


@patch("revenium_middleware.anthropic.middleware._get_thread_safe_client", return_value=MagicMock())
@patch("revenium_middleware.anthropic.middleware.submit_ai_event", return_value=MagicMock(status_code=201))
@patch("revenium_middleware.anthropic.middleware._safe_run_async_in_thread", side_effect=run_metering_synchronously)
class TestMeterRawStream:
    def test_completion_start_time_uses_first_event_time(self, mock_thread, mock_submit, mock_client):
        """completion_start_time must be the first-event timestamp, not wall-clock.

        Uses a first_event_time_dt far in the past so a wall-clock value
        (datetime.now) could never equal the expected string.
        """
        state = StreamUsageState()
        state.saw_message_start = True
        state.message_id = "msg_cst"
        state.model = "claude-x"
        state.input_tokens = 10
        state.output_tokens = 4
        state.cache_creation_input_tokens = 0
        state.cache_read_input_tokens = 0
        state.first_event_time_dt = datetime.datetime(
            2023, 1, 1, 12, 0, 5, tzinfo=datetime.timezone.utc
        )

        anthropic_middleware._meter_raw_stream(
            state,
            {},
            {"model": "claude-x"},
            "2023-01-01T12:00:00Z",
            datetime.datetime(2023, 1, 1, 12, 0, 0, tzinfo=datetime.timezone.utc),
            Provider.ANTHROPIC,
        )

        assert mock_submit.call_count == 1
        payload = mock_submit.call_args[0][1]
        assert payload["completion_start_time"] == "2023-01-01T12:00:05Z"
        assert payload["time_to_first_token"] == 5000
