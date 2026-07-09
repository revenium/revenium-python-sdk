"""The middleware must not alter what the caller sees when it injects
stream_options.include_usage for metering.

The injected option makes OpenAI append a final chunk with empty choices and
populated usage. That chunk exists only because of the middleware, so it must
be consumed for metering but hidden from the caller -- unless the caller asked
for include_usage themselves.
"""
import asyncio
import datetime
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from revenium_middleware.openai.middleware import create_wrapper, async_create_wrapper


def run_coro_in_thread(coro):
    thread = threading.Thread(target=lambda: asyncio.run(coro))
    thread.start()
    thread.join(timeout=10)
    return MagicMock()


def make_content_chunk(i):
    return SimpleNamespace(
        id="chatcmpl-injection-test",
        model="gpt-4o-mini",
        choices=[SimpleNamespace(delta=SimpleNamespace(content=f"tok{i}"), finish_reason=None)],
        system_fingerprint=None,
    )


def make_synthetic_usage_chunk():
    return SimpleNamespace(
        id="chatcmpl-injection-test",
        model="gpt-4o-mini",
        choices=[],
        usage=SimpleNamespace(prompt_tokens=7, completion_tokens=3, total_tokens=10),
        system_fingerprint=None,
    )


def base_kwargs(**extra):
    kwargs = {
        "messages": [{"role": "user", "content": "hi"}],
        "model": "gpt-4o-mini",
        "stream": True,
    }
    kwargs.update(extra)
    return kwargs


class FakeAsyncStream:
    def __init__(self, chunks):
        self._it = iter(chunks)

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self._it)
        except StopIteration:
            raise StopAsyncIteration


@patch("revenium_middleware.openai.middleware.client", object())
@patch("revenium_middleware.openai.middleware.submit_ai_event", return_value=SimpleNamespace(id="evt-1"))
@patch("revenium_middleware.openai.middleware.run_async_in_thread", side_effect=run_coro_in_thread)
class TestSyncStreamOptionsInjection:
    def test_synthetic_usage_chunk_hidden_when_caller_did_not_request(
        self, mock_run, mock_submit
    ):
        chunks = [make_content_chunk(0), make_content_chunk(1), make_synthetic_usage_chunk()]
        mock_wrapped = MagicMock(return_value=iter(chunks))

        result = create_wrapper(mock_wrapped, None, (), base_kwargs())
        visible = list(result)

        # The caller sees exactly what a raw call would have produced.
        assert visible == chunks[:2]
        # The middleware still requested usage from the API...
        sent_kwargs = mock_wrapped.call_args.kwargs
        assert sent_kwargs["stream_options"]["include_usage"] is True
        # ...and metering still received the real token counts.
        assert mock_submit.call_count == 1
        payload = mock_submit.call_args[0][1]
        assert payload["input_token_count"] == 7
        assert payload["output_token_count"] == 3

    def test_usage_chunk_visible_when_caller_requested_it(self, mock_run, mock_submit):
        chunks = [make_content_chunk(0), make_synthetic_usage_chunk()]
        mock_wrapped = MagicMock(return_value=iter(chunks))

        result = create_wrapper(
            mock_wrapped, None, (), base_kwargs(stream_options={"include_usage": True})
        )
        visible = list(result)

        # Caller opted in: the usage chunk is theirs to see.
        assert visible == chunks
        assert mock_submit.call_count == 1


@patch("revenium_middleware.openai.middleware.client", object())
@patch("revenium_middleware.openai.middleware.submit_ai_event", return_value=SimpleNamespace(id="evt-1"))
@patch("revenium_middleware.openai.middleware.run_async_in_thread", side_effect=run_coro_in_thread)
class TestAsyncStreamOptionsInjection:
    def test_synthetic_usage_chunk_hidden_when_caller_did_not_request(
        self, mock_run, mock_submit
    ):
        chunks = [make_content_chunk(0), make_content_chunk(1), make_synthetic_usage_chunk()]
        sent_kwargs = []

        async def fake_create(**kwargs):
            sent_kwargs.append(kwargs)
            return FakeAsyncStream(chunks)

        async def scenario():
            stream = await async_create_wrapper(fake_create, None, (), base_kwargs())
            return [chunk async for chunk in stream]

        visible = asyncio.run(scenario())

        assert visible == chunks[:2]
        # The middleware still requested usage from the API...
        assert sent_kwargs[0]["stream_options"]["include_usage"] is True
        assert mock_submit.call_count == 1
        payload = mock_submit.call_args[0][1]
        assert payload["input_token_count"] == 7
        assert payload["output_token_count"] == 3

    def test_usage_chunk_visible_when_caller_requested_it(self, mock_run, mock_submit):
        chunks = [make_content_chunk(0), make_synthetic_usage_chunk()]

        async def fake_create(**kwargs):
            return FakeAsyncStream(chunks)

        async def scenario():
            stream = await async_create_wrapper(
                fake_create, None, (), base_kwargs(stream_options={"include_usage": True})
            )
            return [chunk async for chunk in stream]

        visible = asyncio.run(scenario())

        assert visible == chunks
        assert mock_submit.call_count == 1
