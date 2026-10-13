"""Every OpenAI chat and Responses entry point, driven through the real
``openai`` client over ``httpx.MockTransport``, meters exactly once and leaves
what the caller sees unchanged.

The baseline for "unchanged" is the same call with the wrappers passing
through untouched (selective metering on, outside any decorated function).
"""
import asyncio
import datetime
import gc
import json
import logging
import itertools
import sys
import threading
import time
import weakref
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import httpx
import openai
import pydantic
import pytest
import wrapt
from openai import NOT_GIVEN, AsyncAzureOpenAI, AsyncOpenAI, AzureOpenAI, OpenAI, Omit
from openai.resources.responses import AsyncResponses, Responses

from revenium_middleware._core import sse_body_tap
from revenium_middleware.openai import middleware as mw

from .responses_stream_events import TERMINAL_STATUS, MODEL, event_payloads, response_payload, usage_payload

RESPONSES_USAGE = usage_payload(input_tokens=11, output_tokens=7, cached_tokens=3, reasoning_tokens=2)
CHAT_USAGE = {
    "prompt_tokens": 11,
    "completion_tokens": 7,
    "total_tokens": 18,
    "prompt_tokens_details": {"cached_tokens": 3},
    "completion_tokens_details": {"reasoning_tokens": 2},
}
CHAT_REQUEST = {"model": MODEL, "messages": [{"role": "user", "content": "hi"}]}
RESPONSES_REQUEST = {"model": MODEL, "input": "hi"}
MISSING_USAGE_WARNING = "No usage data found in streaming Responses API response!"
METERED_FIELDS = (
    "model", "input_token_count", "output_token_count", "total_token_count",
    "cache_read_token_count", "reasoning_token_count",
)


def _sse(events, done_marker):
    lines = []
    for event in events:
        if "type" in event:
            lines.append(f"event: {event['type']}\n")
        lines.append(f"data: {json.dumps(event)}\n\n")
    if done_marker:
        lines.append("data: [DONE]\n\n")
    return "".join(lines).encode()


AZURE_ENDPOINT = "https://revenium-test.openai.azure.com"
AZURE_API_VERSION = "2024-10-21"
# Azure opens every stream with its prompt content-filter verdict: no choices,
# and an empty id and model.
AZURE_PROMPT_FILTER_CHUNK = {
    "choices": [], "created": 0, "id": "", "model": "", "object": "",
    "prompt_filter_results": [{"prompt_index": 0, "content_filter_results": {
        "hate": {"filtered": False, "severity": "safe"}}}],
}


def _chat_chunks(include_usage, azure=False, finish_reason="stop", usage=None):
    base = {"id": "chatcmpl-1", "object": "chat.completion.chunk", "created": 1, "model": MODEL}
    chunks = [
        {**base, "choices": [{"index": 0, "delta": {"role": "assistant", "content": "hi"},
                              "finish_reason": None}]},
        {**base, "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}]},
    ]
    if include_usage:
        chunks.append({**base, "choices": [], "usage": CHAT_USAGE if usage is None else usage})
    if azure:
        chunks.insert(0, AZURE_PROMPT_FILTER_CHUNK)
    return chunks


BODY_CHUNK_SIZE = 64


def in_chunks_of(size):
    def shape(body):
        return [body[i:i + size] for i in range(0, len(body), size)]
    return shape


def whole_body(body):
    return [body]


def split_inside_the_usage_event(body):
    cut = body.index(b'"usage"') if b'"usage"' in body else len(body) // 2
    return [body[:cut], body[cut:]]


def crlf_events(body):
    return [body.replace(b"\n", b"\r\n")]


def crlf_split_between_cr_and_lf(body):
    return crlf_events(body)[0].replace(b"\r\n", b"\r\x00\n").split(b"\x00")


class ChunkedBody(httpx.SyncByteStream, httpx.AsyncByteStream):
    """A response body served chunk by chunk, sync or async. A Response built
    from bytes is read on construction, which would hide whether the body was
    read, and in what pieces. ``fail_after`` raises a read error once that many
    chunks have been served."""

    def __init__(self, chunks, fail_after=None):
        self._chunks = chunks
        self._fail_after = fail_after

    def _served(self):
        for number, chunk in enumerate(self._chunks):
            if number == self._fail_after:
                raise httpx.ReadError("connection reset mid-body")
            yield chunk

    def __iter__(self):
        yield from self._served()

    async def __aiter__(self):
        for chunk in self._served():
            yield chunk


class FakeOpenAI:
    """Serves chat and Responses calls and records what each request asked for.

    Streamed bodies are served in ``shape(body)`` chunks."""

    def __init__(self, responses_terminal="response.completed", shape=in_chunks_of(BODY_CHUNK_SIZE),
                 azure=False, fail_after=None, chat_finish_reason="stop", chat_usage=None, fail_status=None,
                 fail_message="refused", chat_content=None, first_content=None):
        self.requests = []
        self._chat_content = chat_content
        self._first_content = first_content
        self._fail_status = fail_status
        self._fail_message = fail_message
        # A refused call is raised at once rather than retried after a back-off.
        self._client_options = {"max_retries": 0} if fail_status else {}
        self._chat_usage = chat_usage
        self._responses_terminal = responses_terminal
        self._chat_finish_reason = chat_finish_reason
        self._shape = shape
        self._azure = azure
        self._fail_after = fail_after

    def _stream(self, body):
        return httpx.Response(200, headers={"content-type": "text/event-stream"},
                              stream=ChunkedBody(self._shape(body), self._fail_after))

    def __call__(self, request):
        body = json.loads(request.content or b"{}")
        self.requests.append(body)
        if self._fail_status:
            return httpx.Response(self._fail_status, json={"error": {
                "message": self._fail_message, "type": "invalid_request_error", "param": None, "code": "refused"}})
        if request.url.path.endswith("/chat/completions"):
            if body.get("stream"):
                include_usage = bool((body.get("stream_options") or {}).get("include_usage"))
                return self._stream(_sse(self._streamed_chat_chunks(include_usage), True))
            return httpx.Response(200, json={
                "id": "chatcmpl-1", "object": "chat.completion", "created": 1, "model": MODEL,
                "usage": CHAT_USAGE if self._chat_usage is None else self._chat_usage,
                "choices": [{"index": 0, "finish_reason": "stop",
                             "message": {"role": "assistant", "content": "hi"}}],
            })
        if request.url.path.endswith("/responses"):
            if body.get("stream"):
                events = event_payloads(self._responses_terminal, RESPONSES_USAGE)
                return self._stream(_sse(events, False))
            status = TERMINAL_STATUS[self._responses_terminal]
            return httpx.Response(200, json=response_payload(status=status, usage=RESPONSES_USAGE))
        return httpx.Response(404, json={"error": "unrouted"})

    def _streamed_chat_chunks(self, include_usage):
        chunks = _chat_chunks(include_usage, self._azure, self._chat_finish_reason, self._chat_usage)
        if self._chat_content is not None:
            first_with_content = next(chunk for chunk in chunks if chunk["choices"])
            first_with_content["choices"][0]["delta"]["content"] = self._chat_content
        if self._first_content is not None:
            first = next(index for index, chunk in enumerate(chunks) if chunk["choices"])
            chunks.insert(first, {**chunks[first], "choices": [{
                "index": 0, "delta": {"role": "assistant", "content": self._first_content}, "finish_reason": None}]})
        return chunks

    def sync_client(self):
        return OpenAI(api_key="sk-test", base_url="https://api.openai.test/v1",
                      http_client=httpx.Client(transport=httpx.MockTransport(self)), **self._client_options)

    def _async_handler(self):
        async def handler(request):
            await request.aread()
            return self(request)
        return handler

    def async_client(self):
        return AsyncOpenAI(api_key="sk-test", base_url="https://api.openai.test/v1",
                           http_client=httpx.AsyncClient(transport=httpx.MockTransport(self._async_handler())),
                           **self._client_options)

    def azure_client(self):
        return AzureOpenAI(api_key="az-test", azure_endpoint=AZURE_ENDPOINT, api_version=AZURE_API_VERSION,
                           http_client=httpx.Client(transport=httpx.MockTransport(self)))

    def async_azure_client(self):
        return AsyncAzureOpenAI(api_key="az-test", azure_endpoint=AZURE_ENDPOINT, api_version=AZURE_API_VERSION,
                                http_client=httpx.AsyncClient(
                                    transport=httpx.MockTransport(self._async_handler())))


def _run_coro_to_completion(coro):
    errors = []

    def target():
        try:
            asyncio.run(coro)
        except Exception as exc:  # surfaced to the test below
            errors.append(exc)

    thread = threading.Thread(target=target)
    thread.start()
    thread.join()
    if errors:
        raise errors[0]
    return thread


@pytest.fixture
def payloads():
    recorded = []

    def record(kind, args, idempotency_key=None):
        recorded.append(args)
        return SimpleNamespace(id="evt-test")

    with patch.object(mw, "current_client", lambda: object()), \
         patch.object(mw, "submit_ai_event", side_effect=record), \
         patch.object(mw, "run_async_in_thread", side_effect=_run_coro_to_completion):
        yield recorded


# ---- calls: each returns what the caller saw ------------------------------

def _chat_kwargs(stream_options):
    return CHAT_REQUEST if stream_options is None else {**CHAT_REQUEST, "stream_options": stream_options}


def _responses_kwargs(stream_options):
    return RESPONSES_REQUEST if stream_options is None else {**RESPONSES_REQUEST, "stream_options": stream_options}


def sync_chat_create(fake, _options):
    return [fake.sync_client().chat.completions.create(**CHAT_REQUEST)]


def sync_chat_create_stream(fake, options):
    return list(fake.sync_client().chat.completions.create(**_chat_kwargs(options), stream=True))


def sync_chat_stream(fake, options):
    with fake.sync_client().chat.completions.stream(**_chat_kwargs(options)) as stream:
        return [event.type for event in stream]


def async_chat_create(fake, _options):
    async def go():
        return [await fake.async_client().chat.completions.create(**CHAT_REQUEST)]
    return asyncio.run(go())


def async_chat_create_stream(fake, options):
    async def go():
        stream = await fake.async_client().chat.completions.create(**_chat_kwargs(options), stream=True)
        return [chunk async for chunk in stream]
    return asyncio.run(go())


def async_chat_stream(fake, options):
    async def go():
        async with fake.async_client().chat.completions.stream(**_chat_kwargs(options)) as stream:
            return [event.type async for event in stream]
    return asyncio.run(go())


def sync_responses_create(fake, _options):
    return [fake.sync_client().responses.create(**RESPONSES_REQUEST)]


def sync_responses_create_stream(fake, options):
    return [event.type for event in
            fake.sync_client().responses.create(**_responses_kwargs(options), stream=True)]


def sync_responses_stream(fake, options):
    with fake.sync_client().responses.stream(**_responses_kwargs(options)) as stream:
        events = [event.type for event in stream]
        assert stream.get_final_response().usage.total_tokens == RESPONSES_USAGE["total_tokens"]
        return events


def async_responses_create(fake, _options):
    async def go():
        return [await fake.async_client().responses.create(**RESPONSES_REQUEST)]
    return asyncio.run(go())


def async_responses_create_stream(fake, options):
    async def go():
        stream = await fake.async_client().responses.create(**_responses_kwargs(options), stream=True)
        return [event.type async for event in stream]
    return asyncio.run(go())


def async_responses_stream(fake, options):
    async def go():
        async with fake.async_client().responses.stream(**_responses_kwargs(options)) as stream:
            events = [event.type async for event in stream]
            final = await stream.get_final_response()
            assert final.usage.total_tokens == RESPONSES_USAGE["total_tokens"]
            return events
    return asyncio.run(go())


# The raw-response forms LangChain uses when it wants the response headers
# (BACK-3999): the caller gets OpenAI's own response wrapper and parses it.

def sync_chat_raw_stream(fake, options):
    raw = fake.sync_client().chat.completions.with_raw_response.create(**_chat_kwargs(options), stream=True)
    return list(raw.parse())


def sync_chat_streaming_response_stream(fake, options):
    with fake.sync_client().chat.completions.with_streaming_response.create(
            **_chat_kwargs(options), stream=True) as response:
        return list(response.parse())


def async_chat_raw_stream(fake, options):
    async def go():
        raw = await fake.async_client().chat.completions.with_raw_response.create(
            **_chat_kwargs(options), stream=True)
        return [chunk async for chunk in raw.parse()]
    return asyncio.run(go())


def async_chat_streaming_response_stream(fake, options):
    async def go():
        async with fake.async_client().chat.completions.with_streaming_response.create(
                **_chat_kwargs(options), stream=True) as response:
            return [chunk async for chunk in await response.parse()]
    return asyncio.run(go())


def sync_responses_raw_stream(fake, options):
    raw = fake.sync_client().responses.with_raw_response.create(**_responses_kwargs(options), stream=True)
    return list(raw.parse())


def sync_responses_streaming_response_stream(fake, options):
    with fake.sync_client().responses.with_streaming_response.create(
            **_responses_kwargs(options), stream=True) as response:
        return list(response.parse())


def async_responses_raw_stream(fake, options):
    async def go():
        raw = await fake.async_client().responses.with_raw_response.create(
            **_responses_kwargs(options), stream=True)
        return [event async for event in raw.parse()]
    return asyncio.run(go())


def async_responses_streaming_response_stream(fake, options):
    async def go():
        async with fake.async_client().responses.with_streaming_response.create(
                **_responses_kwargs(options), stream=True) as response:
            return [event async for event in await response.parse()]
    return asyncio.run(go())


@dataclass(frozen=True)
class Case:
    call: object
    options: object = None

    def __str__(self):
        suffix = "" if self.options is None else "-caller_stream_options"
        return f"{self.call.__name__}{suffix}"


CHAT_OPTIONS = {"include_usage": True}
RESPONSES_OPTIONS = {"include_obfuscation": False}
NON_STREAMED = [sync_chat_create, async_chat_create, sync_responses_create, async_responses_create]
RAW_STREAMED_CHAT = [sync_chat_raw_stream, sync_chat_streaming_response_stream,
                     async_chat_raw_stream, async_chat_streaming_response_stream]
RAW_STREAMED_RESPONSES = [sync_responses_raw_stream, sync_responses_streaming_response_stream,
                          async_responses_raw_stream, async_responses_streaming_response_stream]
STREAMED_CHAT = [sync_chat_create_stream, sync_chat_stream, async_chat_create_stream, async_chat_stream,
                 *RAW_STREAMED_CHAT]
STREAMED_RESPONSES = [sync_responses_create_stream, sync_responses_stream,
                      async_responses_create_stream, async_responses_stream, *RAW_STREAMED_RESPONSES]

# stream_options only means something on a streamed call, so the non-streamed
# calls run once, without it.
CASES = (
    [Case(call) for call in NON_STREAMED]
    + [Case(call, options) for call in STREAMED_CHAT for options in (None, CHAT_OPTIONS)]
    + [Case(call, options) for call in STREAMED_RESPONSES for options in (None, RESPONSES_OPTIONS)]
)


def _caller_view(items):
    return [item.model_dump() if hasattr(item, "model_dump") else item for item in items]


@pytest.mark.parametrize("case", CASES, ids=str)
def test_entry_point_meters_once_and_caller_view_matches_baseline(case, payloads, caplog):
    with patch.object(mw, "is_selective_metering_enabled", return_value=True):
        baseline = case.call(FakeOpenAI(), case.options)
    assert payloads == []

    with caplog.at_level(logging.WARNING, logger="revenium_middleware.extension"):
        metered = case.call(FakeOpenAI(), case.options)

    assert len(payloads) == 1
    assert _caller_view(metered) == _caller_view(baseline)
    assert MISSING_USAGE_WARNING not in caplog.text
    del metered
    gc.collect()
    assert len(payloads) == 1


@pytest.mark.parametrize("call", STREAMED_CHAT, ids=lambda c: c.__name__)
@pytest.mark.parametrize("options", [None, CHAT_OPTIONS, {"include_obfuscation": False}],
                         ids=["no_options", "include_usage", "other_options"])
def test_chat_stream_requests_usage_and_keeps_caller_options(call, options, payloads):
    fake = FakeOpenAI()

    call(fake, options)

    sent = fake.requests[0]["stream_options"]
    assert sent["include_usage"] is True
    for key, value in (options or {}).items():
        assert sent[key] == value
    assert payloads[0]["total_token_count"] == CHAT_USAGE["total_tokens"]


@pytest.mark.parametrize("call", [sync_chat_create_stream, async_chat_create_stream],
                         ids=lambda c: c.__name__)
def test_injected_usage_chunk_is_visible_only_when_requested(call, payloads):
    hidden = call(FakeOpenAI(), None)
    requested = call(FakeOpenAI(), CHAT_OPTIONS)

    assert not any(chunk.usage for chunk in hidden)
    assert [chunk.usage.total_tokens for chunk in requested if chunk.usage] == [CHAT_USAGE["total_tokens"]]


# ---- raw-response streams read as bytes (BACK-3999) -------------------------

def _sync_streamed_bytes(fake, read):
    with fake.sync_client().chat.completions.with_streaming_response.create(**CHAT_REQUEST, stream=True) as r:
        return read(r)


def _async_streamed_bytes(fake, read):
    async def go():
        async with fake.async_client().chat.completions.with_streaming_response.create(
                **CHAT_REQUEST, stream=True) as r:
            return await read(r)
    return asyncio.run(go())


def _all_bytes(response):
    return list(response.iter_bytes())


async def _all_bytes_async(response):
    return [chunk async for chunk in response.iter_bytes()]


def _seven_byte_pieces(response):
    return list(response.iter_bytes(chunk_size=7))


async def _seven_byte_pieces_async(response):
    return [chunk async for chunk in response.iter_bytes(chunk_size=7)]


BYTE_READERS = [
    pytest.param(_sync_streamed_bytes, _all_bytes, id="sync"),
    pytest.param(_async_streamed_bytes, _all_bytes_async, id="async"),
]
BODY_SHAPES = [
    pytest.param(whole_body, id="whole_body"),
    pytest.param(in_chunks_of(1), id="one_byte_chunks"),
    pytest.param(split_inside_the_usage_event, id="split_inside_usage_event"),
    pytest.param(crlf_split_between_cr_and_lf, id="crlf_split_between_cr_and_lf"),
    pytest.param(crlf_events, id="crlf_events"),
]


@pytest.mark.parametrize("shape", BODY_SHAPES)
@pytest.mark.parametrize("streamed_bytes, read", BYTE_READERS)
def test_raw_stream_read_as_bytes_meters_once_and_matches_baseline(streamed_bytes, read, shape, payloads):
    with patch.object(mw, "is_selective_metering_enabled", return_value=True):
        baseline = b"".join(streamed_bytes(FakeOpenAI(shape=shape), read))

    metered = b"".join(streamed_bytes(FakeOpenAI(shape=shape), read))

    assert metered == baseline
    assert len(payloads) == 1
    assert payloads[0]["total_token_count"] == CHAT_USAGE["total_tokens"]


@pytest.mark.parametrize("streamed_bytes, read", [
    pytest.param(_sync_streamed_bytes, _seven_byte_pieces, id="sync"),
    pytest.param(_async_streamed_bytes, _seven_byte_pieces_async, id="async"),
])
def test_raw_stream_keeps_the_callers_chunk_size(streamed_bytes, read, payloads):
    with patch.object(mw, "is_selective_metering_enabled", return_value=True):
        baseline = streamed_bytes(FakeOpenAI(), read)

    metered = streamed_bytes(FakeOpenAI(), read)

    assert metered == baseline
    assert {len(piece) for piece in metered[:-1]} == {7}
    assert len(payloads) == 1


def _read_first_chunk_then_close(fake):
    with fake.sync_client().chat.completions.with_streaming_response.create(**CHAT_REQUEST, stream=True) as r:
        stream = r.parse()
        first = next(iter(stream))
        stream.close()
    return first


@pytest.mark.filterwarnings("error::pytest.PytestUnraisableExceptionWarning")
def test_raw_stream_early_close_meters_once(payloads):
    first = _read_first_chunk_then_close(FakeOpenAI(shape=in_chunks_of(16)))
    gc.collect()

    assert first.choices[0].delta.content == "hi"
    assert len(payloads) == 1
    assert payloads[0]["transaction_id"] == "chatcmpl-1"


def _sync_close_mid_read(fake, payloads):
    with fake.sync_client().chat.completions.with_streaming_response.create(**CHAT_REQUEST, stream=True) as r:
        pieces = r.iter_bytes()
        next(pieces)
        r.close()
        sent_on_close = len(payloads)
        return sent_on_close, pieces


def _async_close_mid_read(fake, payloads):
    async def go():
        async with fake.async_client().chat.completions.with_streaming_response.create(
                **CHAT_REQUEST, stream=True) as r:
            pieces = r.iter_bytes()
            await pieces.__anext__()
            await r.close()
            sent_on_close = len(payloads)
            await pieces.aclose()
            return sent_on_close, pieces
    return asyncio.run(go())


@pytest.mark.parametrize("close_mid_read", [_sync_close_mid_read, _async_close_mid_read], ids=["sync", "async"])
def test_closing_a_partly_read_raw_stream_sends_the_record(close_mid_read, payloads):
    sent_on_close, pieces = close_mid_read(FakeOpenAI(shape=in_chunks_of(16)), payloads)

    assert sent_on_close == 1
    assert payloads[0]["transaction_id"] == "chatcmpl-1"
    del pieces
    gc.collect()
    assert len(payloads) == 1


def without_the_last_terminator(body):
    return [body[:body.rindex(b"\n\ndata: [DONE]")]]


@pytest.mark.parametrize("streamed_bytes, read", BYTE_READERS)
def test_unterminated_usage_event_at_the_end_stays_hidden(streamed_bytes, read, payloads):
    shape = without_the_last_terminator
    with patch.object(mw, "is_selective_metering_enabled", return_value=True):
        baseline = b"".join(streamed_bytes(FakeOpenAI(shape=shape), read))

    metered = b"".join(streamed_bytes(FakeOpenAI(shape=shape), read))

    assert b'"usage"' not in metered
    assert metered.rstrip(b"\n") == baseline
    assert len(payloads) == 1
    assert payloads[0]["total_token_count"] == CHAT_USAGE["total_tokens"]


def _error_the_bare_sdk_raises(call):
    """openai 2 lets a mid-body read error through as httpx.ReadError; openai 3
    wraps it in APIConnectionError. The middleware must raise whatever the SDK does."""
    from openai.resources.chat.completions import Completions

    create = vars(Completions)["create"]
    while isinstance(create, wrapt.FunctionWrapper):
        create = create.__wrapped__
    with patch.object(Completions, "create", create):
        with pytest.raises(Exception) as bare:
            call()
    return type(bare.value)


@pytest.mark.filterwarnings("error::pytest.PytestUnraisableExceptionWarning")
def test_raw_stream_read_error_propagates(payloads):
    # 320 bytes arrive: the first chunk's event whole, the second cut off.
    def cut_off_raw_stream():
        return sync_chat_raw_stream(FakeOpenAI(shape=in_chunks_of(64), fail_after=5), None)

    expected = _error_the_bare_sdk_raises(cut_off_raw_stream)
    with pytest.raises(expected):
        cut_off_raw_stream()
    gc.collect()

    assert len(payloads) == 1
    assert payloads[0]["transaction_id"] == "chatcmpl-1"
    assert payloads[0]["stop_reason"] == "ERROR"


def test_unread_streaming_response_sends_no_record(payloads):
    with FakeOpenAI().sync_client().chat.completions.with_streaming_response.create(
            **CHAT_REQUEST, stream=True) as response:
        pass
    gc.collect()

    assert response.http_response.is_stream_consumed is False
    assert payloads == []


def _azure_raw_stream(fake):
    return list(fake.azure_client().chat.completions.with_raw_response.create(**CHAT_REQUEST, stream=True).parse())


def _azure_stream(fake):
    return list(fake.azure_client().chat.completions.create(**CHAT_REQUEST, stream=True))


def _async_azure_stream(fake):
    async def go():
        stream = await fake.async_azure_client().chat.completions.create(**CHAT_REQUEST, stream=True)
        return [chunk async for chunk in stream]
    return asyncio.run(go())


def _assert_one_azure_record(payloads):
    assert len(payloads) == 1
    assert payloads[0]["provider"] == "Azure"
    assert payloads[0]["transaction_id"] == "chatcmpl-1"
    assert payloads[0]["model"] not in ("", "unknown")
    assert payloads[0]["total_token_count"] == CHAT_USAGE["total_tokens"]


def test_azure_raw_stream_meters_once(payloads):
    _azure_raw_stream(FakeOpenAI(azure=True))

    _assert_one_azure_record(payloads)


@pytest.mark.parametrize("call", [_azure_stream, _async_azure_stream], ids=["sync", "async"])
def test_azure_stream_meters_once(call, payloads):
    call(FakeOpenAI(azure=True))

    _assert_one_azure_record(payloads)


def test_raw_response_detection(payloads):
    from openai import Stream
    from openai.types.chat import ChatCompletion

    from revenium_middleware.openai.raw_response import is_raw_response

    client = FakeOpenAI().sync_client()
    raw = client.chat.completions.with_raw_response.create(**CHAT_REQUEST)
    with client.chat.completions.with_streaming_response.create(**CHAT_REQUEST) as streaming:
        assert is_raw_response(streaming)
    assert is_raw_response(raw)
    assert not is_raw_response(raw.parse())
    assert not is_raw_response(ChatCompletion.model_validate(raw.parse().model_dump()))
    assert not is_raw_response(Stream.__new__(Stream))
    assert not is_raw_response(MagicMock())
    assert not is_raw_response(iter(()))


# ---- raw-response streams from servers that omit null fields (BACK-4254) ---

_SPEC_CHAT_CHUNKS = _chat_chunks


def _without_null_finish_reasons(chunks):
    for chunk in chunks:
        for choice in chunk.get("choices", []):
            if choice.get("finish_reason", "absent") is None:
                del choice["finish_reason"]
    return chunks


def _chunks_without_null_fields(include_usage, azure=False, finish_reason="stop", usage=None):
    """What live grok-4.7 sends, and the OpenAI spec body with nulls left out."""
    return _without_null_finish_reasons(_SPEC_CHAT_CHUNKS(include_usage, azure, finish_reason, usage))


def _xai_documented_chunks(include_usage, azure=False, finish_reason="stop", usage=None):
    """The docs.x.ai stream: no finish_reason anywhere, usage on every chunk."""
    base = {"id": "chatcmpl-1", "object": "chat.completion.chunk", "created": 1, "model": MODEL,
            "usage": CHAT_USAGE if usage is None else usage}
    return [
        {**base, "choices": [{"index": 0, "delta": {"role": "assistant", "content": "h"}}]},
        {**base, "choices": [{"index": 0, "delta": {"content": "i"}}]},
    ]


def _usage_chunk_without_a_choices_key(include_usage, azure=False, finish_reason="stop", usage=None):
    chunks = _chunks_without_null_fields(include_usage, azure, finish_reason, usage)
    for chunk in chunks:
        if "usage" in chunk and not chunk["choices"]:
            del chunk["choices"]
    return chunks


NULL_OMITTING_BODIES = [_chunks_without_null_fields, _xai_documented_chunks, _usage_chunk_without_a_choices_key]


@pytest.mark.parametrize("body", NULL_OMITTING_BODIES, ids=lambda b: b.__name__.strip("_"))
@pytest.mark.parametrize("call", RAW_STREAMED_CHAT, ids=lambda c: c.__name__)
def test_a_raw_stream_whose_chunks_omit_null_fields_meters_like_the_plain_stream(call, body, payloads, monkeypatch):
    monkeypatch.setattr(sys.modules[__name__], "_chat_chunks", body)
    plain = sync_chat_create_stream(FakeOpenAI(), None)
    assert len(payloads) == 1
    baseline = payloads.pop()

    raw = call(FakeOpenAI(), None)

    assert len(payloads) == 1
    assert _metered_values(payloads[0]) == _metered_values(baseline)
    assert payloads[0]["stop_reason"] == baseline["stop_reason"]
    assert _caller_view(raw) == _caller_view(plain)


def test_an_unreadable_raw_stream_event_is_logged_once_at_warning(payloads, caplog):
    with patch.object(mw, "is_selective_metering_enabled", return_value=True):
        sent_by_the_server = sync_chat_raw_stream(FakeOpenAI(), CHAT_OPTIONS)
    parse_event = sse_body_tap._parse_event
    reads = []

    def unreadable_first_event(raw_event):
        reads.append(raw_event)
        if len(reads) == 1:
            raise RuntimeError("unreadable event")
        return parse_event(raw_event)

    with patch.object(sse_body_tap, "_parse_event", side_effect=unreadable_first_event), \
            caplog.at_level(logging.WARNING, logger="revenium_middleware.extension"):
        received = sync_chat_raw_stream(FakeOpenAI(), None)

    warnings = [r for r in caplog.records
                if r.levelno == logging.WARNING and "could not be metered" in r.getMessage()]
    assert len(warnings) == 1
    assert "chatcmpl-1" not in warnings[0].getMessage()
    assert "data:" not in warnings[0].getMessage()
    assert _caller_view(received) == _caller_view(sent_by_the_server)
    assert payloads == []


def _metered_values(payload):
    return {field: payload[field] for field in METERED_FIELDS}


@pytest.mark.parametrize("call", STREAMED_RESPONSES + [async_responses_create], ids=lambda c: c.__name__)
def test_responses_event_matches_the_non_streamed_call(call, payloads):
    sync_responses_create(FakeOpenAI(), None)
    call(FakeOpenAI(), None)

    non_streamed, other = payloads
    assert _metered_values(other) == _metered_values(non_streamed)
    assert other["model"] == MODEL
    assert other["cache_read_token_count"] == 3
    assert other["reasoning_token_count"] == 2
    assert other["is_streamed"] is (call is not async_responses_create)


@pytest.mark.parametrize("terminal", ["response.incomplete", "response.failed"])
@pytest.mark.parametrize("call", [sync_responses_create_stream, async_responses_create_stream],
                         ids=lambda c: c.__name__)
def test_responses_stream_meters_usage_from_non_completed_terminal_event(call, terminal, payloads):
    call(FakeOpenAI(responses_terminal=terminal), None)

    assert len(payloads) == 1
    assert payloads[0]["total_token_count"] == RESPONSES_USAGE["total_tokens"]
    assert payloads[0]["model"] == MODEL


# ---- stop reasons (BACK-4032) ----------------------------------------------

@pytest.mark.parametrize("call", STREAMED_CHAT, ids=lambda c: c.__name__)
def test_chat_stream_records_the_finish_reason(call, payloads):
    call(FakeOpenAI(chat_finish_reason="length"), None)

    assert len(payloads) == 1
    assert payloads[0]["stop_reason"] == "TOKEN_LIMIT"


RESPONSES_WITHOUT_THE_FINAL_RESPONSE_HELPER = [
    sync_responses_create, async_responses_create, sync_responses_create_stream, async_responses_create_stream,
    *RAW_STREAMED_RESPONSES,
]


@pytest.mark.parametrize("terminal, stop_reason", [
    ("response.incomplete", "TOKEN_LIMIT"),
    ("response.failed", "ERROR"),
])
@pytest.mark.parametrize("call", RESPONSES_WITHOUT_THE_FINAL_RESPONSE_HELPER, ids=lambda c: c.__name__)
def test_responses_record_the_incomplete_or_failed_status(call, terminal, stop_reason, payloads):
    call(FakeOpenAI(responses_terminal=terminal), None)

    assert len(payloads) == 1
    assert payloads[0]["stop_reason"] == stop_reason
    assert payloads[0]["total_token_count"] == RESPONSES_USAGE["total_tokens"]


def _responses_object(status, incomplete_details):
    usage = SimpleNamespace(input_tokens=11, output_tokens=7, total_tokens=18)
    return SimpleNamespace(id="resp_1", model=MODEL, usage=usage, status=status,
                           incomplete_details=incomplete_details)


@pytest.mark.parametrize("status, incomplete_details, stop_reason", [
    ("incomplete", SimpleNamespace(reason="max_output_tokens"), "TOKEN_LIMIT"),
    ("incomplete", SimpleNamespace(reason="content_filter"), "ERROR"),
    ("incomplete", SimpleNamespace(reason="something_new"), "END"),
    ("incomplete", {"reason": "max_output_tokens"}, "TOKEN_LIMIT"),
    ("incomplete", None, "END"),
    ("failed", None, "ERROR"),
    ("completed", None, "END"),
    (None, None, "END"),
], ids=["max_output_tokens", "content_filter", "unknown_reason", "dict_details", "no_details", "failed",
        "completed", "no_status"])
def test_responses_stop_reason_mapping(status, incomplete_details, stop_reason):
    usage_data, _ = mw.extract_usage_data(_responses_object(status, incomplete_details), mw.OperationType.CHAT,
                                          "2026-10-08T00:00:00Z", "2026-10-08T00:00:01Z", 1000)

    assert usage_data["stop_reason"] == stop_reason


@pytest.mark.parametrize("placeholder", [Omit(), NOT_GIVEN, None], ids=["omit", "not_given", "none"])
def test_placeholder_stream_options_count_as_not_supplied(placeholder):
    forced, caller_requested_usage = mw._with_forced_usage_reporting(
        {"stream": True, "stream_options": placeholder})

    assert forced["stream_options"] == {"include_usage": True}
    assert caller_requested_usage is False


def test_forcing_usage_does_not_mutate_the_caller_options():
    caller_options = {"include_obfuscation": False}

    forced, caller_requested_usage = mw._with_forced_usage_reporting({"stream_options": caller_options})

    assert forced["stream_options"] == {"include_obfuscation": False, "include_usage": True}
    assert caller_options == {"include_obfuscation": False}
    assert caller_requested_usage is False


def test_async_responses_create_is_wrapped():
    assert isinstance(AsyncResponses.__dict__["create"], wrapt.FunctionWrapper)
    assert isinstance(Responses.__dict__["create"], wrapt.FunctionWrapper)


# ---- selective metering pass-through (BACK-3991) ---------------------------

def _with_usage_metadata(request):
    return {**request, "usage_metadata": {"organizationName": "AcmeCorp", "subscriber": {"email": "dev@acme.example"}}}


def _sync_chat_create_with_metadata(fake):
    return fake.sync_client().chat.completions.create(**_with_usage_metadata(CHAT_REQUEST))


def _async_chat_create_with_metadata(fake):
    async def go():
        return await fake.async_client().chat.completions.create(**_with_usage_metadata(CHAT_REQUEST))
    return asyncio.run(go())


def _sync_responses_create_with_metadata(fake):
    return fake.sync_client().responses.create(**_with_usage_metadata(RESPONSES_REQUEST))


@pytest.mark.parametrize("call", [
    pytest.param(_sync_chat_create_with_metadata, id="sync_chat_create"),
    pytest.param(_async_chat_create_with_metadata, id="async_chat_create"),
    pytest.param(_sync_responses_create_with_metadata, id="sync_responses_create"),
])
def test_selective_pass_through_strips_usage_metadata_from_the_request(call, payloads):
    fake = FakeOpenAI()

    with patch.object(mw, "is_selective_metering_enabled", return_value=True):
        call(fake)

    assert len(fake.requests) == 1
    assert "usage_metadata" not in fake.requests[0]
    assert payloads == []


class _PerplexityBoundCompletions:
    _client = SimpleNamespace(base_url="https://api.perplexity.ai")


_PERPLEXITY_CHAT_WRAPS = (
    "perplexity:openai.resources.chat.completions.Completions.create",
    "perplexity:openai.resources.chat.completions.AsyncCompletions.create",
)


def _set_perplexity_wraps(monkeypatch, installed):
    from revenium_middleware._core import patch_registry
    patched = set(patch_registry._patched) - set(_PERPLEXITY_CHAT_WRAPS)
    if installed:
        patched |= set(_PERPLEXITY_CHAT_WRAPS)
    monkeypatch.setattr(patch_registry, "_patched", patched)


def _call_perplexity_bound_create():
    seen = []

    def wrapped(*_args, **kwargs):
        seen.append(kwargs)
        return "next wrapper's answer"

    kwargs = {**CHAT_REQUEST, "usage_metadata": {"trace_id": "t-1"}}
    assert mw.create_wrapper(wrapped, _PerplexityBoundCompletions(), (), kwargs) == "next wrapper's answer"
    return seen


def test_perplexity_bound_create_leaves_usage_metadata_for_the_perplexity_wrapper(monkeypatch):
    _set_perplexity_wraps(monkeypatch, installed=True)

    seen = _call_perplexity_bound_create()

    assert seen[0]["usage_metadata"] == {"trace_id": "t-1"}


def test_perplexity_bound_create_without_the_perplexity_wrapper_is_checked_and_takes_usage_metadata(monkeypatch):
    """BACK-4252: with no Perplexity wrapper to hand the call to, the OpenAI
    wrap checks the cap itself and drops the keyword the SDK would reject."""
    _set_perplexity_wraps(monkeypatch, installed=False)
    monkeypatch.setenv("REVENIUM_CIRCUIT_BREAKER_ENABLED", "true")
    checks = []
    monkeypatch.setattr(mw, "check_enforcement", lambda usage_metadata=None, **keywords: checks.append(keywords))

    seen = _call_perplexity_bound_create()

    assert "usage_metadata" not in seen[0]
    assert checks == [{"model": MODEL, "provider": "PERPLEXITY"}]


# ---- one-hour cache writes through an OpenAI-compatible gateway (BACK-4225) ----

GATEWAY_ONE_HOUR_CACHE_USAGE = {
    "prompt_tokens": 1200,
    "completion_tokens": 5,
    "total_tokens": 1205,
    "cache_creation_input_tokens": 1000,
    "cache_read_input_tokens": 0,
    "prompt_tokens_details": {"cache_creation_token_details": {
        "ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 1000}},
}
NON_STREAMED_CHAT = [sync_chat_create, async_chat_create]
SYNC_STREAMED_CHAT = [sync_chat_create_stream, sync_chat_stream, sync_chat_raw_stream,
                      sync_chat_streaming_response_stream]


@pytest.mark.parametrize("call", NON_STREAMED_CHAT + STREAMED_CHAT, ids=lambda c: c.__name__)
def test_one_hour_cache_write_split_is_sent_on_every_chat_entry_point(call, payloads):
    """A Claude call through a gateway prices its one-hour write at the one-hour rate however it is made."""
    call(FakeOpenAI(chat_usage=GATEWAY_ONE_HOUR_CACHE_USAGE), None)

    assert len(payloads) == 1
    split = {name: payloads[0].get(name) for name in ("cache_creation5m_token_count", "cache_creation1h_token_count")}
    assert payloads[0]["cache_creation_token_count"] == 1000
    assert split == {"cache_creation5m_token_count": 0, "cache_creation1h_token_count": 1000}


@pytest.mark.parametrize("call", SYNC_STREAMED_CHAT, ids=lambda c: c.__name__)
def test_openai_native_cache_write_sends_no_split_on_sync_stream(call, payloads):
    """OpenAI reports no TTL split, and a zero split would assert one that never happened."""
    usage = {**CHAT_USAGE, "prompt_tokens_details": {"cached_tokens": 3, "cache_write_tokens": 4}}
    call(FakeOpenAI(chat_usage=usage), None)

    assert len(payloads) == 1
    assert payloads[0]["cache_creation_token_count"] == 4
    assert "cache_creation5m_token_count" not in payloads[0]
    assert "cache_creation1h_token_count" not in payloads[0]


# ---- BACK-4239: a failed call sends one ERROR record --------------------------

REFUSED_CALLS = [
    pytest.param(sync_chat_create, False, id="chat"),
    pytest.param(sync_chat_create_stream, True, id="chat-stream"),
    pytest.param(async_chat_create, False, id="async-chat"),
    pytest.param(async_chat_create_stream, True, id="async-chat-stream"),
    pytest.param(sync_responses_create, False, id="responses"),
    pytest.param(sync_responses_create_stream, True, id="responses-stream"),
    pytest.param(async_responses_create, False, id="async-responses"),
    pytest.param(async_responses_create_stream, True, id="async-responses-stream"),
]


@pytest.fixture
def raised_errors(monkeypatch):
    """Every status error the OpenAI client builds, to check the caller gets that same object."""
    built = []
    for client_class in (OpenAI, AsyncOpenAI):
        build = client_class._make_status_error

        def spy(self, *args, _build=build, **kwargs):
            error = _build(self, *args, **kwargs)
            built.append(error)
            return error

        monkeypatch.setattr(client_class, "_make_status_error", spy)
    return built


def assert_one_error_record(payloads, *, streamed, error_class):
    assert len(payloads) == 1, payloads
    record = payloads[0]
    assert record["stop_reason"] == "ERROR"
    assert error_class in record["error_reason"]
    assert (record["input_token_count"], record["output_token_count"], record["total_token_count"]) == (0, 0, 0)
    assert record["model"] == MODEL
    assert record["is_streamed"] is streamed
    assert record["operation_type"] == "CHAT"
    return record


@pytest.mark.parametrize("status, error_class", [
    (400, "BadRequestError"), (429, "RateLimitError"), (500, "InternalServerError"),
])
@pytest.mark.parametrize("call, streamed", REFUSED_CALLS)
def test_refused_call_sends_one_error_record(payloads, raised_errors, call, streamed, status, error_class):
    import openai

    with pytest.raises(openai.APIStatusError) as exc_info:
        call(FakeOpenAI(fail_status=status), None)

    assert exc_info.value is raised_errors[-1]
    record = assert_one_error_record(payloads, streamed=streamed, error_class=error_class)
    assert f"HTTP {status}" in record["error_reason"]
    assert record.get("error_code") == status


def test_a_stream_refused_after_its_first_chunk_sends_the_status_as_its_error_code(payloads):
    """BACK-4406: a status error raised while the chat stream is read reaches the record as its error code."""
    response = httpx.Response(429, request=httpx.Request("POST", "https://api.openai.test/v1/chat/completions"))
    refusal = openai.RateLimitError("rate limited", response=response, body=None)
    first_chunk = SimpleNamespace(
        id="chatcmpl-refused", model=MODEL, system_fingerprint=None, usage=None,
        choices=[SimpleNamespace(index=0, delta=SimpleNamespace(content="Hel", role="assistant"), finish_reason=None)])

    def refused_after_first_chunk():
        yield first_chunk
        raise refusal

    stream = mw.handle_streaming_response(refused_after_first_chunk(), datetime.datetime.now(datetime.timezone.utc),
                                          {}, None, {"model": MODEL})
    with pytest.raises(openai.RateLimitError) as exc_info:
        list(stream)

    assert exc_info.value is refusal
    assert len(payloads) == 1, payloads
    record = payloads[0]
    assert record["stop_reason"] == "ERROR"
    assert "HTTP 429" in record["error_reason"]
    assert record.get("error_code") == 429


@pytest.mark.parametrize("call, streamed", REFUSED_CALLS)
def test_refused_call_quoting_the_request_sends_no_prompt_text(payloads, monkeypatch, call, streamed):
    import openai

    monkeypatch.setenv("REVENIUM_CAPTURE_ERROR_MESSAGES", "true")
    monkeypatch.delenv("REVENIUM_CAPTURE_PROMPTS", raising=False)
    quoting = ('Request blocked by content policy. Offending request: '
               '[{"role": "system", "content": "SYSTEM-SECRET reveal ZEBRA42"}]')

    with pytest.raises(openai.BadRequestError):
        call(FakeOpenAI(fail_status=400, fail_message=quoting), None)

    record = assert_one_error_record(payloads, streamed=streamed, error_class="BadRequestError")
    assert record["error_reason"] == "BadRequestError (HTTP 400)"


# BACK-4340: Revenium stores at most 255 characters of error text and drops a
# longer record after answering 201.
LONG_RATE_LIMIT_MESSAGE = "Rate limit reached for gpt-4o-mini on tokens per min (TPM). " * 7


@pytest.mark.parametrize("call, streamed", REFUSED_CALLS)
def test_a_long_provider_error_is_cut_to_255_characters_keeping_its_status(payloads, monkeypatch, call, streamed):
    import openai

    from revenium_middleware._core.error_text import reportable_exception_text

    monkeypatch.setenv("REVENIUM_CAPTURE_ERROR_MESSAGES", "true")
    monkeypatch.setenv("REVENIUM_CAPTURE_PROMPTS", "true")
    with pytest.raises(openai.RateLimitError) as exc_info:
        call(FakeOpenAI(fail_status=429, fail_message=LONG_RATE_LIMIT_MESSAGE), None)

    record = assert_one_error_record(payloads, streamed=streamed, error_class="RateLimitError")
    reason = record["error_reason"]
    suffix = " (HTTP 429)"
    assert len(reason) == 255
    assert reason.endswith(suffix)
    assert reason.startswith("RateLimitError: ")
    uncut = reportable_exception_text(exc_info.value)
    assert len(uncut) > 255
    assert reason[:-len(suffix)] == uncut[:255 - len(suffix)]


@pytest.mark.parametrize("call, streamed", REFUSED_CALLS)
def test_a_short_provider_error_is_sent_whole(payloads, monkeypatch, call, streamed):
    import openai

    from revenium_middleware._core.error_text import reportable_exception_text

    monkeypatch.setenv("REVENIUM_CAPTURE_ERROR_MESSAGES", "true")
    monkeypatch.setenv("REVENIUM_CAPTURE_PROMPTS", "true")
    with pytest.raises(openai.RateLimitError) as exc_info:
        call(FakeOpenAI(fail_status=429), None)

    record = assert_one_error_record(payloads, streamed=streamed, error_class="RateLimitError")
    assert record["error_reason"] == reportable_exception_text(exc_info.value) + " (HTTP 429)"


def test_two_failures_have_distinct_transaction_ids(payloads):
    import openai

    for _ in range(2):
        with pytest.raises(openai.RateLimitError):
            sync_chat_create(FakeOpenAI(fail_status=429), None)

    assert len(payloads) == 2
    assert payloads[0]["transaction_id"] != payloads[1]["transaction_id"]


@pytest.mark.parametrize("call", [sync_chat_create_stream, async_chat_create_stream])
def test_stream_failing_part_way_is_recorded_as_error(payloads, call):
    # 320 bytes arrive: the first chunk's event whole, the second cut off.
    with pytest.raises(Exception):
        call(FakeOpenAI(shape=in_chunks_of(64), fail_after=5), None)
    gc.collect()

    assert len(payloads) == 1, payloads
    assert payloads[0]["stop_reason"] == "ERROR"
    assert payloads[0]["transaction_id"] == "chatcmpl-1"


@pytest.mark.parametrize("call", [sync_chat_create_stream, async_chat_create_stream,
                                  sync_responses_create_stream, async_responses_create_stream])
def test_stream_failing_before_first_chunk_sends_one_error_record(payloads, call):
    with pytest.raises(Exception):
        call(FakeOpenAI(fail_after=0), None)
    gc.collect()

    assert_one_error_record(payloads, streamed=True, error_class="Error")


def test_error_record_does_not_consume_the_idempotency_block_key(mock_revenium_client):
    import openai
    from revenium_middleware import idempotency_key
    from revenium_middleware._core.metering_pool import wait_until_idle

    with idempotency_key("k"):
        with pytest.raises(openai.RateLimitError):
            sync_chat_create(FakeOpenAI(fail_status=429), None)
        assert wait_until_idle(10)
        sync_chat_create(FakeOpenAI(), None)
    assert wait_until_idle(10)

    sent = [call.kwargs for call in mock_revenium_client.ai.create_completion.call_args_list]
    assert sorted(record["stop_reason"] for record in sent) == ["END", "ERROR"]
    keys = {record["stop_reason"]: record["extra_headers"]["Idempotency-Key"] for record in sent}
    assert keys["END"] == "k"
    assert keys["ERROR"] not in ("k", "k:2")


OTHER_KEY = "hak_other_key"


@pytest.fixture
def other_key_client(monkeypatch, mock_revenium_client):
    """The client ``metering_key(OTHER_KEY)`` sends under; the process-wide key is ``mock_revenium_client``."""
    from collections import OrderedDict
    from revenium_middleware._core import metering

    client = MagicMock(name=OTHER_KEY)
    client.api_key = OTHER_KEY
    client.base_url = mock_revenium_client.base_url
    monkeypatch.setattr(metering, "_scoped_clients", OrderedDict())
    monkeypatch.setattr(metering, "_new_client", lambda api_key, base_url=None, http_client=None: client)
    return client


def _refused_sync_chat(fake):
    fake.sync_client().chat.completions.create(**CHAT_REQUEST)


def _refused_async_chat(fake):
    async def go():
        await fake.async_client().chat.completions.create(**CHAT_REQUEST)
    asyncio.run(go())


@pytest.mark.parametrize("refused_call, awaited_after_the_block", [
    pytest.param(_refused_sync_chat, False, id="chat"),
    pytest.param(_refused_async_chat, False, id="async-chat"),
    pytest.param(None, True, id="async-chat-awaited-after-the-block"),
])
def test_refused_call_inside_a_metering_key_block_sends_its_error_record_under_that_key(
        mock_revenium_client, other_key_client, refused_call, awaited_after_the_block):
    import openai
    import revenium_middleware
    from revenium_middleware._core.metering_pool import wait_until_idle

    fake = FakeOpenAI(fail_status=429)
    with pytest.raises(openai.RateLimitError):
        with revenium_middleware.metering_key(OTHER_KEY):
            if awaited_after_the_block:
                pending = fake.async_client().chat.completions.create(**CHAT_REQUEST)
            else:
                refused_call(fake)
        asyncio.run(pending)
    assert wait_until_idle(10)

    sent = [call.kwargs for call in other_key_client.ai.create_completion.call_args_list]
    assert [record["stop_reason"] for record in sent] == ["ERROR"]
    mock_revenium_client.ai.create_completion.assert_not_called()


# ---- usage_metadata on OpenAI's stream helpers (BACK-4337) --------------------

HELPER_TAGS = {"organizationName": "AcmeCorp", "productName": "prod-K", "agent": "agent-K"}


class _Answer(pydantic.BaseModel):
    answer: str


ANSWER_JSON = '{"answer": "hi"}'


def _tagged_sync_chat_stream(fake):
    with fake.sync_client().chat.completions.stream(**CHAT_REQUEST, usage_metadata=HELPER_TAGS) as stream:
        return [event.type for event in stream]


def _tagged_async_chat_stream(fake):
    async def go():
        async with fake.async_client().chat.completions.stream(**CHAT_REQUEST, usage_metadata=HELPER_TAGS) as stream:
            return [event.type async for event in stream]
    return asyncio.run(go())


def _tagged_sync_responses_stream(fake):
    with fake.sync_client().responses.stream(**RESPONSES_REQUEST, usage_metadata=HELPER_TAGS) as stream:
        return [event.type for event in stream]


def _tagged_async_responses_stream(fake):
    async def go():
        async with fake.async_client().responses.stream(**RESPONSES_REQUEST, usage_metadata=HELPER_TAGS) as stream:
            return [event.type async for event in stream]
    return asyncio.run(go())


def _tagged_sync_structured_chat_stream(fake):
    with fake.sync_client().chat.completions.stream(
            **CHAT_REQUEST, response_format=_Answer, usage_metadata=HELPER_TAGS) as stream:
        events = [event.type for event in stream]
        assert stream.get_final_completion().choices[0].message.parsed == _Answer(answer="hi")
        return events


@pytest.mark.parametrize("call", [
    _tagged_sync_chat_stream, _tagged_async_chat_stream, _tagged_sync_responses_stream,
    _tagged_async_responses_stream, _tagged_sync_structured_chat_stream,
], ids=lambda c: c.__name__.removeprefix("_tagged_"))
def test_stream_helper_accepts_usage_metadata_and_records_it(call, payloads):
    fake = FakeOpenAI(chat_content=ANSWER_JSON)

    call(fake)

    assert fake.requests and all("usage_metadata" not in request for request in fake.requests)
    assert len(payloads) == 1
    record = payloads[0]
    assert (record.get("organization_name"), record.get("product_name"), record.get("agent")) == (
        "AcmeCorp", "prod-K", "agent-K")


def _plain_create_inside_a_tagged_sync_helper(fake):
    client = fake.sync_client()
    with client.chat.completions.stream(**CHAT_REQUEST, usage_metadata=HELPER_TAGS) as stream:
        client.chat.completions.create(**CHAT_REQUEST)
        list(stream)


def _plain_create_inside_a_tagged_async_helper(fake):
    async def go():
        client = fake.async_client()
        async with client.chat.completions.stream(**CHAT_REQUEST, usage_metadata=HELPER_TAGS) as stream:
            await client.chat.completions.create(**CHAT_REQUEST)
            [event async for event in stream]
    asyncio.run(go())


def _untagged_helper_after_a_tagged_one(fake):
    client = fake.sync_client()
    with client.chat.completions.stream(**CHAT_REQUEST, usage_metadata=HELPER_TAGS) as stream:
        list(stream)
    with client.chat.completions.create(**CHAT_REQUEST, stream=True) as stream:
        list(stream)
    with client.chat.completions.stream(**CHAT_REQUEST) as stream:
        list(stream)


@pytest.mark.parametrize("call, untagged_records", [
    pytest.param(_plain_create_inside_a_tagged_sync_helper, 1, id="sync_create_inside_the_with"),
    pytest.param(_plain_create_inside_a_tagged_async_helper, 1, id="async_create_inside_the_async_with"),
    pytest.param(_untagged_helper_after_a_tagged_one, 2, id="untagged_calls_after_a_tagged_helper"),
])
def test_stream_helper_tag_does_not_reach_a_later_call(call, untagged_records, payloads):
    call(FakeOpenAI())

    organizations = [record.get("organization_name") for record in payloads]
    assert sorted(organizations, key=str) == sorted(["AcmeCorp"] + [None] * untagged_records, key=str)


# ---- a helper stream cut off before its usage chunk (BACK-4336) ---------------

CUT_OFF_USAGE = {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18}


def _sync_structured_chat_stream(fake, **extra):
    with fake.sync_client().chat.completions.stream(**CHAT_REQUEST, response_format=_Answer, **extra) as stream:
        for _ in stream:
            pass


def _async_structured_chat_stream(fake, **extra):
    async def go():
        async with fake.async_client().chat.completions.stream(
                **CHAT_REQUEST, response_format=_Answer, **extra) as stream:
            async for _ in stream:
                pass
    asyncio.run(go())


STRUCTURED_HELPERS = [_sync_structured_chat_stream, _async_structured_chat_stream]


def _assert_one_cut_off_record(payloads, stop_reason):
    assert len(payloads) == 1, payloads
    record = payloads[0]
    assert record["stop_reason"] == stop_reason
    assert (record["input_token_count"], record["output_token_count"]) == (11, 7)
    assert record["is_streamed"] is True
    return record


@pytest.mark.parametrize("finish_reason, raised, stop_reason", [
    ("length", openai.LengthFinishReasonError, "TOKEN_LIMIT"),
    ("content_filter", openai.ContentFilterFinishReasonError, "ERROR"),
    ("stop", ValueError, "END"),
], ids=["length", "content_filter", "stop_with_unparseable_content"])
@pytest.mark.parametrize("call", STRUCTURED_HELPERS, ids=lambda c: c.__name__.strip("_"))
def test_structured_chat_stream_cut_off_is_recorded_with_its_usage_when_the_with_block_exits(
        call, finish_reason, raised, stop_reason, payloads):
    fake = FakeOpenAI(chat_finish_reason=finish_reason, chat_usage=CUT_OFF_USAGE)

    with pytest.raises(raised) as caught:
        call(fake)

    _assert_one_cut_off_record(payloads, stop_reason)
    del caught
    gc.collect()
    assert len(payloads) == 1


class _FinishedStreamStub:
    """A chat stream wrapper that has seen its finish chunk and awaits usage."""

    def __init__(self, stream):
        self.finish_reason = "length"
        self.final_usage = None
        self.stream = stream
        self.observed = []

    def _observe(self, chunk):
        self.observed.append(chunk)
        if chunk == "usage":
            self.final_usage = CUT_OFF_USAGE


class _StalledAsyncStream:
    def __aiter__(self):
        return self

    async def __anext__(self):
        await asyncio.sleep(30)


class _PromptAsyncStream:
    def __init__(self, chunks):
        self._chunks = iter(chunks)

    async def __anext__(self):
        try:
            return next(self._chunks)
        except StopIteration:
            raise StopAsyncIteration


def _chunks_forever():
    while True:
        yield "text"


def test_reading_on_for_usage_after_a_finish_stops_at_the_time_limit_when_the_async_stream_stalls(monkeypatch):
    """BACK-4336: a stalled stream must not hold the caller's with block."""
    monkeypatch.setattr(mw, "_SECONDS_TO_WAIT_FOR_USAGE_AFTER_FINISH", 0.05)
    stub = _FinishedStreamStub(_StalledAsyncStream())

    started = time.monotonic()
    asyncio.run(mw._aread_usage_after_finish(stub, still_open=True))

    assert time.monotonic() - started < 2
    assert stub.final_usage is None


def test_reading_on_for_usage_after_a_finish_stops_at_the_time_limit_when_the_sync_stream_keeps_sending(
        monkeypatch):
    ticks = itertools.count()
    monkeypatch.setattr(mw, "time", SimpleNamespace(monotonic=lambda: float(next(ticks))))
    stub = _FinishedStreamStub(_chunks_forever())

    mw._read_usage_after_finish(stub, still_open=True)

    assert 0 < len(stub.observed) < mw._MAX_CHUNKS_READ_AFTER_FINISH
    assert stub.final_usage is None


def test_reading_on_for_usage_after_a_finish_still_picks_up_a_usage_chunk_that_arrives_promptly():
    sync_stub = _FinishedStreamStub(iter(["text", "usage"]))
    async_stub = _FinishedStreamStub(_PromptAsyncStream(["text", "usage"]))

    mw._read_usage_after_finish(sync_stub, still_open=True)
    asyncio.run(mw._aread_usage_after_finish(async_stub, still_open=True))

    assert sync_stub.final_usage == CUT_OFF_USAGE
    assert async_stub.final_usage == CUT_OFF_USAGE


class _CallerError(RuntimeError):
    pass


def _sync_helper_whose_body_raises(fake):
    with fake.sync_client().chat.completions.stream(**CHAT_REQUEST) as stream:
        for _ in stream:
            raise _CallerError("the caller's own error")


def _async_helper_whose_body_raises(fake):
    async def go():
        async with fake.async_client().chat.completions.stream(**CHAT_REQUEST) as stream:
            async for _ in stream:
                raise _CallerError("the caller's own error")
    asyncio.run(go())


@pytest.mark.parametrize("call", [_sync_helper_whose_body_raises, _async_helper_whose_body_raises],
                         ids=["sync", "async"])
def test_an_error_in_the_callers_with_body_is_not_replaced_and_the_call_is_recorded(call, payloads):
    fake = FakeOpenAI()

    with pytest.raises(_CallerError):
        call(fake)

    assert len(payloads) == 1, payloads
    assert len(fake.requests) == 1


def test_a_stream_the_caller_breaks_out_of_is_recorded_without_waiting_for_gc(payloads):
    stream = FakeOpenAI().sync_client().chat.completions.create(**CHAT_REQUEST, stream=True)
    gc.disable()
    try:
        for _ in stream:
            break
        wrapper = weakref.ref(stream)
        del stream

        assert len(payloads) == 1
        assert wrapper() is None
    finally:
        gc.enable()


@pytest.mark.parametrize("call", STRUCTURED_HELPERS, ids=lambda c: c.__name__.strip("_"))
def test_a_tagged_helper_stream_cut_off_records_its_tags_and_usage(call, payloads):
    fake = FakeOpenAI(chat_finish_reason="length", chat_usage=CUT_OFF_USAGE)

    with pytest.raises(openai.LengthFinishReasonError):
        call(fake, usage_metadata=HELPER_TAGS)

    record = _assert_one_cut_off_record(payloads, "TOKEN_LIMIT")
    assert (record.get("organization_name"), record.get("product_name"), record.get("agent")) == (
        "AcmeCorp", "prod-K", "agent-K")


# ---- a chunk whose content is a list of parts (BACK-4330) ---------------------

THINKING_PART = {"type": "thinking", "thinking": [{"type": "text", "text": "hm"}]}
LIST_CONTENTS = [
    pytest.param([THINKING_PART, {"type": "text", "text": "ok"}], id="thinking_then_text"),
    pytest.param([], id="empty_list"),
    pytest.param([THINKING_PART], id="thinking_only"),
]


@pytest.mark.parametrize("content", LIST_CONTENTS)
@pytest.mark.parametrize("call", [sync_chat_create_stream, async_chat_create_stream, *RAW_STREAMED_CHAT],
                         ids=lambda c: c.__name__)
def test_chat_stream_with_list_content_reads_like_the_bare_sdk_and_is_recorded(call, content, payloads, monkeypatch):
    monkeypatch.setenv("REVENIUM_CAPTURE_PROMPTS", "true")
    with patch.object(mw, "is_selective_metering_enabled", return_value=True):
        baseline = call(FakeOpenAI(first_content=content, chat_usage=CUT_OFF_USAGE), None)

    metered = call(FakeOpenAI(first_content=content, chat_usage=CUT_OFF_USAGE), None)

    assert _caller_view(metered) == _caller_view(baseline)
    assert len(payloads) == 1, payloads
    record = payloads[0]
    assert record["stop_reason"] == "END"
    assert (record["input_token_count"], record["output_token_count"]) == (11, 7)
    assert record.get("error_reason") is None
    captured = record.get("output_response") or ""
    assert "thinking" not in captured and "[" not in captured


def test_an_error_in_the_sdks_own_chunk_reading_is_not_recorded_as_a_provider_failure(payloads, monkeypatch):
    with patch.object(mw, "is_selective_metering_enabled", return_value=True):
        baseline = sync_chat_create_stream(FakeOpenAI(chat_usage=CUT_OFF_USAGE), None)

    def unreadable(_content):
        raise RuntimeError("the middleware could not read this chunk")

    monkeypatch.setattr(mw, "_answer_text_of", unreadable)
    received = sync_chat_create_stream(FakeOpenAI(chat_usage=CUT_OFF_USAGE), None)

    assert _caller_view(received) == _caller_view(baseline)
    assert len(payloads) == 1, payloads
    record = payloads[0]
    assert record["stop_reason"] == "END"
    assert (record["input_token_count"], record["output_token_count"]) == (11, 7)
    assert record.get("error_reason") is None
