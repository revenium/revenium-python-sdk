"""Perplexity raw-response and streaming-response forms record the real tokens
(BACK-4158), on the native client and on the OpenAI client pointed at
Perplexity, driven through the real SDKs over a stub transport.

The stub serves each body either loaded (the shape ``httpx.Response(json=...)``
builds) or chunked, one SSE event per chunk, the shape a real network body has:
only a chunked body shows whether the middleware leaves the body unread for
the caller.
"""
import asyncio
import inspect
import subprocess
import sys
import time
from unittest.mock import MagicMock, patch

import httpx
import openai
import perplexity
import pytest

import revenium_middleware.perplexity  # noqa: F401  installs the patches
from entry_points import calls

MESSAGES = [{"role": "user", "content": "hi"}]
TOKENS = (calls.INPUT_TOKENS, calls.OUTPUT_TOKENS, calls.INPUT_TOKENS + calls.OUTPUT_TOKENS)
STUB_RESPONSE_ID = "chatcmpl-1"

CLIENTS = ("native", "openai_compatible")
MODES = ("sync", "async")
FORMS = ("with_raw_response", "with_streaming_response")
STREAMS = (False, True)
DELIVERIES = ("loaded", "chunked")


class _ChunkedBody(httpx.SyncByteStream, httpx.AsyncByteStream):
    def __init__(self, chunks):
        self._chunks = chunks

    def __iter__(self):
        yield from self._chunks

    async def __aiter__(self):
        for chunk in self._chunks:
            yield chunk


def _sse_events(body):
    events = body.split(b"\n\n")
    return [event + b"\n\n" for event in events if event]


def _halves(body):
    middle = len(body) // 2
    return [body[:middle], body[middle:]]


class StubPerplexity:
    """Answers chat completions as Perplexity does, counting the requests it gets."""

    def __init__(self, delivery="loaded", answer=None):
        self.requests = 0
        self._delivery = delivery
        self._answer = answer or calls._perplexity_handler()

    def __call__(self, request):
        self.requests += 1
        loaded = self._answer(request)
        if self._delivery == "loaded":
            return loaded
        body = loaded.content
        is_sse = loaded.headers.get("content-type", "").startswith("text/event-stream")
        return httpx.Response(loaded.status_code, headers={"content-type": loaded.headers["content-type"]},
                              stream=_ChunkedBody(_sse_events(body) if is_sse else _halves(body)))

    def client(self, kind, mode):
        transport = httpx.MockTransport(self)
        if kind == "native":
            if mode == "sync":
                return perplexity.Perplexity(api_key="stub", max_retries=0, http_client=httpx.Client(transport=transport))
            return perplexity.AsyncPerplexity(api_key="stub", max_retries=0,
                                              http_client=httpx.AsyncClient(transport=transport))
        if mode == "sync":
            return openai.OpenAI(api_key="stub", base_url=calls.PERPLEXITY_BASE_URL, max_retries=0,
                                 http_client=httpx.Client(transport=transport))
        return openai.AsyncOpenAI(api_key="stub", base_url=calls.PERPLEXITY_BASE_URL, max_retries=0,
                                  http_client=httpx.AsyncClient(transport=transport))


def _request(stream):
    request = {"model": calls.PERPLEXITY_MODEL, "messages": MESSAGES}
    if stream:
        request["stream"] = True
    return request


@pytest.fixture
def recorder():
    recording_client = MagicMock()
    with patch("revenium_middleware._core.metering.client", recording_client):
        yield recording_client


def _sent(recording_client):
    calls.wait_for_metering()
    return [recorded.kwargs for recorded in recording_client.ai.create_completion.call_args_list]


def _tokens(record):
    return record["input_token_count"], record["output_token_count"], record["total_token_count"]


async def _resolved(value):
    return await value if inspect.isawaitable(value) else value


def _sdk_leaves_the_body_unread(form, stream):
    """A non-streamed ``with_raw_response`` call is the one form the SDK reads before returning."""
    return stream or form == "with_streaming_response"


class Cell:
    def __init__(self, kind, mode, form, stream, delivery):
        self.kind, self.mode, self.form, self.stream, self.delivery = kind, mode, form, stream, delivery

    @property
    def body_left_unread(self):
        return self.delivery == "chunked" and _sdk_leaves_the_body_unread(self.form, self.stream)

    def run(self, on_response=lambda raw: None, stub=None):
        """Make the call and read it as a caller does; return what the caller read."""
        stub = stub or StubPerplexity(self.delivery)
        create = getattr(stub.client(self.kind, self.mode).chat.completions, self.form).create
        if self.mode == "sync":
            return self._run_sync(create, on_response)
        return asyncio.run(self._run_async(create, on_response))

    def _run_sync(self, create, on_response):
        if self.form == "with_raw_response":
            raw = create(**_request(self.stream))
            on_response(raw)
            return self._read_sync(raw)
        with create(**_request(self.stream)) as raw:
            on_response(raw)
            return self._read_sync(raw)

    def _read_sync(self, raw):
        parsed = raw.parse()
        return [chunk.model_dump() for chunk in parsed] if self.stream else parsed.model_dump()

    async def _run_async(self, create, on_response):
        if self.form == "with_raw_response":
            raw = await create(**_request(self.stream))
            on_response(raw)
            return await self._read_async(raw)
        async with create(**_request(self.stream)) as raw:
            on_response(raw)
            return await self._read_async(raw)

    async def _read_async(self, raw):
        parsed = await _resolved(raw.parse())
        if self.stream:
            return [chunk.model_dump() async for chunk in parsed]
        return parsed.model_dump()


CELLS = [
    pytest.param(Cell(kind, mode, form, stream, delivery),
                 id=f"{kind}-{mode}-{form}-{'stream' if stream else 'whole'}-{delivery}")
    for kind in CLIENTS for mode in MODES for form in FORMS for stream in STREAMS for delivery in DELIVERIES
]


@pytest.mark.parametrize("cell", CELLS)
def test_a_raw_response_call_records_the_real_tokens(cell, recorder):
    seen_before_reading = []

    def before_the_caller_reads(raw):
        if cell.body_left_unread:
            assert raw.http_response.is_stream_consumed is False
            seen_before_reading.extend(_sent(recorder))

    cell.run(before_the_caller_reads)

    assert seen_before_reading == []
    sent = _sent(recorder)
    assert len(sent) == 1, sent
    assert sent[0]["provider"] == "PERPLEXITY"
    assert _tokens(sent[0]) == TOKENS
    assert sent[0]["is_streamed"] is cell.stream


@pytest.mark.parametrize("cell", CELLS)
def test_the_caller_still_reads_the_raw_response(cell, recorder, monkeypatch):
    metered = cell.run()

    monkeypatch.setenv("REVENIUM_SELECTIVE_METERING", "true")
    unmetered = cell.run()

    assert metered == unmetered
    if cell.stream:
        assert len(metered) == 3


@pytest.mark.parametrize("kind", CLIENTS)
def test_a_streaming_response_never_entered_sends_no_request_and_no_record(kind, recorder):
    stub = StubPerplexity()

    stub.client(kind, "sync").chat.completions.with_streaming_response.create(**_request(stream=False))

    assert stub.requests == 0
    assert _sent(recorder) == []


PROVIDER_WAIT_SECONDS = 0.3


def _answer_after_a_provider_wait(request):
    time.sleep(PROVIDER_WAIT_SECONDS)
    return calls._perplexity_handler()(request)


@pytest.mark.parametrize("kind", CLIENTS)
def test_a_streaming_response_duration_includes_the_provider_wait(kind, recorder):
    Cell(kind, "sync", "with_streaming_response", False, "loaded").run(
        stub=StubPerplexity(answer=_answer_after_a_provider_wait))

    sent = _sent(recorder)
    assert len(sent) == 1, sent
    assert sent[0]["request_duration"] >= PROVIDER_WAIT_SECONDS * 1000


def _read_one_chunk_then_close(kind, mode):
    stub = StubPerplexity("chunked")
    create = stub.client(kind, mode).chat.completions.with_raw_response.create

    if mode == "sync":
        stream = create(**_request(stream=True)).parse()
        next(iter(stream))
        stream.close()
        return

    async def consume():
        stream = await _resolved((await create(**_request(stream=True))).parse())
        await stream.__aiter__().__anext__()
        await stream.close()

    asyncio.run(consume())


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("kind", CLIENTS)
def test_a_raw_stream_closed_before_its_usage_chunk_is_metered_once(kind, mode, recorder):
    _read_one_chunk_then_close(kind, mode)

    sent = _sent(recorder)
    assert len(sent) == 1, sent
    assert _tokens(sent[0]) == (0, 0, 0)
    assert sent[0]["is_streamed"] is True


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("kind", CLIENTS)
def test_a_raw_response_record_uses_the_perplexity_response_id(kind, mode, recorder):
    Cell(kind, mode, "with_raw_response", False, "loaded").run()

    sent = _sent(recorder)
    assert len(sent) == 1, sent
    assert sent[0]["transaction_id"] == STUB_RESPONSE_ID


def _failing_answer(request):
    return httpx.Response(500, json={"error": {"message": "stub failure"}})


def _raised_by(call):
    with pytest.raises(Exception) as raised:
        call()
    return type(raised.value)


@pytest.mark.parametrize("kind", CLIENTS)
def test_an_error_on_enter_propagates_and_sends_no_record(kind, recorder, monkeypatch):
    def enter_a_failing_streaming_response():
        stub = StubPerplexity(answer=_failing_answer)
        with stub.client(kind, "sync").chat.completions.with_streaming_response.create(**_request(stream=False)):
            pass

    metered = _raised_by(enter_a_failing_streaming_response)
    assert _sent(recorder) == []

    monkeypatch.setenv("REVENIUM_SELECTIVE_METERING", "true")
    assert metered is _raised_by(enter_a_failing_streaming_response)


EXPECTED_FAILED_STOP_REASONS = {"native": ["ERROR"], "openai_compatible": []}


@pytest.mark.parametrize("kind", CLIENTS)
def test_a_failed_raw_response_call_behaves_as_a_failed_create(kind, recorder):
    completions = StubPerplexity(answer=_failing_answer).client(kind, "sync").chat.completions

    plain = _raised_by(lambda: completions.create(**_request(stream=False)))
    plain_records = [_failure_fields(record) for record in _sent(recorder)]
    assert [record[0] for record in plain_records] == EXPECTED_FAILED_STOP_REASONS[kind]
    raw = _raised_by(lambda: completions.with_raw_response.create(**_request(stream=False)))

    assert raw is plain
    assert [_failure_fields(record) for record in _sent(recorder)] == plain_records * 2


def _failure_fields(record):
    return record.get("stop_reason"), record.get("error_reason"), _tokens(record)


def _not_json_answer(request):
    return httpx.Response(200, content=b"<html>gateway hiccup</html>", headers={"content-type": "application/json"})


@pytest.mark.parametrize("delivery", DELIVERIES)
@pytest.mark.parametrize("form", FORMS)
@pytest.mark.parametrize("kind", CLIENTS)
def test_a_non_json_body_never_raises_into_the_caller(kind, form, delivery, recorder, caplog):
    stub = StubPerplexity(delivery, answer=_not_json_answer)
    create = getattr(stub.client(kind, "sync").chat.completions, form).create

    if form == "with_raw_response":
        body = create(**_request(stream=False)).http_response.read()
    else:
        with create(**_request(stream=False)) as raw:
            body = raw.http_response.read()

    assert body == b"<html>gateway hiccup</html>"
    sent = _sent(recorder)
    assert len(sent) == 1, sent
    assert _tokens(sent[0]) == (0, 0, 0)
    assert "No usage data found in response" in caplog.text


def test_importing_the_perplexity_middleware_does_not_patch_openai():
    probe = ("import sys, revenium_middleware.perplexity; "
             "print('revenium_middleware.openai.middleware' in sys.modules)")
    completed = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, timeout=120)

    assert completed.returncode == 0, completed.stderr[-2000:]
    assert completed.stdout.strip().splitlines()[-1] == "False"


def _chat_openai(stub, **kwargs):
    langchain_openai = pytest.importorskip("langchain_openai")
    transport = httpx.MockTransport(stub)
    return langchain_openai.ChatOpenAI(
        model=calls.PERPLEXITY_MODEL, api_key="stub", base_url=calls.PERPLEXITY_BASE_URL, max_retries=0,
        http_client=httpx.Client(transport=transport), http_async_client=httpx.AsyncClient(transport=transport),
        **kwargs)


LANGCHAIN_CALLS = {
    "invoke": lambda model: model.invoke("hi"),
    "ainvoke": lambda model: asyncio.run(model.ainvoke("hi")),
    "stream_with_headers": lambda model: list(model.stream("hi")),
}


@pytest.mark.parametrize("name", sorted(LANGCHAIN_CALLS))
def test_langchain_chat_openai_at_perplexity_records_the_real_tokens(name, recorder):
    kwargs = {"include_response_headers": True} if name == "stream_with_headers" else {}

    LANGCHAIN_CALLS[name](_chat_openai(StubPerplexity(), **kwargs))

    sent = _sent(recorder)
    assert len(sent) == 1, sent
    assert sent[0]["provider"] == "PERPLEXITY"
    assert _tokens(sent[0])[:2] == TOKENS[:2]


def test_the_revenium_callback_stands_down_for_a_metered_perplexity_call(recorder):
    pytest.importorskip("langchain_openai")
    from revenium_middleware.openai.langchain import ReveniumCallbackHandler

    _chat_openai(StubPerplexity(), callbacks=[ReveniumCallbackHandler()]).invoke("hi")

    sent = _sent(recorder)
    assert len(sent) == 1, sent
    assert _tokens(sent[0])[:2] == TOKENS[:2]
