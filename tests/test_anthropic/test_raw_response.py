"""Raw-response and streaming-response call forms, driven through the real
anthropic package over a stub transport (BACK-3583).

Each cell asserts the call returns what the SDK returns without our middleware,
reaches the stub exactly once, and sends exactly one usage record.
"""
import asyncio
import gc
import json
import threading
from unittest.mock import MagicMock, patch

import pydantic
import pytest

anthropic = pytest.importorskip("anthropic")
httpx2 = pytest.importorskip("httpx2")

import revenium_middleware.anthropic  # noqa: E402,F401  installs the patches
from revenium_middleware.anthropic.raw_response import is_raw_response  # noqa: E402

DATED_MODEL = "claude-sonnet-4-5-20250929"
REQUEST_ID = "req_stub"
MESSAGE_ID = "msg_stub"
INPUT_TOKENS = 13
OUTPUT_TOKENS = 5
STUB_HOST = "anthropic.stub.invalid"
REQUEST = dict(model="claude-sonnet-4-5", max_tokens=16, messages=[{"role": "user", "content": "hi"}])

_USAGE = {
    "input_tokens": INPUT_TOKENS,
    "output_tokens": OUTPUT_TOKENS,
    "cache_creation_input_tokens": 0,
    "cache_read_input_tokens": 0,
}
END_TURN = {"stop_reason": "end_turn", "stop_sequence": None}


def _message_body(finish):
    return json.dumps({
        "id": MESSAGE_ID, "type": "message", "role": "assistant", "model": DATED_MODEL,
        "content": [{"type": "text", "text": "hi"}], **finish, "usage": _USAGE,
    }).encode()


def _sse_body(finish):
    events = [
        ("message_start", {"type": "message_start", "message": {
            "id": MESSAGE_ID, "type": "message", "role": "assistant", "model": DATED_MODEL, "content": [],
            "stop_reason": None, "stop_sequence": None, "usage": {**_USAGE, "output_tokens": 1}}}),
        ("content_block_start", {"type": "content_block_start", "index": 0,
                                 "content_block": {"type": "text", "text": ""}}),
        ("content_block_delta", {"type": "content_block_delta", "index": 0,
                                 "delta": {"type": "text_delta", "text": "hi"}}),
        ("content_block_stop", {"type": "content_block_stop", "index": 0}),
        ("message_delta", {"type": "message_delta", "delta": finish, "usage": _USAGE}),
        ("message_stop", {"type": "message_stop"}),
    ]
    return "".join(f"event: {name}\ndata: {json.dumps(data)}\n\n" for name, data in events).encode()


MESSAGE_BODY = _message_body(END_TURN)
SSE_BODY = _sse_body(END_TURN)


CHUNK_SIZE = 64


def _chunks(body):
    return [body[i:i + CHUNK_SIZE] for i in range(0, len(body), CHUNK_SIZE)]


async def _async_chunks(body):
    for chunk in _chunks(body):
        yield chunk


class StubAnthropic:
    """Serves bodies as chunk iterators: an httpx2 Response built from bytes is
    read on construction, which would hide whether our wrapper consumed it."""

    def __init__(self, finish=END_TURN, status=None):
        self.requests = []
        self._finish = finish
        self._status = status

    def _respond(self, request, as_body):
        assert request.url.host == STUB_HOST
        self.requests.append(request)
        if self._status:
            return httpx2.Response(self._status, headers={"content-type": "application/json", "request-id": REQUEST_ID},
                                   content=as_body(json.dumps({"type": "error", "error": {
                                       "type": "invalid_request_error", "message": "refused"}}).encode()))
        if json.loads(request.content).get("stream"):
            return httpx2.Response(200, headers={"content-type": "text/event-stream", "request-id": REQUEST_ID},
                                   content=as_body(_sse_body(self._finish)))
        return httpx2.Response(200, headers={"content-type": "application/json", "request-id": REQUEST_ID},
                               content=as_body(_message_body(self._finish)))

    def sync_client(self):
        transport = httpx2.MockTransport(lambda request: self._respond(request, lambda body: iter(_chunks(body))))
        return anthropic.Anthropic(api_key="sk-ant-test", base_url=f"http://{STUB_HOST}", max_retries=0,
                                   http_client=httpx2.Client(transport=transport))

    def async_client(self):
        transport = httpx2.MockTransport(lambda request: self._respond(request, _async_chunks))
        return anthropic.AsyncAnthropic(api_key="sk-ant-test", base_url=f"http://{STUB_HOST}", max_retries=0,
                                        http_client=httpx2.AsyncClient(transport=transport))


def _run_metering_synchronously(coro_func, *args, **kwargs):
    thread = threading.Thread(target=lambda: asyncio.run(coro_func(*args, **kwargs)))
    thread.start()
    thread.join(timeout=10)
    return MagicMock()


@pytest.fixture
def stub():
    return StubAnthropic()


@pytest.fixture
def records():
    captured = []

    def record(operation, payload, *args, **kwargs):
        captured.append(payload)
        return MagicMock(status_code=201)

    with patch("revenium_middleware.anthropic.middleware._get_thread_safe_client", return_value=MagicMock()), \
            patch("revenium_middleware.anthropic.middleware.submit_ai_event", side_effect=record), \
            patch("revenium_middleware.anthropic.middleware._safe_run_async_in_thread",
                  side_effect=_run_metering_synchronously):
        yield captured


def assert_one_record(stub, records, *, streamed):
    assert len(stub.requests) == 1
    assert len(records) == 1
    record = records[0]
    assert record["input_token_count"] == INPUT_TOKENS
    assert record["output_token_count"] == OUTPUT_TOKENS
    assert record["provider"] == "ANTHROPIC"
    assert record["model"] == DATED_MODEL
    assert record["transaction_id"] == MESSAGE_ID
    assert record["is_streamed"] is streamed


def assert_sdk_raw_response(response, response_type):
    assert type(response) is response_type
    assert response.request_id == REQUEST_ID
    assert response.status_code == 200


def assert_message(message):
    assert isinstance(message, anthropic.types.Message)
    assert message.id == MESSAGE_ID
    assert message.usage.input_tokens == INPUT_TOKENS


class TestPlainCreateControl:
    def test_create_sync(self, stub, records):
        assert_message(stub.sync_client().messages.create(**REQUEST))
        assert_one_record(stub, records, streamed=False)

    def test_create_async(self, stub, records):
        async def call():
            return await stub.async_client().messages.create(**REQUEST)

        assert_message(asyncio.run(call()))
        assert_one_record(stub, records, streamed=False)

    def test_create_stream_sync(self, stub, records):
        events = list(stub.sync_client().messages.create(stream=True, **REQUEST))
        assert events[-1].type == "message_stop"
        assert_one_record(stub, records, streamed=True)

    def test_create_stream_async(self, stub, records):
        async def call():
            stream = await stub.async_client().messages.create(stream=True, **REQUEST)
            return [event async for event in stream]

        assert asyncio.run(call())[-1].type == "message_stop"
        assert_one_record(stub, records, streamed=True)

    def test_messages_stream_sync(self, stub, records):
        with stub.sync_client().messages.stream(**REQUEST) as stream:
            assert stream.get_final_message().id == MESSAGE_ID
        assert_one_record(stub, records, streamed=True)

    def test_messages_stream_async(self, stub, records):
        async def call():
            async with stub.async_client().messages.stream(**REQUEST) as stream:
                return await stream.get_final_message()

        assert asyncio.run(call()).id == MESSAGE_ID
        assert_one_record(stub, records, streamed=True)


class TestRawResponse:
    def test_raw_create_sync(self, stub, records):
        response = stub.sync_client().messages.with_raw_response.create(**REQUEST)

        assert_sdk_raw_response(response, anthropic.APIResponse)
        message = response.parse()
        assert_message(message)
        assert response.parse() is message
        assert_one_record(stub, records, streamed=False)

    def test_raw_create_async(self, stub, records):
        async def call():
            response = await stub.async_client().messages.with_raw_response.create(**REQUEST)
            return response, await response.parse(), await response.parse()

        response, message, parsed_again = asyncio.run(call())
        assert_sdk_raw_response(response, anthropic.AsyncAPIResponse)
        assert_message(message)
        assert parsed_again is message
        assert_one_record(stub, records, streamed=False)

    def test_raw_create_stream_sync(self, stub, records):
        response = stub.sync_client().messages.with_raw_response.create(stream=True, **REQUEST)

        assert_sdk_raw_response(response, anthropic.APIResponse)
        stream = response.parse()
        assert type(stream) is anthropic.Stream
        assert [event.type for event in stream][-1] == "message_stop"
        assert_one_record(stub, records, streamed=True)

    def test_raw_create_stream_async(self, stub, records):
        async def call():
            response = await stub.async_client().messages.with_raw_response.create(stream=True, **REQUEST)
            stream = await response.parse()
            return response, stream, [event.type async for event in stream]

        response, stream, event_types = asyncio.run(call())
        assert_sdk_raw_response(response, anthropic.AsyncAPIResponse)
        assert type(stream) is anthropic.AsyncStream
        assert event_types[-1] == "message_stop"
        assert_one_record(stub, records, streamed=True)


class TestStreamingResponse:
    def test_streaming_create_sync_leaves_body_for_caller(self, stub, records):
        with stub.sync_client().messages.with_streaming_response.create(**REQUEST) as response:
            assert_sdk_raw_response(response, anthropic.APIResponse)
            assert response.http_response.is_stream_consumed is False
            assert records == []
            assert b"".join(response.iter_bytes()) == MESSAGE_BODY
        assert_one_record(stub, records, streamed=False)

    def test_streaming_create_async_leaves_body_for_caller(self, stub, records):
        async def call():
            async with stub.async_client().messages.with_streaming_response.create(**REQUEST) as response:
                consumed_before_read = response.http_response.is_stream_consumed
                metered_before_read = list(records)
                body = b"".join([chunk async for chunk in response.iter_bytes()])
                return response, consumed_before_read, metered_before_read, body

        response, consumed_before_read, metered_before_read, body = asyncio.run(call())
        assert_sdk_raw_response(response, anthropic.AsyncAPIResponse)
        assert consumed_before_read is False
        assert metered_before_read == []
        assert body == MESSAGE_BODY
        assert_one_record(stub, records, streamed=False)

    def test_streaming_create_sync_parsed(self, stub, records):
        with stub.sync_client().messages.with_streaming_response.create(**REQUEST) as response:
            message = response.parse()
            assert_message(message)
            assert response.parse() is message
        assert_one_record(stub, records, streamed=False)

    def test_streaming_create_async_parsed(self, stub, records):
        async def call():
            async with stub.async_client().messages.with_streaming_response.create(**REQUEST) as response:
                return await response.parse(), await response.parse()

        message, parsed_again = asyncio.run(call())
        assert_message(message)
        assert parsed_again is message
        assert_one_record(stub, records, streamed=False)

    def test_streaming_create_sync_never_read_sends_no_record(self, stub, records):
        with stub.sync_client().messages.with_streaming_response.create(**REQUEST) as response:
            assert response.status_code == 200
        assert len(stub.requests) == 1
        assert records == []

    def test_streaming_create_async_never_read_sends_no_record(self, stub, records):
        async def call():
            async with stub.async_client().messages.with_streaming_response.create(**REQUEST) as response:
                assert response.status_code == 200

        asyncio.run(call())
        assert len(stub.requests) == 1
        assert records == []

    def test_streaming_create_stream_sync_bytes(self, stub, records):
        with stub.sync_client().messages.with_streaming_response.create(stream=True, **REQUEST) as response:
            assert_sdk_raw_response(response, anthropic.APIResponse)
            assert b"".join(response.iter_bytes()) == SSE_BODY
        assert_one_record(stub, records, streamed=True)

    def test_streaming_create_stream_async_bytes(self, stub, records):
        async def call():
            async with stub.async_client().messages.with_streaming_response.create(
                    stream=True, **REQUEST) as response:
                return response, b"".join([chunk async for chunk in response.iter_bytes()])

        response, body = asyncio.run(call())
        assert_sdk_raw_response(response, anthropic.AsyncAPIResponse)
        assert body == SSE_BODY
        assert_one_record(stub, records, streamed=True)

    def test_streaming_create_stream_sync_parsed(self, stub, records):
        with stub.sync_client().messages.with_streaming_response.create(stream=True, **REQUEST) as response:
            assert [event.type for event in response.parse()][-1] == "message_stop"
        assert_one_record(stub, records, streamed=True)

    def test_streaming_create_stream_async_parsed(self, stub, records):
        async def call():
            async with stub.async_client().messages.with_streaming_response.create(
                    stream=True, **REQUEST) as response:
                return [event.type async for event in await response.parse()]

        assert asyncio.run(call())[-1] == "message_stop"
        assert_one_record(stub, records, streamed=True)


class TestStopSequenceFinish:
    """A Claude answer that stopped on one of the caller's stop sequences
    (BACK-4032)."""

    @pytest.fixture
    def stub(self):
        return StubAnthropic(finish={"stop_reason": "stop_sequence", "stop_sequence": "###"})

    def test_create(self, stub, records):
        stub.sync_client().messages.create(stop_sequences=["###"], **REQUEST)

        assert_one_record(stub, records, streamed=False)
        assert records[0]["stop_reason"] == "END_SEQUENCE"

    def test_create_stream(self, stub, records):
        list(stub.sync_client().messages.create(stream=True, stop_sequences=["###"], **REQUEST))

        assert_one_record(stub, records, streamed=True)
        assert records[0]["stop_reason"] == "END_SEQUENCE"

    def test_messages_stream(self, stub, records):
        with stub.sync_client().messages.stream(stop_sequences=["###"], **REQUEST) as stream:
            stream.get_final_message()

        assert_one_record(stub, records, streamed=True)
        assert records[0]["stop_reason"] == "END_SEQUENCE"


class TestRawResponseDetection:
    def test_sdk_class_in_another_anthropic_module_is_recognised(self):
        stand_in_class = type("FutureAPIResponse", (), {
            "__module__": "anthropic._future_response",
            "http_response": None,
            "parse": lambda self: None,
        })
        assert is_raw_response(stand_in_class())

    def test_test_doubles_and_parsed_results_are_not_raw_responses(self, stub, records):
        message = stub.sync_client().messages.create(**REQUEST)
        assert not is_raw_response(MagicMock())
        assert not is_raw_response(message)


class TestChatAnthropicWithMiddlewareAlone:
    @pytest.fixture
    def chat(self, stub):
        langchain_anthropic = pytest.importorskip("langchain_anthropic")
        model = langchain_anthropic.ChatAnthropic(model=REQUEST["model"], max_tokens=16, max_retries=0,
                                                  api_key="sk-ant-test", base_url=f"http://{STUB_HOST}")
        model.__dict__["_client"] = stub.sync_client()
        model.__dict__["_async_client"] = stub.async_client()
        return model

    def test_invoke(self, chat, stub, records):
        assert chat.invoke("hi").content == "hi"
        assert_one_record(stub, records, streamed=False)

    def test_ainvoke(self, chat, stub, records):
        assert asyncio.run(chat.ainvoke("hi")).content == "hi"
        assert_one_record(stub, records, streamed=False)

    def test_stream(self, chat, stub, records):
        assert "".join(chunk.text for chunk in chat.stream("hi")) == "hi"
        assert_one_record(stub, records, streamed=True)

    def test_astream(self, chat, stub, records):
        async def call():
            return "".join([chunk.text async for chunk in chat.astream("hi")])

        assert asyncio.run(call()) == "hi"
        assert_one_record(stub, records, streamed=True)


# ---- BACK-4239: a refused call sends one ERROR record --------------------------

def _refused_create(client):
    return client.messages.create(**REQUEST)


def _refused_create_stream(client):
    return list(client.messages.create(**REQUEST, stream=True))


def _refused_messages_stream(client):
    with client.messages.stream(**REQUEST) as stream:
        return list(stream)


def _refused_async_create(stub):
    return asyncio.run(stub.async_client().messages.create(**REQUEST))


class _Answer(pydantic.BaseModel):
    answer: str


REFUSED_ANTHROPIC_CALLS = [
    pytest.param(lambda stub: _refused_create(stub.sync_client()), False, id="create"),
    pytest.param(lambda stub: stub.sync_client().messages.parse(**REQUEST, output_format=_Answer), False, id="parse"),
    pytest.param(lambda stub: _refused_create_stream(stub.sync_client()), True, id="create-stream"),
    pytest.param(_refused_async_create, False, id="async-create"),
    pytest.param(lambda stub: _refused_messages_stream(stub.sync_client()), True, id="messages-stream"),
]


@pytest.fixture
def raised_errors(monkeypatch):
    """Every status error the Anthropic client builds, to check the caller gets that same object."""
    built = []
    for client_class in (anthropic.Anthropic, anthropic.AsyncAnthropic):
        build = client_class._make_status_error

        def spy(self, *args, _build=build, **kwargs):
            error = _build(self, *args, **kwargs)
            built.append(error)
            return error

        monkeypatch.setattr(client_class, "_make_status_error", spy)
    return built


@pytest.mark.parametrize("status, error_class", [
    (400, "BadRequestError"), (429, "RateLimitError"), (500, "InternalServerError"),
])
@pytest.mark.parametrize("call, streamed", REFUSED_ANTHROPIC_CALLS)
def test_refused_create_sends_one_error_record(records, raised_errors, call, streamed, status, error_class):
    stub = StubAnthropic(status=status)

    with pytest.raises(anthropic.APIStatusError) as exc_info:
        call(stub)

    assert exc_info.value is raised_errors[-1]
    assert len(records) == 1, records
    record = records[0]
    assert record["stop_reason"] == "ERROR"
    assert error_class in record["error_reason"]
    assert f"HTTP {status}" in record["error_reason"]
    assert record.get("error_code") == status
    assert (record["input_token_count"], record["output_token_count"], record["total_token_count"]) == (0, 0, 0)
    assert record["model"] == REQUEST["model"]
    assert record["provider"] == "ANTHROPIC"
    assert record["is_streamed"] is streamed


# ---- BACK-4239: a stream that breaks off part-way is recorded as ERROR ----------

def _sse_events(body):
    return [event + b"\n\n" for event in body.split(b"\n\n") if event]


class CutOffStubAnthropic(StubAnthropic):
    """Serves a stream body that breaks off with ``error`` after ``events_before_cut`` events."""

    def __init__(self, events_before_cut):
        super().__init__()
        self.error = httpx2.ReadError("connection reset mid-stream")
        self._events_before_cut = events_before_cut

    def _cut_off(self, body):
        yield from _sse_events(body)[:self._events_before_cut]
        raise self.error

    async def _async_cut_off(self, body):
        for event in self._cut_off(body):
            yield event

    def sync_client(self):
        transport = httpx2.MockTransport(lambda request: self._respond(request, self._cut_off))
        return anthropic.Anthropic(api_key="sk-ant-test", base_url=f"http://{STUB_HOST}", max_retries=0,
                                   http_client=httpx2.Client(transport=transport))

    def async_client(self):
        transport = httpx2.MockTransport(lambda request: self._respond(request, self._async_cut_off))
        return anthropic.AsyncAnthropic(api_key="sk-ant-test", base_url=f"http://{STUB_HOST}", max_retries=0,
                                        http_client=httpx2.AsyncClient(transport=transport))


def _cut_messages_stream(stub):
    with stub.sync_client().messages.stream(**REQUEST) as stream:
        for _ in stream:
            pass


def _cut_async_messages_stream(stub):
    async def call():
        async with stub.async_client().messages.stream(**REQUEST) as stream:
            async for _ in stream:
                pass

    asyncio.run(call())


def _cut_raw_create_stream(stub):
    response = stub.sync_client().messages.with_raw_response.create(stream=True, **REQUEST)
    for _ in response.parse():
        pass


def _cut_async_raw_create_stream(stub):
    async def call():
        response = await stub.async_client().messages.with_raw_response.create(stream=True, **REQUEST)
        async for _ in await response.parse():
            pass

    asyncio.run(call())


CUT_OFF_STREAMS = [
    pytest.param(_cut_messages_stream, id="messages-stream"),
    pytest.param(_cut_async_messages_stream, id="async-messages-stream"),
    pytest.param(_cut_raw_create_stream, id="raw-create-stream"),
    pytest.param(_cut_async_raw_create_stream, id="async-raw-create-stream"),
]


@pytest.fixture
def keyed_records():
    captured = []

    def record(operation, payload, idempotency_key=None):
        captured.append((payload, idempotency_key))
        return MagicMock(status_code=201)

    with patch("revenium_middleware.anthropic.middleware._get_thread_safe_client", return_value=MagicMock()), \
            patch("revenium_middleware.anthropic.middleware.submit_ai_event", side_effect=record), \
            patch("revenium_middleware.anthropic.middleware._safe_run_async_in_thread",
                  side_effect=_run_metering_synchronously):
        yield captured


def _broken_stream_record(keyed_records, monkeypatch, call, events_before_cut):
    from revenium_middleware import idempotency_key as idempotency_block

    monkeypatch.setenv("REVENIUM_CAPTURE_PROMPTS", "true")
    stub = CutOffStubAnthropic(events_before_cut)
    with idempotency_block("blk"):
        with pytest.raises(httpx2.ReadError) as exc_info:
            call(stub)

    assert exc_info.value is stub.error
    assert len(keyed_records) == 1, keyed_records
    record, key = keyed_records[0]
    assert record["stop_reason"] == "ERROR"
    assert "ReadError" in record["error_reason"]
    assert record.get("error_code") is None
    assert record["is_streamed"] is True
    assert key.startswith("blk:err:")
    assert (record["system_prompt"], record["input_messages"], record["output_response"]) == (None, None, None)
    return record


@pytest.mark.parametrize("call", CUT_OFF_STREAMS)
def test_stream_breaking_off_part_way_is_recorded_as_error(keyed_records, monkeypatch, call):
    record = _broken_stream_record(keyed_records, monkeypatch, call, events_before_cut=3)

    assert record["input_token_count"] == INPUT_TOKENS
    assert record["transaction_id"] == MESSAGE_ID
    assert record["model"] == DATED_MODEL


@pytest.mark.parametrize("call", CUT_OFF_STREAMS)
def test_stream_breaking_off_before_message_start_sends_one_error_record(keyed_records, monkeypatch, call):
    record = _broken_stream_record(keyed_records, monkeypatch, call, events_before_cut=0)

    assert (record["input_token_count"], record["output_token_count"]) == (0, 0)
    assert record["model"] == REQUEST["model"]


OVERLOADED_EVENT = ("event: error\ndata: " + json.dumps({
    "type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}}) + "\n\n").encode()


class ErrorEventStubAnthropic(StubAnthropic):
    """Answers 200 and serves a stream that ends in an ``error`` event after ``events_before_error`` events."""

    def __init__(self, events_before_error):
        super().__init__()
        self._events_before_error = events_before_error

    def _with_error_event(self, body):
        return _sse_events(body)[:self._events_before_error] + [OVERLOADED_EVENT]

    async def _async_with_error_event(self, body):
        for event in self._with_error_event(body):
            yield event

    def sync_client(self):
        transport = httpx2.MockTransport(lambda request: self._respond(request, self._with_error_event))
        return anthropic.Anthropic(api_key="sk-ant-test", base_url=f"http://{STUB_HOST}", max_retries=0,
                                   http_client=httpx2.Client(transport=transport))

    def async_client(self):
        transport = httpx2.MockTransport(lambda request: self._respond(request, self._async_with_error_event))
        return anthropic.AsyncAnthropic(api_key="sk-ant-test", base_url=f"http://{STUB_HOST}", max_retries=0,
                                        http_client=httpx2.AsyncClient(transport=transport))


@pytest.mark.parametrize("call", [
    pytest.param(_cut_messages_stream, id="messages-stream"),
    pytest.param(_cut_async_messages_stream, id="async-messages-stream"),
])
def test_an_error_event_inside_a_200_stream_sends_no_error_code(records, call):
    """BACK-4406: the provider answered 200, so the record has no HTTP status to carry as its error code."""
    with pytest.raises(anthropic.APIStatusError) as exc_info:
        call(ErrorEventStubAnthropic(events_before_error=3))

    assert exc_info.value.status_code == 200
    assert len(records) == 1, records
    record = records[0]
    assert record["stop_reason"] == "ERROR"
    assert "error_code" not in record


# ---- a delivered parse() answer that fails to parse (BACK-4331) ---------------

MAX_TOKENS = {"stop_reason": "max_tokens", "stop_sequence": None}


def _parse(stub):
    return stub.sync_client().messages.parse(**REQUEST, output_format=_Answer)


def _async_parse(stub):
    return asyncio.run(stub.async_client().messages.parse(**REQUEST, output_format=_Answer))


def _beta_parse(stub):
    return stub.sync_client().beta.messages.parse(**REQUEST, output_format=_Answer)


def _async_beta_parse(stub):
    return asyncio.run(stub.async_client().beta.messages.parse(**REQUEST, output_format=_Answer))


@pytest.mark.parametrize("finish, stop_reason", [
    pytest.param(MAX_TOKENS, "TOKEN_LIMIT", id="max_tokens"),
    pytest.param(END_TURN, "END", id="end_turn_with_unparseable_text"),
])
@pytest.mark.parametrize("call", [_parse, _async_parse, _beta_parse, _async_beta_parse],
                         ids=lambda c: c.__name__.strip("_"))
def test_parse_cut_off_at_the_token_limit_is_recorded_with_the_billed_tokens(call, finish, stop_reason, records):
    stub = StubAnthropic(finish=finish)

    with pytest.raises(pydantic.ValidationError):
        call(stub)

    assert len(stub.requests) == 1
    assert len(records) == 1, records
    record = records[0]
    assert record["stop_reason"] == stop_reason
    assert (record["input_token_count"], record["output_token_count"]) == (INPUT_TOKENS, OUTPUT_TOKENS)
    assert record["transaction_id"] == MESSAGE_ID
    assert record.get("error_reason") is None
    gc.collect()
    assert len(records) == 1
