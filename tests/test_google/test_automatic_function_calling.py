"""Every model request of a Gemini automatic function calling loop is recorded (BACK-4242).

When a Python function is passed as a tool, google-genai's public
``generate_content`` (and ``generate_content_stream``, sync and async) asks
Gemini, runs the function and asks again, all inside one call. Each of those
requests is billed. The real google-genai client runs here against a local
HTTP stub that answers with a function call until the request carries enough
function results, then with text.
"""
import asyncio
import gc
import json
import logging
import threading
import time
import weakref
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
import wrapt

from revenium_middleware import idempotency_key, revenium_meter
from revenium_middleware.google.google_ai import middleware as genai_mw

pytestmark = pytest.mark.skipif(genai_mw is None, reason="google-genai SDK not installed")

genai = pytest.importorskip("google.genai")
types = pytest.importorskip("google.genai.types")

from entry_points import calls  # noqa: E402
from entry_points.caps import breached_rule, record_cap_checks, seed_rules  # noqa: E402
from revenium_middleware._core.exceptions import BudgetExceededError  # noqa: E402

MODEL = "gemini-2.5-flash"
ROUND_USAGE = [(100, 10), (115, 8), (130, 5)]
INNER_USAGE = (7, 3)
THOUGHTS_TOKENS = 6
CACHED_TOKENS = 40


def _usage(prompt, candidates, **extra):
    return {"promptTokenCount": prompt, "candidatesTokenCount": candidates,
            "totalTokenCount": prompt + candidates, **extra}


def _function_call(usage):
    return {"candidates": [{"content": {"role": "model", "parts": [
        {"functionCall": {"name": "get_weather", "args": {"city": "Paris"}}}]}, "finishReason": "STOP"}],
        "usageMetadata": usage, "modelVersion": MODEL}


def _text(usage, text="It is sunny in Paris."):
    return {"candidates": [{"content": {"role": "model", "parts": [{"text": text}]}, "finishReason": "STOP"}],
            "usageMetadata": usage, "modelVersion": MODEL}


def _function_results_sent(body):
    return sum(1 for content in body.get("contents", []) for part in content.get("parts", [])
               if "functionResponse" in part)


class GeminiStub:
    """Answers with a function call for the first ``tool_rounds`` rounds, then with text."""

    def __init__(self):
        self.reset()

    def reset(self, tool_rounds=1, fail_round=None, split_first_round=False):
        self.tool_rounds = tool_rounds
        self.fail_round = fail_round
        self.split_first_round = split_first_round
        self.requests = []

    def answer(self, path, body):
        """The status and the chunks to send for one request."""
        if not body.get("tools"):
            self.requests.append(("inner", path))
            return 200, [_text(_usage(*INNER_USAGE), "inner")]
        round_number = _function_results_sent(body) + 1
        self.requests.append((round_number, path))
        if round_number == self.fail_round:
            return 500, [{"error": {"code": 500, "message": "stub failure", "status": "INTERNAL"}}]
        usage = _usage(*ROUND_USAGE[round_number - 1 if round_number <= self.tool_rounds else -1])
        if round_number > self.tool_rounds:
            return 200, [_text(usage)]
        if self.split_first_round and round_number == 1:
            first = _function_call(_usage(100, 4))
            final = {"candidates": [{"content": {"role": "model", "parts": [{"text": ""}]}, "finishReason": "STOP"}],
                     "usageMetadata": {**_usage(100, 10, thoughtsTokenCount=THOUGHTS_TOKENS,
                                                cachedContentTokenCount=CACHED_TOKENS),
                                       "totalTokenCount": 100 + 10 + THOUGHTS_TOKENS},
                     "modelVersion": MODEL}
            return 200, [first, final]
        return 200, [_function_call(usage)]


STUB = GeminiStub()


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("content-length", 0))) or b"{}")
        status, chunks = STUB.answer(self.path, body)
        if status == 200 and "streamGenerateContent" in self.path:
            data = "".join(f"data: {json.dumps(chunk)}\r\n\r\n" for chunk in chunks).encode()
            content_type = "text/event-stream"
        else:
            data = json.dumps(chunks[0]).encode()
            content_type = "application/json"
        self.send_response(status)
        self.send_header("content-type", content_type)
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


@pytest.fixture(scope="module")
def server_url():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()


@pytest.fixture
def stub():
    STUB.reset()
    return STUB


_clients = []


@pytest.fixture
def client(server_url, stub):
    made = genai.Client(api_key="stub", http_options=types.HttpOptions(
        base_url=server_url, retry_options=types.HttpRetryOptions(attempts=1)))
    _clients.append(made)
    return made


def get_weather(city: str) -> str:
    """Look up the weather in a city."""
    return "sunny"


async def async_get_weather(city: str) -> str:
    """Look up the weather in a city."""
    return "sunny"


async_get_weather.__name__ = "get_weather"


def _config(tool=get_weather, **afc):
    config = {"tools": [tool]}
    if afc:
        config["automatic_function_calling"] = types.AutomaticFunctionCallingConfig(**afc)
    return types.GenerateContentConfig(**config)


def completions(recorder):
    calls.wait_for_metering()
    return [c.kwargs for c in recorder.ai.create_completion.call_args_list]


def completions_by_input_tokens(recorder):
    return sorted(completions(recorder), key=lambda record: record["input_token_count"])


def token_pairs(recorder):
    return [(record["input_token_count"], record["output_token_count"])
            for record in completions_by_input_tokens(recorder)]


def _sync(client, config, contents="weather in Paris?", **extra):
    return client.models.generate_content(model=MODEL, contents=contents, config=config, **extra)


def _sync_stream(client, config, contents="weather in Paris?", **extra):
    return list(client.models.generate_content_stream(model=MODEL, contents=contents, config=config, **extra))


def _async(client, config, contents="weather in Paris?", **extra):
    return asyncio.run(client.aio.models.generate_content(model=MODEL, contents=contents, config=config, **extra))


def _async_stream(client, config, contents="weather in Paris?", **extra):
    async def run():
        stream = await client.aio.models.generate_content_stream(
            model=MODEL, contents=contents, config=config, **extra)
        return [chunk async for chunk in stream]
    return asyncio.run(run())


PATHS = [
    pytest.param(_sync, False, False, id="sync"),
    pytest.param(_sync_stream, True, False, id="sync_stream"),
    pytest.param(_async, False, True, id="async"),
    pytest.param(_async_stream, True, True, id="async_stream"),
]


def _tool_for(is_async):
    return async_get_weather if is_async else get_weather


@pytest.mark.parametrize("call, streamed, is_async", PATHS)
def test_function_calling_records_every_round(mock_revenium_client, stub, client, call, streamed, is_async):
    call(client, _config(_tool_for(is_async)))
    assert [request[0] for request in stub.requests] == [1, 2]
    records = completions(mock_revenium_client)
    assert token_pairs(mock_revenium_client) == [(100, 10), (130, 5)]
    assert sum(r["input_token_count"] for r in records) == 230
    assert sum(r["output_token_count"] for r in records) == 15
    for record in records:
        assert record["is_streamed"] is streamed
        assert record["provider"] == "Google"


@pytest.mark.parametrize("call, streamed, is_async", PATHS)
def test_three_rounds_send_three_records(mock_revenium_client, stub, client, call, streamed, is_async):
    stub.reset(tool_rounds=2)
    call(client, _config(_tool_for(is_async)))
    assert token_pairs(mock_revenium_client) == ROUND_USAGE


@pytest.mark.parametrize("call, streamed, is_async", PATHS)
def test_one_remote_call_sends_one_record(mock_revenium_client, stub, client, call, streamed, is_async):
    call(client, _config(_tool_for(is_async), maximum_remote_calls=1))
    assert len(stub.requests) == 1
    assert token_pairs(mock_revenium_client) == [(100, 10)]


@pytest.mark.parametrize("call, streamed, is_async", PATHS)
def test_a_call_without_tools_sends_one_record(mock_revenium_client, stub, client, call, streamed, is_async):
    call(client, None)
    assert token_pairs(mock_revenium_client) == [INNER_USAGE]


@pytest.mark.parametrize("call, streamed, is_async", PATHS)
def test_function_calling_switched_off_sends_one_record(mock_revenium_client, stub, client, call, streamed, is_async):
    call(client, _config(_tool_for(is_async), disable=True))
    assert len(stub.requests) == 1
    assert token_pairs(mock_revenium_client) == [(100, 10)]


@pytest.mark.parametrize("call", [pytest.param(_sync_stream, id="sync_stream"),
                                  pytest.param(_async_stream, id="async_stream")])
def test_a_streamed_round_split_over_chunks_counts_its_final_usage(mock_revenium_client, stub, client, call):
    stub.reset(split_first_round=True)
    call(client, _config(async_get_weather if call is _async_stream else get_weather))
    records = completions_by_input_tokens(mock_revenium_client)
    assert len(records) == 2, records
    first, second = records
    assert first["total_token_count"] == 100 + 10 + THOUGHTS_TOKENS
    assert first["cache_read_token_count"] == CACHED_TOKENS
    assert (second["input_token_count"], second["output_token_count"]) == (130, 5)
    assert second["total_token_count"] == 135
    assert not second.get("cache_read_token_count")


def _chat_send(client):
    client.chats.create(model=MODEL, config=_config()).send_message("weather in Paris?")


def _chat_send_stream(client):
    list(client.chats.create(model=MODEL, config=_config()).send_message_stream("weather in Paris?"))


def _async_chat_send(client):
    asyncio.run(client.aio.chats.create(model=MODEL, config=_config(async_get_weather)).send_message("weather?"))


def _async_chat_send_stream(client):
    async def run():
        chat = client.aio.chats.create(model=MODEL, config=_config(async_get_weather))
        return [chunk async for chunk in await chat.send_message_stream("weather?")]
    asyncio.run(run())


@pytest.mark.parametrize("send", [
    pytest.param(_chat_send, id="send_message"),
    pytest.param(_chat_send_stream, id="send_message_stream"),
    pytest.param(_async_chat_send, id="async"),
    pytest.param(_async_chat_send_stream, id="async_stream"),
])
def test_chat_records_one_per_round_without_double_count(mock_revenium_client, stub, client, send):
    send(client)
    assert len(stub.requests) == 2
    assert token_pairs(mock_revenium_client) == [(100, 10), (130, 5)]


BLOCK_KEY = "caller-block"


def assert_the_round_before_and_one_error_record(recorder):
    finished, failed = sorted(completions(recorder), key=lambda record: -record["input_token_count"])
    assert (finished["input_token_count"], finished["output_token_count"]) == (100, 10)
    assert finished["stop_reason"] == "END"
    assert finished["trace_id"] == "t-4335"
    assert (failed["input_token_count"], failed["output_token_count"]) == (0, 0)
    assert failed["stop_reason"] == "ERROR"
    assert "HTTP" in failed["error_reason"]
    assert failed["trace_id"] == "t-4335"
    assert failed["extra_headers"]["Idempotency-Key"] not in (BLOCK_KEY, finished["extra_headers"]["Idempotency-Key"])


@pytest.mark.parametrize("call, streamed, is_async", PATHS)
def test_a_round_that_fails_still_records_the_round_before(
        mock_revenium_client, stub, client, call, streamed, is_async):
    """The finished round keeps its record and the refused one sends an ERROR record (BACK-4242, BACK-4335)."""
    stub.reset(fail_round=2)
    with idempotency_key(BLOCK_KEY), pytest.raises(genai.errors.ServerError):
        call(client, _config(_tool_for(is_async)), usage_metadata={"trace_id": "t-4335"})
    assert len(stub.requests) == 2
    assert_the_round_before_and_one_error_record(mock_revenium_client)


class _InsideTool:
    active = False


@pytest.fixture
def unmetered_inside_tools(monkeypatch):
    monkeypatch.setattr(genai_mw, "_metering_skipped", lambda: _InsideTool.active)


@pytest.mark.parametrize("is_async", [False, True], ids=["sync", "async"])
def test_an_unmetered_gemini_call_inside_a_tool_is_not_billed_to_the_outer_call(
        mock_revenium_client, stub, client, unmetered_inside_tools, is_async):
    def get_weather(city: str) -> str:
        """Look up the weather in a city."""
        _InsideTool.active = True
        try:
            client.models.generate_content(model=MODEL, contents="inner")
        finally:
            _InsideTool.active = False
        return "sunny"

    async def async_get_weather(city: str) -> str:
        """Look up the weather in a city."""
        _InsideTool.active = True
        try:
            await client.aio.models.generate_content(model=MODEL, contents="inner")
        finally:
            _InsideTool.active = False
        return "sunny"

    async_get_weather.__name__ = "get_weather"

    @revenium_meter()
    def outer():
        if is_async:
            return _async(client, _config(async_get_weather))
        return _sync(client, _config(get_weather))

    outer()
    assert ("inner", f"/v1beta/models/{MODEL}:generateContent") in stub.requests
    assert token_pairs(mock_revenium_client) == [(100, 10), (130, 5)]


@pytest.mark.parametrize("is_async", [False, True], ids=["sync", "async"])
def test_an_unmetered_gemini_stream_inside_a_tool_is_not_billed_to_the_outer_call(
        mock_revenium_client, stub, client, unmetered_inside_tools, is_async):
    def get_weather(city: str) -> str:
        """Look up the weather in a city."""
        _InsideTool.active = True
        try:
            list(client.models.generate_content_stream(model=MODEL, contents="inner"))
        finally:
            _InsideTool.active = False
        return "sunny"

    async def async_get_weather(city: str) -> str:
        """Look up the weather in a city."""
        _InsideTool.active = True
        try:
            stream = await client.aio.models.generate_content_stream(model=MODEL, contents="inner")
            return "sunny" if [chunk async for chunk in stream] else "empty"
        finally:
            _InsideTool.active = False

    async_get_weather.__name__ = "get_weather"

    if is_async:
        _async_stream(client, _config(async_get_weather))
    else:
        _sync_stream(client, _config(get_weather))
    assert ("inner", f"/v1beta/models/{MODEL}:streamGenerateContent?alt=sse") in stub.requests
    assert token_pairs(mock_revenium_client) == [(100, 10), (130, 5)]


@pytest.mark.parametrize("is_async", [False, True], ids=["sync", "async"])
def test_a_call_that_breaks_after_an_answered_round_sends_no_error_record(
        mock_revenium_client, stub, client, is_async):
    def not_the_requested_tool(city: str) -> str:
        """Look up the weather in a city."""
        return "sunny"

    with pytest.raises(KeyError):
        if is_async:
            _async(client, _config(not_the_requested_tool))
        else:
            _sync(client, _config(not_the_requested_tool))
    assert [request[0] for request in stub.requests] == [1]
    assert token_pairs(mock_revenium_client) == [(100, 10)]
    assert all(record["stop_reason"] != "ERROR" for record in completions(mock_revenium_client))


@pytest.mark.parametrize("call, streamed, is_async", PATHS)
def test_a_breached_cap_is_checked_once_before_the_first_round(
        monkeypatch, caplog, mock_revenium_client, stub, client, call, streamed, is_async):
    seed_rules(monkeypatch, breached_rule(4242))
    with pytest.raises(BudgetExceededError):
        call(client, _config(_tool_for(is_async)))
    assert stub.requests == []

    checks = record_cap_checks(monkeypatch)
    seed_rules(monkeypatch, breached_rule(4242, action="WARN_ONLY"))
    with caplog.at_level(logging.WARNING, logger="revenium_middleware.extension"):
        call(client, _config(_tool_for(is_async)))
    assert len(stub.requests) == 2
    assert len(checks) == 1, checks
    assert len([r for r in caplog.records if "is over its threshold" in r.getMessage()]) == 1


ROUND_TEXT = "It is sunny in Paris."


def output_responses(recorder):
    return [record.get("output_response") for record in completions_by_input_tokens(recorder)]


@pytest.fixture
def captured_prompts(monkeypatch):
    monkeypatch.setenv("REVENIUM_CAPTURE_PROMPTS", "true")


@pytest.mark.parametrize("send", [
    pytest.param(_sync_stream, id="sync_stream"),
    pytest.param(_async_stream, id="async_stream"),
    pytest.param(_chat_send_stream, id="send_message_stream"),
])
def test_each_streamed_round_captures_its_text_once(mock_revenium_client, stub, client, captured_prompts, send):
    """Every chunk is read once even though it passes both the round tap and the caller's pull (BACK-4333)."""
    if send is _chat_send_stream:
        send(client)
    else:
        send(client, _config(_tool_for(send is _async_stream)))
    assert token_pairs(mock_revenium_client) == [(100, 10), (130, 5)]
    assert output_responses(mock_revenium_client)[1] == ROUND_TEXT


def _old_shape_chunks(round_number, fail_round=None):
    if round_number == fail_round:
        raise genai.errors.ServerError(500, {"error": {"code": 500, "message": "stub failure", "status": "INTERNAL"}})
    usage = _usage(*ROUND_USAGE[0]) if round_number == 1 else _usage(*ROUND_USAGE[-1])
    chunk = _function_call(usage) if round_number == 1 else _text(usage, ROUND_TEXT)
    return [types.GenerateContentResponse.model_validate(chunk)]


def _max_remote_calls(config):
    afc = config.automatic_function_calling if config else None
    return afc.maximum_remote_calls if afc and afc.maximum_remote_calls else 10


def _old_shape_loop(model, contents, config, stream_round):
    """google-genai 1.65.0's streamed tool loop (2.17.0 is the same): the chunks
    of the round that asks for a tool are read by the loop and never yielded."""
    for round_number in range(1, _max_remote_calls(config) + 1):
        asked_for_tool = False
        for chunk in stream_round(model=model, contents=contents, config=config):
            asked_for_tool = bool(chunk.function_calls)
            if round_number > 1 or not asked_for_tool:
                yield chunk
        if not asked_for_tool:
            return


async def _async_old_shape_loop(model, contents, config, stream_round):
    for round_number in range(1, _max_remote_calls(config) + 1):
        asked_for_tool = False
        async for chunk in await stream_round(model=model, contents=contents, config=config):
            asked_for_tool = bool(chunk.function_calls)
            if round_number > 1 or not asked_for_tool:
                yield chunk
        if not asked_for_tool:
            return


async def _async_chunks(chunks):
    for chunk in chunks:
        yield chunk


class _OldShapeModels:
    def __init__(self, fail_round=None):
        self.requests = 0
        self.fail_round = fail_round

    def _generate_content_stream(self, *, model, contents, config):
        self.requests += 1
        return iter(_old_shape_chunks(self.requests, self.fail_round))

    def generate_content_stream(self, *, model, contents, config=None):
        return _old_shape_loop(model, contents, config, self._generate_content_stream)


class _AsyncOldShapeModels:
    def __init__(self, fail_round=None):
        self.requests = 0
        self.fail_round = fail_round

    async def _generate_content_stream(self, *, model, contents, config):
        self.requests += 1
        return _async_chunks(_old_shape_chunks(self.requests, self.fail_round))

    async def generate_content_stream(self, *, model, contents, config=None):
        return _async_old_shape_loop(model, contents, config, self._generate_content_stream)


for _models, _public, _private in [
        (_OldShapeModels, genai_mw.generate_content_stream_wrapper, genai_mw._stream_round_tap),
        (_AsyncOldShapeModels, genai_mw.async_generate_content_stream_wrapper, genai_mw._async_stream_round_tap)]:
    _models.generate_content_stream = wrapt.FunctionWrapper(_models.generate_content_stream, _public)
    _models._generate_content_stream = wrapt.FunctionWrapper(_models._generate_content_stream, _private)


def _old_shape_stream(config, fail_round=None, **extra):
    models = _OldShapeModels(fail_round)
    return list(models.generate_content_stream(model=MODEL, contents="weather in Paris?", config=config, **extra))


def _async_old_shape_stream(config, fail_round=None, **extra):
    async def run():
        stream = await _AsyncOldShapeModels(fail_round).generate_content_stream(
            model=MODEL, contents="weather in Paris?", config=config, **extra)
        return [chunk async for chunk in stream]
    return asyncio.run(run())


OLD_SHAPE_PATHS = [pytest.param(_old_shape_stream, id="sync"), pytest.param(_async_old_shape_stream, id="async")]


@pytest.mark.parametrize("call", OLD_SHAPE_PATHS)
def test_a_stream_whose_client_hides_the_tool_round_records_every_round(
        mock_revenium_client, captured_prompts, call):
    """google-genai below 2.18.0 never yields the tool round's chunks; that round is still recorded (BACK-4333)."""
    call(_config())
    assert token_pairs(mock_revenium_client) == [(100, 10), (130, 5)]
    assert all(record["is_streamed"] is True for record in completions(mock_revenium_client))
    assert output_responses(mock_revenium_client)[1] == ROUND_TEXT


@pytest.mark.parametrize("call", OLD_SHAPE_PATHS)
def test_a_stream_whose_client_hides_its_only_round_records_it(mock_revenium_client, call):
    """A loop held to one request yields nothing on google-genai below 2.18.0; its request is recorded (BACK-4333)."""
    assert call(_config(maximum_remote_calls=1)) == []
    assert token_pairs(mock_revenium_client) == [(100, 10)]
    assert all(record["is_streamed"] is True for record in completions(mock_revenium_client))


@pytest.mark.parametrize("call", OLD_SHAPE_PATHS)
def test_a_hidden_tool_round_then_a_refused_round_records_both(mock_revenium_client, call):
    """On google-genai below 2.18.0 the hidden tool round and the refused round each send a record
    (BACK-4333, BACK-4335)."""
    with idempotency_key(BLOCK_KEY), pytest.raises(genai.errors.ServerError):
        call(_config(), fail_round=2, usage_metadata={"trace_id": "t-4335"})
    assert_the_round_before_and_one_error_record(mock_revenium_client)


TOOL_SECONDS = 0.5


def slow_get_weather(city: str) -> str:
    """Look up the weather in a city."""
    time.sleep(TOOL_SECONDS)
    return "sunny"


async def async_slow_get_weather(city: str) -> str:
    """Look up the weather in a city."""
    await asyncio.sleep(TOOL_SECONDS)
    return "sunny"


slow_get_weather.__name__ = async_slow_get_weather.__name__ = "get_weather"


@pytest.mark.parametrize("call, tool", [pytest.param(_sync_stream, slow_get_weather, id="sync_stream"),
                                        pytest.param(_async_stream, async_slow_get_weather, id="async_stream")])
def test_a_streamed_round_ends_when_its_last_chunk_arrives_not_after_the_tool_runs(
        mock_revenium_client, stub, client, call, tool):
    call(client, _config(tool))
    first_round, final_round = completions(mock_revenium_client)
    assert first_round["request_duration"] < TOOL_SECONDS * 1000 / 2, first_round["request_duration"]
    assert final_round["request_duration"] < TOOL_SECONDS * 1000 / 2, final_round["request_duration"]


def _stream_config(tools, is_async):
    return _config(_tool_for(is_async)) if tools else None


def _pull(chunks, count):
    for pulled, _ in enumerate(chunks, start=1):
        if pulled == count:
            return


async def _async_pull(chunks, count):
    pulled = 0
    async for _ in chunks:
        pulled += 1
        if pulled == count:
            return


def _abandon_unbound(client, recorder, tools, count):
    for _ in client.models.generate_content_stream(model=MODEL, contents="weather in Paris?",
                                                   config=_stream_config(tools, False)):
        break
    return token_pairs(recorder), True


def _abandon(client, recorder, tools, count):
    stream = client.models.generate_content_stream(model=MODEL, contents="weather in Paris?",
                                                   config=_stream_config(tools, False))
    _pull(stream, count)
    dropped = weakref.ref(stream)
    del stream
    return token_pairs(recorder), dropped() is None


def _abandon_async(client, recorder, tools, count):
    async def run():
        stream = await client.aio.models.generate_content_stream(model=MODEL, contents="weather in Paris?",
                                                                 config=_stream_config(tools, True))
        await _async_pull(stream, count)
        dropped = weakref.ref(stream)
        del stream
        return token_pairs(recorder), dropped() is None
    return asyncio.run(run())


def _abandon_chat(client, recorder, tools, count):
    chat = client.chats.create(model=MODEL, config=_stream_config(tools, False))
    for _ in chat.send_message_stream("weather in Paris?"):
        break
    return token_pairs(recorder), True


def _usage_of_each_request(stub):
    return [INNER_USAGE if round_number == "inner"
            else ROUND_USAGE[round_number - 1 if round_number <= stub.tool_rounds else -1]
            for round_number, _ in stub.requests]


@pytest.mark.parametrize("abandon, tools, tool_rounds, count", [
    pytest.param(_abandon_unbound, False, 1, 1, id="sync_unbound"),
    pytest.param(_abandon, False, 1, 1, id="sync"),
    pytest.param(_abandon_async, False, 1, 1, id="async"),
    pytest.param(_abandon, True, 1, 1, id="sync_tool_round"),
    pytest.param(_abandon_async, True, 1, 1, id="async_tool_round"),
    pytest.param(_abandon, True, 2, 2, id="sync_two_tool_rounds"),
    pytest.param(_abandon_async, True, 2, 2, id="async_two_tool_rounds"),
    pytest.param(_abandon_chat, True, 1, 1, id="chat_send_message_stream"),
])
def test_an_abandoned_stream_is_recorded_when_dropped_with_the_collector_off(
        mock_revenium_client, stub, client, abandon, tools, tool_rounds, count):
    """A stream left early is recorded on its last reference, not by the cycle collector (BACK-4410).

    Every request the stream made is recorded once. How many that is depends on
    the google-genai version: below 2.18.0 the tool round's chunks are never
    yielded, so the first chunk pulled already comes from the answer round.
    Async records are read inside the coroutine: closing the event loop would
    otherwise free the stream and hide the bug.
    """
    stub.reset(tool_rounds=tool_rounds)
    gc.disable()
    try:
        pairs, freed = abandon(client, mock_revenium_client, tools, count)
    finally:
        gc.enable()
    assert len(stub.requests) >= count
    assert pairs == _usage_of_each_request(stub)
    assert freed
