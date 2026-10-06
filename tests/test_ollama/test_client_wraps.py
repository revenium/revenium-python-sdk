"""Every Ollama call shape is metered exactly once (BACK-3604).

Explicit ``Client``/``AsyncClient`` instances pointed at a self-hosted server,
the module-level functions bound to ollama's default client, and the legacy
``embeddings`` endpoint all go through the class-level wraps. Ollama traffic
is answered by ``httpx.MockTransport``; metering is recorded at
``submit_ai_event`` and dispatched synchronously.
"""
import asyncio
import json
import threading
from unittest.mock import MagicMock, patch

import httpx
import ollama
import pytest
import wrapt

from revenium_middleware._core.call_ownership import OLLAMA, transport_claim_mark
from revenium_middleware.ollama import middleware as mw

MODEL = "llama3"
SELF_HOSTED = "http://ollama.internal:8080"
INPUT_TOKENS = 11
OUTPUT_TOKENS = 7
COUNTS = {"prompt_eval_count": INPUT_TOKENS, "eval_count": OUTPUT_TOKENS}
DONE = {"model": MODEL, "created_at": "2026-01-01T00:00:00Z", "done": True, "done_reason": "stop", **COUNTS}


class StubOllama:
    def __init__(self):
        self.hosts = []

    def respond(self, request):
        self.hosts.append(request.url.host)
        body = json.loads(request.content or b"{}")
        path = request.url.path
        streamed = body.get("stream", True) is not False
        if path in ("/api/chat", "/api/generate"):
            key = "message" if path == "/api/chat" else "response"
            text, empty = ({"role": "assistant", "content": "hi"}, {"role": "assistant", "content": ""}) \
                if key == "message" else ("hi", "")
            if not streamed:
                return httpx.Response(200, json={**DONE, key: text})
            head = {"model": MODEL, "created_at": DONE["created_at"], "done": False, key: text}
            lines = [json.dumps(head), json.dumps({**DONE, key: empty})]
            return httpx.Response(200, content=("\n".join(lines) + "\n").encode())
        if path == "/api/embed":
            return httpx.Response(200, json={"model": MODEL, "embeddings": [[0.1, 0.2]],
                                             "prompt_eval_count": INPUT_TOKENS})
        if path == "/api/embeddings":
            return httpx.Response(200, json={"embedding": [0.1, 0.2]})
        return httpx.Response(404, json={"error": path})


def _run_now(coroutine):
    thread = threading.Thread(target=lambda: asyncio.run(coroutine))
    thread.start()
    thread.join(timeout=10)
    return thread


@pytest.fixture
def stub():
    return StubOllama()


@pytest.fixture
def records():
    captured = []

    def record(operation, payload, *args, **kwargs):
        captured.append(payload)
        return MagicMock(id="recorded")

    with patch.object(mw, "get_client", return_value=MagicMock()), \
            patch.object(mw, "submit_ai_event", side_effect=record), \
            patch.object(mw, "run_async_in_thread", side_effect=_run_now):
        yield captured


@pytest.fixture
def client(stub):
    return ollama.Client(host=SELF_HOSTED, transport=httpx.MockTransport(stub.respond))


@pytest.fixture
def async_client(stub):
    return ollama.AsyncClient(host=SELF_HOSTED, transport=httpx.MockTransport(stub.respond))


@pytest.fixture
def default_client(stub):
    default = ollama.chat.__self__
    original = default._client
    default._client = httpx.Client(transport=httpx.MockTransport(stub.respond), base_url="http://127.0.0.1:11434")
    try:
        yield default
    finally:
        default._client = original


def _consume(result):
    if hasattr(result, "__next__"):
        for _ in result:
            pass


async def _aconsume(awaitable):
    result = await awaitable
    if hasattr(result, "__anext__"):
        async for _ in result:
            pass


CALLS = {
    "chat": dict(model=MODEL, messages=[{"role": "user", "content": "hi"}]),
    "generate": dict(model=MODEL, prompt="hi"),
    "embed": dict(model=MODEL, input="hi"),
    "embeddings": dict(model=MODEL, prompt="hi"),
}
EXPECTED = {
    # endpoint: (operation_type, input tokens, output tokens)
    "chat": ("CHAT", INPUT_TOKENS, OUTPUT_TOKENS),
    "generate": ("GENERATE", INPUT_TOKENS, OUTPUT_TOKENS),
    "embed": ("EMBED", INPUT_TOKENS, 0),
    "embeddings": ("EMBED", 0, 0),
}
CASES = [(endpoint, False) for endpoint in CALLS] + [("chat", True), ("generate", True)]


def assert_one_payload(records, endpoint, streamed):
    assert len(records) == 1, records
    payload = records[0]
    operation_type, input_tokens, output_tokens = EXPECTED[endpoint]
    assert payload["provider"] == "OLLAMA"
    assert payload["model"] == MODEL
    assert payload["operation_type"] == operation_type
    assert payload["input_token_count"] == input_tokens
    assert payload["output_token_count"] == output_tokens
    assert payload["is_streamed"] is streamed
    assert payload["transaction_id"].startswith("ollama-")
    return payload


class TestExplicitSyncClient:
    @pytest.mark.parametrize("endpoint, streamed", CASES)
    def test_one_payload_against_a_self_hosted_server(self, client, stub, records, endpoint, streamed):
        kwargs = dict(CALLS[endpoint], **({"stream": True} if streamed else {}))
        _consume(getattr(client, endpoint)(**kwargs))
        assert stub.hosts == ["ollama.internal"]
        assert_one_payload(records, endpoint, streamed)

    def test_usage_metadata_is_taken_off_the_request_and_metered(self, client, records):
        client.chat(**CALLS["chat"], usage_metadata={"organizationName": "SelfHostedOrg"})
        assert records[0]["organization_name"] == "SelfHostedOrg"

    def test_a_positional_model_names_the_legacy_embeddings_record(self, client, records):
        client.embeddings(MODEL, "hi")
        assert records[0]["model"] == MODEL

    def test_a_stream_abandoned_after_the_first_chunk_is_metered_once(self, client, records):
        stream = client.chat(**CALLS["chat"], stream=True)
        next(stream)
        stream.close()
        assert len(records) == 1
        assert records[0]["is_streamed"] is True


class TestExplicitAsyncClient:
    @pytest.mark.parametrize("endpoint, streamed", CASES)
    def test_one_payload_against_a_self_hosted_server(self, async_client, stub, records, endpoint, streamed):
        kwargs = dict(CALLS[endpoint], **({"stream": True} if streamed else {}))
        asyncio.run(_aconsume(getattr(async_client, endpoint)(**kwargs)))
        assert stub.hosts == ["ollama.internal"]
        assert_one_payload(records, endpoint, streamed)

    def test_a_stream_abandoned_after_the_first_chunk_is_metered_once(self, async_client, records):
        async def call():
            stream = await async_client.generate(**CALLS["generate"], stream=True)
            await stream.__anext__()
            await stream.aclose()

        asyncio.run(call())
        assert len(records) == 1
        assert records[0]["is_streamed"] is True

    def test_a_stream_left_by_break_is_metered_once(self, async_client, records):
        async def call():
            async for _ in await async_client.chat(**CALLS["chat"], stream=True):
                break

        asyncio.run(call())
        assert len(records) == 1


class TestModuleLevelFunctions:
    @pytest.mark.parametrize("endpoint, streamed", CASES)
    def test_the_default_client_call_is_metered_once_not_twice(self, default_client, records, endpoint, streamed):
        kwargs = dict(CALLS[endpoint], **({"stream": True} if streamed else {}))
        _consume(getattr(ollama, endpoint)(**kwargs))
        assert_one_payload(records, endpoint, streamed)

    @pytest.mark.parametrize("endpoint", list(CALLS))
    def test_the_module_function_carries_exactly_one_revenium_wrap(self, endpoint):
        layers = []
        function = getattr(ollama, endpoint)
        while isinstance(function, (wrapt.FunctionWrapper, wrapt.BoundFunctionWrapper)):
            layers.append(function._self_wrapper.__module__)
            function = function.__wrapped__
        assert layers == ["revenium_middleware.ollama.middleware"]

    def test_rebinding_again_does_not_stack_a_second_wrap(self, default_client, records):
        mw.rebind_default_client_functions()
        ollama.chat(**CALLS["chat"])
        assert len(records) == 1


class TestTransportClaim:
    def test_a_metered_call_claims_the_ollama_transport(self, client, records):
        mark = transport_claim_mark()
        client.chat(**CALLS["chat"])
        assert transport_claim_mark().get(OLLAMA, 0) == mark.get(OLLAMA, 0) + 1

    def test_a_stream_is_claimed_when_it_is_handed_back(self, client, records):
        mark = transport_claim_mark()
        stream = client.chat(**CALLS["chat"], stream=True)
        assert transport_claim_mark().get(OLLAMA, 0) == mark.get(OLLAMA, 0) + 1
        _consume(stream)
        assert transport_claim_mark().get(OLLAMA, 0) == mark.get(OLLAMA, 0) + 1

    def test_an_async_call_claims_in_the_awaiting_task(self, async_client, records):
        async def call():
            mark = transport_claim_mark()
            await async_client.generate(**CALLS["generate"])
            return mark, transport_claim_mark()

        before, after = asyncio.run(call())
        assert after.get(OLLAMA, 0) == before.get(OLLAMA, 0) + 1

    def test_nothing_is_claimed_or_sent_when_metering_is_disabled(self, client, records):
        mark = transport_claim_mark()
        with patch.object(mw, "get_client", return_value=None):
            client.chat(**CALLS["chat"])
        assert records == []
        assert transport_claim_mark() == mark


class TestInterruptedStreamTokens:
    def test_a_stream_cut_before_the_final_chunk_meters_zero_tokens_without_raising(self, records):
        chunk = ollama.ChatResponse(model=MODEL, done=False, message={"role": "assistant", "content": "hi"})
        stream = mw.handle_streaming_response(iter([chunk, chunk]), mw.datetime.datetime.now(mw.datetime.timezone.utc),
                                              {}, "ollama-cut", "chat", {}, request_model=MODEL)
        next(stream)
        stream.close()
        assert len(records) == 1
        assert records[0]["input_token_count"] == 0
        assert records[0]["output_token_count"] == 0


class TestGriptapeOllamaDriver:
    def test_the_driver_is_metered_once_with_its_usage_metadata(self, client, stub, records):
        pytest.importorskip("griptape")
        from griptape.common import PromptStack
        from revenium_middleware.griptape import ReveniumOllamaDriver

        driver = ReveniumOllamaDriver(model=MODEL, client=client, usage_metadata={"organizationName": "GriptapeOrg"})
        prompt_stack = PromptStack()
        prompt_stack.add_user_message("hi")

        assert driver.run(prompt_stack).to_text() == "hi"
        assert stub.hosts == ["ollama.internal"]
        payload = assert_one_payload(records, "chat", streamed=False)
        assert payload["organization_name"] == "GriptapeOrg"


class Chunk:
    def __init__(self, done=False, **counts):
        self.done = done
        self.message = {"role": "assistant", "content": "tok"}
        self.model = MODEL
        self.done_reason = "stop" if done else None
        self.prompt_eval_count = counts.get("prompt_eval_count")
        self.eval_count = counts.get("eval_count")


NOW = mw.datetime.datetime.now(mw.datetime.timezone.utc)


def _chunks(n):
    return [Chunk() for _ in range(n - 1)] + [Chunk(done=True, **COUNTS)]


class ClosableStream:
    def __init__(self, chunks):
        self.chunks = chunks
        self.closed = False

    def sync(self):
        try:
            yield from self.chunks
        finally:
            self.closed = True

    async def asynchronous(self):
        try:
            for chunk in self.chunks:
                yield chunk
        finally:
            self.closed = True


class TestStreamClosing:
    """The caller may still hold Ollama's generator (here the test does), so
    closing ours must close it rather than leave it to garbage collection."""

    def test_closing_the_returned_stream_closes_ollamas_stream(self, records):
        inner = ClosableStream(_chunks(5))
        source = inner.sync()
        stream = mw.handle_streaming_response(source, NOW, {}, "ollama-close", "chat", {})
        next(stream)
        stream.close()
        assert inner.closed
        assert len(records) == 1

    def test_acclosing_the_returned_async_stream_closes_ollamas_stream(self, records):
        inner = ClosableStream(_chunks(5))

        async def call():
            source = inner.asynchronous()
            stream = mw.handle_async_streaming_response(source, NOW, {}, "ollama-aclose", "chat", {})
            await stream.__anext__()
            await stream.aclose()
            return inner.closed

        assert asyncio.run(call()) is True
        assert len(records) == 1

    def test_a_real_async_client_stream_releases_its_http_response_on_aclose(self, stub, records):
        responses = []

        class RecordingTransport(httpx.MockTransport):
            async def handle_async_request(self, request):
                response = await super().handle_async_request(request)
                responses.append(response)
                return response

        async def call():
            client = ollama.AsyncClient(host=SELF_HOSTED, transport=RecordingTransport(stub.respond))
            stream = await client.chat(**CALLS["chat"], stream=True)
            await stream.__anext__()
            await stream.aclose()
            return responses[0].is_closed

        assert asyncio.run(call()) is True
        assert len(records) == 1


class TestStreamRetention:
    def test_a_long_stream_keeps_only_the_chunk_it_meters(self, records):
        import gc
        import weakref

        chunks = _chunks(200)
        refs = [weakref.ref(chunk) for chunk in chunks]
        tally = mw.StreamTally()
        stream = mw.handle_streaming_response(iter(chunks), NOW, {}, "ollama-long", "chat", {}, tally=tally)
        del chunks
        for _ in stream:
            pass
        del _
        gc.collect()

        assert sum(ref() is not None for ref in refs) == 1
        assert tally.metered_chunk is refs[-1]()
        assert records[0]["input_token_count"] == INPUT_TOKENS

    def test_a_long_async_stream_keeps_only_the_chunk_it_meters(self, records):
        import gc
        import weakref

        chunks = _chunks(200)
        refs = [weakref.ref(chunk) for chunk in chunks]
        tally = mw.StreamTally()

        async def source(items):
            while items:
                yield items.pop(0)

        async def call():
            async for _ in mw.handle_async_streaming_response(source(chunks), NOW, {}, "ollama-along", "chat", {},
                                                              tally=tally):
                pass

        asyncio.run(call())
        gc.collect()

        assert sum(ref() is not None for ref in refs) == 1
        assert tally.metered_chunk is refs[-1]()
        assert records[0]["output_token_count"] == OUTPUT_TOKENS

    def test_a_chunk_after_the_counted_one_does_not_replace_it(self):
        tally = mw.StreamTally()
        counted = Chunk(done=True, **COUNTS)
        for chunk in (Chunk(), counted, Chunk()):
            tally.observe(chunk)
        assert tally.metered_chunk is counted

    def test_without_a_counted_chunk_the_latest_is_kept(self):
        tally = mw.StreamTally()
        latest = Chunk()
        for chunk in (Chunk(), Chunk(), latest):
            tally.observe(chunk)
        assert tally.metered_chunk is latest


class TestSelectiveMeteringSkip:
    @pytest.fixture
    def skipped(self):
        with patch.object(mw, "is_selective_metering_enabled", return_value=True), \
                patch.object(mw, "is_inside_decorated_function", return_value=False):
            yield

    def test_a_skipped_sync_call_does_not_forward_usage_metadata(self, skipped, client, stub, records):
        assert client.chat(**CALLS["chat"], usage_metadata={"organizationName": "Org"}).message.content == "hi"
        assert stub.hosts == ["ollama.internal"]
        assert records == []

    def test_a_skipped_async_call_does_not_forward_usage_metadata(self, skipped, async_client, records):
        response = asyncio.run(async_client.generate(**CALLS["generate"], usage_metadata={"organizationName": "Org"}))
        assert response.response == "hi"
        assert records == []

    def test_the_griptape_driver_works_outside_a_decorated_function(self, skipped, client, records):
        pytest.importorskip("griptape")
        from griptape.common import PromptStack
        from revenium_middleware.griptape import ReveniumOllamaDriver

        driver = ReveniumOllamaDriver(model=MODEL, client=client, usage_metadata={"organizationName": "Org"})
        prompt_stack = PromptStack()
        prompt_stack.add_user_message("hi")
        assert driver.run(prompt_stack).to_text() == "hi"
        assert records == []
