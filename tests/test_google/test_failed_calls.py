"""A Gemini call the provider refuses sends one ERROR record (BACK-4335).

The real google-genai client, on the Gemini API and on Vertex AI, runs here
against a local HTTP stub that answers every request with 429
RESOURCE_EXHAUSTED.
"""
import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from revenium_middleware import idempotency_key
from revenium_middleware.google.google_ai import middleware as genai_mw

pytestmark = pytest.mark.skipif(genai_mw is None, reason="google-genai SDK not installed")

genai = pytest.importorskip("google.genai")
types = pytest.importorskip("google.genai.types")
errors = pytest.importorskip("google.genai.errors")

from entry_points import calls  # noqa: E402
from entry_points.caps import breached_rule, seed_rules  # noqa: E402
from revenium_middleware._core.exceptions import BudgetExceededError  # noqa: E402

MODEL = "gemini-2.5-flash"
BLOCK_KEY = "caller-block"
REFUSAL = {"error": {"code": 429, "message": "Resource has been exhausted", "status": "RESOURCE_EXHAUSTED"}}


class _Refuses(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    requests = []

    def log_message(self, *args):
        pass

    def do_POST(self):
        self.rfile.read(int(self.headers.get("content-length", 0)))
        _Refuses.requests.append(self.path)
        data = json.dumps(REFUSAL).encode()
        self.send_response(429)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


@pytest.fixture(scope="module")
def server_url():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Refuses)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()


def _http_options(server_url):
    return types.HttpOptions(base_url=server_url, retry_options=types.HttpRetryOptions(attempts=1))


@pytest.fixture(params=["gemini_api", "vertex_ai"])
def client(request, server_url):
    _Refuses.requests = []
    if request.param == "vertex_ai":
        return genai.Client(vertexai=True, api_key="stub", http_options=_http_options(server_url))
    return genai.Client(api_key="stub", http_options=_http_options(server_url))


def _sync(client):
    return client.models.generate_content(model=MODEL, contents="hi")


def _sync_stream(client):
    return list(client.models.generate_content_stream(model=MODEL, contents="hi"))


def _async(client):
    return asyncio.run(client.aio.models.generate_content(model=MODEL, contents="hi"))


def _async_stream(client):
    async def run():
        return [chunk async for chunk in await client.aio.models.generate_content_stream(model=MODEL, contents="hi")]
    return asyncio.run(run())


PATHS = [
    pytest.param(_sync, False, id="sync"),
    pytest.param(_async, False, id="async"),
    pytest.param(_sync_stream, True, id="sync_stream"),
    pytest.param(_async_stream, True, id="async_stream"),
]


def completion_calls(recorder):
    calls.wait_for_metering()
    return [c.kwargs for c in recorder.ai.create_completion.call_args_list]


@pytest.mark.parametrize("call, streamed", PATHS)
def test_a_refused_gemini_call_sends_one_error_record(mock_revenium_client, client, call, streamed):
    with idempotency_key(BLOCK_KEY), pytest.raises(errors.ClientError):
        call(client)
    assert _Refuses.requests
    records = completion_calls(mock_revenium_client)
    assert len(records) == 1, records
    record = records[0]
    assert record["stop_reason"] == "ERROR"
    assert "ClientError" in record["error_reason"]
    assert "(HTTP 429)" in record["error_reason"]
    assert record.get("error_code") == 429
    assert (record["input_token_count"], record["output_token_count"], record["total_token_count"]) == (0, 0, 0)
    assert record["model"] == MODEL
    assert record["provider"] == "Google"
    assert record["is_streamed"] is streamed
    assert record["extra_headers"]["Idempotency-Key"] != BLOCK_KEY


@pytest.mark.parametrize("call, streamed", PATHS)
def test_a_gemini_call_a_cap_refuses_sends_no_record(monkeypatch, mock_revenium_client, client, call, streamed):
    seed_rules(monkeypatch, breached_rule(4335))
    with pytest.raises(BudgetExceededError):
        call(client)
    assert _Refuses.requests == []
    assert completion_calls(mock_revenium_client) == []


def test_a_refused_gemini_call_sends_no_error_text_without_prompt_capture(monkeypatch, mock_revenium_client, client):
    """google-genai's message quotes the provider's answer, which can quote the request (BACK-4359)."""
    monkeypatch.setenv("REVENIUM_CAPTURE_ERROR_MESSAGES", "true")
    monkeypatch.delenv("REVENIUM_CAPTURE_PROMPTS", raising=False)
    with pytest.raises(errors.ClientError):
        _sync(client)
    records = completion_calls(mock_revenium_client)
    assert [record["error_reason"] for record in records] == ["ClientError (HTTP 429)"]


def test_a_refused_gemini_embedding_sends_no_record(mock_revenium_client, client):
    """Embeddings are the documented exception: a refused one is not recorded yet."""
    with pytest.raises(errors.ClientError):
        client.models.embed_content(model="text-embedding-004", contents="hi")
    calls.wait_for_metering()
    assert mock_revenium_client.ai.create_completion.call_args_list == []
