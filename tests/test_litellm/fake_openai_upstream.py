"""A local OpenAI-compatible upstream for driving a real LiteLLM proxy offline.

Serves ``/chat/completions`` (plain and streamed, with the usage chunk when
``stream_options.include_usage`` asks for it) and ``/embeddings`` on a loopback
port, so a proxy test exercises LiteLLM's own HTTP client and response parsing
instead of ``mock_response``, which short-circuits the provider call.

Not a test module (no test_ prefix) so pytest imports it rather than collecting it.
"""
import json
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

UPSTREAM_MODEL = "gpt-4o-mini-2024-07-18"
USAGE = {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10}
EMBEDDING_USAGE = {"prompt_tokens": 2, "total_tokens": 2}


def _chunk(completion_id, delta, finish_reason=None):
    return {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": UPSTREAM_MODEL,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }


def _stream_events(completion_id, include_usage):
    chunks = [
        _chunk(completion_id, {"role": "assistant", "content": "hi"}),
        _chunk(completion_id, {}, finish_reason="stop"),
    ]
    if include_usage:
        usage_chunk = _chunk(completion_id, {})
        usage_chunk["choices"] = []
        usage_chunk["usage"] = USAGE
        chunks.append(usage_chunk)
    events = "".join("data: {}\n\n".format(json.dumps(chunk)) for chunk in chunks)
    return (events + "data: [DONE]\n\n").encode()


def _completion(completion_id):
    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": UPSTREAM_MODEL,
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}
        ],
        "usage": USAGE,
    }


def _embedding():
    return {
        "object": "list",
        "model": "text-embedding-3-small",
        "data": [{"object": "embedding", "index": 0, "embedding": [0.1, 0.2]}],
        "usage": EMBEDDING_USAGE,
    }


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def do_POST(self):
        length = int(self.headers.get("content-length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        if self.path.endswith("/embeddings"):
            self._send("application/json", json.dumps(_embedding()).encode())
            return
        completion_id = "chatcmpl-" + uuid.uuid4().hex
        if body.get("stream"):
            include_usage = bool((body.get("stream_options") or {}).get("include_usage"))
            self._send("text/event-stream", _stream_events(completion_id, include_usage))
            return
        self._send("application/json", json.dumps(_completion(completion_id)).encode())

    def _send(self, content_type, payload):
        self.send_response(200)
        self.send_header("content-type", content_type)
        self.send_header("content-length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def start_upstream():
    """Serve the fake upstream on a free loopback port; the caller shuts it down."""
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def api_base(server):
    return "http://127.0.0.1:{}/v1".format(server.server_address[1])
