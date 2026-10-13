"""A local Perplexity upstream for driving real LiteLLM calls offline.

Answers ``/chat/completions`` the way Perplexity does: ``usage.cost`` carries
Perplexity's own charge, whose ``total_cost`` includes the request and search
fees. A streamed answer repeats a zero-cost skeleton on interim chunks and puts
the charge on the last one, as Perplexity's real stream does.

Not a test module (no test_ prefix) so pytest imports it rather than collecting it.
"""
import json
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

UPSTREAM_MODEL = "sonar"
COST = {"input_tokens_cost": 0.00124, "output_tokens_cost": 2e-05, "request_cost": 0.0025,
        "search_queries_cost": 0.005, "total_cost": 0.00876}
ZERO_COST_SKELETON = {"input_tokens_cost": 0.0, "output_tokens_cost": 0.0, "total_cost": 0.0}
TOKENS = {"prompt_tokens": 6, "completion_tokens": 3, "total_tokens": 9}
USAGE = {**TOKENS, "cost": COST}
INTERIM_USAGE = {**TOKENS, "cost": ZERO_COST_SKELETON}


def _chunk(completion_id, delta, usage, finish_reason=None):
    return {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": UPSTREAM_MODEL,
        "usage": usage,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }


def _stream_events(completion_id):
    chunks = [
        _chunk(completion_id, {"role": "assistant", "content": "o"}, INTERIM_USAGE),
        _chunk(completion_id, {"content": "k"}, INTERIM_USAGE),
        _chunk(completion_id, {}, USAGE, finish_reason="stop"),
    ]
    events = "".join("data: {}\n\n".format(json.dumps(chunk)) for chunk in chunks)
    return (events + "data: [DONE]\n\n").encode()


def _completion(completion_id):
    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": UPSTREAM_MODEL,
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}
        ],
        "usage": USAGE,
    }


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def do_POST(self):
        length = int(self.headers.get("content-length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        completion_id = "pplx-" + uuid.uuid4().hex
        if body.get("stream"):
            self._send("text/event-stream", _stream_events(completion_id))
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


def api_base(server, path=""):
    return "http://127.0.0.1:{}{}".format(server.server_address[1], path)
