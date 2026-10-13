"""A local Anthropic Messages upstream for driving real LiteLLM calls offline.

Serves ``/v1/messages`` plain and streamed on a loopback port, answering with
the usage block the server was started with, so a test exercises LiteLLM's own
Anthropic transformation (the one that folds cache tokens into
``prompt_tokens`` and nests the cache-write split) instead of ``mock_response``.
A streamed answer carries the ``cache_creation`` split on ``message_start``
only, never on ``message_delta``, which is what the live API sends.

Not a test module (no test_ prefix) so pytest imports it rather than collecting it.
"""
import gc
import json
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

UPSTREAM_MODEL = "claude-haiku-4-5-20251001"


def _message(message_id, usage):
    return {
        "id": message_id,
        "type": "message",
        "role": "assistant",
        "model": UPSTREAM_MODEL,
        "content": [{"type": "text", "text": "ok"}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": usage,
    }


def _delta_usage(usage):
    """The cumulative counts ``message_delta`` reports: the flat fields, no split."""
    return {name: value for name, value in usage.items() if name != "cache_creation"}


def _stream_events(message_id, usage, start_usage=None):
    start = _message(message_id, {**(start_usage or usage), "output_tokens": 1})
    start["content"] = []
    start["stop_reason"] = None
    events = [
        ("message_start", {"type": "message_start", "message": start}),
        ("content_block_start", {"type": "content_block_start", "index": 0,
                                 "content_block": {"type": "text", "text": ""}}),
        ("content_block_delta", {"type": "content_block_delta", "index": 0,
                                 "delta": {"type": "text_delta", "text": "ok"}}),
        ("content_block_stop", {"type": "content_block_stop", "index": 0}),
        ("message_delta", {"type": "message_delta",
                           "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                           "usage": _delta_usage(usage)}),
        ("message_stop", {"type": "message_stop"}),
    ]
    return "".join("event: {}\ndata: {}\n\n".format(name, json.dumps(data)) for name, data in events).encode()


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def do_POST(self):
        length = int(self.headers.get("content-length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        message_id = "msg_" + uuid.uuid4().hex
        usage = self.server.usage
        if body.get("stream"):
            self._send("text/event-stream", _stream_events(message_id, usage, self.server.start_usage))
            return
        self._send("application/json", json.dumps(_message(message_id, usage)).encode())

    def _send(self, content_type, payload):
        self.send_response(200)
        self.send_header("content-type", content_type)
        self.send_header("content-length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def start_upstream(usage, start_usage=None):
    """Serve the fake upstream on a free loopback port, answering with ``usage``; the caller shuts it down.

    ``start_usage`` replaces ``usage`` on a stream's ``message_start``, for a
    usage the live API sends only on ``message_delta``, such as ``iterations``.
    """
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    server.usage = usage
    server.start_usage = start_usage
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def stop_upstreams(servers):
    """Shut ``servers`` down after closing every stream response a LiteLLM call left open on them."""
    # A sync LiteLLM stream, even read to the end, leaves its httpcore response
    # open in a reference cycle. If the cycle collector closes it later, inside
    # the next request on the same pool, the close takes the pool's lock that
    # request already holds, and the suite hangs. Collecting here closes it
    # while no lock is held.
    gc.collect()
    for server in servers:
        server.shutdown()
        server.server_close()


def api_base(server):
    return "http://127.0.0.1:{}".format(server.server_address[1])
