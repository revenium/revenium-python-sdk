"""A local Bedrock Converse upstream for driving real LiteLLM calls offline.

Serves ``/model/<id>/converse`` and ``/model/<id>/converse-stream`` on a
loopback port, answering with the Converse usage block the server was started
with, so a test exercises LiteLLM's own Converse transformation (the one that
folds cache tokens into ``prompt_tokens``) instead of ``mock_response``.
LiteLLM reaches it through ``aws_bedrock_runtime_endpoint``; any AWS keys sign
the request, and nothing checks the signature.

A streamed answer is AWS's binary event stream: botocore ships a decoder for
it, not an encoder, so the frames are built here (prelude, headers, payload,
two CRC32s), which is what LiteLLM's ``AWSEventStreamDecoder`` reads.

Not a test module (no test_ prefix) so pytest imports it rather than collecting it.
"""
import json
import struct
import threading
import zlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

_STRING_HEADER = 7


def _header(name, value):
    name_bytes, value_bytes = name.encode(), value.encode()
    return (struct.pack(">B", len(name_bytes)) + name_bytes
            + struct.pack(">BH", _STRING_HEADER, len(value_bytes)) + value_bytes)


def _frame(event_type, payload):
    headers = b"".join(_header(name, value) for name, value in (
        (":event-type", event_type),
        (":content-type", "application/json"),
        (":message-type", "event"),
    ))
    body = json.dumps(payload).encode()
    prelude = struct.pack(">II", 12 + len(headers) + len(body) + 4, len(headers))
    message = prelude + struct.pack(">I", zlib.crc32(prelude)) + headers + body
    return message + struct.pack(">I", zlib.crc32(message))


def _converse_body(usage):
    return {
        "output": {"message": {"role": "assistant", "content": [{"text": "ok"}]}},
        "stopReason": "end_turn",
        "usage": usage,
        "metrics": {"latencyMs": 10},
    }


def _stream_frames(usage):
    return b"".join(_frame(event_type, payload) for event_type, payload in (
        ("messageStart", {"role": "assistant"}),
        ("contentBlockDelta", {"contentBlockIndex": 0, "delta": {"text": "ok"}}),
        ("contentBlockStop", {"contentBlockIndex": 0}),
        ("messageStop", {"stopReason": "end_turn"}),
        ("metadata", {"usage": usage, "metrics": {"latencyMs": 10}}),
    ))


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def do_POST(self):
        self.rfile.read(int(self.headers.get("content-length", 0)))
        usage = self.server.usage
        if self.path.endswith("/converse-stream"):
            self._send("application/vnd.amazon.eventstream", _stream_frames(usage))
            return
        self._send("application/json", json.dumps(_converse_body(usage)).encode())

    def _send(self, content_type, payload):
        self.send_response(200)
        self.send_header("content-type", content_type)
        self.send_header("content-length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def start_upstream(usage):
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    server.usage = usage
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def endpoint(server):
    return "http://127.0.0.1:{}".format(server.server_address[1])


def converse_usage(input_tokens=12, cache_read=0, cache_write=0, output_tokens=5):
    """A Converse ``usage`` block, totalled the way Bedrock totals it."""
    return {
        "inputTokens": input_tokens,
        "outputTokens": output_tokens,
        "totalTokens": input_tokens + cache_read + cache_write + output_tokens,
        "cacheReadInputTokens": cache_read,
        "cacheWriteInputTokens": cache_write,
    }
