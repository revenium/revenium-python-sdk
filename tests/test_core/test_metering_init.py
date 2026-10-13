"""Tests for the metering client bootstrap (`_build_metering_client`).

Tests call the builder directly to keep module-level state untouched — earlier
versions used ``importlib.reload`` and caused state-leakage failures in
``tests/test_metering.py``.
"""

import logging
import os
import subprocess
import sys
import textwrap
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx
import pytest

import revenium_middleware._core as core
from revenium_middleware._core import metering
from revenium_middleware._core.metering import _build_metering_client, get_client, metering_key
from revenium_middleware._metering import APIConnectionError


def test_unset_api_key_returns_none():
    assert _build_metering_client(None, None) is None


def test_empty_api_key_returns_none():
    assert _build_metering_client("", None) is None


def test_invalid_prefix_raises():
    with pytest.raises(ValueError, match='should start with "hak_" or "rev_"'):
        _build_metering_client("invalid_no_prefix", None)


def test_hak_prefix_initializes_client():
    client = _build_metering_client("hak_test123abc", None)
    assert client is not None


def test_rev_prefix_initializes_client():
    client = _build_metering_client("rev_mk_test123abc", None)
    assert client is not None


MALFORMED_KEY_PROBE = textwrap.dedent("""
    import os
    import revenium_middleware
    from revenium_middleware._core.metering import get_client
    from revenium_middleware._core.metering_submission import submit_ai_event

    print("client-is-none=%s" % (get_client() is None))
    print("submit-returned-none=%s" % (submit_ai_event("completion", {}) is None))
    os.environ["REVENIUM_METERING_API_KEY"] = "hak_fixed_later_dummy"
    print("client-built-after-fix=%s" % (get_client() is not None))
""")


def test_malformed_env_key_at_import_warns_and_runs_unmetered():
    env = {k: v for k, v in os.environ.items() if not k.startswith("REVENIUM_")}
    env["REVENIUM_METERING_API_KEY"] = "sk-not-a-revenium-key-MARKER123"
    env["REVENIUM_METERING_BASE_URL"] = "http://127.0.0.1:9"

    completed = subprocess.run([sys.executable, "-c", MALFORMED_KEY_PROBE], capture_output=True,
                               text=True, timeout=120, env=env)

    assert completed.returncode == 0, completed.stderr[-2000:]
    assert "client-is-none=True" in completed.stdout
    assert "submit-returned-none=True" in completed.stdout
    assert "client-built-after-fix=True" in completed.stdout
    assert "malformed" in completed.stderr
    assert "MARKER123" not in completed.stdout + completed.stderr


PADDED_KEY = "hak_padded_dummy"


class MeteringStub:
    """A local usage endpoint that answers 200 and keeps the ``x-api-key`` of each request."""

    def __init__(self):
        api_keys = self.api_keys = []

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                self.rfile.read(int(self.headers.get("Content-Length", 0)))
                api_keys.append(self.headers.get("x-api-key"))
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", "2")
                self.end_headers()
                self.wfile.write(b"{}")

            def log_message(self, *args):
                pass

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._server.daemon_threads = True
        threading.Thread(target=self._server.serve_forever, daemon=True).start()
        self.meter_url = f"http://127.0.0.1:{self._server.server_address[1]}/meter/"

    def close(self):
        self._server.shutdown()
        self._server.server_close()


@pytest.fixture
def usage_endpoint(monkeypatch):
    monkeypatch.setenv("REVENIUM_METERING_MAX_RETRIES", "0")
    stub = MeteringStub()
    yield stub
    stub.close()


def deliver_one_record(client):
    """Post one usage record; the name of the error it failed with, or None when it was sent."""
    try:
        client.post("/v2/ai/completions", body={}, cast_to=httpx.Response)
    except APIConnectionError as error:
        return type(error).__name__
    return None


@pytest.mark.parametrize("suffix", ["\n", " ", "\r\n", "\t"], ids=["newline", "space", "crlf", "tab"])
def test_a_key_with_a_trailing_newline_or_space_is_delivered_trimmed(usage_endpoint, suffix, caplog):
    caplog.set_level(logging.DEBUG)
    client = _build_metering_client(PADDED_KEY + suffix, usage_endpoint.meter_url)

    assert deliver_one_record(client) is None
    assert usage_endpoint.api_keys == [PADDED_KEY]
    assert not [record for record in caplog.records if "Connection error" in record.getMessage()]


def test_an_unpadded_key_still_delivers_one_request(usage_endpoint):
    client = _build_metering_client(PADDED_KEY, usage_endpoint.meter_url)

    assert deliver_one_record(client) is None
    assert usage_endpoint.api_keys == [PADDED_KEY]


@pytest.mark.parametrize("key, code_point, position", [
    ("hak_a b", "U+0020", 6),
    ("hak_a\x00b", "U+0000", 6),
    ("hak_a\nb", "U+000A", 6),
    ("hak_a\u00e9b", "U+00E9", 6),
    ("hak_a\u200bb", "U+200B", 6),
], ids=["inner-space", "nul", "inner-newline", "accented", "zero-width-space"])
def test_a_key_with_a_space_or_control_character_inside_is_refused_at_setup(key, code_point, position):
    with pytest.raises(ValueError) as refused:
        _build_metering_client(key, None)

    message = str(refused.value)
    assert f"{code_point} at position {position}" in message
    assert key not in message


def test_a_whitespace_only_key_counts_as_unset():
    assert _build_metering_client(" \n", None) is None


def test_metering_key_trims_a_padded_key():
    with metering_key("hak_scoped_dummy\n", base_url="http://127.0.0.1:9/meter/"):
        assert get_client().api_key == "hak_scoped_dummy"


def test_a_whitespace_only_env_key_logs_once_not_per_event(monkeypatch, caplog):
    monkeypatch.setenv("REVENIUM_METERING_API_KEY", " \n")
    monkeypatch.setattr(metering, "client", None)
    monkeypatch.setattr(core, "client", core.client)
    monkeypatch.setattr(metering, "_last_failed_key", None)
    caplog.set_level(logging.WARNING, logger="revenium_middleware")

    results = [metering.process_client() for _event in range(3)]

    assert results == [None, None, None]
    assert len([record for record in caplog.records if record.levelno == logging.ERROR]) <= 1
