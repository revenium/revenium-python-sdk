"""BACK-3910: a LiteLLM proxy restart sends the usage records the SDK still holds.

uvicorn ends a process stopped by SIGTERM by re-raising the signal right after
the app's lifespan shutdown, before ``atexit`` runs, so the SDK's exit drain
never ran under the proxy and the last records were lost. LiteLLM calls no
shutdown method on a guardrail or callback, so the drain runs from the ASGI
middleware ``DrainMeteringOnShutdown`` around the proxy's app.
"""
import asyncio
import json
import logging
import os
import signal
import socket
import subprocess
import sys
import threading
import time
import uuid

import httpx
import pytest
import yaml

pytest.importorskip("litellm")

from revenium_middleware._core import metering, metering_buffer, metering_pool, metering_status  # noqa: E402
from revenium_middleware._core.config import Config  # noqa: E402
from revenium_middleware._core.metering_buffer import MeteringBuffer  # noqa: E402
from revenium_middleware._core.metering_pool import MeteringWorkerPool  # noqa: E402
from revenium_middleware.litellm.proxy.shutdown import DrainMeteringOnShutdown  # noqa: E402

from .fake_openai_upstream import api_base, start_upstream  # noqa: E402

BUDGET_ENV = Config.ENV_REVENIUM_SHUTDOWN_TIMEOUT_SECONDS

posix_signals = pytest.mark.skipif(sys.platform == "win32", reason="POSIX signal delivery")


@pytest.fixture
def isolated_shutdown(monkeypatch):
    """Run the exit drain against only the buffer a test installs, and reopen metering afterwards."""
    monkeypatch.setattr(metering_buffer, "_buffer", None)
    monkeypatch.setattr(metering_pool, "_pool", None)
    metering.shutdown_event.clear()
    yield
    metering.shutdown_event.clear()
    if metering_pool._pool is not None:
        metering_pool._pool.stop(timeout=5)


def _buffer_holding_one_record(monkeypatch, replay):
    buffer = MeteringBuffer(max_size=10, flush_interval=9999.0, replay_fn=replay)
    buffer.push("ai", {"seq": "held"})
    monkeypatch.setattr(metering_buffer, "_buffer", buffer)
    return buffer


class _LifespanApp:
    """An ASGI app that answers the lifespan protocol and records what it saw."""

    def __init__(self, events, shutdown_reply="lifespan.shutdown.complete"):
        self.events = events
        self.shutdown_reply = shutdown_reply

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            self.events.append("app-request")
            await send({"type": "http.response.start", "status": 204, "headers": []})
            await send({"type": "http.response.body", "body": b""})
            return
        while True:
            message = await receive()
            if message["type"] == "lifespan.startup":
                await send({"type": "lifespan.startup.complete"})
            elif message["type"] == "lifespan.shutdown":
                self.events.append("app-shutdown")
                await send({"type": self.shutdown_reply})
                return


async def _run_lifespan(app, events):
    """Drive ``app`` through startup and shutdown as an ASGI server does, recording what the server is sent."""
    inbox = asyncio.Queue()
    for message in ("lifespan.startup", "lifespan.shutdown"):
        inbox.put_nowait({"type": message})

    async def send(message):
        events.append("server-heard " + message["type"])

    await app({"type": "lifespan", "asgi": {"version": "3.0"}}, inbox.get, send)


@pytest.mark.asyncio
@pytest.mark.parametrize("shutdown_reply", ["lifespan.shutdown.complete", "lifespan.shutdown.failed"])
async def test_the_drain_runs_after_the_apps_own_shutdown_and_before_the_server_hears_it_is_over(
    isolated_shutdown, monkeypatch, shutdown_reply
):
    events = []
    _buffer_holding_one_record(monkeypatch, lambda event, timeout: events.append("replayed"))

    await _run_lifespan(DrainMeteringOnShutdown(_LifespanApp(events, shutdown_reply)), events)

    assert events == [
        "server-heard lifespan.startup.complete", "app-shutdown", "replayed", "server-heard " + shutdown_reply,
    ]
    assert metering.shutdown_event.is_set()


@pytest.mark.asyncio
async def test_a_request_passes_through_without_a_drain(isolated_shutdown, monkeypatch):
    events = []
    _buffer_holding_one_record(monkeypatch, lambda event, timeout: events.append("replayed"))
    app = DrainMeteringOnShutdown(_LifespanApp(events))

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://proxy.test") as client:
        response = await client.get("/v1/models")

    assert response.status_code == 204
    assert events == ["app-request"]
    assert not metering.shutdown_event.is_set()


@pytest.mark.asyncio
async def test_a_shutdown_with_revenium_not_answering_waits_out_the_budget_and_no_longer(
    isolated_shutdown, monkeypatch, caplog
):
    budget = 0.5
    monkeypatch.setenv(BUDGET_ENV, str(budget))
    release = threading.Event()

    async def delivery_revenium_never_answers():
        release.wait(30)

    monkeypatch.setattr(metering_pool, "_pool", MeteringWorkerPool(1, queue_size=10, overflow=lambda task: task.discard()))
    metering.run_async_in_thread(delivery_revenium_never_answers())
    try:
        started = time.monotonic()
        with caplog.at_level(logging.WARNING, logger="revenium_middleware"):
            await _run_lifespan(DrainMeteringOnShutdown(_LifespanApp([])), [])
        elapsed = time.monotonic() - started
    finally:
        release.set()

    assert metering.shutdown_event.is_set()
    assert "1 metering event(s) still queued" in caplog.text
    assert budget * 0.8 <= elapsed < budget + 0.5


def _rejected(status_code):
    request = httpx.Request("POST", "http://revenium.test/meter/v2/ai/completions")
    return httpx.HTTPStatusError(str(status_code), request=request, response=httpx.Response(status_code, request=request))


def _join_drain_threads():
    for thread in threading.enumerate():
        if thread.name == "revenium-shutdown-drain":
            thread.join(5)


@pytest.mark.asyncio
async def test_a_shutdown_ends_within_the_budget_while_an_error_subscriber_is_still_running(
    isolated_shutdown, monkeypatch, caplog
):
    """A record Revenium rejects for good reaches ``on_metering_error`` subscribers after the flush's deadline checks."""
    budget = 0.5
    monkeypatch.setenv(BUDGET_ENV, str(budget))
    subscriber_called, release = threading.Event(), threading.Event()

    def replay_rejected(event, timeout):
        raise _rejected(401)

    def alert_to_a_service_that_hangs(event):
        subscriber_called.set()
        release.wait(30)

    _buffer_holding_one_record(monkeypatch, replay_rejected)
    metering_status.on_metering_error(alert_to_a_service_that_hangs)
    try:
        started = time.monotonic()
        with caplog.at_level(logging.WARNING, logger="revenium_middleware"):
            await _run_lifespan(DrainMeteringOnShutdown(_LifespanApp([])), [])
        elapsed = time.monotonic() - started
    finally:
        release.set()
        metering_status.remove_metering_error_callback(alert_to_a_service_that_hangs)
        _join_drain_threads()

    assert subscriber_called.is_set()
    assert elapsed < budget + 1.0
    assert "still running when its 0.5s shutdown budget" in caplog.text


@pytest.mark.asyncio
async def test_the_event_loop_keeps_running_while_the_drain_waits(isolated_shutdown, monkeypatch):
    """Callbacks LiteLLM still has scheduled on the loop can run, and queue their records, during the drain."""
    replaying, release = threading.Event(), threading.Event()

    def slow_replay(event, timeout):
        replaying.set()
        release.wait(5)

    _buffer_holding_one_record(monkeypatch, slow_replay)

    async def release_once_the_drain_is_replaying():
        while not replaying.is_set():
            await asyncio.sleep(0.01)
        release.set()

    helper = asyncio.ensure_future(release_once_the_drain_is_replaying())
    try:
        started = time.monotonic()
        await _run_lifespan(DrainMeteringOnShutdown(_LifespanApp([])), [])
        elapsed = time.monotonic() - started
    finally:
        release.set()
        helper.cancel()

    assert replaying.is_set()
    assert elapsed < 2


@pytest.mark.asyncio
async def test_a_second_shutdown_does_not_drain_again(isolated_shutdown, monkeypatch):
    replayed = []
    buffer = _buffer_holding_one_record(monkeypatch, lambda event, timeout: replayed.append(event.payload["seq"]))
    app = DrainMeteringOnShutdown(DrainMeteringOnShutdown(_LifespanApp([])))

    await _run_lifespan(app, [])
    buffer.push("ai", {"seq": "after-the-drain"})
    started = time.monotonic()
    await _run_lifespan(app, [])

    assert replayed == ["held"]
    assert time.monotonic() - started < 0.5


@pytest.mark.asyncio
async def test_a_drain_already_running_from_a_signal_or_atexit_is_waited_for_not_repeated(isolated_shutdown, monkeypatch):
    replayed, replaying, release = [], threading.Event(), threading.Event()

    def slow_replay(event, timeout):
        replaying.set()
        release.wait(5)
        replayed.append(event.payload["seq"])

    _buffer_holding_one_record(monkeypatch, slow_replay)
    other_drain = threading.Thread(target=metering.handle_exit)
    other_drain.start()
    assert replaying.wait(5)
    threading.Timer(0.3, release.set).start()

    await _run_lifespan(DrainMeteringOnShutdown(_LifespanApp([])), [])

    assert replayed == ["held"]
    other_drain.join(5)


# uvicorn under a real LiteLLM proxy. One metering worker and a Revenium that
# takes a while to answer leave the second record queued when SIGTERM arrives,
# so it reaches Revenium only if something drains the queue in the shutdown.
REVENIUM_ANSWER_SECONDS = 2.0
MASTER_KEY = "sk-back-3910-" + uuid.uuid4().hex


class _ReveniumStub:
    """A local metering endpoint that counts the records it receives and answers each after ``delay`` seconds."""

    def __init__(self, delay):
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

        received = self.received = []

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                received.append(json.loads(self.rfile.read(int(self.headers["Content-Length"])))["transactionId"])
                time.sleep(delay)
                reply = json.dumps({"id": "m", "label": "m", "resourceType": "metering", "signature": "s"}).encode()
                self.send_response(201)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(reply)))
                self.end_headers()
                self.wfile.write(reply)

            def log_message(self, *args):
                pass

        class Server(ThreadingHTTPServer):
            daemon_threads = True
            block_on_close = False

        self._server = Server(("127.0.0.1", 0), Handler)
        self.base_url = f"http://127.0.0.1:{self._server.server_address[1]}/meter/"

    def __enter__(self):
        threading.Thread(target=self._server.serve_forever, daemon=True).start()
        return self

    def __exit__(self, *exc):
        self._server.shutdown()
        self._server.server_close()


@pytest.fixture(scope="module")
def upstream():
    server = start_upstream()
    yield server
    server.shutdown()


@pytest.fixture
def proxy_config(upstream, tmp_path):
    config = {
        "model_list": [{"model_name": "gpt-4o-mini", "litellm_params": {
            "model": "openai/gpt-4o-mini", "api_key": "sk-back-3910-fake", "api_base": api_base(upstream)}}],
        "guardrails": [{"guardrail_name": "revenium", "litellm_params": {
            "guardrail": "revenium_middleware.litellm.proxy.guardrail.ReveniumGuardrail",
            "mode": ["pre_call", "post_call"], "default_on": True}}],
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def _free_port():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def _proxy_env(config_path, revenium, budget):
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("REVENIUM_", "LITELLM_")) and k not in ("DATABASE_URL", "CONFIG_FILE_PATH")}
    env.update({
        "CONFIG_FILE_PATH": str(config_path),
        "LITELLM_LOCAL_MODEL_COST_MAP": "True",
        "LITELLM_MASTER_KEY": MASTER_KEY,
        "NO_PROXY": "127.0.0.1",
        "REVENIUM_METERING_API_KEY": "hak_test_back_3910",
        "REVENIUM_METERING_BASE_URL": revenium.base_url,
        "REVENIUM_METERING_WORKERS": "1",
        BUDGET_ENV: str(budget),
    })
    return env


def _serve_two_calls_then_sigterm(app_path, config_path, revenium, budget):
    """Run the proxy under uvicorn, make two calls, send SIGTERM; return (exit code, seconds to exit, stderr)."""
    port = _free_port()
    proxy = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", app_path, "--host", "127.0.0.1", "--port", str(port), "--log-level", "info"],
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, env=_proxy_env(config_path, revenium, budget),
    )
    try:
        with httpx.Client(base_url=f"http://127.0.0.1:{port}", headers={"Authorization": f"Bearer {MASTER_KEY}"},
                          timeout=30, trust_env=False) as client:
            _wait_until_ready(client, proxy)
            for _ in range(2):
                response = client.post("/v1/chat/completions", json={
                    "model": "gpt-4o-mini", "messages": [{"role": "user", "content": "hi"}]})
                assert response.status_code == 200, response.text
        signalled = time.monotonic()
        proxy.send_signal(signal.SIGTERM)
        _, stderr = proxy.communicate(timeout=60)
        return proxy.returncode, time.monotonic() - signalled, stderr
    finally:
        if proxy.poll() is None:
            proxy.kill()
            proxy.communicate()


def _wait_until_ready(client, proxy):
    deadline = time.monotonic() + 90
    while time.monotonic() < deadline:
        assert proxy.poll() is None, proxy.stderr.read()[-4000:]
        try:
            if client.get("/health/liveliness").status_code == 200:
                return
        except httpx.TransportError:
            pass
        time.sleep(0.2)
    pytest.fail("the LiteLLM proxy did not start within 90 seconds")


def _require_proxy():
    pytest.importorskip("uvicorn")
    pytest.importorskip("litellm.proxy.proxy_server")


@posix_signals
def test_a_sigterm_to_the_proxy_under_uvicorn_delivers_the_records_still_queued(proxy_config):
    _require_proxy()
    budget = 10.0
    with _ReveniumStub(delay=REVENIUM_ANSWER_SECONDS) as revenium:
        returncode, seconds_to_exit, stderr = _serve_two_calls_then_sigterm(
            "revenium_middleware.litellm.proxy.asgi:app", proxy_config, revenium, budget)

    assert len(revenium.received) == 2, stderr[-4000:]
    assert returncode == -signal.SIGTERM, stderr[-4000:]
    assert "still queued" not in stderr
    assert seconds_to_exit < budget + 5


@posix_signals
def test_a_sigterm_to_the_proxy_with_revenium_not_answering_ends_within_the_budget(proxy_config):
    _require_proxy()
    budget = 1.0
    with _ReveniumStub(delay=60) as revenium:
        returncode, seconds_to_exit, stderr = _serve_two_calls_then_sigterm(
            "revenium_middleware.litellm.proxy.asgi:app", proxy_config, revenium, budget)

    assert returncode == -signal.SIGTERM, stderr[-4000:]
    assert seconds_to_exit < budget + 5, stderr[-4000:]
    assert "2 metering event(s) still queued" in stderr, stderr[-4000:]


@posix_signals
def test_guard_the_unwrapped_proxy_app_still_loses_the_records_still_queued(proxy_config):
    """Passes before and after the fix: it pins the loss the wrapper exists for, so a LiteLLM or uvicorn
    release that starts draining on its own shows up here."""
    _require_proxy()
    with _ReveniumStub(delay=REVENIUM_ANSWER_SECONDS) as revenium:
        returncode, _, stderr = _serve_two_calls_then_sigterm(
            "litellm.proxy.proxy_server:app", proxy_config, revenium, 10.0)

    assert returncode == -signal.SIGTERM, stderr[-4000:]
    assert len(revenium.received) == 1, stderr[-4000:]
