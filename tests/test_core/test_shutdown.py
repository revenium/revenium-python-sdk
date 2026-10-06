"""BACK-3903: the SDK leaves the host's signals alone and drains within one budget.

Importing ``revenium_middleware`` used to replace SIGINT and SIGTERM with a
handler that ended the process through ``os._exit(0)``, skipping the host's
own graceful shutdown (uvicorn, the LiteLLM proxy's spend-log flush). The
exit drain also joined each metering thread for 5s, one after another; since
BACK-3904 it waits for the worker pool's queue within the same single budget.
"""
import json
import logging
import os
import signal
import subprocess
import sys
import textwrap
import threading
import time

import pytest

from revenium_middleware._core import metering, metering_buffer, metering_pool
from revenium_middleware._core.config import Config
from revenium_middleware._core.metering_buffer import MeteringBuffer
from revenium_middleware._core.metering_pool import MeteringWorkerPool
from revenium_middleware._core.shutdown_signals import chain_sigterm

BUDGET_ENV = Config.ENV_REVENIUM_SHUTDOWN_TIMEOUT_SECONDS
OPT_IN_ENV = Config.ENV_REVENIUM_INSTALL_SIGNAL_HANDLERS

posix_signals = pytest.mark.skipif(sys.platform == "win32", reason="POSIX signal delivery")


def _child_env(**extra):
    env = {k: v for k, v in os.environ.items() if not k.startswith("REVENIUM_")}
    env["REVENIUM_METERING_API_KEY"] = "hak_test_shutdown"
    env.update(extra)
    return env


IMPORT_PROBE = textwrap.dedent("""
    import json, signal, sys

    def sentinel(signum, frame):
        pass

    pre_import = {"default": None, "sentinel": sentinel, "ignore": signal.SIG_IGN}[sys.argv[1]]
    signals = {"SIGTERM": signal.SIGTERM, "SIGINT": signal.SIGINT}
    if pre_import is not None:
        for sig in signals.values():
            signal.signal(sig, pre_import)
    before = {name: signal.getsignal(sig) for name, sig in signals.items()}

    import revenium_middleware  # noqa: F401

    print(json.dumps({
        "unchanged": {name: signal.getsignal(sig) is before[name] for name, sig in signals.items()},
        "sigterm_was_default": before["SIGTERM"] is signal.SIG_DFL,
    }))
""")


def _probe_import(pre_import, **env):
    completed = subprocess.run([sys.executable, "-c", IMPORT_PROBE, pre_import], capture_output=True,
                               text=True, timeout=120, env=_child_env(**env))
    assert completed.returncode == 0, completed.stderr[-2000:]
    return json.loads(completed.stdout.strip().splitlines()[-1])


@pytest.mark.parametrize("pre_import", ["default", "sentinel", "ignore"])
def test_import_leaves_sigterm_and_sigint_unchanged(pre_import):
    result = _probe_import(pre_import)

    assert result["unchanged"] == {"SIGTERM": True, "SIGINT": True}
    assert result["sigterm_was_default"] is (pre_import == "default")


def test_opt_in_replaces_only_sigterm():
    result = _probe_import("default", **{OPT_IN_ENV: "1"})

    assert result["unchanged"] == {"SIGTERM": False, "SIGINT": True}


def test_opt_in_keeps_an_ignored_sigterm_ignored():
    result = _probe_import("ignore", **{OPT_IN_ENV: "1"})

    assert result["unchanged"] == {"SIGTERM": True, "SIGINT": True}


SIGTERM_CHILD = textwrap.dedent("""
    import os, signal, sys, time

    def say(word):
        os.write(1, (word + "\\n").encode())

    if sys.argv[1] == "sentinel":
        def sentinel(signum, frame):
            say("sentinel")
            sys.exit(17)
        signal.signal(signal.SIGTERM, sentinel)

    from revenium_middleware._core import metering_buffer
    from revenium_middleware._core.metering_buffer import MeteringBuffer

    buffer = MeteringBuffer(max_size=10, flush_interval=9999.0, replay_fn=lambda event, timeout: say("replayed"))
    buffer.push("ai", {"seq": "pending"})
    metering_buffer._buffer = buffer

    say("ready")
    while True:
        time.sleep(0.05)
""")


def _terminate_child(pre_import, opt_in):
    env = _child_env(**({OPT_IN_ENV: "1"} if opt_in else {}))
    child = subprocess.Popen([sys.executable, "-c", SIGTERM_CHILD, pre_import], stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE, text=True, env=env)
    try:
        assert child.stdout.readline().strip() == "ready"
        child.send_signal(signal.SIGTERM)
        stdout, stderr = child.communicate(timeout=60)
    finally:
        if child.poll() is None:
            child.kill()
            child.communicate()
    return child.returncode, stdout.split(), stderr


@posix_signals
@pytest.mark.parametrize("opt_in", [False, True])
def test_sigterm_exits_through_the_hosts_own_handler(opt_in):
    returncode, lines, stderr = _terminate_child("sentinel", opt_in)

    assert returncode == 17, stderr[-2000:]
    assert lines == ["sentinel", "replayed"]


@posix_signals
def test_opt_in_drains_before_a_default_sigterm_terminates_the_process():
    returncode, lines, stderr = _terminate_child("default", opt_in=True)

    assert returncode == -signal.SIGTERM, stderr[-2000:]
    assert lines == ["replayed"]


@posix_signals
def test_default_sigterm_without_opt_in_terminates_without_a_drain():
    returncode, lines, stderr = _terminate_child("default", opt_in=False)

    assert returncode == -signal.SIGTERM, stderr[-2000:]
    assert lines == []


UVICORN_HOST = textwrap.dedent("""
    import os, uvicorn

    def say(word):
        os.write(1, (word + "\\n").encode())

    async def app(scope, receive, send):
        if scope["type"] != "lifespan":
            return
        while True:
            message = await receive()
            if message["type"] == "lifespan.startup":
                import revenium_middleware  # noqa: F401
                await send({"type": "lifespan.startup.complete"})
                say("ready")
            elif message["type"] == "lifespan.shutdown":
                say("host-lifespan-shutdown")
                await send({"type": "lifespan.shutdown.complete"})
                return

    uvicorn.run(app, host="127.0.0.1", port=0, log_level="warning")
""")


@posix_signals
def test_uvicorn_shuts_down_gracefully_when_the_sdk_is_imported_during_startup():
    """The LiteLLM proxy imports its guardrails and callbacks during lifespan
    startup, after uvicorn has taken SIGTERM; the old import-time handler
    replaced uvicorn's and the lifespan shutdown never ran."""
    pytest.importorskip("uvicorn")
    child = subprocess.Popen([sys.executable, "-c", UVICORN_HOST], stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE, text=True, env=_child_env())
    try:
        assert child.stdout.readline().strip() == "ready"
        child.send_signal(signal.SIGTERM)
        stdout, stderr = child.communicate(timeout=60)
    finally:
        if child.poll() is None:
            child.kill()
            child.communicate()

    assert stdout.split() == ["host-lifespan-shutdown"], stderr[-2000:]
    assert child.returncode == -signal.SIGTERM


def test_chained_handler_defers_to_a_callable_without_draining():
    calls = []
    handler = chain_sigterm(lambda signum, frame: calls.append(("previous", signum, frame)),
                            drain=lambda: calls.append(("drain",)))

    handler(signal.SIGTERM, None)

    assert calls == [("previous", signal.SIGTERM, None)]


@pytest.fixture
def isolated_shutdown(monkeypatch):
    """Run handle_exit against only the worker pool and buffer a test installs."""
    monkeypatch.setattr(metering_buffer, "_buffer", None)
    monkeypatch.setattr(metering_pool, "_pool", None)
    metering.shutdown_event.clear()
    yield
    metering.shutdown_event.clear()
    if metering_pool._pool is not None:
        metering_pool._pool.stop(timeout=5)


def _queue_blocked_events(monkeypatch, count, release, workers=2):
    pool = MeteringWorkerPool(workers, queue_size=100, overflow=lambda task: task.discard())
    monkeypatch.setattr(metering_pool, "_pool", pool)

    async def blocked():
        release.wait(10)

    return [metering.run_async_in_thread(blocked()) for _ in range(count)]


def test_drain_with_many_queued_events_finishes_within_one_budget(isolated_shutdown, monkeypatch, caplog):
    budget = 0.5
    monkeypatch.setenv(BUDGET_ENV, str(budget))
    release = threading.Event()
    _queue_blocked_events(monkeypatch, 10, release)

    try:
        started = time.monotonic()
        with caplog.at_level(logging.DEBUG, logger="revenium_middleware"):
            metering.handle_exit()
        elapsed = time.monotonic() - started
    finally:
        release.set()

    assert budget - 0.05 <= elapsed < budget + 0.5
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert warnings[0].startswith("10 metering event(s) still queued, in flight or unbuilt after the 0.5s shutdown budget")


def test_drain_waits_for_events_that_finish_inside_the_budget(isolated_shutdown, monkeypatch, caplog):
    monkeypatch.setenv(BUDGET_ENV, "5")
    release = threading.Event()
    tasks = _queue_blocked_events(monkeypatch, 3, release)
    threading.Timer(0.1, release.set).start()

    with caplog.at_level(logging.WARNING, logger="revenium_middleware"):
        metering.handle_exit()

    assert not any(task.is_alive() for task in tasks)
    assert "still queued" not in caplog.text


def test_the_queue_drain_and_the_flush_share_one_deadline(isolated_shutdown, monkeypatch):
    budget = 0.6
    monkeypatch.setenv(BUDGET_ENV, str(budget))
    release = threading.Event()
    _queue_blocked_events(monkeypatch, 1, release)
    threading.Timer(0.4, release.set).start()
    per_call_timeouts = []
    buffer = MeteringBuffer(max_size=10, flush_interval=9999.0,
                            replay_fn=lambda event, timeout: per_call_timeouts.append(timeout))
    buffer.push("ai", {"seq": "pending"})
    monkeypatch.setattr(metering_buffer, "_buffer", buffer)

    started = time.monotonic()
    metering.handle_exit()

    assert time.monotonic() - started < budget + 0.3
    assert len(per_call_timeouts) == 1
    assert 0 < per_call_timeouts[0] <= budget - 0.35


def test_buffer_flush_shares_the_budget(isolated_shutdown, monkeypatch):
    budget = 0.75
    monkeypatch.setenv(BUDGET_ENV, str(budget))
    per_call_timeouts = []
    buffer = MeteringBuffer(max_size=10, flush_interval=9999.0,
                            replay_fn=lambda event, timeout: per_call_timeouts.append(timeout))
    buffer.push("ai", {"seq": "pending"})
    monkeypatch.setattr(metering_buffer, "_buffer", buffer)

    metering.handle_exit()

    assert len(per_call_timeouts) == 1
    assert 0 < per_call_timeouts[0] <= budget


def test_drain_gives_up_on_a_flush_still_running_at_the_deadline(isolated_shutdown, monkeypatch):
    budget = 0.3
    monkeypatch.setenv(BUDGET_ENV, str(budget))
    replayed = []
    buffer = MeteringBuffer(max_size=10, flush_interval=9999.0, replay_fn=lambda event, timeout: replayed.append(event))
    buffer.push("ai", {"seq": "pending"})
    monkeypatch.setattr(metering_buffer, "_buffer", buffer)
    buffer._flush_lock.acquire()

    try:
        started = time.monotonic()
        metering.handle_exit()
        elapsed = time.monotonic() - started
    finally:
        buffer._flush_lock.release()

    assert elapsed < budget + 0.5
    assert replayed == []
    assert buffer.stats()["size"] == 1


COMPLETION_ARGS = {
    "completion_start_time": "2026-10-06T00:00:00Z", "cost_type": "AI", "input_token_count": 1,
    "is_streamed": False, "model": "gpt-test", "output_token_count": 1, "provider": "OPENAI",
    "request_duration": 1, "request_time": "2026-10-06T00:00:00Z", "response_time": "2026-10-06T00:00:01Z",
    "stop_reason": "END", "total_token_count": 2, "transaction_id": "txn-shutdown",
    "extra_headers": {"Idempotency-Key": "txn-shutdown"},
}


def test_drain_against_a_failing_endpoint_sends_once_and_stays_within_the_budget(isolated_shutdown, monkeypatch):
    """The metering client retries a 503 twice with backoff, which alone
    outlasts a one-second budget; the replay must make a single attempt."""
    import httpx
    from revenium_middleware._metering import ReveniumMetering

    budget = 1.0
    monkeypatch.setenv(BUDGET_ENV, str(budget))
    requests = []

    def unavailable(request):
        requests.append(request)
        return httpx.Response(503, json={"error": "unavailable"})

    client = ReveniumMetering(api_key="hak_test_shutdown", base_url="https://metering.test/meter/",
                              http_client=httpx.Client(transport=httpx.MockTransport(unavailable)))
    monkeypatch.setattr(metering, "get_client", lambda: client)
    buffer = MeteringBuffer(max_size=10, flush_interval=9999.0)
    buffer.push("ai", {"operation": "completion", "args": dict(COMPLETION_ARGS)})
    monkeypatch.setattr(metering_buffer, "_buffer", buffer)

    started = time.monotonic()
    metering.handle_exit()
    elapsed = time.monotonic() - started

    assert len(requests) == 1
    assert elapsed < budget + 0.5
    assert buffer.stats()["size"] == 1


@pytest.mark.parametrize("raw, expected", [
    (None, metering.DEFAULT_SHUTDOWN_TIMEOUT_SECONDS),
    ("", metering.DEFAULT_SHUTDOWN_TIMEOUT_SECONDS),
    ("2.5", 2.5),
    ("0", 0.0),
])
def test_shutdown_budget_reads_the_environment(monkeypatch, raw, expected):
    if raw is None:
        monkeypatch.delenv(BUDGET_ENV, raising=False)
    else:
        monkeypatch.setenv(BUDGET_ENV, raw)

    assert metering.shutdown_budget_seconds() == expected


@pytest.mark.parametrize("raw", ["soon", "-1", "nan", "inf"])
def test_invalid_shutdown_budget_falls_back_to_the_default(monkeypatch, caplog, raw):
    monkeypatch.setenv(BUDGET_ENV, raw)

    with caplog.at_level(logging.WARNING, logger="revenium_middleware"):
        budget = metering.shutdown_budget_seconds()

    assert budget == metering.DEFAULT_SHUTDOWN_TIMEOUT_SECONDS
    assert f"Invalid {BUDGET_ENV}" in caplog.text
