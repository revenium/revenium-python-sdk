"""BACK-3903: the SDK leaves the host's signals alone and drains within one budget.

Importing ``revenium_middleware`` used to replace SIGINT and SIGTERM with a
handler that ended the process through ``os._exit(0)``, skipping the host's
own graceful shutdown (uvicorn, the LiteLLM proxy's spend-log flush). The
exit drain also joined each metering thread for 5s, one after another; since
BACK-3904 it waits for the worker pool's queue within the same single budget.
"""
import asyncio
import contextvars
import json
import logging
import os
import signal
import subprocess
import sys
import textwrap
import threading
import time

import httpx
import pytest

from revenium_middleware._core import metering, metering_buffer, metering_pool
from revenium_middleware._core.config import Config
from revenium_middleware._core.metering_buffer import MeteringBuffer
from revenium_middleware._core.metering_pool import MeteringTask, MeteringWorkerPool
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


def test_a_padded_opt_in_replaces_sigterm():
    result = _probe_import("default", **{OPT_IN_ENV: "1 "})

    assert result["unchanged"] == {"SIGTERM": False, "SIGINT": True}


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


SIGTERM_WHILE_HOLDING_CHILD = textwrap.dedent("""
    import os, signal, sys, time

    from revenium_middleware._core import metering_buffer, metering_status
    from revenium_middleware._core.metering_buffer import MeteringBuffer

    def say(word):
        os.write(1, (word + "\\n").encode())

    buffer = MeteringBuffer(max_size=10, flush_interval=9999.0, replay_fn=lambda event, timeout: say("replayed"))
    buffer.push("ai", {"seq": "pending"})
    metering_buffer._buffer = buffer
    in_progress = metering_buffer._OverflowBuild([])
    buffer._active_builds.add(in_progress)
    held = {"buffer": buffer._lock, "status": metering_status._lock, "overflow-build": in_progress._lock}[sys.argv[1]]

    with held:
        signal.raise_signal(signal.SIGTERM)
    say("survived")
""")


@posix_signals
@pytest.mark.parametrize("held", ["buffer", "status", "overflow-build"])
def test_opt_in_sigterm_drains_even_when_it_interrupts_the_main_thread_inside_a_drain_lock(held):
    """The handler runs the drain on the main thread, which may be inside one of the locks the drain takes."""
    env = _child_env(**{OPT_IN_ENV: "1", BUDGET_ENV: "2"})
    completed = subprocess.run([sys.executable, "-c", SIGTERM_WHILE_HOLDING_CHILD, held], capture_output=True,
                               text=True, timeout=30, env=env)

    assert completed.returncode == -signal.SIGTERM, completed.stderr[-2000:]
    assert completed.stdout.split() == ["replayed"]


SIGTERM_DURING_DRAIN_CHILD = textwrap.dedent("""
    import os, sys, threading, time

    from revenium_middleware._core import metering, metering_buffer
    from revenium_middleware._core.metering_buffer import MeteringBuffer

    def say(word):
        os.write(1, (word + "\\n").encode())

    def slow_replay(event, timeout):
        seq = event.payload["seq"]
        say("replaying-%d" % seq)
        time.sleep(0.5)
        say("replayed-%d" % seq)

    buffer = MeteringBuffer(max_size=10, flush_interval=9999.0, replay_fn=slow_replay)
    for seq in range(3):
        buffer.push("ai", {"seq": seq})
    metering_buffer._buffer = buffer

    if sys.argv[1] == "drain-on-another-thread":
        threading.Thread(target=metering.handle_exit).start()
        while True:
            time.sleep(0.05)
""")


@posix_signals
@pytest.mark.parametrize("drain_runs", ["atexit-on-the-main-thread", "drain-on-another-thread"])
def test_a_sigterm_during_the_exit_drain_lets_it_send_the_remaining_records_first(drain_runs):
    """The opt-in handler used to run a second drain under the interrupted one, which waited on the
    flush lock its own thread held until the deadline, then ended the process before the rest were sent."""
    env = _child_env(**{OPT_IN_ENV: "1", BUDGET_ENV: "10"})
    child = subprocess.Popen([sys.executable, "-c", SIGTERM_DURING_DRAIN_CHILD, drain_runs], stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE, text=True, env=env)
    try:
        assert child.stdout.readline().strip() == "replaying-0"
        signalled = time.monotonic()
        child.send_signal(signal.SIGTERM)
        stdout, stderr = child.communicate(timeout=60)
        elapsed = time.monotonic() - signalled
    finally:
        if child.poll() is None:
            child.kill()
            child.communicate()

    assert stdout.split() == ["replayed-0", "replaying-1", "replayed-1", "replaying-2", "replayed-2"], stderr[-2000:]
    assert child.returncode == -signal.SIGTERM
    assert elapsed < 5


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
    assert warnings[0].startswith(
        "10 metering event(s) still queued, in flight, unbuilt or unsent after the 0.5s shutdown budget")


def test_manual_records_queued_before_exit_are_delivered_by_the_exit_drain(
    isolated_metering, isolated_shutdown, recording_meters, monkeypatch
):
    """BACK-4451: emit_completion and emit_tool_event queue their records, and the exit drain sends them."""
    from revenium_middleware.agentic_outcomes import AgenticOutcomeClient, AgenticOutcomeSettings

    monkeypatch.setenv(BUDGET_ENV, "5")
    meter, _ = recording_meters
    meter.delay = 0.3
    settings = AgenticOutcomeSettings(api_key="rev_sk_fake_exit_drain_write_key", meter_base_url=meter.base_url)
    completion = {
        "completionStartTime": "2026-10-11T00:00:00Z", "costType": "AI", "inputTokenCount": 3,
        "isStreamed": False, "model": "gpt-4o-mini", "outputTokenCount": 2, "provider": "OPENAI",
        "requestDuration": 10, "requestTime": "2026-10-11T00:00:00Z", "responseTime": "2026-10-11T00:00:01Z",
        "stopReason": "END", "totalTokenCount": 5, "transactionId": "txn-exit-drain",
    }
    with AgenticOutcomeClient(settings) as outcomes:
        for _ in range(5):
            outcomes.emit_completion(dict(completion))
        for _ in range(2):
            outcomes.emit_tool_event({"transactionId": "tool-exit-drain", "toolId": "t1"})
        queued_at_exit = 0 if metering_pool._pool is None else metering_pool._pool.pending()

        metering.handle_exit()

    assert queued_at_exit > 0
    paths = sorted(request["path"] for request in meter.requests)
    assert paths == ["/meter/v2/ai/completions"] * 5 + ["/meter/v2/tool/events"] * 2


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


def _unreachable(event, timeout):
    raise httpx.ConnectError("metering endpoint unreachable")


def _shutdown_warnings(caplog):
    return [r.getMessage() for r in caplog.records
            if r.levelno == logging.WARNING and "shutdown budget" in r.getMessage()]


def test_a_failed_final_replay_is_counted_in_the_shutdown_warning(isolated_shutdown, monkeypatch, caplog):
    monkeypatch.setenv(BUDGET_ENV, "1")
    buffer = MeteringBuffer(max_size=10, flush_interval=9999.0, replay_fn=_unreachable)
    buffer.push("ai", {"seq": "unsent"})
    monkeypatch.setattr(metering_buffer, "_buffer", buffer)

    with caplog.at_level(logging.WARNING, logger="revenium_middleware"):
        metering.handle_exit()

    assert [w.split(" metering event(s)")[0] for w in _shutdown_warnings(caplog)] == ["1"]


def test_records_a_background_replay_still_holds_at_the_deadline_are_counted_in_the_shutdown_warning(
    isolated_shutdown, monkeypatch, caplog
):
    monkeypatch.setenv(BUDGET_ENV, "0.3")
    release = threading.Event()
    replaying = threading.Barrier(5)

    def stuck(event, timeout):
        replaying.wait(5)
        release.wait(10)

    buffer = MeteringBuffer(max_size=10, flush_interval=9999.0, replay_fn=stuck, replay_concurrency=4)
    for seq in range(4):
        buffer.push("ai", {"seq": seq})
    monkeypatch.setattr(metering_buffer, "_buffer", buffer)
    background = threading.Thread(target=buffer.flush, daemon=True)
    background.start()
    replaying.wait(5)

    try:
        with caplog.at_level(logging.WARNING, logger="revenium_middleware"):
            metering.handle_exit()
    finally:
        release.set()
        background.join(5)

    assert [w.split(" metering event(s)")[0] for w in _shutdown_warnings(caplog)] == ["4"]
    assert buffer.undelivered() == 0


def test_the_final_flush_stops_at_the_first_refused_record_and_the_warning_counts_the_rest(
    isolated_shutdown, monkeypatch, caplog
):
    monkeypatch.setenv(BUDGET_ENV, "1")
    attempts = []

    def unreachable(event, timeout):
        attempts.append(event.payload["seq"])
        _unreachable(event, timeout)

    buffer = MeteringBuffer(max_size=10, flush_interval=9999.0, replay_fn=unreachable)
    for seq in range(3):
        buffer.push("ai", {"seq": seq})
    monkeypatch.setattr(metering_buffer, "_buffer", buffer)

    with caplog.at_level(logging.WARNING, logger="revenium_middleware"):
        metering.handle_exit()

    assert attempts == [0]
    assert [w.split(" metering event(s)")[0] for w in _shutdown_warnings(caplog)] == ["3"]


def test_the_shutdown_warning_counts_queued_unbuilt_and_unsent_events_once_each(isolated_shutdown, monkeypatch, caplog):
    monkeypatch.setenv(BUDGET_ENV, "0.3")
    release = threading.Event()
    _queue_blocked_events(monkeypatch, 3, release, workers=1)
    buffer = MeteringBuffer(max_size=10, flush_interval=9999.0, replay_fn=_unreachable)
    buffer.push("ai", {"seq": "unsent"})

    async def blocked_build():
        release.wait(10)

    overflowed = [MeteringTask(blocked_build(), contextvars.copy_context()) for _ in range(2)]
    for task in overflowed:
        buffer.push_overflow(task)
    monkeypatch.setattr(metering_buffer, "_buffer", buffer)

    try:
        with caplog.at_level(logging.WARNING, logger="revenium_middleware"):
            metering.handle_exit()
    finally:
        release.set()
        for task in overflowed:
            task.discard()

    assert [w.split(" metering event(s)")[0] for w in _shutdown_warnings(caplog)] == ["6"]


def test_a_build_the_deadline_cuts_short_on_the_exiting_thread_is_counted_in_the_shutdown_warning(
    isolated_shutdown, monkeypatch, caplog
):
    monkeypatch.setenv(BUDGET_ENV, "0.3")

    def refuse(thread):
        raise RuntimeError("can't create new thread at interpreter shutdown")

    monkeypatch.setattr(threading.Thread, "start", refuse)
    buffer = MeteringBuffer(max_size=10, flush_interval=9999.0, replay_fn=lambda event, timeout: None)

    async def awaits_past_the_budget():
        await asyncio.sleep(5)

    task = MeteringTask(awaits_past_the_budget(), contextvars.copy_context())
    buffer.push_overflow(task)
    monkeypatch.setattr(metering_buffer, "_buffer", buffer)

    with caplog.at_level(logging.WARNING, logger="revenium_middleware"):
        metering.handle_exit()

    assert not task.is_alive()
    assert buffer.stats()["size"] == 0
    assert [w.split(" metering event(s)")[0] for w in _shutdown_warnings(caplog)] == ["1"]


def _overflow_with_no_thread_to_start(monkeypatch, buffer, coroutine_or_func):
    """Overflow one event through run_async_in_thread, then leave the process unable to start a thread."""
    monkeypatch.setattr(metering_buffer, "_buffer", buffer)
    pool = MeteringWorkerPool(1, queue_size=1, overflow=metering_pool._overflow_to_buffer)
    monkeypatch.setattr(metering_pool, "_pool", pool)
    picked_up, release = threading.Event(), threading.Event()

    async def occupy_the_worker():
        picked_up.set()
        release.wait(10)

    async def fill_the_queue():
        release.wait(10)

    metering.run_async_in_thread(occupy_the_worker())
    assert picked_up.wait(5)
    queued = metering.run_async_in_thread(fill_the_queue())

    def refuse(thread):
        raise RuntimeError("can't create new thread at interpreter shutdown")

    monkeypatch.setattr(threading.Thread, "start", refuse)
    overflowed = metering.run_async_in_thread(coroutine_or_func)
    release.set()
    queued.join(5)
    assert not queued.is_alive()
    assert buffer.stats()["total_overflowed"] == 1
    return overflowed


def test_with_no_thread_to_start_shutdown_leaves_an_overflowed_synchronous_callable_unbuilt_and_counts_it(
    isolated_shutdown, monkeypatch, caplog
):
    """asyncio.wait_for cannot interrupt a synchronous callable queued through run_async_in_thread,
    so the exiting thread must not build one when no other thread can."""
    budget = 0.3
    monkeypatch.setenv(BUDGET_ENV, str(budget))
    started, release = threading.Event(), threading.Event()

    def blocks_past_the_budget():
        started.set()
        release.wait(5)

    buffer = MeteringBuffer(max_size=10, flush_interval=9999.0, replay_fn=lambda event, timeout: None)
    overflowed = _overflow_with_no_thread_to_start(monkeypatch, buffer, blocks_past_the_budget)

    try:
        drain_started = time.monotonic()
        with caplog.at_level(logging.WARNING, logger="revenium_middleware"):
            metering.handle_exit()
        elapsed = time.monotonic() - drain_started
    finally:
        release.set()
        overflowed.discard()

    assert not started.is_set()
    assert elapsed < budget + 0.3
    assert [w.split(" metering event(s)")[0] for w in _shutdown_warnings(caplog)] == ["1"]


def test_with_no_thread_to_start_shutdown_still_builds_an_overflowed_coroutine_on_the_exiting_thread(
    isolated_shutdown, monkeypatch, caplog
):
    monkeypatch.setenv(BUDGET_ENV, "0.5")
    built_on, replayed = [], []

    async def integration_style_call():
        built_on.append(threading.current_thread())
        metering_buffer.buffer_deferred_event("ai", {"seq": "overflowed"})

    buffer = MeteringBuffer(max_size=10, flush_interval=9999.0,
                            replay_fn=lambda event, timeout: replayed.append(event.payload["seq"]))
    _overflow_with_no_thread_to_start(monkeypatch, buffer, integration_style_call())

    with caplog.at_level(logging.WARNING, logger="revenium_middleware"):
        metering.handle_exit()

    assert built_on == [threading.current_thread()]
    assert replayed == ["overflowed"]
    assert _shutdown_warnings(caplog) == []


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
    buffer.push("ai", {"operation": "completion", "args": dict(COMPLETION_ARGS),
                       "api_key": client.api_key, "base_url": str(client.base_url)})
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


class _MeteringStub:
    """A local metering endpoint that records the transaction id of every completion it receives.

    Each answer takes ``delay`` seconds; ``peak_in_flight`` is the most requests it was answering at once.
    """

    def __init__(self, delay=0.0):
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

        received = self.received = []
        self.peak_in_flight = 0
        in_flight = [0]
        lock = threading.Lock()
        stub = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                with lock:
                    received.append(body["transactionId"])
                    in_flight[0] += 1
                    stub.peak_in_flight = max(stub.peak_in_flight, in_flight[0])
                try:
                    time.sleep(delay)
                finally:
                    with lock:
                        in_flight[0] -= 1
                reply = json.dumps({"id": "m", "label": "m", "resourceType": "metering", "signature": "s"}).encode()
                self.send_response(201)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(reply)))
                self.end_headers()
                self.wfile.write(reply)

            def log_message(self, *args):
                pass

        class Server(ThreadingHTTPServer):
            # socketserver's backlog of 5 drops connects beyond it on Linux, and a dropped
            # connect is retried only after about a second.
            request_queue_size = 128
            block_on_close = False

        self._server = Server(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self.base_url = f"http://127.0.0.1:{self._server.server_address[1]}/meter/"

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._server.shutdown()
        self._server.server_close()


OVERFLOW_AT_EXIT_CHILD = textwrap.dedent("""
    import atexit, sys, threading, time

    drain_started = []
    atexit.register(lambda: print("drain-seconds %.3f" % (time.monotonic() - drain_started[0])))

    from revenium_middleware._core import metering, metering_pool
    from revenium_middleware._core.metering_buffer import get_buffer_stats
    from revenium_middleware._core.metering_submission import submit_ai_event

    ARGS = {args}
    OVERFLOWED = {overflowed}

    async def occupy_the_worker():
        time.sleep(0.3)

    async def integration_style_call(n):
        if metering.shutdown_event.is_set():
            return
        submit_ai_event("completion", dict(ARGS, transaction_id="txn-%d" % n))

    def overflow():
        metering.run_async_in_thread(occupy_the_worker())
        pool = metering_pool.get_pool(metering.shutdown_event.is_set)
        while pool._queue.qsize():
            time.sleep(0.01)
        for n in range(OVERFLOWED + 1):
            metering.run_async_in_thread(integration_style_call(n))
        assert get_buffer_stats()["total_overflowed"] == OVERFLOWED, get_buffer_stats()

    def refuse_new_threads_like_python_3_12_0_to_3_12_2():
        def refuse(thread):
            raise RuntimeError("can't create new thread at interpreter shutdown")
        threading.Thread.start = refuse

    atexit.register(lambda: drain_started.append(time.monotonic()))
    if sys.argv[1] == "while-running":
        overflow()
    else:
        metering.run_async_in_thread(integration_style_call(-1)).join(5)
        atexit.register(overflow)
    atexit.register(refuse_new_threads_like_python_3_12_0_to_3_12_2)
""")


STUCK_DELIVERY_CHILD = textwrap.dedent("""
    import contextvars, time

    from revenium_middleware._core import metering
    from revenium_middleware._core.metering_buffer import get_buffer
    from revenium_middleware._core.metering_pool import MeteringTask

    async def never_finishes():
        time.sleep(3600)

    metering.run_async_in_thread(never_finishes())
    get_buffer().push_overflow(MeteringTask(never_finishes(), contextvars.copy_context()))
""")


def test_the_interpreter_exits_with_a_delivery_stuck_on_every_metering_thread():
    """Worker, build and flush threads are daemons: a stuck one cannot hold the process past the exit budget."""
    started = time.monotonic()
    completed = subprocess.run([sys.executable, "-c", STUCK_DELIVERY_CHILD], capture_output=True, text=True,
                               timeout=60, env=_child_env(REVENIUM_METERING_WORKERS="1", **{BUDGET_ENV: "0.5"}))

    assert completed.returncode == 0, completed.stderr[-2000:]
    assert time.monotonic() - started < 30


@pytest.mark.parametrize("overflowed_when", ["while-running", "during-atexit"])
def test_records_overflowed_before_exit_are_delivered_when_exit_cannot_start_a_thread(overflowed_when):
    """Python 3.12.0 to 3.12.2 refuse to start a thread from atexit, where the drain used to build overflowed records."""
    overflowed, budget = 5, 2.0
    args = {key: value for key, value in COMPLETION_ARGS.items() if key not in ("transaction_id", "extra_headers")}
    child_source = OVERFLOW_AT_EXIT_CHILD.format(args=repr(args), overflowed=overflowed)
    with _MeteringStub() as stub:
        env = _child_env(REVENIUM_METERING_BASE_URL=stub.base_url, REVENIUM_METERING_WORKERS="1",
                         REVENIUM_METERING_QUEUE_SIZE="1", **{BUDGET_ENV: str(budget)})
        completed = subprocess.run([sys.executable, "-c", child_source, overflowed_when], capture_output=True,
                                   text=True, timeout=60, env=env)

    assert completed.returncode == 0, completed.stderr[-2000:]
    first = 0 if overflowed_when == "while-running" else -1
    assert sorted(stub.received) == sorted(f"txn-{n}" for n in range(first, overflowed + 1)), completed.stderr[-2000:]
    assert "still queued" not in completed.stderr
    drain_seconds = float(completed.stdout.split("drain-seconds")[-1])
    assert drain_seconds < budget


class _RecordsWhenBuilt:
    """An overflowed task that, like an integration's coroutine, skips itself once shutdown_event is set."""

    def __init__(self, seq):
        self.seq = seq
        self.outcome = None

    def materialize(self, enqueued_at, timeout=None):
        if metering.shutdown_event.is_set():
            self.outcome = "skipped"
            return
        self.outcome = "built"
        with metering_buffer.delivery_deferred_to_buffer(enqueued_at):
            metering_buffer.buffer_deferred_event("ai", {"seq": self.seq})

    def discard(self):
        self.outcome = "discarded"


def test_shutdown_builds_a_task_a_running_flush_took_out_of_the_buffer(isolated_shutdown, monkeypatch, caplog):
    """A periodic flush moves an overflow task it meets mid-replay aside, to build after replaying;
    shutdown must take it over then, or it is built after shutdown_event, skips itself and goes uncounted."""
    monkeypatch.setenv(BUDGET_ENV, "0.5")
    replaying = {seq: threading.Event() for seq in ("first", "second")}
    release = {seq: threading.Event() for seq in ("first", "second")}

    def gated_replay(event, timeout):
        seq = event.payload["seq"]
        if seq in replaying:
            replaying[seq].set()
            release[seq].wait(10)

    buffer = MeteringBuffer(max_size=10, flush_interval=9999.0, replay_fn=gated_replay)
    monkeypatch.setattr(metering_buffer, "_buffer", buffer)
    buffer.push("ai", {"seq": "first"})
    periodic_flush = threading.Thread(target=buffer.flush, daemon=True)
    periodic_flush.start()
    assert replaying["first"].wait(5)
    late = _RecordsWhenBuilt("late")
    buffer.push_overflow(late)
    buffer.push("ai", {"seq": "second"})
    release["first"].set()
    assert replaying["second"].wait(5)

    try:
        with caplog.at_level(logging.WARNING, logger="revenium_middleware"):
            metering.handle_exit()
    finally:
        release["second"].set()
        periodic_flush.join(5)

    assert late.outcome == "built"
    assert [w.split(" metering event(s)")[0] for w in _shutdown_warnings(caplog)] == ["2"]


def test_an_overflow_after_the_exit_build_stays_buffered_and_counted(isolated_shutdown):
    buffer = MeteringBuffer(max_size=10, flush_interval=9999.0, replay_fn=lambda event, timeout: None)
    buffer.build_all_overflow(deadline_seconds=0.5)
    straggler = _RecordsWhenBuilt("straggler")
    buffer.push_overflow(straggler)
    metering.shutdown_event.set()

    result = buffer.flush(deadline_seconds=0.5)

    assert straggler.outcome is None
    assert result["remaining"] == 1


@posix_signals
def test_a_drain_interrupted_by_the_sigterm_handler_keeps_its_one_deadline(isolated_shutdown, monkeypatch):
    """The opt-in handler runs handle_exit on the main thread, which may already be in the atexit drain."""
    budget = 1.0
    monkeypatch.setenv(BUDGET_ENV, str(budget))
    release = threading.Event()
    _queue_blocked_events(monkeypatch, 1, release)
    interrupted = []

    def drain_again(signum, frame):
        interrupted.append(signum)
        metering.handle_exit()

    previous = signal.signal(signal.SIGUSR1, drain_again)
    threading.Timer(budget / 2, os.kill, (os.getpid(), signal.SIGUSR1)).start()
    try:
        started = time.monotonic()
        metering.handle_exit()
        elapsed = time.monotonic() - started
    finally:
        signal.signal(signal.SIGUSR1, previous)
        release.set()

    assert interrupted == [signal.SIGUSR1]
    assert elapsed < budget + 0.3


def test_shutdown_does_not_wait_past_its_budget_on_a_reclaimed_task_that_blocks_synchronously(
    isolated_shutdown, monkeypatch, caplog
):
    """With the build thread stuck, reclaimed tasks build on a thread of their own: a synchronous callable
    run through run_async_in_thread blocks where asyncio.wait_for cannot interrupt it."""
    budget = 0.5
    monkeypatch.setenv(BUDGET_ENV, str(budget))
    release = threading.Event()
    stuck_started = threading.Event()

    async def stuck_call():
        stuck_started.set()
        release.wait(10)

    def blocks_on_io():
        release.wait(10)

    buffer = MeteringBuffer(max_size=10, flush_interval=9999.0, replay_fn=lambda event, timeout: None)
    monkeypatch.setattr(metering_buffer, "_buffer", buffer)
    buffer.push_overflow(MeteringTask(stuck_call(), contextvars.copy_context()))
    buffer.push_overflow(MeteringTask(metering._run_sync_callable(blocks_on_io), contextvars.copy_context()))
    periodic_build = threading.Thread(target=buffer.materialize_overflow, daemon=True)
    periodic_build.start()
    assert stuck_started.wait(5)

    try:
        started = time.monotonic()
        with caplog.at_level(logging.WARNING, logger="revenium_middleware"):
            metering.handle_exit()
        elapsed = time.monotonic() - started
    finally:
        release.set()
        periodic_build.join(5)

    assert elapsed < budget + 0.3
    assert [w.split(" metering event(s)")[0] for w in _shutdown_warnings(caplog)] == ["2"]


WORKER_PROCESS_JOBS = textwrap.dedent("""
    import multiprocessing, sys, threading, time

    from revenium_middleware._core import metering
    from revenium_middleware._core.metering_submission import submit_ai_event

    ARGS = {args}

    async def integration_style_call(txn):
        if metering.shutdown_event.is_set():
            return
        submit_ai_event("completion", dict(ARGS, transaction_id=txn))

    def job(n):
        metering.run_async_in_thread(integration_style_call("txn-%d" % n))

    def job_whose_thread_meters_after_it_returns(n):
        metering.run_async_in_thread(integration_style_call("txn-early"))

        def meter_later():
            time.sleep(0.3)
            metering.run_async_in_thread(integration_style_call("txn-late"))

        threading.Thread(target=meter_later).start()

    def job_that_reports_when_it_returns(n, returned_at):
        job(n)
        returned_at.put(time.time())

    if __name__ == "__main__":
        mode, start_method, jobs = sys.argv[1], sys.argv[2], int(sys.argv[3])
        if mode == "warm-up":
            metering.run_async_in_thread(integration_style_call("txn-parent")).join(10)
        context = multiprocessing.get_context(start_method)
        returned_at = context.Queue()
        if mode == "timed":
            processes = [context.Process(target=job_that_reports_when_it_returns, args=(n, returned_at))
                         for n in range(jobs)]
        else:
            target = job_whose_thread_meters_after_it_returns if mode == "late-thread" else job
            processes = [context.Process(target=target, args=(n,)) for n in range(jobs)]
        for process in processes:
            process.start()
        for process in processes:
            process.join(20)
            if process.is_alive():
                process.kill()
                process.join()
        if mode == "timed":
            print("exit-seconds %.3f" % (time.time() - returned_at.get(timeout=5)))
        print("exit-codes", " ".join(str(process.exitcode) for process in processes))
""")


# A metering client built in a forked child on macOS reads the system proxy settings through
# Apple's SystemConfiguration, which can crash a child forked from a threaded parent; a proxy
# variable in the environment, as here for the local stub, makes the client read that instead.
FORK_SAFE_PROXY_ENV = {"NO_PROXY": "127.0.0.1"}


def _run_worker_process_jobs(tmp_path, stub, mode, start_method, jobs, **env):
    args = {key: value for key, value in COMPLETION_ARGS.items() if key not in ("transaction_id", "extra_headers")}
    script = tmp_path / "worker_process_jobs.py"
    script.write_text(WORKER_PROCESS_JOBS.format(args=repr(args)))
    return subprocess.run([sys.executable, str(script), mode, start_method, str(jobs)], capture_output=True,
                          text=True, timeout=60,
                          env=_child_env(REVENIUM_METERING_BASE_URL=stub.base_url, **FORK_SAFE_PROXY_ENV, **env))


def _start_method(name):
    import multiprocessing

    return pytest.param(name, marks=pytest.mark.skipif(
        name not in multiprocessing.get_all_start_methods(), reason=f"the {name} start method is not available"))


@pytest.mark.parametrize("start_method", [_start_method("fork"), _start_method("forkserver")])
def test_a_job_run_in_its_own_worker_process_delivers_its_usage_records_when_it_returns(tmp_path, start_method):
    """BACK-4353: multiprocessing ends a fork or forkserver child with os._exit, which skips atexit,
    so a job's records died with the daemon workers that were about to send them."""
    import collections

    jobs = 5
    with _MeteringStub() as stub:
        completed = _run_worker_process_jobs(tmp_path, stub, "warm-up", start_method, jobs)

    assert completed.returncode == 0, completed.stderr[-2000:]
    assert completed.stdout.split("exit-codes")[-1].split() == ["0"] * jobs, completed.stderr[-2000:]
    expected = ["txn-parent"] + [f"txn-{n}" for n in range(jobs)]
    assert collections.Counter(stub.received) == collections.Counter(expected), completed.stderr[-2000:]


@pytest.mark.parametrize("start_method", [_start_method("fork"), _start_method("forkserver")])
def test_a_worker_process_that_ends_with_records_unsent_logs_a_warning_with_the_count(tmp_path, start_method):
    """BACK-4353: a worker process that ended with a record still unsent lost it without a word."""
    with _MeteringStub(delay=2.0) as stub:
        completed = _run_worker_process_jobs(tmp_path, stub, "plain", start_method, 1, **{BUDGET_ENV: "0.5"})

    assert completed.returncode == 0, completed.stderr[-2000:]
    assert "1 metering event(s) still queued" in completed.stderr, completed.stderr[-2000:]


@pytest.mark.parametrize("start_method", [_start_method("spawn")])
def test_a_spawned_job_whose_thread_meters_after_the_target_returns_still_delivers(tmp_path, start_method):
    """Guard for BACK-4353, passes before and after the fix: the flush that runs when a job's target returns
    must not end metering, because a spawned child's non-daemon threads still run, and meter, after it."""
    import collections

    with _MeteringStub() as stub:
        completed = _run_worker_process_jobs(tmp_path, stub, "late-thread", start_method, 1)

    assert completed.returncode == 0, completed.stderr[-2000:]
    assert collections.Counter(stub.received) == collections.Counter(["txn-early", "txn-late"]), \
        completed.stderr[-2000:]


@pytest.mark.parametrize("start_method", [_start_method("spawn"), _start_method("fork"), _start_method("forkserver")])
def test_a_worker_process_that_ends_with_revenium_slow_exits_within_one_budget_and_warns_once(tmp_path, start_method):
    """BACK-4353: a spawned child ran the flush when its job returned and then the interpreter's exit drain,
    each with a full budget, so it took twice the budget to exit and logged the unsent record twice."""
    budget = 2.0
    with _MeteringStub(delay=3 * budget) as stub:
        completed = _run_worker_process_jobs(tmp_path, stub, "timed", start_method, 1, **{BUDGET_ENV: str(budget)})

    assert completed.returncode == 0, completed.stderr[-2000:]
    assert float(completed.stdout.split("exit-seconds")[-1].split()[0]) < budget + 1, completed.stdout
    assert completed.stderr.count("1 metering event(s) still queued") == 1, completed.stderr[-2000:]


def test_flush_metering_sends_what_is_queued_and_metering_continues_afterwards(isolated_shutdown, monkeypatch):
    import revenium_middleware

    pool = MeteringWorkerPool(1, queue_size=100, overflow=lambda task: task.discard())
    monkeypatch.setattr(metering_pool, "_pool", pool)
    delivered = []

    async def slow_delivery(n):
        await asyncio.sleep(0.1)
        delivered.append(n)

    for n in range(3):
        metering.run_async_in_thread(slow_delivery(n))

    assert revenium_middleware.flush_metering(2) == 0
    assert delivered == [0, 1, 2]
    assert not metering.shutdown_event.is_set()
    later = metering.run_async_in_thread(slow_delivery(3))
    assert later is not None
    later.join(5)
    assert delivered == [0, 1, 2, 3]


def test_flush_metering_returns_how_many_records_are_left_when_its_timeout_passes(isolated_shutdown, monkeypatch):
    import revenium_middleware

    release = threading.Event()
    _queue_blocked_events(monkeypatch, 1, release, workers=1)
    try:
        started = time.monotonic()
        left = revenium_middleware.flush_metering(0.2)
        elapsed = time.monotonic() - started
    finally:
        release.set()

    assert left == 1
    assert elapsed < 0.5


def test_flush_metering_counts_an_overflowed_record_its_timeout_cut_short_with_no_thread_to_start(
    isolated_shutdown, monkeypatch
):
    """BACK-4353: an overflowed record whose build the timeout cut short was in neither count, so
    flush_metering() returned 0 and a forked child's exit flush logged no warning."""
    import revenium_middleware

    async def outlasts_the_timeout():
        await asyncio.sleep(5)

    buffer = MeteringBuffer(max_size=10, flush_interval=9999.0, replay_fn=lambda event, timeout: None)
    _overflow_with_no_thread_to_start(monkeypatch, buffer, outlasts_the_timeout())

    started = time.monotonic()
    left = revenium_middleware.flush_metering(0.2)

    assert left >= 1
    assert time.monotonic() - started < 0.6


@pytest.mark.parametrize("timeout", [-1, float("nan"), float("inf")])
def test_flush_metering_rejects_a_timeout_that_is_negative_or_not_finite(timeout):
    import revenium_middleware

    with pytest.raises(ValueError):
        revenium_middleware.flush_metering(timeout)


def test_flush_metering_after_the_exit_drain_has_nothing_left_to_do(isolated_shutdown):
    import revenium_middleware

    metering.shutdown_event.set()

    assert revenium_middleware.flush_metering(0.1) == 0


HELD_AT_EXIT_CHILD = textwrap.dedent("""
    import atexit, sys, threading, time

    drain_started = []
    atexit.register(lambda: print("drain-seconds %.3f" % (time.monotonic() - drain_started[0])))

    from revenium_middleware._core import metering
    from revenium_middleware._core.metering_buffer import get_buffer

    ARGS = {args}
    held, new_threads = int(sys.argv[1]), sys.argv[2]
    client = metering.process_client()
    for n in range(held):
        txn = "txn-%d" % n
        get_buffer().push("ai", dict(operation="completion", api_key=client.api_key, base_url=str(client.base_url),
                                     args=dict(ARGS, transaction_id=txn, extra_headers={{"Idempotency-Key": txn}})))

    def refuse_new_threads_like_python_3_12_0_to_3_12_2():
        def refuse(thread):
            raise RuntimeError("can't create new thread at interpreter shutdown")
        threading.Thread.start = refuse

    atexit.register(lambda: drain_started.append(time.monotonic()))
    if new_threads == "refused":
        atexit.register(refuse_new_threads_like_python_3_12_0_to_3_12_2)
""")


def _exit_holding_records(stub, held, new_threads, budget):
    args = {key: value for key, value in COMPLETION_ARGS.items() if key not in ("transaction_id", "extra_headers")}
    env = _child_env(REVENIUM_METERING_BASE_URL=stub.base_url, REVENIUM_BUFFER_FLUSH_INTERVAL="9999",
                     **{BUDGET_ENV: str(budget)})
    completed = subprocess.run([sys.executable, "-c", HELD_AT_EXIT_CHILD.format(args=repr(args)), str(held), new_threads],
                               capture_output=True, text=True, timeout=60, env=env)
    assert completed.returncode == 0, completed.stderr[-2000:]
    return completed, float(completed.stdout.split("drain-seconds")[-1])


def test_an_exit_soon_after_an_outage_delivers_its_held_records_within_the_budget():
    """BACK-4355: the exit flush sent held records one at a time, so at a quarter-second round trip
    a 5 second budget delivered about 20 of them and dropped the rest."""
    held, budget = 200, metering.DEFAULT_SHUTDOWN_TIMEOUT_SECONDS
    with _MeteringStub(delay=0.25) as stub:
        completed, drain_seconds = _exit_holding_records(stub, held, "allowed", budget)

    assert sorted(stub.received) == sorted(f"txn-{n}" for n in range(held)), completed.stderr[-2000:]
    assert stub.peak_in_flight > 1
    assert "still queued" not in completed.stderr
    assert drain_seconds < budget + 1


def test_an_exit_that_cannot_start_threads_still_ends_within_its_budget():
    """Guard for BACK-4355, passes before and after the fix: Python 3.12.0 to 3.12.2 refuse to start a thread
    inside atexit, so a wide exit batch replays inline, and must still stop at the budget and count what is left."""
    held, budget = 64, 2.0
    with _MeteringStub(delay=0.25) as stub:
        completed, drain_seconds = _exit_holding_records(stub, held, "refused", budget)

    assert drain_seconds < budget + 1
    warnings = [line.split(" metering event(s) still queued")[0].split()[-1]
                for line in completed.stderr.splitlines() if "metering event(s) still queued" in line]
    assert len(warnings) == 1, completed.stderr[-2000:]
    # The record cut off by the deadline may have reached the stub before the SDK gave up on its answer.
    assert held - len(stub.received) <= int(warnings[0]) <= held - len(stub.received) + 1


def test_a_background_flush_running_at_exit_hands_over_to_the_exit_flush(isolated_shutdown, monkeypatch, caplog):
    """BACK-4355: a background flush replaying when the exit drain starts holds the flush lock; it must stop
    after its batch so the exit flush replays the rest at its own width within the budget."""
    budget = 3.0
    monkeypatch.setenv(BUDGET_ENV, str(budget))
    replayed, in_flight, peak = [], [0], [0]
    lock, replaying = threading.Lock(), threading.Event()

    def slow_replay(event, timeout):
        with lock:
            in_flight[0] += 1
            peak[0] = max(peak[0], in_flight[0])
        replaying.set()
        time.sleep(0.2)
        with lock:
            in_flight[0] -= 1
            replayed.append(event.payload["seq"])

    buffer = MeteringBuffer(max_size=100, flush_interval=9999.0, replay_fn=slow_replay, replay_concurrency=1,
                            exit_replay_concurrency=16)
    monkeypatch.setattr(metering_buffer, "_buffer", buffer)
    for seq in range(40):
        buffer.push("ai", {"seq": seq})
    background = threading.Thread(target=buffer.flush, daemon=True)
    background.start()
    assert replaying.wait(5)

    started = time.monotonic()
    with caplog.at_level(logging.WARNING, logger="revenium_middleware"):
        metering.handle_exit()
    elapsed = time.monotonic() - started
    background.join(5)

    assert sorted(replayed) == list(range(40))
    assert peak[0] > 1
    assert _shutdown_warnings(caplog) == []
    assert elapsed < budget
