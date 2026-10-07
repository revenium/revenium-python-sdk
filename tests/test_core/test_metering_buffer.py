"""Store-and-forward buffer: failed metering events are buffered and replayed.

Covers the buffer unit contract (FIFO, bounds, TTL, stop-on-retryable,
discard-on-permanent), the retryability classification, and the integration
points: submit_ai_event, tool-event dispatch, and shutdown drain.
"""
import datetime
import socket
import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import httpx
import pytest

from revenium_middleware._core import metering, metering_buffer
from revenium_middleware._core.metering_buffer import (
    BufferedEvent,
    MeteringBuffer,
    is_retryable_failure,
)
from revenium_middleware._core.metering_status import (
    get_metering_status,
    on_metering_error,
    reset_metering_status,
)
from revenium_middleware._metering._exceptions import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
)


def make_status_error(status_code, headers=None):
    request = httpx.Request("POST", "https://api.test/meter/v2/ai/completions")
    response = httpx.Response(status_code, headers=headers or {}, request=request)
    return APIStatusError("boom", response=response, body=None)


def make_buffer(**overrides):
    defaults = dict(max_size=5, flush_interval=9999.0)
    defaults.update(overrides)
    return MeteringBuffer(**defaults)


class RecordingReplayer:
    def __init__(self, failures=None):
        self.calls = []
        self.timeouts = []
        self.failures = dict(failures or {})  # index -> exception

    def __call__(self, event, timeout_seconds):
        index = len(self.calls)
        self.calls.append(event)
        self.timeouts.append(timeout_seconds)
        if index in self.failures:
            raise self.failures[index]


def make_failing_async_client(status_code):
    """Async-client double whose post() always fails with ``status_code``."""
    request = httpx.Request("POST", "https://api.test/meter/v2/tool/events")

    class FailingAsyncClient:
        def __init__(self, timeout=None):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def post(self, url, headers=None, json=None):
            response = httpx.Response(status_code, request=request)
            raise httpx.HTTPStatusError(str(status_code), request=request, response=response)

    return FailingAsyncClient


class TestRetryabilityClassification:
    @pytest.mark.parametrize("status", [408, 429, 500, 503])
    def test_retryable_statuses(self, status):
        assert is_retryable_failure(make_status_error(status)) is True

    @pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
    def test_permanent_statuses(self, status):
        assert is_retryable_failure(make_status_error(status)) is False

    def test_409_without_retry_after_is_permanent(self):
        # idempotency_key_mismatch: same key, different body -- a retry can
        # never resolve it, so it must not circle in the buffer until TTL.
        assert is_retryable_failure(make_status_error(409)) is False

    def test_409_with_retry_after_is_retryable(self):
        # idempotency_key_in_progress: backend signals Retry-After: 1.
        assert is_retryable_failure(
            make_status_error(409, {"Retry-After": "1"})) is True

    def test_retry_after_header_makes_any_status_retryable(self):
        assert is_retryable_failure(
            make_status_error(422, {"Retry-After": "2"})) is True

    def test_x_should_retry_header_wins(self):
        assert is_retryable_failure(make_status_error(400, {"x-should-retry": "true"})) is True
        assert is_retryable_failure(make_status_error(500, {"x-should-retry": "false"})) is False

    def test_connection_errors_are_retryable(self):
        request = httpx.Request("POST", "https://api.test")
        assert is_retryable_failure(APIConnectionError(request=request)) is True
        assert is_retryable_failure(APITimeoutError(request=request)) is True

    def test_httpx_transport_errors_are_retryable(self):
        assert is_retryable_failure(httpx.ConnectError("refused")) is True
        assert is_retryable_failure(httpx.ReadTimeout("slow")) is True

    def test_httpx_status_errors_follow_status_rules(self):
        request = httpx.Request("POST", "https://api.test/meter/v2/tool/events")
        retryable = httpx.HTTPStatusError(
            "503", request=request, response=httpx.Response(503, request=request))
        permanent = httpx.HTTPStatusError(
            "422", request=request, response=httpx.Response(422, request=request))
        assert is_retryable_failure(retryable) is True
        assert is_retryable_failure(permanent) is False

    def test_unknown_exceptions_are_not_buffered(self):
        assert is_retryable_failure(ValueError("bug")) is False


class TestBufferContract:
    def test_flush_replays_oldest_first_and_drains(self):
        replayer = RecordingReplayer()
        buf = make_buffer(replay_fn=replayer)
        for i in range(3):
            buf.push("ai", {"seq": i})

        buf.flush()

        assert [e.payload["seq"] for e in replayer.calls] == [0, 1, 2]
        assert buf.stats()["size"] == 0
        assert buf.stats()["total_replayed"] == 3

    def test_flush_stops_on_first_retryable_failure(self):
        replayer = RecordingReplayer(failures={1: make_status_error(503)})
        buf = make_buffer(replay_fn=replayer)
        for i in range(3):
            buf.push("ai", {"seq": i})

        buf.flush()

        # 0 delivered; 1 failed retryably -> kept; 2 never attempted.
        assert len(replayer.calls) == 2
        assert buf.stats()["size"] == 2

    def test_permanent_failure_during_replay_discards_and_continues(self):
        replayer = RecordingReplayer(failures={1: make_status_error(422)})
        buf = make_buffer(replay_fn=replayer)
        for i in range(3):
            buf.push("ai", {"seq": i})

        buf.flush()

        assert [e.payload["seq"] for e in replayer.calls] == [0, 1, 2]
        assert buf.stats()["size"] == 0
        assert buf.stats()["total_discarded"] == 1
        assert buf.stats()["total_replayed"] == 2

    def test_fifo_eviction_at_max_size(self, caplog):
        buf = make_buffer(max_size=3, replay_fn=RecordingReplayer())
        for i in range(4):
            buf.push("ai", {"seq": i})

        stats = buf.stats()
        assert stats["size"] == 3
        assert stats["total_evicted"] == 1
        assert "buffer" in caplog.text.lower()
        # Oldest (seq 0) was evicted.
        replayer = RecordingReplayer()
        buf._replay_fn = replayer
        buf.flush()
        assert [e.payload["seq"] for e in replayer.calls] == [1, 2, 3]

    def test_events_older_than_max_age_expire_during_flush(self):
        clock = {"now": 1_000_000.0}
        replayer = RecordingReplayer()
        buf = make_buffer(replay_fn=replayer, now_fn=lambda: clock["now"], max_age_seconds=3600)
        buf.push("ai", {"seq": "old"})
        clock["now"] += 3601
        buf.push("ai", {"seq": "fresh"})

        buf.flush()

        assert [e.payload["seq"] for e in replayer.calls] == ["fresh"]
        assert buf.stats()["total_expired"] == 1

    def test_flush_respects_deadline(self):
        slow_calls = []

        def slow_replayer(event, timeout_seconds):
            slow_calls.append(event)
            time.sleep(0.2)

        buf = make_buffer(max_size=100, replay_fn=slow_replayer)
        for i in range(10):
            buf.push("ai", {"seq": i})

        buf.flush(deadline_seconds=0.3)

        assert 0 < len(slow_calls) < 10
        assert buf.stats()["size"] == 10 - len(slow_calls)

    def test_deadline_shrinks_per_call_replay_timeout(self):
        """A 10s network timeout must not blow a smaller flush deadline."""
        replayer = RecordingReplayer()
        buf = make_buffer(replay_fn=replayer)
        buf.push("ai", {"seq": 0})

        buf.flush(deadline_seconds=2.0)

        assert len(replayer.timeouts) == 1
        assert replayer.timeouts[0] <= 2.0
        assert replayer.timeouts[0] >= 0.5

    def test_no_deadline_uses_full_replay_timeout(self):
        replayer = RecordingReplayer()
        buf = make_buffer(replay_fn=replayer)
        buf.push("ai", {"seq": 0})

        buf.flush()

        assert replayer.timeouts == [metering.DEFAULT_TIMEOUT_SECONDS]

    def test_a_replay_waits_as_long_as_the_metering_timeout_setting(self, monkeypatch):
        monkeypatch.setenv(metering.TIMEOUT_ENV, "3")
        replayer = RecordingReplayer()
        buf = make_buffer(replay_fn=replayer)
        buf.push("ai", {"seq": 0})
        buf.push("ai", {"seq": 1})

        buf.flush()
        buf.push("ai", {"seq": 2})
        buf.flush(deadline_seconds=30.0)

        assert replayer.timeouts == [3.0, 3.0, 3.0]

    def test_stats_shape(self):
        buf = make_buffer(replay_fn=RecordingReplayer())
        stats = buf.stats()
        for field in ("size", "max_size", "total_buffered", "total_replayed",
                      "total_evicted", "total_expired", "total_discarded"):
            assert field in stats

    def test_push_is_thread_safe(self):
        buf = make_buffer(max_size=10000, replay_fn=RecordingReplayer())

        def hammer():
            for i in range(200):
                buf.push("ai", {"seq": i})

        threads = [threading.Thread(target=hammer) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert buf.stats()["size"] == 1000
        assert buf.stats()["total_buffered"] == 1000


class TestModuleSingleton:
    def test_get_buffer_stats_exported_publicly(self):
        import revenium_middleware
        assert "get_buffer_stats" in revenium_middleware.__all__
        assert callable(revenium_middleware.get_buffer_stats)

    def test_env_config_honored(self, monkeypatch):
        monkeypatch.setenv("REVENIUM_BUFFER_MAX_SIZE", "7")
        monkeypatch.setenv("REVENIUM_BUFFER_FLUSH_INTERVAL", "1.5")
        monkeypatch.setattr(metering_buffer, "_buffer", None)

        buf = metering_buffer.get_buffer()

        assert buf.stats()["max_size"] == 7
        assert buf._flush_interval == 1.5

    @pytest.mark.parametrize("raw, expected", [("500", 500), ("500.0", 500), ("1e3", 1000)])
    def test_max_size_accepts_any_whole_number_spelling(self, monkeypatch, raw, expected):
        monkeypatch.setenv("REVENIUM_BUFFER_MAX_SIZE", raw)
        monkeypatch.setattr(metering_buffer, "_buffer", None)

        assert metering_buffer.get_buffer().stats()["max_size"] == expected

    @pytest.mark.parametrize("raw", ["0", "-5", "inf", "nan"])
    def test_max_size_below_one_or_not_finite_uses_the_default(self, monkeypatch, caplog, raw):
        monkeypatch.setenv("REVENIUM_BUFFER_MAX_SIZE", raw)
        monkeypatch.setattr(metering_buffer, "_buffer", None)

        with caplog.at_level("WARNING"):
            buf = metering_buffer.get_buffer()

        assert buf.stats()["max_size"] == metering_buffer.DEFAULT_MAX_SIZE
        assert "REVENIUM_BUFFER_MAX_SIZE" in caplog.text

    @pytest.mark.parametrize("raw", ["0", "-1", "nan", "inf", "1e300"])
    def test_flush_interval_that_would_spin_or_stop_the_flush_thread_uses_the_default(self, monkeypatch, caplog, raw):
        """0, negative and nan made the flush loop spin and replay without pause; inf and
        values past threading.TIMEOUT_MAX made Event.wait raise, ending the flush thread."""
        monkeypatch.setenv("REVENIUM_BUFFER_FLUSH_INTERVAL", raw)
        monkeypatch.setattr(metering_buffer, "_buffer", None)

        with caplog.at_level("WARNING"):
            buf = metering_buffer.get_buffer()

        assert buf._flush_interval == metering_buffer.DEFAULT_FLUSH_INTERVAL
        assert f"Invalid REVENIUM_BUFFER_FLUSH_INTERVAL={raw!r}" in caplog.text

    @pytest.mark.parametrize("raw", ["", "lots"])
    def test_empty_or_malformed_env_uses_defaults(self, monkeypatch, caplog, raw):
        monkeypatch.setenv("REVENIUM_BUFFER_MAX_SIZE", raw)
        monkeypatch.setenv("REVENIUM_BUFFER_FLUSH_INTERVAL", raw)
        monkeypatch.setattr(metering_buffer, "_buffer", None)

        with caplog.at_level("WARNING"):
            buf = metering_buffer.get_buffer()

        assert buf.stats()["max_size"] == metering_buffer.DEFAULT_MAX_SIZE
        assert buf._flush_interval == metering_buffer.DEFAULT_FLUSH_INTERVAL
        assert ("REVENIUM_BUFFER_MAX_SIZE" in caplog.text) == bool(raw)


@pytest.fixture()
def fresh_buffer(monkeypatch):
    """Give integration tests an isolated singleton with replay disabled."""
    buf = MeteringBuffer(max_size=100, flush_interval=9999.0,
                         replay_fn=RecordingReplayer())
    monkeypatch.setattr(metering_buffer, "_buffer", buf)
    return buf


class TestSubmitAiEventIntegration:
    def _client_raising(self, exc):
        client = MagicMock()
        client.ai.create_completion.side_effect = exc
        return client

    def test_retryable_failure_is_buffered_with_original_key(self, fresh_buffer, monkeypatch):
        from revenium_middleware._core import metering_submission
        client = self._client_raising(make_status_error(503))
        monkeypatch.setattr(metering_submission, "get_client", lambda: client)

        result = metering_submission.submit_ai_event(
            "completion", {"model": "gpt-test"}, idempotency_key="order-42")

        assert result is None  # not delivered now, but not lost
        assert fresh_buffer.stats()["size"] == 1
        event = fresh_buffer._events[0]
        assert event.kind == "ai"
        assert event.payload["operation"] == "completion"
        assert event.payload["args"]["extra_headers"]["Idempotency-Key"] == "order-42"

    def test_contextvar_key_is_frozen_into_buffered_event(self, fresh_buffer, monkeypatch):
        from revenium_middleware import idempotency_key
        from revenium_middleware._core import metering_submission
        client = self._client_raising(APIConnectionError(
            request=httpx.Request("POST", "https://api.test")))
        monkeypatch.setattr(metering_submission, "get_client", lambda: client)

        with idempotency_key("ctx-key-99"):
            metering_submission.submit_ai_event("completion", {"model": "gpt-test"})

        event = fresh_buffer._events[0]
        assert event.payload["args"]["extra_headers"]["Idempotency-Key"] == "ctx-key-99"

    def test_permanent_failure_is_not_buffered_and_raises(self, fresh_buffer, monkeypatch):
        from revenium_middleware._core import metering_submission
        client = self._client_raising(make_status_error(422))
        monkeypatch.setattr(metering_submission, "get_client", lambda: client)

        with pytest.raises(APIStatusError):
            metering_submission.submit_ai_event("completion", {"model": "gpt-test"})

        assert fresh_buffer.stats()["size"] == 0

    def test_replay_reuses_original_idempotency_key(self, monkeypatch):
        """Card scenario: 503 -> buffered -> backend restored -> replayed, same key."""
        from revenium_middleware._core import metering_submission
        buf = MeteringBuffer(max_size=10, flush_interval=9999.0)  # real replayer
        monkeypatch.setattr(metering_buffer, "_buffer", buf)

        failing = self._client_raising(make_status_error(503))
        monkeypatch.setattr(metering_submission, "get_client", lambda: failing)
        metering_submission.submit_ai_event(
            "completion", {"model": "gpt-test"}, idempotency_key="replay-me")
        assert buf.stats()["size"] == 1

        healthy = MagicMock()
        monkeypatch.setattr(
            "revenium_middleware._core.metering.get_client", lambda: healthy)
        buf.flush()

        assert buf.stats()["size"] == 0
        healthy.with_options.assert_called_once_with(max_retries=0)
        call = healthy.with_options.return_value.ai.create_completion.call_args
        assert call.kwargs["extra_headers"]["Idempotency-Key"] == "replay-me"
        assert call.kwargs["model"] == "gpt-test"


class TestToolEventIntegration:
    def test_retryable_tool_failure_is_buffered(self, fresh_buffer, monkeypatch):
        import asyncio
        from revenium_middleware._metering import decorator as tool_metering
        from revenium_middleware._metering.context import get_context

        monkeypatch.setattr(tool_metering, "httpx",
                            SimpleNamespace(AsyncClient=make_failing_async_client(503)))

        asyncio.run(tool_metering._send_tool_event_async(
            "https://api.test/meter/v2/tool/events", "hak_k",
            tool_id="buffered-tool", operation="run", duration_ms=5,
            success=True, error_message=None, usage_metadata=None,
            context=get_context(), occurred_at=datetime.datetime.now(datetime.timezone.utc)))

        assert fresh_buffer.stats()["size"] == 1
        event = fresh_buffer._events[0]
        assert event.kind == "tool"
        assert event.payload["event_payload"]["toolId"] == "buffered-tool"

    def test_permanent_tool_failure_is_not_buffered(self, fresh_buffer, monkeypatch):
        import asyncio
        from revenium_middleware._metering import decorator as tool_metering
        from revenium_middleware._metering.context import get_context

        monkeypatch.setattr(tool_metering, "httpx",
                            SimpleNamespace(AsyncClient=make_failing_async_client(422)))

        asyncio.run(tool_metering._send_tool_event_async(
            "https://api.test/meter/v2/tool/events", "hak_k",
            tool_id="poison-tool", operation="run", duration_ms=5,
            success=True, error_message=None, usage_metadata=None,
            context=get_context(), occurred_at=datetime.datetime.now(datetime.timezone.utc)))

        assert fresh_buffer.stats()["size"] == 0


class TestShutdownIntegration:
    def test_handle_exit_drains_buffer_before_joining_threads(self, monkeypatch):
        from revenium_middleware._core import metering

        # Quiesce straggler metering threads left over from earlier tests
        # before installing the buffer under test: the drain deliberately
        # runs before the thread join, so a late push from a leftover
        # thread would otherwise land in this test's buffer mid-drain.
        monkeypatch.setattr(metering_buffer, "_buffer", None)
        metering.handle_exit()
        metering.shutdown_event.clear()

        replayer = RecordingReplayer()
        buf = MeteringBuffer(max_size=10, flush_interval=9999.0, replay_fn=replayer)
        buf.push("ai", {"seq": "pending"})
        monkeypatch.setattr(metering_buffer, "_buffer", buf)

        try:
            metering.handle_exit()
            assert [e.payload["seq"] for e in replayer.calls] == ["pending"]
            assert buf.stats()["size"] == 0
        finally:
            metering.shutdown_event.clear()


class TestToolReplayPath:
    """Drive flush() through the real tool replayer (not a test double)."""

    def _buffered_tool_buffer(self):
        buf = MeteringBuffer(max_size=10, flush_interval=9999.0)  # real replayer
        buf.push("tool", {
            "url": "https://frozen.example/meter/v2/tool/events",
            "key": "hak_frozen",
            "event_payload": {"transactionId": "txn-replay-1", "toolId": "t"},
        })
        return buf

    def _capture_httpx(self, monkeypatch):
        calls = []

        class Client:
            def __init__(self, timeout=None):
                calls.append({"timeout": timeout})

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def post(self, url, headers=None, json=None):
                calls[-1].update(url=url, headers=headers, json=json)
                return SimpleNamespace(raise_for_status=lambda: None)

        monkeypatch.setattr(metering_buffer, "httpx",
                            SimpleNamespace(Client=Client,
                                            HTTPStatusError=httpx.HTTPStatusError,
                                            TimeoutException=httpx.TimeoutException,
                                            TransportError=httpx.TransportError,
                                            ConnectError=httpx.ConnectError))
        return calls

    def test_replay_uses_current_credentials_and_idempotency_key(self, monkeypatch):
        buf = self._buffered_tool_buffer()
        calls = self._capture_httpx(monkeypatch)
        monkeypatch.setattr(
            "revenium_middleware._metering.decorator._resolve_endpoint",
            lambda: ("https://current.example/meter/v2/tool/events", "hak_rotated"))

        buf.flush()

        assert buf.stats()["size"] == 0
        call = calls[0]
        assert call["url"] == "https://current.example/meter/v2/tool/events"
        assert call["headers"]["x-api-key"] == "hak_rotated"
        assert call["headers"]["Idempotency-Key"] == "txn-replay-1"

    def test_replay_falls_back_to_frozen_endpoint(self, monkeypatch):
        buf = self._buffered_tool_buffer()
        calls = self._capture_httpx(monkeypatch)
        monkeypatch.setattr(
            "revenium_middleware._metering.decorator._resolve_endpoint",
            lambda: (None, None))

        buf.flush()

        call = calls[0]
        assert call["url"] == "https://frozen.example/meter/v2/tool/events"
        assert call["headers"]["x-api-key"] == "hak_frozen"


@pytest.fixture()
def clean_metering_status():
    """Isolate the global metering status counters and callbacks."""
    reset_metering_status()
    yield
    reset_metering_status()


class TestMeteringStatusIntegration:
    """flush() must feed the metering status counters and error subscribers."""

    def test_replay_success_records_metering_success(self, clean_metering_status):
        buf = make_buffer(replay_fn=RecordingReplayer())
        buf.push("ai", {"seq": 0})

        buf.flush()

        assert get_metering_status().success_count == 1

    def test_permanent_discard_records_metering_error(self, clean_metering_status):
        exc = make_status_error(422)
        buf = make_buffer(replay_fn=RecordingReplayer(failures={0: exc}))
        buf.push("tool", {"seq": 0})
        received = []
        on_metering_error(received.append)

        buf.flush()

        status = get_metering_status()
        assert status.error_count == 1
        assert status.last_error is exc
        assert len(received) == 1
        assert received[0].operation == "tool"
        assert received[0].error is exc

    def test_expired_event_records_metering_error(self, clean_metering_status):
        clock = {"now": 1_000_000.0}
        buf = make_buffer(replay_fn=RecordingReplayer(),
                          now_fn=lambda: clock["now"], max_age_seconds=3600)
        buf.push("ai", {"seq": "old"})
        clock["now"] += 3601
        received = []
        on_metering_error(received.append)

        buf.flush()

        assert get_metering_status().error_count == 1
        assert len(received) == 1
        assert received[0].operation == "ai"

    def test_error_callback_may_touch_buffer_without_deadlock(self, clean_metering_status):
        # Subscriber callbacks run synchronously from flush(); recording
        # status while _flush_lock is held would self-deadlock any callback
        # that calls back into the buffer.
        exc = make_status_error(422)
        buf = make_buffer(replay_fn=RecordingReplayer(failures={0: exc}))
        buf.push("ai", {"seq": 0})
        reentered = []
        on_metering_error(lambda event: reentered.append(buf.flush(deadline_seconds=0.05)))

        t = threading.Thread(target=buf.flush, daemon=True)
        t.start()
        t.join(timeout=5.0)

        assert not t.is_alive(), "flush() deadlocked when an error callback re-entered the buffer"
        assert len(reentered) == 1

    def test_retryable_flush_failure_records_nothing(self, clean_metering_status):
        # The event stays buffered for the next cycle: neither a success nor
        # a terminal error, so counters must not move.
        buf = make_buffer(replay_fn=RecordingReplayer(
            failures={0: make_status_error(503)}))
        buf.push("ai", {"seq": 0})

        buf.flush()

        status = get_metering_status()
        assert status.success_count == 0
        assert status.error_count == 0


def test_deadline_bounds_the_wait_for_a_flush_already_in_progress():
    replayer = RecordingReplayer()
    buf = MeteringBuffer(max_size=10, flush_interval=9999.0, replay_fn=replayer)
    buf.push("ai", {"seq": 0})
    buf._flush_lock.acquire()

    try:
        started = time.monotonic()
        result = buf.flush(deadline_seconds=0.2)
        elapsed = time.monotonic() - started
    finally:
        buf._flush_lock.release()

    assert 0.15 <= elapsed < 0.7
    assert result == {"sent": 0, "expired": 0, "discarded": 0, "remaining": 1}
    assert replayer.calls == []


class _SlowOverflowTask:
    def __init__(self, seconds):
        self._seconds = seconds

    def materialize(self, enqueued_at, timeout=None):
        time.sleep(self._seconds)

    def discard(self):
        pass


@pytest.mark.parametrize("build_seconds, deadline, low, high", [
    # The build spends 0.3s, so the lock wait gets the remaining 0.1s, not another 0.4s.
    (0.3, 0.4, 0.35, 0.6),
    # The build outlasts the whole deadline, so the held lock is not waited on at all.
    (1.0, 0.2, 0.15, 0.45),
])
def test_lock_wait_after_a_slow_overflow_build_uses_only_the_remaining_budget(build_seconds, deadline, low, high):
    replayer = RecordingReplayer()
    buf = MeteringBuffer(max_size=10, flush_interval=9999.0, replay_fn=replayer)
    buf.push_overflow(_SlowOverflowTask(build_seconds))
    buf.push("ai", {"seq": 0})
    buf._flush_lock.acquire()

    try:
        started = time.monotonic()
        result = buf.flush(deadline_seconds=deadline)
        elapsed = time.monotonic() - started
    finally:
        buf._flush_lock.release()

    assert low <= elapsed < high
    assert result["sent"] == 0
    assert replayer.calls == []


def test_tiny_deadline_strictly_bounds_per_call_timeout():
    received = []

    def replayer(event, timeout_seconds):
        received.append(timeout_seconds)

    buf = MeteringBuffer(max_size=10, flush_interval=9999.0, replay_fn=replayer)
    buf.push("ai", {"seq": 0})

    buf.flush(deadline_seconds=0.2)

    assert received and received[0] <= 0.2


class TestEvictionVisibility:
    """Evicted events are lost usage: counted in the status and logged with a count (BACK-3919)."""

    def test_evictions_are_counted_in_the_metering_status(self, clean_metering_status):
        buf = make_buffer(max_size=2, replay_fn=RecordingReplayer())

        for i in range(5):
            buf.push("ai", {"seq": i})

        assert get_metering_status().evicted_count == 3
        assert buf.stats()["total_evicted"] == 3

    def test_an_eviction_does_not_hold_the_buffer_lock_while_it_waits_for_the_status_lock(
        self, clean_metering_status
    ):
        from revenium_middleware._core import metering_status

        buf = make_buffer(max_size=1, replay_fn=RecordingReplayer())
        buf.push("ai", {"seq": 0})
        with metering_status._lock:
            evicting = threading.Thread(target=buf.push, args=("ai", {"seq": 1}), daemon=True)
            evicting.start()
            evicting.join(0.2)
            reader = threading.Thread(target=buf.undelivered, daemon=True)
            reader.start()
            reader.join(2.0)
            assert not reader.is_alive(), "the buffer lock was held while waiting for the status lock"
        evicting.join(2.0)
        assert get_metering_status().evicted_count == 1

    def test_reset_clears_the_eviction_count(self, clean_metering_status):
        buf = make_buffer(max_size=1, replay_fn=RecordingReplayer())
        buf.push("ai", {"seq": 0})
        buf.push("ai", {"seq": 1})

        reset_metering_status()

        assert get_metering_status().evicted_count == 0

    def test_each_report_logs_the_evictions_since_the_previous_one(self, caplog):
        buf = make_buffer(max_size=2, replay_fn=RecordingReplayer())
        for i in range(5):
            buf.push("ai", {"seq": i})

        with caplog.at_level("WARNING", logger="revenium_middleware"):
            buf.report_evictions()
            buf.report_evictions()
            buf.push("ai", {"seq": 5})
            buf.report_evictions()

        reports = [r.getMessage() for r in caplog.records if "evicted" in r.getMessage()]
        assert len(reports) == 2
        assert "evicted 3 undelivered event(s)" in reports[0]
        assert "evicted 1 undelivered event(s)" in reports[1] and "(4 in total)" in reports[1]

    def test_the_flush_thread_reports_evictions_every_cycle(self, caplog, monkeypatch):
        from revenium_middleware._core import metering
        monkeypatch.setattr(metering, "shutdown_event", threading.Event())
        buf = make_buffer(max_size=2, flush_interval=0.05,
                          replay_fn=RecordingReplayer(failures={n: make_status_error(503) for n in range(1000)}))

        with caplog.at_level("WARNING", logger="revenium_middleware"):
            for i in range(4):
                buf.push("ai", {"seq": i})
            deadline = time.monotonic() + 5
            while "evicted 2 undelivered" not in caplog.text and time.monotonic() < deadline:
                time.sleep(0.01)
            buf._flush_interval = 3600

        assert "evicted 2 undelivered event(s)" in caplog.text


class ConcurrencyRecorder:
    def __init__(self, seconds, failures=None):
        self._seconds = seconds
        self._lock = threading.Lock()
        self._in_flight = 0
        self.peak = 0
        self.calls = []
        self.failures = dict(failures or {})  # seq -> exception

    def __call__(self, event, timeout_seconds):
        with self._lock:
            self._in_flight += 1
            self.peak = max(self.peak, self._in_flight)
            self.calls.append(event)
        try:
            time.sleep(self._seconds)
            if event.payload["seq"] in self.failures:
                raise self.failures[event.payload["seq"]]
        finally:
            with self._lock:
                self._in_flight -= 1


class TestConcurrentReplay:
    """A backlog drains several events at a time instead of one round trip each (BACK-3919)."""

    def test_the_default_replay_sends_several_events_at_once(self):
        assert MeteringBuffer()._replay_concurrency == metering_buffer.REPLAY_CONCURRENCY > 1

    def test_a_backlog_replays_a_batch_at_a_time(self):
        replayer = ConcurrencyRecorder(0.1)
        buf = make_buffer(max_size=100, replay_fn=replayer, replay_concurrency=16)
        for i in range(48):
            buf.push("ai", {"seq": i})

        started = time.monotonic()
        result = buf.flush()

        assert time.monotonic() - started < 1.5
        assert replayer.peak == 16
        assert result["sent"] == 48
        assert buf.stats()["total_replayed"] == 48

    def test_retryable_failures_in_a_batch_go_back_to_the_front_in_order(self):
        unreachable = make_status_error(503)
        replayer = ConcurrencyRecorder(0.01, failures={1: unreachable, 3: unreachable})
        buf = make_buffer(max_size=100, replay_fn=replayer, replay_concurrency=4)
        for i in range(6):
            buf.push("ai", {"seq": i})

        buf.flush()

        assert [event.payload["seq"] for event in buf._events] == [1, 3, 4, 5]
        assert buf.stats()["total_replayed"] == 2

    def test_a_flush_with_a_deadline_replays_one_at_a_time(self):
        replayer = ConcurrencyRecorder(0.01)
        buf = make_buffer(max_size=100, replay_fn=replayer, replay_concurrency=16)
        for i in range(8):
            buf.push("ai", {"seq": i})

        buf.flush(deadline_seconds=5)

        assert replayer.peak == 1
        assert buf.stats()["total_replayed"] == 8

    def test_with_the_circuit_open_a_flush_sends_one_event_to_probe(self):
        from revenium_middleware._core import delivery_circuit
        for _ in range(delivery_circuit.FAILURES_TO_OPEN):
            delivery_circuit.get_circuit().record_failure()
        replayer = ConcurrencyRecorder(0, failures={n: make_status_error(503) for n in range(10)})
        buf = make_buffer(max_size=100, replay_fn=replayer, replay_concurrency=16)
        for i in range(10):
            buf.push("ai", {"seq": i})

        buf.flush()

        assert len(replayer.calls) == 1
        assert buf.stats()["size"] == 10

    def test_replay_outcomes_open_and_close_the_circuit(self):
        from revenium_middleware._core import delivery_circuit
        circuit = delivery_circuit.get_circuit()
        failing = RecordingReplayer(failures={n: make_status_error(503) for n in range(10)})
        buf = make_buffer(replay_fn=failing)
        buf.push("ai", {"seq": 0})

        for _ in range(delivery_circuit.FAILURES_TO_OPEN):
            buf.flush()
        assert circuit.is_open()

        buf._replay_fn = RecordingReplayer()
        buf.flush()
        assert not circuit.is_open()

    def test_a_batch_with_deliveries_counts_as_the_endpoint_answering(self):
        from revenium_middleware._core import delivery_circuit
        circuit = delivery_circuit.get_circuit()
        for _ in range(delivery_circuit.FAILURES_TO_OPEN + 1):
            mixed = RecordingReplayer(failures={15: make_status_error(503)})
            buf = make_buffer(max_size=100, replay_fn=mixed, replay_concurrency=16)
            for i in range(16):
                buf.push("ai", {"seq": i})
            buf.flush()
            assert not circuit.is_open()

    def test_the_size_stat_counts_events_still_being_replayed(self):
        release = threading.Event()
        started = threading.Event()

        def slow_replay(event, timeout_seconds):
            started.set()
            release.wait(5)

        buf = make_buffer(max_size=100, replay_fn=slow_replay, replay_concurrency=16)
        for i in range(3):
            buf.push("ai", {"seq": i})
        flushing = threading.Thread(target=buf.flush, daemon=True)
        flushing.start()
        try:
            assert started.wait(5)
            assert buf.stats()["size"] == 3
        finally:
            release.set()
            flushing.join(5)
        assert buf.stats()["size"] == 0


class KindReplayer:
    """Replays events whose kind's endpoint answers; ``unreachable`` kinds, and ``failing`` seqs, fail with a 503."""

    def __init__(self, unreachable=(), failing=()):
        self.unreachable = set(unreachable)
        self.failing = set(failing)
        self.calls = []

    def __call__(self, event, timeout_seconds):
        self.calls.append(event)
        if event.kind in self.unreachable or event.payload["seq"] in self.failing:
            raise make_status_error(503)

    def delivered(self):
        return [
            e.payload["seq"] for e in self.calls
            if e.kind not in self.unreachable and e.payload["seq"] not in self.failing
        ]


def open_delivery_circuit():
    from revenium_middleware._core import delivery_circuit
    for _ in range(delivery_circuit.FAILURES_TO_OPEN):
        delivery_circuit.get_circuit().record_failure()
    return delivery_circuit.get_circuit()


class TestReplayPerKind:
    """An unreachable AI endpoint never holds back the tool events behind its records (BACK-3919)."""

    @pytest.mark.parametrize("circuit_open", [False, True], ids=["circuit closed", "circuit open"])
    @pytest.mark.parametrize("concurrency", [1, 16])
    def test_tool_records_behind_a_failing_ai_record_are_replayed(self, circuit_open, concurrency):
        if circuit_open:
            open_delivery_circuit()
        replayer = KindReplayer(unreachable={"ai"})
        buf = make_buffer(max_size=100, replay_fn=replayer, replay_concurrency=concurrency)
        buf.push("ai", {"seq": "ai-0"})
        for i in range(3):
            buf.push("tool", {"seq": f"tool-{i}"})

        result = buf.flush()

        assert replayer.delivered() == ["tool-0", "tool-1", "tool-2"]
        assert [(e.kind, e.payload["seq"]) for e in buf._events] == [("ai", "ai-0")]
        assert result["sent"] == 3

    def test_a_flush_with_a_deadline_also_replays_past_a_failing_ai_record(self):
        replayer = KindReplayer(unreachable={"ai"})
        buf = make_buffer(max_size=100, replay_fn=replayer)
        buf.push("ai", {"seq": "ai-0"})
        buf.push("tool", {"seq": "tool-0"})

        buf.flush(deadline_seconds=5)

        assert replayer.delivered() == ["tool-0"]
        assert [e.payload["seq"] for e in buf._events] == ["ai-0"]

    def test_an_unreachable_kind_is_tried_once_per_flush(self):
        replayer = KindReplayer(unreachable={"ai"})
        buf = make_buffer(max_size=100, replay_fn=replayer)
        for i in range(3):
            buf.push("ai", {"seq": f"ai-{i}"})
            buf.push("tool", {"seq": f"tool-{i}"})

        buf.flush()

        assert [e.payload["seq"] for e in replayer.calls if e.kind == "ai"] == ["ai-0"]
        assert [e.payload["seq"] for e in buf._events] == ["ai-0", "ai-1", "ai-2"]

    def test_records_keep_their_order_within_each_kind_when_both_endpoints_fail(self):
        replayer = KindReplayer(unreachable={"ai"}, failing={"tool-1"})
        buf = make_buffer(max_size=100, replay_fn=replayer)
        buf.push("ai", {"seq": "ai-0"})
        for i in range(3):
            buf.push("tool", {"seq": f"tool-{i}"})
        buf.push("ai", {"seq": "ai-1"})

        buf.flush()

        assert replayer.delivered() == ["tool-0"]
        remaining = [(e.kind, e.payload["seq"]) for e in buf._events]
        assert [seq for kind, seq in remaining if kind == "ai"] == ["ai-0", "ai-1"]
        assert [seq for kind, seq in remaining if kind == "tool"] == ["tool-1", "tool-2"]

    def test_a_flush_examines_each_blocked_record_once(self):
        examined = []

        def counting_clock():
            examined.append(None)
            return time.time()

        buf = make_buffer(max_size=5000, replay_fn=KindReplayer(unreachable={"ai"}), replay_concurrency=16,
                          now_fn=counting_clock)
        for i in range(2000):
            buf.push("ai", {"seq": f"ai-{i}"})
        for i in range(2000):
            buf.push("tool", {"seq": f"tool-{i}"})
        examined.clear()

        result = buf.flush()

        assert result["sent"] == 2000
        assert len(examined) <= 4000
        assert [e.payload["seq"] for e in buf._events] == [f"ai-{i}" for i in range(2000)]

    def test_parked_records_count_toward_capacity_and_are_evicted_first(self):
        seen = {}

        def replay(event, timeout_seconds):
            if event.kind == "ai":
                raise make_status_error(503)
            for i in range(4):
                buf.push("ai", {"seq": f"new-{i}"})
                if i == 1:
                    seen["undelivered"] = buf.undelivered()

        buf = make_buffer(max_size=5, replay_fn=replay)
        buf.push("ai", {"seq": "ai-0"})
        buf.push("ai", {"seq": "ai-1"})
        buf.push("tool", {"seq": "tool-0"})

        buf.flush()

        assert seen["undelivered"] == 5
        assert [e.payload["seq"] for e in buf._events] == ["ai-1", "new-0", "new-1", "new-2", "new-3"]
        assert buf.stats()["total_evicted"] == 1

    def test_parked_records_return_in_order_when_the_deadline_ends_the_flush(self):
        def replay(event, timeout_seconds):
            if event.kind == "ai":
                raise make_status_error(503)
            time.sleep(0.3)

        buf = make_buffer(max_size=100, replay_fn=replay)
        buf.push("ai", {"seq": "ai-0"})
        buf.push("ai", {"seq": "ai-1"})
        buf.push("tool", {"seq": "tool-0"})
        buf.push("tool", {"seq": "tool-1"})

        buf.flush(deadline_seconds=0.2)

        assert [e.payload["seq"] for e in buf._events] == ["ai-0", "ai-1", "tool-1"]
        assert buf.undelivered() == 3

    def test_parked_records_return_in_order_when_the_flush_raises(self):
        class Interrupted(BaseException):
            pass

        def replay(event, timeout_seconds):
            if event.kind == "ai":
                raise make_status_error(503)
            raise Interrupted()

        buf = make_buffer(max_size=100, replay_fn=replay)
        buf.push("ai", {"seq": "ai-0"})
        buf.push("ai", {"seq": "ai-1"})
        buf.push("tool", {"seq": "tool-0"})
        buf.push("ai", {"seq": "ai-2"})

        with pytest.raises(Interrupted):
            buf.flush()

        assert [e.payload["seq"] for e in buf._events] == ["ai-0", "ai-1", "ai-2"]

    def test_while_the_circuit_is_open_a_batch_still_probes_with_one_ai_record(self):
        open_delivery_circuit()
        replayer = KindReplayer(unreachable={"ai"})
        buf = make_buffer(max_size=100, replay_fn=replayer, replay_concurrency=16)
        for i in range(10):
            buf.push("ai", {"seq": f"ai-{i}"})

        buf.flush()

        assert len(replayer.calls) == 1
        assert buf.stats()["size"] == 10

    def test_tool_replays_neither_close_nor_open_the_circuit(self):
        circuit = open_delivery_circuit()
        buf = make_buffer(max_size=100, replay_fn=KindReplayer())
        buf.push("tool", {"seq": "tool-0"})
        buf.flush()
        assert circuit.is_open()

        from revenium_middleware._core import delivery_circuit
        delivery_circuit.reset()
        failing = make_buffer(max_size=100, replay_fn=KindReplayer(unreachable={"tool"}))
        failing.push("tool", {"seq": "tool-1"})
        for _ in range(delivery_circuit.FAILURES_TO_OPEN):
            failing.flush()
        assert not delivery_circuit.get_circuit().is_open()


class TestReplayTimeout:
    """A replay near or past its flush deadline still gets a timeout httpx accepts."""

    def test_a_spent_budget_still_yields_a_positive_timeout(self):
        timeout = make_buffer()._replay_timeout(0.1, started=time.monotonic() - 5.0)

        assert timeout > 0

    def test_a_spent_budget_yields_a_timeout_the_socket_layer_accepts(self):
        timeout = make_buffer()._replay_timeout(0.1, started=time.monotonic() - 5.0)
        left, right = socket.socketpair()
        try:
            left.settimeout(timeout)
        finally:
            left.close()
            right.close()

