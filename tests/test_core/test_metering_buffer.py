"""Store-and-forward buffer: failed metering events are buffered and replayed.

Covers the buffer unit contract (FIFO, bounds, TTL, stop-on-retryable,
discard-on-permanent), the retryability classification, and the integration
points: submit_ai_event, tool-event dispatch, and shutdown drain.
"""
import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import httpx
import pytest

from revenium_middleware._core import metering_buffer
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

        assert replayer.timeouts == [metering_buffer.REPLAY_TIMEOUT_SECONDS]

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
        call = healthy.ai.create_completion.call_args
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
            context=get_context()))

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
            context=get_context()))

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


def test_tiny_deadline_strictly_bounds_per_call_timeout():
    received = []

    def replayer(event, timeout_seconds):
        received.append(timeout_seconds)

    buf = MeteringBuffer(max_size=10, flush_interval=9999.0, replay_fn=replayer)
    buf.push("ai", {"seq": 0})

    buf.flush(deadline_seconds=0.2)

    assert received and received[0] <= 0.2
