"""Store-and-forward buffer: failed metering events are buffered and replayed.

Covers the buffer unit contract (FIFO, bounds, TTL, stop-on-retryable,
discard-on-permanent), the retryability classification, and the integration
points: submit_ai_event, tool-event dispatch, and shutdown drain.
"""
import datetime
import json
import socket
import threading
import time
from unittest.mock import MagicMock, patch

import httpx
import pytest

from revenium_middleware._core import metering, metering_buffer, tool_event_transport
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


class FailingToolEventClient:
    """Shared tool-event client double whose post() always fails with ``status_code``."""

    def __init__(self, status_code):
        self.status_code = status_code
        self.timeouts = []
        self.sent = []

    def post(self, url, headers=None, json=None, timeout=None):
        self.timeouts.append(timeout)
        self.sent.append((headers, json))
        request = httpx.Request("POST", url)
        response = httpx.Response(self.status_code, request=request)
        raise httpx.HTTPStatusError(str(self.status_code), request=request, response=response)


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
        """Any status but a redirect, which is never retried (see test_a_redirect_is_never_retryable)."""
        assert is_retryable_failure(
            make_status_error(422, {"Retry-After": "2"})) is True

    @pytest.mark.parametrize("status, headers", [
        (301, None),
        (302, None),
        (303, None),
        (307, None),
        (308, None),
        (307, {"Retry-After": "1"}),
        (307, {"x-should-retry": "true"}),
    ])
    def test_a_redirect_is_never_retryable(self, status, headers):
        assert is_retryable_failure(make_status_error(status, headers)) is False

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
        monkeypatch.setattr(metering_submission, "current_client", lambda: client)

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
        monkeypatch.setattr(metering_submission, "current_client", lambda: client)

        with idempotency_key("ctx-key-99"):
            metering_submission.submit_ai_event("completion", {"model": "gpt-test"})

        event = fresh_buffer._events[0]
        assert event.payload["args"]["extra_headers"]["Idempotency-Key"] == "ctx-key-99"

    def test_second_event_in_a_block_is_buffered_and_replayed_under_its_own_key(
        self, monkeypatch, switchable_client
    ):
        from revenium_middleware import idempotency_key
        from revenium_middleware._core import metering_submission
        buf = MeteringBuffer(max_size=10, flush_interval=9999.0)  # real replayer
        monkeypatch.setattr(metering_buffer, "_buffer", buf)
        sent_keys = []

        def second_request_drops(request):
            sent_keys.append(request.headers.get("Idempotency-Key"))
            if len(sent_keys) == 2:
                raise httpx.ConnectError("connection dropped", request=request)
            return httpx.Response(200, json={})

        from revenium_middleware._metering import ReveniumMetering
        switchable_client(ReveniumMetering(
            api_key="hak_org_a_dummy", base_url="https://org-a.example/meter/", max_retries=0,
            http_client=httpx.Client(transport=httpx.MockTransport(second_request_drops))))

        with idempotency_key("order-42"):
            metering_submission.submit_ai_event("completion", dict(FROZEN_COMPLETION_ARGS))
            metering_submission.submit_ai_event("completion", dict(FROZEN_COMPLETION_ARGS))
        assert buf.stats()["size"] == 1
        assert buf._events[0].payload["args"]["extra_headers"]["Idempotency-Key"] == "order-42:2"

        buf.flush()

        assert buf.stats()["size"] == 0
        assert sent_keys == ["order-42", "order-42:2", "order-42:2"]

    def test_permanent_failure_is_not_buffered_and_raises(self, fresh_buffer, monkeypatch):
        from revenium_middleware._core import metering_submission
        client = self._client_raising(make_status_error(422))
        monkeypatch.setattr(metering_submission, "current_client", lambda: client)

        with pytest.raises(APIStatusError):
            metering_submission.submit_ai_event("completion", {"model": "gpt-test"})

        assert fresh_buffer.stats()["size"] == 0

    def test_replay_reuses_original_idempotency_key(self, monkeypatch, switchable_client):
        """Card scenario: 503 -> buffered -> backend restored -> replayed, same key."""
        from revenium_middleware._core import metering_submission
        buf = MeteringBuffer(max_size=10, flush_interval=9999.0)  # real replayer
        monkeypatch.setattr(metering_buffer, "_buffer", buf)
        wire = MeteringWire(failures=1)
        switchable_client(wire.client("hak_org_a_dummy", "https://org-a.example/meter/"))

        metering_submission.submit_ai_event(
            "completion", dict(FROZEN_COMPLETION_ARGS), idempotency_key="replay-me")
        assert buf.stats()["size"] == 1
        buf.flush()

        assert buf.stats()["size"] == 0
        assert wire.requests == [
            ("hak_org_a_dummy", "org-a.example", "replay-me"),
            ("hak_org_a_dummy", "org-a.example", "replay-me"),
        ]

    def test_ai_replay_after_a_key_switch_uses_the_key_it_was_recorded_with(self, monkeypatch, switchable_client):
        from revenium_middleware._core import metering_submission
        buf = MeteringBuffer(max_size=10, flush_interval=9999.0)  # real replayer
        monkeypatch.setattr(metering_buffer, "_buffer", buf)
        wire = MeteringWire(failures=1)
        monkeypatch.setattr(metering, "_new_client", wire.client)
        switchable_client(wire.client("hak_org_a_dummy", "https://org-a.example/meter/"))

        metering_submission.submit_ai_event("completion", dict(FROZEN_COMPLETION_ARGS), idempotency_key="held-1")
        assert buf.stats()["size"] == 1
        switchable_client(wire.client("hak_org_b_dummy", "https://org-b.example/meter/"))

        buf.flush()

        assert buf.stats()["size"] == 0
        assert wire.requests == [
            ("hak_org_a_dummy", "org-a.example", "held-1"),
            ("hak_org_a_dummy", "org-a.example", "held-1"),
        ]


    def test_ai_event_without_frozen_credentials_is_discarded_not_sent_with_the_current_client(
        self, monkeypatch, switchable_client
    ):
        buf = MeteringBuffer(max_size=10, flush_interval=9999.0)  # real replayer
        wire = MeteringWire()
        switchable_client(wire.client("hak_org_b_dummy", "https://org-b.example/meter/"))
        buf.push("ai", {"operation": "completion", "args": dict(FROZEN_COMPLETION_ARGS)})

        result = buf.flush()

        assert wire.requests == []
        assert result["discarded"] == 1
        assert buf.stats()["size"] == 0

FROZEN_COMPLETION_ARGS = {
    "completion_start_time": "2026-10-08T00:00:00Z", "cost_type": "AI", "input_token_count": 1,
    "is_streamed": False, "model": "gpt-test", "output_token_count": 1, "provider": "OPENAI",
    "request_duration": 1, "request_time": "2026-10-08T00:00:00Z", "response_time": "2026-10-08T00:00:01Z",
    "stop_reason": "END", "total_token_count": 2, "transaction_id": "txn-key-switch",
}


class MeteringWire:
    """Records the key and host of every metering request; answers the first ``failures`` with 503."""

    def __init__(self, failures=0):
        self.requests = []
        self._failures = failures

    def __call__(self, request):
        self.requests.append((request.headers["x-api-key"], request.url.host,
                              request.headers.get("Idempotency-Key")))
        if len(self.requests) <= self._failures:
            return httpx.Response(503, json={"error": "unavailable"})
        return httpx.Response(200, json={})

    def client(self, api_key, base_url):
        from revenium_middleware._metering import ReveniumMetering
        return ReveniumMetering(api_key=api_key, base_url=base_url, max_retries=0,
                                http_client=httpx.Client(transport=httpx.MockTransport(self)))


@pytest.fixture()
def switchable_client(monkeypatch):
    """Install a metering client the way initialize_metering does; returns a setter."""
    import revenium_middleware
    from revenium_middleware import _core

    def install(client):
        for module in (metering, _core, revenium_middleware):
            monkeypatch.setattr(module, "client", client, raising=False)

    return install


class TestOverflowedAiEventKeepsItsKey:
    """BACK-4016: an AI record that overflowed the delivery queue keeps the key current when it was queued."""

    def test_overflowed_ai_event_is_built_with_the_client_current_when_it_was_queued(
        self, monkeypatch, switchable_client
    ):
        from revenium_middleware._core import metering_pool, metering_submission
        from revenium_middleware._core.metering_pool import MeteringWorkerPool
        buf = MeteringBuffer(max_size=10, flush_interval=9999.0)  # real replayer
        monkeypatch.setattr(metering_buffer, "_buffer", buf)
        pool = MeteringWorkerPool(1, queue_size=1, overflow=metering_pool._overflow_to_buffer)
        monkeypatch.setattr(metering_pool, "_pool", pool)
        wire = MeteringWire()
        monkeypatch.setattr(metering, "_new_client", wire.client)
        switchable_client(wire.client("hak_org_a_dummy", "https://org-a.example/meter/"))
        picked_up, release = threading.Event(), threading.Event()

        async def occupy_the_worker():
            picked_up.set()
            release.wait(10)

        async def fill_the_queue():
            release.wait(10)

        async def meter():
            metering_submission.submit_ai_event(
                "completion", dict(FROZEN_COMPLETION_ARGS), idempotency_key="overflowed-1")

        try:
            metering.run_async_in_thread(occupy_the_worker())
            assert picked_up.wait(5)
            metering.run_async_in_thread(fill_the_queue())
            metering.run_async_in_thread(meter())
            assert buf.stats()["size"] == 1
            switchable_client(wire.client("hak_org_b_dummy", "https://org-b.example/meter/"))

            buf.build_all_overflow(deadline_seconds=5)
            buf.flush()
        finally:
            release.set()
            pool.stop(timeout=5)

        assert wire.requests == [("hak_org_a_dummy", "org-a.example", "overflowed-1")]

    def test_ai_event_queued_while_metering_was_disabled_is_not_sent_under_a_key_enabled_later(
        self, monkeypatch, switchable_client
    ):
        from revenium_middleware._core import metering_pool, metering_submission
        from revenium_middleware._core.metering_pool import MeteringWorkerPool
        buf = MeteringBuffer(max_size=10, flush_interval=9999.0)  # real replayer
        monkeypatch.setattr(metering_buffer, "_buffer", buf)
        pool = MeteringWorkerPool(1, queue_size=1, overflow=metering_pool._overflow_to_buffer)
        monkeypatch.setattr(metering_pool, "_pool", pool)
        wire = MeteringWire()
        monkeypatch.setattr(metering, "_new_client", wire.client)
        monkeypatch.setattr(metering, "get_client", lambda: None)
        picked_up, release = threading.Event(), threading.Event()

        async def occupy_the_worker():
            picked_up.set()
            release.wait(10)

        async def fill_the_queue():
            release.wait(10)

        async def meter():
            metering_submission.submit_ai_event(
                "completion", dict(FROZEN_COMPLETION_ARGS), idempotency_key="disabled-1")

        try:
            metering.run_async_in_thread(occupy_the_worker())
            assert picked_up.wait(5)
            metering.run_async_in_thread(fill_the_queue())
            metering.run_async_in_thread(meter())
            client_b = wire.client("hak_org_b_dummy", "https://org-b.example/meter/")
            monkeypatch.setattr(metering, "get_client", lambda: client_b)
            switchable_client(client_b)

            buf.build_all_overflow(deadline_seconds=5)
            buf.flush()
        finally:
            release.set()
            pool.stop(timeout=5)

        assert wire.requests == []


class TransactionRecordingWire(MeteringWire):
    """A ``MeteringWire`` that also records which key each transaction was sent under."""

    def __init__(self, failures=0):
        super().__init__(failures=failures)
        self.key_per_transaction = {}

    def __call__(self, request):
        transaction_id = json.loads(request.content)["transactionId"]
        self.key_per_transaction[transaction_id] = request.headers["x-api-key"]
        return super().__call__(request)


class CountingClientFactory:
    """Stands in for ``metering._new_client``: counts builds per key and address and keeps every client built."""

    def __init__(self, wire, close_error=None):
        self.wire = wire
        self.built = {}
        self.clients = []
        self._close_error = close_error
        self._lock = threading.Lock()

    def __call__(self, api_key, base_url):
        time.sleep(0.01)  # widens the window in which concurrent replays miss together
        client = self.wire.client(api_key, base_url)
        if self._close_error is not None:
            def close():
                raise self._close_error
            client.close = close
        with self._lock:
            self.built[(api_key, base_url)] = self.built.get((api_key, base_url), 0) + 1
            self.clients.append(client)
        return client

    def builds(self):
        return sum(self.built.values())

    def builds_per_address(self):
        per_address = {}
        for (_, base_url), count in self.built.items():
            per_address[base_url] = per_address.get(base_url, 0) + count
        return per_address

    def all_closed(self):
        return all(client.is_closed() for client in self.clients)


class TestAiReplayUnderAnOldKeyReusesOneClient:
    """BACK-4264: AI records held under another key than the current one share a client per address in a flush."""

    KEY_A, ADDRESS_A = "hak_org_a_dummy", "https://org-a.example/meter/"
    KEY_C, ADDRESS_C = "hak_org_c_dummy", "https://org-c.example/meter/"

    @staticmethod
    def push_held(buf, wire, api_key, base_url, transaction_id):
        from revenium_middleware._core import metering_submission
        recorded_with = wire.client(api_key, base_url)
        buf.push("ai", metering_submission._replay_payload(
            "completion", dict(FROZEN_COMPLETION_ARGS, transaction_id=transaction_id), recorded_with))

    @pytest.fixture()
    def wire_on_key_b(self, monkeypatch, switchable_client):
        def install(failures=0, close_error=None):
            wire = TransactionRecordingWire(failures=failures)
            factory = CountingClientFactory(wire, close_error=close_error)
            monkeypatch.setattr(metering, "_new_client", factory)
            switchable_client(wire.client("hak_org_b_dummy", "https://org-b.example/meter/"))
            return wire, factory

        return install

    def test_a_backlog_held_under_an_old_key_builds_one_client_for_that_key(self, wire_on_key_b, switchable_client):
        wire, factory = wire_on_key_b()
        buf = MeteringBuffer(max_size=100, flush_interval=9999.0)  # real replayer
        for i in range(40):
            self.push_held(buf, wire, self.KEY_A, self.ADDRESS_A, f"t{i}")

        result = buf.flush()

        assert factory.built == {(self.KEY_A, self.ADDRESS_A): 1}
        assert result["sent"] == 40
        assert result["discarded"] == 0
        assert len(wire.requests) == 40
        assert {(key, host) for key, host, _ in wire.requests} == {(self.KEY_A, "org-a.example")}
        assert factory.all_closed()

        switchable_client(wire.client(self.KEY_A, self.ADDRESS_A))
        for i in range(10):
            self.push_held(buf, wire, self.KEY_A, self.ADDRESS_A, f"again{i}")

        buf.flush()

        assert factory.builds() == 1
        assert len(wire.requests) == 50

    def test_the_exit_drain_builds_one_client_per_old_address(self, wire_on_key_b):
        wire, factory = wire_on_key_b()
        buf = MeteringBuffer(max_size=100, flush_interval=9999.0)  # real replayer
        for i in range(20):
            self.push_held(buf, wire, self.KEY_A, self.ADDRESS_A, f"a{i}")
            self.push_held(buf, wire, self.KEY_C, self.ADDRESS_C, f"c{i}")

        result = buf.flush(deadline_seconds=5)

        assert factory.builds_per_address() == {self.ADDRESS_A: 1, self.ADDRESS_C: 1}
        assert result["sent"] == 40
        sent_as = [(key, host) for key, host, _ in wire.requests]
        assert sent_as.count((self.KEY_A, "org-a.example")) == 20
        assert sent_as.count((self.KEY_C, "org-c.example")) == 20
        assert factory.all_closed()

    def test_records_still_held_after_a_flush_get_one_new_client_in_the_next(self, wire_on_key_b):
        wire, factory = wire_on_key_b(failures=1)
        buf = MeteringBuffer(max_size=100, flush_interval=9999.0, replay_concurrency=1)  # real replayer
        for i in range(5):
            self.push_held(buf, wire, self.KEY_A, self.ADDRESS_A, f"t{i}")

        first = buf.flush()

        assert first["sent"] == 0
        assert buf.stats()["size"] == 5
        assert factory.builds() == 1
        assert factory.all_closed()

        second = buf.flush()

        assert second["sent"] == 5
        assert buf.stats()["size"] == 0
        assert factory.builds() == 2
        assert factory.all_closed()
        assert len(wire.requests) == 6
        assert {key for key, _, _ in wire.requests} == {self.KEY_A}

    def test_a_backlog_held_under_many_old_keys_at_one_address_builds_one_client(self, wire_on_key_b):
        wire, factory = wire_on_key_b()
        buf = MeteringBuffer(max_size=100, flush_interval=9999.0)  # real replayer
        keys = [f"hak_cust{i}_dummy" for i in range(40)]
        for i, key in enumerate(keys):
            self.push_held(buf, wire, key, self.ADDRESS_A, f"t{i}")

        buf.flush()

        assert factory.builds() == 1
        assert len(wire.requests) == 40
        assert wire.key_per_transaction == {f"t{i}": key for i, key in enumerate(keys)}

    def test_a_replay_client_that_fails_to_close_does_not_block_the_next_flush(self, wire_on_key_b):
        wire, factory = wire_on_key_b(close_error=RuntimeError("close failed"))
        buf = MeteringBuffer(max_size=100, flush_interval=9999.0)  # real replayer
        self.push_held(buf, wire, self.KEY_A, self.ADDRESS_A, "first")

        first = buf.flush()
        self.push_held(buf, wire, self.KEY_A, self.ADDRESS_A, "second")
        second = buf.flush(deadline_seconds=1)

        assert first["sent"] == 1
        assert second["sent"] == 1
        assert buf.stats()["size"] == 0


class TestToolEventIntegration:
    def test_a_built_tool_event_keeps_its_transaction_id_through_retry_and_replay(self, monkeypatch):
        import time
        from revenium_middleware._metering import decorator as tool_metering
        from revenium_middleware._metering.context import context

        buf = MeteringBuffer(max_size=10, flush_interval=9999.0)
        monkeypatch.setattr(metering_buffer, "_buffer", buf)
        first_attempt = FailingToolEventClient(503)
        monkeypatch.setattr(tool_event_transport, "_client", first_attempt)
        tool_metering.configure(metering_url="http://metering.test", api_key="rev_mk_test")
        try:
            with context(transaction_id="txn-1"):
                tool_metering.report_tool_call(tool_id="firecrawl", operation="scrape")

            deadline = time.monotonic() + 6
            while time.monotonic() < deadline and not (first_attempt.sent and buf.stats()["size"] == 1):
                time.sleep(0.05)
            assert buf.stats()["size"] == 1

            replayed = []

            class Recovered:
                def post(self, url, headers=None, json=None, timeout=None):
                    replayed.append((headers, json))
                    return httpx.Response(201, request=httpx.Request("POST", url))

            monkeypatch.setattr(tool_event_transport, "_client", Recovered())
            buf.flush()
        finally:
            tool_metering.configure()

        posts = first_attempt.sent + replayed
        assert len(posts) >= 2
        transaction_ids = {body["transactionId"] for _, body in posts}
        assert len(transaction_ids) == 1
        assert {headers["Idempotency-Key"] for headers, _ in posts} == transaction_ids

    def test_buffered_tool_event_replays_with_the_ids_of_its_first_attempt(self, monkeypatch):
        import asyncio
        from revenium_middleware._metering import decorator as tool_metering
        from revenium_middleware._metering.context import ReveniumContext

        buf = MeteringBuffer(max_size=10, flush_interval=9999.0)  # real replayer
        monkeypatch.setattr(metering_buffer, "_buffer", buf)
        first_attempt = FailingToolEventClient(503)
        monkeypatch.setattr(tool_event_transport, "_client", first_attempt)

        asyncio.run(tool_metering._send_tool_event_async(
            "https://api.test/meter/v2/tool/events", "hak_k",
            tool_id="buffered-tool", operation="run", duration_ms=5,
            success=True, error_message=None, usage_metadata=None,
            context=ReveniumContext(transaction_id="txn-group"),
            occurred_at=datetime.datetime.now(datetime.timezone.utc), event_sequence="test:1"))

        replayed = []

        class Recovered:
            def post(self, url, headers=None, json=None, timeout=None):
                replayed.append((headers, json))
                return httpx.Response(201, request=httpx.Request("POST", url))

        monkeypatch.setattr(tool_event_transport, "_client", Recovered())
        buf.flush()

        (first_headers, first_body), = first_attempt.sent
        (replay_headers, replay_body), = replayed
        assert replay_body["transactionId"] == first_body["transactionId"]
        assert replay_headers["Idempotency-Key"] == first_headers["Idempotency-Key"]
        assert replay_headers["Idempotency-Key"] == replay_body["transactionId"]

    def test_retryable_tool_failure_is_buffered(self, fresh_buffer, monkeypatch):
        import asyncio
        from revenium_middleware._metering import decorator as tool_metering
        from revenium_middleware._metering.context import get_context

        monkeypatch.setattr(tool_event_transport, "_client", FailingToolEventClient(503))

        asyncio.run(tool_metering._send_tool_event_async(
            "https://api.test/meter/v2/tool/events", "hak_k",
            tool_id="buffered-tool", operation="run", duration_ms=5,
            success=True, error_message=None, usage_metadata=None,
            context=get_context(), occurred_at=datetime.datetime.now(datetime.timezone.utc), event_sequence="test:1"))

        assert fresh_buffer.stats()["size"] == 1
        event = fresh_buffer._events[0]
        assert event.kind == "tool"
        assert event.payload["event_payload"]["toolId"] == "buffered-tool"

    def test_permanent_tool_failure_is_not_buffered(self, fresh_buffer, monkeypatch):
        import asyncio
        from revenium_middleware._metering import decorator as tool_metering
        from revenium_middleware._metering.context import get_context

        monkeypatch.setattr(tool_event_transport, "_client", FailingToolEventClient(422))

        asyncio.run(tool_metering._send_tool_event_async(
            "https://api.test/meter/v2/tool/events", "hak_k",
            tool_id="poison-tool", operation="run", duration_ms=5,
            success=True, error_message=None, usage_metadata=None,
            context=get_context(), occurred_at=datetime.datetime.now(datetime.timezone.utc), event_sequence="test:1"))

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

    def _capture_httpx(self, monkeypatch, status_code=200):
        calls = []

        def respond(url):
            request = httpx.Request("POST", url)
            return httpx.Response(status_code, request=request)

        class Client:
            def post(self, url, headers=None, json=None, timeout=None):
                calls.append(dict(url=url, headers=headers, json=json, timeout=timeout))
                return respond(url)

        monkeypatch.setattr(tool_event_transport, "_client", Client())
        return calls

    def test_replayed_tool_events_reuse_one_connection(self, keep_alive_meter):
        payload = {
            "url": keep_alive_meter.url,
            "key": "hak_frozen",
            "event_payload": {"transactionId": "txn-replay-1", "toolId": "t"},
        }

        for _ in range(10):
            metering_buffer._replay_tool_event(payload, 5.0)

        assert len(keep_alive_meter.ports) == 10
        assert len(set(keep_alive_meter.ports)) == 1

    def test_tool_send_and_replay_share_one_connection(self, keep_alive_meter):
        import asyncio
        from revenium_middleware._metering import decorator as tool_metering
        from revenium_middleware._metering.context import ReveniumContext

        asyncio.run(tool_metering._send_tool_event_async(
            keep_alive_meter.url, "hak_k",
            tool_id="t", operation="run", duration_ms=5,
            success=True, error_message=None, usage_metadata=None,
            context=ReveniumContext(), occurred_at=datetime.datetime.now(datetime.timezone.utc), event_sequence="test:1"))
        metering_buffer._replay_tool_event({
            "url": keep_alive_meter.url,
            "key": "hak_k",
            "event_payload": {"transactionId": "txn-replay-1", "toolId": "t"},
        }, 5.0)

        assert len(keep_alive_meter.ports) == 2
        assert len(set(keep_alive_meter.ports)) == 1

    def test_replay_uses_the_key_the_event_was_recorded_under(self, monkeypatch):
        buf = self._buffered_tool_buffer()
        calls = self._capture_httpx(monkeypatch)
        monkeypatch.setattr(
            "revenium_middleware._metering.decorator._resolve_endpoint",
            lambda: ("https://other-org.example/meter/v2/tool/events", "hak_other_org"))

        buf.flush()

        assert buf.stats()["size"] == 0
        assert len(calls) == 1
        call = calls[0]
        assert call["url"] == "https://frozen.example/meter/v2/tool/events"
        assert call["headers"]["x-api-key"] == "hak_frozen"
        assert call["headers"]["Idempotency-Key"] == "txn-replay-1"

    def test_replay_keeps_the_frozen_key_when_the_key_was_rotated_since(self, monkeypatch):
        buf = self._buffered_tool_buffer()
        calls = self._capture_httpx(monkeypatch)
        monkeypatch.setattr(
            "revenium_middleware._metering.decorator._resolve_endpoint",
            lambda: ("https://frozen.example/meter/v2/tool/events", "hak_rotated"))

        buf.flush()

        call = calls[0]
        assert call["url"] == "https://frozen.example/meter/v2/tool/events"
        assert call["headers"]["x-api-key"] == "hak_frozen"

    def test_a_held_event_whose_key_is_rejected_is_discarded_not_resent_under_the_current_key(self, monkeypatch):
        buf = self._buffered_tool_buffer()
        calls = self._capture_httpx(monkeypatch, status_code=401)
        monkeypatch.setattr(
            "revenium_middleware._metering.decorator._resolve_endpoint",
            lambda: ("https://other-org.example/meter/v2/tool/events", "hak_other_org"))

        result = buf.flush()

        assert len(calls) == 1
        assert calls[0]["headers"]["x-api-key"] == "hak_frozen"
        assert result["discarded"] == 1
        assert buf.stats()["size"] == 0


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

    def test_a_flush_with_a_deadline_replays_held_records_in_parallel(self):
        """BACK-4355: the exit flush replayed one record at a time."""
        replayer = ConcurrencyRecorder(0.2)
        buf = make_buffer(max_size=1000, replay_fn=replayer, replay_concurrency=1, exit_replay_concurrency=16)
        for i in range(64):
            buf.push("ai", {"seq": i})

        started = time.monotonic()
        result = buf.flush(deadline_seconds=2.0)

        assert time.monotonic() - started < 2.5
        assert result["sent"] == 64
        assert replayer.peak > 1

    def test_the_default_exit_flush_replays_more_records_at_once_than_the_background_flush(self):
        assert MeteringBuffer()._exit_replay_concurrency == metering_buffer.EXIT_REPLAY_CONCURRENCY
        assert metering_buffer.EXIT_REPLAY_CONCURRENCY > metering_buffer.REPLAY_CONCURRENCY

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


    def test_a_replay_that_waited_for_its_client_gets_only_the_time_left_and_the_flush_keeps_its_deadline(
        self, monkeypatch
    ):
        """BACK-4355: replays waiting on one another's client builds started with the batch's full timeout,
        and the flush waited for them past its deadline."""
        build_seconds, budget = 0.3, 0.75
        started_requests = []

        class SlowToBuildClient:
            def __init__(self):
                self.ai = self

            def with_options(self, **options):
                return self

            def create_completion(self, timeout, **kwargs):
                started_requests.append((time.monotonic(), timeout))

            def close(self):
                pass

        def build_slowly(api_key, base_url):
            time.sleep(build_seconds)
            return SlowToBuildClient()

        monkeypatch.setattr(metering, "_new_client", build_slowly)
        monkeypatch.setattr(metering, "get_client", lambda: None)
        buf = MeteringBuffer(max_size=10, flush_interval=9999.0, replay_timeout_seconds=10.0)
        for n in range(3):
            buf.push("ai", {"api_key": "hak_test_dummy", "base_url": f"https://org-{n}.example/meter/",
                            "operation": "completion", "args": {}})

        flush_started = time.monotonic()
        result = buf.flush(deadline_seconds=budget)
        elapsed = time.monotonic() - flush_started

        assert elapsed < budget + 0.15
        assert started_requests
        for request_started, timeout in started_requests:
            assert request_started - flush_started + timeout <= budget + 0.05
        assert result["sent"] == len(started_requests) < 3
        assert result["remaining"] == 3 - result["sent"]
