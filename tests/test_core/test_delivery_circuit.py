"""An unreachable metering endpoint stops holding the delivery workers (BACK-3919)."""
import contextvars
import logging
import time

import httpx
import pytest

from revenium_middleware._core import delivery_circuit, metering_buffer
from revenium_middleware._core.delivery_circuit import DeliveryCircuit
from revenium_middleware._core.metering_buffer import MeteringBuffer
from revenium_middleware._core.metering_pool import MeteringTask, MeteringWorkerPool
from revenium_middleware._core.metering_submission import submit_ai_event
from revenium_middleware._metering import decorator as tool_metering
from revenium_middleware._metering._exceptions import APIConnectionError

COMPLETION_ARGS = {"model": "gpt-4o-mini", "input_token_count": 3, "output_token_count": 2}


class FakeClock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


def connection_error():
    return APIConnectionError(request=httpx.Request("POST", "http://metering.test/v2/ai/completions"))


def open_circuit(circuit):
    for _ in range(delivery_circuit.FAILURES_TO_OPEN):
        circuit.record_failure()


@pytest.fixture
def clock():
    return FakeClock()


@pytest.fixture
def circuit(clock):
    return DeliveryCircuit(failures_to_open=3, probe_interval=5.0, clock=clock)


@pytest.fixture
def buffer(monkeypatch):
    buf = MeteringBuffer(flush_interval=3600, replay_fn=lambda event, timeout: None)
    monkeypatch.setattr(metering_buffer, "_buffer", buf)
    return buf


class TestCircuit:
    def test_sends_while_failures_stay_below_the_threshold(self, circuit):
        circuit.record_failure()
        circuit.record_failure()

        assert circuit.allows_send()
        assert not circuit.is_open()

    def test_a_success_resets_the_failure_count(self, circuit):
        circuit.record_failure()
        circuit.record_failure()
        circuit.record_success()
        circuit.record_failure()
        circuit.record_failure()

        assert not circuit.is_open()

    def test_opens_after_consecutive_failures_and_holds_records_back(self, circuit):
        open_circuit(circuit)

        assert circuit.is_open()
        assert not circuit.allows_send()

    def test_lets_one_probe_through_per_interval_while_open(self, circuit, clock):
        open_circuit(circuit)

        clock.now += 5.0
        assert circuit.allows_send()
        assert not circuit.allows_send()
        clock.now += 4.9
        assert not circuit.allows_send()
        clock.now += 0.1
        assert circuit.allows_send()

    def test_a_failed_probe_pushes_the_next_one_back(self, circuit, clock):
        open_circuit(circuit)
        clock.now += 5.0
        assert circuit.allows_send()

        clock.now += 3.0
        circuit.record_failure()
        clock.now += 4.0

        assert not circuit.allows_send()

    def test_a_success_closes_it_and_says_so_once(self, circuit):
        open_circuit(circuit)

        assert circuit.record_success() is True
        assert circuit.record_success() is False
        assert circuit.allows_send()

    def test_warns_once_when_it_opens_and_once_when_it_closes(self, circuit, caplog):
        with caplog.at_level(logging.WARNING, logger="revenium_middleware"):
            open_circuit(circuit)
            circuit.record_failure()
            circuit.record_failure()
            circuit.record_success()
            circuit.record_success()

        messages = [record.getMessage() for record in caplog.records]
        assert sum("failed 3 deliveries in a row" in message for message in messages) == 1
        assert sum("answering again" in message for message in messages) == 1


class ToolEndpoint:
    """Stands in for the httpx module the tool-event sender posts through; records each URL it is sent to."""

    def __init__(self):
        self.urls = []
        endpoint = self

        class AsyncClient:
            def __init__(self, timeout=None):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *exc):
                return False

            async def post(self, url, headers=None, json=None):
                endpoint.urls.append(url)
                return httpx.Response(202, request=httpx.Request("POST", url))

        self.AsyncClient = AsyncClient

    def wait_for_post(self, timeout=5.0):
        deadline = time.monotonic() + timeout
        while not self.urls and time.monotonic() < deadline:
            time.sleep(0.01)


@pytest.fixture
def tool_endpoint(monkeypatch):
    endpoint = ToolEndpoint()
    monkeypatch.setattr(tool_metering, "httpx", endpoint)
    tool_metering.configure(metering_url="http://tools.test", api_key="rev_mk_test")
    yield endpoint
    tool_metering.configure()


class TestWorkersWhileOpen:
    @pytest.fixture
    def pool(self):
        pool = MeteringWorkerPool(2, 100, lambda task: None)
        yield pool
        pool.stop(timeout=5)

    def submit(self, pool):
        async def meter():
            submit_ai_event("completion", dict(COMPLETION_ARGS))

        task = MeteringTask(meter(), contextvars.copy_context())
        pool.submit(task)
        return task

    def test_records_are_buffered_without_a_send(self, pool, buffer, mock_revenium_client):
        open_circuit(delivery_circuit.get_circuit())
        delivery_circuit.get_circuit()._next_probe_at = time.monotonic() + 3600

        tasks = [self.submit(pool) for _ in range(5)]
        for task in tasks:
            task.join(5)

        mock_revenium_client.ai.create_completion.assert_not_called()
        assert [event.kind for event in buffer._events] == ["ai"] * 5
        assert buffer.stats()["total_buffered"] == 5

    def test_one_record_per_probe_interval_is_still_sent(self, pool, buffer, mock_revenium_client):
        mock_revenium_client.ai.create_completion.side_effect = connection_error()
        open_circuit(delivery_circuit.get_circuit())
        delivery_circuit.get_circuit()._next_probe_at = 0.0

        tasks = [self.submit(pool) for _ in range(5)]
        for task in tasks:
            task.join(5)

        assert mock_revenium_client.ai.create_completion.call_count == 1
        assert buffer.stats()["size"] == 5

    def test_a_tool_event_is_sent_not_buffered(self, buffer, mock_revenium_client, tool_endpoint):
        mock_revenium_client.ai.create_completion.side_effect = connection_error()
        for _ in range(delivery_circuit.FAILURES_TO_OPEN):
            submit_ai_event("completion", dict(COMPLETION_ARGS))
        assert delivery_circuit.get_circuit().is_open()

        tool_metering.report_tool_call(tool_id="search", duration_ms=5)
        tool_endpoint.wait_for_post()

        assert tool_endpoint.urls == ["http://tools.test/meter/v2/tool/events"]
        assert [event.kind for event in buffer._events] == ["ai"] * delivery_circuit.FAILURES_TO_OPEN
        assert delivery_circuit.get_circuit().is_open()

    def test_a_tool_event_does_not_spend_the_probe(self, pool, buffer, mock_revenium_client):
        open_circuit(delivery_circuit.get_circuit())
        delivery_circuit.get_circuit()._next_probe_at = 0.0

        async def tool_event():
            pass

        tool_task = pool.new_task(tool_event(), contextvars.copy_context(), gated_by_circuit=False)
        pool.submit(tool_task)
        tool_task.join(5)
        self.submit(pool).join(5)

        mock_revenium_client.ai.create_completion.assert_called_once()

    def test_the_first_success_reopens_delivery(self, pool, buffer, mock_revenium_client):
        open_circuit(delivery_circuit.get_circuit())
        delivery_circuit.get_circuit()._next_probe_at = 0.0
        self.submit(pool).join(5)

        tasks = [self.submit(pool) for _ in range(3)]
        for task in tasks:
            task.join(5)

        assert mock_revenium_client.ai.create_completion.call_count == 4
        assert buffer.stats()["size"] == 0


class TestSubmissionFeedsTheCircuit:
    def test_retryable_failures_open_it(self, buffer, mock_revenium_client):
        mock_revenium_client.ai.create_completion.side_effect = connection_error()

        for _ in range(delivery_circuit.FAILURES_TO_OPEN):
            submit_ai_event("completion", dict(COMPLETION_ARGS))

        assert delivery_circuit.get_circuit().is_open()
        assert buffer.stats()["size"] == delivery_circuit.FAILURES_TO_OPEN

    def test_permanent_failures_do_not(self, buffer, mock_revenium_client):
        request = httpx.Request("POST", "http://metering.test/v2/ai/completions")
        from revenium_middleware._metering._exceptions import APIStatusError
        mock_revenium_client.ai.create_completion.side_effect = APIStatusError(
            "bad", response=httpx.Response(422, request=request), body=None)

        for _ in range(delivery_circuit.FAILURES_TO_OPEN):
            with pytest.raises(APIStatusError):
                submit_ai_event("completion", dict(COMPLETION_ARGS))

        assert not delivery_circuit.get_circuit().is_open()

    def test_the_success_that_closes_it_wakes_the_buffer_replay(self, mock_revenium_client, monkeypatch):
        replayed = []
        buf = MeteringBuffer(flush_interval=3600, replay_fn=lambda event, timeout: replayed.append(event))
        monkeypatch.setattr(metering_buffer, "_buffer", buf)
        buf.push("ai", {"seq": "held back"})
        open_circuit(delivery_circuit.get_circuit())

        submit_ai_event("completion", dict(COMPLETION_ARGS))

        deadline = time.monotonic() + 5
        while not replayed and time.monotonic() < deadline:
            time.sleep(0.01)
        assert [event.payload["seq"] for event in replayed] == ["held back"]
