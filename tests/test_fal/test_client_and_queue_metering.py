"""One metering record per fal generation, whichever client or queue call started it (BACK-3608).

fal is stubbed at the httpx transport of each client, so these tests run the
real SyncClient / AsyncClient / request-handle code under our wraps.
"""
import asyncio
import datetime
import itertools
import json

import httpx
import pytest

pytest.importorskip("fal_client")

import fal_client  # noqa: E402

import revenium_middleware.fal  # noqa: E402,F401
from revenium_middleware import revenium_meter, revenium_metadata  # noqa: E402
from revenium_middleware._core.metering_pool import wait_until_idle  # noqa: E402
from revenium_middleware.fal._call import FalCall  # noqa: E402
from revenium_middleware.fal._queue import QueuedJobs, application_from_queue_url  # noqa: E402

APP = "fal-ai/flux/schnell"
MODEL = "fal_ai/" + APP
ARGUMENTS = {"prompt": "cat", "num_images": 2}
RESULT = {"images": [{"url": "https://stub/a.png", "width": 512, "height": 512},
                     {"url": "https://stub/b.png", "width": 512, "height": 512}]}
STREAM_EVENTS = ({"status": "IN_PROGRESS"}, RESULT)
REQUESTS_URL = "https://queue.fal.run/" + APP + "/requests/"


class FalStub:
    """The run, stream and queue hosts, with one IN_PROGRESS poll before a job completes."""

    def __init__(self):
        self._ids = itertools.count(1)
        self._polls = {}
        self.result_fetches = 0

    def __call__(self, request):
        if request.url.host != "queue.fal.run":
            return self._run(request)
        if request.method == "POST":
            return httpx.Response(200, json=self._submitted())
        request_id = request.url.path.rpartition("/requests/")[2].split("/")[0]
        if request.url.path.endswith("/status"):
            return httpx.Response(200, json=self._status(request_id))
        self.result_fetches += 1
        return httpx.Response(200, json=RESULT)

    @staticmethod
    def _run(request):
        if request.url.path.endswith("/stream"):
            body = b"".join(f"data: {json.dumps(event)}\n\n".encode() for event in STREAM_EVENTS)
            return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=body)
        return httpx.Response(200, json=RESULT)

    def _submitted(self):
        request_id = f"job-{next(self._ids)}-{id(self)}"
        base = REQUESTS_URL + request_id
        return {"request_id": request_id, "response_url": base, "status_url": base + "/status",
                "cancel_url": base + "/cancel"}

    def _status(self, request_id):
        self._polls[request_id] = self._polls.get(request_id, 0) + 1
        if self._polls[request_id] == 1:
            return {"status": "IN_PROGRESS", "logs": []}
        return {"status": "COMPLETED", "logs": [], "metrics": {}}


class _Awaited:
    """AsyncClient caches ``_client`` as an awaitable, not the httpx client itself."""

    def __init__(self, value):
        self._value = value

    def __await__(self):
        if False:
            yield
        return self._value


@pytest.fixture
def stub():
    return FalStub()


@pytest.fixture
def sync_client(stub):
    client = fal_client.SyncClient(key="stub:stub")
    client.__dict__["_client"] = httpx.Client(transport=httpx.MockTransport(stub))
    return client


@pytest.fixture
def async_client(stub):
    client = fal_client.AsyncClient(key="stub:stub")
    client.__dict__["_client"] = _Awaited(httpx.AsyncClient(transport=httpx.MockTransport(stub)))
    return client


@pytest.fixture
def default_clients(stub, monkeypatch):
    monkeypatch.setenv("FAL_KEY", "stub:stub")
    monkeypatch.setitem(fal_client.sync_client.__dict__, "_client",
                        httpx.Client(transport=httpx.MockTransport(stub)))
    monkeypatch.setitem(fal_client.async_client.__dict__, "_client",
                        _Awaited(httpx.AsyncClient(transport=httpx.MockTransport(stub))))


@pytest.fixture
def payloads(mock_revenium_client):
    def recorded():
        wait_until_idle(10)
        return [call.kwargs for call in mock_revenium_client.ai.create_image.call_args_list]
    return recorded


def _assert_one_image_payload(payloads, **expected):
    recorded = payloads()
    assert len(recorded) == 1, recorded
    payload = recorded[0]
    assert payload["model"] == MODEL
    assert payload["actual_image_count"] == 2
    assert "input_token_count" not in payload
    for field, value in expected.items():
        assert payload[field] == value, field
    return payload


class TestExplicitClients:
    def test_sync_client_run(self, sync_client, payloads):
        sync_client.run(APP, arguments=ARGUMENTS)
        _assert_one_image_payload(payloads, requested_image_count=2)

    def test_async_client_run(self, async_client, payloads):
        asyncio.run(async_client.run(APP, arguments=ARGUMENTS))
        _assert_one_image_payload(payloads, requested_image_count=2)

    def test_positional_arguments_are_read(self, sync_client, payloads):
        sync_client.run(APP, ARGUMENTS)
        _assert_one_image_payload(payloads, requested_image_count=2)

    def test_usage_metadata_is_ours_and_not_forwarded_to_fal(self, sync_client, payloads):
        sync_client.run(APP, arguments=ARGUMENTS, usage_metadata={"trace_id": "t-run"})
        _assert_one_image_payload(payloads, trace_id="t-run")

    def test_sync_client_subscribe(self, sync_client, payloads):
        assert sync_client.subscribe(APP, arguments=ARGUMENTS) == RESULT
        _assert_one_image_payload(payloads)

    def test_async_client_subscribe(self, async_client, payloads):
        assert asyncio.run(async_client.subscribe(APP, arguments=ARGUMENTS)) == RESULT
        _assert_one_image_payload(payloads)


class TestModuleLevelControls:
    """The module-level functions reach the wrapped class methods once, not a second wrap."""

    def test_run(self, default_clients, payloads):
        fal_client.run(APP, arguments=ARGUMENTS)
        _assert_one_image_payload(payloads)

    def test_run_async(self, default_clients, payloads):
        asyncio.run(fal_client.run_async(APP, arguments=ARGUMENTS))
        _assert_one_image_payload(payloads)

    def test_subscribe(self, default_clients, payloads):
        fal_client.subscribe(APP, arguments=ARGUMENTS)
        _assert_one_image_payload(payloads)

    def test_subscribe_async(self, default_clients, payloads):
        asyncio.run(fal_client.subscribe_async(APP, arguments=ARGUMENTS))
        _assert_one_image_payload(payloads)

    @pytest.mark.parametrize("alias, client, method", [
        ("run", "sync_client", "run"), ("subscribe", "sync_client", "subscribe"),
        ("submit", "sync_client", "submit"), ("stream", "sync_client", "stream"),
        ("run_async", "async_client", "run"), ("subscribe_async", "async_client", "subscribe"),
        ("submit_async", "async_client", "submit"), ("stream_async", "async_client", "stream"),
    ])
    def test_alias_is_the_default_clients_wrapped_method(self, alias, client, method):
        assert getattr(fal_client, alias) == getattr(getattr(fal_client, client), method)


class TestQueueFlow:
    def test_submit_and_status_emit_nothing_and_result_emits_once(self, sync_client, payloads):
        handle = sync_client.submit(APP, arguments=ARGUMENTS, usage_metadata={"trace_id": "t-queue"})
        sync_client.status(APP, handle.request_id)
        sync_client.status(APP, handle.request_id)
        assert payloads() == []

        assert sync_client.result(APP, handle.request_id) == RESULT
        _assert_one_image_payload(payloads, trace_id="t-queue", requested_image_count=2)

    def test_async_submit_status_result(self, async_client, payloads):
        async def queue_flow():
            handle = await async_client.submit(APP, arguments=ARGUMENTS)
            await async_client.status(APP, handle.request_id)
            assert payloads() == []
            return await async_client.result(APP, handle.request_id)

        assert asyncio.run(queue_flow()) == RESULT
        _assert_one_image_payload(payloads, requested_image_count=2)

    def test_handle_get(self, sync_client, payloads):
        assert sync_client.submit(APP, arguments=ARGUMENTS).get() == RESULT
        _assert_one_image_payload(payloads)

    def test_async_handle_get(self, async_client, payloads):
        async def queue_flow():
            return await (await async_client.submit(APP, arguments=ARGUMENTS)).get()

        assert asyncio.run(queue_flow()) == RESULT
        _assert_one_image_payload(payloads)

    def test_module_level_queue_functions(self, default_clients, payloads):
        request_id = fal_client.submit(APP, arguments=ARGUMENTS).request_id
        fal_client.status(APP, request_id)
        fal_client.result(APP, request_id)
        _assert_one_image_payload(payloads)

    def test_fetching_a_result_again_does_not_bill_the_job_again(self, sync_client, stub, payloads):
        handle = sync_client.submit(APP, arguments=ARGUMENTS)
        handle.get()
        handle.get()
        sync_client.result(APP, handle.request_id)
        assert stub.result_fetches == 3
        _assert_one_image_payload(payloads)

    def test_a_job_whose_result_is_never_fetched_emits_nothing(self, sync_client, payloads):
        handle = sync_client.submit(APP, arguments=ARGUMENTS)
        sync_client.status(APP, handle.request_id)
        assert payloads() == []

    def test_a_result_fetched_by_id_without_a_tracked_submit_is_metered_once(self, sync_client, payloads):
        assert sync_client.result(APP, "submitted-elsewhere") == RESULT
        sync_client.result(APP, "submitted-elsewhere")
        _assert_one_image_payload(payloads)

    def test_a_handle_built_from_a_request_id_is_metered_from_its_queue_url(self, sync_client, payloads):
        handle = fal_client.SyncRequestHandle.from_request_id(sync_client._client, APP, "handle-only")
        handle.get()
        recorded = payloads()
        assert len(recorded) == 1
        assert recorded[0]["model"] == "fal_ai/fal-ai/flux"


class TestSubscribeIsMeteredOnce:
    def test_the_callers_on_enqueue_still_runs(self, sync_client, payloads):
        enqueued = []
        sync_client.subscribe(APP, arguments=ARGUMENTS, on_enqueue=enqueued.append)
        assert len(enqueued) == 1
        _assert_one_image_payload(payloads)

    def test_an_awaitable_on_enqueue_is_still_awaited(self, async_client, payloads):
        enqueued = []

        async def on_enqueue(request_id):
            enqueued.append(request_id)

        asyncio.run(async_client.subscribe(APP, arguments=ARGUMENTS, on_enqueue=on_enqueue))
        assert len(enqueued) == 1
        _assert_one_image_payload(payloads)

    def test_on_queue_update_polling_does_not_add_a_record(self, sync_client, payloads):
        updates = []
        sync_client.subscribe(APP, arguments=ARGUMENTS, on_queue_update=updates.append)
        assert len(updates) >= 2
        _assert_one_image_payload(payloads)

    def test_client_timeout_keeps_the_callers_metadata(self, sync_client, payloads):
        """With client_timeout fal submits on its executor thread, outside the caller's context."""
        @revenium_metadata(trace_id="t-scoped")
        def generate():
            return sync_client.subscribe(APP, arguments=ARGUMENTS, client_timeout=30)

        generate()
        _assert_one_image_payload(payloads, trace_id="t-scoped")

    def test_client_timeout_inside_revenium_meter_is_metered_under_selective_metering(
            self, sync_client, payloads, monkeypatch):
        monkeypatch.setenv("REVENIUM_SELECTIVE_METERING", "true")

        @revenium_meter()
        def generate():
            return sync_client.subscribe(APP, arguments=ARGUMENTS, client_timeout=30)

        generate()
        _assert_one_image_payload(payloads)

    def test_a_foreign_job_polled_outside_scope_is_metered_when_fetched_inside(
            self, sync_client, payloads, monkeypatch):
        monkeypatch.setenv("REVENIUM_SELECTIVE_METERING", "true")
        sync_client.status(APP, "foreign-polled-outside")

        @revenium_meter()
        def fetch():
            return sync_client.result(APP, "foreign-polled-outside")

        fetch()
        _assert_one_image_payload(payloads)

    def test_a_foreign_job_polled_inside_scope_is_not_metered_when_fetched_outside(
            self, sync_client, payloads, monkeypatch):
        monkeypatch.setenv("REVENIUM_SELECTIVE_METERING", "true")

        @revenium_meter()
        def poll():
            return sync_client.status(APP, "foreign-polled-inside")

        poll()
        sync_client.result(APP, "foreign-polled-inside")
        assert payloads() == []

    def test_selective_metering_outside_a_metered_function_emits_nothing(self, sync_client, payloads,
                                                                         monkeypatch):
        monkeypatch.setenv("REVENIUM_SELECTIVE_METERING", "true")
        sync_client.subscribe(APP, arguments=ARGUMENTS, usage_metadata={"trace_id": "dropped"})
        sync_client.run(APP, arguments=ARGUMENTS, usage_metadata={"trace_id": "dropped"})
        assert payloads() == []


class TestStreams:
    def test_sync_client_stream_is_metered_once_after_the_last_event(self, sync_client, payloads):
        events = sync_client.stream(APP, arguments=ARGUMENTS)
        assert next(events) == {"status": "IN_PROGRESS"}
        assert payloads() == []
        assert list(events) == [RESULT]
        _assert_one_image_payload(payloads)

    def test_async_client_stream_is_an_async_iterator_metered_once(self, async_client, payloads):
        async def drain():
            return [event async for event in async_client.stream(APP, arguments=ARGUMENTS)]

        assert asyncio.run(drain()) == [{"status": "IN_PROGRESS"}, RESULT]
        _assert_one_image_payload(payloads)

    def test_module_level_stream_async_is_an_async_iterator(self, default_clients, payloads):
        async def drain():
            return [event async for event in fal_client.stream_async(APP, arguments=ARGUMENTS)]

        assert asyncio.run(drain())[-1] == RESULT
        _assert_one_image_payload(payloads)


def _call(application="app"):
    return FalCall(application=application, arguments={}, usage_metadata={},
                   request_time_dt=datetime.datetime.now(datetime.timezone.utc),
                   transaction_id="fal-x", metered=True)


class TestQueuedJobs:
    def test_a_job_is_claimed_once(self):
        jobs = QueuedJobs()
        tracked = _call("tracked")
        jobs.track("r1", tracked)
        assert jobs.claim("r1", _call("fallback")) is tracked
        assert jobs.claim("r1", _call("fallback")) is None

    def test_an_untracked_job_is_claimed_with_the_fallback_once(self):
        jobs = QueuedJobs()
        fallback = _call("fallback")
        assert jobs.claim("r1", fallback) is fallback
        assert jobs.claim("r1", fallback) is None

    def test_a_remembered_application_is_forgotten_once_the_job_is_claimed(self):
        jobs = QueuedJobs()
        jobs.remember_application("r1", "by-id")
        assert jobs.application_of("r1") == "by-id"
        jobs.claim("r1", _call())
        assert jobs.application_of("r1") is None

    def test_remembered_applications_are_bounded(self):
        jobs = QueuedJobs(application_capacity=1)
        jobs.remember_application("r1", "a")
        jobs.remember_application("r2", "b")
        assert jobs.application_of("r1") is None
        assert jobs.application_of("r2") == "b"

    def test_the_oldest_jobs_are_evicted(self):
        jobs = QueuedJobs(tracked_capacity=2, claimed_capacity=2)
        for request_id in ("r1", "r2", "r3"):
            jobs.track(request_id, _call(request_id))
        fallback = _call("fallback")
        assert jobs.claim("r1", fallback) is fallback
        assert jobs.claim("r3", fallback).application == "r3"

    @pytest.mark.parametrize("url, application", [
        ("https://queue.fal.run/fal-ai/flux/requests/abc", "fal-ai/flux"),
        ("https://queue.fal.run/fal-ai/flux/dev/requests/abc/status", "fal-ai/flux/dev"),
        ("https://queue.fal.run/nothing-here", "unknown"),
        ("", "unknown"),
    ])
    def test_application_from_queue_url(self, url, application):
        assert application_from_queue_url(url) == application
