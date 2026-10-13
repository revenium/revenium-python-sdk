"""One metering record per fal generation, whichever client or queue call started it (BACK-3608).

fal is stubbed at the httpx transport of each client, so these tests run the
real SyncClient / AsyncClient / request-handle code under our wraps.
"""
import asyncio
import datetime
import gc
import itertools
import json
import logging

import httpx
import pytest

pytest.importorskip("fal_client")

import fal_client  # noqa: E402

import revenium_middleware.fal  # noqa: E402,F401
from revenium_middleware import idempotency_key  # noqa: E402
from revenium_middleware._core import delivery_circuit, metering_buffer  # noqa: E402
from revenium_middleware._core.context import injected_metadata_scope  # noqa: E402
from revenium_middleware._core.metering import recorded_under  # noqa: E402
from revenium_middleware._core.metering_buffer import MeteringBuffer  # noqa: E402
from revenium_middleware._metering._exceptions import APIStatusError  # noqa: E402
import revenium_middleware.fal.middleware as fal_middleware  # noqa: E402
from revenium_middleware.fal import _call as fal_call_module, _metering as fal_metering  # noqa: E402
from revenium_middleware import JobContext, revenium_meter, revenium_metadata  # noqa: E402
from revenium_middleware._core.metering_pool import wait_until_idle  # noqa: E402
from revenium_middleware.fal._call import FalCall, billing_arguments  # noqa: E402
from revenium_middleware.fal._queue import QueuedJobs, application_from_queue_url, job_transaction_id  # noqa: E402

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
        ("result", "sync_client", "result"), ("result_async", "async_client", "result"),
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


_FIRST_EVENT = f"data: {json.dumps(STREAM_EVENTS[0])}\n\n".encode()


def _assert_one_abandoned_stream_payload(payloads):
    recorded = payloads()
    assert len(recorded) == 1, recorded
    assert recorded[0]["model"] == MODEL
    assert recorded[0]["actual_image_count"] == 1


def _dropping_body(sent):
    yield sent
    raise httpx.ReadError("connection dropped")


async def _dropping_async_body(sent):
    yield sent
    raise httpx.ReadError("connection dropped")


def _read_broken_stream(mode, sent, seen):
    """Stream from a fal host that sends ``sent`` and then drops the connection."""
    headers = {"content-type": "text/event-stream"}
    if mode == "sync":
        client = fal_client.SyncClient(key="stub:stub")
        client.__dict__["_client"] = httpx.Client(transport=httpx.MockTransport(
            lambda request: httpx.Response(200, headers=headers, content=_dropping_body(sent))))
        for event in client.stream(APP, arguments=ARGUMENTS):
            seen.append(event)
        return

    async def read():
        client = fal_client.AsyncClient(key="stub:stub")
        client.__dict__["_client"] = _Awaited(httpx.AsyncClient(transport=httpx.MockTransport(
            lambda request: httpx.Response(200, headers=headers, content=_dropping_async_body(sent)))))
        async for event in client.stream(APP, arguments=ARGUMENTS):
            seen.append(event)

    asyncio.run(read())


class TestStreams:
    def test_sync_client_stream_is_metered_once_after_the_last_event(self, sync_client, payloads):
        events = sync_client.stream(APP, arguments=ARGUMENTS)
        assert next(events) == {"status": "IN_PROGRESS"}
        assert payloads() == []
        assert list(events) == [RESULT]
        _assert_one_image_payload(payloads)

        events.close()
        del events
        gc.collect()
        _assert_one_image_payload(payloads)

    def test_a_sync_stream_closed_after_its_first_event_is_metered_once(self, sync_client, payloads):
        events = sync_client.stream(APP, arguments=ARGUMENTS)
        assert next(events) == {"status": "IN_PROGRESS"}

        events.close()

        _assert_one_abandoned_stream_payload(payloads)

    def test_a_sync_stream_broken_out_of_is_metered_once(self, sync_client, payloads):
        for _event in sync_client.stream(APP, arguments=ARGUMENTS):
            break

        _assert_one_abandoned_stream_payload(payloads)

    def test_an_async_stream_closed_with_aclose_is_metered_once(self, async_client, payloads):
        async def first_event_then_close():
            events = async_client.stream(APP, arguments=ARGUMENTS)
            assert await events.__anext__() == {"status": "IN_PROGRESS"}
            await events.aclose()

        asyncio.run(first_event_then_close())

        _assert_one_abandoned_stream_payload(payloads)

    @pytest.mark.parametrize("mode", ["sync", "async"])
    def test_a_stream_that_drops_part_way_is_metered_once_and_the_caller_sees_the_error(self, payloads, mode):
        seen = []
        with pytest.raises(httpx.ReadError):
            _read_broken_stream(mode, _FIRST_EVENT, seen)

        assert seen == [{"status": "IN_PROGRESS"}]
        _assert_one_abandoned_stream_payload(payloads)

    @pytest.mark.parametrize("mode", ["sync", "async"])
    def test_a_stream_that_fails_before_its_first_event_sends_nothing(self, payloads, mode):
        with pytest.raises(httpx.ReadError):
            _read_broken_stream(mode, b"", [])

        assert payloads() == []

    def test_an_abandoned_stream_keeps_the_job_it_started_in(self, sync_client, payloads):
        with JobContext(job_id="job-a"):
            events = sync_client.stream(APP, arguments=ARGUMENTS)
            next(events)
        with JobContext(job_id="job-b"):
            events.close()

        recorded = payloads()
        assert [(payload.get("extra_body") or {}).get("agenticJobId") for payload in recorded] == ["job-a"]

    def test_an_abandoned_async_stream_keeps_the_job_it_started_in(self, async_client, payloads):
        async def start_in_one_job_and_read_in_another():
            with JobContext(job_id="job-a"):
                events = async_client.stream(APP, arguments=ARGUMENTS)
            with JobContext(job_id="job-b"):
                await events.__anext__()
                await events.aclose()

        asyncio.run(start_in_one_job_and_read_in_another())

        recorded = payloads()
        assert [(payload.get("extra_body") or {}).get("agenticJobId") for payload in recorded] == ["job-a"]

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

    def test_a_job_transaction_id_is_the_request_id(self):
        assert job_transaction_id("abc-123") == "fal-abc-123"

    @pytest.mark.parametrize("request_id", ["r" * 300, "r" * 252], ids=["300 characters", "252 characters"])
    def test_an_overlong_request_id_is_hashed_into_a_valid_key(self, request_id):
        key = job_transaction_id(request_id)
        assert len(key) <= 255
        assert key.startswith("fal-") and len(key) == len("fal-") + 64
        assert all(character in "0123456789abcdef" for character in key[len("fal-"):])
        assert job_transaction_id(request_id) == key

    def test_a_request_id_with_a_space_is_hashed(self):
        key = job_transaction_id("job 1")
        assert " " not in key
        assert key.startswith("fal-") and len(key) == len("fal-") + 64

    @pytest.mark.parametrize("url, application", [
        ("https://queue.fal.run/fal-ai/flux/requests/abc", "fal-ai/flux"),
        ("https://queue.fal.run/fal-ai/flux/dev/requests/abc/status", "fal-ai/flux/dev"),
        ("https://queue.fal.run/nothing-here", "unknown"),
        ("", "unknown"),
    ])
    def test_application_from_queue_url(self, url, application):
        assert application_from_queue_url(url) == application


INLINE_IMAGE = "data:image/png;base64," + "A" * 4096
INLINE_AUDIO = "data:audio/wav;base64," + "A" * 4096


class TestPendingJobs:
    """A queued job waits in memory until its result is fetched, so it keeps only what its record reads (BACK-4380)."""

    def test_a_pending_job_holds_only_what_its_record_needs(self, sync_client, payloads):
        arguments = {"prompt": "x" * 100_000, "image_url": INLINE_IMAGE, "num_images": 2,
                     "image_size": {"width": 768, "height": 768}, "audio_url": INLINE_AUDIO}
        handle = sync_client.submit(APP, arguments=arguments)

        pending = fal_middleware.queued_jobs.pending(handle.request_id)
        assert "prompt" not in pending.arguments
        assert "image_url" not in pending.arguments
        assert pending.arguments["audio_url"] is True

        handle.get()
        _assert_one_image_payload(payloads, requested_image_count=2)

    @pytest.mark.parametrize("extract, result, arguments", [
        ("extract_image_fields", {"images": [{"url": "https://stub/a.png"}]},
         {"prompt": "cat", "image_url": INLINE_IMAGE, "num_images": 3,
          "image_size": {"width": 768, "height": 768}, "quality": "hd", "aspect_ratio": "16:9"}),
        ("extract_image_fields", {"images": [{"url": "https://stub/a.png"}]},
         {"prompt": "cat", "image_size": "square_hd"}),
        ("extract_video_fields", {"video": {"url": "https://stub/a.mp4"}},
         {"prompt": "cat", "image_url": INLINE_IMAGE, "duration": "5", "resolution": "720p", "fps": 24,
          "aspect_ratio": "16:9"}),
        ("extract_audio_fields", {"audio": {"duration": 1.5}}, {"text": "hello there"}),
        ("extract_audio_fields", {"audio": {"duration": 1.5}}, {"input": "hello there", "audio_url": ""}),
        ("extract_audio_fields", {"audio": {"duration": 1.5}}, {"text": "hello there", "audio_url": None}),
        ("extract_audio_fields", {"text": "heard"}, {"audio_url": INLINE_AUDIO}),
        ("extract_audio_fields", {"text": "heard"}, {"audio": INLINE_AUDIO}),
    ])
    def test_billing_arguments_give_the_same_record_fields(self, extract, result, arguments):
        extractor = getattr(fal_metering, extract)
        assert extractor(result, billing_arguments(arguments)) == extractor(result, arguments)

    def test_run_and_stream_records_still_see_full_arguments(self, sync_client, monkeypatch):
        seen = []
        monkeypatch.setattr(fal_call_module, "handle_metering", lambda **kwargs: seen.append(kwargs["arguments"]))
        arguments = {"prompt": "cat", "image_url": INLINE_IMAGE, "num_images": 2}

        sync_client.run(APP, arguments=arguments)
        list(sync_client.stream(APP, arguments=arguments))

        assert seen == [arguments, arguments]


TAGS = {"organization_name": "org-tagged", "product_name": "product-tagged", "trace_id": "trace-tagged",
        "agent": "agent-tagged", "task_type": "task-tagged", "subscriber": {"id": "subscriber-tagged"}}


@pytest.fixture
def fresh_process(monkeypatch):
    """Stand in for another worker process: a job ledger that never saw this process's submits or fetches."""
    def start(jobs=None):
        monkeypatch.setattr(fal_middleware, "queued_jobs", jobs if jobs is not None else QueuedJobs())
    return start


@pytest.fixture
def sent_ids(payloads):
    """The transaction id and Idempotency-Key of each record sent, in order."""
    def recorded():
        return [(payload["transaction_id"], payload["extra_headers"]["Idempotency-Key"]) for payload in payloads()]
    return recorded


class TestOneIdPerJob:
    """Every fetch of one queued job is sent under one id, so Revenium stores it once (BACK-4380)."""

    def test_a_result_fetched_again_by_a_fresh_ledger_is_sent_under_the_jobs_own_id(
            self, sync_client, sent_ids, fresh_process):
        handle = sync_client.submit(APP, arguments=ARGUMENTS, usage_metadata=TAGS)
        handle.get()
        fresh_process()
        sync_client.result(APP, handle.request_id)

        job_id = "fal-" + handle.request_id
        assert sent_ids() == [(job_id, job_id), (job_id, job_id)]

    def test_a_result_fetched_after_the_ledger_forgot_it_is_sent_under_the_jobs_own_id(
            self, sync_client, sent_ids, fresh_process):
        fresh_process(QueuedJobs(claimed_capacity=1))
        first = sync_client.submit(APP, arguments=ARGUMENTS)
        first.get()
        sync_client.submit(APP, arguments=ARGUMENTS).get()
        first.get()

        job_id = "fal-" + first.request_id
        recorded = sent_ids()
        assert len(recorded) == 3
        assert recorded.count((job_id, job_id)) == 2

    @pytest.mark.parametrize("client_timeout", [None, 30])
    def test_a_subscribed_job_is_sent_under_its_request_id(self, sync_client, sent_ids, client_timeout):
        enqueued = []
        sync_client.subscribe(APP, arguments=ARGUMENTS, on_enqueue=enqueued.append, client_timeout=client_timeout)

        assert sent_ids() == [("fal-" + enqueued[0], "fal-" + enqueued[0])]

    @pytest.mark.parametrize("client_timeout", [None, 30])
    def test_an_async_subscribed_job_is_sent_under_its_request_id(self, async_client, sent_ids, client_timeout):
        enqueued = []
        asyncio.run(async_client.subscribe(APP, arguments=ARGUMENTS, on_enqueue=enqueued.append,
                                           client_timeout=client_timeout))

        assert sent_ids() == [("fal-" + enqueued[0], "fal-" + enqueued[0])]

    def test_a_job_fetched_inside_an_idempotency_block_keeps_its_own_key(self, sync_client, sent_ids):
        handle = sync_client.submit(APP, arguments=ARGUMENTS)
        with idempotency_key("order-42"):
            handle.get()
            sync_client.run(APP, arguments=ARGUMENTS)

        keys = {key for _transaction_id, key in sent_ids()}
        assert keys == {"fal-" + handle.request_id, "order-42"}

    def test_a_refused_repeat_does_not_reach_the_host_or_the_buffer(
            self, sync_client, sent_ids, fresh_process, mock_revenium_client, monkeypatch):
        buffer = MeteringBuffer(flush_interval=3600, replay_fn=lambda event, timeout: None)
        monkeypatch.setattr(metering_buffer, "_buffer", buffer)
        circuit_failures = []
        monkeypatch.setattr(delivery_circuit.get_circuit(), "record_failure",
                            lambda: circuit_failures.append(True))
        request = httpx.Request("POST", "https://api.test.revenium.ai/meter/v2/ai/images")
        refusal = APIStatusError("idempotency_key_mismatch", response=httpx.Response(409, request=request), body=None)
        create_image = mock_revenium_client.ai.create_image
        create_image.side_effect = [create_image.return_value, refusal]

        handle = sync_client.submit(APP, arguments=ARGUMENTS)
        handle.get()
        fresh_process()

        assert sync_client.result(APP, handle.request_id) == RESULT
        job_id = "fal-" + handle.request_id
        assert sent_ids() == [(job_id, job_id), (job_id, job_id)]
        assert buffer.stats()["size"] == 0
        assert circuit_failures == []


def _assert_tagged(payload, tags):
    assert payload["organization_name"] == tags["organization_name"]
    assert payload["product_name"] == tags["product_name"]
    assert payload["trace_id"] == tags["trace_id"]
    assert payload["agent"] == tags["agent"]
    assert payload["task_type"] == tags["task_type"]
    assert payload["subscriber"]["id"] == tags["subscriber"]["id"]


def _fetch_with_sync_client_result(sync_client, async_client, request_id, tags):
    return sync_client.result(APP, request_id, usage_metadata=tags)


def _fetch_with_async_client_result(sync_client, async_client, request_id, tags):
    return asyncio.run(async_client.result(APP, request_id, usage_metadata=tags))


def _fetch_with_module_result(sync_client, async_client, request_id, tags):
    return fal_client.result(APP, request_id, usage_metadata=tags)


def _fetch_with_module_result_async(sync_client, async_client, request_id, tags):
    return asyncio.run(fal_client.result_async(APP, request_id, usage_metadata=tags))


def _fetch_with_handle_get(sync_client, async_client, request_id, tags):
    return sync_client.get_handle(APP, request_id).get(usage_metadata=tags)


def _fetch_with_async_handle_get(sync_client, async_client, request_id, tags):
    async def fetch():
        return await (await async_client.get_handle(APP, request_id)).get(usage_metadata=tags)
    return asyncio.run(fetch())


FETCHES_WITH_TAGS = [_fetch_with_sync_client_result, _fetch_with_async_client_result, _fetch_with_module_result,
                     _fetch_with_module_result_async, _fetch_with_handle_get, _fetch_with_async_handle_get]

UNTAGGED_JOB_WARNING = "has no customer tags"


def _untagged_job_warnings(caplog):
    return [record for record in caplog.records
            if record.name == "revenium_middleware.fal" and record.levelno == logging.WARNING
            and UNTAGGED_JOB_WARNING in record.getMessage()]


class TestJobsFetchedElsewhere:
    """A job fetched by a process that did not submit it is tagged from the fetch (BACK-4380).

    A fresh ledger stands in for that other process: the module-level job ledger the wraps read is replaced.
    """

    @pytest.fixture(autouse=True)
    def _ledger_of_this_test_only(self, fresh_process):
        fresh_process()

    @pytest.mark.parametrize("fetch", FETCHES_WITH_TAGS, ids=lambda fetch: fetch.__name__[len("_fetch_with_"):])
    def test_tags_passed_to_result_are_on_a_job_this_process_never_submitted(
            self, sync_client, async_client, default_clients, payloads, fetch):
        assert fetch(sync_client, async_client, "submitted-elsewhere", TAGS) == RESULT

        _assert_tagged(_assert_one_image_payload(payloads), TAGS)

    def test_a_job_submitted_with_tags_and_fetched_by_a_fresh_ledger_with_tags_keeps_them(
            self, sync_client, payloads, fresh_process):
        handle = sync_client.submit(APP, arguments=ARGUMENTS, usage_metadata=TAGS)
        fresh_process()

        sync_client.result(APP, handle.request_id, usage_metadata=TAGS)

        _assert_tagged(_assert_one_image_payload(payloads), TAGS)

    def test_tags_given_at_submit_win_in_the_submitting_process(self, sync_client, payloads):
        handle = sync_client.submit(APP, arguments=ARGUMENTS, usage_metadata={"trace_id": "t-submit"})

        sync_client.result(APP, handle.request_id, usage_metadata={"trace_id": "t-fetch"})

        _assert_one_image_payload(payloads, trace_id="t-submit")

    def test_tags_given_at_fetch_fill_fields_the_submit_left_unset(self, sync_client, payloads):
        handle = sync_client.submit(APP, arguments=ARGUMENTS, usage_metadata={"trace_id": "t-submit"})

        sync_client.result(APP, handle.request_id,
                           usage_metadata={"trace_id": "t-fetch", "organization_name": "org-fetch"})

        _assert_one_image_payload(payloads, trace_id="t-submit", organization_name="org-fetch")

    def test_an_empty_tag_given_at_submit_does_not_erase_the_one_given_at_fetch(self, sync_client, payloads):
        handle = sync_client.submit(APP, arguments=ARGUMENTS,
                                    usage_metadata={"trace_id": "t-submit", "organization_name": None})

        sync_client.result(APP, handle.request_id, usage_metadata={"organization_name": "org-fetch"})

        _assert_one_image_payload(payloads, trace_id="t-submit", organization_name="org-fetch")

    def test_an_empty_subscriber_given_at_submit_does_not_erase_the_one_given_at_fetch(self, sync_client, payloads):
        handle = sync_client.submit(APP, arguments=ARGUMENTS,
                                    usage_metadata={"trace_id": "t-submit", "subscriber": {}})

        sync_client.result(APP, handle.request_id, usage_metadata={"subscriber": {"id": "sub-fetch"}})

        payload = _assert_one_image_payload(payloads, trace_id="t-submit")
        assert payload["subscriber"]["id"] == "sub-fetch"

    def test_a_job_recorded_without_tags_logs_why(self, sync_client, payloads, caplog):
        caplog.set_level(logging.WARNING, logger="revenium_middleware.fal")

        sync_client.result(APP, "untagged-job")

        _assert_one_image_payload(payloads)
        warnings = _untagged_job_warnings(caplog)
        assert len(warnings) == 1
        assert "untagged-job" in warnings[0].getMessage()
        assert APP in warnings[0].getMessage()

    def test_a_job_with_only_a_global_default_tag_still_logs_why(self, sync_client, payloads, caplog):
        caplog.set_level(logging.WARNING, logger="revenium_middleware.fal")

        with injected_metadata_scope({"environment": "prod-default"}):
            sync_client.result(APP, "default-tagged-job")

        _assert_one_image_payload(payloads)
        warnings = _untagged_job_warnings(caplog)
        assert len(warnings) == 1
        assert "default-tagged-job" in warnings[0].getMessage()
        assert "prod-default" not in warnings[0].getMessage()

    def test_no_warning_for_a_job_this_process_submitted_with_tags(self, sync_client, payloads, caplog):
        caplog.set_level(logging.WARNING, logger="revenium_middleware.fal")

        sync_client.submit(APP, arguments=ARGUMENTS, usage_metadata=TAGS).get()

        _assert_one_image_payload(payloads)
        assert _untagged_job_warnings(caplog) == []

    def test_no_warning_for_a_job_tagged_at_fetch(self, sync_client, payloads, caplog):
        caplog.set_level(logging.WARNING, logger="revenium_middleware.fal")

        sync_client.result(APP, "tagged-at-fetch", usage_metadata=TAGS)

        _assert_one_image_payload(payloads)
        assert _untagged_job_warnings(caplog) == []

    def test_no_warning_for_a_job_selective_metering_leaves_out(self, sync_client, payloads, caplog, monkeypatch):
        monkeypatch.setenv("REVENIUM_SELECTIVE_METERING", "true")
        caplog.set_level(logging.WARNING, logger="revenium_middleware.fal")

        sync_client.result(APP, "out-of-scope-job")

        assert payloads() == []
        assert _untagged_job_warnings(caplog) == []

    def test_no_warning_when_metering_is_off(self, sync_client, payloads, caplog):
        caplog.set_level(logging.WARNING, logger="revenium_middleware.fal")

        with recorded_under(None):
            sync_client.result(APP, "unmetered-job")

        assert payloads() == []
        assert _untagged_job_warnings(caplog) == []
