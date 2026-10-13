"""OpenAI background responses are metered once, when this process first sees them finish (BACK-4244).

``responses.create(background=True)`` returns a queued response with no usage.
The usage arrives later, on the ``responses.retrieve`` poll, resumed stream or
``responses.cancel`` that sees the job reach a terminal status. The real
``openai`` clients run over an httpx MockTransport that plays a background job.
"""
import asyncio
import contextlib
import contextvars
import json
import logging
import threading
import time
import uuid
from collections import OrderedDict
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import httpx
import pytest

from entry_points.caps import breached_rule, seed_rules
from openai import AsyncOpenAI, OpenAI

from revenium_middleware import JobContext, metering_key
from revenium_middleware._core import metering, metering_pool
from revenium_middleware.openai import middleware as mw

from .responses_stream_events import event_payloads, response_payload, usage_payload

USAGE = usage_payload(input_tokens=1000, output_tokens=400, reasoning_tokens=300)
FOREGROUND_USAGE = usage_payload(input_tokens=11, output_tokens=7)
TRACE_ID = "background-job-trace"
REQUEST = dict(model="o3-deep-research", input="research X", usage_metadata={"trace_id": TRACE_ID})
INTERRUPT_AFTER = 3


def _sse(events):
    return "".join(f"event: {event['type']}\ndata: {json.dumps(event)}\n\n" for event in events).encode()


class BackgroundJobFake:
    """Plays one OpenAI background job: queued at create, ``polls_until_done`` polls later completed."""

    def __init__(self, polls_until_done=2, cancel_usage=None):
        self.response_id = f"resp_bg_{uuid.uuid4().hex[:12]}"
        self.polls_until_done = polls_until_done
        self.cancel_usage = cancel_usage
        self.requests = []
        self.polls = 0

    def _response(self, status, usage=None, background=True):
        return {**response_payload(status=status, usage=usage, with_output=status == "completed"),
                "id": self.response_id, "background": background}

    def _events(self, created_status="queued", background=True, usage=USAGE):
        events = event_payloads("response.completed", usage)
        for event in events:
            if "response" in event:
                event["response"] = {**event["response"], "id": self.response_id, "background": background}
        events[0]["response"]["status"] = created_status
        return events

    def _stream(self, events):
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=_sse(events))

    def __call__(self, request):
        self.requests.append((request.method, request.url.path))
        if request.method == "GET" and "/responses/" in request.url.path:
            return self._retrieve(request)
        if request.method == "POST" and request.url.path.endswith("/cancel"):
            return httpx.Response(200, json=self._response("cancelled", self.cancel_usage))
        body = json.loads(request.content or b"{}")
        background = bool(body.get("background"))
        if body.get("stream"):
            usage = USAGE if background else FOREGROUND_USAGE
            return self._stream(self._events("queued" if background else "in_progress", background, usage))
        if background:
            return httpx.Response(200, json=self._response("queued"))
        return httpx.Response(200, json=self._response("completed", FOREGROUND_USAGE, background=False))

    def _retrieve(self, request):
        if request.url.params.get("stream") == "true":
            after = int(request.url.params.get("starting_after", -1))
            return self._stream([event for event in self._events() if event["sequence_number"] > after])
        self.polls += 1
        if self.polls < self.polls_until_done:
            return httpx.Response(200, json=self._response("in_progress"))
        return httpx.Response(200, json=self._response("completed", USAGE))


class SyncDriver:
    asynchronous = False

    def __init__(self, fake):
        self.client = OpenAI(api_key="sk-test", base_url="https://api.openai.test/v1", max_retries=0,
                             http_client=httpx.Client(transport=httpx.MockTransport(fake)))

    def run(self, call):
        return call(self.client)


class AsyncDriver:
    asynchronous = True

    def __init__(self, fake):
        async def handler(request):
            await request.aread()
            return fake(request)

        self.client = AsyncOpenAI(api_key="sk-test", base_url="https://api.openai.test/v1", max_retries=0,
                                  http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
        self.loop = asyncio.new_event_loop()

    def run(self, call):
        result = call(self.client)
        if asyncio.iscoroutine(result):
            return self.loop.run_until_complete(result)
        return result


DRIVERS = [pytest.param(SyncDriver, id="sync"), pytest.param(AsyncDriver, id="async")]


def _run_coro_to_completion(coro):
    # Copies the caller's context at submit time, as run_async_in_thread does.
    context = contextvars.copy_context()
    thread = threading.Thread(target=lambda: context.run(asyncio.run, coro))
    thread.start()
    thread.join()
    return thread


@pytest.fixture
def payloads():
    recorded = []

    def record(kind, args):
        recorded.append(args)
        return SimpleNamespace(id="evt-test")

    with patch.object(mw, "current_client", lambda: object()), \
            patch.object(mw, "submit_ai_event", side_effect=record), \
            patch.object(mw, "run_async_in_thread", side_effect=_run_coro_to_completion):
        yield recorded


def _create(**extra):
    return lambda client: client.responses.create(background=True, **REQUEST, **extra)


def _retrieve(response_id, **extra):
    return lambda client: client.responses.retrieve(response_id, **extra)


def _poll_until_done(driver, response_id):
    response = driver.run(_retrieve(response_id))
    while response.status in ("queued", "in_progress"):
        response = driver.run(_retrieve(response_id))
    return response


def assert_the_job_record(payloads, fake):
    assert len(payloads) == 1, payloads
    record = payloads[0]
    assert record["input_token_count"] == 1000
    assert record["output_token_count"] == 400
    assert record["reasoning_token_count"] == 300
    assert record["transaction_id"] == fake.response_id
    assert record["trace_id"] == TRACE_ID


def _drain(driver, stream_call):
    if driver.asynchronous:
        async def drain(client):
            return [event async for event in await stream_call(client)]
        return driver.run(drain)
    return driver.run(lambda client: list(stream_call(client)))


def _interrupt(driver, stream_call):
    """Read the stream up to the first text delta, then close it, as a dropped connection would."""
    if driver.asynchronous:
        async def interrupt(client):
            stream = await stream_call(client)
            async for event in stream:
                if event.sequence_number == INTERRUPT_AFTER:
                    break
            await stream.close()
        return driver.run(interrupt)

    def interrupt_sync(client):
        stream = stream_call(client)
        for event in stream:
            if event.sequence_number == INTERRUPT_AFTER:
                break
        stream.close()
    return driver.run(interrupt_sync)


def _resume_with_stream_helper(driver, response_id):
    if driver.asynchronous:
        async def resume(client):
            async with client.responses.stream(response_id=response_id, starting_after=INTERRUPT_AFTER) as stream:
                return [event async for event in stream]
        return driver.run(resume)

    def resume_sync(client):
        with client.responses.stream(response_id=response_id, starting_after=INTERRUPT_AFTER) as stream:
            return list(stream)
    return driver.run(resume_sync)


@pytest.mark.parametrize("driver_class", DRIVERS)
def test_a_polled_background_job_sends_one_record_with_its_usage(payloads, driver_class):
    fake = BackgroundJobFake()
    driver = driver_class(fake)
    queued = driver.run(_create())
    assert queued.status == "queued"
    assert payloads == []
    assert _poll_until_done(driver, queued.id).status == "completed"
    assert_the_job_record(payloads, fake)


@pytest.mark.parametrize("driver_class", DRIVERS)
def test_retrieving_a_finished_job_again_sends_no_second_record(payloads, driver_class):
    fake = BackgroundJobFake()
    driver = driver_class(fake)
    queued = driver.run(_create())
    _poll_until_done(driver, queued.id)
    driver.run(_retrieve(queued.id))
    driver.run(_retrieve(queued.id))
    assert_the_job_record(payloads, fake)


@pytest.mark.parametrize("driver_class", DRIVERS)
def test_a_raw_retrieve_of_a_finished_job_sends_one_record(payloads, driver_class):
    fake = BackgroundJobFake(polls_until_done=1)
    driver = driver_class(fake)
    queued = driver.run(_create())
    raw = driver.run(lambda client: client.responses.with_raw_response.retrieve(queued.id))
    assert raw.parse().status == "completed"
    assert_the_job_record(payloads, fake)


def test_still_running_retrieves_send_nothing(payloads, caplog):
    fake = BackgroundJobFake(polls_until_done=10)
    driver = SyncDriver(fake)
    with caplog.at_level(logging.WARNING):
        queued = driver.run(_create())
        for _ in range(3):
            assert driver.run(_retrieve(queued.id)).status == "in_progress"
    assert payloads == []
    assert not [r for r in caplog.records if "Skipping metering" in r.getMessage()]


@pytest.mark.parametrize("driver_class", DRIVERS)
def test_a_streamed_background_job_sends_one_record_and_a_later_retrieve_adds_none(payloads, driver_class):
    fake = BackgroundJobFake(polls_until_done=1)
    driver = driver_class(fake)
    events = _drain(driver, _create(stream=True))
    assert events[-1].type == "response.completed"
    driver.run(_retrieve(fake.response_id))
    assert_the_job_record(payloads, fake)
    assert payloads[0]["is_streamed"] is True


class SlowStreamFake(BackgroundJobFake):
    """Answers the streamed create only after ``FIRST_BYTE_SECONDS``."""

    def __call__(self, request):
        if request.method == "POST":
            time.sleep(FIRST_BYTE_SECONDS)
        return super().__call__(request)


FIRST_BYTE_SECONDS = 0.2


@pytest.mark.parametrize("driver_class", DRIVERS)
def test_a_streamed_background_job_records_its_first_token_time(payloads, driver_class):
    """A fresh background stream times its first output text, as a foreground one does (BACK-4413)."""
    fake = SlowStreamFake(polls_until_done=1)
    _drain(driver_class(fake), _create(stream=True))
    assert len(payloads) == 1
    assert FIRST_BYTE_SECONDS * 1000 <= payloads[0]["time_to_first_token"] <= payloads[0]["request_duration"]


@pytest.mark.parametrize("driver_class", DRIVERS)
def test_a_resumed_background_stream_records_zero_first_token_time(payloads, driver_class):
    """The resumed stream's text was not timed from the original request, so it records 0 (BACK-4413)."""
    fake = BackgroundJobFake()
    driver = driver_class(fake)
    _interrupt(driver, _create(stream=True))
    _drain(driver, lambda client: client.responses.retrieve(
        fake.response_id, stream=True, starting_after=INTERRUPT_AFTER))
    assert len(payloads) == 1
    assert payloads[0]["time_to_first_token"] == 0


@pytest.mark.parametrize("driver_class", DRIVERS)
@pytest.mark.parametrize("resume", ["retrieve-stream", "stream-helper"])
def test_an_interrupted_background_stream_is_metered_once_when_resumed(payloads, driver_class, resume):
    fake = BackgroundJobFake()
    driver = driver_class(fake)
    _interrupt(driver, _create(stream=True))
    assert payloads == []
    if resume == "retrieve-stream":
        events = _drain(driver, lambda client: client.responses.retrieve(
            fake.response_id, stream=True, starting_after=INTERRUPT_AFTER))
        assert events[0].sequence_number == INTERRUPT_AFTER + 1
    else:
        _resume_with_stream_helper(driver, fake.response_id)
    assert_the_job_record(payloads, fake)
    assert payloads[0]["is_streamed"] is True
    _poll_until_done(driver, fake.response_id)
    assert len(payloads) == 1


@pytest.mark.parametrize("driver_class", DRIVERS)
def test_an_interrupted_background_stream_is_metered_when_polled_instead(payloads, driver_class):
    fake = BackgroundJobFake(polls_until_done=1)
    driver = driver_class(fake)
    _interrupt(driver, _create(stream=True))
    assert payloads == []
    _poll_until_done(driver, fake.response_id)
    assert_the_job_record(payloads, fake)


@pytest.mark.parametrize("driver_class", DRIVERS)
def test_an_interrupted_foreground_stream_still_sends_its_zero_token_record(payloads, driver_class):
    fake = BackgroundJobFake()
    driver = driver_class(fake)
    _interrupt(driver, lambda client: client.responses.create(stream=True, **REQUEST))
    assert len(payloads) == 1, payloads
    assert payloads[0]["input_token_count"] == 0
    assert payloads[0]["output_token_count"] == 0
    assert payloads[0]["transaction_id"] == fake.response_id


@pytest.mark.parametrize("driver_class", DRIVERS)
def test_retrieving_a_synchronously_created_response_sends_no_second_record(payloads, driver_class):
    fake = BackgroundJobFake(polls_until_done=1)
    driver = driver_class(fake)
    created = driver.run(lambda client: client.responses.create(**REQUEST))
    assert len(payloads) == 1
    driver.run(_retrieve(created.id))
    assert len(payloads) == 1


@pytest.mark.parametrize("driver_class", DRIVERS)
def test_a_breached_cap_does_not_block_retrieving_a_finished_job(monkeypatch, payloads, driver_class):
    fake = BackgroundJobFake(polls_until_done=1)
    driver = driver_class(fake)
    queued = driver.run(_create())
    seed_rules(monkeypatch, breached_rule(4244))
    assert driver.run(_retrieve(queued.id)).status == "completed"
    assert_the_job_record(payloads, fake)


def test_an_unread_streaming_response_retrieve_leaves_the_job_pending(payloads):
    fake = BackgroundJobFake(polls_until_done=1)
    driver = AsyncDriver(fake)
    queued = driver.run(_create())

    async def read_through_streaming_response(client):
        async with client.responses.with_streaming_response.retrieve(queued.id) as response:
            return await response.parse()

    assert driver.run(read_through_streaming_response).status == "completed"
    assert payloads == []
    driver.run(_retrieve(queued.id))
    assert_the_job_record(payloads, fake)


@pytest.mark.parametrize("driver_class", DRIVERS)
def test_a_cancelled_background_job_frees_its_entry_and_sends_no_record(payloads, driver_class):
    fake = BackgroundJobFake(polls_until_done=1)
    driver = driver_class(fake)
    queued = driver.run(_create())
    assert driver.run(lambda client: client.responses.cancel(queued.id)).status == "cancelled"
    driver.run(_retrieve(queued.id))
    assert payloads == []


@pytest.mark.parametrize("driver_class", DRIVERS)
def test_a_cancelled_background_job_that_reports_usage_sends_one_record(payloads, driver_class):
    fake = BackgroundJobFake(polls_until_done=1, cancel_usage=USAGE)
    driver = driver_class(fake)
    queued = driver.run(_create())
    driver.run(lambda client: client.responses.cancel(queued.id))
    driver.run(_retrieve(queued.id))
    assert_the_job_record(payloads, fake)


JOB_A = {"agenticJobId": "job-a", "agenticJobName": "Job A"}


def _job_fields_of(record):
    extra_body = record.get("extra_body") or {}
    return {key: extra_body[key] for key in ("agenticJobId", "agenticJobName") if key in extra_body}


def _create_in_job_a(driver, call):
    with JobContext(job_id="job-a", name="Job A"):
        return driver.run(call)


@pytest.mark.parametrize("driver_class", DRIVERS)
def test_a_job_created_in_one_agentic_job_and_polled_in_another_keeps_the_first_job(payloads, driver_class):
    fake = BackgroundJobFake(polls_until_done=1)
    driver = driver_class(fake)
    queued = _create_in_job_a(driver, _create())
    with JobContext(job_id="job-b", name="Job B"):
        _poll_until_done(driver, queued.id)
    assert_the_job_record(payloads, fake)
    assert _job_fields_of(payloads[0]) == JOB_A


@pytest.mark.parametrize("driver_class", DRIVERS)
def test_a_job_created_in_an_agentic_job_and_polled_outside_any_job_keeps_its_job(payloads, driver_class):
    fake = BackgroundJobFake(polls_until_done=1)
    driver = driver_class(fake)
    queued = _create_in_job_a(driver, _create())
    _poll_until_done(driver, queued.id)
    assert_the_job_record(payloads, fake)
    assert _job_fields_of(payloads[0]) == JOB_A


@pytest.mark.parametrize("driver_class", DRIVERS)
def test_an_explicit_usage_metadata_job_id_still_wins_over_the_create_time_job(payloads, driver_class):
    fake = BackgroundJobFake(polls_until_done=1)
    driver = driver_class(fake)
    usage_metadata = {"trace_id": TRACE_ID, "agentic_job_id": "job-explicit"}
    queued = _create_in_job_a(driver, lambda client: client.responses.create(
        background=True, model=REQUEST["model"], input=REQUEST["input"], usage_metadata=usage_metadata))
    with JobContext(job_id="job-b"):
        _poll_until_done(driver, queued.id)
    assert_the_job_record(payloads, fake)
    assert _job_fields_of(payloads[0]) == {"agenticJobId": "job-explicit", "agenticJobName": "Job A"}
    assert usage_metadata == {"trace_id": TRACE_ID, "agentic_job_id": "job-explicit"}


@pytest.mark.parametrize("driver_class", DRIVERS)
def test_an_interrupted_stream_created_in_one_agentic_job_and_resumed_in_another_keeps_the_first_job(
        payloads, driver_class):
    fake = BackgroundJobFake()
    driver = driver_class(fake)
    with JobContext(job_id="job-a", name="Job A"):
        _interrupt(driver, _create(stream=True))
    with JobContext(job_id="job-b", name="Job B"):
        _resume_with_stream_helper(driver, fake.response_id)
    assert_the_job_record(payloads, fake)
    assert _job_fields_of(payloads[0]) == JOB_A


@pytest.mark.parametrize("driver_class", DRIVERS)
def test_a_job_created_outside_any_agentic_job_and_polled_in_one_records_no_job(payloads, driver_class):
    fake = BackgroundJobFake(polls_until_done=1)
    driver = driver_class(fake)
    queued = driver.run(_create())
    with JobContext(job_id="job-b", name="Job B"):
        _poll_until_done(driver, queued.id)
    assert_the_job_record(payloads, fake)
    assert _job_fields_of(payloads[0]) == {}


@pytest.mark.parametrize("driver_class", DRIVERS)
def test_an_interrupted_stream_created_outside_any_agentic_job_and_resumed_in_one_records_no_job(
        payloads, driver_class):
    fake = BackgroundJobFake()
    driver = driver_class(fake)
    _interrupt(driver, _create(stream=True))
    with JobContext(job_id="job-b", name="Job B"):
        _resume_with_stream_helper(driver, fake.response_id)
    assert_the_job_record(payloads, fake)
    assert _job_fields_of(payloads[0]) == {}


@pytest.mark.parametrize("driver_class", DRIVERS)
@pytest.mark.parametrize("stream", [False, True], ids=["non-streamed", "streamed"])
def test_a_foreground_response_created_in_an_agentic_job_records_that_job(payloads, driver_class, stream):
    fake = BackgroundJobFake()
    driver = driver_class(fake)
    create = lambda client: client.responses.create(stream=stream, **REQUEST)  # noqa: E731
    with JobContext(job_id="job-b", name="Job B"):
        if stream:
            _drain(driver, create)
        else:
            driver.run(create)
    assert len(payloads) == 1, payloads
    assert _job_fields_of(payloads[0]) == {"agenticJobId": "job-b", "agenticJobName": "Job B"}


# --- the Revenium key of the customer who created the job ----------------------

KEY_A = "hak_background_key_a_fake"
KEY_B = "hak_background_key_b_fake"


@pytest.fixture
def scoped_clients(monkeypatch):
    """No process-wide key: a record is delivered only under a ``metering_key()`` client, one fake per key."""
    built = {}

    def new_client(api_key, base_url=None, http_client=None):
        return built.setdefault(api_key, MagicMock(name=api_key))

    monkeypatch.setattr(metering, "_new_client", new_client)
    monkeypatch.setattr(metering, "_scoped_clients", OrderedDict(), raising=False)
    monkeypatch.setattr(metering, "client", None)
    monkeypatch.delenv("REVENIUM_METERING_API_KEY", raising=False)
    return built


def _delivered_under(scoped_clients):
    assert metering_pool.drain(5) == 0
    return {key: [name for name, _args, _kwargs in client.ai.method_calls if name.startswith("create_")]
            for key, client in scoped_clients.items()}


POLLER_SCOPES = [
    pytest.param(contextlib.nullcontext, id="after-the-block"),
    pytest.param(lambda: metering_key(KEY_B), id="under-another-key"),
]


def _finish_the_job(driver, fake, finish):
    if finish == "resume":
        _resume_with_stream_helper(driver, fake.response_id)
    else:
        _poll_until_done(driver, fake.response_id)


@pytest.mark.parametrize("driver_class", DRIVERS)
@pytest.mark.parametrize("poller_scope", POLLER_SCOPES)
@pytest.mark.parametrize("start, finish", [("create", "poll"), ("interrupted-stream", "poll"),
                                           ("interrupted-stream", "resume")])
def test_a_background_job_is_delivered_under_the_key_it_was_created_under(
        scoped_clients, driver_class, poller_scope, start, finish):
    fake = BackgroundJobFake(polls_until_done=1)
    driver = driver_class(fake)
    with metering_key(KEY_A):
        if start == "create":
            assert driver.run(_create()).status == "queued"
        else:
            _interrupt(driver, _create(stream=True))
    with poller_scope():
        _finish_the_job(driver, fake, finish)

    delivered = _delivered_under(scoped_clients)
    assert delivered[KEY_A] == ["create_completion"]
    assert delivered.get(KEY_B, []) == []


def test_a_sync_streaming_response_retrieve_leaves_the_job_pending(payloads):
    fake = BackgroundJobFake(polls_until_done=1)
    driver = SyncDriver(fake)
    queued = driver.run(_create())

    def read_through_streaming_response(client):
        with client.responses.with_streaming_response.retrieve(queued.id) as response:
            return response.parse()

    assert driver.run(read_through_streaming_response).status == "completed"
    assert payloads == []
    driver.run(_retrieve(queued.id))
    assert_the_job_record(payloads, fake)
