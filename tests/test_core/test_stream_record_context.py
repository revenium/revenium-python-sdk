"""A streamed call is recorded with the job and idempotency key it was created in (BACK-4370).

A service that runs each request as a Revenium job, and hands a streamed
answer to a response writer or another task, had the stream recorded under
whichever job was open when the stream finished: the next request's job and
key, or none at all once the request's block had closed. An OpenAI stream
abandoned without ``close()`` took the job of whichever request happened to
run the garbage collector.
"""
import contextvars
import gc
import threading
from types import SimpleNamespace

import pytest

from revenium_middleware._core import metering, metering_pool
from revenium_middleware._core.context import (
    _agentic_job_context, _idempotency_key_context, next_idempotency_key, set_agentic_job_fields,
    set_idempotency_key,
)

from .test_metering_key_switch import (  # noqa: F401  keys is a fixture
    NOW, STREAM_SITES, anthropic_kwargs, anthropic_messages, drain, gemini_chunk, keys, ollama_chunks,
    openai_chunks, openai_completion,
)

CUSTOMER_A = {"organizationName": "CustomerA"}


@pytest.fixture(autouse=True)
def no_job_from_the_environment(monkeypatch):
    monkeypatch.delenv("REVENIUM_AGENTIC_JOB_ID", raising=False)


def enter_job(name):
    """What ``JobContext(job)`` and ``idempotency_key(idem)`` set for a request's block."""
    set_agentic_job_fields(job_id=f"job-{name}")
    set_idempotency_key(f"idem-{name}")


def leave_job():
    """The state once the request's block has closed."""
    _agentic_job_context.set(None)
    _idempotency_key_context.set(None)


def sent(client):
    """``(job id, Idempotency-Key, organization)`` of every usage record ``client`` was asked to deliver."""
    return [
        ((kwargs.get("extra_body") or {}).get("agenticJobId"),
         (kwargs.get("extra_headers") or {}).get("Idempotency-Key"),
         kwargs.get("organization_name"))
        for name, _args, kwargs in client.ai.method_calls
        if name.startswith("create_")
    ]


def enqueue_one_record_with_its_job(organization_name):
    """A provider's record builder, reduced to the fields under test: it resolves the
    job when it builds the record, as every real builder does."""
    from revenium_middleware._core.fields import extract_agentic_job_fields
    from revenium_middleware._core.metering_submission import submit_ai_event

    async def send():
        submit_ai_event("completion", {"organization_name": organization_name,
                                       "extra_body": extract_agentic_job_fields({})})

    return lambda *_: metering.run_async_in_thread(send())


def drive_perplexity_stream(switch):
    from revenium_middleware.perplexity.streaming import MeteredStream, StreamMeter
    stream = MeteredStream(iter(["Hel", "lo"]), StreamMeter(enqueue_one_record_with_its_job("CustomerA")))
    next(stream)
    switch()
    list(stream)


def drive_bedrock_transport_stream(switch):
    from revenium_middleware.anthropic.bedrock_transport import _MeteredEventStream
    stream = iter(_MeteredEventStream(iter([{"chunk": 1}, {"chunk": 2}]), lambda event: None,
                                      enqueue_one_record_with_its_job("CustomerA")))
    next(stream)
    switch()
    list(stream)


CALLBACK_RECORDED = pytest.mark.xfail(
    strict=True, reason="callback-recorded models build the record at on_llm_end; out of scope, follow-up")

SITES = {**STREAM_SITES, "perplexity": drive_perplexity_stream, "anthropic_bedrock_transport": drive_bedrock_transport_stream}
SITE_PARAMS = [pytest.param(site, marks=CALLBACK_RECORDED) if site == "langchain_callback" else site
               for site in sorted(SITES)]


def in_job_a(body):
    """Run ``body(a_block)`` in a fresh context inside job A; return job A's block."""
    def run():
        enter_job("A")
        a_block = contextvars.copy_context()
        body(a_block)
        return a_block
    return contextvars.copy_context().run(run)


def entering_job_b(b_blocks):
    def switch():
        enter_job("B")
        b_blocks.append(contextvars.copy_context())
    return switch


@pytest.mark.parametrize("site", SITE_PARAMS)
def test_a_stream_handed_to_another_job_is_recorded_under_the_job_and_key_it_was_created_in(site, keys):
    b_blocks = []
    a_block = in_job_a(lambda _a: SITES[site](entering_job_b(b_blocks)))
    drain()

    assert sent(keys.a) == [("job-A", "idem-A", "CustomerA")]
    assert sent(keys.b) == []
    assert b_blocks[0].run(next_idempotency_key) == "idem-B"
    assert a_block.run(next_idempotency_key) == "idem-A:2"


def plain_openai_call():
    from revenium_middleware.openai import middleware as mw
    mw.create_wrapper(lambda *a, **k: openai_completion("CustomerA"), SimpleNamespace(_client=None), (),
                      {"model": "gpt-4o-mini", "messages": [], "usage_metadata": dict(CUSTOMER_A)})


def test_a_stream_and_a_plain_call_in_one_job_get_different_keys(keys):
    from revenium_middleware.openai import middleware as mw

    def body(_a_block):
        stream = mw.handle_streaming_response(iter(openai_chunks()), NOW, dict(CUSTOMER_A), None, {})
        next(stream)
        plain_openai_call()
        enter_job("B")
        list(stream)

    in_job_a(body)
    drain()

    records = sent(keys.a)
    assert sorted(records) == [("job-A", "idem-A", "CustomerA"), ("job-A", "idem-A:2", "CustomerA")]


@pytest.mark.parametrize("site", SITE_PARAMS)
def test_a_stream_read_after_its_job_block_closed_keeps_that_job_and_key(site, keys):
    in_job_a(lambda _a: SITES[site](leave_job))
    drain()

    assert sent(keys.a) == [("job-A", "idem-A", "CustomerA")]


# --- a stream dropped without close() and collected on another thread -------------

def started(stream):
    iterator = iter(stream)
    next(iterator)
    return [stream, iterator]


def started_openai_stream():
    from revenium_middleware.openai import middleware as mw
    return started(mw.handle_streaming_response(iter(openai_chunks()), NOW, dict(CUSTOMER_A), None, {}))


def started_anthropic_raw_stream():
    return started(anthropic_messages(async_client=False).create(**anthropic_kwargs(stream=True)))


def started_gemini_stream():
    from revenium_middleware.google.google_ai import middleware as genai_mw
    return started(genai_mw.handle_streaming_response(
        iter([gemini_chunk("Hel"), gemini_chunk("lo", total=8)]), NOW, dict(CUSTOMER_A)))


def started_litellm_stream():
    pytest.importorskip("litellm")
    from revenium_middleware.litellm.client import middleware as litellm_mw
    usage = SimpleNamespace(prompt_tokens=3, completion_tokens=5, total_tokens=8)
    chunks = [SimpleNamespace(usage=None, choices=[], model="gpt-4o-mini", id="chatcmpl-drop"),
              SimpleNamespace(usage=usage, choices=[], model="gpt-4o-mini", id="chatcmpl-drop")]
    return started(litellm_mw.handle_streaming_response(iter(chunks), NOW, dict(CUSTOMER_A)))


def started_ollama_stream():
    pytest.importorskip("ollama")
    from revenium_middleware.ollama import middleware as ollama_mw
    return started(ollama_mw.handle_streaming_response(
        iter(ollama_chunks()), NOW, dict(CUSTOMER_A), "txn-drop", "chat", {"model": "llama3"}))


def started_bedrock_transport_stream():
    from revenium_middleware.anthropic.bedrock_transport import _MeteredEventStream
    return started(_MeteredEventStream(iter([{"chunk": 1}, {"chunk": 2}]), lambda event: None,
                                       enqueue_one_record_with_its_job("CustomerA")))


ABANDONED_STREAMS = {
    "openai_chat": started_openai_stream,
    "anthropic_raw_stream": started_anthropic_raw_stream,
    "gemini": started_gemini_stream,
    "litellm": started_litellm_stream,
    "ollama": started_ollama_stream,
    "anthropic_bedrock_transport": started_bedrock_transport_stream,
}


@pytest.fixture
def submitting_threads(monkeypatch):
    """The thread id each usage record was queued from."""
    threads = []
    submit = metering_pool.MeteringWorkerPool.submit

    def spy(pool, task):
        threads.append(threading.get_ident())
        submit(pool, task)

    monkeypatch.setattr(metering_pool.MeteringWorkerPool, "submit", spy)
    return threads


def abandon_in_a_cycle(stream):
    holder = [stream]
    holder.append(holder)


def collect_inside_job_b():
    collector = {}

    def run():
        enter_job("B")
        collector["thread"] = threading.get_ident()
        gc.collect()

    thread = threading.Thread(target=lambda: contextvars.copy_context().run(run))
    thread.start()
    thread.join(10)
    return collector["thread"]


@pytest.mark.parametrize("site", sorted(ABANDONED_STREAMS))
def test_an_abandoned_stream_collected_on_another_thread_inside_another_job_keeps_its_job(
        site, keys, submitting_threads):
    gc.collect()
    gc.disable()
    try:
        in_job_a(lambda _a: abandon_in_a_cycle(ABANDONED_STREAMS[site]()))
        collector_thread = collect_inside_job_b()
    finally:
        gc.enable()
    drain()

    assert submitting_threads == [collector_thread]
    assert sent(keys.a) == [("job-A", "idem-A", "CustomerA")]


def test_an_openai_stream_broken_off_inside_another_job_gets_an_error_key_from_its_own_block(keys):
    from revenium_middleware.openai import middleware as mw

    def broken_off():
        yield openai_chunks()[0]
        raise ConnectionError("connection reset by peer")

    def body(_a_block):
        stream = mw.handle_streaming_response(broken_off(), NOW, dict(CUSTOMER_A), None, {})
        next(stream)
        enter_job("B")
        with pytest.raises(ConnectionError):
            list(stream)

    in_job_a(body)
    drain()

    [(job, key, organization)] = sent(keys.a)
    assert (job, organization) == ("job-A", "CustomerA")
    assert key.startswith("idem-A:err:")


def job_id():
    return (_agentic_job_context.get() or {}).get("agenticJobId")


def test_call_context_runs_nested_and_concurrently_without_error():
    from revenium_middleware._core.call_context import CallContext

    def created_in_job_a():
        enter_job("A")
        return CallContext()

    call_context = contextvars.copy_context().run(created_in_job_a)
    assert call_context.run(call_context.run, job_id) == "job-A"

    both_inside = threading.Barrier(2, timeout=5)
    seen, errors = [], []

    def finish():
        both_inside.wait()
        return job_id()

    def finish_on_this_thread():
        try:
            seen.append(call_context.run(finish))
        except Exception as error:  # noqa: BLE001 - the assertion below reports it
            errors.append(error)

    threads = [threading.Thread(target=finish_on_this_thread) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(10)

    assert errors == []
    assert seen == ["job-A", "job-A"]


# --- a background Responses stream read inside another job -----------------------

def background_event(response_id):
    return SimpleNamespace(response=SimpleNamespace(
        id=response_id, model="o3", usage=None, status="queued", incomplete_details=None))


def test_a_background_responses_stream_read_inside_another_job_keeps_the_job_it_was_created_in():
    from revenium_middleware.openai import middleware as mw
    response_id = "resp_bg_context"
    created = []
    in_job_a(lambda _a: created.append(mw._ResponsesStreamMeter(NOW, {}, None, "o3", background=True)))
    stream_meter = created[0]

    def read_in_job_b():
        enter_job("B")
        stream_meter.observe(background_event(response_id))

    contextvars.copy_context().run(read_in_job_b)
    pending = mw._pending_background_responses.take(response_id)

    assert pending.usage_metadata.get("agenticJobId") == "job-A"
