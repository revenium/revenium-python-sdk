"""Tool metering must never block the wrapped call.

meter_tool and report_tool_call previously issued the metering POST inline
(sync client in the decorator's finally, awaited client in the async path),
stalling the caller for the full round-trip on a slow endpoint. They must
dispatch fire-and-forget like AI-completion metering does.
"""
import asyncio
import contextlib
import inspect
import json
import logging
import time
import warnings

import pytest

from revenium_middleware._core.context import _agentic_job_context, set_agentic_job_fields
from revenium_middleware._metering import decorator as tool_metering
from revenium_middleware._metering.decorator import configure, meter_tool, report_tool_call

SLOW_SECONDS = 1.0
FAST_BUDGET_SECONDS = 0.5


class RecordedCall:
    def __init__(self, url, headers, json):
        self.url = url
        self.headers = headers
        self.json = json


class FakeResponse:
    def raise_for_status(self):
        return None


class HttpxStub:
    """Stands in for the httpx module; every POST sleeps for ``delay`` seconds.

    The dispatch tests need a slow endpoint to prove the caller is not blocked;
    the payload tests only care about the posted body, so they use delay=0 and
    stay fast.
    """

    def __init__(self, delay=SLOW_SECONDS):
        self.calls = []
        self.delay = delay
        stub = self

        class Client:
            def __init__(self, timeout=None):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def post(self, url, headers=None, json=None):
                time.sleep(stub.delay)
                stub.calls.append(RecordedCall(url, headers, json))
                return FakeResponse()

        class AsyncClient:
            def __init__(self, timeout=None):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *exc):
                return False

            async def post(self, url, headers=None, json=None):
                await asyncio.sleep(stub.delay)
                stub.calls.append(RecordedCall(url, headers, json))
                return FakeResponse()

        self.Client = Client
        self.AsyncClient = AsyncClient


@pytest.fixture()
def slow_endpoint(monkeypatch):
    stub = HttpxStub()
    monkeypatch.setattr(tool_metering, "httpx", stub)
    configure(metering_url="http://metering.test", api_key="rev_mk_test")
    yield stub
    configure()  # clear overrides; resolution falls back to the environment


@pytest.fixture()
def endpoint(monkeypatch):
    """A metering endpoint that answers immediately, for body assertions.

    Also clears REVENIUM_AGENTIC_JOB_ID so a developer machine that exports it
    cannot make the "no job id anywhere" case pass or fail by accident.
    """
    monkeypatch.delenv("REVENIUM_AGENTIC_JOB_ID", raising=False)
    stub = HttpxStub(delay=0)
    monkeypatch.setattr(tool_metering, "httpx", stub)
    configure(metering_url="http://metering.test", api_key="rev_mk_test")
    yield stub
    configure()  # clear overrides; resolution falls back to the environment


def wait_for_calls(stub, count=1, timeout=6.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if len(stub.calls) >= count:
            return True
        time.sleep(0.05)
    return False


def test_meter_tool_sync_returns_before_metering_completes(slow_endpoint):
    @meter_tool("audit-probe-tool")
    def fast_tool(x):
        return x * 2

    start = time.perf_counter()
    result = fast_tool(21)
    elapsed = time.perf_counter() - start

    assert result == 42
    assert elapsed < FAST_BUDGET_SECONDS, f"caller blocked for {elapsed:.2f}s"
    # The event is still delivered, just in the background.
    assert wait_for_calls(slow_endpoint)
    assert slow_endpoint.calls[0].json["toolId"] == "audit-probe-tool"


def test_report_tool_call_returns_before_metering_completes(slow_endpoint):
    start = time.perf_counter()
    report_tool_call(tool_id="manual-tool", duration_ms=5, success=True)
    elapsed = time.perf_counter() - start

    assert elapsed < FAST_BUDGET_SECONDS, f"caller blocked for {elapsed:.2f}s"
    assert wait_for_calls(slow_endpoint)
    assert slow_endpoint.calls[0].json["toolId"] == "manual-tool"


def test_meter_tool_async_returns_before_metering_completes(slow_endpoint):
    @meter_tool("async-probe-tool")
    async def fast_async_tool(x):
        return x + 1

    async def scenario():
        start = time.perf_counter()
        result = await fast_async_tool(1)
        return result, time.perf_counter() - start

    result, elapsed = asyncio.run(scenario())

    assert result == 2
    assert elapsed < FAST_BUDGET_SECONDS, f"caller blocked for {elapsed:.2f}s"
    assert wait_for_calls(slow_endpoint)
    assert slow_endpoint.calls[0].json["toolId"] == "async-probe-tool"


def test_dispatch_time_failure_never_reaches_the_caller(slow_endpoint, monkeypatch):
    # Synchronous stub: raises while _dispatch_tool_event constructs the
    # coroutine, i.e. before anything is handed to the background thread.
    def boom(**kwargs):
        raise RuntimeError("metering exploded")

    monkeypatch.setattr(tool_metering, "_send_tool_event_async", boom)

    @meter_tool("resilient-tool")
    def fast_tool():
        return "ok"

    assert fast_tool() == "ok"


def test_background_metering_failure_never_reaches_the_caller(slow_endpoint, monkeypatch, caplog):
    # Async stub: the failure only surfaces when the coroutine is awaited on
    # the metering thread, exercising the background swallow-and-log path.
    async def boom(*args, **kwargs):
        raise RuntimeError("metering exploded in background")

    monkeypatch.setattr(tool_metering, "_send_tool_event_async", boom)

    @meter_tool("resilient-tool-async")
    def fast_tool():
        return "ok"

    with caplog.at_level(logging.WARNING, logger="revenium_middleware"):
        assert fast_tool() == "ok"
        # The metering thread must catch AND log the failure -- if its
        # exception handling were removed, nothing would be logged and this
        # assertion would fail (the thread would die with stderr noise only).
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and "metering exploded in background" not in caplog.text:
            time.sleep(0.05)

    assert "metering exploded in background" in caplog.text


def test_dispatch_failure_does_not_leak_unawaited_coroutine(slow_endpoint, monkeypatch):
    # If run_async_in_thread raises after the coroutine is created, the
    # dispatcher must close() it; otherwise garbage collection emits a
    # "coroutine was never awaited" RuntimeWarning. GC timing is not
    # deterministic under pytest (log capture keeps the raised exception --
    # and through its traceback, the coroutine -- alive past the test body),
    # so capture the coroutine and assert on its state directly.
    captured = []

    def raiser(coro):
        captured.append(coro)
        raise RuntimeError("thread pool exploded")

    # _dispatch_tool_event imports the symbol lazily from the package root.
    monkeypatch.setattr("revenium_middleware.run_async_in_thread", raiser)

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        report_tool_call(tool_id="leak-guard-tool", duration_ms=1, success=True)

    assert len(captured) == 1
    coro = captured.pop()
    try:
        assert inspect.getcoroutinestate(coro) == inspect.CORO_CLOSED, (
            "dispatcher must close the coroutine when run_async_in_thread fails"
        )
    finally:
        # Idempotent; keeps a failing run from also spewing the leak warning.
        coro.close()


# ---------------------------------------------------------------------------
# agenticJobId attribution (BACK-2751)
#
# Tool spend only joins the agentic job it belongs to if the tool event carries
# the same agenticJobId the process already puts on its AI completions. The
# precedence below is the completion path's precedence, reused verbatim.
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def ambient_job(job_id):
    """Set the ambient agentic job for the block, the way JobContext does."""
    token = set_agentic_job_fields(job_id=job_id)
    try:
        yield
    finally:
        _agentic_job_context.reset(token)


def posted_body(stub):
    """Return the single posted event body, waiting for the background thread."""
    assert wait_for_calls(stub), "no tool event was posted"
    return stub.calls[0].json


def test_usage_metadata_snake_case_job_id_beats_context_and_env(endpoint, monkeypatch):
    monkeypatch.setenv("REVENIUM_AGENTIC_JOB_ID", "job-from-env")

    with ambient_job("job-from-context"):
        report_tool_call(
            tool_id="firecrawl",
            usage_metadata={"agentic_job_id": "job-from-metadata"},
        )

    assert posted_body(endpoint)["agenticJobId"] == "job-from-metadata"


def test_usage_metadata_camel_case_alias_resolves(endpoint):
    report_tool_call(tool_id="firecrawl", usage_metadata={"agenticJobId": "job-camel"})

    assert posted_body(endpoint)["agenticJobId"] == "job-camel"


def test_ambient_job_context_supplies_job_id_when_metadata_is_silent(endpoint, monkeypatch):
    # Env is set to a different value so this also pins context above env.
    monkeypatch.setenv("REVENIUM_AGENTIC_JOB_ID", "job-from-env")

    @meter_tool("firecrawl", operation="scrape")
    def scrape(url):
        return {"pages": 1}

    with ambient_job("job-from-context"):
        assert scrape("https://example.test") == {"pages": 1}

    assert posted_body(endpoint)["agenticJobId"] == "job-from-context"


def test_env_var_is_the_last_fallback(endpoint, monkeypatch):
    monkeypatch.setenv("REVENIUM_AGENTIC_JOB_ID", "job-from-env")

    @meter_tool("firecrawl")
    def scrape():
        return "ok"

    assert scrape() == "ok"

    assert posted_body(endpoint)["agenticJobId"] == "job-from-env"


def test_job_id_key_is_absent_when_nothing_resolves(endpoint):
    # The `endpoint` fixture clears REVENIUM_AGENTIC_JOB_ID; no context, no metadata.
    @meter_tool("firecrawl")
    def scrape():
        return "ok"

    scrape()

    body = posted_body(endpoint)
    assert "agenticJobId" not in body, "unset job id must be omitted, not sent as null"
    # The field must not reach the wire as an explicit null either.
    assert "agenticJobId" not in json.loads(json.dumps(body))


def test_unusable_metadata_job_id_is_omitted_and_never_raises(endpoint):
    # A non-string (here also unhashable) value cannot be a job id on the wire.
    # Resolution happens inside a fire-and-forget dispatch, so it must degrade
    # to omitting the field rather than surfacing an error to the caller.
    @meter_tool("firecrawl", output_fields=["agentic_job_id"])
    def scrape():
        return {"agentic_job_id": {"nested": ["not", "a", "string"]}}

    assert scrape() == {"agentic_job_id": {"nested": ["not", "a", "string"]}}

    body = posted_body(endpoint)
    assert "agenticJobId" not in body
    # The raw metadata is still forwarded verbatim, as it always was.
    assert body["usageMetadata"] == {"agentic_job_id": {"nested": ["not", "a", "string"]}}


def test_non_mapping_usage_metadata_still_resolves_the_ambient_job(endpoint):
    # report_tool_call takes whatever the caller passes; a non-mapping must not
    # break resolution, it must simply contribute nothing to it.
    with ambient_job("job-from-context"):
        report_tool_call(tool_id="firecrawl", usage_metadata=["not", "a", "mapping"])

    assert posted_body(endpoint)["agenticJobId"] == "job-from-context"


class TestAgenticJobIdServerPatternGate:
    """Ids the server's pattern rejects are dropped, not sent.

    The tool-events controller validates agenticJobId against
    JobValidation.AGENTIC_JOB_ID_PATTERN and 400s the WHOLE event on a
    violation — and dispatch is fire-and-forget, so the event and its cost
    would vanish silently. Dropping the field loses one call's attribution;
    sending it loses the event.
    """

    @pytest.mark.parametrize(
        "bad_id",
        [
            "has space",
            "colon:sep",
            "slash/sep",
            "at@sign",
            "-leading-dash",
            "_leading_underscore",
            "x" * 256,
            "types",
            "conversion-funnel",
            # Python-specific traps: $ accepts a trailing newline and .match()
            # does not require full consumption — both must still be rejected,
            # because the server's Java matches() demands the whole string.
            "job-1\n",
            "job-1\nx",
        ],
    )
    def test_invalid_id_is_dropped_and_event_still_posts(self, endpoint, bad_id):
        report_tool_call(tool_id="hammer", usage_metadata={"agentic_job_id": bad_id})
        body = posted_body(endpoint)
        assert "agenticJobId" not in body
        assert body["toolId"] == "hammer"

    @pytest.mark.parametrize(
        "good_id",
        ["job-1", "A", "j.o.b_2-x", "types2", "conversion-funnel-a", "x" * 255],
    )
    def test_valid_id_passes_verbatim(self, endpoint, good_id):
        report_tool_call(tool_id="hammer", usage_metadata={"agentic_job_id": good_id})
        assert posted_body(endpoint)["agenticJobId"] == good_id
