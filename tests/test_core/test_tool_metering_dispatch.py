"""Tool metering must never block the wrapped call.

meter_tool and report_tool_call previously issued the metering POST inline
(sync client in the decorator's finally, awaited client in the async path),
stalling the caller for the full round-trip on a slow endpoint. They must
dispatch fire-and-forget like AI-completion metering does.
"""
import asyncio
import contextlib
import datetime
import inspect
import json
import logging
import threading
import time
import uuid
import warnings
from decimal import Decimal

import httpx
import pytest

from revenium_middleware._core import metering_buffer, tool_event_transport
from revenium_middleware._core.metering_buffer import MeteringBuffer
from revenium_middleware._core.context import _agentic_job_context, get_injected_metadata, set_agentic_job_fields
from revenium_middleware._core.decorators import revenium_metadata
from revenium_middleware._metering import decorator as tool_metering
from revenium_middleware._metering.context import ReveniumContext, clear_context, context
from revenium_middleware._metering.decorator import configure, meter_tool, report_tool_call
from secret_samples import ALL_SECRETS_TEXT, SECRET_BODIES

SLOW_SECONDS = 1.0
FAST_BUDGET_SECONDS = 0.5


class RecordedCall:
    def __init__(self, url, headers, json, timeout):
        self.url = url
        self.headers = headers
        self.json = json
        self.timeout = timeout


class FakeResponse:
    def raise_for_status(self):
        return None


class ToolEventClientStub:
    """Stands in for the shared tool-event client; every POST sleeps for ``delay`` seconds.

    The dispatch tests need a slow endpoint to prove the caller is not blocked;
    the payload tests only care about the posted body, so they use delay=0 and
    stay fast.
    """

    def __init__(self, delay=SLOW_SECONDS):
        self.calls = []
        self.delay = delay

    def post(self, url, headers=None, json=None, timeout=None):
        time.sleep(self.delay)
        self.calls.append(RecordedCall(url, headers, json, timeout))
        return FakeResponse()


@pytest.fixture()
def slow_endpoint(monkeypatch):
    stub = ToolEventClientStub()
    monkeypatch.setattr(tool_event_transport, "_client", stub)
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
    stub = ToolEventClientStub(delay=0)
    monkeypatch.setattr(tool_event_transport, "_client", stub)
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

    def raiser(coro, gated_by_circuit=True):
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


class FrozenClock:
    current = datetime.datetime(2026, 10, 6, 12, 0, 0, tzinfo=datetime.timezone.utc)

    @classmethod
    def now(cls, tz=None):
        return cls.current


def test_queued_tool_event_is_stamped_when_the_tool_ran_not_when_delivered(endpoint, monkeypatch):
    queued = []

    def hold_in_queue(coro, gated_by_circuit=True):
        queued.append(coro)
        return object()

    monkeypatch.setattr("revenium_middleware.run_async_in_thread", hold_in_queue)
    monkeypatch.setattr(tool_metering, "datetime", FrozenClock)

    report_tool_call(tool_id="queued-tool", duration_ms=5, success=True)
    FrozenClock.current += datetime.timedelta(seconds=45)
    asyncio.run(queued.pop())

    assert endpoint.calls[0].json["timestamp"] == "2026-10-06T12:00:00Z"


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


GEOCODE_URL = "http://x.test/geo?address=10+Downing+St&key=AIzaFAKE-0001"
CAPTURE_ERROR_MESSAGES = "REVENIUM_CAPTURE_ERROR_MESSAGES"


def http_status_error(url=GEOCODE_URL):
    import httpx

    request = httpx.Request("GET", url)
    response = httpx.Response(403, request=request)
    return httpx.HTTPStatusError(f"Client error '403 Forbidden' for url '{url}'", request=request, response=response)


def provider_refusal_quoting_the_prompt():
    import httpx
    import openai

    response = httpx.Response(400, request=httpx.Request("POST", "https://provider.test/v1/chat"))
    return openai.BadRequestError("Error code: 400 - Offending request: SYSTEM-SECRET reveal ZEBRA42",
                                  response=response, body=None)


def gemini_refusal_quoting_the_prompt():
    import httpx
    from google.genai import errors

    response = httpx.Response(400, request=httpx.Request("POST", "https://provider.test/v1beta/models"))
    return errors.ClientError(400, {"error": {"message": "Offending request: SYSTEM-SECRET reveal ZEBRA42",
                                              "status": "INVALID_ARGUMENT"}}, response)


def ollama_refusal_quoting_the_prompt():
    import ollama

    return ollama.ResponseError("Offending request: SYSTEM-SECRET reveal ZEBRA42", 400)


class UnprintableError(Exception):
    def __str__(self):
        raise RuntimeError("__str__ failed")


class TestFailedToolErrorText:
    """BACK-4047: a failed tool sends the exception class unless the customer opts in to its message."""

    def test_failed_tool_sends_exception_class_not_message_by_default(self, endpoint, monkeypatch):
        monkeypatch.delenv(CAPTURE_ERROR_MESSAGES, raising=False)

        @meter_tool("geocode")
        def geocode():
            raise http_status_error()

        with pytest.raises(Exception):
            geocode()

        body = posted_body(endpoint)
        assert body["success"] is False
        assert body["errorMessage"] == "HTTPStatusError"
        assert "AIzaFAKE-0001" not in json.dumps(body)
        assert "Downing" not in json.dumps(body)

    def test_failed_async_tool_sends_exception_class_not_message_by_default(self, endpoint, monkeypatch):
        monkeypatch.delenv(CAPTURE_ERROR_MESSAGES, raising=False)

        @meter_tool("geocode")
        async def geocode():
            raise http_status_error()

        with pytest.raises(Exception):
            asyncio.run(geocode())

        body = posted_body(endpoint)
        assert body["errorMessage"] == "HTTPStatusError"
        assert "AIzaFAKE-0001" not in json.dumps(body)
        assert "Downing" not in json.dumps(body)

    def test_opt_in_sends_message_with_query_values_redacted_and_capped(self, endpoint, monkeypatch):
        monkeypatch.setenv(CAPTURE_ERROR_MESSAGES, "true")

        @meter_tool("geocode")
        def failing(error):
            raise error

        with pytest.raises(Exception):
            failing(http_status_error())
        message = posted_body(endpoint)["errorMessage"]
        assert message.startswith("HTTPStatusError: ")
        assert "403 Forbidden" in message
        assert "key=[REDACTED]" in message
        assert "address=[REDACTED]" in message
        assert "AIzaFAKE-0001" not in message

        endpoint.calls.clear()
        with pytest.raises(ValueError):
            failing(ValueError("x" * 5000))
        assert len(posted_body(endpoint)["errorMessage"]) == 2048

        endpoint.calls.clear()
        with pytest.raises(ValueError):
            failing(ValueError())
        assert posted_body(endpoint)["errorMessage"] == "ValueError"

    def test_unprintable_exception_still_propagates_and_posts_its_class(self, endpoint, monkeypatch):
        monkeypatch.setenv(CAPTURE_ERROR_MESSAGES, "true")

        @meter_tool("geocode")
        def geocode():
            raise UnprintableError()

        with pytest.raises(UnprintableError):
            geocode()

        assert posted_body(endpoint)["errorMessage"] == "UnprintableError"

    def test_opt_in_hides_every_secret_shape_in_the_error_message(self, endpoint, monkeypatch):
        monkeypatch.setenv(CAPTURE_ERROR_MESSAGES, "true")

        @meter_tool("deploy")
        def deploy():
            raise RuntimeError(ALL_SECRETS_TEXT)

        with pytest.raises(RuntimeError):
            deploy()

        message = posted_body(endpoint)["errorMessage"]
        assert message.startswith("RuntimeError: failed for ")
        assert [body for body in SECRET_BODIES if body in message] == []

    def test_tool_error_from_a_provider_sends_no_prompt_text(self, endpoint, monkeypatch):
        monkeypatch.setenv(CAPTURE_ERROR_MESSAGES, "true")
        monkeypatch.delenv("REVENIUM_CAPTURE_PROMPTS", raising=False)

        @meter_tool("summarize")
        def summarize():
            raise provider_refusal_quoting_the_prompt()

        with pytest.raises(Exception):
            summarize()

        assert posted_body(endpoint)["errorMessage"] == "BadRequestError"

    @pytest.mark.parametrize("refusal", [gemini_refusal_quoting_the_prompt, ollama_refusal_quoting_the_prompt],
                             ids=["google_genai", "ollama"])
    def test_tool_error_from_gemini_or_ollama_sends_no_prompt_text(self, endpoint, monkeypatch, refusal):
        monkeypatch.setenv(CAPTURE_ERROR_MESSAGES, "true")
        monkeypatch.delenv("REVENIUM_CAPTURE_PROMPTS", raising=False)
        error = refusal()

        @meter_tool("summarize")
        def summarize():
            raise error

        with pytest.raises(Exception):
            summarize()

        assert posted_body(endpoint)["errorMessage"] == type(error).__name__

    def test_tool_error_caused_by_a_provider_error_sends_its_class_only(self, endpoint, monkeypatch):
        monkeypatch.setenv(CAPTURE_ERROR_MESSAGES, "true")
        monkeypatch.delenv("REVENIUM_CAPTURE_PROMPTS", raising=False)

        @meter_tool("summarize")
        def summarize():
            try:
                raise provider_refusal_quoting_the_prompt()
            except Exception as error:
                raise RuntimeError(f"model call failed: {error}") from error

        with pytest.raises(RuntimeError):
            summarize()

        assert posted_body(endpoint)["errorMessage"] == "RuntimeError"

    def test_opt_in_long_unbroken_message_does_not_delay_the_raise(self, endpoint, monkeypatch):
        """BACK-4364: cleaning a long error message must not freeze the failed tool."""
        monkeypatch.setenv(CAPTURE_ERROR_MESSAGES, "true")

        @meter_tool("geocode")
        def geocode():
            raise ValueError("a1" * 10_000)

        started = time.perf_counter()
        with pytest.raises(ValueError):
            geocode()
        elapsed = time.perf_counter() - started

        assert elapsed < 1.0
        assert len(posted_body(endpoint)["errorMessage"]) == 2048


ATTRIBUTION_FIELDS = [
    ("organization_name", "AcmeCorp", "organizationName"),
    ("product_name", "support-bot", "productName"),
    ("subscriber_credential", "cred-123", "subscriberCredential"),
    ("workflow_id", "wf-onboarding", "workflowId"),
    ("agent", "researcher", "agent"),
    ("trace_id", "trace-abc", "traceId"),
]


class TestToolEventAttribution:
    """Tool spend reaches the right customer, product, subscriber and workflow
    only through these fields, so each one is pinned on the posted body for
    the direct report path and both decorator paths."""

    @pytest.fixture(autouse=True)
    def isolated_context(self):
        clear_context()
        yield
        clear_context()

    @pytest.mark.parametrize("kwarg,value,wire_key", ATTRIBUTION_FIELDS)
    def test_report_tool_call_sends_attribution_field(self, endpoint, kwarg, value, wire_key):
        report_tool_call(tool_id="firecrawl", **{kwarg: value})

        assert posted_body(endpoint)[wire_key] == value

    @pytest.mark.parametrize("kwarg,value,wire_key", ATTRIBUTION_FIELDS)
    def test_meter_tool_sends_attribution_field_from_the_ambient_context(self, endpoint, kwarg, value, wire_key):
        @meter_tool("firecrawl")
        def scrape():
            return "ok"

        with context(**{kwarg: value}):
            assert scrape() == "ok"

        assert posted_body(endpoint)[wire_key] == value

    @pytest.mark.parametrize("kwarg,value,wire_key", ATTRIBUTION_FIELDS)
    def test_meter_tool_sends_attribution_field_from_its_own_arguments(self, endpoint, kwarg, value, wire_key):
        @meter_tool("firecrawl", **{kwarg: value})
        async def scrape():
            return "ok"

        assert asyncio.run(scrape()) == "ok"

        assert posted_body(endpoint)[wire_key] == value

    def test_deprecated_organization_id_and_product_still_attribute(self, endpoint):
        report_tool_call(tool_id="firecrawl", organization_id="org-old", product="prod-old")

        body = posted_body(endpoint)
        assert body["organizationName"] == "org-old"
        assert body["productName"] == "prod-old"

    def test_context_holding_only_the_deprecated_fields_still_attributes(self, endpoint):
        # The public entry points copy the deprecated values into the new
        # fields before the event is built, so only a context that carries
        # the deprecated fields alone reaches the builder's fallback branch.
        legacy = ReveniumContext(organization_id="org-old", product="prod-old")
        asyncio.run(
            tool_metering._send_tool_event_async(
                "http://metering.test/meter/v2/tool/events",
                "rev_mk_test",
                tool_id="firecrawl",
                operation=None,
                duration_ms=1,
                success=True,
                error_message=None,
                usage_metadata=None,
                context=legacy,
                occurred_at=datetime.datetime.now(datetime.timezone.utc),
                event_sequence="test:1",
            )
        )

        body = posted_body(endpoint)
        assert body["organizationName"] == "org-old"
        assert body["productName"] == "prod-old"


def send_tool_event(url, tool_id="firecrawl"):
    return tool_metering._send_tool_event_async(
        url,
        "rev_mk_test",
        tool_id=tool_id,
        operation=None,
        duration_ms=1,
        success=True,
        error_message=None,
        usage_metadata=None,
        context=ReveniumContext(),
        occurred_at=datetime.datetime.now(datetime.timezone.utc),
        event_sequence="test:1",
    )


def test_consecutive_tool_events_reuse_one_connection(keep_alive_meter):
    for _ in range(10):
        asyncio.run(send_tool_event(keep_alive_meter.url))

    assert keep_alive_meter.wait_for_requests(10)
    assert len(keep_alive_meter.ports) == 10
    assert len(set(keep_alive_meter.ports)) == 1


def test_tool_events_through_the_pool_share_connections(keep_alive_meter):
    from revenium_middleware._metering._constants import DEFAULT_CONNECTION_LIMITS

    # Each caller waits until every event issued so far has arrived before it
    # issues its next one, so at most one event per caller is in flight. A
    # burst wider than the pool's keep-alive limit opens one connection per
    # concurrent request, which says nothing about reuse.
    issued = [0]
    issued_lock = threading.Lock()

    def report_some(count):
        for _ in range(count):
            with issued_lock:
                issued[0] += 1
                target = issued[0]
            report_tool_call(tool_id="pooled-tool", duration_ms=1, success=True)
            keep_alive_meter.wait_for_requests(target)

    configure(metering_url=keep_alive_meter.base_url, api_key="rev_mk_test")
    try:
        callers = [threading.Thread(target=report_some, args=(n,)) for n in [7] * 6 + [4, 4]]
        for caller in callers:
            caller.start()
        for caller in callers:
            caller.join()

        assert keep_alive_meter.wait_for_requests(50)
    finally:
        configure()

    assert len(keep_alive_meter.ports) == 50
    assert len(set(keep_alive_meter.ports)) <= DEFAULT_CONNECTION_LIMITS.max_keepalive_connections


def is_uuid(value):
    try:
        uuid.UUID(value)
    except (TypeError, ValueError):
        return False
    return True


class TestToolEventTransactionIds:
    """A transaction id set by the caller groups tool calls; each tool call still gets its own id."""

    @pytest.fixture(autouse=True)
    def isolated_context(self):
        clear_context()
        yield
        clear_context()

    def test_tool_calls_sharing_a_transaction_id_each_get_their_own_transaction_id(self, endpoint):
        @meter_tool("web-search")
        def search(query):
            return query

        @meter_tool("scrape")
        def scrape(url):
            return url

        with context(transaction_id="chatcmpl-abc123"):
            search("q")
            scrape("u")

        assert wait_for_calls(endpoint, count=2)
        first, second = endpoint.calls[0], endpoint.calls[1]
        assert first.json["transactionId"] != second.json["transactionId"]
        for call in (first, second):
            assert is_uuid(call.json["transactionId"])
            assert call.json["transactionId"] != "chatcmpl-abc123"
            assert call.json["traceId"] == "chatcmpl-abc123"
            assert call.headers["Idempotency-Key"] == call.json["transactionId"]

    def test_caller_transaction_id_moves_to_usage_metadata_when_a_trace_id_is_set(self, endpoint):
        caller_metadata = {"pages": 3}

        with context(transaction_id="t1", trace_id="tr1"):
            report_tool_call(tool_id="firecrawl", usage_metadata=caller_metadata)

        body = posted_body(endpoint)
        assert body["traceId"] == "tr1"
        assert body["usageMetadata"] == {"pages": 3, "callerTransactionId": "t1"}
        assert caller_metadata == {"pages": 3}
        assert is_uuid(body["transactionId"])

    def test_report_tool_call_with_a_transaction_id_sends_a_fresh_transaction_id_each_time(self, endpoint):
        report_tool_call(tool_id="firecrawl", transaction_id="txn-1")
        report_tool_call(tool_id="firecrawl", transaction_id="txn-1")

        assert wait_for_calls(endpoint, count=2)
        ids = [call.json["transactionId"] for call in endpoint.calls]
        assert len(set(ids)) == 2
        assert all(is_uuid(value) for value in ids)
        assert all(call.json["traceId"] == "txn-1" for call in endpoint.calls)

    def test_tool_event_without_a_transaction_id_still_gets_a_random_uuid(self, endpoint):
        @meter_tool("firecrawl")
        def scrape():
            return "ok"

        scrape()
        report_tool_call(tool_id="firecrawl")

        assert wait_for_calls(endpoint, count=2)
        for call in endpoint.calls:
            assert uuid.UUID(call.json["transactionId"]).version == 4
            assert "traceId" not in call.json
            assert call.headers["Idempotency-Key"] == call.json["transactionId"]

    def test_calls_stamped_with_the_same_instant_still_get_distinct_transaction_ids(self, endpoint, monkeypatch):
        monkeypatch.setattr(tool_metering, "datetime", FrozenClock)

        with context(transaction_id="txn-1"):
            report_tool_call(tool_id="firecrawl", operation="scrape")
            report_tool_call(tool_id="firecrawl", operation="scrape")

        assert wait_for_calls(endpoint, count=2)
        ids = [call.json["transactionId"] for call in endpoint.calls]
        assert len(set(ids)) == 2
        assert all(call.headers["Idempotency-Key"] == call.json["transactionId"] for call in endpoint.calls)

    def test_a_forked_child_restarts_the_event_sequence_under_a_new_nonce(self):
        parent_nonce = tool_metering._dispatch_nonce
        parent_lock = tool_metering._dispatch_lock
        tool_metering._next_event_sequence()

        tool_metering._restart_event_sequence()

        assert tool_metering._dispatch_nonce != parent_nonce
        assert tool_metering._dispatch_lock is not parent_lock
        assert tool_metering._next_event_sequence().endswith(":0")


# ---------------------------------------------------------------------------
# Streaming (generator) tools are recorded once, when the stream ends
# ---------------------------------------------------------------------------

STREAM_ITEM_SECONDS = 0.2
ONCE_GRACE_SECONDS = 0.3


def only_event(stub):
    """The single posted event body, after a grace that would let a second one land."""
    assert wait_for_calls(stub), "no tool event was posted"
    time.sleep(ONCE_GRACE_SECONDS)
    assert len(stub.calls) == 1, [call.json for call in stub.calls]
    return stub.calls[0].json


def no_event_yet(stub):
    time.sleep(ONCE_GRACE_SECONDS)
    return stub.calls == []


def streaming_tool(kind, items, fail_at=None, **meter_kwargs):
    decorate = meter_tool("streamer", **meter_kwargs)
    if kind == "sync":
        @decorate
        def tool():
            for index, item in enumerate(items):
                if index == fail_at:
                    raise RuntimeError("boom")
                time.sleep(STREAM_ITEM_SECONDS)
                yield item
        return tool

    @decorate
    async def async_tool():
        for index, item in enumerate(items):
            if index == fail_at:
                raise RuntimeError("boom")
            await asyncio.sleep(STREAM_ITEM_SECONDS)
            yield item
    return async_tool


def drain(kind, generator):
    if kind == "sync":
        return list(generator)

    async def collect():
        return [item async for item in generator]
    return asyncio.run(collect())


KIND = pytest.mark.parametrize("kind", ["sync", "async"])


class TestStreamingToolMetering:
    @pytest.fixture(autouse=True)
    def isolated_context(self, monkeypatch):
        monkeypatch.delenv(CAPTURE_ERROR_MESSAGES, raising=False)
        clear_context()
        yield
        clear_context()

    @KIND
    def test_streaming_tool_is_recorded_once_with_its_real_duration(self, endpoint, kind):
        generator = streaming_tool(kind, ["a", "b", "c"])()

        assert drain(kind, generator) == ["a", "b", "c"]
        if kind == "sync":
            generator.close()

        body = only_event(endpoint)
        assert body["durationMs"] >= 600
        assert body["success"] is True

    @KIND
    def test_streaming_tool_that_raises_mid_stream_is_recorded_once_as_failed(self, endpoint, kind):
        generator = streaming_tool(kind, [{"pages": 1}, {"pages": 2}], fail_at=1, output_fields=["pages"])()

        with pytest.raises(RuntimeError, match="boom"):
            drain(kind, generator)

        body = only_event(endpoint)
        assert body["success"] is False
        assert body["errorMessage"] == "RuntimeError"
        assert body["usageMetadata"] == {"pages": 1}

    def test_streaming_tool_failure_sends_its_message_when_the_customer_opts_in(self, endpoint, monkeypatch):
        monkeypatch.setenv(CAPTURE_ERROR_MESSAGES, "true")
        generator = streaming_tool("sync", ["a", "b"], fail_at=1)()

        with pytest.raises(RuntimeError):
            list(generator)

        body = only_event(endpoint)
        assert body["success"] is False
        assert "boom" in body["errorMessage"]

    @KIND
    def test_streaming_tool_closed_early_is_recorded_once(self, endpoint, kind):
        if kind == "sync":
            generator = streaming_tool(kind, ["a", "b", "c"])()
            assert next(generator) == "a"
            assert no_event_yet(endpoint)
            generator.close()
        else:
            async def take_one_then_close():
                generator = streaming_tool(kind, ["a", "b", "c"])()
                assert await generator.__anext__() == "a"
                assert no_event_yet(endpoint)
                await generator.aclose()
            asyncio.run(take_one_then_close())

        assert only_event(endpoint)["success"] is True

    def test_streaming_tool_never_iterated_sends_nothing(self, endpoint):
        generator = streaming_tool("sync", ["a"])()

        assert no_event_yet(endpoint)
        generator.close()
        assert no_event_yet(endpoint)

    @KIND
    def test_send_throw_and_close_reach_the_streaming_tool_body(self, endpoint, kind):
        seen = []

        if kind == "sync":
            @meter_tool("conversation")
            def conversation():
                try:
                    seen.append(("sent", (yield "first")))
                    try:
                        yield "second"
                    except KeyError:
                        seen.append("thrown")
                    yield "third"
                    yield "unreached"
                finally:
                    seen.append("finally")

            generator = conversation()
            assert next(generator) == "first"
            assert generator.send("hello") == "second"
            assert generator.throw(KeyError("boom")) == "third"
            generator.close()
        else:
            @meter_tool("conversation")
            async def conversation():
                try:
                    seen.append(("sent", (yield "first")))
                    try:
                        yield "second"
                    except KeyError:
                        seen.append("thrown")
                    yield "third"
                    yield "unreached"
                finally:
                    seen.append("finally")

            async def drive():
                generator = conversation()
                assert await generator.__anext__() == "first"
                assert await generator.asend("hello") == "second"
                assert await generator.athrow(KeyError("boom")) == "third"
                await generator.aclose()
            asyncio.run(drive())

        assert seen == [("sent", "hello"), "thrown", "finally"]
        assert only_event(endpoint)["success"] is True

    def test_streaming_tool_output_fields_come_from_the_last_item(self, endpoint):
        generator = streaming_tool("sync", [{"pages": 1}, {"pages": 3}], output_fields=["pages"])()

        list(generator)

        assert only_event(endpoint)["usageMetadata"] == {"pages": 3}

    def test_metered_streaming_tools_stay_generator_functions(self):
        assert inspect.isgeneratorfunction(streaming_tool("sync", []))
        assert inspect.isasyncgenfunction(streaming_tool("async", []))

    def test_streaming_tool_reads_the_ambient_context_at_its_first_step(self, endpoint):
        tool = streaming_tool("sync", ["a"])
        with context(trace_id="tr-1"):
            generator = tool()
        with context(trace_id="tr-2"):
            list(generator)

        assert only_event(endpoint)["traceId"] == "tr-2"

    def test_streaming_tool_keeps_the_job_active_at_its_first_step_when_drained_outside_it(self, endpoint):
        generator = streaming_tool("sync", ["a", "b"])()
        with ambient_job("job-a"):
            next(generator)
        list(generator)

        assert only_event(endpoint)["agenticJobId"] == "job-a"

    def test_streaming_tool_keeps_its_first_step_job_when_drained_inside_another_job(self, endpoint):
        generator = streaming_tool("sync", ["a", "b"])()
        with ambient_job("job-a"):
            next(generator)
        with ambient_job("job-b"):
            list(generator)

        assert only_event(endpoint)["agenticJobId"] == "job-a"

    def test_async_streaming_tool_keeps_the_job_active_at_its_first_step(self, endpoint):
        tool = streaming_tool("async", ["a", "b"])

        async def drive():
            generator = tool()
            with ambient_job("job-a"):
                await generator.__anext__()
            return [item async for item in generator]

        assert asyncio.run(drive()) == ["b"]
        assert only_event(endpoint)["agenticJobId"] == "job-a"

    def test_streaming_tool_output_job_id_beats_its_first_step_job(self, endpoint):
        generator = streaming_tool(
            "sync", [{"agentic_job_id": "job-last"}], output_fields=["agentic_job_id"]
        )()
        with ambient_job("job-a"):
            list(generator)

        assert only_event(endpoint)["agenticJobId"] == "job-last"

    def test_the_return_value_of_a_streaming_tool_reaches_a_yield_from_caller(self, endpoint):
        @meter_tool("returner")
        def tool():
            yield 1
            return "done"

        def outer():
            result = yield from tool()
            yield result

        assert list(outer()) == [1, "done"]
        assert only_event(endpoint)["success"] is True

    def test_revenium_metadata_over_a_streaming_meter_tool_tags_the_body(self, endpoint):
        @revenium_metadata(organizationName="ToolOrg")
        @meter_tool("tagged-streamer")
        def tool():
            yield (get_injected_metadata() or {}).get("organizationName")

        assert list(tool()) == ["ToolOrg"]
        assert get_injected_metadata() is None
        only_event(endpoint)



# A tool event holding a value JSON cannot write (BACK-4374). The stub
# endpoint above records the payload without writing it, so these tests post
# through a real httpx client on a MockTransport, which writes the body exactly
# as a live send would.

TOOL_LOGGERS = ("revenium_middleware", "revenium_metering")


class JsonWritingEndpoint:
    """Records every tool event body a real httpx client writes; ``statuses`` scripts the answers."""

    def __init__(self):
        self.bodies = []
        self.statuses = []

    def answer(self, request):
        status = self.statuses.pop(0) if self.statuses else 201
        if status == 201:
            self.bodies.append(json.loads(request.content))
        return httpx.Response(status, json={})


@pytest.fixture()
def json_writing_endpoint(monkeypatch):
    monkeypatch.delenv("REVENIUM_AGENTIC_JOB_ID", raising=False)
    meter = JsonWritingEndpoint()
    monkeypatch.setattr(tool_event_transport, "_client", httpx.Client(transport=httpx.MockTransport(meter.answer)))
    configure(metering_url="http://metering.test", api_key="rev_mk_test")
    yield meter
    configure()  # clear overrides; resolution falls back to the environment


@pytest.fixture()
def isolated_buffer(monkeypatch):
    buf = MeteringBuffer(flush_interval=3600, replay_fn=lambda event, timeout: None)
    monkeypatch.setattr(metering_buffer, "_buffer", buf)
    return buf


@pytest.fixture(autouse=True)
def forget_json_safe_warnings():
    try:
        from revenium_middleware._core import json_safe
    except ImportError:
        yield
        return
    json_safe._warn_once.cache_clear()
    json_safe._warn_once_sent_as_given.cache_clear()
    yield
    json_safe._warn_once.cache_clear()
    json_safe._warn_once_sent_as_given.cache_clear()


@contextlib.contextmanager
def tool_logs(caplog, level=logging.WARNING):
    with contextlib.ExitStack() as stack:
        for name in TOOL_LOGGERS:
            stack.enter_context(caplog.at_level(level, logger=name))
        yield


def wait_until(condition, timeout=6.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.05)
    return False


def wait_for_bodies(meter, count=1, timeout=6.0):
    return wait_until(lambda: len(meter.bodies) >= count, timeout)


def as_json(value):
    """Compare as written JSON, so 8 and 8.0 differ."""
    return json.dumps(value, sort_keys=True)


SAMPLE_UUID = uuid.UUID("12345678-1234-5678-1234-567812345678")
VIDEO_OUTPUT_CASES = [
    pytest.param({"duration_seconds": Decimal("8.0"), "resolution": "720p"},
                 {"duration_seconds": 8, "resolution": "720p"}, id="decimal-whole"),
    pytest.param({"duration_seconds": Decimal("8.5"), "resolution": "720p"},
                 {"duration_seconds": 8.5, "resolution": "720p"}, id="decimal-fraction"),
    pytest.param({"duration_seconds": datetime.datetime(2026, 10, 11, 12, 30, tzinfo=datetime.timezone.utc),
                  "resolution": "720p"},
                 {"duration_seconds": "2026-10-11T12:30:00+00:00", "resolution": "720p"}, id="datetime"),
    pytest.param({"duration_seconds": SAMPLE_UUID, "resolution": "720p"},
                 {"duration_seconds": str(SAMPLE_UUID), "resolution": "720p"}, id="uuid"),
    pytest.param({"duration_seconds": float("nan"), "resolution": "720p"},
                 {"resolution": "720p"}, id="nan"),
    pytest.param({"duration_seconds": 8, "resolution": "720p\ud83d"},
                 {"duration_seconds": 8, "resolution": "720p\ufffd"}, id="half-emoji"),
]


@pytest.mark.parametrize("output, expected_usage", VIDEO_OUTPUT_CASES)
def test_meter_tool_sends_its_event_when_an_output_value_is_not_plain_json(json_writing_endpoint, output,
                                                                           expected_usage):
    @meter_tool("video-maker", output_fields=["duration_seconds", "resolution"])
    def make_video():
        return output

    returned = make_video()

    assert returned is output
    assert wait_for_bodies(json_writing_endpoint), "no tool event was posted"
    body = json_writing_endpoint.bodies[0]
    assert body["toolId"] == "video-maker"
    assert body["success"] is True
    assert as_json(body["usageMetadata"]) == as_json(expected_usage)


def test_report_tool_call_sends_its_event_when_usage_metadata_holds_a_decimal(json_writing_endpoint):
    usage = {"records": Decimal("42")}

    report_tool_call("record-counter", usage_metadata=usage)

    assert wait_for_bodies(json_writing_endpoint), "no tool event was posted"
    assert as_json(json_writing_endpoint.bodies[0]["usageMetadata"]) == as_json({"records": 42})
    assert usage == {"records": Decimal("42")}


def test_report_tool_call_sends_a_large_plain_usage_metadata_exactly_as_given(json_writing_endpoint):
    usage = {"rows": list(range(10_001)), "records": 10_001}

    report_tool_call("record-counter", usage_metadata=usage)

    assert wait_for_bodies(json_writing_endpoint), "no tool event was posted"
    assert json_writing_endpoint.bodies[0]["usageMetadata"] == usage


def test_failed_tool_with_half_an_emoji_in_its_error_text_still_sends_the_failure(json_writing_endpoint,
                                                                                  monkeypatch):
    monkeypatch.setenv("REVENIUM_CAPTURE_ERROR_MESSAGES", "true")

    @meter_tool("video-maker")
    def make_video():
        raise RuntimeError("bad row Jos\ud83d")

    with pytest.raises(RuntimeError):
        make_video()

    assert wait_for_bodies(json_writing_endpoint), "no tool event was posted"
    body = json_writing_endpoint.bodies[0]
    assert body["success"] is False
    assert body["errorMessage"] == "RuntimeError: bad row Jos\ufffd"


def test_a_dropped_tool_output_value_is_warned_about_by_tool_and_field_never_by_value(json_writing_endpoint,
                                                                                     caplog):
    @meter_tool("video-maker", output_fields=["duration_seconds", "resolution"])
    def make_video():
        return {"duration_seconds": float("nan"), "resolution": "sk-live-secret\ud83d"}

    with tool_logs(caplog):
        make_video()
        assert wait_for_bodies(json_writing_endpoint), "no tool event was posted"

    messages = [record.getMessage() for record in caplog.records]
    for field in ("usageMetadata.duration_seconds", "usageMetadata.resolution"):
        assert any(field in message and "video-maker" in message for message in messages), messages
    assert all("sk-live-secret" not in message for message in messages), messages


def test_a_tool_event_delivery_error_line_names_the_tool(json_writing_endpoint, caplog):
    json_writing_endpoint.statuses = [401]

    @meter_tool("video-maker")
    def make_video():
        return "ok"

    with tool_logs(caplog, logging.ERROR):
        make_video()
        assert wait_until(lambda: "metering error" in caplog.text), "no metering error was logged"

    assert "metering error for tool video-maker" in caplog.text, caplog.text


def test_a_tool_event_buffered_after_a_503_is_already_clean(json_writing_endpoint, isolated_buffer):
    json_writing_endpoint.statuses = [503]

    @meter_tool("video-maker", output_fields=["duration_seconds", "resolution"])
    def make_video():
        return {"duration_seconds": Decimal("8.0"), "resolution": "720p"}

    make_video()

    assert wait_until(lambda: isolated_buffer.stats()["size"] == 1), "the tool event was not buffered"
    (event,) = list(isolated_buffer._events)
    assert event.kind == "tool"
    assert as_json(event.payload["event_payload"]["usageMetadata"]) == as_json(
        {"duration_seconds": 8, "resolution": "720p"})
