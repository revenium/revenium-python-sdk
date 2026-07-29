"""JobContext public surface (BACK-777 Phase 2)."""
import json
import os
from unittest.mock import patch

import httpx
import pytest

from revenium_middleware import JobContext, OutcomeAlreadyReportedError, OutcomeReportingError
from revenium_middleware._core.fields import extract_agentic_job_fields

WRITE_KEY = "rev_sk_TENANT_abc"
ENV = {"REVENIUM_OUTCOME_API_KEY": WRITE_KEY, "REVENIUM_TEAM_ID": "team-1"}


def _recording_client(status=200, body=None):
    calls = []

    def handler(req: httpx.Request) -> httpx.Response:
        calls.append(req)
        return httpx.Response(status, json=body if body is not None else {})

    return httpx.Client(transport=httpx.MockTransport(handler)), calls


class TestContextPropagation:
    def test_fields_visible_inside_and_gone_after(self):
        with patch.dict(os.environ, {}, clear=True):
            with JobContext(job_id="loan-1", name="Loan", type="loan", version="1.0"):
                assert extract_agentic_job_fields({}) == {
                    "agenticJobId": "loan-1",
                    "agenticJobName": "Loan",
                    "agenticJobType": "loan",
                    "agenticJobVersion": "1.0",
                }
            assert extract_agentic_job_fields({}) == {}

    def test_nesting_replaces_not_merges(self):
        with patch.dict(os.environ, {}, clear=True):
            with JobContext(job_id="outer", name="Outer"):
                with JobContext(job_id="inner"):
                    fields = extract_agentic_job_fields({})
                    assert fields == {"agenticJobId": "inner"}  # no inherited name
                assert extract_agentic_job_fields({})["agenticJobId"] == "outer"

    @pytest.mark.asyncio
    async def test_async_context_manager(self):
        with patch.dict(os.environ, {}, clear=True):
            async with JobContext(job_id="async-1"):
                assert extract_agentic_job_fields({})["agenticJobId"] == "async-1"
            assert extract_agentic_job_fields({}) == {}

    def test_job_id_must_be_non_empty_string(self):
        with pytest.raises(ValueError):
            JobContext(job_id="")
        with pytest.raises(ValueError):
            JobContext(job_id="   ")
        with pytest.raises(ValueError):
            JobContext(job_id=None)


class TestReportOutcome:
    def test_posts_camel_case_payload_with_team_and_key(self):
        http, calls = _recording_client()
        with patch.dict(os.environ, ENV):
            with JobContext(job_id="loan-1", http_client=http) as job:
                job.report_outcome(
                    execution_status="SUCCESS",
                    outcome_type="CONVERTED",
                    outcome_value=500.0,
                    outcome_currency="USD",
                    metadata={"notes": "approved"},
                    reported_by="orchestrator",
                )
        assert len(calls) == 1
        req = calls[0]
        assert req.url.path.endswith("/profitstream/v2/api/jobs/loan-1/outcome")
        assert req.url.params["teamId"] == "team-1"
        assert req.headers["x-api-key"] == WRITE_KEY
        body = json.loads(req.content)
        assert body["executionStatus"] == "SUCCESS"
        assert body["outcomeType"] == "CONVERTED"
        assert body["outcomeValue"] == 500.0
        assert body["outcomeCurrency"] == "USD"
        assert json.loads(body["metadata"]) == {"notes": "approved"}
        assert body["reportedBy"] == "orchestrator"

    def test_invalid_execution_status_rejected(self):
        with patch.dict(os.environ, ENV):
            with JobContext(job_id="j1") as job:
                with pytest.raises(ValueError):
                    job.report_outcome(execution_status="DONE")

    def test_outcome_value_requires_outcome_type(self):
        with patch.dict(os.environ, ENV):
            with JobContext(job_id="j1") as job:
                with pytest.raises(ValueError):
                    job.report_outcome(execution_status="SUCCESS", outcome_value=10.0)

    def test_metering_key_fails_fast_before_http(self):
        http, calls = _recording_client()
        env = {"REVENIUM_OUTCOME_API_KEY": "rev_mk_TENANT_abc", "REVENIUM_TEAM_ID": "team-1"}
        with patch.dict(os.environ, env):
            with JobContext(job_id="j1", http_client=http) as job:
                with pytest.raises(ValueError, match="write-scope"):
                    job.report_outcome(execution_status="SUCCESS")
        assert calls == []

    def test_missing_key_raises_reporting_error(self):
        with patch.dict(os.environ, {}, clear=True):
            with JobContext(job_id="j1", team_id="team-1") as job:
                with pytest.raises(OutcomeReportingError):
                    job.report_outcome(execution_status="SUCCESS")

    def test_unresolvable_team_raises_reporting_error(self):
        http, _ = _recording_client(status=503)
        with patch.dict(os.environ, {"REVENIUM_OUTCOME_API_KEY": "bogus_key"}, clear=True):
            with JobContext(job_id="j1", http_client=http) as job:
                with pytest.raises(OutcomeReportingError):
                    job.report_outcome(execution_status="SUCCESS")

    def test_structured_409_raises_typed(self):
        # The body the backend actually sends: conflict fields nested under
        # "details", count named updateCount and serialized as a string.
        body = {
            "status": 409,
            "error": "Conflict",
            "message": "Outcome already reported",
            "details": {
                "guidance": "Use PATCH /v2/api/jobs/{jobId}/outcome to update",
                "reportedAt": "2026-04-02T10:00:00Z",
                "updateCount": "1",
            },
        }
        http, _ = _recording_client(status=409, body=body)
        with patch.dict(os.environ, ENV):
            with JobContext(job_id="j1", http_client=http) as job:
                with pytest.raises(OutcomeAlreadyReportedError) as exc_info:
                    job.report_outcome(execution_status="SUCCESS")
        assert exc_info.value.amendment_count == 1
        assert isinstance(exc_info.value.amendment_count, int)
        assert exc_info.value.reported_at == "2026-04-02T10:00:00Z"
        assert str(exc_info.value) == "Outcome already reported"


class TestAutoFailed:
    def test_exception_auto_reports_failed_and_propagates(self):
        http, calls = _recording_client()
        with patch.dict(os.environ, ENV):
            with pytest.raises(RuntimeError, match="boom"):
                with JobContext(job_id="j1", http_client=http):
                    raise RuntimeError("boom")
        assert len(calls) == 1
        body = json.loads(calls[0].content)
        assert body["executionStatus"] == "FAILED"
        meta = json.loads(body["metadata"])
        assert meta["error"] == "boom"
        assert meta["errorType"] == "RuntimeError"

    def test_no_auto_report_when_outcome_already_reported(self):
        http, calls = _recording_client()
        with patch.dict(os.environ, ENV):
            with pytest.raises(RuntimeError):
                with JobContext(job_id="j1", http_client=http) as job:
                    job.report_outcome(execution_status="SUCCESS")
                    raise RuntimeError("after report")
        assert len(calls) == 1  # only the explicit SUCCESS report

    def test_reporting_failure_never_masks_user_exception(self, caplog):
        import logging
        http, _ = _recording_client(status=500)
        with patch.dict(os.environ, ENV):
            with caplog.at_level(logging.WARNING):
                with pytest.raises(RuntimeError, match="original"):
                    with JobContext(job_id="j1", http_client=http):
                        raise RuntimeError("original")
        assert "FAILED" in caplog.text or "auto-report" in caplog.text.lower()

    def test_no_auto_report_on_clean_exit(self):
        http, calls = _recording_client()
        with patch.dict(os.environ, ENV):
            with JobContext(job_id="j1", http_client=http):
                pass
        assert calls == []

    def test_keyboard_interrupt_skips_auto_report(self):
        http, calls = _recording_client()
        with patch.dict(os.environ, ENV):
            with pytest.raises(KeyboardInterrupt):
                with JobContext(job_id="j1", http_client=http):
                    raise KeyboardInterrupt()
        assert calls == []
        assert extract_agentic_job_fields({}) == {}

    def test_reenter_resets_outcome_reported(self):
        http, calls = _recording_client()
        with patch.dict(os.environ, ENV):
            job = JobContext(job_id="j1", http_client=http)
            with job as j:
                j.report_outcome(execution_status="SUCCESS")
            with pytest.raises(RuntimeError):
                with job:
                    raise RuntimeError("second run")
        # one explicit SUCCESS + one auto-FAILED from the second run
        assert len(calls) == 2

    def test_context_reset_even_when_auto_report_fails(self):
        http, _ = _recording_client(status=500)
        with patch.dict(os.environ, ENV):
            with pytest.raises(RuntimeError):
                with JobContext(job_id="j1", http_client=http):
                    raise RuntimeError("x")
        assert extract_agentic_job_fields({}) == {}

    def test_409_inside_block_does_not_trigger_auto_failed(self):
        body = {
            "error": "Outcome already reported",
            "guidance": "Use PATCH /v2/api/jobs/{id}/outcome to amend",
            "reportedAt": "2026-04-02T10:00:00Z",
            "amendmentCount": 1,
        }
        http, calls = _recording_client(status=409, body=body)
        with patch.dict(os.environ, ENV):
            with pytest.raises(OutcomeAlreadyReportedError):
                with JobContext(job_id="j1", http_client=http) as job:
                    job.report_outcome(execution_status="SUCCESS")
        assert len(calls) == 1  # no doomed auto-FAILED second POST

    def test_failed_explicit_report_suppresses_auto_failed(self):
        http, calls = _recording_client(status=500)
        with patch.dict(os.environ, ENV):
            with pytest.raises(httpx.HTTPStatusError):
                with JobContext(job_id="j1", http_client=http) as job:
                    job.report_outcome(execution_status="SUCCESS")
        assert len(calls) == 1  # only the user's failed attempt; no FAILED overwrite

    @pytest.mark.asyncio
    async def test_async_auto_failed_reports_and_propagates(self):
        http, calls = _recording_client()
        with patch.dict(os.environ, ENV):
            with pytest.raises(RuntimeError, match="boom"):
                async with JobContext(job_id="j1", http_client=http):
                    raise RuntimeError("boom")
        assert len(calls) == 1
        body = json.loads(calls[0].content)
        assert body["executionStatus"] == "FAILED"
        assert extract_agentic_job_fields({}) == {}

    def test_reentering_active_instance_raises(self):
        with patch.dict(os.environ, {}, clear=True):
            job = JobContext(job_id="outer-job")
            with job:
                with pytest.raises(RuntimeError, match="already active"):
                    with job:
                        pass
            assert extract_agentic_job_fields({}) == {}  # outer exit restored cleanly

    def test_reenter_resets_resolved_team_cache(self):
        http, calls = _recording_client()
        with patch.dict(os.environ, ENV):
            job = JobContext(job_id="j1", http_client=http)
            with job as j:
                j.report_outcome(execution_status="SUCCESS")
            job._resolved_team_id = "stale-team"
            with job:
                assert job._resolved_team_id is None


class TestAmendOutcome:
    def test_attach_then_amend_patches_camel_case(self):
        seen = {}

        def handler(req: httpx.Request) -> httpx.Response:
            seen["method"] = req.method
            seen["body"] = json.loads(req.content)
            return httpx.Response(200, json={"id": "job-1", "outcomeAmendmentCount": 2})

        http = httpx.Client(transport=httpx.MockTransport(handler))
        with patch.dict(os.environ, ENV):
            handle = JobContext.attach(job_id="loan-1", http_client=http)
            result = handle.amend_outcome(
                reason="Customer expanded contract",
                outcome_value=750.0,
                metadata={"expansion_event": "upsell_q2"},
            )
        assert seen["method"] == "PATCH"
        assert seen["body"]["reason"] == "Customer expanded contract"
        assert seen["body"]["outcomeValue"] == 750.0
        assert json.loads(seen["body"]["metadata"]) == {"expansion_event": "upsell_q2"}
        assert result["outcomeAmendmentCount"] == 2

    def test_blank_reason_rejected_before_http(self):
        http, calls = _recording_client()
        with patch.dict(os.environ, ENV):
            handle = JobContext.attach(job_id="j1", http_client=http)
            with pytest.raises(ValueError):
                handle.amend_outcome(reason="")
            with pytest.raises(ValueError):
                handle.amend_outcome(reason="   ")
        assert calls == []

    def test_422_maps_to_not_reported(self):
        http, _ = _recording_client(status=422, body={"error": "no outcome"})
        with patch.dict(os.environ, ENV):
            handle = JobContext.attach(job_id="j1", http_client=http)
            from revenium_middleware import OutcomeNotReportedError
            with pytest.raises(OutcomeNotReportedError):
                handle.amend_outcome(reason="r")

    def test_409_maps_to_amend_conflict(self):
        http, _ = _recording_client(status=409, body={"error": "conflict"})
        with patch.dict(os.environ, ENV):
            handle = JobContext.attach(job_id="j1", http_client=http)
            from revenium_middleware import OutcomeAmendConflictError
            with pytest.raises(OutcomeAmendConflictError):
                handle.amend_outcome(reason="r")

    def test_invalid_execution_status_rejected(self):
        with patch.dict(os.environ, ENV):
            handle = JobContext.attach(job_id="j1")
            with pytest.raises(ValueError):
                handle.amend_outcome(reason="r", execution_status="DONE")

    def test_attach_does_not_touch_context(self):
        with patch.dict(os.environ, {}, clear=True):
            JobContext.attach(job_id="j1")
            assert extract_agentic_job_fields({}) == {}


class TestRetryKnobs:
    def test_attach_knobs_reach_transport(self):
        """attach() handles must be tunable too — CrewAI builds them this way."""
        counter = {"n": 0}

        def handler(req):
            counter["n"] += 1
            return httpx.Response(502, json={})

        http = httpx.Client(transport=httpx.MockTransport(handler))
        with patch.dict(os.environ, ENV):
            handle = JobContext.attach(
                job_id="j1", http_client=http,
                retry_attempts=2, retry_initial_seconds=0.01, retry_max_seconds=0.02,
            )
            with pytest.raises(httpx.HTTPStatusError):
                handle.report_outcome(execution_status="SUCCESS")
        assert counter["n"] == 2  # not the 10-attempt default schedule

    def test_knobs_reach_transport(self):
        counter = {"n": 0}

        def handler(req):
            counter["n"] += 1
            return httpx.Response(502, json={})

        http = httpx.Client(transport=httpx.MockTransport(handler))
        with patch.dict(os.environ, ENV):
            job = JobContext(
                job_id="j1", http_client=http,
                retry_attempts=2, retry_initial_seconds=0.01, retry_max_seconds=0.02,
            )
            with pytest.raises(httpx.HTTPStatusError):
                job.report_outcome(execution_status="SUCCESS")
        assert counter["n"] == 2  # attempts honored, sub-second sleeps

    @pytest.mark.asyncio
    async def test_knobs_reach_the_async_auto_report(self):
        """The async auto-report worker must honor the knobs too.

        It no longer goes through report_outcome(), and __aexit__ awaits it, so a
        dropped knob shows up as duration rather than attempt count: the retry
        sleeps ~0.01s with the configured backoff and ~2s with the default one.
        """
        import time as _time

        counter = {"n": 0}

        def handler(req):
            counter["n"] += 1
            status = 502 if counter["n"] == 1 else 200
            return httpx.Response(status, json={})

        http = httpx.Client(transport=httpx.MockTransport(handler))
        with patch.dict(os.environ, ENV):
            job = JobContext(
                job_id="j1", http_client=http,
                retry_attempts=3, retry_initial_seconds=0.01, retry_max_seconds=0.02,
            )
            started = _time.monotonic()
            with pytest.raises(RuntimeError, match="boom"):
                async with job:
                    raise RuntimeError("boom")
            elapsed = _time.monotonic() - started
        assert counter["n"] == 2  # retried once
        # 50x the configured backoff, 4x under the default first sleep (2s).
        assert elapsed < 0.5, f"retry knobs ignored by the worker ({elapsed:.2f}s)"


class _TrackingClient(httpx.Client):
    """httpx client that counts close() calls (ownership assertions)."""

    def __init__(self, *args, **kwargs):
        self.closed_count = 0
        super().__init__(*args, **kwargs)

    def close(self):
        self.closed_count += 1
        super().close()


def _tracking_client(status=200, body=None):
    calls = []

    def handler(req: httpx.Request) -> httpx.Response:
        calls.append(req)
        return httpx.Response(status, json=body if body is not None else {})

    return _TrackingClient(transport=httpx.MockTransport(handler)), calls


class TestConcurrencyAndCancellation:
    def _run_concurrent_enter(self, job):
        """Enter ``job`` from two threads at once; return (entered, errors)."""
        import threading as _t

        errors, entered = [], []
        barrier = _t.Barrier(2)
        # Held by whoever enters until both threads have decided, so a fast
        # winner cannot exit before the loser runs its check.
        decided = _t.Barrier(2)

        def worker():
            barrier.wait(timeout=10)
            try:
                with job:
                    entered.append(True)
                    decided.wait(timeout=10)
            except RuntimeError:
                errors.append(True)
                decided.wait(timeout=10)

        threads = [_t.Thread(target=worker) for _ in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=15)
        return entered, errors

    def test_concurrent_enter_is_atomic(self):
        """The check-and-set must be atomic even under aggressive preemption."""
        import sys as _s

        previous = _s.getswitchinterval()
        _s.setswitchinterval(1e-9)  # force interleaving inside the guard
        try:
            with patch.dict(os.environ, {}, clear=True):
                for _ in range(100):
                    job = JobContext(job_id="shared-job")
                    entered, errors = self._run_concurrent_enter(job)
                    # exactly one entered; the other was rejected deterministically
                    assert len(entered) == 1 and len(errors) == 1
                    assert job._token is None  # the winner's exit cleaned up
                    assert extract_agentic_job_fields({}) == {}
        finally:
            _s.setswitchinterval(previous)

    def test_concurrent_enter_guard_covers_the_token_assignment(self):
        """Deterministic form: the token assignment must be inside the guard.

        Slowing ``set_agentic_job_fields`` widens the check-then-set window, so
        an unsynchronised guard lets both threads enter every time.
        """
        import time as _time

        from revenium_middleware import job_context as _jc

        real_set = _jc.set_agentic_job_fields

        def slow_set(*args, **kwargs):
            _time.sleep(0.05)
            return real_set(*args, **kwargs)

        with patch.dict(os.environ, {}, clear=True):
            with patch.object(_jc, "set_agentic_job_fields", slow_set):
                job = JobContext(job_id="shared-job")
                entered, errors = self._run_concurrent_enter(job)
        assert len(entered) == 1 and len(errors) == 1
        assert job._token is None
        assert extract_agentic_job_fields({}) == {}

    @pytest.mark.asyncio
    async def test_cancellation_during_auto_report_does_not_race_cleanup(self):
        http, calls = _tracking_client()
        with patch.dict(os.environ, ENV):
            # owned=False for an injected client: the SDK must never close it
            with pytest.raises(RuntimeError, match="boom"):
                async with JobContext(job_id="j1", http_client=http):
                    raise RuntimeError("boom")
        assert len(calls) == 1
        assert http.closed_count == 0  # injected client untouched
        assert extract_agentic_job_fields({}) == {}

    @pytest.mark.asyncio
    async def test_owned_client_closed_once_by_async_auto_report(self):
        http, calls = _tracking_client()
        with patch.dict(os.environ, ENV):
            job = JobContext(job_id="j1")  # no injected client -> SDK-owned
            assert job._owns_http_client
            job._http_client = http  # stand in for the lazily created client
            with pytest.raises(RuntimeError, match="boom"):
                async with job:
                    raise RuntimeError("boom")
        assert len(calls) == 1
        assert http.closed_count == 1  # closed exactly once, by the worker
        assert job._http_client is None
        assert extract_agentic_job_fields({}) == {}

    @pytest.mark.asyncio
    async def test_reenter_restores_ownership_of_sdk_created_client(self):
        """Sequential reuse must not leak a client per run.

        The async auto-report hands ownership to the worker, so re-entry has to
        restore it or every later run creates a client nobody closes.
        """
        first, first_calls = _tracking_client()
        second, second_calls = _tracking_client()
        observed = {}
        with patch.dict(os.environ, ENV):
            job = JobContext(job_id="j1")  # no injected client -> SDK-owned
            job._http_client = first  # stand in for the lazily created client
            with pytest.raises(RuntimeError, match="first"):
                async with job:
                    raise RuntimeError("first")
            observed["between_runs"] = job._owns_http_client
            with pytest.raises(RuntimeError, match="second"):
                async with job:
                    observed["second_run"] = job._owns_http_client
                    job._http_client = second
                    raise RuntimeError("second")
        assert observed["between_runs"] is False  # transferred to the worker
        assert observed["second_run"] is True  # ...and restored on re-entry
        assert len(first_calls) == 1 and first.closed_count == 1
        assert len(second_calls) == 1 and second.closed_count == 1

    @pytest.mark.asyncio
    async def test_reenter_keeps_using_the_injected_client(self):
        """A user-supplied client stays the client for every run.

        The worker detaches it when it hands the client back, so re-entry must
        restore it instead of fabricating (and leaking) a replacement.
        """
        http, calls = _tracking_client()
        observed = {}
        with patch.dict(os.environ, ENV):
            job = JobContext(job_id="j1", http_client=http)
            with pytest.raises(RuntimeError, match="boom"):
                async with job:
                    raise RuntimeError("boom")
            with job as j:
                observed["client_is_injected"] = job._http_client is http
                observed["owned"] = job._owns_http_client
                # Guarded so a regression cannot fall through to a real client.
                if observed["client_is_injected"]:
                    j.report_outcome(execution_status="SUCCESS")
        assert observed["client_is_injected"] is True
        assert observed["owned"] is False  # never owned, on any run
        assert len(calls) == 2  # both reports went over the injected client
        assert http.closed_count == 0  # and it is still never closed by the SDK

    def test_exit_cleanup_is_atomic_against_reentry(self):
        """A run's teardown must not close the client of a concurrent new run.

        The token is what __enter__ gates on, so if it is released before the
        owned-client cleanup, a second thread can enter mid-teardown and have its
        client closed by the outgoing run.
        """
        import threading as _t
        import time as _time

        from revenium_middleware import job_context as _jc

        created = []
        calls = []

        def handler(req: httpx.Request) -> httpx.Response:
            calls.append(req)
            return httpx.Response(200, json={})

        def client_factory(*args, **kwargs):
            client = _TrackingClient(transport=httpx.MockTransport(handler))
            created.append(client)
            return client

        real_close = JobContext.close
        in_close = _t.Event()

        def slow_close(self):
            in_close.set()
            _time.sleep(0.2)  # widen the teardown window
            return real_close(self)

        observed = {}

        with patch.dict(os.environ, ENV):
            with patch.object(_jc.httpx, "Client", client_factory):
                with patch.object(JobContext, "close", slow_close):
                    job = JobContext(job_id="j1")  # SDK-owned client

                    def run_a():
                        with job:
                            pass

                    thread_a = _t.Thread(target=run_a)
                    thread_a.start()
                    assert in_close.wait(5), "thread A never reached its cleanup"

                    # Thread B (here) enters while A is inside its teardown.
                    with job as j:
                        j.report_outcome(execution_status="SUCCESS")
                        b_client = job._http_client
                        observed["closed_on_entry"] = b_client.closed_count
                        thread_a.join(timeout=5)  # A's cleanup fully completes
                        observed["closed_after_a"] = b_client.closed_count
                        observed["still_attached"] = job._http_client is b_client
                        j.report_outcome(execution_status="SUCCESS")

        assert observed["closed_on_entry"] == 0
        assert observed["closed_after_a"] == 0  # A must not close B's client
        assert observed["still_attached"] is True  # nor detach it
        assert len(created) == 1  # B never had to fabricate a replacement
        assert len(calls) == 2  # both of B's reports went over its own client
        assert created[0].closed_count == 1  # closed once, by B's own exit

    @pytest.mark.asyncio
    async def test_detached_worker_does_not_touch_a_new_run(self):
        """A worker left running by a cancellation must not touch instance state.

        __aexit__ resets the token, so the instance can legitimately be
        re-entered while the previous run's auto-report is still in flight.
        """
        import asyncio as _a
        import threading as _t

        from revenium_middleware import job_context as _jc

        in_first, release_first = _t.Event(), _t.Event()
        first_calls = []

        def first_handler(req: httpx.Request) -> httpx.Response:
            first_calls.append(req)
            in_first.set()
            release_first.wait(timeout=5)
            return httpx.Response(200, json={})

        first = _TrackingClient(transport=httpx.MockTransport(first_handler))
        second, second_calls = _tracking_client()
        observed = {}

        def no_new_clients(*args, **kwargs):
            raise AssertionError("the SDK fabricated a replacement client")

        async def run_first(job):
            async with job:
                raise RuntimeError("first")

        with patch.dict(os.environ, ENV):
            # Guard: nothing here may fall through to a real httpx client.
            with patch.object(_jc.httpx, "Client", no_new_clients):
                job = JobContext(job_id="j1")  # SDK-owned client
                job._http_client = first
                task = _a.ensure_future(run_first(job))
                loop = _a.get_event_loop()
                # The first run's worker is now inside its outcome POST.
                await loop.run_in_executor(None, in_first.wait, 5)
                task.cancel()
                with pytest.raises(_a.CancelledError):
                    await task

                # Second run starts while the first worker is still in flight.
                with job as j:
                    job._http_client = second
                    observed["owned_on_reentry"] = job._owns_http_client
                    release_first.set()  # let the stale worker finish
                    for _ in range(250):
                        if first.closed_count:
                            break
                        await _a.sleep(0.02)
                    observed["client_kept"] = job._http_client is second
                    observed["reported_flag"] = j._outcome_reported
                    observed["attempted_flag"] = j._outcome_attempted
                    raise_in_second = RuntimeError("second")
                    try:
                        raise raise_in_second
                    except RuntimeError:
                        j._auto_report_failed(raise_in_second)

        assert observed["owned_on_reentry"] is True
        assert observed["client_kept"] is True  # stale worker did not clobber it
        assert observed["reported_flag"] is False  # nor the new run's flags
        assert observed["attempted_flag"] is False
        assert len(first_calls) == 1 and first.closed_count == 1
        # The new run's own report went over the new run's client.
        assert len(second_calls) == 1
        assert json.loads(second_calls[0].content)["executionStatus"] == "FAILED"
        assert second.closed_count == 1  # closed once, by the new run's own exit

    @pytest.mark.asyncio
    async def test_cancelled_report_leaves_injected_client_attached(self):
        """The worker must not detach or close a caller-supplied client."""
        import asyncio as _a
        import threading as _t

        in_handler, release = _t.Event(), _t.Event()
        calls, finished = [], []

        def handler(req: httpx.Request) -> httpx.Response:
            calls.append(req)
            in_handler.set()
            release.wait(timeout=5)
            finished.append(True)  # the POST is done; the worker is unwinding
            return httpx.Response(200, json={})

        http = _TrackingClient(transport=httpx.MockTransport(handler))

        async def run(job):
            async with job:
                raise RuntimeError("boom")

        with patch.dict(os.environ, ENV):
            job = JobContext(job_id="j1", http_client=http)
            task = _a.ensure_future(run(job))
            loop = _a.get_event_loop()
            await loop.run_in_executor(None, in_handler.wait, 5)
            task.cancel()
            with pytest.raises(_a.CancelledError):
                await task
            release.set()
            for _ in range(250):
                if finished:
                    break
                await _a.sleep(0.02)
            assert finished, "worker never completed its POST"
            # Settle: the worker's remaining statements are microseconds away, so
            # any write-back to the instance shows up well within this window.
            await _a.sleep(0.2)
        assert len(calls) == 1
        assert job._http_client is http  # never detached from the instance
        assert job._owns_http_client is False  # and never adopted
        assert http.closed_count == 0  # never closed on the caller's behalf

    @pytest.mark.asyncio
    async def test_cancel_mid_report_leaves_close_to_the_worker(self):
        import asyncio as _a
        import threading as _t

        in_handler = _t.Event()
        release = _t.Event()
        calls = []
        seen = {}

        def handler(req: httpx.Request) -> httpx.Response:
            calls.append(req)
            in_handler.set()
            release.wait(timeout=5)
            # Recorded after the caller has already been cancelled: the client
            # carrying this in-flight request must still be usable.
            seen["closed_mid_flight"] = http.is_closed
            return httpx.Response(200, json={})

        http = _TrackingClient(transport=httpx.MockTransport(handler))

        async def run():
            with patch.dict(os.environ, ENV):
                async with JobContext(job_id="j1") as job:
                    job._http_client = http  # SDK-owned tracking client
                    raise RuntimeError("boom")

        task = _a.ensure_future(run())
        loop = _a.get_event_loop()
        # Wait off-loop until the worker thread is inside the outcome POST.
        await loop.run_in_executor(None, in_handler.wait, 5)
        assert in_handler.is_set()
        task.cancel()
        # Let the caller finish unwinding (its cleanup runs here) before the
        # worker's request is allowed to complete.
        with pytest.raises(_a.CancelledError):
            await task
        release.set()
        # The worker owns the close; poll briefly for it to finish.
        for _ in range(250):
            if "closed_mid_flight" in seen and http.closed_count:
                break
            await _a.sleep(0.02)
        assert calls  # the report was in flight
        assert seen["closed_mid_flight"] is False  # cleanup did not race the worker
        assert http.closed_count == 1  # closed once, by the worker, not the caller
        assert extract_agentic_job_fields({}) == {}
