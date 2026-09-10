"""JobContext public surface (BACK-777 Phase 2)."""
import json
import os
import warnings
from unittest.mock import patch

import httpx
import pytest

from revenium_middleware import (
    JobContext,
    OutcomeAlreadyReportedError,
    OutcomeAmendConflictError,
    OutcomeReportingError,
)
from revenium_middleware._core.config import Config
from revenium_middleware._core.fields import extract_agentic_job_fields

WRITE_KEY = "rev_sk_TENANT_write"
LEGACY_KEY = "rev_sk_TENANT_legacy"
EXPLICIT_KEY = "rev_sk_TENANT_explicit"
METERING_KEY = "rev_mk_TENANT_metering"
ENV = {Config.ENV_REVENIUM_WRITE_API_KEY: WRITE_KEY, "REVENIUM_TEAM_ID": "team-1"}


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
    def test_write_env_used_without_deprecation_warning(self):
        http, calls = _recording_client()
        with patch.dict(os.environ, ENV, clear=True):
            with warnings.catch_warnings(record=True) as recorded:
                warnings.simplefilter("always")
                with JobContext(job_id="loan-1", http_client=http) as job:
                    job.report_outcome(execution_status="SUCCESS")

        assert calls[0].headers["x-api-key"] == WRITE_KEY
        assert [
            warning for warning in recorded if issubclass(warning.category, DeprecationWarning)
        ] == []

    def test_outcome_env_still_works_but_warns(self):
        http, calls = _recording_client()
        env = {
            Config.ENV_REVENIUM_OUTCOME_API_KEY: LEGACY_KEY,
            "REVENIUM_TEAM_ID": "team-1",
        }

        with patch.dict(os.environ, env, clear=True):
            with pytest.warns(DeprecationWarning, match="REVENIUM_WRITE_API_KEY"):
                with JobContext(job_id="loan-1", http_client=http) as job:
                    job.report_outcome(execution_status="SUCCESS")

        assert calls[0].headers["x-api-key"] == LEGACY_KEY

    def test_write_env_wins_over_outcome_env_without_deprecation_warning(self):
        http, calls = _recording_client()
        env = {
            Config.ENV_REVENIUM_WRITE_API_KEY: WRITE_KEY,
            Config.ENV_REVENIUM_OUTCOME_API_KEY: LEGACY_KEY,
            "REVENIUM_TEAM_ID": "team-1",
        }

        with patch.dict(os.environ, env, clear=True):
            with warnings.catch_warnings(record=True) as recorded:
                warnings.simplefilter("always")
                with JobContext(job_id="loan-1", http_client=http) as job:
                    job.report_outcome(execution_status="SUCCESS")

        assert calls[0].headers["x-api-key"] == WRITE_KEY
        assert [
            warning for warning in recorded if issubclass(warning.category, DeprecationWarning)
        ] == []

    def test_explicit_api_key_wins_over_env_vars_without_deprecation_warning(self):
        http, calls = _recording_client()
        env = {
            Config.ENV_REVENIUM_WRITE_API_KEY: WRITE_KEY,
            Config.ENV_REVENIUM_OUTCOME_API_KEY: LEGACY_KEY,
            "REVENIUM_TEAM_ID": "team-1",
        }

        with patch.dict(os.environ, env, clear=True):
            with warnings.catch_warnings(record=True) as recorded:
                warnings.simplefilter("always")
                with JobContext(job_id="loan-1", api_key=EXPLICIT_KEY, http_client=http) as job:
                    job.report_outcome(execution_status="SUCCESS")

        assert calls[0].headers["x-api-key"] == EXPLICIT_KEY
        assert [
            warning for warning in recorded if issubclass(warning.category, DeprecationWarning)
        ] == []

    def test_metering_env_fallback_still_fails_fast(self):
        http, calls = _recording_client()
        env = {Config.ENV_REVENIUM_API_KEY: METERING_KEY, "REVENIUM_TEAM_ID": "team-1"}

        with patch.dict(os.environ, env, clear=True):
            with warnings.catch_warnings(record=True) as recorded:
                warnings.simplefilter("always")
                with JobContext(job_id="j1", http_client=http) as job:
                    with pytest.raises(ValueError, match="write-scope"):
                        job.report_outcome(execution_status="SUCCESS")

        assert calls == []
        assert [
            warning for warning in recorded if issubclass(warning.category, DeprecationWarning)
        ] == []

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

    def test_outcome_reason_sent_as_its_own_field(self):
        http, calls = _recording_client()
        with patch.dict(os.environ, ENV):
            with JobContext(job_id="loan-1", http_client=http) as job:
                job.report_outcome(
                    execution_status="FAILED",
                    outcome_reason="Applicant withdrew before underwriting",
                )
        body = json.loads(calls[0].content)
        assert body["outcomeReason"] == "Applicant withdrew before underwriting"
        # The prescribed field, not a metadata key.
        assert "metadata" not in body

    def test_outcome_reason_omitted_when_not_passed(self):
        http, calls = _recording_client()
        with patch.dict(os.environ, ENV):
            with JobContext(job_id="loan-1", http_client=http) as job:
                job.report_outcome(execution_status="SUCCESS")
        assert "outcomeReason" not in json.loads(calls[0].content)

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
        env = {Config.ENV_REVENIUM_WRITE_API_KEY: "rev_mk_TENANT_abc", "REVENIUM_TEAM_ID": "team-1"}
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
        with patch.dict(os.environ, {Config.ENV_REVENIUM_WRITE_API_KEY: "bogus_key"}, clear=True):
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
        # The failure explanation also rides the prescribed first-class field,
        # so history rows do not come back with outcome_reason=None.
        assert body["outcomeReason"] == "boom"

    def test_auto_report_truncates_an_oversized_exception_message(self):
        """The ingest API rejects outcomeReason over 2048 chars with a 400 —
        the safety net must degrade to a truncated reason, never lose the
        whole outcome to validation."""
        http, calls = _recording_client()
        with patch.dict(os.environ, ENV):
            with pytest.raises(RuntimeError):
                with JobContext(job_id="j1", http_client=http):
                    raise RuntimeError("x" * 5000)
        body = json.loads(calls[0].content)
        assert len(body["outcomeReason"]) == 2048
        # metadata keeps the full text (no size constraint on that field)
        assert len(json.loads(body["metadata"])["error"]) == 5000

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
        assert body["outcomeReason"] == "boom"
        assert extract_agentic_job_fields({}) == {}

    @pytest.mark.asyncio
    async def test_async_auto_report_truncates_an_oversized_exception_message(self):
        http, calls = _recording_client()
        with patch.dict(os.environ, ENV):
            with pytest.raises(RuntimeError):
                async with JobContext(job_id="j1", http_client=http):
                    raise RuntimeError("y" * 5000)
        body = json.loads(calls[0].content)
        assert len(body["outcomeReason"]) == 2048

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

    def test_outcome_reason_omitted_vs_cleared(self):
        """Omitting leaves the stored reason untouched; "" clears it."""
        bodies = []

        def handler(req: httpx.Request) -> httpx.Response:
            bodies.append(json.loads(req.content))
            return httpx.Response(200, json={})

        http = httpx.Client(transport=httpx.MockTransport(handler))
        with patch.dict(os.environ, ENV):
            handle = JobContext.attach(job_id="loan-1", http_client=http)
            handle.amend_outcome(reason="Audit pass, outcome unchanged")
            handle.amend_outcome(reason="Reason no longer applies", outcome_reason="")
            handle.amend_outcome(reason="Chargeback", outcome_reason="Payment reversed")
        assert "outcomeReason" not in bodies[0]
        assert bodies[1]["outcomeReason"] == ""
        assert bodies[2]["outcomeReason"] == "Payment reversed"
        # reason (amendment audit trail) stays independent of outcomeReason.
        assert bodies[1]["reason"] == "Reason no longer applies"

    def test_blank_reason_rejected_before_http(self):
        http, calls = _recording_client()
        with patch.dict(os.environ, ENV):
            handle = JobContext.attach(job_id="j1", http_client=http)
            with pytest.raises(ValueError):
                handle.amend_outcome(reason="")
            with pytest.raises(ValueError):
                handle.amend_outcome(reason="   ")
        assert calls == []

    def test_reason_is_omitted_when_not_supplied(self):
        seen = {}

        def handler(req: httpx.Request) -> httpx.Response:
            seen["body"] = json.loads(req.content)
            return httpx.Response(200, json={})

        http = httpx.Client(transport=httpx.MockTransport(handler))
        with patch.dict(os.environ, ENV):
            JobContext.attach(job_id="j1", http_client=http).amend_outcome()
        assert seen["body"] == {}

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


class TestEntityVersion:
    """Optimistic locking on outcome amendment (BACK-3079).

    The platform returns ``entityVersion`` on every job/outcome response and
    accepts it back as ``expectedEntityVersion`` on the amendment PATCH. Without
    it, two writers amending the same outcome silently overwrite each other.
    """

    def _versioned_client(self, versions):
        """A PATCH/POST recorder answering with the given versions in order."""
        bodies = []

        def handler(req: httpx.Request) -> httpx.Response:
            bodies.append(json.loads(req.content))
            version = versions[min(len(bodies) - 1, len(versions) - 1)]
            body = {"id": "job-1"} if version is None else {"id": "job-1", "entityVersion": version}
            return httpx.Response(200, json=body)

        return httpx.Client(transport=httpx.MockTransport(handler)), bodies

    def test_report_records_the_returned_version(self):
        http, _ = self._versioned_client([7])
        with patch.dict(os.environ, ENV):
            with JobContext(job_id="loan-1", http_client=http) as job:
                job.report_outcome(execution_status="SUCCESS")
                assert job.entity_version == 7

    def test_report_tolerates_a_response_without_a_version(self):
        """An older backend sends no version; reporting must still succeed."""
        http, _ = self._versioned_client([None])
        with patch.dict(os.environ, ENV):
            with JobContext(job_id="loan-1", http_client=http) as job:
                job.report_outcome(execution_status="SUCCESS")
                assert job.entity_version is None

    def test_amend_sends_the_version_recorded_by_the_report(self):
        http, bodies = self._versioned_client([7, 8])
        with patch.dict(os.environ, ENV):
            with JobContext(job_id="loan-1", http_client=http) as job:
                job.report_outcome(execution_status="SUCCESS")
                job.amend_outcome(reason="Value corrected", outcome_value=750.0)
        assert "expectedEntityVersion" not in bodies[0]  # POST has no such field
        assert bodies[1]["expectedEntityVersion"] == 7

    def test_amend_records_the_version_from_its_own_response(self):
        """Chained amendments each need the version the previous one produced."""
        http, bodies = self._versioned_client([7, 8, 9])
        with patch.dict(os.environ, ENV):
            with JobContext(job_id="loan-1", http_client=http) as job:
                job.report_outcome(execution_status="SUCCESS")
                job.amend_outcome(reason="First correction")
                assert job.entity_version == 8
                job.amend_outcome(reason="Second correction")
        assert bodies[2]["expectedEntityVersion"] == 8

    def test_amend_omits_the_version_when_none_is_known(self):
        """An attach() handle has reported nothing, so it must not lock."""
        http, bodies = self._versioned_client([5])
        with patch.dict(os.environ, ENV):
            handle = JobContext.attach(job_id="loan-1", http_client=http)
            handle.amend_outcome(reason="Blind correction")
        assert "expectedEntityVersion" not in bodies[0]

    def test_explicit_version_wins_over_the_recorded_one(self):
        http, bodies = self._versioned_client([7, 8])
        with patch.dict(os.environ, ENV):
            with JobContext(job_id="loan-1", http_client=http) as job:
                job.report_outcome(execution_status="SUCCESS")
                job.amend_outcome(reason="Refetched", expected_entity_version=11)
        assert bodies[1]["expectedEntityVersion"] == 11

    def test_version_zero_is_sent_not_dropped(self):
        """A freshly created job sits at version 0 — a real version, not "unknown"."""
        http, bodies = self._versioned_client([8])
        with patch.dict(os.environ, ENV):
            handle = JobContext.attach(job_id="loan-1", http_client=http)
            handle.amend_outcome(reason="Correcting the first report",
                                 expected_entity_version=0)
        assert bodies[0]["expectedEntityVersion"] == 0

    def test_versionless_success_discards_the_stale_lock(self):
        """A completed amendment has advanced the version, so the old one is wrong.

        Keeping it would make the next amendment send a token the backend
        cannot match and report a conflict that never happened.
        """
        http, bodies = self._versioned_client([7, None, None])
        with patch.dict(os.environ, ENV):
            with JobContext(job_id="loan-1", http_client=http) as job:
                job.report_outcome(execution_status="SUCCESS")
                job.amend_outcome(reason="Correction")
                assert job.entity_version is None
                job.amend_outcome(reason="Second correction")
        assert bodies[1]["expectedEntityVersion"] == 7
        # Unlocked, not locked to the version the first amendment invalidated.
        assert "expectedEntityVersion" not in bodies[2]

    def test_bodiless_success_discards_the_stale_lock(self):
        """The transport accepts a bodiless 2xx as success; so must the lock."""
        bodies = []

        def handler(req: httpx.Request) -> httpx.Response:
            bodies.append(json.loads(req.content))
            if len(bodies) == 1:
                return httpx.Response(200, json={"entityVersion": 7})
            return httpx.Response(204)

        http = httpx.Client(transport=httpx.MockTransport(handler))
        with patch.dict(os.environ, ENV):
            with JobContext(job_id="loan-1", http_client=http) as job:
                job.report_outcome(execution_status="SUCCESS")
                job.amend_outcome(reason="Correction")
                assert job.entity_version is None
                job.amend_outcome(reason="Second correction")
        assert "expectedEntityVersion" not in bodies[2]

    def test_reenter_forgets_the_previous_run_version(self):
        http, bodies = self._versioned_client([7, None])
        with patch.dict(os.environ, ENV):
            job = JobContext(job_id="loan-1", http_client=http)
            with job:
                job.report_outcome(execution_status="SUCCESS")
            assert job.entity_version == 7
            with job:
                assert job.entity_version is None
                job.amend_outcome(reason="Correction on a new run")
        assert "expectedEntityVersion" not in bodies[1]

    def test_bad_explicit_version_rejected_before_http(self):
        """A value the SDK cannot legally send must not reach the wire.

        ``3.5`` is the one that matters: truncating it to ``3`` would lock the
        amendment against a version the caller never named.
        """
        http, calls = _recording_client()
        with patch.dict(os.environ, ENV):
            handle = JobContext.attach(job_id="j1", http_client=http)
            for bad in (-1, 3.5, "not-a-number", "3.0", True, object()):
                with pytest.raises(ValueError, match="non-negative integer"):
                    handle.amend_outcome(reason="r", expected_entity_version=bad)
        assert calls == []

    def test_integral_versions_accepted(self):
        """A JSON-ish integral value is a version: 3.0 and "3" both mean 3."""
        http, bodies = self._versioned_client([9])
        with patch.dict(os.environ, ENV):
            handle = JobContext.attach(job_id="j1", http_client=http)
            handle.amend_outcome(reason="r", expected_entity_version=3.0)
            handle.amend_outcome(reason="r", expected_entity_version="3")
        assert bodies[0]["expectedEntityVersion"] == 3
        assert bodies[1]["expectedEntityVersion"] == 3

    def test_conflict_records_the_version_the_backend_holds(self):
        """The 409 is what closes the loop: it reports the current version."""
        bodies = []

        def handler(req: httpx.Request) -> httpx.Response:
            bodies.append(json.loads(req.content))
            if len(bodies) == 1:
                return httpx.Response(200, json={"entityVersion": 7})
            if len(bodies) == 2:
                return httpx.Response(409, json={
                    "status": 409, "error": "Conflict",
                    "message": "Outcome has changed since entity version 7; "
                               "current version is 9. Refetch and retry.",
                })
            return httpx.Response(200, json={"entityVersion": 10})

        http = httpx.Client(transport=httpx.MockTransport(handler))
        with patch.dict(os.environ, ENV):
            with JobContext(job_id="loan-1", http_client=http) as job:
                job.report_outcome(execution_status="SUCCESS")
                with pytest.raises(OutcomeAmendConflictError) as conflict:
                    job.amend_outcome(reason="Correction", outcome_value=750.0)
                assert conflict.value.current_entity_version == 9
                # Recorded, so a caller who decides the amendment still applies
                # can retry through this handle without threading it back.
                assert job.entity_version == 9
                job.amend_outcome(reason="Correction, re-checked", outcome_value=750.0)
        assert bodies[1]["expectedEntityVersion"] == 7
        assert bodies[2]["expectedEntityVersion"] == 9

    def test_versionless_conflict_leaves_the_next_amendment_unlocked(self):
        """No version in the body means no version to retry with: unlock."""
        bodies = []

        def handler(req: httpx.Request) -> httpx.Response:
            bodies.append(json.loads(req.content))
            if len(bodies) == 1:
                return httpx.Response(200, json={"entityVersion": 7})
            if len(bodies) == 2:
                return httpx.Response(409, json={"error": "Conflict"})
            return httpx.Response(200, json={"entityVersion": 11})

        http = httpx.Client(transport=httpx.MockTransport(handler))
        with patch.dict(os.environ, ENV):
            with JobContext(job_id="loan-1", http_client=http) as job:
                job.report_outcome(execution_status="SUCCESS")
                with pytest.raises(OutcomeAmendConflictError) as conflict:
                    job.amend_outcome(reason="Correction")
                assert conflict.value.current_entity_version is None
                assert job.entity_version is None
                job.amend_outcome(reason="Correction, unlocked")
        assert "expectedEntityVersion" not in bodies[2]


class TestAmendReasonOptional:
    """``reason`` is optional for API-key callers (BACK-3079).

    ``UpdateOutcomeRequest.reason`` is nullable: an API-key caller that omits it
    gets an automated correction reason from the server, so the SDK must be able
    to omit the key rather than force a placeholder into the audit trail.
    """

    def test_omitted_reason_is_absent_from_the_payload(self):
        bodies = []

        def handler(req: httpx.Request) -> httpx.Response:
            bodies.append(json.loads(req.content))
            return httpx.Response(200, json={"id": "job-1", "entityVersion": 2})

        http = httpx.Client(transport=httpx.MockTransport(handler))
        with patch.dict(os.environ, ENV):
            handle = JobContext.attach(job_id="loan-1", http_client=http)
            handle.amend_outcome(outcome_value=750.0)
        assert "reason" not in bodies[0]
        assert bodies[0]["outcomeValue"] == 750.0

    def test_reason_still_positional(self):
        """Existing callers pass reason as the first positional argument."""
        bodies = []

        def handler(req: httpx.Request) -> httpx.Response:
            bodies.append(json.loads(req.content))
            return httpx.Response(200, json={})

        http = httpx.Client(transport=httpx.MockTransport(handler))
        with patch.dict(os.environ, ENV):
            handle = JobContext.attach(job_id="loan-1", http_client=http)
            handle.amend_outcome("Customer expanded contract")
        assert bodies[0]["reason"] == "Customer expanded contract"

    def test_blank_reason_still_rejected(self):
        """Omitting is deliberate; a blank string is a caller bug either way."""
        http, calls = _recording_client()
        with patch.dict(os.environ, ENV):
            handle = JobContext.attach(job_id="j1", http_client=http)
            with pytest.raises(ValueError):
                handle.amend_outcome(reason="   ")
        assert calls == []


class TestOutcomeMetrics:
    """Declared metric facts on the outcome bodies and the late-append call (BACK-3080).

    A fact only lands if the job type's economics contract declares the metric
    (BACK-3078), and the server range-checks ``quality_rate`` — so these tests
    are about the wire shape and about shape errors surfacing before any HTTP,
    not about which values the platform accepts.
    """

    METRICS = [
        {"key": "quality_rate", "value": 0.93, "provenance": "MEASURED",
         "reason": "graded sample of 200 cases"},
        {"key": "cases_closed", "value": 12},
    ]

    def test_report_body_carries_the_metrics_array_unchanged(self):
        http, calls = _recording_client()
        with patch.dict(os.environ, ENV):
            with JobContext(job_id="loan-1", http_client=http) as job:
                job.report_outcome(execution_status="SUCCESS", metrics=self.METRICS)
        body = json.loads(calls[0].content)
        # Entries reach the wire exactly as supplied: provenance, recordedBy and
        # source all have server-side defaults, and filling them in client-side
        # would misattribute the fact.
        assert body["metrics"] == self.METRICS

    def test_amend_body_carries_the_metrics_array_unchanged(self):
        http, calls = _recording_client()
        with patch.dict(os.environ, ENV):
            handle = JobContext.attach(job_id="loan-1", http_client=http)
            handle.amend_outcome(reason="Graded after review", metrics=self.METRICS)
        assert calls[0].method == "PATCH"
        assert json.loads(calls[0].content)["metrics"] == self.METRICS

    def test_metrics_key_absent_when_not_passed(self):
        http, calls = _recording_client()
        with patch.dict(os.environ, ENV):
            with JobContext(job_id="loan-1", http_client=http) as job:
                job.report_outcome(execution_status="SUCCESS")
            handle = JobContext.attach(job_id="loan-1", http_client=http)
            handle.amend_outcome(reason="No facts to add")
        assert "metrics" not in json.loads(calls[0].content)
        assert "metrics" not in json.loads(calls[1].content)

    def test_empty_metrics_omits_the_key(self):
        """An empty array is what the server ignores, so do not send it."""
        http, calls = _recording_client()
        with patch.dict(os.environ, ENV):
            with JobContext(job_id="loan-1", http_client=http) as job:
                job.report_outcome(execution_status="SUCCESS", metrics=[])
        assert "metrics" not in json.loads(calls[0].content)

    @pytest.mark.parametrize("bad", [
        {"key": "quality_rate", "value": 0.9},          # a single mapping, not a sequence
        "quality_rate",                                  # a bare string
        [["quality_rate", 0.9]],                         # entries that are not mappings
        [{"value": 0.9}],                                # no key
        [{"key": "quality_rate"}],                       # no value
        [{"key": "", "value": 0.9}],                     # blank key
        [{"key": "quality_rate", "value": None}],        # null value
    ])
    def test_malformed_metrics_raise_before_any_request(self, bad):
        http, calls = _recording_client()
        with patch.dict(os.environ, ENV):
            with JobContext(job_id="loan-1", http_client=http) as job:
                with pytest.raises(ValueError, match="metrics"):
                    job.report_outcome(execution_status="SUCCESS", metrics=bad)
                with pytest.raises(ValueError, match="metrics"):
                    job.append_outcome_metrics(bad)
            handle = JobContext.attach(job_id="loan-1", http_client=http)
            with pytest.raises(ValueError, match="metrics"):
                handle.amend_outcome(reason="r", metrics=bad)
        # Nothing reached the network, including the team-resolution GET.
        assert calls == []

    def test_append_posts_a_bare_array_to_the_metrics_path(self):
        http, calls = _recording_client(status=201)
        with patch.dict(os.environ, ENV):
            handle = JobContext.attach(job_id="loan-1", http_client=http)
            handle.append_outcome_metrics(self.METRICS)
        assert len(calls) == 1
        req = calls[0]
        assert req.method == "POST"
        assert req.url.path.endswith("/profitstream/v2/api/jobs/loan-1/outcome/metrics")
        assert req.url.params["teamId"] == "team-1"
        assert req.headers["x-api-key"] == WRITE_KEY
        assert json.loads(req.content) == self.METRICS

    def test_append_rejects_empty_entries_before_http(self):
        """There is nothing to append, and the server answers 400 for it."""
        http, calls = _recording_client(status=201)
        with patch.dict(os.environ, ENV):
            handle = JobContext.attach(job_id="loan-1", http_client=http)
            with pytest.raises(ValueError, match="metrics"):
                handle.append_outcome_metrics([])
        assert calls == []

    def test_append_rejects_none_entries_before_http(self):
        """An unset variable forwarded here is a caller bug, not an empty POST."""
        http, calls = _recording_client(status=201)
        with patch.dict(os.environ, ENV):
            handle = JobContext.attach(job_id="loan-1", http_client=http)
            with pytest.raises(ValueError, match="metrics"):
                handle.append_outcome_metrics(None)
        assert calls == []

    def test_omitted_metrics_is_still_not_an_error_on_the_outcome_bodies(self):
        """``metrics=None`` means "no facts" there; only the append demands entries."""
        http, calls = _recording_client()
        with patch.dict(os.environ, ENV):
            with JobContext(job_id="loan-1", http_client=http) as job:
                job.report_outcome(execution_status="SUCCESS", metrics=None)
        assert "metrics" not in json.loads(calls[0].content)

    def test_append_rejects_a_metering_key_before_http(self):
        http, calls = _recording_client(status=201)
        # The metering key has to sit in the resolved position, which is now
        # REVENIUM_WRITE_API_KEY; overriding the deprecated fallback would be
        # shadowed by the write key ENV already sets, and the call would pass.
        env = dict(ENV, **{Config.ENV_REVENIUM_WRITE_API_KEY: "rev_mk_TENANT_abc"})
        with patch.dict(os.environ, env):
            handle = JobContext.attach(job_id="loan-1", http_client=http)
            with pytest.raises(ValueError, match="write-scope"):
                handle.append_outcome_metrics(self.METRICS)
        assert calls == []

    def test_append_records_an_entity_version_when_the_response_carries_one(self):
        http, _ = _recording_client(body={"entityVersion": 11})
        with patch.dict(os.environ, ENV):
            handle = JobContext.attach(job_id="loan-1", http_client=http)
            handle.append_outcome_metrics(self.METRICS)
            assert handle.entity_version == 11

    def test_append_leaves_the_recorded_version_alone_on_a_bodiless_201(self):
        """Appending facts does not advance the job's version, so the lock holds.

        report/amend clear the recorded token on a versionless response because
        they mutate the Job row and anything held from before them is stale. The
        facts append writes no Job row and answers a bodiless 201, so clearing
        would unlock the next amendment for nothing — re-opening the lost update
        the lock exists to catch.
        """
        bodies = []

        def handler(req: httpx.Request) -> httpx.Response:
            bodies.append(req)
            if req.url.path.endswith("/outcome/metrics"):
                return httpx.Response(201)
            return httpx.Response(200, json={"id": "job-1", "entityVersion": 4})

        http = httpx.Client(transport=httpx.MockTransport(handler))
        with patch.dict(os.environ, ENV):
            with JobContext(job_id="loan-1", http_client=http) as job:
                job.report_outcome(execution_status="SUCCESS")
                assert job.entity_version == 4
                job.append_outcome_metrics(self.METRICS)
                assert job.entity_version == 4
                job.amend_outcome(reason="Graded after the facts landed")
        assert json.loads(bodies[-1].content)["expectedEntityVersion"] == 4

    def test_append_does_not_count_as_reporting_an_outcome(self):
        """Facts are not a terminal outcome: auto-FAILED must still fire."""
        http, calls = _recording_client(status=201)
        with patch.dict(os.environ, ENV):
            with pytest.raises(RuntimeError):
                with JobContext(job_id="loan-1", http_client=http) as job:
                    job.append_outcome_metrics(self.METRICS)
                    raise RuntimeError("agent crashed after recording facts")
        paths = [c.url.path for c in calls]
        assert paths[0].endswith("/outcome/metrics")
        assert paths[1].endswith("/outcome")
        assert json.loads(calls[1].content)["executionStatus"] == "FAILED"
