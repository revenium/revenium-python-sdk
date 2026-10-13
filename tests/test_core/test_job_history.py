"""get_outcome_history (BACK-777 Phase 3, addendum §C)."""
import os
import warnings
from datetime import datetime
from unittest.mock import patch

import httpx
import pytest

import revenium_middleware
import revenium_middleware.job_history as job_history
from revenium_middleware import JobOutcomeAmendment, get_outcome_history
from revenium_middleware._core import outcomes
from revenium_middleware._core.config import Config
from revenium_middleware._core.exceptions import OutcomeAmendConflictError, OutcomeReportingError

BASE = "https://api.revenium.example"
WRITE_KEY = "rev_sk_TENANT_write"
LEGACY_KEY = "rev_sk_TENANT_legacy"
EXPLICIT_KEY = "rev_sk_TENANT_explicit"
METERING_KEY = "rev_mk_TENANT_metering"
ENV = {Config.ENV_REVENIUM_WRITE_API_KEY: WRITE_KEY, "REVENIUM_TEAM_ID": "team-1"}

HISTORY = [
    {
        "sequence": 2,
        "executionStatus": "SUCCESS",
        "outcomeType": "CONVERTED",
        "outcomeValue": 750.0,
        "outcomeCurrency": "USD",
        "outcomeMetadata": "{\"expansion_event\": \"upsell_q2\"}",
        "reportedBy": "orchestrator",
        "reportedAt": "2026-07-01T10:00:00Z",
        "reason": "Customer expanded contract",
    },
    {
        "sequence": 1,
        "executionStatus": "SUCCESS",
        "outcomeType": "CONVERTED",
        "outcomeValue": 500.0,
        "outcomeCurrency": "USD",
        "outcomeMetadata": None,
        "reportedBy": "orchestrator",
        "reportedAt": "2026-06-01T10:00:00Z",
        "reason": None,
    },
]


def _client(body, status=200):
    seen = {"requests": 0}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["requests"] += 1
        seen["method"] = req.method
        seen["path"] = req.url.path
        seen["team"] = req.url.params.get("teamId")
        seen["headers"] = dict(req.headers)
        seen["query"] = dict(req.url.params)
        seen["raw_path"] = req.url.raw_path.decode().split("?", 1)[0]
        seen["url"] = str(req.url)
        return httpx.Response(status, json=body)

    return httpx.Client(transport=httpx.MockTransport(handler)), seen


class TestGetOutcomeHistory:
    def test_write_env_used_without_deprecation_warning(self):
        http, seen = _client(HISTORY)
        with patch.dict(os.environ, ENV, clear=True):
            with warnings.catch_warnings(record=True) as recorded:
                warnings.simplefilter("always")
                get_outcome_history("job-1", profitstream_base_url=BASE, http_client=http)

        assert seen["headers"]["x-api-key"] == WRITE_KEY
        assert [
            warning for warning in recorded if issubclass(warning.category, DeprecationWarning)
        ] == []

    def test_outcome_env_still_works_but_warns(self):
        http, seen = _client(HISTORY)
        env = {
            Config.ENV_REVENIUM_OUTCOME_API_KEY: LEGACY_KEY,
            "REVENIUM_TEAM_ID": "team-1",
        }

        with patch.dict(os.environ, env, clear=True):
            with pytest.warns(DeprecationWarning, match="REVENIUM_WRITE_API_KEY"):
                get_outcome_history("job-1", profitstream_base_url=BASE, http_client=http)

        assert seen["headers"]["x-api-key"] == LEGACY_KEY

    def test_write_env_wins_over_outcome_env_without_deprecation_warning(self):
        http, seen = _client(HISTORY)
        env = {
            Config.ENV_REVENIUM_WRITE_API_KEY: WRITE_KEY,
            Config.ENV_REVENIUM_OUTCOME_API_KEY: LEGACY_KEY,
            "REVENIUM_TEAM_ID": "team-1",
        }

        with patch.dict(os.environ, env, clear=True):
            with warnings.catch_warnings(record=True) as recorded:
                warnings.simplefilter("always")
                get_outcome_history("job-1", profitstream_base_url=BASE, http_client=http)

        assert seen["headers"]["x-api-key"] == WRITE_KEY
        assert [
            warning for warning in recorded if issubclass(warning.category, DeprecationWarning)
        ] == []

    def test_explicit_api_key_wins_over_env_vars_without_deprecation_warning(self):
        http, seen = _client(HISTORY)
        env = {
            Config.ENV_REVENIUM_WRITE_API_KEY: WRITE_KEY,
            Config.ENV_REVENIUM_OUTCOME_API_KEY: LEGACY_KEY,
            "REVENIUM_TEAM_ID": "team-1",
        }

        with patch.dict(os.environ, env, clear=True):
            with warnings.catch_warnings(record=True) as recorded:
                warnings.simplefilter("always")
                get_outcome_history(
                    "job-1",
                    api_key=EXPLICIT_KEY,
                    profitstream_base_url=BASE,
                    http_client=http,
                )

        assert seen["headers"]["x-api-key"] == EXPLICIT_KEY
        assert [
            warning for warning in recorded if issubclass(warning.category, DeprecationWarning)
        ] == []

    def test_metering_env_fallback_still_fails_fast(self):
        http, seen = _client(HISTORY)
        env = {Config.ENV_REVENIUM_API_KEY: METERING_KEY, "REVENIUM_TEAM_ID": "team-1"}

        with patch.dict(os.environ, env, clear=True):
            with warnings.catch_warnings(record=True) as recorded:
                warnings.simplefilter("always")
                with pytest.raises(ValueError, match="write-scope"):
                    get_outcome_history("job-1", profitstream_base_url=BASE, http_client=http)

        assert "method" not in seen
        assert [
            warning for warning in recorded if issubclass(warning.category, DeprecationWarning)
        ] == []

    def test_returns_ordered_dataclasses(self):
        http, seen = _client(HISTORY)
        with patch.dict(os.environ, ENV):
            history = get_outcome_history("job-1", profitstream_base_url=BASE, http_client=http)
        assert seen["method"] == "GET"
        assert seen["path"].endswith("/profitstream/v2/api/jobs/job-1/outcome/history")
        assert seen["team"] == "team-1"
        assert [a.amendment_sequence for a in history] == [1, 2]  # defensive ASC sort
        first = history[0]
        assert isinstance(first, JobOutcomeAmendment)
        assert first.reason is None and first.outcome_value == 500.0
        assert first.reported_at == datetime.fromisoformat("2026-06-01T10:00:00+00:00")
        assert history[1].reason == "Customer expanded contract"

    def test_embedded_list_shape_also_parsed(self):
        # HAL-style wrapper, mirroring the teams API response shape
        body = {"_embedded": {"jobOutcomeRevisionResourceList": HISTORY}}
        http, _ = _client(body)
        with patch.dict(os.environ, ENV):
            history = get_outcome_history("job-1", profitstream_base_url=BASE, http_client=http)
        assert len(history) == 2

    def test_unknown_embedded_relation_yields_empty(self):
        body = {"_embedded": {"someOtherResourceList": HISTORY}}
        http, _ = _client(body)
        with patch.dict(os.environ, ENV):
            assert get_outcome_history("job-1", profitstream_base_url=BASE, http_client=http) == []

    def test_retry_knobs_reach_transport(self):
        counter = {"n": 0}

        def handler(req: httpx.Request) -> httpx.Response:
            counter["n"] += 1
            return httpx.Response(503, json={})

        http = httpx.Client(transport=httpx.MockTransport(handler))
        with patch.dict(os.environ, ENV):
            with pytest.raises(httpx.HTTPStatusError):
                get_outcome_history(
                    "job-1", profitstream_base_url=BASE, http_client=http,
                    retry_attempts=2, retry_initial_seconds=0.01, retry_max_seconds=0.02,
                )
        assert counter["n"] == 2  # not the 10-attempt default schedule

    def test_outcome_reason_mapped_and_distinct_from_amendment_reason(self):
        rows = [
            {
                "sequence": 1,
                "executionStatus": "FAILED",
                "reason": None,
                "outcomeReason": "Applicant withdrew before underwriting",
            },
            {
                "sequence": 2,
                "executionStatus": "CANCELLED",
                "reason": "Corrected after manual review",
                "outcomeReason": "",
            },
        ]
        http, _ = _client(rows)
        with patch.dict(os.environ, ENV):
            history = get_outcome_history("job-1", profitstream_base_url=BASE, http_client=http)
        assert history[0].outcome_reason == "Applicant withdrew before underwriting"
        assert history[0].reason is None
        assert history[1].outcome_reason == ""  # cleared by the amendment
        assert history[1].reason == "Corrected after manual review"

    def test_row_without_outcome_reason_maps_to_none(self):
        http, _ = _client(HISTORY)
        with patch.dict(os.environ, ENV):
            history = get_outcome_history("job-1", profitstream_base_url=BASE, http_client=http)
        assert all(a.outcome_reason is None for a in history)

    def test_positional_construction_still_works(self):
        # outcome_reason is appended last and defaulted, so user code that builds
        # rows positionally keeps compiling.
        row = JobOutcomeAmendment(
            2, "SUCCESS", "CONVERTED", 750.0, "USD", None, "orchestrator", None, "expanded",
        )
        assert row.reason == "expanded"
        assert row.outcome_reason is None

    def test_metering_key_fails_fast(self):
        http, seen = _client(HISTORY)
        env = {Config.ENV_REVENIUM_WRITE_API_KEY: "rev_mk_TENANT_abc", "REVENIUM_TEAM_ID": "team-1"}
        with patch.dict(os.environ, env):
            with pytest.raises(ValueError, match="write-scope"):
                get_outcome_history("job-1", profitstream_base_url=BASE, http_client=http)
        assert "method" not in seen


def test_history_goes_to_the_usage_address_when_only_that_is_set(monkeypatch):
    monkeypatch.delenv("REVENIUM_PROFITSTREAM_BASE_URL", raising=False)
    monkeypatch.setenv("REVENIUM_METERING_BASE_URL", "https://usage.example.test/meter")
    monkeypatch.setenv(Config.ENV_REVENIUM_WRITE_API_KEY, WRITE_KEY)
    monkeypatch.setenv("REVENIUM_TEAM_ID", "team-1")
    urls = []

    def handler(req: httpx.Request) -> httpx.Response:
        urls.append(str(req.url))
        return httpx.Response(200, json=HISTORY)

    get_outcome_history("job-1", http_client=httpx.Client(transport=httpx.MockTransport(handler)))

    assert urls == [
        "https://usage.example.test/profitstream/v2/api/jobs/job-1/outcome/history?teamId=team-1"
    ]


JOBS = "/profitstream/v2/api/jobs"
JOB_READS = [
    "list_jobs", "get_job", "get_job_types", "get_job_roi",
    "get_job_transactions", "get_conversion_funnel",
]
JOB = {
    "id": "j1", "agenticJobId": "job-1", "name": "Loan 1", "type": "loan_processing",
    "environment": "production", "executionStatus": "SUCCESS", "outcomeType": "CONVERTED",
    "outcomeValue": 500.0, "hasOutcome": True, "entityVersion": 3,
    "_links": {"self": {"href": "https://api.revenium.example/jobs/job-1"}},
}
PAGE = {"size": 20, "totalElements": 1, "totalPages": 1, "number": 0}


def _read(name):
    fn = getattr(job_history, name, None)
    assert callable(fn), f"revenium_middleware.job_history has no {name}"
    return fn


def _args_for(name):
    return ("job-1",) if name in ("get_job", "get_job_roi", "get_job_transactions") else ()


class TestJobReads:
    @pytest.mark.parametrize("name", JOB_READS)
    def test_job_reads_are_public(self, name):
        assert callable(getattr(revenium_middleware, name, None))
        assert name in revenium_middleware.__all__

    @pytest.mark.parametrize(
        "name, args, path",
        [
            ("list_jobs", (), JOBS),
            ("get_job", ("job-1",), JOBS + "/job-1"),
            ("get_job_types", (), JOBS + "/types"),
            ("get_job_roi", ("job-1",), JOBS + "/job-1/roi"),
            ("get_job_transactions", ("job-1",), JOBS + "/job-1/transactions"),
            ("get_conversion_funnel", (), JOBS + "/conversion-funnel"),
        ],
        ids=JOB_READS,
    )
    def test_each_job_read_sends_one_get_to_its_endpoint(self, name, args, path):
        http, seen = _client({})
        with patch.dict(os.environ, ENV, clear=True):
            _read(name)(*args, profitstream_base_url=BASE, http_client=http)
        assert seen["method"] == "GET"
        assert seen["raw_path"] == path
        assert seen["query"] == {"teamId": "team-1"}
        assert seen["headers"]["x-api-key"] == WRITE_KEY
        assert seen["requests"] == 1

    def test_list_jobs_sends_only_the_filters_given(self):
        list_jobs = _read("list_jobs")
        http, seen = _client({"content": [], "page": PAGE})
        with patch.dict(os.environ, ENV, clear=True):
            list_jobs(
                job_type="loan_processing", execution_status="SUCCESS",
                outcome_type="CONVERTED", environment="production",
                start_date="2026-09-01T00:00:00Z", end_date="2026-09-30T23:59:59Z",
                page=2, size=50, sort="created,desc",
                profitstream_base_url=BASE, http_client=http,
            )
            assert seen["query"] == {
                "teamId": "team-1", "type": "loan_processing", "executionStatus": "SUCCESS",
                "outcomeType": "CONVERTED", "environment": "production",
                "startDate": "2026-09-01T00:00:00Z", "endDate": "2026-09-30T23:59:59Z",
                "page": "2", "size": "50", "sort": "created,desc",
            }
            list_jobs(size=5, profitstream_base_url=BASE, http_client=http)
            assert seen["query"] == {"teamId": "team-1", "size": "5"}
            list_jobs(page=0, profitstream_base_url=BASE, http_client=http)
            assert seen["query"] == {"teamId": "team-1", "page": "0"}

    def test_list_jobs_reads_the_hal_page(self):
        body = {"_embedded": {"jobResourceList": [JOB]}, "_links": {}, "page": PAGE}
        http, _ = _client(body)
        with patch.dict(os.environ, ENV, clear=True):
            result = _read("list_jobs")(profitstream_base_url=BASE, http_client=http)
        assert result == {"content": [JOB], "page": PAGE}

    def test_list_jobs_reads_a_plain_content_page(self):
        http, _ = _client({"content": [JOB], "page": PAGE})
        with patch.dict(os.environ, ENV, clear=True):
            result = _read("list_jobs")(profitstream_base_url=BASE, http_client=http)
        assert result == {"content": [JOB], "page": PAGE}

    def test_list_jobs_keeps_paging_details_beside_a_plain_content_page(self):
        paging = {"size": 20, "totalElements": 200, "totalPages": 10, "number": 1}
        http, _ = _client({"content": [JOB], **paging})
        with patch.dict(os.environ, ENV, clear=True):
            result = _read("list_jobs")(profitstream_base_url=BASE, http_client=http)
        assert result == {"content": [JOB], "page": paging}

    def test_job_reads_follow_the_usage_address_when_no_outcome_address_is_set(self):
        http, seen = _client({})
        env = dict(ENV, REVENIUM_METERING_BASE_URL="https://usage.example.test")
        with patch.dict(os.environ, env, clear=True):
            _read("list_jobs")(http_client=http)
        assert seen["url"].startswith("https://usage.example.test/profitstream/v2/api/jobs")

    def test_list_jobs_empty_page_has_no_embedded_key(self):
        empty_page = {"size": 20, "totalElements": 0, "totalPages": 0, "number": 0}
        http, _ = _client({"_links": {}, "page": empty_page})
        with patch.dict(os.environ, ENV, clear=True):
            result = _read("list_jobs")(profitstream_base_url=BASE, http_client=http)
        assert result == {"content": [], "page": empty_page}

    def test_get_job_returns_the_entity_version(self):
        http, _ = _client(JOB)
        with patch.dict(os.environ, ENV, clear=True):
            job = _read("get_job")("job-1", profitstream_base_url=BASE, http_client=http)
        assert job["entityVersion"] == 3
        assert job == JOB

    @pytest.mark.parametrize(
        "name, body",
        [
            ("get_job_roi", {"agenticJobId": "job-1", "roi": 4.5, "totalCost": 0.12}),
            (
                "get_job_transactions",
                {"transactions": [{"transactionId": "t1", "cost": 0.12}], "totalCount": 1},
            ),
            ("get_job_types", ["claims", "loan_processing"]),
            (
                "get_conversion_funnel",
                {"totalJobs": 10, "convertedJobs": 4, "deflectedJobs": 2, "conversionRate": 0.4},
            ),
        ],
        ids=["roi", "transactions", "types", "funnel"],
    )
    def test_roi_transactions_types_and_funnel_return_the_api_json(self, name, body):
        http, _ = _client(body)
        with patch.dict(os.environ, ENV, clear=True):
            result = _read(name)(*_args_for(name), profitstream_base_url=BASE, http_client=http)
        assert result == body

    def test_get_job_types_on_a_non_list_reply_is_empty(self):
        http, _ = _client({})
        with patch.dict(os.environ, ENV, clear=True):
            assert _read("get_job_types")(profitstream_base_url=BASE, http_client=http) == []

    def test_get_conversion_funnel_sends_its_filters(self):
        http, seen = _client({})
        with patch.dict(os.environ, ENV, clear=True):
            _read("get_conversion_funnel")(
                start_date="2026-09-01T00:00:00Z", end_date="2026-09-30T23:59:59Z",
                job_type="loan_processing", environment="production",
                profitstream_base_url=BASE, http_client=http,
            )
        assert seen["query"] == {
            "teamId": "team-1", "startDate": "2026-09-01T00:00:00Z",
            "endDate": "2026-09-30T23:59:59Z", "jobType": "loan_processing",
            "environment": "production",
        }

    def test_job_id_is_escaped_into_one_path_segment(self):
        http, seen = _client({})
        with patch.dict(os.environ, ENV, clear=True):
            _read("get_job_roi")("loan/42 a", profitstream_base_url=BASE, http_client=http)
        assert seen["raw_path"] == JOBS + "/loan%2F42%20a/roi"

    @pytest.mark.parametrize("job_id", ["", "   "], ids=["empty", "blank"])
    @pytest.mark.parametrize("name", ["get_job", "get_job_roi", "get_job_transactions"])
    def test_blank_job_id_is_rejected_before_any_request(self, name, job_id):
        fn = _read(name)
        http, seen = _client({})
        with patch.dict(os.environ, ENV, clear=True):
            with pytest.raises(ValueError, match="job_id"):
                fn(job_id, profitstream_base_url=BASE, http_client=http)
        assert seen["requests"] == 0

    @pytest.mark.parametrize("name", JOB_READS)
    def test_job_reads_reject_a_metering_key_before_any_request(self, name):
        fn = _read(name)
        http, seen = _client({})
        env = {Config.ENV_REVENIUM_WRITE_API_KEY: METERING_KEY, "REVENIUM_TEAM_ID": "team-1"}
        with patch.dict(os.environ, env, clear=True):
            with pytest.raises(ValueError, match="write-scope"):
                fn(*_args_for(name), profitstream_base_url=BASE, http_client=http)
        assert seen["requests"] == 0

    def test_job_reads_raise_when_no_team_resolves(self):
        list_jobs = _read("list_jobs")
        paths = []

        def handler(req: httpx.Request) -> httpx.Response:
            paths.append(req.url.path)
            if req.url.path == "/profitstream/v2/api/teams":
                return httpx.Response(200, json={"_embedded": {"teamResourceList": []}})
            return httpx.Response(200, json={})

        http = httpx.Client(transport=httpx.MockTransport(handler))
        env = {Config.ENV_REVENIUM_WRITE_API_KEY: WRITE_KEY}
        with patch.dict(os.environ, env, clear=True):
            with pytest.raises(OutcomeReportingError, match="team_id"):
                list_jobs(profitstream_base_url=BASE, http_client=http)
        assert paths == ["/profitstream/v2/api/teams"]

    def test_missing_job_raises_without_retrying(self):
        http, seen = _client({"status": 404}, status=404)
        with patch.dict(os.environ, ENV, clear=True):
            with pytest.raises(httpx.HTTPStatusError):
                _read("get_job")("nope", profitstream_base_url=BASE, http_client=http)
        assert seen["requests"] == 1

    def test_retry_knobs_reach_the_job_reads(self):
        http, seen = _client({}, status=503)
        with patch.dict(os.environ, ENV, clear=True):
            with pytest.raises(httpx.HTTPStatusError):
                _read("get_job_roi")(
                    "job-1", profitstream_base_url=BASE, http_client=http,
                    retry_attempts=2, retry_initial_seconds=0.01, retry_max_seconds=0.02,
                )
        assert seen["requests"] == 2

    def test_amend_conflict_message_points_at_get_job(self):
        assert "get_job(" in outcomes._AMEND_CONFLICT_RECOVERY
        assert "outside this SDK" not in outcomes._AMEND_CONFLICT_RECOVERY
        body = {"status": 409, "error": "Conflict", "message": "Outcome was changed by another writer"}
        http = httpx.Client(transport=httpx.MockTransport(lambda req: httpx.Response(409, json=body)))
        with pytest.raises(OutcomeAmendConflictError) as raised:
            outcomes.amend_outcome_request(
                http, BASE, "job-1", {"outcomeType": "CONVERTED"},
                team_id="team-1", api_key=WRITE_KEY,
            )
        assert "get_job(" in str(raised.value)
        assert "outside this SDK" not in str(raised.value)
        assert raised.value.current_entity_version is None

    def test_job_reads_use_the_configured_profitstream_address(self):
        http, seen = _client({})
        env = dict(ENV, **{Config.ENV_REVENIUM_PROFITSTREAM_BASE_URL: "https://cfg.example.test"})
        with patch.dict(os.environ, env, clear=True):
            _read("list_jobs")(http_client=http)
        assert seen["url"].startswith("https://cfg.example.test/profitstream/v2/api/jobs")
