"""get_outcome_history (BACK-777 Phase 3, addendum §C)."""
import os
import warnings
from datetime import datetime
from unittest.mock import patch

import httpx
import pytest

from revenium_middleware import JobOutcomeAmendment, get_outcome_history
from revenium_middleware._core.config import Config

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
    seen = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["method"] = req.method
        seen["path"] = req.url.path
        seen["team"] = req.url.params.get("teamId")
        seen["headers"] = dict(req.headers)
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
