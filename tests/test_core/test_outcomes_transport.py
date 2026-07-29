"""Shared outcome transport (BACK-777 Phase 2): key validation, team-id chain, typed 409."""
import os
from unittest.mock import patch

import httpx
import pytest

from revenium_middleware._core.exceptions import (
    OutcomeAlreadyReportedError,
    OutcomeAmendConflictError,
    OutcomeNotReportedError,
    OutcomeReportingError,
)
from revenium_middleware._core.outcomes import (
    amend_outcome_request,
    post_with_retry,
    resolve_team_id,
    validate_outcome_key,
)

BASE = "https://api.revenium.example"


def _client(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


class TestValidateOutcomeKey:
    def test_metering_key_rejected_with_clear_message(self):
        with pytest.raises(ValueError, match="write-scope"):
            validate_outcome_key("rev_mk_TENANT_abc")

    def test_write_key_passes(self):
        assert validate_outcome_key("rev_sk_TENANT_abc") == "rev_sk_TENANT_abc"

    def test_legacy_key_not_rejected(self):
        assert validate_outcome_key("hak_legacy_key") == "hak_legacy_key"


class TestResolveTeamId:
    def test_explicit_wins(self):
        client = _client(lambda req: httpx.Response(500))
        assert resolve_team_id("team-X", "rev_sk_T_a", client, BASE) == "team-X"

    def test_env_used_when_enabled(self):
        client = _client(lambda req: httpx.Response(500))
        with patch.dict(os.environ, {"REVENIUM_TEAM_ID": "team-env"}):
            assert resolve_team_id("", "rev_sk_T_a", client, BASE) == "team-env"

    def test_env_skipped_when_disabled(self):
        def handler(req):
            assert req.url.path.endswith("/profitstream/v2/api/teams")
            assert req.url.params["tenantId"] == "TENANT"
            return httpx.Response(200, json={"_embedded": {"teamResourceList": [{"id": "team-api"}]}})

        with patch.dict(os.environ, {"REVENIUM_TEAM_ID": "team-env"}):
            assert resolve_team_id("", "rev_sk_TENANT_a", _client(handler), BASE, use_env=False) == "team-api"

    def test_auto_resolution_from_teams_api(self):
        def handler(req):
            return httpx.Response(200, json={"_embedded": {"teamResourceList": [{"id": "team-42"}]}})

        with patch.dict(os.environ, {}, clear=True):
            assert resolve_team_id("", "rev_sk_TENANT_a", _client(handler), BASE) == "team-42"

    def test_returns_empty_on_malformed_key(self):
        client = _client(lambda req: httpx.Response(500))
        with patch.dict(os.environ, {}, clear=True):
            assert resolve_team_id("", "bogus", client, BASE) == ""

    def test_returns_empty_on_api_failure(self):
        client = _client(lambda req: httpx.Response(503))
        with patch.dict(os.environ, {}, clear=True):
            assert resolve_team_id("", "rev_sk_TENANT_a", client, BASE) == ""

    def test_returns_empty_on_malformed_json_body(self):
        client = _client(lambda req: httpx.Response(200, content=b"not json"))
        with patch.dict(os.environ, {}, clear=True):
            assert resolve_team_id("", "rev_sk_TENANT_a", client, BASE) == ""


def _raise_on_409(body):
    client = _client(lambda req: httpx.Response(409, json=body))
    return post_with_retry(client, f"{BASE}/x", params=None, body={},
                           api_key="rev_sk_T_a", raise_typed_on_409=True)


# The body the backend actually sends (ErrorHandler.handleOutcomeAlreadyReported):
# human message in "message" ("error" is the HTTP reason phrase), conflict fields
# nested under "details", count named updateCount and serialized as a string.
REAL_409_BODY = {
    "timestamp": "2026-04-02T10:00:01.123+00:00",
    "status": 409,
    "error": "Conflict",
    "message": "Outcome already reported",
    "path": "/profitstream/v2/api/jobs/job-1/outcome",
    "details": {
        "guidance": "Use PATCH /v2/api/jobs/{jobId}/outcome to update",
        "reportedAt": "2026-04-02T10:00:00Z",
        "updateCount": "2",
    },
}


class TestTyped409:
    def test_real_backend_body_raises_typed_exception(self):
        with pytest.raises(OutcomeAlreadyReportedError) as exc_info:
            _raise_on_409(REAL_409_BODY)
        exc = exc_info.value
        assert exc.reported_at == "2026-04-02T10:00:00Z"
        assert exc.amendment_count == 2
        # The backend sends "2"; callers must never receive a str here.
        assert isinstance(exc.amendment_count, int)
        # "error" is the reason phrase ("Conflict"); the human message wins.
        assert str(exc) == "Outcome already reported"

    def test_concurrent_write_race_body_raises_with_empty_details(self):
        body = dict(REAL_409_BODY, details={
            "guidance": "Use PATCH /v2/api/jobs/{jobId}/outcome to update",
        })
        with pytest.raises(OutcomeAlreadyReportedError) as exc_info:
            _raise_on_409(body)
        assert exc_info.value.reported_at is None
        assert exc_info.value.amendment_count is None

    def test_legacy_flat_409_still_raises_typed_exception(self):
        body = {
            "error": "Outcome already reported",
            "guidance": "Use PATCH /v2/api/jobs/{id}/outcome to amend",
            "reportedAt": "2026-04-02T10:00:00Z",
            "amendmentCount": 2,
        }
        with pytest.raises(OutcomeAlreadyReportedError) as exc_info:
            _raise_on_409(body)
        assert exc_info.value.reported_at == "2026-04-02T10:00:00Z"
        assert exc_info.value.amendment_count == 2
        assert isinstance(exc_info.value.amendment_count, int)
        assert str(exc_info.value) == "Outcome already reported"

    def test_non_numeric_count_is_dropped_not_raised_on(self):
        body = dict(REAL_409_BODY, details=dict(REAL_409_BODY["details"], updateCount="many"))
        with pytest.raises(OutcomeAlreadyReportedError) as exc_info:
            _raise_on_409(body)
        assert exc_info.value.amendment_count is None
        assert exc_info.value.reported_at == "2026-04-02T10:00:00Z"

    def test_json_409_without_guidance_warns_and_returns(self, caplog):
        import logging
        body = {"status": 409, "error": "Conflict", "message": "Something else", "details": {}}
        with caplog.at_level(logging.WARNING, logger="revenium_middleware._core.outcomes"):
            response = _raise_on_409(body)
        assert response is not None and response.status_code == 409
        assert "409" in caplog.text

    def test_unparseable_409_warns_and_returns(self, caplog):
        import logging
        client = _client(lambda req: httpx.Response(409, content=b"conflict"))
        with caplog.at_level(logging.WARNING, logger="revenium_middleware._core.outcomes"):
            response = post_with_retry(client, f"{BASE}/x", params=None, body={},
                                       api_key="rev_sk_T_a", raise_typed_on_409=True)
        assert response is not None and response.status_code == 409
        assert "409" in caplog.text or "already" in caplog.text.lower()

    def test_accept_409_returns_without_raising(self):
        client = _client(lambda req: httpx.Response(409, json={"error": "exists"}))
        response = post_with_retry(client, f"{BASE}/x", params=None, body={},
                                   api_key="rev_sk_T_a", accept_409=True)
        assert response is not None and response.status_code == 409

    def test_typed_exception_is_subclass_of_family_base(self):
        assert issubclass(OutcomeAlreadyReportedError, OutcomeReportingError)


class TestUnexpected2xxSuccess:
    def test_unexpected_2xx_returns_once_on_post(self):
        counter = {"n": 0}

        def handler(req):
            counter["n"] += 1
            return httpx.Response(204)

        response = post_with_retry(_client(handler), f"{BASE}/x", params=None, body={},
                                   api_key="rev_sk_T_a", retry_attempts=5)
        assert counter["n"] == 1 and response.status_code == 204


class TestAmendOutcomeRequest:
    def test_patch_method_url_and_payload(self):
        seen = {}

        def handler(req: httpx.Request) -> httpx.Response:
            seen["method"] = req.method
            # httpx's url.path percent-decodes; use the raw wire path so the
            # assertion actually verifies the job id was URL-encoded.
            seen["path"] = req.url.raw_path.decode().split("?", 1)[0]
            seen["team"] = req.url.params.get("teamId")
            return httpx.Response(200, json={"id": "job-1", "outcomeAmendmentCount": 1})

        response = amend_outcome_request(
            _client(handler), BASE, "job 1", {"reason": "value changed"},
            team_id="team-9", api_key="rev_sk_T_a",
        )
        assert seen["method"] == "PATCH"
        assert seen["path"].endswith("/profitstream/v2/api/jobs/job%201/outcome")
        assert seen["team"] == "team-9"
        assert response.json()["outcomeAmendmentCount"] == 1

    def test_422_raises_not_reported(self):
        client = _client(lambda req: httpx.Response(422, json={"error": "No outcome to amend"}))
        with pytest.raises(OutcomeNotReportedError):
            amend_outcome_request(client, BASE, "j", {"reason": "r"},
                                  team_id="t", api_key="rev_sk_T_a")

    def test_409_raises_amend_conflict(self):
        client = _client(lambda req: httpx.Response(409, json={"error": "concurrent amendment"}))
        with pytest.raises(OutcomeAmendConflictError):
            amend_outcome_request(client, BASE, "j", {"reason": "r"},
                                  team_id="t", api_key="rev_sk_T_a")

    def test_real_backend_bodies_carry_the_human_message(self):
        """On real bodies "error" is the reason phrase; "message" is the message."""
        conflict_body = {
            "status": 409, "error": "Conflict",
            "message": "Outcome was updated concurrently; refetch and retry",
            "details": {},
        }
        client = _client(lambda req: httpx.Response(409, json=conflict_body))
        with pytest.raises(OutcomeAmendConflictError) as conflict:
            amend_outcome_request(client, BASE, "j", {"reason": "r"},
                                  team_id="t", api_key="rev_sk_T_a")
        assert str(conflict.value) == "Outcome was updated concurrently; refetch and retry"

        missing_body = {
            "status": 422, "error": "Unprocessable Entity",
            "message": "Job has no outcome to update",
        }
        client = _client(lambda req: httpx.Response(422, json=missing_body))
        with pytest.raises(OutcomeNotReportedError) as missing:
            amend_outcome_request(client, BASE, "j", {"reason": "r"},
                                  team_id="t", api_key="rev_sk_T_a")
        assert str(missing.value) == "Job has no outcome to update"

    def test_404_is_not_retried(self):
        counter = {"n": 0}

        def handler(req):
            counter["n"] += 1
            return httpx.Response(404, json={"error": "no such job"})

        with pytest.raises(httpx.HTTPStatusError):
            amend_outcome_request(_client(handler), BASE, "j", {"reason": "r"},
                                  team_id="t", api_key="rev_sk_T_a")
        assert counter["n"] == 1

    def test_5xx_is_not_retried_on_amend(self):
        counter = {"n": 0}

        def handler(req):
            counter["n"] += 1
            return httpx.Response(502, json={})

        with pytest.raises(httpx.HTTPStatusError):
            amend_outcome_request(_client(handler), BASE, "j", {"reason": "r"},
                                  team_id="t", api_key="rev_sk_T_a")
        assert counter["n"] == 1

    def test_429_still_retried_on_amend(self):
        counter = {"n": 0}

        def handler(req):
            counter["n"] += 1
            if counter["n"] == 1:
                return httpx.Response(429, json={})
            return httpx.Response(200, json={"id": "job-1"})

        response = amend_outcome_request(_client(handler), BASE, "j", {"reason": "r"},
                                         team_id="t", api_key="rev_sk_T_a",
                                         retry_initial_seconds=0.01, retry_max_seconds=0.02)
        assert counter["n"] == 2 and response.status_code == 200

    def test_unexpected_2xx_returns_once_on_amend(self):
        counter = {"n": 0}

        def handler(req):
            counter["n"] += 1
            return httpx.Response(204)

        response = amend_outcome_request(_client(handler), BASE, "j", {"reason": "r"},
                                         team_id="t", api_key="rev_sk_T_a", retry_attempts=5)
        assert counter["n"] == 1 and response.status_code == 204


class TestHistoryRequestRetry:
    def test_history_get_retries_transient_failures(self):
        counter = {"n": 0}

        def handler(req):
            counter["n"] += 1
            if counter["n"] == 1:
                return httpx.Response(503, json={})
            return httpx.Response(200, json=[])

        from revenium_middleware._core.outcomes import get_outcome_history_request
        response = get_outcome_history_request(_client(handler), BASE, "j",
                                               team_id="t", api_key="rev_sk_T_a",
                                               retry_initial_seconds=0.01,
                                               retry_max_seconds=0.02)
        assert counter["n"] == 2 and response.status_code == 200
