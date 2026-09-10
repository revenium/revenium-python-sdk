"""Shared outcome transport (BACK-777 Phase 2): key validation, team-id chain, typed 409."""
import json
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
    append_outcome_metrics_request,
    coerce_entity_version,
    parse_entity_version,
    post_with_retry,
    report_outcome_request,
    resolve_team_id,
    validate_outcome_key,
)

BASE = "https://api.revenium.example"


def _client(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


class TestValidateOutcomeKey:
    def test_metering_key_rejected_with_clear_message(self):
        with pytest.raises(ValueError, match="REVENIUM_WRITE_API_KEY"):
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
        # The server's own message stays the head of the string; the SDK appends
        # the recovery loop after it (BACK-3079).
        assert str(conflict.value).startswith(
            "Outcome was updated concurrently; refetch and retry"
        )

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

    def test_409_exposes_the_version_the_backend_holds(self):
        """The only way a caller can retry with a lock: the 409 reports it.

        The live backend puts the value in the message text only (no structured
        field), so that text is what has to be read.
        """
        body = {
            "timestamp": "2026-09-09T14:14:11Z", "status": 409, "error": "Conflict",
            "message": "Outcome has changed since entity version 0; "
                       "current version is 1. Refetch and retry.",
            "details": {"error": "Outcome has changed since entity version 0; "
                                 "current version is 1. Refetch and retry."},
        }
        client = _client(lambda req: httpx.Response(409, json=body))
        with pytest.raises(OutcomeAmendConflictError) as conflict:
            amend_outcome_request(client, BASE, "j", {"reason": "r", "expectedEntityVersion": 0},
                                  team_id="t", api_key="rev_sk_T_a")
        assert conflict.value.current_entity_version == 1

    def test_409_prefers_a_structured_version_when_offered(self):
        """A backend that starts sending the field must win over the prose."""
        body = {
            "status": 409, "error": "Conflict",
            "message": "Outcome has changed since entity version 0; current version is 1.",
            "details": {"currentEntityVersion": 4},
        }
        client = _client(lambda req: httpx.Response(409, json=body))
        with pytest.raises(OutcomeAmendConflictError) as conflict:
            amend_outcome_request(client, BASE, "j", {"reason": "r"},
                                  team_id="t", api_key="rev_sk_T_a")
        assert conflict.value.current_entity_version == 4

    def test_409_without_a_version_reports_none(self):
        """Still a conflict; the caller just has to read the job themselves."""
        for body in ({"error": "concurrent amendment"}, {"message": "Conflict"}):
            client = _client(lambda req: httpx.Response(409, json=body))
            with pytest.raises(OutcomeAmendConflictError) as conflict:
                amend_outcome_request(client, BASE, "j", {"reason": "r"},
                                      team_id="t", api_key="rev_sk_T_a")
            assert conflict.value.current_entity_version is None

    def test_409_with_a_non_json_body_reports_none(self):
        client = _client(lambda req: httpx.Response(409, content=b"<html>gateway</html>"))
        with pytest.raises(OutcomeAmendConflictError) as conflict:
            amend_outcome_request(client, BASE, "j", {"reason": "r"},
                                  team_id="t", api_key="rev_sk_T_a")
        assert conflict.value.current_entity_version is None

    def test_409_message_names_the_recovery_loop(self):
        """A conflict has to tell the caller how to recover, not just that it lost.

        The SDK never auto-retries an amendment, so the message is the only
        place the refetch-then-retry-with-the-new-version loop is stated at the
        moment the caller needs it.
        """
        body = {
            "status": 409, "error": "Conflict",
            "message": "Outcome has changed since entity version 3; current version is 5. "
                       "Refetch and retry.",
        }
        client = _client(lambda req: httpx.Response(409, json=body))
        with pytest.raises(OutcomeAmendConflictError) as conflict:
            amend_outcome_request(client, BASE, "j", {"reason": "r", "expectedEntityVersion": 3},
                                  team_id="t", api_key="rev_sk_T_a")
        message = str(conflict.value)
        assert "current version is 5" in message
        assert "current_entity_version" in message
        assert "get_outcome_history" in message
        assert "expected_entity_version" in message


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


class TestParseEntityVersion:
    """``entityVersion`` is the optimistic-lock token; parsing it must never raise.

    The field arrives on every job/outcome response from a current backend and
    is simply absent on an older one, so every unusable shape has to degrade to
    ``None`` rather than break an outcome call that otherwise succeeded.
    """

    def _response(self, **kwargs):
        return httpx.Response(request=httpx.Request("POST", f"{BASE}/x"), **kwargs)

    def test_parses_an_int(self):
        assert parse_entity_version(self._response(status_code=200, json={"entityVersion": 4})) == 4

    def test_parses_version_zero(self):
        # A freshly created job is at version 0, which is a real version and
        # must not be confused with "no version known".
        assert parse_entity_version(self._response(status_code=200, json={"entityVersion": 0})) == 0

    def test_coerces_a_string_version(self):
        assert parse_entity_version(
            self._response(status_code=200, json={"entityVersion": "12"})
        ) == 12

    def test_absent_field_is_none(self):
        assert parse_entity_version(self._response(status_code=200, json={"id": "job-1"})) is None

    def test_non_json_body_is_none(self):
        assert parse_entity_version(self._response(status_code=204, content=b"")) is None

    def test_non_dict_body_is_none(self):
        assert parse_entity_version(self._response(status_code=200, json=[1, 2])) is None

    def test_unparseable_value_is_none(self):
        assert parse_entity_version(
            self._response(status_code=200, json={"entityVersion": "not-a-number"})
        ) is None

    def test_fractional_value_is_none(self):
        """3.5 is not version 3 — truncating it would invent a lock token."""
        assert parse_entity_version(
            self._response(status_code=200, json={"entityVersion": 3.5})
        ) is None

    def test_integral_float_is_accepted(self):
        assert parse_entity_version(
            self._response(status_code=200, json={"entityVersion": 3.0})
        ) == 3

    def test_negative_value_is_none(self):
        """The SDK could not legally send it back, so it is not worth recording."""
        assert parse_entity_version(
            self._response(status_code=200, json={"entityVersion": -1})
        ) is None

    def test_bool_is_not_a_version(self):
        assert parse_entity_version(
            self._response(status_code=200, json={"entityVersion": True})
        ) is None

    def test_missing_response_is_none(self):
        assert parse_entity_version(None) is None


class TestCoerceEntityVersion:
    """One definition of "is a version", shared by the parse and caller paths."""

    def test_accepts_integers_including_zero(self):
        assert coerce_entity_version(0) == 0
        assert coerce_entity_version(12) == 12

    def test_accepts_integral_floats_and_digit_strings(self):
        assert coerce_entity_version(3.0) == 3
        assert coerce_entity_version("3") == 3
        assert coerce_entity_version(" 3 ") == 3

    def test_rejects_fractions_negatives_bools_and_junk(self):
        for value in (3.5, -1, -1.0, "-1", "3.0", True, False, None, object(), [3]):
            assert coerce_entity_version(value) is None, value


class TestReportOutcomeRequest:
    def test_returns_the_response_so_entity_version_reaches_the_caller(self):
        def handler(req: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"id": "job-1", "entityVersion": 3})

        response = report_outcome_request(
            _client(handler), BASE, "job-1", {"executionStatus": "SUCCESS"},
            team_id="team-9", api_key="rev_sk_T_a",
        )
        assert response is not None
        assert parse_entity_version(response) == 3

    def test_response_without_entity_version_is_still_returned(self):
        """An older backend answers 200 with no version; the report still succeeded."""
        response = report_outcome_request(
            _client(lambda req: httpx.Response(200, json={"id": "job-1"})),
            BASE, "job-1", {"executionStatus": "SUCCESS"},
            team_id="team-9", api_key="rev_sk_T_a",
        )
        assert parse_entity_version(response) is None


def test_the_exact_409_sentence_hypercurrent_emits_is_the_wire_contract_until_back_3122():
    """Pin the sentence _conflict_entity_version parses (JobService, hypercurrent).

    The platform returns the current version only in prose today; BACK-3122 asks
    for a structured details.currentEntityVersion, which this parser already
    prefers. Until that ships, a rewording upstream must fail here, not silently
    turn the SDK's recovery loop into "version unknown".
    """
    from revenium_middleware._core.outcomes import _CURRENT_VERSION_IN_MESSAGE

    sentence = "Outcome has changed since entity version 7; current version is 8. Refetch and retry."
    match = _CURRENT_VERSION_IN_MESSAGE.search(sentence)
    assert match is not None and match.group(1) == "8"
    assert _CURRENT_VERSION_IN_MESSAGE.search("Outcome has changed; please retry") is None


class TestAppendOutcomeMetricsRequest:
    """Appending late per-job facts (BACK-3080).

    The endpoint's request body is a bare JSON array of OutcomeMetricEntry, not
    an object, and it answers a bodiless 201 — so the assertions here are about
    the wire shape and about the error paths staying untyped, since the SDK has
    no typed exception for an undeclared metric.
    """

    def _recorder(self, responses):
        seen = []

        def handler(req: httpx.Request) -> httpx.Response:
            seen.append(req)
            return responses[min(len(seen) - 1, len(responses) - 1)]

        return _client(handler), seen

    def test_sends_a_bare_array_to_the_metrics_path(self):
        client, seen = self._recorder([httpx.Response(201)])
        entries = [
            {"key": "quality_rate", "value": 0.93, "provenance": "MEASURED"},
            {"key": "cases_closed", "value": 4},
        ]
        append_outcome_metrics_request(
            client, BASE, "job 1", entries, team_id="team-9", api_key="rev_sk_T_a",
        )
        assert len(seen) == 1
        req = seen[0]
        assert req.method == "POST"
        # raw_path keeps the percent-encoding, so this really checks the quoting.
        path = req.url.raw_path.decode().split("?", 1)[0]
        assert path.endswith("/profitstream/v2/api/jobs/job%201/outcome/metrics")
        assert req.url.params["teamId"] == "team-9"
        assert req.headers["x-api-key"] == "rev_sk_T_a"
        assert req.headers["Content-Type"] == "application/json"
        # The body is the array itself — never wrapped in a "metrics" key.
        assert json.loads(req.content) == entries

    def test_entries_are_forwarded_without_server_side_defaults(self):
        """provenance/recordedBy/source have documented server defaults.

        Filling any of them in client-side would misattribute a fact, so an
        entry reaches the wire with exactly the keys the caller supplied.
        """
        client, seen = self._recorder([httpx.Response(201)])
        append_outcome_metrics_request(
            client, BASE, "job-1", [{"key": "quality_rate", "value": 1}],
            team_id="t", api_key="rev_sk_T_a",
        )
        assert json.loads(seen[0].content) == [{"key": "quality_rate", "value": 1}]

    @pytest.mark.parametrize("status", [502, 503, 504, 404])
    def test_ambiguous_failures_are_not_retried(self, status):
        """A repeat of this POST is a second fact, so only a provable no-op repeats.

        A gateway 5xx can arrive after the origin committed the append, and this
        endpoint has no idempotency key — a retry would either duplicate the
        fact or answer 409 for a write that already succeeded. 404 goes with
        them: unlike the outcome POST, which retries it through the Job's
        asynchronous creation, a fact is appended to a job that already exists,
        so an immediate answer beats minutes of blocking retry.
        """
        client, seen = self._recorder([httpx.Response(status, json={})])
        with pytest.raises(httpx.HTTPStatusError):
            append_outcome_metrics_request(
                client, BASE, "job-1", [{"key": "quality_rate", "value": 0.5}],
                team_id="t", api_key="rev_sk_T_a",
                retry_initial_seconds=0.01, retry_max_seconds=0.02,
            )
        assert len(seen) == 1

    def test_429_is_retried_because_it_proves_the_request_was_rejected(self, monkeypatch):
        slept = []
        monkeypatch.setattr("revenium_middleware._core.outcomes.time.sleep", slept.append)
        client, seen = self._recorder([
            httpx.Response(429, headers={"Retry-After": "1"}, json={}),
            httpx.Response(201),
        ])
        response = append_outcome_metrics_request(
            client, BASE, "job-1", [{"key": "quality_rate", "value": 0.5}],
            team_id="t", api_key="rev_sk_T_a", retry_max_seconds=5.0,
        )
        assert len(seen) == 2 and response.status_code == 201
        # Retry-After honored with the usual 1s buffer, under the cap.
        assert slept == [2.0]

    def test_400_undeclared_metric_raises_immediately(self):
        """A metric the job type does not declare is a caller error, not transient."""
        client, seen = self._recorder([
            httpx.Response(400, json={"message": "key 'quality_rate' is not declared for the job type"}),
        ])
        with pytest.raises(httpx.HTTPStatusError):
            append_outcome_metrics_request(
                client, BASE, "job-1", [{"key": "quality_rate", "value": 0.5}],
                team_id="t", api_key="rev_sk_T_a",
            )
        assert len(seen) == 1

    def test_409_uses_the_generic_error_path(self):
        """A duplicate active fact is a 409, but it is not an outcome conflict.

        Neither OutcomeAlreadyReportedError (an outcome already exists) nor
        OutcomeAmendConflictError (an optimistic-lock mismatch) describes it, so
        the append deliberately raises the plain transport error instead of
        borrowing an exception whose recovery advice would be wrong.
        """
        client, seen = self._recorder([httpx.Response(409, json={"message": "duplicate fact"})])
        with pytest.raises(httpx.HTTPStatusError):
            append_outcome_metrics_request(
                client, BASE, "job-1", [{"key": "quality_rate", "value": 0.5}],
                team_id="t", api_key="rev_sk_T_a",
            )
        assert len(seen) == 1

    def test_bodiless_201_yields_no_entity_version(self):
        """The endpoint answers 201 with no body, so there is no version to read."""
        client, _ = self._recorder([httpx.Response(201)])
        response = append_outcome_metrics_request(
            client, BASE, "job-1", [{"key": "quality_rate", "value": 0.5}],
            team_id="t", api_key="rev_sk_T_a",
        )
        assert response.status_code == 201
        assert parse_entity_version(response) is None

    def test_entity_version_is_read_when_a_backend_returns_one(self):
        client, _ = self._recorder([httpx.Response(200, json={"entityVersion": 6})])
        response = append_outcome_metrics_request(
            client, BASE, "job-1", [{"key": "quality_rate", "value": 0.5}],
            team_id="t", api_key="rev_sk_T_a",
        )
        assert parse_entity_version(response) == 6

    def test_team_id_param_omitted_when_unresolved(self):
        client, seen = self._recorder([httpx.Response(201)])
        append_outcome_metrics_request(
            client, BASE, "job-1", [{"key": "quality_rate", "value": 0.5}],
            team_id="", api_key="rev_sk_T_a",
        )
        assert "teamId" not in seen[0].url.params


class TestSharedHelperAcceptsAnArrayBody:
    def test_post_with_retry_sends_a_list_body_unwrapped(self):
        """The shared helper had to be widened, not the payload wrapped.

        Every other outcome call sends an object; the facts endpoint takes a
        bare array, and wrapping it in a key the server does not expect would
        be a 400.
        """
        seen = {}

        def handler(req: httpx.Request) -> httpx.Response:
            seen["body"] = json.loads(req.content)
            return httpx.Response(201)

        post_with_retry(_client(handler), f"{BASE}/x", params=None,
                        body=[{"key": "quality_rate", "value": 0.5}],
                        api_key="rev_sk_T_a")
        assert seen["body"] == [{"key": "quality_rate", "value": 0.5}]


class TestRetryStatusesPolicy:
    def test_default_policy_is_unchanged_for_the_other_callers(self):
        """Narrowing the policy for one caller must not narrow it for the rest."""
        seen = []

        def handler(req: httpx.Request) -> httpx.Response:
            seen.append(req)
            return httpx.Response(503, json={}) if len(seen) == 1 else httpx.Response(200, json={})

        response = report_outcome_request(
            _client(handler), BASE, "job-1", {"executionStatus": "SUCCESS"},
            team_id="t", api_key="rev_sk_T_a",
            retry_initial_seconds=0.01, retry_max_seconds=0.02,
        )
        assert len(seen) == 2 and response.status_code == 200

    def test_explicit_retry_statuses_replaces_the_default_set(self):
        seen = []

        def handler(req: httpx.Request) -> httpx.Response:
            seen.append(req)
            return httpx.Response(503, json={})

        with pytest.raises(httpx.HTTPStatusError):
            post_with_retry(_client(handler), f"{BASE}/x", params=None, body={},
                            api_key="rev_sk_T_a", retry_statuses={429},
                            retry_initial_seconds=0.01, retry_max_seconds=0.02)
        assert len(seen) == 1
