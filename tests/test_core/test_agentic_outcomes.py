"""Unit tests for revenium_middleware.agentic_outcomes."""
from __future__ import annotations

import contextlib
import copy
import dataclasses
import inspect
import json
import socket
import threading
import time
import uuid

import httpx
import pytest
from typing_extensions import Annotated, Literal, Required, get_args, get_origin

from revenium_middleware._core import metering, metering_pool, tool_event_transport
from revenium_middleware._core.exceptions import OutcomeAlreadyReportedError
from revenium_middleware._core.prompt_extraction import sanitize_credentials
from revenium_middleware._metering import ReveniumMetering
from revenium_middleware._metering._utils._transform import _maybe_transform_key, get_type_hints
from revenium_middleware._metering.resources.ai import AIResource
from revenium_middleware._metering.types.ai_create_completion_params import AICreateCompletionParams
from revenium_middleware.agentic_outcomes import AgenticOutcomeClient, AgenticOutcomeSettings

# The body the backend answers a second outcome report with, as recorded on dev.
_ALREADY_REPORTED_BODY = {
    "details": {
        "guidance": "Use PATCH /v2/api/jobs/job-1/outcome to update",
        "reportedAt": "2026-10-08T16:03:15.556Z",
        "updateCount": "0",
    },
    "error": "Conflict",
    "message": "Outcome already reported",
    "path": "/profitstream/v2/api/jobs/job-1/outcome",
    "status": 409,
    "timestamp": "2026-10-08T16:03:16.561558419Z",
}


def _settings(**overrides) -> AgenticOutcomeSettings:
    base = dict(
        api_key="rev_sk_test_xxx",
        meter_base_url="https://api.example.test",
        profitstream_base_url="https://api.example.test",
        team_id="TEAM",
        outcome_retry_attempts=3,
        outcome_retry_initial_seconds=0.0,
        outcome_retry_max_seconds=10.0,
    )
    base.update(overrides)
    return AgenticOutcomeSettings(**base)


def test_settings_is_frozen_dataclass():
    s = _settings()
    assert dataclasses.is_dataclass(s)
    with pytest.raises(dataclasses.FrozenInstanceError):
        s.api_key = "mutated"  # type: ignore[misc]


def test_settings_urls_built_correctly():
    s = _settings(meter_base_url="https://api.example.test/")
    assert s.completion_url == "https://api.example.test/meter/v2/ai/completions"
    assert s.tool_url == "https://api.example.test/meter/v2/tool/events"
    assert s.sdk_meter_base_url == "https://api.example.test/meter/"


def test_settings_accepts_outcome_api_key_kwarg():
    s = _settings(outcome_api_key="rev_sk_other")
    assert s.outcome_api_key == "rev_sk_other"


def test_client_close_lifecycle():
    transport = httpx.MockTransport(lambda req: httpx.Response(200, json={}))
    http = httpx.Client(transport=transport)
    client = AgenticOutcomeClient(_settings(), http_client=http)
    # Caller-provided http_client → close() must NOT close it.
    client.close()
    assert not http.is_closed
    http.close()


def test_client_owns_default_http_client():
    client = AgenticOutcomeClient(_settings())
    assert not client.http_client.is_closed
    client.close()
    assert client.http_client.is_closed


def test_emit_tool_event_dry_run(capsys):
    client = AgenticOutcomeClient(_settings())
    payload = {"transactionId": "tx-1", "toolId": "demo.tool"}
    out = client.emit_tool_event(payload, dry_run=True)
    captured = capsys.readouterr()
    assert "DRY-RUN" in captured.out
    assert out == payload
    client.close()


def _tool_events_sent(monkeypatch, settings, *payloads, status=200) -> list:
    stub = _ToolEventClientAnswering(status)
    monkeypatch.setattr(tool_event_transport, "_client", stub)
    with AgenticOutcomeClient(settings) as client:
        for payload in payloads:
            client.emit_tool_event(payload)
        _wait_for_delivery()
    return stub.calls


@pytest.mark.usefixtures("isolated_metering")
def test_emit_tool_event_uses_single_api_key(monkeypatch):
    (sent,) = _tool_events_sent(monkeypatch, _settings(outcome_api_key="ignored"), {"transactionId": "tx-1"})
    assert sent["headers"]["x-api-key"] == "rev_sk_test_xxx"  # api_key, NOT outcome_api_key
    assert sent["url"].endswith("/meter/v2/tool/events")


def test_report_outcome_dry_run(capsys):
    client = AgenticOutcomeClient(_settings())
    client.report_outcome("job-123", {"executionStatus": "success"}, dry_run=True)
    out = capsys.readouterr().out
    assert "DRY-RUN" in out
    assert "/profitstream/v2/api/jobs/job-123/outcome" in out
    client.close()


def test_report_outcome_retries_404_then_succeeds():
    calls = []

    def handler(req: httpx.Request) -> httpx.Response:
        calls.append(req)
        if len(calls) == 1:
            return httpx.Response(404, json={"error": "not found yet"})
        return httpx.Response(200, json={})

    transport = httpx.MockTransport(handler)
    http = httpx.Client(transport=transport)
    client = AgenticOutcomeClient(_settings(), http_client=http)
    client.report_outcome("job-async", {"executionStatus": "success"})
    assert len(calls) == 2
    client.close()


def test_retry_after_header_clamped_by_max():
    """HIGH adversary finding: server-supplied Retry-After must be clamped, never raw."""
    import time as time_mod

    sleeps: list[float] = []
    monkey_sleep = lambda secs: sleeps.append(secs)  # noqa: E731

    calls = []

    def handler(req: httpx.Request) -> httpx.Response:
        calls.append(req)
        if len(calls) == 1:
            # Server says "wait 3600 seconds" — implementation MUST clamp.
            return httpx.Response(429, headers={"Retry-After": "3600"}, json={})
        return httpx.Response(200, json={})

    transport = httpx.MockTransport(handler)
    http = httpx.Client(transport=transport)
    settings = _settings(outcome_retry_max_seconds=5.0)
    client = AgenticOutcomeClient(settings, http_client=http)

    # Patch time.sleep so the test doesn't actually wait.
    original_sleep = time_mod.sleep
    time_mod.sleep = monkey_sleep
    try:
        client.report_outcome("job-X", {"executionStatus": "success"})
    finally:
        time_mod.sleep = original_sleep

    assert len(sleeps) == 1
    assert sleeps[0] <= 5.0, f"Retry-After=3600 was not clamped to max=5.0 (got {sleeps[0]})"
    client.close()


def test_create_job_treats_409_as_idempotent_success():
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(409, json={"error": "exists"})

    transport = httpx.MockTransport(handler)
    http = httpx.Client(transport=transport)
    client = AgenticOutcomeClient(_settings(), http_client=http)
    # Should NOT raise — re-runs of the same agenticJobId are idempotent.
    body = client.create_job("job-456", name="re-run")
    assert body["agenticJobId"] == "job-456"
    client.close()


def test_report_outcome_passes_outcome_reason_through():
    seen = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(req.content)
        return httpx.Response(200, json={})

    transport = httpx.MockTransport(handler)
    http = httpx.Client(transport=transport)
    client = AgenticOutcomeClient(_settings(), http_client=http)
    client.report_outcome(
        "job-fail",
        {"executionStatus": "FAILED", "outcomeReason": "Customer unreachable"},
    )
    assert seen["body"]["outcomeReason"] == "Customer unreachable"
    client.close()


def test_create_job_never_sends_outcome_reason():
    """outcomeReason is readOnly on the job resource — job create must not send it."""
    seen = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(req.content)
        return httpx.Response(200, json={})

    transport = httpx.MockTransport(handler)
    http = httpx.Client(transport=transport)
    client = AgenticOutcomeClient(_settings(), http_client=http)
    body = client.create_job("job-1", name="run", type="agent", version="1.0", environment="demo")
    assert "outcomeReason" not in body
    assert "outcomeReason" not in seen["body"]
    client.close()


def _tool_event_answered_then_replayed(meter, buffer, status) -> list:
    meter.answer_with(status)
    with AgenticOutcomeClient(_stub_settings(meter)) as client:
        client.emit_tool_event({"transactionId": "tool-tx-1", "toolId": "t1"})
        _wait_for_delivery()
        assert buffer.stats()["size"] == 1
        meter.answer_with(201)
        buffer.flush(deadline_seconds=5)
    assert buffer.stats()["size"] == 0
    return meter.requests


def test_emit_tool_event_buffers_a_429_and_replays_it(recording_meters, isolated_metering):
    requests = _tool_event_answered_then_replayed(recording_meters[0], isolated_metering, 429)
    assert [request["path"] for request in requests] == ["/meter/v2/tool/events"] * 2


def test_top_level_imports_available():
    import revenium_middleware as rm

    assert rm.AgenticOutcomeClient is AgenticOutcomeClient
    assert rm.AgenticOutcomeSettings is AgenticOutcomeSettings
    assert "AgenticOutcomeClient" in rm.__all__
    assert "AgenticOutcomeSettings" in rm.__all__


# ---------------------------------------------------------------------------
# Fix #10: KeyError → ValueError for missing required keys
# ---------------------------------------------------------------------------
@pytest.mark.usefixtures("isolated_metering")
def test_emit_completion_raises_value_error_on_missing_keys(recording_meters):
    a, _ = recording_meters
    client = AgenticOutcomeClient(_settings(meter_base_url=a.base_url))
    with pytest.raises(ValueError, match="Missing required keys"):
        client.emit_completion({"model": "gpt-5"})
    metering_pool.wait_until_idle(10)
    assert a.requests == []
    client.close()


# ---------------------------------------------------------------------------
# Fix #8: outcome_api_key, when set, is used for outcome calls (not metering)
# ---------------------------------------------------------------------------
def test_report_outcome_uses_outcome_api_key_when_set():
    seen = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["x_api_key"] = req.headers.get("x-api-key")
        return httpx.Response(200, json={})

    transport = httpx.MockTransport(handler)
    http = httpx.Client(transport=transport)
    settings = _settings(api_key="rev_mk_metering", outcome_api_key="rev_sk_write")
    client = AgenticOutcomeClient(settings, http_client=http)
    client.report_outcome("job-1", {"executionStatus": "success"})
    assert seen["x_api_key"] == "rev_sk_write"
    client.close()


def test_create_job_uses_outcome_api_key_when_set():
    seen = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["x_api_key"] = req.headers.get("x-api-key")
        return httpx.Response(200, json={})

    transport = httpx.MockTransport(handler)
    http = httpx.Client(transport=transport)
    settings = _settings(api_key="rev_mk_metering", outcome_api_key="rev_sk_write")
    client = AgenticOutcomeClient(settings, http_client=http)
    client.create_job("job-2")
    assert seen["x_api_key"] == "rev_sk_write"
    client.close()


def test_report_outcome_falls_back_to_api_key_when_outcome_key_unset():
    seen = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["x_api_key"] = req.headers.get("x-api-key")
        return httpx.Response(200, json={})

    transport = httpx.MockTransport(handler)
    http = httpx.Client(transport=transport)
    settings = _settings(api_key="rev_sk_only", outcome_api_key=None)
    client = AgenticOutcomeClient(settings, http_client=http)
    client.report_outcome("job-3", {"executionStatus": "success"})
    assert seen["x_api_key"] == "rev_sk_only"
    client.close()


@pytest.mark.usefixtures("isolated_metering")
def test_tool_event_still_uses_api_key_not_outcome_key(monkeypatch):
    settings = _settings(api_key="rev_mk_metering", outcome_api_key="rev_sk_write")
    (sent,) = _tool_events_sent(monkeypatch, settings, {"transactionId": "tx-1"})
    assert sent["headers"]["x-api-key"] == "rev_mk_metering"


# ---------------------------------------------------------------------------
# Fix #9: _get_team_id failures are logged, not silently swallowed
# ---------------------------------------------------------------------------
def test_get_team_id_logs_on_http_failure(caplog):
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "boom"})

    transport = httpx.MockTransport(handler)
    http = httpx.Client(transport=transport)
    settings = _settings(team_id="", api_key="rev_sk_tenantA_xyz")
    client = AgenticOutcomeClient(settings, http_client=http)
    with caplog.at_level("WARNING"):
        result = client._get_team_id()
    assert result == ""
    assert any("team" in r.message.lower() for r in caplog.records)
    client.close()


def test_get_team_id_logs_on_malformed_key(caplog):
    settings = _settings(team_id="", api_key="malformed")
    client = AgenticOutcomeClient(settings)
    with caplog.at_level("WARNING"):
        result = client._get_team_id()
    assert result == ""
    assert any("team" in r.message.lower() for r in caplog.records)
    client.close()


# ---------------------------------------------------------------------------
# Tests for emit_completion (previously uncovered)
# ---------------------------------------------------------------------------
def test_emit_completion_dry_run(capsys):
    client = AgenticOutcomeClient(_settings())
    payload = {
        "completionStartTime": "2026-01-01T00:00:00Z",
        "costType": "AI",
        "inputTokenCount": 100,
        "isStreamed": False,
        "model": "gpt-5",
        "outputTokenCount": 50,
        "provider": "chatgpt",
        "requestDuration": 1000,
        "requestTime": "2026-01-01T00:00:00Z",
        "responseTime": "2026-01-01T00:00:01Z",
        "stopReason": "END",
        "totalTokenCount": 150,
        "transactionId": "tx-1",
    }
    out = client.emit_completion(payload, dry_run=True)
    captured = capsys.readouterr()
    assert "DRY-RUN" in captured.out
    assert "/meter/v2/ai/completions" in captured.out
    assert out == payload
    client.close()


@pytest.mark.usefixtures("isolated_metering")
def test_emit_completion_forwards_to_metering_sdk():
    captured = {}

    class FakeAI:
        def create_completion(self, **kwargs):
            captured["kwargs"] = kwargs

    class FakeMetering:
        def __init__(self):
            self.ai = FakeAI()

    fake = FakeMetering()
    client = AgenticOutcomeClient(_settings(), metering_client=fake)
    payload = {
        "completionStartTime": "2026-01-01T00:00:00Z",
        "costType": "AI",
        "inputTokenCount": 100,
        "isStreamed": False,
        "model": "gpt-5",
        "outputTokenCount": 50,
        "provider": "chatgpt",
        "requestDuration": 1000,
        "requestTime": "2026-01-01T00:00:00Z",
        "responseTime": "2026-01-01T00:00:01Z",
        "stopReason": "END",
        "totalTokenCount": 150,
        "transactionId": "tx-1",
        "agenticJobId": "job-1",
    }
    client.emit_completion(payload)
    _wait_for_delivery()
    assert captured["kwargs"]["model"] == "gpt-5"
    assert captured["kwargs"]["transaction_id"] == "tx-1"
    # extra_body carries the agentic job fields
    assert captured["kwargs"]["extra_body"]["agenticJobId"] == "job-1"
    # agentVersion is absent from this payload, so it must not reach the
    # typed client at all (an explicit None would go on the wire as null).
    assert "agent_version" not in captured["kwargs"]
    client.close()


@pytest.mark.usefixtures("isolated_metering")
def test_emit_completion_forwards_agent_version_as_the_typed_keyword():
    captured = {}

    class FakeAI:
        def create_completion(self, **kwargs):
            captured["kwargs"] = kwargs

    class FakeMetering:
        def __init__(self):
            self.ai = FakeAI()

    client = AgenticOutcomeClient(_settings(), metering_client=FakeMetering())
    payload = {
        "completionStartTime": "2026-01-01T00:00:00Z",
        "costType": "AI",
        "inputTokenCount": 100,
        "isStreamed": False,
        "model": "gpt-5",
        "outputTokenCount": 50,
        "provider": "chatgpt",
        "requestDuration": 1000,
        "requestTime": "2026-01-01T00:00:00Z",
        "responseTime": "2026-01-01T00:00:01Z",
        "stopReason": "END",
        "totalTokenCount": 150,
        "transactionId": "tx-2",
        "agentVersion": "1.4.2",
        "agenticJobVersion": "7",
    }
    client.emit_completion(payload)
    _wait_for_delivery()
    assert captured["kwargs"]["agent_version"] == "1.4.2"
    # The job definition's version stays a separate field in extra_body.
    assert captured["kwargs"]["extra_body"]["agenticJobVersion"] == "7"
    client.close()


@pytest.mark.usefixtures("isolated_metering")
def test_emit_completion_validates_agent_version_like_the_other_paths():
    captured = {}

    class FakeAI:
        def create_completion(self, **kwargs):
            captured["kwargs"] = kwargs

    class FakeMetering:
        def __init__(self):
            self.ai = FakeAI()

    client = AgenticOutcomeClient(_settings(), metering_client=FakeMetering())
    base = {
        "completionStartTime": "2026-01-01T00:00:00Z",
        "costType": "AI",
        "inputTokenCount": 100,
        "isStreamed": False,
        "model": "gpt-5",
        "outputTokenCount": 50,
        "provider": "chatgpt",
        "requestDuration": 1000,
        "requestTime": "2026-01-01T00:00:00Z",
        "responseTime": "2026-01-01T00:00:01Z",
        "stopReason": "END",
        "totalTokenCount": 150,
        "transactionId": "tx-3",
    }
    # A non-string is dropped, not stringified.
    client.emit_completion({**base, "agentVersion": 142})
    _wait_for_delivery()
    assert "agent_version" not in captured["kwargs"]
    # An over-long value is capped at the ingest limit, not sent unchanged.
    client.emit_completion({**base, "agentVersion": "v" * 200})
    _wait_for_delivery()
    assert len(captured["kwargs"]["agent_version"]) == 64
    client.close()


# ---------------------------------------------------------------------------
# _get_team_id: auto-resolution success path
# ---------------------------------------------------------------------------
def test_get_team_id_resolves_from_api_key_prefix():
    def handler(req: httpx.Request) -> httpx.Response:
        assert "tenantId=tenantA" in str(req.url)
        return httpx.Response(
            200,
            json={"_embedded": {"teamResourceList": [{"id": "team-A1"}]}},
        )

    transport = httpx.MockTransport(handler)
    http = httpx.Client(transport=transport)
    settings = _settings(team_id="", api_key="rev_sk_tenantA_xyz")
    client = AgenticOutcomeClient(settings, http_client=http)
    assert client._get_team_id() == "team-A1"
    # Second call uses cache, no extra HTTP
    assert client._get_team_id() == "team-A1"
    client.close()


# ---------------------------------------------------------------------------
# Retry exhaustion: after N failed 429s, the final response.raise_for_status() runs
# ---------------------------------------------------------------------------
def test_report_outcome_raises_after_retry_exhaustion():
    calls = []

    def handler(req: httpx.Request) -> httpx.Response:
        calls.append(req)
        return httpx.Response(429, json={"error": "rate limited"})

    transport = httpx.MockTransport(handler)
    http = httpx.Client(transport=transport)
    settings = _settings(outcome_retry_attempts=3, outcome_retry_initial_seconds=0.0)
    client = AgenticOutcomeClient(settings, http_client=http)
    with pytest.raises(httpx.HTTPStatusError):
        client.report_outcome("job-X", {"executionStatus": "success"})
    assert len(calls) == 3
    client.close()


# ---------------------------------------------------------------------------
# dry_run must NOT trigger _get_team_id() network calls (tessie 3266673888).
# ---------------------------------------------------------------------------
def test_report_outcome_dry_run_makes_no_http_calls():
    calls = []

    def handler(req: httpx.Request) -> httpx.Response:
        calls.append(req)
        return httpx.Response(200, json={})

    transport = httpx.MockTransport(handler)
    http = httpx.Client(transport=transport)
    # team_id="" forces _get_team_id to attempt a network resolution if invoked.
    settings = _settings(team_id="")
    client = AgenticOutcomeClient(settings, http_client=http)
    client.report_outcome("job-dry", {"executionStatus": "success"}, dry_run=True)
    assert calls == []
    client.close()


def test_create_job_dry_run_makes_no_http_calls():
    calls = []

    def handler(req: httpx.Request) -> httpx.Response:
        calls.append(req)
        return httpx.Response(200, json={})

    transport = httpx.MockTransport(handler)
    http = httpx.Client(transport=transport)
    settings = _settings(team_id="")
    client = AgenticOutcomeClient(settings, http_client=http)
    client.create_job("job-dry", name="x", dry_run=True)
    assert calls == []
    client.close()


# ---------------------------------------------------------------------------
# _get_team_id uses _outcome_key() so split-key setups authenticate teams API
# with the same credential as create_job/report_outcome (tessie 3266673901).
# ---------------------------------------------------------------------------
def test_get_team_id_uses_outcome_api_key_for_teams_lookup():
    seen = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["x_api_key"] = req.headers.get("x-api-key")
        seen["params"] = dict(req.url.params)
        return httpx.Response(
            200,
            json={"_embedded": {"teamResourceList": [{"id": "team-write"}]}},
        )

    transport = httpx.MockTransport(handler)
    http = httpx.Client(transport=transport)
    settings = _settings(
        team_id="",
        api_key="rev_mk_tenantMK_metering",
        outcome_api_key="rev_sk_tenantSK_write",
    )
    client = AgenticOutcomeClient(settings, http_client=http)
    assert client._get_team_id() == "team-write"
    # Teams API authenticated with the outcome (write) key, not the metering key.
    assert seen["x_api_key"] == "rev_sk_tenantSK_write"
    # Tenant ID parsed from the outcome key, not the metering key.
    assert seen["params"]["tenantId"] == "tenantSK"
    client.close()


# ---------------------------------------------------------------------------
# Outcomes require a write-scope key: a metering key must fail client-side
# before any HTTP call, matching JobContext._resolve_api_key.
# ---------------------------------------------------------------------------
def test_report_outcome_rejects_metering_key_before_http():
    calls = []

    def handler(req: httpx.Request) -> httpx.Response:
        calls.append(req)
        return httpx.Response(200, json={})

    transport = httpx.MockTransport(handler)
    http = httpx.Client(transport=transport)
    settings = _settings(api_key="rev_mk_TENANT_abc", outcome_api_key=None)
    client = AgenticOutcomeClient(settings, http_client=http)
    with pytest.raises(ValueError, match="write-scope"):
        client.report_outcome("job-1", {"executionStatus": "SUCCESS"})
    assert calls == []
    client.close()


def test_create_job_rejects_metering_key_before_http():
    calls = []

    def handler(req: httpx.Request) -> httpx.Response:
        calls.append(req)
        return httpx.Response(200, json={})

    transport = httpx.MockTransport(handler)
    http = httpx.Client(transport=transport)
    settings = _settings(api_key="rev_mk_TENANT_abc", outcome_api_key=None)
    client = AgenticOutcomeClient(settings, http_client=http)
    with pytest.raises(ValueError, match="write-scope"):
        client.create_job("job-1")
    assert calls == []
    client.close()


@pytest.mark.usefixtures("isolated_metering")
def test_metering_paths_still_accept_a_metering_key(monkeypatch):
    # emit_tool_event authenticates with settings.api_key, not the outcome key —
    # a metering key must keep working there.
    settings = _settings(api_key="rev_mk_TENANT_abc", outcome_api_key=None)
    (sent,) = _tool_events_sent(monkeypatch, settings, {"transactionId": "tx-mk"})
    assert sent["headers"]["x-api-key"] == "rev_mk_TENANT_abc"


def test_create_job_returns_the_response_body_with_entity_version():
    """The examples-pack client needs the version without a second call (BACK-3079)."""
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"agenticJobId": "job-777", "entityVersion": 0})

    http = httpx.Client(transport=httpx.MockTransport(handler))
    client = AgenticOutcomeClient(_settings(), http_client=http)
    body = client.create_job("job-777", name="run")
    assert body["entityVersion"] == 0
    assert body["agenticJobId"] == "job-777"
    # Merged over the request body, so a response that omits a field the caller
    # supplied does not drop it from the return value.
    assert body["name"] == "run"
    client.close()


def test_create_job_response_wins_where_the_two_overlap():
    """The server is authoritative on the fields it does return."""
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"agenticJobId": "job-777", "name": "normalized",
                                         "id": "5jXkgO", "entityVersion": 2})

    http = httpx.Client(transport=httpx.MockTransport(handler))
    client = AgenticOutcomeClient(_settings(), http_client=http)
    body = client.create_job("job-777", name="run", type="agent")
    assert body["name"] == "normalized"
    assert body["type"] == "agent"
    assert body["id"] == "5jXkgO"
    client.close()


def test_create_job_409_still_returns_the_request_body_shape():
    """An idempotent re-run answers 409 with an error body, not a job resource."""
    http = httpx.Client(
        transport=httpx.MockTransport(lambda req: httpx.Response(409, json={"error": "exists"}))
    )
    client = AgenticOutcomeClient(_settings(), http_client=http)
    body = client.create_job("job-456", name="re-run")
    assert body["agenticJobId"] == "job-456"
    assert body["name"] == "re-run"
    assert "error" not in body
    client.close()


def test_create_job_tolerates_a_bodiless_response():
    """A 204 (or non-JSON) response must not turn a created job into an empty dict."""
    http = httpx.Client(transport=httpx.MockTransport(lambda req: httpx.Response(204)))
    client = AgenticOutcomeClient(_settings(), http_client=http)
    body = client.create_job("job-204")
    assert body["agenticJobId"] == "job-204"
    client.close()


# ---------------------------------------------------------------------------
# Late per-job metric facts (BACK-3080). The examples-pack client forwards
# entries as-is, like report_outcome forwards its payload.
# ---------------------------------------------------------------------------
def test_append_outcome_metrics_posts_a_bare_array():
    seen = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["method"] = req.method
        seen["path"] = req.url.path
        seen["team"] = req.url.params.get("teamId")
        seen["body"] = json.loads(req.content)
        return httpx.Response(201)

    http = httpx.Client(transport=httpx.MockTransport(handler))
    client = AgenticOutcomeClient(_settings(), http_client=http)
    entries = [{"key": "quality_rate", "value": 0.87, "provenance": "MEASURED"}]
    client.append_outcome_metrics("job-777", entries)
    assert seen["method"] == "POST"
    assert seen["path"] == "/profitstream/v2/api/jobs/job-777/outcome/metrics"
    assert seen["team"] == "TEAM"
    assert seen["body"] == entries
    client.close()


def test_append_outcome_metrics_dry_run_makes_no_http_calls(capsys):
    calls = []

    def handler(req: httpx.Request) -> httpx.Response:
        calls.append(req)
        return httpx.Response(201)

    http = httpx.Client(transport=httpx.MockTransport(handler))
    client = AgenticOutcomeClient(_settings(), http_client=http)
    client.append_outcome_metrics(
        "job-dry", [{"key": "quality_rate", "value": 0.5}], dry_run=True
    )
    out = capsys.readouterr().out
    assert "DRY-RUN" in out
    assert "/profitstream/v2/api/jobs/job-dry/outcome/metrics" in out
    assert calls == []
    client.close()


def test_append_outcome_metrics_rejects_metering_key_before_http():
    calls = []

    def handler(req: httpx.Request) -> httpx.Response:
        calls.append(req)
        return httpx.Response(201)

    http = httpx.Client(transport=httpx.MockTransport(handler))
    settings = _settings(api_key="rev_mk_TENANT_abc", outcome_api_key=None)
    client = AgenticOutcomeClient(settings, http_client=http)
    with pytest.raises(ValueError, match="write-scope"):
        client.append_outcome_metrics("job-1", [{"key": "quality_rate", "value": 0.5}])
    assert calls == []
    client.close()


@pytest.mark.parametrize("bad", [
    [],                                       # nothing to append
    None,                                     # an unset variable forwarded here
    {"key": "quality_rate", "value": 0.9},    # one entry passed unwrapped
    [{"value": 0.9}],                         # no key
])
def test_append_outcome_metrics_rejects_malformed_entries_before_http(bad):
    """The examples client shape-checks entries exactly as JobContext does.

    Without it an empty list reached the server, and a bare mapping became a
    JSON array of its keys.
    """
    calls = []

    def handler(req: httpx.Request) -> httpx.Response:
        calls.append(req)
        return httpx.Response(201)

    http = httpx.Client(transport=httpx.MockTransport(handler))
    client = AgenticOutcomeClient(_settings(), http_client=http)
    with pytest.raises(ValueError, match="metrics"):
        client.append_outcome_metrics("job-1", bad)
    assert calls == []
    client.close()


@pytest.mark.parametrize("bad", [[], None, {"key": "quality_rate", "value": 0.9}])
def test_append_outcome_metrics_dry_run_refuses_to_print_a_payload_it_would_reject(bad, capsys):
    """A dry run that prints an invalid body is worse than no dry run."""
    client = AgenticOutcomeClient(_settings())
    with pytest.raises(ValueError, match="metrics"):
        client.append_outcome_metrics("job-1", bad, dry_run=True)
    assert "DRY-RUN" not in capsys.readouterr().out
    client.close()


# ---------------------------------------------------------------------------
# emit_completion: what reaches the completions API
# ---------------------------------------------------------------------------
_REQUIRED_COMPLETION_FIELDS = {
    "completionStartTime": "2026-01-01T00:00:00Z",
    "costType": "AI",
    "inputTokenCount": 100,
    "isStreamed": False,
    "model": "gpt-5",
    "outputTokenCount": 50,
    "provider": "chatgpt",
    "requestDuration": 1000,
    "requestTime": "2026-01-01T00:00:00Z",
    "responseTime": "2026-01-01T00:00:01Z",
    "stopReason": "END",
    "totalTokenCount": 150,
    "transactionId": "tx-wire",
}

# create_completion sends the Name form and falls back to the deprecated Id
# form only when the Name is absent.
_DEPRECATED_ID_TO_NAME = {
    "organizationId": "organizationName",
    "productId": "productName",
}

_TRANSPORT_ONLY_PARAMETERS = {"self", "extra_headers", "extra_query", "extra_body", "timeout"}


def _completion_client_recording_requests():
    requests = []

    def handler(req: httpx.Request) -> httpx.Response:
        requests.append(req)
        return httpx.Response(201, json={})

    metering = ReveniumMetering(
        api_key="rev_sk_test_xxx",
        base_url="https://api.example.test",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        max_retries=0,
    )
    client = AgenticOutcomeClient(_settings(), metering_client=metering)
    return client, requests


def _sent_body(request: httpx.Request) -> dict:
    return json.loads(request.content)


def _completion_field_hints() -> dict:
    return get_type_hints(AICreateCompletionParams, include_extras=True)


def _sample_value(wire_name: str, hint):
    while get_origin(hint) in (Required, Annotated):
        hint = get_args(hint)[0]
    if get_origin(hint) is Literal:
        return get_args(hint)[0]
    if hint is bool:
        return True
    if hint is int:
        return 7
    if hint is float:
        return 1.5
    if hint is str:
        return f"{wire_name}-value"
    return {"id": "sub-1", "email": "user@example.test"}


def _payload_setting_every_completion_field() -> dict:
    payload = {}
    for python_name, hint in _completion_field_hints().items():
        wire_name = _maybe_transform_key(python_name, hint)
        payload[wire_name] = _sample_value(wire_name, hint)
    return payload


@pytest.mark.usefixtures("isolated_metering")
def test_emit_completion_sends_every_field_the_metering_client_accepts():
    accepted = set(inspect.signature(AIResource.create_completion).parameters)
    assert set(_completion_field_hints()) == accepted - _TRANSPORT_ONLY_PARAMETERS

    client, requests = _completion_client_recording_requests()
    payload = _payload_setting_every_completion_field()

    client.emit_completion(payload)
    _wait_for_delivery()
    expected = {
        k: v for k, v in payload.items()
        if k not in _DEPRECATED_ID_TO_NAME and k != "credentialAlias"
    }
    assert _sent_body(requests[0]) == expected

    without_names = {
        k: v for k, v in payload.items() if k not in _DEPRECATED_ID_TO_NAME.values()
    }
    client.emit_completion(without_names)
    _wait_for_delivery()
    body = _sent_body(requests[1])
    for id_field, name_field in _DEPRECATED_ID_TO_NAME.items():
        assert body[name_field] == payload[id_field]
    client.close()


@pytest.mark.usefixtures("isolated_metering")
def test_emit_completion_sends_token_counts_cost_and_subscriber():
    client, requests = _completion_client_recording_requests()
    usage = {
        "inputTokenCount": 1200,
        "outputTokenCount": 340,
        "totalTokenCount": 1540,
        "reasoningTokenCount": 90,
        "cacheReadTokenCount": 800,
        "cacheCreationTokenCount": 4000,
        "cacheCreation1hTokenCount": 0,
        "cacheCreation5mTokenCount": 4000,
        "totalCost": 0.0421,
        "inputTokenCost": 0.012,
        "outputTokenCost": 0.0204,
        "cacheReadTokenCost": 0.0008,
        "cacheCreationTokenCost": 0.0089,
        "subscriber": {"id": "sub-42", "email": "buyer@example.test"},
        "billingSkipped": False,
        "retryNumber": 0,
        "errorReason": "",
    }
    client.emit_completion({**_REQUIRED_COMPLETION_FIELDS, **usage})
    _wait_for_delivery()
    body = _sent_body(requests[0])
    assert {name: body.get(name, "<absent>") for name in usage} == usage
    client.close()


@pytest.mark.usefixtures("isolated_metering")
def test_emit_completion_omits_fields_the_caller_did_not_set():
    client, requests = _completion_client_recording_requests()
    client.emit_completion(dict(_REQUIRED_COMPLETION_FIELDS))
    _wait_for_delivery()
    assert _sent_body(requests[0]) == _REQUIRED_COMPLETION_FIELDS
    client.close()


@pytest.mark.usefixtures("isolated_metering")
def test_emit_completion_still_sends_a_required_key_set_to_none():
    client, requests = _completion_client_recording_requests()
    client.emit_completion({**_REQUIRED_COMPLETION_FIELDS, "stopReason": None})
    _wait_for_delivery()
    body = _sent_body(requests[0])
    assert "stopReason" in body and body["stopReason"] is None
    client.close()


@pytest.mark.usefixtures("isolated_metering")
def test_emit_completion_warns_about_a_key_the_api_does_not_know(caplog):
    client, requests = _completion_client_recording_requests()
    payload = {**_REQUIRED_COMPLETION_FIELDS, "bogusKey": "secret-prompt-text"}

    with caplog.at_level("WARNING", logger="revenium_middleware.agentic_outcomes"):
        client.emit_completion(payload)
        _wait_for_delivery()
        client.emit_completion(payload)
        _wait_for_delivery()

    warnings = [
        r for r in caplog.records
        if r.name == "revenium_middleware.agentic_outcomes" and r.levelname == "WARNING"
    ]
    assert len(warnings) == 1
    assert "bogusKey" in warnings[0].getMessage()
    assert "secret-prompt-text" not in warnings[0].getMessage()
    assert all("bogusKey" not in _sent_body(request) for request in requests)
    assert len(requests) == 2
    client.close()


@pytest.mark.usefixtures("isolated_metering")
def test_emit_completion_names_each_unknown_key_once_per_client(caplog):
    client, _ = _completion_client_recording_requests()

    with caplog.at_level("WARNING", logger="revenium_middleware.agentic_outcomes"):
        client.emit_completion({**_REQUIRED_COMPLETION_FIELDS, "bogusA": 1, "bogusB": 1})
        _wait_for_delivery()
        client.emit_completion({**_REQUIRED_COMPLETION_FIELDS, "bogusA": 1, "bogusC": 1})
        _wait_for_delivery()
        client.emit_completion({**_REQUIRED_COMPLETION_FIELDS, "bogusB": 1})
        _wait_for_delivery()

    messages = [
        r.getMessage() for r in caplog.records
        if r.name == "revenium_middleware.agentic_outcomes" and r.levelname == "WARNING"
    ]
    assert len(messages) == 2
    assert "bogusA" in messages[0] and "bogusB" in messages[0]
    assert "bogusC" in messages[1] and "bogusA" not in messages[1]
    client.close()


@pytest.mark.usefixtures("isolated_metering")
def test_emit_completion_sends_a_fresh_idempotency_key_per_call():
    client, requests = _completion_client_recording_requests()
    client.emit_completion(dict(_REQUIRED_COMPLETION_FIELDS))
    _wait_for_delivery()
    client.emit_completion(dict(_REQUIRED_COMPLETION_FIELDS))
    _wait_for_delivery()
    keys = [request.headers["Idempotency-Key"] for request in requests]
    assert [str(uuid.UUID(key)) for key in keys] == keys
    assert keys[0] != keys[1]
    client.close()


_PROMPT_TEXT_FIELDS = ("systemPrompt", "inputMessages", "outputResponse")


def _payload_with_credentials_in_prompt_text() -> dict:
    openai_key = "sk-proj-FAKE" + "0" * 48
    anthropic_key = "sk-ant-api03-FAKE" + "0" * 32
    bearer = "Bearer FAKEBEARERTOKENFAKEBEARERTOKEN0002"
    return {
        **_REQUIRED_COMPLETION_FIELDS,
        "systemPrompt": f"Call the tools with token={openai_key}",
        "inputMessages": json.dumps([
            {"role": "user", "content": f"my key is {openai_key} and password: FAKEpassw0rd!"},
        ]),
        "outputResponse": f"Use the header Authorization: {bearer} or the key {anthropic_key}",
    }


@pytest.mark.usefixtures("isolated_metering")
def test_emit_completion_masks_credentials_in_prompt_and_reply_text():
    client, requests = _completion_client_recording_requests()
    payload = _payload_with_credentials_in_prompt_text()

    client.emit_completion(payload)
    _wait_for_delivery()

    body = _sent_body(requests[0])
    for field in _PROMPT_TEXT_FIELDS:
        assert body[field] == sanitize_credentials(payload[field])
        assert "FAKE" not in body[field]
        assert "***REDACTED***" in body[field]
    json.loads(body["inputMessages"])
    client.close()


@pytest.mark.usefixtures("isolated_metering")
def test_emit_completion_sends_prompt_text_without_credentials_unchanged():
    client, requests = _completion_client_recording_requests()
    benign = {
        "systemPrompt": "The bearer of this token must meet password requirements; secret sauce",
        "inputMessages": json.dumps([{"role": "user", "content": "What is a good password policy?"}]),
        "outputResponse": "Use a token bucket; keep the secret recipe in a vault.",
    }

    client.emit_completion({**_REQUIRED_COMPLETION_FIELDS, **benign})
    _wait_for_delivery()

    body = _sent_body(requests[0])
    assert {field: body[field] for field in benign} == benign
    client.close()


@pytest.mark.usefixtures("isolated_metering")
def test_emit_completion_does_not_modify_the_callers_payload():
    client, _ = _completion_client_recording_requests()
    payload = _payload_with_credentials_in_prompt_text()
    snapshot = copy.deepcopy(payload)

    returned = client.emit_completion(payload)
    _wait_for_delivery()

    assert payload == snapshot
    assert returned is payload
    client.close()


@pytest.mark.usefixtures("isolated_metering")
def test_emit_completion_masks_with_prompt_capture_off(monkeypatch):
    monkeypatch.delenv("REVENIUM_CAPTURE_PROMPTS", raising=False)
    client, requests = _completion_client_recording_requests()

    client.emit_completion(_payload_with_credentials_in_prompt_text())
    _wait_for_delivery()

    body = _sent_body(requests[0])
    for field in _PROMPT_TEXT_FIELDS:
        assert "FAKE" not in body[field]
        assert "***REDACTED***" in body[field]
    client.close()


def test_report_outcome_raises_already_reported_on_a_second_report():
    calls = []

    def handler(req: httpx.Request) -> httpx.Response:
        calls.append(req)
        return httpx.Response(409, json=_ALREADY_REPORTED_BODY)

    http = httpx.Client(transport=httpx.MockTransport(handler))
    client = AgenticOutcomeClient(_settings(), http_client=http)
    with pytest.raises(OutcomeAlreadyReportedError) as exc_info:
        client.report_outcome("job-1", {"executionStatus": "SUCCESS"})
    assert exc_info.value.reported_at == "2026-10-08T16:03:15.556Z"
    assert exc_info.value.amendment_count == 0
    assert [call.method for call in calls] == ["POST"]
    client.close()


# ---------------------------------------------------------------------------
# Retries after a gateway error
# ---------------------------------------------------------------------------
def test_report_outcome_returns_when_a_502_hid_its_own_success():
    stored_job = {
        "agenticJobId": "job-1",
        "entityVersion": 1,
        "executionStatus": "SUCCESS",
        "hasOutcome": True,
        "outcomeCurrency": None,
        "outcomeMetadata": None,
        "outcomeReason": None,
        "outcomeType": "CONVERTED",
        "outcomeUpdateCount": 0,
        "outcomeValue": 12.5,
    }
    calls = []

    def handler(req: httpx.Request) -> httpx.Response:
        calls.append(req)
        if req.method == "GET":
            return httpx.Response(200, json=stored_job)
        if sum(1 for call in calls if call.method == "POST") == 1:
            return httpx.Response(502, json={})
        return httpx.Response(409, json=_ALREADY_REPORTED_BODY)

    http = httpx.Client(transport=httpx.MockTransport(handler))
    client = AgenticOutcomeClient(_settings(), http_client=http)
    client.report_outcome(
        "job-1",
        {"executionStatus": "SUCCESS", "outcomeType": "CONVERTED", "outcomeValue": 12.5},
    )
    assert [(call.method, call.url.path) for call in calls] == [
        ("POST", "/profitstream/v2/api/jobs/job-1/outcome"),
        ("POST", "/profitstream/v2/api/jobs/job-1/outcome"),
        ("GET", "/profitstream/v2/api/jobs/job-1"),
    ]
    client.close()


def test_a_replayed_tool_event_carries_the_same_idempotency_key(recording_meters, isolated_metering):
    requests = _tool_event_answered_then_replayed(recording_meters[0], isolated_metering, 502)
    first, second = (_header(request, "Idempotency-Key") for request in requests)
    assert first is not None and str(uuid.UUID(first)) == first
    assert second == first


@pytest.mark.usefixtures("isolated_metering")
def test_each_tool_event_call_gets_its_own_idempotency_key(monkeypatch):
    payload = {"transactionId": "tool-tx-1", "toolId": "t1"}
    calls = _tool_events_sent(monkeypatch, _settings(), dict(payload), dict(payload))
    first, second = (call["headers"].get("Idempotency-Key") for call in calls)
    assert None not in (first, second)
    assert first != second


# ---------------------------------------------------------------------------
# Which Revenium address the client uses
# ---------------------------------------------------------------------------
def _no_address_is_set_in_code(monkeypatch):
    monkeypatch.delenv("REVENIUM_METERING_API_KEY", raising=False)
    monkeypatch.setattr(metering, "client", None)


def _only_the_usage_address_is_set(monkeypatch, usage_address):
    _no_address_is_set_in_code(monkeypatch)
    monkeypatch.delenv("REVENIUM_PROFITSTREAM_BASE_URL", raising=False)
    monkeypatch.setenv("REVENIUM_METERING_BASE_URL", usage_address)


def _outcome_url_sent_by(settings: AgenticOutcomeSettings) -> str:
    urls = []

    def handler(req: httpx.Request) -> httpx.Response:
        urls.append(str(req.url))
        return httpx.Response(200, json={})

    http = httpx.Client(transport=httpx.MockTransport(handler))
    client = AgenticOutcomeClient(settings, http_client=http)
    client.report_outcome("job-1", {"executionStatus": "SUCCESS"})
    client.close()
    return urls[0]


def test_settings_use_the_usage_address_for_usage_and_outcomes(monkeypatch):
    _only_the_usage_address_is_set(monkeypatch, "https://usage.example.test/meter")

    settings = AgenticOutcomeSettings(api_key="rev_sk_test_xxx", team_id="TEAM")

    assert settings.completion_url == "https://usage.example.test/meter/v2/ai/completions"
    assert settings.tool_url == "https://usage.example.test/meter/v2/tool/events"
    assert settings.sdk_meter_base_url == "https://usage.example.test/meter/"
    assert _outcome_url_sent_by(settings) == (
        "https://usage.example.test/profitstream/v2/api/jobs/job-1/outcome?teamId=TEAM"
    )


def test_explicit_meter_base_url_carries_outcomes_with_it(monkeypatch):
    _only_the_usage_address_is_set(monkeypatch, "https://usage.example.test")

    settings = AgenticOutcomeSettings(
        api_key="rev_sk_test_xxx", meter_base_url="https://self-hosted.example.test/meter", team_id="TEAM",
    )

    assert _outcome_url_sent_by(settings) == (
        "https://self-hosted.example.test/profitstream/v2/api/jobs/job-1/outcome?teamId=TEAM"
    )


def test_separate_outcome_address_still_wins(monkeypatch):
    monkeypatch.setenv("REVENIUM_METERING_BASE_URL", "https://usage.example.test")
    monkeypatch.setenv("REVENIUM_PROFITSTREAM_BASE_URL", "https://outcomes.example.test/ctx")

    from_env = AgenticOutcomeSettings(api_key="rev_sk_test_xxx", team_id="TEAM")
    explicit = AgenticOutcomeSettings(
        api_key="rev_sk_test_xxx", profitstream_base_url="https://explicit.example.test", team_id="TEAM",
    )

    assert from_env.completion_url == "https://usage.example.test/meter/v2/ai/completions"
    assert _outcome_url_sent_by(from_env) == (
        "https://outcomes.example.test/ctx/profitstream/v2/api/jobs/job-1/outcome?teamId=TEAM"
    )
    assert _outcome_url_sent_by(explicit) == (
        "https://explicit.example.test/profitstream/v2/api/jobs/job-1/outcome?teamId=TEAM"
    )


def test_default_address_matches_the_usage_default(monkeypatch):
    _no_address_is_set_in_code(monkeypatch)
    monkeypatch.delenv("REVENIUM_PROFITSTREAM_BASE_URL", raising=False)
    monkeypatch.delenv("REVENIUM_METERING_BASE_URL", raising=False)

    settings = AgenticOutcomeSettings(api_key="rev_sk_test_xxx", team_id="TEAM")

    assert settings.completion_url == "https://api.revenium.ai/meter/v2/ai/completions"
    assert _outcome_url_sent_by(settings) == (
        "https://api.revenium.ai/profitstream/v2/api/jobs/job-1/outcome?teamId=TEAM"
    )


def test_usage_address_path_prefix_is_kept_for_outcomes(monkeypatch):
    _only_the_usage_address_is_set(monkeypatch, "https://proxy.corp.test/revenium/meter")

    settings = AgenticOutcomeSettings(api_key="rev_sk_test_xxx", team_id="TEAM")

    assert settings.completion_url == "https://proxy.corp.test/revenium/meter/v2/ai/completions"
    assert _outcome_url_sent_by(settings) == (
        "https://proxy.corp.test/revenium/profitstream/v2/api/jobs/job-1/outcome?teamId=TEAM"
    )


def test_settings_trim_padded_keys():
    settings = _settings(api_key="rev_sk_test_xxx\n", outcome_api_key=" rev_sk_test_outcome ")

    assert (settings.api_key, settings.outcome_api_key) == ("rev_sk_test_xxx", "rev_sk_test_outcome")


# ---------------------------------------------------------------------------
# BACK-4451: hand-sent records are delivered like automatic ones
# ---------------------------------------------------------------------------
WRITE_KEY = "rev_sk_fake_emit_delivery_write_key"
PROCESS_KEY = "hak_fake_emit_delivery_process_key"
SCOPED_KEY = "hak_fake_emit_delivery_scoped_key"
SLOW_ENDPOINT_SECONDS = 1.2
QUEUE_BOUND_SECONDS = 0.2
COMPLETIONS_PATH = "/meter/v2/ai/completions"
TOOL_EVENTS_PATH = "/meter/v2/tool/events"
_TOOL_PAYLOAD = {"transactionId": "tool-tx-delivery", "toolId": "t1"}


def _stub_settings(meter, **overrides) -> AgenticOutcomeSettings:
    base = dict(
        api_key=WRITE_KEY,
        meter_base_url=meter.base_url,
        team_id="TEAM",
        outcome_retry_attempts=3,
        outcome_retry_initial_seconds=0.0,
    )
    base.update(overrides)
    return AgenticOutcomeSettings(**base)


def _payload_for(emit: str) -> dict:
    return dict(_REQUIRED_COMPLETION_FIELDS) if emit == "emit_completion" else dict(_TOOL_PAYLOAD)


def _path_for(emit: str) -> str:
    return COMPLETIONS_PATH if emit == "emit_completion" else TOOL_EVENTS_PATH


def _header(request: dict, name: str):
    return {key.lower(): value for key, value in request["headers"].items()}.get(name.lower())


def _wait_for_delivery():
    assert metering_pool.wait_until_idle(10)


def _sdk_threads() -> int:
    """Live threads other than the stub server's per-connection handlers."""
    return sum(1 for thread in threading.enumerate() if "process_request_thread" not in thread.name)


def _spy_on_queueing(monkeypatch) -> list:
    from revenium_middleware import agentic_outcomes

    queued = []
    real = metering.run_async_in_thread

    def spy(*args, **kwargs):
        queued.append(args)
        return real(*args, **kwargs)

    monkeypatch.setattr(agentic_outcomes, "run_async_in_thread", spy, raising=False)
    return queued


def _closed_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


@pytest.fixture()
def process_client(monkeypatch):
    """Lets a test call initialize_metering(); the process client and its re-exports are restored after."""
    import revenium_middleware
    import revenium_middleware._core as core

    monkeypatch.setattr(metering, "client", metering.client)
    for module in (revenium_middleware, core):
        if hasattr(module, "client"):
            monkeypatch.setattr(module, "client", module.client)
    built = []

    def initialize(base_url):
        metering.initialize_metering(api_key=PROCESS_KEY, base_url=base_url)
        built.append(metering.client)

    yield initialize
    for client in built:
        client.close()


class _ToolEventClientAnswering:
    """Stands in for the shared tool-event client; records each post and answers with ``status``."""

    def __init__(self, status):
        self.status = status
        self.calls = []

    def post(self, url, headers=None, json=None, timeout=None):
        self.calls.append({"url": url, "headers": headers, "json": json, "timeout": timeout})
        return httpx.Response(self.status, request=httpx.Request("POST", url))


def _time_three_calls_after_a_warm_up(emit, payload) -> tuple:
    emit(dict(payload))
    threads_after_warm_up = _sdk_threads()
    elapsed = []
    for _ in range(3):
        started = time.perf_counter()
        emit(dict(payload))
        elapsed.append(time.perf_counter() - started)
    return max(elapsed), threads_after_warm_up, _sdk_threads()


def _assert_returns_before_a_slow_endpoint_answers(emit, recording_meters, monkeypatch):
    a, _ = recording_meters
    a.delay = SLOW_ENDPOINT_SECONDS
    queued = _spy_on_queueing(monkeypatch)
    with AgenticOutcomeClient(_stub_settings(a)) as client:
        slowest, threads_before, threads_after = _time_three_calls_after_a_warm_up(
            getattr(client, emit), _payload_for(emit)
        )
        _wait_for_delivery()
    print(f"{emit} caller seconds against a {SLOW_ENDPOINT_SECONDS}s endpoint: max {slowest:.4f}")
    assert slowest < QUEUE_BOUND_SECONDS
    assert len(queued) == 4
    assert threads_after == threads_before
    assert [request["path"] for request in a.requests] == [_path_for(emit)] * 4
    assert all(_header(request, "x-api-key") == WRITE_KEY for request in a.requests)


@pytest.mark.usefixtures("isolated_metering")
def test_emit_completion_returns_before_a_slow_endpoint_answers(recording_meters, monkeypatch):
    _assert_returns_before_a_slow_endpoint_answers("emit_completion", recording_meters, monkeypatch)


@pytest.mark.usefixtures("isolated_metering")
def test_emit_tool_event_returns_before_a_slow_endpoint_answers(recording_meters, monkeypatch):
    _assert_returns_before_a_slow_endpoint_answers("emit_tool_event", recording_meters, monkeypatch)


def test_emit_completion_uses_the_configured_timeouts_and_retries(recording_meters, monkeypatch, isolated_metering):
    a, _ = recording_meters
    monkeypatch.setenv(metering.TIMEOUT_ENV, "0.4")
    monkeypatch.setenv(metering.CONNECT_TIMEOUT_ENV, "0.3")
    monkeypatch.setenv(metering.MAX_RETRIES_ENV, "1")
    a.answer_with(503)
    with AgenticOutcomeClient(_stub_settings(a)) as client:
        sender = client._metering()
        assert sender.timeout == httpx.Timeout(0.4, connect=0.3)
        assert sender.max_retries == 1
        client.emit_completion(dict(_REQUIRED_COMPLETION_FIELDS))
        _wait_for_delivery()
    assert len(a.requests) == 2
    assert isolated_metering.stats()["size"] == 1


def test_emit_completion_times_out_at_the_configured_timeout(recording_meters, monkeypatch, isolated_metering):
    a, _ = recording_meters
    monkeypatch.setenv(metering.TIMEOUT_ENV, "0.4")
    a.delay = SLOW_ENDPOINT_SECONDS
    with AgenticOutcomeClient(_stub_settings(a)) as client:
        client.emit_completion(dict(_REQUIRED_COMPLETION_FIELDS))
        _wait_for_delivery()
    assert len(a.requests) == 1
    assert isolated_metering.stats()["size"] == 1


def test_emit_tool_event_uses_the_metering_timeouts_without_retries(monkeypatch, isolated_metering):
    monkeypatch.setenv(metering.TIMEOUT_ENV, "0.4")
    monkeypatch.setenv(metering.CONNECT_TIMEOUT_ENV, "0.3")
    stub = _ToolEventClientAnswering(503)
    monkeypatch.setattr(tool_event_transport, "_client", stub)
    jobs_only = httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200, json={})))
    settings = _settings(api_key=WRITE_KEY, meter_base_url="http://metering.test")
    with AgenticOutcomeClient(settings, http_client=jobs_only) as client:
        client.emit_tool_event(dict(_TOOL_PAYLOAD))
        _wait_for_delivery()
    jobs_only.close()
    assert len(stub.calls) == 1
    assert stub.calls[0]["timeout"] == httpx.Timeout(0.4, connect=0.3)
    assert isolated_metering.stats()["size"] == 1


def _emit_inside(emit, settings, block):
    with AgenticOutcomeClient(settings) as client:
        with block:
            getattr(client, emit)(_payload_for(emit))
        _wait_for_delivery()


def _assert_sent_to_a_only(emit, a, b):
    assert [request["path"] for request in a.requests] == [_path_for(emit)]
    assert _header(a.requests[0], "x-api-key") == WRITE_KEY
    assert b.requests == []


def _assert_goes_to_the_address_set_by_initialize_metering(emit, recording_meters, monkeypatch, process_client):
    a, b = recording_meters
    monkeypatch.setenv("REVENIUM_METERING_BASE_URL", b.meter_url)
    process_client(a.meter_url)
    _emit_inside(emit, AgenticOutcomeSettings(api_key=WRITE_KEY), contextlib.nullcontext())
    _assert_sent_to_a_only(emit, a, b)


def _assert_goes_to_the_address_of_a_metering_key_block(emit, recording_meters, monkeypatch):
    a, b = recording_meters
    monkeypatch.setenv("REVENIUM_METERING_BASE_URL", b.meter_url)
    settings = AgenticOutcomeSettings(api_key=WRITE_KEY)
    _emit_inside(emit, settings, metering.metering_key(SCOPED_KEY, base_url=a.meter_url))
    _assert_sent_to_a_only(emit, a, b)


@pytest.mark.usefixtures("isolated_metering")
def test_emit_completion_goes_to_the_address_set_by_initialize_metering(recording_meters, monkeypatch, process_client):
    _assert_goes_to_the_address_set_by_initialize_metering(
        "emit_completion", recording_meters, monkeypatch, process_client
    )


@pytest.mark.usefixtures("isolated_metering")
def test_emit_completion_goes_to_the_address_of_a_metering_key_block(recording_meters, monkeypatch):
    _assert_goes_to_the_address_of_a_metering_key_block("emit_completion", recording_meters, monkeypatch)


@pytest.mark.usefixtures("isolated_metering")
def test_emit_tool_event_goes_to_the_address_set_by_initialize_metering(recording_meters, monkeypatch, process_client):
    _assert_goes_to_the_address_set_by_initialize_metering(
        "emit_tool_event", recording_meters, monkeypatch, process_client
    )


@pytest.mark.usefixtures("isolated_metering")
def test_emit_tool_event_goes_to_the_address_of_a_metering_key_block(recording_meters, monkeypatch):
    _assert_goes_to_the_address_of_a_metering_key_block("emit_tool_event", recording_meters, monkeypatch)


@pytest.mark.usefixtures("isolated_metering")
def test_an_explicit_meter_base_url_still_wins_over_initialize_metering(recording_meters, process_client):
    """Guard: passes before and after the fix."""
    a, b = recording_meters
    process_client(b.meter_url)
    _emit_inside("emit_completion", _stub_settings(a), contextlib.nullcontext())
    _assert_sent_to_a_only("emit_completion", a, b)


@pytest.mark.parametrize("cause", ["503", "timeout"])
@pytest.mark.parametrize("emit", ["emit_completion", "emit_tool_event"])
def test_a_retryable_failure_is_buffered_and_delivered_on_recovery(
    emit, cause, recording_meters, monkeypatch, isolated_metering
):
    a, _ = recording_meters
    if cause == "503":
        a.answer_with(503)
    else:
        monkeypatch.setenv(metering.TIMEOUT_ENV, "0.4")
        a.delay = SLOW_ENDPOINT_SECONDS
    with AgenticOutcomeClient(_stub_settings(a)) as client:
        getattr(client, emit)(_payload_for(emit))
        _wait_for_delivery()
        assert isolated_metering.stats()["size"] == 1

        a.answer_with(201)
        a.delay = 0.0
        isolated_metering.flush(deadline_seconds=5)

    assert isolated_metering.stats()["size"] == 0
    first, last = a.requests[0], a.requests[-1]
    assert len(a.requests) == 2
    assert last["path"] == _path_for(emit)
    assert _header(last, "Idempotency-Key") == _header(first, "Idempotency-Key")
    assert _header(last, "x-api-key") == WRITE_KEY


@pytest.mark.parametrize("emit", ["emit_completion", "emit_tool_event"])
def test_connection_refused_is_buffered(emit, isolated_metering):
    settings = _settings(api_key=WRITE_KEY, meter_base_url=f"http://127.0.0.1:{_closed_port()}")
    with AgenticOutcomeClient(settings) as client:
        getattr(client, emit)(_payload_for(emit))
        _wait_for_delivery()
    assert isolated_metering.stats()["size"] == 1


@pytest.mark.usefixtures("isolated_metering")
def test_jobs_and_outcomes_follow_the_address_set_by_initialize_metering(recording_meters, monkeypatch, process_client):
    a, b = recording_meters
    monkeypatch.delenv("REVENIUM_PROFITSTREAM_BASE_URL", raising=False)
    monkeypatch.setenv("REVENIUM_METERING_BASE_URL", b.meter_url)
    process_client(a.meter_url)
    settings = AgenticOutcomeSettings(
        api_key=WRITE_KEY, team_id="TEAM", outcome_retry_attempts=3, outcome_retry_initial_seconds=0.0,
    )
    with AgenticOutcomeClient(settings) as client:
        client.emit_completion(dict(_REQUIRED_COMPLETION_FIELDS))
        _wait_for_delivery()
        client.create_job("job-1")
        client.report_outcome("job-1", {"executionStatus": "SUCCESS"})

    assert [request["path"] for request in a.requests] == [
        COMPLETIONS_PATH,
        "/profitstream/v2/api/jobs?teamId=TEAM",
        "/profitstream/v2/api/jobs/job-1/outcome?teamId=TEAM",
    ]
    assert b.requests == []


@pytest.mark.usefixtures("isolated_metering")
def test_auto_resolved_team_is_looked_up_again_at_a_new_address(recording_meters, monkeypatch, process_client):
    a, b = recording_meters
    monkeypatch.delenv("REVENIUM_PROFITSTREAM_BASE_URL", raising=False)
    monkeypatch.setenv("REVENIUM_METERING_BASE_URL", b.meter_url)
    process_client(a.meter_url)
    team_by_port = {a.port: "TEAM-A", b.port: "TEAM-B"}
    outcome_paths = []

    def handler(request: httpx.Request) -> httpx.Response:
        team = team_by_port[request.url.port]
        if request.url.path.endswith("/teams"):
            return httpx.Response(200, json={"_embedded": {"teamResourceList": [{"id": team}]}})
        outcome_paths.append((request.url.port, request.url.query.decode()))
        return httpx.Response(201, json={})

    settings = AgenticOutcomeSettings(
        api_key=WRITE_KEY, outcome_retry_attempts=3, outcome_retry_initial_seconds=0.0,
    )
    http = httpx.Client(transport=httpx.MockTransport(handler))
    with AgenticOutcomeClient(settings, http_client=http) as client:
        client.report_outcome("job-1", {"executionStatus": "SUCCESS"})
        with metering.metering_key(SCOPED_KEY, base_url=b.meter_url):
            client.report_outcome("job-1", {"executionStatus": "SUCCESS"})

    assert outcome_paths == [(a.port, "teamId=TEAM-A"), (b.port, "teamId=TEAM-B")]


@pytest.mark.usefixtures("isolated_metering")
def test_an_explicit_profitstream_base_url_still_wins(recording_meters, monkeypatch, process_client):
    """Guard: passes before and after the fix."""
    a, b = recording_meters
    monkeypatch.delenv("REVENIUM_PROFITSTREAM_BASE_URL", raising=False)
    monkeypatch.setenv("REVENIUM_METERING_BASE_URL", a.meter_url)
    process_client(a.meter_url)
    settings = AgenticOutcomeSettings(
        api_key=WRITE_KEY, profitstream_base_url=b.base_url, team_id="TEAM",
        outcome_retry_attempts=3, outcome_retry_initial_seconds=0.0,
    )
    with AgenticOutcomeClient(settings) as client:
        client.report_outcome("job-1", {"executionStatus": "SUCCESS"})
    assert a.requests == []
    assert [request["path"] for request in b.requests] == ["/profitstream/v2/api/jobs/job-1/outcome?teamId=TEAM"]


def test_an_in_progress_conflict_with_retry_after_is_buffered_and_replayed(recording_meters, isolated_metering):
    a, _ = recording_meters
    a.answer_with(409, {"Retry-After": "1"})
    with AgenticOutcomeClient(_stub_settings(a)) as client:
        client.emit_tool_event(dict(_TOOL_PAYLOAD))
        _wait_for_delivery()
        assert isolated_metering.stats()["size"] == 1

        a.answer_with(201)
        isolated_metering.flush(deadline_seconds=5)

    assert isolated_metering.stats()["size"] == 0
    assert len(a.requests) == 2
    assert _header(a.requests[1], "Idempotency-Key") == _header(a.requests[0], "Idempotency-Key")


def test_a_bare_conflict_is_dropped_not_buffered(recording_meters, isolated_metering):
    a, _ = recording_meters
    a.answer_with(409)
    with AgenticOutcomeClient(_stub_settings(a)) as client:
        client.emit_tool_event(dict(_TOOL_PAYLOAD))
        _wait_for_delivery()
    assert len(a.requests) == 1
    assert isolated_metering.stats()["size"] == 0


def test_emit_completion_names_agent_name_and_currency_and_does_not_send_them(caplog):
    captured = {}

    class FakeAI:
        def create_completion(self, **kwargs):
            captured["kwargs"] = kwargs

    class FakeMetering:
        def __init__(self):
            self.ai = FakeAI()

    client = AgenticOutcomeClient(_settings(), metering_client=FakeMetering())
    payload = {
        **_REQUIRED_COMPLETION_FIELDS,
        "agent": "support-bot",
        "agentName": "support-bot-name",
        "currency": "EUR",
    }

    with caplog.at_level("WARNING", logger="revenium_middleware.agentic_outcomes"):
        client.emit_completion(payload)
        _wait_for_delivery()

    kwargs = captured["kwargs"]
    assert kwargs["agent"] == "support-bot"
    extra_body = kwargs.get("extra_body") or {}
    assert "agentName" not in extra_body
    assert "currency" not in extra_body
    messages = [
        r.getMessage() for r in caplog.records
        if r.name == "revenium_middleware.agentic_outcomes" and r.levelname == "WARNING"
    ]
    assert len(messages) == 1
    assert "agentName" in messages[0] and "currency" in messages[0]
    client.close()
