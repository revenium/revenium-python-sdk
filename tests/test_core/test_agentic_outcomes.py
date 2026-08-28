"""Unit tests for revenium_middleware.agentic_outcomes."""
from __future__ import annotations

import dataclasses
import json

import httpx
import pytest

from revenium_middleware.agentic_outcomes import AgenticOutcomeClient, AgenticOutcomeSettings


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


def test_emit_tool_event_uses_single_api_key():
    seen = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["url"] = str(req.url)
        seen["x_api_key"] = req.headers.get("x-api-key")
        return httpx.Response(200, json={})

    transport = httpx.MockTransport(handler)
    http = httpx.Client(transport=transport)
    client = AgenticOutcomeClient(_settings(outcome_api_key="ignored"), http_client=http)
    client.emit_tool_event({"transactionId": "tx-1"})
    assert seen["x_api_key"] == "rev_sk_test_xxx"  # api_key, NOT outcome_api_key
    assert seen["url"].endswith("/meter/v2/tool/events")
    client.close()


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


def test_emit_tool_event_retries_on_429():
    calls = []

    def handler(req: httpx.Request) -> httpx.Response:
        calls.append(req)
        if len(calls) == 1:
            return httpx.Response(429, json={})
        return httpx.Response(200, json={})

    transport = httpx.MockTransport(handler)
    http = httpx.Client(transport=transport)
    settings = _settings(outcome_retry_initial_seconds=0.0)
    client = AgenticOutcomeClient(settings, http_client=http)
    # Closes the MEDIUM finding: tool events used to have NO retry path; failed silently on 429.
    client.emit_tool_event({"transactionId": "tx-1"})
    assert len(calls) == 2
    client.close()


def test_top_level_imports_available():
    import revenium_middleware as rm

    assert rm.AgenticOutcomeClient is AgenticOutcomeClient
    assert rm.AgenticOutcomeSettings is AgenticOutcomeSettings
    assert "AgenticOutcomeClient" in rm.__all__
    assert "AgenticOutcomeSettings" in rm.__all__


# ---------------------------------------------------------------------------
# Fix #10: KeyError → ValueError for missing required keys
# ---------------------------------------------------------------------------
def test_emit_completion_raises_value_error_on_missing_keys():
    client = AgenticOutcomeClient(_settings())
    with pytest.raises(ValueError, match="Missing required keys"):
        client.emit_completion({"model": "gpt-5"})
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


def test_tool_event_still_uses_api_key_not_outcome_key():
    seen = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["x_api_key"] = req.headers.get("x-api-key")
        return httpx.Response(200, json={})

    transport = httpx.MockTransport(handler)
    http = httpx.Client(transport=transport)
    settings = _settings(api_key="rev_mk_metering", outcome_api_key="rev_sk_write")
    client = AgenticOutcomeClient(settings, http_client=http)
    client.emit_tool_event({"transactionId": "tx-1"})
    assert seen["x_api_key"] == "rev_mk_metering"
    client.close()


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
        "currency": "USD",
    }
    client.emit_completion(payload)
    assert captured["kwargs"]["model"] == "gpt-5"
    assert captured["kwargs"]["transaction_id"] == "tx-1"
    # extra_body carries agentic + currency fields
    assert captured["kwargs"]["extra_body"]["agenticJobId"] == "job-1"
    assert captured["kwargs"]["extra_body"]["currency"] == "USD"
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


def test_metering_paths_still_accept_a_metering_key():
    # emit_tool_event authenticates with settings.api_key, not the outcome key —
    # a metering key must keep working there.
    seen = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["x_api_key"] = req.headers.get("x-api-key")
        return httpx.Response(200, json={})

    transport = httpx.MockTransport(handler)
    http = httpx.Client(transport=transport)
    settings = _settings(api_key="rev_mk_TENANT_abc", outcome_api_key=None)
    client = AgenticOutcomeClient(settings, http_client=http)
    client.emit_tool_event({"transactionId": "tx-mk"})
    assert seen["x_api_key"] == "rev_mk_TENANT_abc"
    client.close()
