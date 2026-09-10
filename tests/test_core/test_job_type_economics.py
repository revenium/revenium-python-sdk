"""Management-plane job economics requests use the fixed v2 REST contract."""
import json
import os
import warnings
from pathlib import Path
from unittest.mock import patch

import httpx
import pytest

from revenium_middleware import (
    Baseline, JobTypeEconomics, PeriodFactEntry,
    create_baseline, get_job_type_economics, list_baselines,
    report_period_facts, upsert_job_type_economics,
)
from revenium_middleware._core.config import Config
import revenium_middleware.job_type_economics as economics_module

BASE = "https://api.revenium.example"
WRITE_KEY = "rev_sk_TENANT_write"
LEGACY_KEY = "rev_sk_TENANT_legacy"
EXPLICIT_KEY = "rev_sk_TENANT_explicit"
METERING_KEY = "rev_mk_TENANT_metering"
ENV = {Config.ENV_REVENIUM_WRITE_API_KEY: WRITE_KEY, "REVENIUM_TEAM_ID": "team-1"}
CONTRACT = json.loads((Path(__file__).parents[1] / "fixtures" / "job_type_economics_contract.json").read_text())


def _client(reply=None, expected_body=...):
    seen = {}
    def handler(request):
        body = json.loads(request.content or b"null")
        if expected_body is not ...:
            assert body == expected_body
        seen.update(method=request.method, path=request.url.raw_path.decode().split("?", 1)[0], team=request.url.params.get("teamId"), headers=dict(request.headers), body=body)
        return httpx.Response(200, json=[] if reply is None else reply)
    return httpx.Client(transport=httpx.MockTransport(handler)), seen


def _contract_call(function, *args, expected_body, **kwargs):
    client, seen = _client(kwargs.pop("reply", {"ok": True}), expected_body=expected_body)
    with patch.dict(os.environ, ENV):
        function(*args, profitstream_base_url=BASE, http_client=client, **kwargs)
    assert seen["team"] == "team-1"
    assert seen["headers"]["x-api-key"] == ENV[Config.ENV_REVENIUM_WRITE_API_KEY]
    return seen


def test_all_job_economics_requests_match_contract():
    payloads = CONTRACT["canonicalPayloads"]
    fact = PeriodFactEntry(
        "2026-08-01T00:00:00Z", "2026-08-02T00:00:00Z", "region", "us", "completed_claims", 4,
        "MEASURED", "claims-worker", "workflow",
        "restating August after the warehouse reload",
    )
    economics = JobTypeEconomics(
        "completed_claims", "claim",
        [{"key": "completed_claims", "type": "COUNT", "direction": "HIGHER_IS_BETTER", "aggregation": "SUM", "resolution": "PER_JOB"}],
        [{"key": "region", "allowedValues": ["us", "ca"]}],
        monetization={"metricKey": "completed_claims", "valuePerUnit": 4.25, "currency": "USD", "category": "COST_AVOIDED", "basis": "REALIZED"},
    )
    baseline = Baseline(cost_per_unit=4.5, currency="USD", provenance="CUSTOMER_DECLARED", declared_by="claims-operations", effective_from="2026-08-01T00:00:00Z")
    cases = [
        (report_period_facts, ("claims", [fact]), "POST", "/profitstream/v2/api/jobs/types/claims/facts", [payloads["periodFact"]]),
        (get_job_type_economics, ("claims",), "GET", "/profitstream/v2/api/jobs/types/claims/economics", None),
        (upsert_job_type_economics, ("claims", economics), "PUT", "/profitstream/v2/api/jobs/types/claims/economics", payloads["economics"]),
        (create_baseline, ("claims", baseline), "POST", "/profitstream/v2/api/jobs/types/claims/baselines", payloads["baseline"]),
        (list_baselines, ("claims",), "GET", "/profitstream/v2/api/jobs/types/claims/baselines", None),
    ]
    for function, args, method, path, body in cases:
        seen = _contract_call(function, *args, expected_body=body)
        assert seen["method"] == method and seen["path"] == path and seen["body"] == body


def test_attribution_fields_are_omitted_when_not_supplied():
    fact = PeriodFactEntry(
        "2026-08-01T00:00:00Z", "2026-08-02T00:00:00Z", "region", "us", "completed_claims", 4,
    )
    baseline = Baseline(effective_from="2026-08-01T00:00:00Z", cost_per_unit=4.5, currency="USD")

    assert fact.to_wire() == {
        "periodStart": "2026-08-01T00:00:00Z",
        "periodEnd": "2026-08-02T00:00:00Z",
        "dimensionKey": "region",
        "dimensionValue": "us",
        "key": "completed_claims",
        "value": 4,
    }
    assert baseline.to_wire() == {
        "effectiveFrom": "2026-08-01T00:00:00Z", "costPerUnit": 4.5, "currency": "USD",
    }


def test_explicit_attribution_fields_are_sent_as_overrides():
    baseline = Baseline(
        effective_from="2026-08-01T00:00:00Z",
        cost_per_unit=4.5,
        currency="USD",
        provenance="CUSTOMER_DECLARED",
        declared_by="claims-operations",
        evidence_url="https://example.com/evidence",
    )

    assert baseline.to_wire()["provenance"] == "CUSTOMER_DECLARED"
    assert baseline.to_wire()["declaredBy"] == "claims-operations"
    assert baseline.to_wire()["evidenceUrl"] == "https://example.com/evidence"


def test_contract_fixture_matches_backend_request_enum_sets():
    assert CONTRACT["enums"] == {
        "baselineProvenance": ["CUSTOMER_DECLARED", "MEASURED", "SIGNED_OFF"],
        "outcomeMetricProvenance": ["MEASURED", "SELF_REPORTED", "DERIVED", "ATTESTED"],
        "metricDirection": ["HIGHER_IS_BETTER", "LOWER_IS_BETTER"],
        "monetizationCategory": ["REVENUE", "COST_AVOIDED", "TIME_SAVED", "LEADING_VALUE"],
        "monetizationBasis": ["REALIZED", "EXPECTED"],
    }


def test_management_calls_reject_metering_key_before_request():
    client, seen = _client()
    with patch.dict(os.environ, {Config.ENV_REVENIUM_WRITE_API_KEY: "rev_mk_TENANT_abc", "REVENIUM_TEAM_ID": "team-1"}):
        with pytest.raises(ValueError, match="write-scope"):
            get_job_type_economics("claims", profitstream_base_url=BASE, http_client=client)
    assert not seen


def test_write_env_used_without_deprecation_warning():
    client, seen = _client(reply={"ok": True})
    with patch.dict(os.environ, ENV, clear=True):
        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always")
            get_job_type_economics("claims", profitstream_base_url=BASE, http_client=client)

    assert seen["headers"]["x-api-key"] == WRITE_KEY
    assert [
        warning for warning in recorded if issubclass(warning.category, DeprecationWarning)
    ] == []


def test_outcome_env_still_works_but_warns():
    client, seen = _client(reply={"ok": True})
    env = {
        Config.ENV_REVENIUM_OUTCOME_API_KEY: LEGACY_KEY,
        "REVENIUM_TEAM_ID": "team-1",
    }

    with patch.dict(os.environ, env, clear=True):
        with pytest.warns(DeprecationWarning, match="REVENIUM_WRITE_API_KEY"):
            get_job_type_economics("claims", profitstream_base_url=BASE, http_client=client)

    assert seen["headers"]["x-api-key"] == LEGACY_KEY


def test_write_env_wins_over_outcome_env_without_deprecation_warning():
    client, seen = _client(reply={"ok": True})
    env = {
        Config.ENV_REVENIUM_WRITE_API_KEY: WRITE_KEY,
        Config.ENV_REVENIUM_OUTCOME_API_KEY: LEGACY_KEY,
        "REVENIUM_TEAM_ID": "team-1",
    }

    with patch.dict(os.environ, env, clear=True):
        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always")
            get_job_type_economics("claims", profitstream_base_url=BASE, http_client=client)

    assert seen["headers"]["x-api-key"] == WRITE_KEY
    assert [
        warning for warning in recorded if issubclass(warning.category, DeprecationWarning)
    ] == []


def test_explicit_api_key_wins_over_env_vars_without_deprecation_warning():
    client, seen = _client(reply={"ok": True})
    env = {
        Config.ENV_REVENIUM_WRITE_API_KEY: WRITE_KEY,
        Config.ENV_REVENIUM_OUTCOME_API_KEY: LEGACY_KEY,
        "REVENIUM_TEAM_ID": "team-1",
    }

    with patch.dict(os.environ, env, clear=True):
        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always")
            get_job_type_economics(
                "claims",
                api_key=EXPLICIT_KEY,
                profitstream_base_url=BASE,
                http_client=client,
            )

    assert seen["headers"]["x-api-key"] == EXPLICIT_KEY
    assert [
        warning for warning in recorded if issubclass(warning.category, DeprecationWarning)
    ] == []


def test_metering_env_fallback_still_fails_fast_without_deprecation_warning():
    client, seen = _client(reply={"ok": True})
    env = {Config.ENV_REVENIUM_API_KEY: METERING_KEY, "REVENIUM_TEAM_ID": "team-1"}

    with patch.dict(os.environ, env, clear=True):
        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always")
            with pytest.raises(ValueError, match="write-scope"):
                get_job_type_economics("claims", profitstream_base_url=BASE, http_client=client)

    assert not seen
    assert [
        warning for warning in recorded if issubclass(warning.category, DeprecationWarning)
    ] == []


def test_post_helpers_do_not_retry_on_502():
    call_count = {"n": 0}

    def handler(request):
        call_count["n"] += 1
        return httpx.Response(502, json={"error": "bad gateway"})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    fact = PeriodFactEntry("2026-08-01T00:00:00Z", "2026-08-02T00:00:00Z", "region", "us", "completed", 1)
    with patch.dict(os.environ, ENV):
        with pytest.raises(httpx.HTTPStatusError):
            report_period_facts("claims", [fact], profitstream_base_url=BASE, http_client=client)
    assert call_count["n"] == 1


def test_a_baseline_cannot_be_built_without_effective_from():
    """effectiveFrom is the one @NotNull field on the server's BaselineRequest.

    Every other field is optional, so a Baseline that let it default to None
    produced a request the server rejects with 400 on every call. Making it a
    required constructor argument turns that runtime 400 into a TypeError at
    the call site.
    """
    with pytest.raises(TypeError, match="effective_from"):
        Baseline(cost_per_unit=4.5)  # type: ignore[call-arg]

    assert Baseline(effective_from="2026-08-01T00:00:00Z").to_wire() == {
        "effectiveFrom": "2026-08-01T00:00:00Z"
    }


def test_a_period_fact_can_carry_the_reason_a_correction_requires():
    """A fact that supersedes an active one is 400ed without a reason.

    Facts are keyed on (periodStart, periodEnd, dimensionKey, dimensionValue,
    key). Re-appending that tuple deactivates the prior row, and the server
    answers "reason is required when correcting an existing fact" unless the
    entry carries one. Without the field there was no way to restate a period.
    """
    corrected = PeriodFactEntry(
        "2026-08-01T00:00:00Z", "2026-08-02T00:00:00Z", "region", "us", "completed_claims", 9,
        reason="restating August after the warehouse reload",
    )
    assert corrected.to_wire()["reason"] == "restating August after the warehouse reload"

    first_write = PeriodFactEntry(
        "2026-08-01T00:00:00Z", "2026-08-02T00:00:00Z", "region", "us", "completed_claims", 4,
    )
    assert "reason" not in first_write.to_wire()


def test_economics_declares_only_the_fields_the_request_contract_has():
    """JobTypeEconomicsRequest has no guardrail field.

    The SDK accepted a guardrail= mapping and put it on the wire, where
    Jackson drops the unknown property without complaint, so a caller who set
    a guardrail was told nothing and got no guardrail.
    """
    with pytest.raises(TypeError, match="guardrail"):
        JobTypeEconomics(  # type: ignore[call-arg]
            "completed_claims", "claim", [], [],
            guardrail={"metricKey": "completed_claims", "threshold": 1},
        )

    wire = JobTypeEconomics(
        "completed_claims", "claim",
        [{"key": "completed_claims", "type": "COUNT", "direction": "HIGHER_IS_BETTER",
          "aggregation": "SUM", "resolution": "PER_JOB"}],
        [],
    ).to_wire()
    assert set(wire) == {"unitMetricKey", "unitLabel", "metrics", "dimensions"}


def test_the_per_job_append_is_delegated_to_the_shared_outcome_helper():
    """POST /jobs/{id}/outcome/metrics belongs to BACK-3080, not to this module.

    This module briefly carried its own module-level transport for that one
    endpoint. Two public surfaces on one endpoint is one too many, and the one
    that survived is the one that can send the reason a correction needs. The
    append now reaches the wire through _core.outcomes, so a caller gets that
    path whether they come in via JobContext or the client.
    """
    import revenium_middleware
    import revenium_middleware.job_type_economics as economics
    from revenium_middleware import AgenticOutcomeClient, JobContext
    from revenium_middleware._core import outcomes

    # The shared helper is the single definition of the append.
    assert callable(outcomes.append_outcome_metrics_request)
    assert callable(JobContext.append_outcome_metrics)
    assert callable(AgenticOutcomeClient.append_outcome_metrics)

    # This module neither redefines nor re-exports it.
    for name in ("report_outcome_metrics", "OutcomeMetricEntry"):
        assert not hasattr(economics, name), f"{name} is BACK-3080's to define"
        assert name not in revenium_middleware.__all__


def test_the_append_only_posts_share_the_non_idempotent_retry_policy():
    """A retried append is a second row, not a no-op - only 429 is safe.

    Baselines and period facts are append-only, so they take the same narrowed
    policy as append_outcome_metrics_request rather than restating one: a 429
    proves the origin rejected the request before processing it, while a
    502/503/504 may have committed. Pinning retry_attempts=1 instead, as this
    module used to, is safe but throws away that one safe retry.
    """
    from revenium_middleware._core.outcomes import _NON_IDEMPOTENT_RETRY_STATUSES

    assert _NON_IDEMPOTENT_RETRY_STATUSES == frozenset({429})

    seen = {}

    def fake_request_with_retry(client, method, url, **kwargs):
        seen[url.rsplit("/", 1)[-1]] = kwargs.get("retry_statuses")
        return httpx.Response(200, json={}, request=httpx.Request(method, url))

    client, _ = _client()
    with patch.dict(os.environ, ENV):
        with patch.object(economics_module, "_request_with_retry", fake_request_with_retry):
            economics_module.create_baseline(
                "claims", Baseline(effective_from="2026-08-01T00:00:00Z"),
                profitstream_base_url=BASE, http_client=client,
            )
            economics_module.report_period_facts(
                "claims",
                [PeriodFactEntry("2026-08-01T00:00:00Z", "2026-08-02T00:00:00Z",
                                 "region", "us", "completed_claims", 4)],
                profitstream_base_url=BASE, http_client=client,
            )
            economics_module.upsert_job_type_economics(
                "claims",
                JobTypeEconomics("completed_claims", "claim", [{"key": "completed_claims"}], []),
                profitstream_base_url=BASE, http_client=client,
            )
            economics_module.get_job_type_economics(
                "claims", profitstream_base_url=BASE, http_client=client,
            )

    # The append-only POSTs narrow the policy; the idempotent PUT and GET
    # keep the shared default (None -> _DEFAULT_RETRY_STATUSES).
    assert seen["baselines"] == _NON_IDEMPOTENT_RETRY_STATUSES
    assert seen["facts"] == _NON_IDEMPOTENT_RETRY_STATUSES
    assert seen["economics"] is None


def test_a_caller_can_still_override_the_retry_policy():
    """The narrowed policy is a default, not a lock."""
    seen = {}

    def fake_request_with_retry(client, method, url, **kwargs):
        seen["retry_statuses"] = kwargs.get("retry_statuses")
        seen["retry_attempts"] = kwargs.get("retry_attempts")
        return httpx.Response(200, json={}, request=httpx.Request(method, url))

    client, _ = _client()
    with patch.dict(os.environ, ENV):
        with patch.object(economics_module, "_request_with_retry", fake_request_with_retry):
            economics_module.create_baseline(
                "claims", Baseline(effective_from="2026-08-01T00:00:00Z"),
                profitstream_base_url=BASE, http_client=client,
                retry_statuses={429, 503}, retry_attempts=5,
            )

    assert seen["retry_statuses"] == {429, 503}
    assert seen["retry_attempts"] == 5


def test_the_contract_fixture_records_what_the_server_requires():
    assert CONTRACT["requiredFields"] == {
        "economics": ["unitMetricKey", "unitLabel", "metrics"],
        "baseline": ["effectiveFrom"],
        "periodFact": [
            "periodStart", "periodEnd", "dimensionKey", "dimensionValue", "key", "value",
        ],
    }
