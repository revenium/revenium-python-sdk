"""Job-type economics and period-fact API helpers.

These management-plane calls require a write-scope ``rev_sk_`` key. Metering
keys remain valid only for AI metering events.

Scope: this module owns the *job type* surface - the economics declaration,
its baselines, and period facts. The *per-job* outcome-metric append
(``POST /jobs/{id}/outcome/metrics``) is not here and must not be added here.
BACK-3080 owns it, as ``JobContext.append_outcome_metrics`` and
``AgenticOutcomeClient.append_outcome_metrics`` over
``_core.outcomes.append_outcome_metrics_request``. That is also where a caller
supplies the ``reason`` the server requires when an append supersedes an
active fact, so a second surface here would be the weaker of the two.

Retry policy is shared with those calls, not restated. The append-only POSTs
below - a baseline version and a batch of period facts - are not repeats of
themselves: a second delivery is a second row, or a 409 against the
active-fact constraint for a write that already landed. They pass
``_core.outcomes._NON_IDEMPOTENT_RETRY_STATUSES``, the same frozenset
``append_outcome_metrics_request`` uses, so only a 429 is retried - the one
status that proves the origin rejected the request before processing it. A
502/503/504 is ambiguous and surfaces to the caller. The upsert is a PUT and
the reads are GETs, so both keep the default policy.
"""

import os
import urllib.parse
from dataclasses import asdict, dataclass
from typing import (
    Any, Collection, Dict, List, Literal, Mapping, Optional, Sequence, Union,
)

import httpx

from ._core.config import Config, resolve_write_api_key
from ._core.exceptions import OutcomeReportingError
from ._core.outcomes import (
    _NON_IDEMPOTENT_RETRY_STATUSES,
    _request_with_retry,
    resolve_team_id,
    validate_outcome_key,
)

_DEFAULT_PROFITSTREAM_BASE_URL = "https://api.revenium.io"

# Keep these literals aligned with HyperCurrent's JobTypeEconomicsRequests.kt
# and JobTypeEconomicsResource.kt request contract.
BaselineProvenance = Literal["CUSTOMER_DECLARED", "MEASURED", "SIGNED_OFF"]
OutcomeMetricProvenance = Literal["MEASURED", "SELF_REPORTED", "DERIVED", "ATTESTED"]
JobTypeMetricDirection = Literal["HIGHER_IS_BETTER", "LOWER_IS_BETTER"]
JobTypeMonetizationCategory = Literal["REVENUE", "COST_AVOIDED", "TIME_SAVED", "LEADING_VALUE"]
JobTypeMonetizationBasis = Literal["REALIZED", "EXPECTED"]


@dataclass(frozen=True)
class PeriodFactEntry:
    """A period fact with optional attribution fields.

    Omit attribution fields to use the server defaults, or set them to record
    an explicit override with MEASURED, SELF_REPORTED, or DERIVED provenance.

    Facts are append-only and keyed on
    ``(period_start, period_end, dimension_key, dimension_value, key)``. An
    append on a key that already has an active fact supersedes it, and the
    server rejects that correction with 400 "reason is required when
    correcting an existing fact" unless ``reason`` is set. Whether an append
    corrects anything is a property of server state, so a caller who may be
    restating a period must supply ``reason``.
    """
    period_start: str
    period_end: str
    dimension_key: str
    dimension_value: str
    key: str
    value: float
    provenance: Optional[OutcomeMetricProvenance] = None
    recorded_by: Optional[str] = None
    source: Optional[str] = None
    reason: Optional[str] = None

    def to_wire(self) -> Dict[str, Any]:
        return _camelize(asdict(self))


@dataclass(frozen=True)
class Baseline:
    """A baseline with optional attribution fields.

    ``effective_from`` is required: it is the only @NotNull field on the
    server's BaselineRequest, so a baseline without it is a guaranteed 400.
    It is declared first here so the required field cannot be omitted.

    Currency values must be USD. Omit attribution fields to use the server
    defaults, or set them to record an explicit override with CUSTOMER_DECLARED
    or MEASURED provenance.
    """
    effective_from: str
    cost_per_unit: Optional[float] = None
    minutes_per_unit: Optional[float] = None
    quality_rate: Optional[float] = None
    hourly_rate: Optional[float] = None
    currency: Optional[str] = None
    provenance: Optional[BaselineProvenance] = None
    declared_by: Optional[str] = None
    evidence_url: Optional[str] = None

    def to_wire(self) -> Dict[str, Any]:
        return _camelize(asdict(self))


@dataclass(frozen=True)
class JobTypeEconomics:
    """A complete economics declaration, including the nested monetization rule.

    Mirrors the server's JobTypeEconomicsRequest field for field.
    ``overhead_per_unit`` and ``overhead_currency`` must be supplied together
    or the server rejects the declaration.
    """
    unit_metric_key: str
    unit_label: str
    metrics: Sequence[Mapping[str, Any]]
    dimensions: Sequence[Mapping[str, Any]]
    monetization: Optional[Mapping[str, Any]] = None
    overhead_per_unit: Optional[float] = None
    overhead_currency: Optional[str] = None

    def to_wire(self) -> Dict[str, Any]:
        return _camelize(asdict(self))


WireValue = Union[Mapping[str, Any], PeriodFactEntry, Baseline, JobTypeEconomics]


def _camelize(values: Mapping[str, Any]) -> Dict[str, Any]:
    def wire_name(name: str) -> str:
        parts = name.split("_")
        return parts[0] + "".join(part.title() for part in parts[1:])
    return {wire_name(key): value for key, value in values.items() if value is not None}


def _wire(value: WireValue) -> Dict[str, Any]:
    if hasattr(value, "to_wire"):
        return value.to_wire()  # type: ignore[no-any-return]
    return dict(value)


def _request(
    method: str, path: str, body: Optional[Any] = None, *, team_id: Optional[str] = None,
    api_key: Optional[str] = None, profitstream_base_url: Optional[str] = None,
    http_client: Optional[httpx.Client] = None, retry_attempts: Optional[int] = None,
    retry_statuses: Optional[Collection[int]] = None,
) -> Any:
    key = resolve_write_api_key(api_key)
    if not key:
        raise OutcomeReportingError(
            "No API key available: pass api_key= or set REVENIUM_WRITE_API_KEY "
            "(REVENIUM_OUTCOME_API_KEY is a deprecated fallback; write-scope "
            "rev_sk_ key required)."
        )
    key = validate_outcome_key(key)
    base_url = profitstream_base_url or os.getenv(Config.ENV_REVENIUM_PROFITSTREAM_BASE_URL) or _DEFAULT_PROFITSTREAM_BASE_URL
    owns_client = http_client is None
    client = http_client or httpx.Client(timeout=60.0)
    try:
        resolved_team = resolve_team_id(team_id or "", key, client, base_url, use_env=True)
        if not resolved_team:
            raise OutcomeReportingError("Could not resolve team_id: pass team_id= or set REVENIUM_TEAM_ID.")
        retry_kwargs: Dict[str, Any] = {}
        if retry_attempts is not None:
            retry_kwargs["retry_attempts"] = retry_attempts
        if retry_statuses is not None:
            retry_kwargs["retry_statuses"] = retry_statuses
        response = _request_with_retry(
            client, method, f"{base_url.rstrip('/')}/profitstream/v2/api{path}",
            params={"teamId": resolved_team}, body=body, api_key=key, **retry_kwargs,
        )
        return response.json() if response is not None and response.content else None
    finally:
        if owns_client:
            client.close()


def report_period_facts(job_type: str, facts: Sequence[WireValue], **kwargs: Any) -> Any:
    """Append period facts with optional attribution fields for a job type.

    ``facts`` is the request body itself - a bare JSON array, which is what the
    endpoint declares. Append-only: set ``reason`` on an entry that restates a
    period already recorded, or the server rejects the correction.
    """
    kwargs.setdefault("retry_statuses", _NON_IDEMPOTENT_RETRY_STATUSES)
    return _request("POST", f"/jobs/types/{urllib.parse.quote(job_type, safe='')}/facts", [_wire(fact) for fact in facts], **kwargs)


def get_job_type_economics(job_type: str, **kwargs: Any) -> Any:
    return _request("GET", f"/jobs/types/{urllib.parse.quote(job_type, safe='')}/economics", **kwargs)


def upsert_job_type_economics(job_type: str, economics: WireValue, **kwargs: Any) -> Any:
    return _request("PUT", f"/jobs/types/{urllib.parse.quote(job_type, safe='')}/economics", _wire(economics), **kwargs)


def create_baseline(job_type: str, baseline: WireValue, **kwargs: Any) -> Any:
    """Append a new immutable baseline version for a job type.

    Append-only: each call adds a version rather than replacing one, so this
    carries the same narrowed retry policy as the facts append.
    """
    kwargs.setdefault("retry_statuses", _NON_IDEMPOTENT_RETRY_STATUSES)
    return _request("POST", f"/jobs/types/{urllib.parse.quote(job_type, safe='')}/baselines", _wire(baseline), **kwargs)


def list_baselines(job_type: str, **kwargs: Any) -> List[Any]:
    result = _request("GET", f"/jobs/types/{urllib.parse.quote(job_type, safe='')}/baselines", **kwargs)
    return result if isinstance(result, list) else []
