"""Outcome amendment history (BACK-777 Phase 3, addendum §C)."""

import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

import httpx

from ._core.config import Config
from ._core.exceptions import OutcomeReportingError
from ._core.outcomes import (
    get_outcome_history_request,
    resolve_team_id,
    validate_outcome_key,
)

_DEFAULT_PROFITSTREAM_BASE_URL = "https://api.revenium.io"


@dataclass(frozen=True)
class JobOutcomeAmendment:
    """One row of a job's outcome history; sequence 1 is the initial report."""

    amendment_sequence: int
    execution_status: str
    outcome_type: Optional[str] = None
    outcome_value: Optional[float] = None
    outcome_currency: Optional[str] = None
    outcome_metadata: Optional[str] = None
    reported_by: Optional[str] = None
    reported_at: Optional[datetime] = None
    reason: Optional[str] = None  # None for the initial report (sequence=1)
    # The outcome's own business failure/cancellation explanation, distinct from
    # ``reason`` above (this amendment's audit justification). Kept last and
    # defaulted so positional construction of the earlier fields still works.
    outcome_reason: Optional[str] = None


def get_outcome_history(
    job_id: str,
    *,
    team_id: Optional[str] = None,
    api_key: Optional[str] = None,
    profitstream_base_url: Optional[str] = None,
    http_client: Optional[httpx.Client] = None,
    retry_attempts: Optional[int] = None,
    retry_initial_seconds: Optional[float] = None,
    retry_max_seconds: Optional[float] = None,
) -> List[JobOutcomeAmendment]:
    """Return a job's full outcome history, ordered by amendment_sequence ASC.

    Requires a write-scope API key (``rev_sk_``): explicit ``api_key`` >
    ``REVENIUM_OUTCOME_API_KEY`` > ``REVENIUM_METERING_API_KEY``.

    The retry knobs match ``JobContext``: omit them for the default bounded
    schedule, or pass any of them to tune this GET's transient-error retry.
    """
    if not isinstance(job_id, str) or not job_id.strip():
        raise ValueError("job_id must be a non-empty string")
    key = (
        api_key
        or os.getenv(Config.ENV_REVENIUM_OUTCOME_API_KEY)
        or os.getenv(Config.ENV_REVENIUM_API_KEY)
    )
    if not key:
        raise OutcomeReportingError(
            "No API key available: pass api_key= or set REVENIUM_OUTCOME_API_KEY "
            "(write-scope rev_sk_ key required)."
        )
    key = validate_outcome_key(key)
    base_url = (
        profitstream_base_url
        or os.getenv(Config.ENV_REVENIUM_PROFITSTREAM_BASE_URL)
        or _DEFAULT_PROFITSTREAM_BASE_URL
    )
    retry_kwargs: Dict[str, Any] = {}
    if retry_attempts is not None:
        retry_kwargs["retry_attempts"] = retry_attempts
    if retry_initial_seconds is not None:
        retry_kwargs["retry_initial_seconds"] = retry_initial_seconds
    if retry_max_seconds is not None:
        retry_kwargs["retry_max_seconds"] = retry_max_seconds
    owns_client = http_client is None
    client = http_client or httpx.Client(timeout=60.0)
    try:
        resolved_team = resolve_team_id(team_id or "", key, client, base_url, use_env=True)
        if not resolved_team:
            raise OutcomeReportingError(
                "Could not resolve team_id: pass team_id= or set REVENIUM_TEAM_ID."
            )
        response = get_outcome_history_request(
            client, base_url, job_id, team_id=resolved_team, api_key=key,
            **retry_kwargs,
        )
        return _parse_history(response.json())
    finally:
        if owns_client:
            client.close()


def _parse_history(data: Any) -> List[JobOutcomeAmendment]:
    # The backend returns a bare JSON array of JobOutcomeRevisionResource;
    # tolerate a HAL-style wrapper keyed by the Spring relation name.
    if isinstance(data, dict):
        data = data.get("_embedded", {}).get("jobOutcomeRevisionResourceList", [])
    rows = data if isinstance(data, list) else []
    parsed = [
        JobOutcomeAmendment(
            amendment_sequence=int(row.get("sequence", 0)),
            execution_status=row.get("executionStatus", ""),
            outcome_type=row.get("outcomeType"),
            outcome_value=row.get("outcomeValue"),
            outcome_currency=row.get("outcomeCurrency"),
            outcome_metadata=row.get("outcomeMetadata"),
            reported_by=row.get("reportedBy"),
            reported_at=_parse_dt(row.get("reportedAt")),
            reason=row.get("reason"),
            outcome_reason=row.get("outcomeReason"),
        )
        for row in rows
    ]
    return sorted(parsed, key=lambda a: a.amendment_sequence)


def _parse_dt(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
