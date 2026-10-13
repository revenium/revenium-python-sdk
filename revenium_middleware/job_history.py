"""Reads of agentic jobs: the job list, one job, job types, ROI, transactions, conversion funnel and outcome history."""

import urllib.parse
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Mapping, Optional

import httpx

from ._core.config import get_profitstream_base_url, resolve_write_api_key
from ._core.exceptions import OutcomeReportingError
from ._core.outcomes import (
    _request_with_retry,
    get_outcome_history_request,
    resolve_team_id,
    validate_outcome_key,
)


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
    ``REVENIUM_WRITE_API_KEY`` > ``REVENIUM_OUTCOME_API_KEY`` (deprecated
    fallback) > ``REVENIUM_METERING_API_KEY``.

    The retry knobs match ``JobContext``: omit them for the default bounded
    schedule, or pass any of them to tune this GET's transient-error retry.
    """
    if not isinstance(job_id, str) or not job_id.strip():
        raise ValueError("job_id must be a non-empty string")
    key = resolve_write_api_key(api_key)
    if not key:
        raise OutcomeReportingError(
            "No API key available: pass api_key= or set REVENIUM_WRITE_API_KEY "
            "(REVENIUM_OUTCOME_API_KEY is a deprecated fallback; write-scope "
            "rev_sk_ key required)."
        )
    key = validate_outcome_key(key)
    base_url = get_profitstream_base_url(profitstream_base_url)
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


def _read_jobs_api(
    path: str,
    query: Optional[Mapping[str, Any]] = None,
    *,
    team_id: Optional[str] = None,
    api_key: Optional[str] = None,
    profitstream_base_url: Optional[str] = None,
    http_client: Optional[httpx.Client] = None,
    retry_attempts: Optional[int] = None,
    retry_initial_seconds: Optional[float] = None,
    retry_max_seconds: Optional[float] = None,
) -> Any:
    key = resolve_write_api_key(api_key)
    if not key:
        raise OutcomeReportingError(
            "No API key available: pass api_key= or set REVENIUM_WRITE_API_KEY "
            "(REVENIUM_OUTCOME_API_KEY is a deprecated fallback; write-scope "
            "rev_sk_ key required)."
        )
    key = validate_outcome_key(key)
    base_url = get_profitstream_base_url(profitstream_base_url)
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
        params = {"teamId": resolved_team}
        params.update({name: str(value) for name, value in (query or {}).items() if value is not None})
        response = _request_with_retry(
            client,
            "GET",
            f"{base_url.rstrip('/')}/profitstream/v2/api/jobs{path}",
            params=params,
            body=None,
            api_key=key,
            **retry_kwargs,
        )
        return response.json() if response is not None and response.content else None
    finally:
        if owns_client:
            client.close()


def _job_path(job_id: str, suffix: str = "") -> str:
    if not isinstance(job_id, str) or not job_id.strip():
        raise ValueError("job_id must be a non-empty string")
    return "/" + urllib.parse.quote(job_id, safe="") + suffix


def _page(data: Any) -> Dict[str, Any]:
    # The API answers a Spring HAL page, which omits ``_embedded`` when empty;
    # a plain ``content`` page is accepted too, as the Go and Node SDKs do.
    body = data if isinstance(data, dict) else {}
    content = body.get("content")
    if not isinstance(content, list):
        embedded = body.get("_embedded")
        content = embedded.get("jobResourceList") if isinstance(embedded, dict) else None
    if not isinstance(content, list):
        content = []
    page = body.get("page")
    if not isinstance(page, dict):
        page = {
            "size": body.get("size", len(content)),
            "totalElements": body.get("totalElements", len(content)),
            "totalPages": body.get("totalPages", 1),
            "number": body.get("number", 0),
        }
    return {"content": content, "page": page}


def list_jobs(
    *,
    job_type: Optional[str] = None,
    execution_status: Optional[str] = None,
    outcome_type: Optional[str] = None,
    environment: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    page: Optional[int] = None,
    size: Optional[int] = None,
    sort: Optional[str] = None,
    team_id: Optional[str] = None,
    api_key: Optional[str] = None,
    profitstream_base_url: Optional[str] = None,
    http_client: Optional[httpx.Client] = None,
    retry_attempts: Optional[int] = None,
    retry_initial_seconds: Optional[float] = None,
    retry_max_seconds: Optional[float] = None,
) -> Dict[str, Any]:
    """Return one page of jobs as ``{"content": [job, ...], "page": {...}}``.

    Only the filters you pass are sent. ``page`` is 0-based; dates are ISO 8601
    strings passed through as given. Each job is the API's JSON (camelCase).

    Requires a write-scope API key (``rev_sk_``): explicit ``api_key`` >
    ``REVENIUM_WRITE_API_KEY`` > ``REVENIUM_OUTCOME_API_KEY`` (deprecated
    fallback) > ``REVENIUM_METERING_API_KEY``.
    """
    query = {
        "type": job_type,
        "executionStatus": execution_status,
        "outcomeType": outcome_type,
        "environment": environment,
        "startDate": start_date,
        "endDate": end_date,
        "page": page,
        "size": size,
        "sort": sort,
    }
    return _page(_read_jobs_api(
        "", query, team_id=team_id, api_key=api_key,
        profitstream_base_url=profitstream_base_url, http_client=http_client,
        retry_attempts=retry_attempts, retry_initial_seconds=retry_initial_seconds,
        retry_max_seconds=retry_max_seconds,
    ))


def get_job(
    job_id: str,
    *,
    team_id: Optional[str] = None,
    api_key: Optional[str] = None,
    profitstream_base_url: Optional[str] = None,
    http_client: Optional[httpx.Client] = None,
    retry_attempts: Optional[int] = None,
    retry_initial_seconds: Optional[float] = None,
    retry_max_seconds: Optional[float] = None,
) -> Dict[str, Any]:
    """Return one job as the API's JSON, including its ``entityVersion``.

    A job that does not exist raises ``httpx.HTTPStatusError`` (404). An id
    equal to ``types`` or ``conversion-funnel`` reaches those routes on the
    server instead of a job.

    Requires a write-scope API key (``rev_sk_``): explicit ``api_key`` >
    ``REVENIUM_WRITE_API_KEY`` > ``REVENIUM_OUTCOME_API_KEY`` (deprecated
    fallback) > ``REVENIUM_METERING_API_KEY``.
    """
    return _read_jobs_api(
        _job_path(job_id), team_id=team_id, api_key=api_key,
        profitstream_base_url=profitstream_base_url, http_client=http_client,
        retry_attempts=retry_attempts, retry_initial_seconds=retry_initial_seconds,
        retry_max_seconds=retry_max_seconds,
    )


def get_job_types(
    *,
    team_id: Optional[str] = None,
    api_key: Optional[str] = None,
    profitstream_base_url: Optional[str] = None,
    http_client: Optional[httpx.Client] = None,
    retry_attempts: Optional[int] = None,
    retry_initial_seconds: Optional[float] = None,
    retry_max_seconds: Optional[float] = None,
) -> List[str]:
    """Return the job types the team has used.

    Requires a write-scope API key (``rev_sk_``): explicit ``api_key`` >
    ``REVENIUM_WRITE_API_KEY`` > ``REVENIUM_OUTCOME_API_KEY`` (deprecated
    fallback) > ``REVENIUM_METERING_API_KEY``.
    """
    data = _read_jobs_api(
        "/types", team_id=team_id, api_key=api_key,
        profitstream_base_url=profitstream_base_url, http_client=http_client,
        retry_attempts=retry_attempts, retry_initial_seconds=retry_initial_seconds,
        retry_max_seconds=retry_max_seconds,
    )
    return data if isinstance(data, list) else []


def get_job_roi(
    job_id: str,
    *,
    team_id: Optional[str] = None,
    api_key: Optional[str] = None,
    profitstream_base_url: Optional[str] = None,
    http_client: Optional[httpx.Client] = None,
    retry_attempts: Optional[int] = None,
    retry_initial_seconds: Optional[float] = None,
    retry_max_seconds: Optional[float] = None,
) -> Dict[str, Any]:
    """Return a job's return on investment (``roi``, ``totalCost``, ``outcomeValue``, tokens).

    Requires a write-scope API key (``rev_sk_``): explicit ``api_key`` >
    ``REVENIUM_WRITE_API_KEY`` > ``REVENIUM_OUTCOME_API_KEY`` (deprecated
    fallback) > ``REVENIUM_METERING_API_KEY``.
    """
    return _read_jobs_api(
        _job_path(job_id, "/roi"), team_id=team_id, api_key=api_key,
        profitstream_base_url=profitstream_base_url, http_client=http_client,
        retry_attempts=retry_attempts, retry_initial_seconds=retry_initial_seconds,
        retry_max_seconds=retry_max_seconds,
    )


def get_job_transactions(
    job_id: str,
    *,
    team_id: Optional[str] = None,
    api_key: Optional[str] = None,
    profitstream_base_url: Optional[str] = None,
    http_client: Optional[httpx.Client] = None,
    retry_attempts: Optional[int] = None,
    retry_initial_seconds: Optional[float] = None,
    retry_max_seconds: Optional[float] = None,
) -> Dict[str, Any]:
    """Return a job's AI calls as ``{"transactions": [...], "totalCount": n}``.

    Requires a write-scope API key (``rev_sk_``): explicit ``api_key`` >
    ``REVENIUM_WRITE_API_KEY`` > ``REVENIUM_OUTCOME_API_KEY`` (deprecated
    fallback) > ``REVENIUM_METERING_API_KEY``.
    """
    return _read_jobs_api(
        _job_path(job_id, "/transactions"), team_id=team_id, api_key=api_key,
        profitstream_base_url=profitstream_base_url, http_client=http_client,
        retry_attempts=retry_attempts, retry_initial_seconds=retry_initial_seconds,
        retry_max_seconds=retry_max_seconds,
    )


def get_conversion_funnel(
    *,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    job_type: Optional[str] = None,
    environment: Optional[str] = None,
    team_id: Optional[str] = None,
    api_key: Optional[str] = None,
    profitstream_base_url: Optional[str] = None,
    http_client: Optional[httpx.Client] = None,
    retry_attempts: Optional[int] = None,
    retry_initial_seconds: Optional[float] = None,
    retry_max_seconds: Optional[float] = None,
) -> Dict[str, Any]:
    """Return the conversion funnel: job counts by outcome plus success and conversion rates.

    Only the filters you pass are sent; dates are ISO 8601 strings.

    Requires a write-scope API key (``rev_sk_``): explicit ``api_key`` >
    ``REVENIUM_WRITE_API_KEY`` > ``REVENIUM_OUTCOME_API_KEY`` (deprecated
    fallback) > ``REVENIUM_METERING_API_KEY``.
    """
    query = {
        "startDate": start_date,
        "endDate": end_date,
        "jobType": job_type,
        "environment": environment,
    }
    return _read_jobs_api(
        "/conversion-funnel", query, team_id=team_id, api_key=api_key,
        profitstream_base_url=profitstream_base_url, http_client=http_client,
        retry_attempts=retry_attempts, retry_initial_seconds=retry_initial_seconds,
        retry_max_seconds=retry_max_seconds,
    )
