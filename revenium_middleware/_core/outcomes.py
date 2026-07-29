"""Shared outcome-reporting transport (BACK-777 Phase 2).

Extracted from ``AgenticOutcomeClient`` so the public ``JobContext`` and the
examples-pack client share one implementation of team-id resolution, retry
semantics (bounded, Retry-After-aware), and outcome POSTs. The 404 retry on
outcome POSTs is deliberate: metric ingestion creates the Job asynchronously,
so a fresh outcome can race ahead until Kafka catches up.
"""

import logging
import time
import urllib.parse
from typing import Any, Dict, Optional

import httpx

from .config import get_team_id
from .exceptions import (
    OutcomeAlreadyReportedError,
    OutcomeAmendConflictError,
    OutcomeNotReportedError,
)

logger = logging.getLogger(__name__)

DEFAULT_RETRY_ATTEMPTS = 10
DEFAULT_RETRY_INITIAL_SECONDS = 2.0
DEFAULT_RETRY_MAX_SECONDS = 90.0

_METERING_KEY_PREFIX = "rev_mk_"


def build_headers(api_key: str) -> Dict[str, str]:
    return {
        "x-api-key": api_key,
        "Content-Type": "application/json",
        "Accept": "application/json",
    }


def validate_outcome_key(api_key: str) -> str:
    """Outcomes require a write-scope key — fail fast on a metering key.

    Metering keys (``rev_mk_``) can only meter completions and tool events;
    job/outcome control needs a write-scope key (``rev_sk_``). Other prefixes
    (legacy ``hak_`` keys) are not rejected client-side.
    """
    if api_key.startswith(_METERING_KEY_PREFIX):
        raise ValueError(
            "Outcome reporting requires a write-scope API key (rev_sk_); "
            "a metering key (rev_mk_) cannot report outcomes. Pass a write "
            "key via api_key= or set REVENIUM_OUTCOME_API_KEY."
        )
    return api_key


def resolve_team_id(
    explicit: str,
    api_key: str,
    http_client: httpx.Client,
    profitstream_base_url: str,
    *,
    use_env: bool = True,
) -> str:
    """Resolve the team id for outcome calls. Returns "" when unresolvable.

    Chain: explicit > REVENIUM_TEAM_ID (when use_env) > key-prefix +
    ProfitStream teams API. Failures are warning-logged, not raised — callers
    decide whether an empty result is fatal (JobContext raises; the examples
    client preserves its historical warn-and-continue behavior).
    """
    if explicit:
        return explicit
    if use_env:
        env_team = get_team_id()
        if env_team:
            return env_team

    # Automatic resolution via key-prefix (format: rev_[ms]k_TENANTID_...) and
    # the ProfitStream teams API.
    parts = api_key.split("_")
    if len(parts) < 3:
        logger.warning(
            "Could not infer team ID: API key does not match expected "
            "rev_[ms]k_TENANTID_... format. Outcome calls may target the wrong team."
        )
        return ""
    tenant_id = parts[2]

    url = f"{profitstream_base_url.rstrip('/')}/profitstream/v2/api/teams"
    try:
        response = http_client.get(
            url,
            headers=build_headers(api_key),
            params={"tenantId": tenant_id},
            timeout=10.0,
        )
        response.raise_for_status()
        data = response.json()
        teams = data.get("_embedded", {}).get("teamResourceList", [])
        if teams:
            return teams[0].get("id", "") or ""
        logger.warning("No teams returned for tenantId=%s; outcome calls may fail.", tenant_id)
    except (httpx.HTTPError, ValueError) as exc:
        logger.warning(
            "Failed to resolve team ID for tenantId=%s: %s. Outcome calls may target the wrong team.",
            tenant_id, exc,
        )
    return ""


def _request_with_retry(
    http_client: httpx.Client,
    method: str,
    url: str,
    *,
    params: Optional[Dict[str, str]],
    body: Optional[Dict[str, Any]],
    api_key: str,
    retry_on_404: bool = False,
    accept_409: bool = False,
    raise_typed_on_409: bool = False,
    retry_attempts: int = DEFAULT_RETRY_ATTEMPTS,
    retry_initial_seconds: float = DEFAULT_RETRY_INITIAL_SECONDS,
    retry_max_seconds: float = DEFAULT_RETRY_MAX_SECONDS,
) -> Optional[httpx.Response]:
    """Issue ``method`` with bounded retry. Returns the response on success.

    ``body=None`` sends no JSON body (httpx treats ``json=None`` as absent) —
    used by bodiless GETs.

    409 handling modes: ``accept_409`` treats conflict as idempotent success
    (job creation); ``raise_typed_on_409`` parses the structured body and
    raises ``OutcomeAlreadyReportedError`` when it carries amendment guidance,
    falling back to warn-and-return for unparseable bodies (older backends).
    """
    attempts = max(1, retry_attempts)
    retryable = {429, 502, 503, 504}
    if retry_on_404:
        retryable.add(404)
    response: Optional[httpx.Response] = None
    for attempt in range(attempts):
        response = http_client.request(
            method,
            url,
            headers=build_headers(api_key),
            params=params,
            json=body,
            timeout=30.0,
        )
        if 200 <= response.status_code < 300:
            return response
        if response.status_code == 409:
            if accept_409:
                return response
            if raise_typed_on_409:
                _handle_conflict(response)
                return response
        if response.status_code in retryable and attempt < attempts - 1:
            delay = _retry_delay(response, attempt, retry_initial_seconds, retry_max_seconds)
            time.sleep(delay)
            continue
        response.raise_for_status()
    return response


def post_with_retry(
    http_client: httpx.Client,
    url: str,
    *,
    params: Optional[Dict[str, str]],
    body: Dict[str, Any],
    api_key: str,
    retry_on_404: bool = False,
    accept_409: bool = False,
    raise_typed_on_409: bool = False,
    retry_attempts: int = DEFAULT_RETRY_ATTEMPTS,
    retry_initial_seconds: float = DEFAULT_RETRY_INITIAL_SECONDS,
    retry_max_seconds: float = DEFAULT_RETRY_MAX_SECONDS,
) -> Optional[httpx.Response]:
    """POST with bounded retry. Returns the response on success/accepted-409.

    Thin wrapper over ``_request_with_retry`` kept with this exact signature —
    the examples-pack client (``agentic_outcomes.py``) imports and calls it.
    """
    return _request_with_retry(
        http_client,
        "POST",
        url,
        params=params,
        body=body,
        api_key=api_key,
        retry_on_404=retry_on_404,
        accept_409=accept_409,
        raise_typed_on_409=raise_typed_on_409,
        retry_attempts=retry_attempts,
        retry_initial_seconds=retry_initial_seconds,
        retry_max_seconds=retry_max_seconds,
    )


def _conflict_field(data: Dict[str, Any], *names: str) -> Any:
    """Read a 409 conflict field from ``details`` first, then the body root.

    The backend nests these under ``details`` and calls the count
    ``updateCount``; earlier drafts put them at the root as ``amendmentCount``.
    A tenant can be on either backend, and a published SDK outlives both, so
    accept every spelling instead of picking one.
    """
    details = data.get("details")
    for source in (details if isinstance(details, dict) else {}, data):
        for name in names:
            value = source.get(name)
            if value is not None:
                return value
    return None


def _conflict_count(value: Any) -> Optional[int]:
    """Coerce a conflict count to ``int`` — the backend serializes it as a string.

    Anything non-numeric yields ``None``: the count is informational, so a
    surprising value must never turn a typed conflict into a parse error, and a
    ``str`` must never reach callers through ``amendment_count``.
    """
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _handle_conflict(response: httpx.Response) -> None:
    """Raise a typed exception on a structured 409; warn on unrecognised bodies.

    A 409 whose body carries PATCH guidance means the outcome is already
    reported, so callers get ``OutcomeAlreadyReportedError`` with whatever
    context the body offered (``reportedAt`` is absent when the duplicate came
    from a concurrent-write race). Bodies without that guidance are treated as
    already-reported too, but only warn, since there is nothing actionable to
    hand the caller.
    """
    try:
        data = response.json()
    except Exception:  # noqa: BLE001 — a non-JSON 409 is just an unrecognised body
        data = {}
    if not isinstance(data, dict):
        data = {}
    guidance = _conflict_field(data, "guidance")
    if "PATCH" in str(guidance or ""):
        # "error" is the HTTP reason phrase ("Conflict") on the real backend; the
        # human-readable text is in "message". Older bodies only have "error".
        message = data.get("message") or data.get("error") or "Outcome already reported"
        reported_at = _conflict_field(data, "reportedAt")
        raise OutcomeAlreadyReportedError(
            str(message),
            reported_at=None if reported_at is None else str(reported_at),
            amendment_count=_conflict_count(
                _conflict_field(data, "updateCount", "amendmentCount")
            ),
        )
    logger.warning(
        "Outcome POST returned 409 without structured amendment guidance "
        "(older backend?); treating as already-reported. Body: %.200s",
        response.text,
    )


def _retry_delay(
    response: httpx.Response,
    attempt: int,
    initial_seconds: float,
    max_seconds: float,
) -> float:
    """Calculate retry delay honoring Retry-After with a max cap."""
    retry_after_header = response.headers.get("Retry-After")
    server_retry_after: Optional[float] = None
    if retry_after_header:
        try:
            server_retry_after = float(retry_after_header)
        except (TypeError, ValueError):
            server_retry_after = None

    if server_retry_after and server_retry_after > 0:
        # Honor server hint with a 1s buffer, but never exceed the configured
        # max — protects callers from unbounded blocking sleeps.
        return min(server_retry_after + 1.0, max_seconds)

    return min(initial_seconds * (2 ** attempt), max_seconds)


def report_outcome_request(
    http_client: httpx.Client,
    profitstream_base_url: str,
    agentic_job_id: str,
    payload: Dict[str, Any],
    *,
    team_id: str,
    api_key: str,
    retry_attempts: int = DEFAULT_RETRY_ATTEMPTS,
    retry_initial_seconds: float = DEFAULT_RETRY_INITIAL_SECONDS,
    retry_max_seconds: float = DEFAULT_RETRY_MAX_SECONDS,
) -> None:
    safe_id = urllib.parse.quote(agentic_job_id, safe="")
    url = (
        f"{profitstream_base_url.rstrip('/')}"
        f"/profitstream/v2/api/jobs/{safe_id}/outcome"
    )
    params = {"teamId": team_id} if team_id else None
    post_with_retry(
        http_client,
        url,
        params=params,
        body=payload,
        api_key=api_key,
        retry_on_404=True,
        raise_typed_on_409=True,
        retry_attempts=retry_attempts,
        retry_initial_seconds=retry_initial_seconds,
        retry_max_seconds=retry_max_seconds,
    )


def amend_outcome_request(
    http_client: httpx.Client,
    profitstream_base_url: str,
    agentic_job_id: str,
    payload: Dict[str, Any],
    *,
    team_id: str,
    api_key: str,
    retry_attempts: int = DEFAULT_RETRY_ATTEMPTS,
    retry_initial_seconds: float = DEFAULT_RETRY_INITIAL_SECONDS,
    retry_max_seconds: float = DEFAULT_RETRY_MAX_SECONDS,
) -> httpx.Response:
    """PATCH an existing outcome (addendum: outcomes are amendable).

    Raises OutcomeNotReportedError on 422 (no outcome yet) and
    OutcomeAmendConflictError on 409 (optimistic-lock conflict — caller
    refetches and retries; the SDK never auto-retries amendments).

    Only 429 is auto-retried: the amendment definitely was not processed.
    5xx responses raise immediately because PATCH-amend is not idempotent —
    a 502/504 from a proxy can arrive after the origin already applied the
    amendment, and a blind retry would append a duplicate history row.
    Callers should check ``get_outcome_history`` before retrying.
    """
    safe_id = urllib.parse.quote(agentic_job_id, safe="")
    url = (
        f"{profitstream_base_url.rstrip('/')}"
        f"/profitstream/v2/api/jobs/{safe_id}/outcome"
    )
    params = {"teamId": team_id} if team_id else None
    attempts = max(1, retry_attempts)
    retryable = {429}
    response: Optional[httpx.Response] = None
    for attempt in range(attempts):
        response = http_client.request(
            "PATCH", url, headers=build_headers(api_key), params=params,
            json=payload, timeout=30.0,
        )
        if 200 <= response.status_code < 300:
            return response
        if response.status_code == 422:
            raise OutcomeNotReportedError(_error_message(response, "Job has no outcome to amend"))
        if response.status_code == 409:
            raise OutcomeAmendConflictError(_error_message(response, "Concurrent amendment conflict"))
        if response.status_code in retryable and attempt < attempts - 1:
            time.sleep(_retry_delay(response, attempt, retry_initial_seconds, retry_max_seconds))
            continue
        response.raise_for_status()
    return response


def get_outcome_history_request(
    http_client: httpx.Client,
    profitstream_base_url: str,
    agentic_job_id: str,
    *,
    team_id: str,
    api_key: str,
    retry_attempts: int = DEFAULT_RETRY_ATTEMPTS,
    retry_initial_seconds: float = DEFAULT_RETRY_INITIAL_SECONDS,
    retry_max_seconds: float = DEFAULT_RETRY_MAX_SECONDS,
) -> httpx.Response:
    """GET a job's outcome history with the standard transient-error retry.

    GET is idempotent, so {429, 502, 503, 504} are safely retried; other
    non-2xx responses raise via ``raise_for_status``.
    """
    safe_id = urllib.parse.quote(agentic_job_id, safe="")
    url = (
        f"{profitstream_base_url.rstrip('/')}"
        f"/profitstream/v2/api/jobs/{safe_id}/outcome/history"
    )
    params = {"teamId": team_id} if team_id else None
    return _request_with_retry(
        http_client,
        "GET",
        url,
        params=params,
        body=None,
        api_key=api_key,
        retry_attempts=retry_attempts,
        retry_initial_seconds=retry_initial_seconds,
        retry_max_seconds=retry_max_seconds,
    )


def _error_message(response: httpx.Response, fallback: str) -> str:
    """Best human-readable text for an error body.

    The backend puts the message in ``message`` and the HTTP reason phrase
    ("Conflict", "Unprocessable Entity") in ``error``; older bodies only carry
    ``error``, with the message in it. Prefer ``message``, then ``error``.
    """
    try:
        data = response.json()
        if isinstance(data, dict):
            return str(data.get("message") or data.get("error") or fallback)
    except Exception:  # noqa: BLE001 — a non-JSON error body is expected
        pass
    return fallback
