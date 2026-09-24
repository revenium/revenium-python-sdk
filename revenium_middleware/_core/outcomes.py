"""Shared outcome-reporting transport (BACK-777 Phase 2).

Extracted from ``AgenticOutcomeClient`` so the public ``JobContext`` and the
examples-pack client share one implementation of team-id resolution, retry
semantics (bounded, Retry-After-aware), and outcome POSTs. The 404 retry on
outcome POSTs is deliberate: metric ingestion creates the Job asynchronously,
so a fresh outcome can race ahead until Kafka catches up.
"""

import logging
import re
import time
import urllib.parse
from collections import abc
from typing import Any, Collection, Dict, List, Mapping, Optional, Sequence, Union

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

# Every outcome call sends a JSON object except the facts append
# (POST .../outcome/metrics), whose request body is a bare JSON array of
# OutcomeMetricEntry. The shared request helper therefore accepts either shape:
# wrapping the array in a key the server does not declare would just be a 400.
JsonBody = Union[Dict[str, Any], List[Any]]

# Retried by default: 429 plus the gateway 5xx a proxy answers when it never
# reached the origin. Safe for the calls that carry it, all of which either
# create-or-affirm one named thing (the job) or set one terminal outcome, so a
# repeat is the same write.
_DEFAULT_RETRY_STATUSES = frozenset({429, 502, 503, 504})

# For a write that is NOT a repeat of itself — the append-only facts endpoint,
# where a second delivery is a second fact. Only 429 proves the request was
# rejected before the origin processed it; 502/503/504 are ambiguous, since the
# origin can have committed before the proxy gave up. ``amend_outcome_request``
# holds the same line in its own loop for the same reason.
_NON_IDEMPOTENT_RETRY_STATUSES = frozenset({429})

# Appended to every amendment 409. The backend's own message reports the two
# versions; this names what to do about them. It is stated at the raise site
# because the SDK deliberately does not auto-retry an amendment, so the caller
# is the only party that can close the loop.
_AMEND_CONFLICT_RECOVERY = (
    "Read the conflict's current_entity_version, re-check the outcome you "
    "intend to write (get_outcome_history shows what the other writer "
    "changed), then re-issue the amendment with expected_entity_version set to "
    "that version. When current_entity_version is None, read the job outside "
    "this SDK (GET /v2/api/jobs/{agenticJobId}) to obtain it. The SDK never "
    "auto-retries an amendment: PATCH-amend is not idempotent, so a blind "
    "retry can append a duplicate history row."
)

# The backend reports the version it actually holds only inside the 409 message
# ("Outcome has changed since entity version 0; current version is 1. Refetch
# and retry." — verified live on dev), so recovering it means reading that text.
# A structured field is preferred whenever a backend starts sending one.
# Wire-contract coupling, deliberately narrow: hypercurrent's amendment 409 carries the
# current version only inside the sentence composed in JobService ("Outcome has changed
# since entity version N; current version is M. Refetch and retry."). No structured field
# exists yet; BACK-3122 asks the platform to add details.currentEntityVersion, which
# _conflict_entity_version already prefers. Until then this regex is the contract, and
# tests/test_core/test_outcomes_transport.py pins the exact sentence it must keep matching.
_CURRENT_VERSION_IN_MESSAGE = re.compile(r"current version is\s+(\d+)")


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
            "key via api_key= or set REVENIUM_WRITE_API_KEY. "
            "REVENIUM_OUTCOME_API_KEY remains a deprecated fallback."
        )
    return api_key


def normalize_outcome_metrics(
    metrics: Optional[Sequence[Mapping[str, Any]]],
    *,
    require_entries: bool = False,
) -> Optional[List[Dict[str, Any]]]:
    """Shape-check declared metric facts before any HTTP; return them unchanged.

    Only the shape is checked: a sequence of mappings, each carrying a non-blank
    ``key`` and a ``value`` that is present and not null. Which keys a job type
    declares, and the 0..1 range on ``quality_rate``, belong to the server — it
    owns the economics contract (declared per BACK-3078), and a client-side copy
    of those rules would drift from it. Entry keys are otherwise passed through
    verbatim, so ``provenance``, ``recordedBy`` and ``source`` keep their
    documented server-side defaults instead of being filled in here.

    Shared by every entry point that accepts facts — ``JobContext`` and the
    examples-pack client, on the outcome bodies and on the append — so one
    malformed input is rejected the same way everywhere, before a request, a
    key/team resolution or a dry-run print.

    ``None`` and an empty sequence both mean "no facts": they yield ``None``, so
    the payload key is simply omitted, matching the server, which ignores an
    empty array on the outcome bodies. The dedicated append call passes
    ``require_entries=True`` and both then raise instead, because a request with
    nothing to append is a 400 and an unset variable arriving here is a caller
    bug worth naming.
    """
    # ``None`` is the same input as an empty sequence — nothing to send — and
    # folding it in here is what stops it slipping past ``require_entries``,
    # which is the likeliest way to arrive with nothing (an unset variable).
    if metrics is None:
        metrics = ()
    if isinstance(metrics, (str, bytes)) or not isinstance(metrics, abc.Sequence):
        # A single entry passed unwrapped is the likely mistake, and it would
        # reach the wire as a JSON object where an array is declared.
        raise ValueError(
            "metrics must be a sequence of {'key': ..., 'value': ...} mappings, "
            f"got {type(metrics).__name__}"
        )
    if not metrics:
        if require_entries:
            raise ValueError("metrics must contain at least one entry to append")
        return None
    normalized: List[Dict[str, Any]] = []
    for index, entry in enumerate(metrics):
        if not isinstance(entry, abc.Mapping):
            raise ValueError(
                f"metrics[{index}] must be a mapping with 'key' and 'value', "
                f"got {type(entry).__name__}"
            )
        key = entry.get("key")
        if not isinstance(key, str) or not key.strip():
            raise ValueError(f"metrics[{index}]['key'] must be a non-blank string")
        if entry.get("value") is None:
            raise ValueError(f"metrics[{index}]['value'] is required")
        normalized.append(dict(entry))
    return normalized


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
    body: Optional[JsonBody],
    api_key: str,
    retry_on_404: bool = False,
    accept_409: bool = False,
    raise_typed_on_409: bool = False,
    retry_statuses: Optional[Collection[int]] = None,
    retry_attempts: int = DEFAULT_RETRY_ATTEMPTS,
    retry_initial_seconds: float = DEFAULT_RETRY_INITIAL_SECONDS,
    retry_max_seconds: float = DEFAULT_RETRY_MAX_SECONDS,
) -> Optional[httpx.Response]:
    """Issue ``method`` with bounded retry. Returns the response on success.

    ``body=None`` sends no JSON body (httpx treats ``json=None`` as absent) —
    used by bodiless GETs. A ``list`` body is sent as a bare JSON array, which
    is what the facts endpoint declares; see ``JsonBody``.

    ``retry_statuses`` overrides which status codes are retried; omit it for
    ``_DEFAULT_RETRY_STATUSES``. A non-idempotent write narrows it (see
    ``_NON_IDEMPOTENT_RETRY_STATUSES``) rather than reimplementing this loop, so
    the Retry-After handling and the attempt bound stay in one place.

    409 handling modes: ``accept_409`` treats conflict as idempotent success
    (job creation); ``raise_typed_on_409`` parses the structured body and
    raises ``OutcomeAlreadyReportedError`` when it carries amendment guidance,
    falling back to warn-and-return for unparseable bodies (older backends).
    """
    attempts = max(1, retry_attempts)
    retryable = set(_DEFAULT_RETRY_STATUSES if retry_statuses is None else retry_statuses)
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
    body: JsonBody,
    api_key: str,
    retry_on_404: bool = False,
    accept_409: bool = False,
    raise_typed_on_409: bool = False,
    retry_statuses: Optional[Collection[int]] = None,
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
        retry_statuses=retry_statuses,
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


def coerce_entity_version(value: Any) -> Optional[int]:
    """A job ``entityVersion`` as a non-negative ``int``, or ``None``.

    The single definition of what counts as a version, shared by the parse path
    (which treats ``None`` as "no version available") and the caller-facing
    validation in ``JobContext`` (which turns it into a ``ValueError``). One
    validator so the two can never disagree about a value.

    Accepts an ``int``, an integral ``float`` (a JSON number can arrive that
    way) and a string of digits, because the field is a Kotlin ``Long`` that
    serializes as a JSON number or a string. Rejects everything else: a
    fractional value such as ``3.5`` is not a version and must never be
    silently truncated to ``3``, a ``bool`` is not a version even though it is
    an ``int``, and a negative value is one the backend answers 400 to
    (``@PositiveOrZero``) — so it is not worth caching or sending either.
    """
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        coerced = value
    elif isinstance(value, float):
        if not value.is_integer():
            return None
        coerced = int(value)
    elif isinstance(value, str):
        text = value.strip()
        # ASCII digits only: str.isdigit() also accepts superscripts and other
        # Unicode digit-like code points that int() cannot parse, and this
        # function must never raise (both callers rely on None for "no
        # version"). The field is a Kotlin Long, so ASCII is the whole contract.
        if not (text.isascii() and text.isdigit()):
            return None
        coerced = int(text)
    else:
        return None
    return coerced if coerced >= 0 else None


def parse_entity_version(response: Optional[httpx.Response]) -> Optional[int]:
    """Read ``entityVersion`` (the optimistic-lock token) off a job/outcome response.

    Every job and outcome response from a current backend carries it, and it is
    what the amendment PATCH accepts back as ``expectedEntityVersion``. Returns
    ``None`` for every shape that cannot yield one — an older backend that sends
    no such field, a bodiless 204, a non-JSON or non-object body, a value that is
    not a number. Reading a version is never worth failing an outcome call that
    the backend already accepted, so this raises nothing.

    ``0`` is a real version (a freshly created job sits there), so callers must
    test the result against ``None`` rather than for truthiness. What counts as
    a usable version is ``coerce_entity_version``.
    """
    if response is None:
        return None
    try:
        data = response.json()
    except Exception:  # noqa: BLE001 — a bodiless or non-JSON 2xx is expected
        return None
    if not isinstance(data, dict):
        return None
    return coerce_entity_version(data.get("entityVersion"))


def _conflict_entity_version(response: httpx.Response) -> Optional[int]:
    """The version the backend actually holds, read off an amendment 409.

    This is what makes the conflict recoverable: history rows carry a
    ``sequence`` and not a version, and this SDK wraps no job read, so without
    it a caller has nowhere to get the version their retry needs.

    A structured field wins when a backend offers one; today the value exists
    only inside the human-readable message, so that text is parsed as a
    fallback. Returns ``None`` when neither yields a version — the conflict is
    still raised, the caller just has to read the job themselves.
    """
    try:
        data = response.json()
    except Exception:  # noqa: BLE001 — a non-JSON 409 is an unrecognised body
        return None
    if not isinstance(data, dict):
        return None
    structured = coerce_entity_version(
        _conflict_field(data, "currentEntityVersion", "entityVersion")
    )
    if structured is not None:
        return structured
    # "details" mirrors the message on the real backend; check both.
    for text in (data.get("message"), _conflict_field(data, "error"), data.get("error")):
        match = _CURRENT_VERSION_IN_MESSAGE.search(str(text or ""))
        if match:
            return coerce_entity_version(match.group(1))
    return None


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
) -> Optional[httpx.Response]:
    """POST a job's terminal outcome. Returns the response on success.

    The response body carries the job's new ``entityVersion``; pass it through
    ``parse_entity_version`` and hold on to it, because it is what the
    amendment PATCH accepts as ``expectedEntityVersion`` to detect a lost
    update. Returning the response (rather than nothing) is what lets a caller
    do that without a second read of the job.
    """
    safe_id = urllib.parse.quote(agentic_job_id, safe="")
    url = (
        f"{profitstream_base_url.rstrip('/')}"
        f"/profitstream/v2/api/jobs/{safe_id}/outcome"
    )
    params = {"teamId": team_id} if team_id else None
    return post_with_retry(
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


def append_outcome_metrics_request(
    http_client: httpx.Client,
    profitstream_base_url: str,
    agentic_job_id: str,
    entries: Sequence[Mapping[str, Any]],
    *,
    team_id: str,
    api_key: str,
    retry_attempts: int = DEFAULT_RETRY_ATTEMPTS,
    retry_initial_seconds: float = DEFAULT_RETRY_INITIAL_SECONDS,
    retry_max_seconds: float = DEFAULT_RETRY_MAX_SECONDS,
) -> Optional[httpx.Response]:
    """POST declared per-job metric facts to an existing job (append-only).

    ``entries`` is sent as the request body itself — a bare JSON array of
    OutcomeMetricEntry objects, which is what this endpoint declares — with each
    entry forwarded exactly as supplied. ``provenance``, ``recordedBy`` and
    ``source`` have documented server-side defaults (SELF_REPORTED, the calling
    principal, ``api``), so filling any of them in here would misattribute a
    fact.

    The facts have to be declared on the job type's economics contract first:
    the server rejects an entry whose ``key`` the type does not declare as a
    PER_JOB metric, and range-checks ``quality_rate`` to 0..1. Both answer 400,
    which is a caller error and therefore not retried.

    Only 429 is retried, and nothing else — not the gateway 5xx the other
    outcome POSTs retry, and not the 404 the outcome POST retries through the
    Job's asynchronous creation. This endpoint is append-only: a second delivery
    of the same entries is a second fact, or a 409 against the active-fact
    constraint reported for a write that already succeeded. 429 is the one
    status that proves the request was rejected before the origin processed it,
    so it is the one that can be repeated safely; a 502/503/504 is ambiguous and
    surfaces to the caller, who can check the recorded facts before deciding to
    resend. The 404 goes with them because a fact is appended to a job that
    exists — a caller who is racing ingestion is better served by an immediate
    404 than by minutes of blocking retry. (``amend_outcome_request`` draws the
    same line for the same reason; the ``job_type_economics`` helpers on
    ``feature/BACK-2746-job-type-economics`` go further and disable retry
    outright, which also gives up the safe 429.)

    A 409 (a duplicate active fact for the same metric) raises the plain
    transport error: neither ``OutcomeAlreadyReportedError`` nor
    ``OutcomeAmendConflictError`` describes it, and borrowing one would hand the
    caller the wrong recovery advice.

    Returns the response so the caller can read an ``entityVersion`` off it via
    ``parse_entity_version``. The current backend answers a bodiless 201, so in
    practice there is none — the response is returned anyway rather than
    discarded, so a backend that starts sending one needs no change here.
    """
    safe_id = urllib.parse.quote(agentic_job_id, safe="")
    url = (
        f"{profitstream_base_url.rstrip('/')}"
        f"/profitstream/v2/api/jobs/{safe_id}/outcome/metrics"
    )
    params = {"teamId": team_id} if team_id else None
    return post_with_retry(
        http_client,
        url,
        params=params,
        # The array is the body. list() only fixes the sequence type for the
        # JSON encoder; the entries themselves are untouched.
        body=list(entries),
        api_key=api_key,
        retry_statuses=_NON_IDEMPOTENT_RETRY_STATUSES,
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

    ``payload`` may carry ``expectedEntityVersion``: the backend compares it
    with the job's current ``entityVersion`` and answers 409 instead of
    overwriting a change the caller has not seen. Omit the key to keep the old
    last-write-wins behavior.

    Raises OutcomeNotReportedError on 422 (no outcome yet) and
    OutcomeAmendConflictError on 409 (optimistic-lock conflict). Recovery is
    the caller's: the raised exception carries
    ``current_entity_version`` — the version the backend actually holds, read
    off the conflict body — so the loop is re-check the outcome you intend to
    write (``get_outcome_history`` shows what the other writer changed), then
    re-issue the amendment with that version as ``expectedEntityVersion``. When
    the conflict body carries no version, it takes a job read outside this SDK
    (``GET /v2/api/jobs/{agenticJobId}``), since history rows carry a
    ``sequence`` and not an entity version. The SDK never auto-retries an
    amendment (see below), so the raised message names that loop too.

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
            raise OutcomeAmendConflictError(
                f"{_error_message(response, 'Concurrent amendment conflict')} "
                f"{_AMEND_CONFLICT_RECOVERY}",
                current_entity_version=_conflict_entity_version(response),
            )
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
