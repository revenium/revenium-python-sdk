"""Public job-context surface for agentic workflows.

``JobContext`` scopes all AI completions inside its block to one agentic job
(via the ``_core.context`` contextvar that every provider middleware already
reads) and reports the job's business outcome to Revenium.

Nesting semantics are replace-not-merge: a nested ``JobContext`` is a
different job and does not inherit the outer job's name/type/version; exiting
the inner context restores the outer job's fields via token reset.

Outcome reporting requires a write-scope API key (``rev_sk_``) — resolution:
explicit ``api_key`` > ``REVENIUM_WRITE_API_KEY`` >
``REVENIUM_OUTCOME_API_KEY`` (deprecated fallback) >
``REVENIUM_METERING_API_KEY``, failing fast on a metering key (``rev_mk_``).
"""

import json
import logging
import os
import threading
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

import httpx

# 3.8 support: asyncio.to_thread only exists on Python 3.9+, so use the backport.
from ._metering._utils._sync import _asyncio_to_thread
from ._core.config import Config, resolve_write_api_key
from ._core.context import _agentic_job_context, set_agentic_job_fields
from ._core.exceptions import (
    OutcomeAlreadyReportedError,
    OutcomeAmendConflictError,
    OutcomeReportingError,
)
from ._core.outcomes import (
    amend_outcome_request,
    append_outcome_metrics_request,
    coerce_entity_version,
    normalize_outcome_metrics,
    parse_entity_version,
    report_outcome_request,
    resolve_team_id,
    validate_outcome_key,
)

logger = logging.getLogger(__name__)

# The ingest API caps outcomeReason at 2048 characters (a Jakarta @Size
# constraint on the request schema); anything longer fails the WHOLE request
# with a 400. The auto-FAILED safety net must never trade "outcome recorded"
# for "full exception text", so it truncates to the cap before sending.
_OUTCOME_REASON_MAX_LEN = 2048


_VALID_EXECUTION_STATUSES = frozenset({"SUCCESS", "FAILED", "CANCELLED"})
_DEFAULT_PROFITSTREAM_BASE_URL = "https://api.revenium.io"


def _coerce_expected_entity_version(version: Optional[Any]) -> Optional[int]:
    """Validate a caller-supplied lock version. ``None`` stays ``None`` (no lock).

    Version 0 is real (a freshly created job sits there), so the test is against
    ``None``, never truthiness. What counts as a version is
    ``coerce_entity_version`` — one definition shared with the parse path, so a
    value the SDK would refuse to send is also one it refuses to record. Here a
    rejected value raises rather than vanishing: the caller typed it.
    """
    if version is None:
        return None
    coerced = coerce_entity_version(version)
    if coerced is None:
        raise ValueError(
            "expected_entity_version must be a non-negative integer, got "
            f"{version!r}"
        )
    return coerced


class JobContext:
    """Scope AI calls to an agentic job and report its outcome.

    Example:
        with JobContext(job_id="loan-app-12345", type="loan_processing") as job:
            response = client.chat.completions.create(...)
            job.report_outcome(execution_status="SUCCESS",
                               outcome_type="CONVERTED", outcome_value=500.0)

    On an unhandled exception inside the block, the context auto-reports
    ``execution_status="FAILED"`` (error message and class in metadata) —
    unless an outcome was already reported — and ALWAYS re-raises the
    original exception. Works as a sync or async context manager. This
    auto-report blocks context exit; tune how long it may retry via the
    ``retry_attempts``/``retry_initial_seconds``/``retry_max_seconds`` knobs,
    which apply to every outcome HTTP call this instance makes. ``attach()``
    and ``get_outcome_history()`` accept the same three.

    Nesting is replace-not-merge: a nested ``JobContext`` is a different job
    and does not inherit the outer job's name/type/version; exiting the inner
    context restores the outer job's fields.

    One instance is single-use-at-a-time: concurrent or nested jobs need
    separate ``JobContext`` instances, and a second concurrent ``__enter__``
    on an already-active instance raises ``RuntimeError``.
    """

    def __init__(
        self,
        job_id: str,
        *,
        name: Optional[str] = None,
        type: Optional[str] = None,
        version: Optional[str] = None,
        team_id: Optional[str] = None,
        api_key: Optional[str] = None,
        profitstream_base_url: Optional[str] = None,
        http_client: Optional[httpx.Client] = None,
        retry_attempts: Optional[int] = None,
        retry_initial_seconds: Optional[float] = None,
        retry_max_seconds: Optional[float] = None,
    ) -> None:
        if not isinstance(job_id, str) or not job_id.strip():
            raise ValueError("job_id must be a non-empty string")
        self.job_id = job_id
        self.name = name
        self.type = type
        self.version = version
        self._team_id = team_id or ""
        self._api_key = api_key
        self._base_url = (
            profitstream_base_url
            or os.getenv(Config.ENV_REVENIUM_PROFITSTREAM_BASE_URL)
            or _DEFAULT_PROFITSTREAM_BASE_URL
        )
        self._http_client = http_client
        # The async auto-report hands the client (and its ownership) to a worker
        # thread, so both are per-run state. Record the original intent once and
        # restore from it in __enter__ — otherwise a re-entered instance either
        # leaks an SDK-created client or silently stops using the injected one.
        self._client_is_user_supplied = http_client is not None
        self._user_http_client = http_client
        self._owns_http_client = not self._client_is_user_supplied
        self._retry_kwargs: Dict[str, Any] = {}
        if retry_attempts is not None:
            self._retry_kwargs["retry_attempts"] = retry_attempts
        if retry_initial_seconds is not None:
            self._retry_kwargs["retry_initial_seconds"] = retry_initial_seconds
        if retry_max_seconds is not None:
            self._retry_kwargs["retry_max_seconds"] = retry_max_seconds
        # Guards the re-entry check-and-set so two threads cannot both enter
        # the same instance and clobber each other's contextvar token.
        self._enter_lock = threading.Lock()
        self._token = None
        self._outcome_reported = False
        self._outcome_attempted = False
        self._resolved_team_id: Optional[str] = None
        # Optimistic-lock token from the last report/amend response; see
        # ``entity_version``. ``None`` means "no version known", which is
        # distinct from version 0 (a freshly created job).
        self._entity_version: Optional[int] = None

    @property
    def entity_version(self) -> Optional[int]:
        """The job's ``entityVersion`` as of this instance's last outcome call.

        ``None`` until this instance reports or amends an outcome, and on a
        backend old enough not to return the field. ``amend_outcome`` sends it
        as ``expectedEntityVersion`` by default, so a concurrent writer that
        changed the outcome in between is reported as
        ``OutcomeAmendConflictError`` instead of being silently overwritten.

        Every outcome call replaces this, never merely refreshes it: a response
        without a version drops it back to ``None`` (unlocked) rather than
        leaving behind a token the completed call has already invalidated, and a
        conflict replaces it with the version the backend reported holding.
        """
        return self._entity_version

    @classmethod
    def attach(
        cls,
        job_id: str,
        *,
        team_id: Optional[str] = None,
        api_key: Optional[str] = None,
        profitstream_base_url: Optional[str] = None,
        http_client: Optional[httpx.Client] = None,
        retry_attempts: Optional[int] = None,
        retry_initial_seconds: Optional[float] = None,
        retry_max_seconds: Optional[float] = None,
    ) -> "JobContext":
        """Attach to an existing job for outcome/amendment calls.

        Returns a handle that is NOT entered as a context manager — it does
        not touch the job-field contextvar. Use it to report or amend an
        outcome from a different process than the one that ran the job.

        Accepts the same retry knobs as the constructor, so a handle can be
        tuned down from the default (bounded) schedule.
        """
        return cls(
            job_id,
            team_id=team_id,
            api_key=api_key,
            profitstream_base_url=profitstream_base_url,
            http_client=http_client,
            retry_attempts=retry_attempts,
            retry_initial_seconds=retry_initial_seconds,
            retry_max_seconds=retry_max_seconds,
        )

    # ------------------------------------------------------------------ context management

    def __enter__(self) -> "JobContext":
        # Lifecycle invariant: the instance is "active" from the moment __enter__
        # takes this lock until exit cleanup releases the token under it again.
        # ``_token``, ``_http_client``, ``_owns_http_client`` and the outcome flags
        # belong to the run that holds that window and must not be mutated from
        # outside it — that is why the exit paths clear the token last, under this
        # same lock, and why the async auto-report worker is handed copies instead
        # of the instance.
        with self._enter_lock:
            if self._token is not None:
                raise RuntimeError(
                    "This JobContext is already active; use a separate JobContext "
                    "instance for nested or concurrent jobs"
                )
            self._outcome_reported = False
            self._outcome_attempted = False
            self._resolved_team_id = None
            # A new run is a new outcome: a version carried over from the last
            # one would lock this run's amendment against a stale value.
            self._entity_version = None
            self._owns_http_client = not self._client_is_user_supplied
            if self._client_is_user_supplied:
                self._http_client = self._user_http_client
            self._token = set_agentic_job_fields(
                job_id=self.job_id, name=self.name, type=self.type, version=self.version
            )
        return self

    def _should_auto_report(self, exc) -> bool:
        return (
            exc is not None
            and isinstance(exc, Exception)
            and not self._outcome_reported
            and not self._outcome_attempted
        )

    def _reset_token(self) -> None:
        if self._token is not None:
            _agentic_job_context.reset(self._token)
            self._token = None

    def __exit__(self, exc_type, exc, tb) -> bool:
        try:
            if self._should_auto_report(exc):
                self._auto_report_failed(exc)
        finally:
            # The token is what __enter__ gates on, so release it last and do it
            # atomically with the cleanup: clearing it first would let another
            # thread enter mid-teardown and have its client closed by this run.
            # Held for microseconds — close() is local teardown, never a request.
            with self._enter_lock:
                try:
                    self.close()
                finally:
                    # Never leave the instance permanently "active" if close raises.
                    self._reset_token()
        return False

    async def __aenter__(self) -> "JobContext":
        return self.__enter__()

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        try:
            report = (
                self._prepare_auto_report(exc) if self._should_auto_report(exc) else None
            )
            if report is None:
                self.close()
            else:
                # Auto-report is blocking HTTP (+ retry sleeps), so it runs in a
                # worker thread. Cancelling this await does not stop a thread that
                # already started, and the token reset below lets the instance be
                # re-entered immediately — so the worker is fully self-contained:
                # it reads and writes no instance state and closes the client
                # itself.
                #
                # Only an SDK-created client changes hands: ownership moves to the
                # worker so our cleanup cannot close it mid-request, and __enter__
                # restores ours on the next run. A caller-supplied client is left
                # attached — close() already leaves those alone.
                #
                # Accepted residual: if the cancellation lands before the executor
                # ever invokes the callable, the report is skipped and a
                # transferred client is left for the garbage collector. There is
                # no reliable signal for that case (the awaited future reports
                # cancelled() either way, whether or not the callable will still
                # run), and closing it from here would risk pulling the client out
                # from under a worker that is about to POST — dropping the FAILED
                # outcome, which is the worse failure.
                if self._owns_http_client:
                    self._http_client = None
                    self._owns_http_client = False
                await _asyncio_to_thread(report)
        finally:
            # Cleanup above already ran while the token still marked the instance
            # active, so a concurrent __enter__ was rejected throughout. Release
            # the token under the same lock the guard uses, so no entry can ever
            # observe a half-finished teardown from either exit path. This is a
            # microsecond-scale acquisition, not a meaningful event-loop block.
            with self._enter_lock:
                self._reset_token()
        return False

    def close(self) -> None:
        if self._owns_http_client and self._http_client is not None:
            self._http_client.close()
            self._http_client = None

    # ------------------------------------------------------------------ outcome reporting

    def report_outcome(
        self,
        execution_status: str,
        *,
        outcome_type: Optional[str] = None,
        outcome_value: Optional[float] = None,
        outcome_currency: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        reported_by: Optional[str] = None,
        outcome_reason: Optional[str] = None,
        metrics: Optional[Sequence[Mapping[str, Any]]] = None,
    ) -> None:
        """Report the job's terminal business outcome.

        ``outcome_reason`` (wire field ``outcomeReason``) is the prescribed field
        for the business explanation of a FAILED or CANCELLED job — pass it here
        instead of encoding the reason inside ``metadata``. The ingest API caps
        it at 2048 characters and rejects longer values with a 400.

        ``metrics`` records declared per-job metric facts alongside the outcome
        — ``[{"key": "quality_rate", "value": 0.93}, ...]``, each entry
        optionally carrying ``provenance`` (MEASURED | SELF_REPORTED | DERIVED |
        ATTESTED), ``recordedBy``, ``source``, ``reason`` and ``recordedAt``.
        Entries are forwarded exactly as supplied, so the three fields with
        server-side defaults (SELF_REPORTED, the calling principal, ``api``) are
        left to the server rather than misattributed here. A metric must already
        be declared as PER_JOB on the job type's economics contract (BACK-3078)
        or the server rejects the whole request, and ``quality_rate`` is
        range-checked to 0..1 server-side. Facts are append-only; use
        ``append_outcome_metrics`` to add more after the outcome is reported.

        Records the job's ``entityVersion`` from the response on this instance
        (see ``entity_version``), so a later ``amend_outcome`` on the same
        instance can detect a lost update. A backend that returns no version
        leaves it unset — that amendment is simply unlocked, as before — and
        everything else works unchanged.

        Raises:
            ValueError: invalid ``execution_status``, ``outcome_value`` without
                ``outcome_type``, a malformed ``metrics`` entry, or a metering
                (``rev_mk_``) API key.
            OutcomeReportingError: no API key available, or team_id unresolvable.
            OutcomeAlreadyReportedError: the job already has an outcome
                (backend 409 with amendment guidance).
        """
        # Any explicit attempt means the user owns the outcome; auto-FAILED
        # must not second-guess it — set before validation on purpose.
        self._outcome_attempted = True
        if execution_status not in _VALID_EXECUTION_STATUSES:
            raise ValueError(
                f"execution_status must be one of {sorted(_VALID_EXECUTION_STATUSES)}, "
                f"got {execution_status!r}"
            )
        if outcome_value is not None and outcome_type is None:
            raise ValueError("outcome_type is required when outcome_value is provided")
        # Shape-checked before the key/team resolution below, because team
        # resolution can itself issue a request.
        normalized_metrics = normalize_outcome_metrics(metrics)

        api_key = self._resolve_api_key()
        team_id = self._resolve_team_id(api_key)

        payload: Dict[str, Any] = {"executionStatus": execution_status}
        self._add_optional_outcome_fields(
            payload,
            outcome_type=outcome_type,
            outcome_value=outcome_value,
            outcome_currency=outcome_currency,
            metadata=metadata,
            reported_by=reported_by,
            outcome_reason=outcome_reason,
            metrics=normalized_metrics,
        )

        try:
            response = report_outcome_request(
                self._http(),
                self._base_url,
                self.job_id,
                payload,
                team_id=team_id,
                api_key=api_key,
                **self._retry_kwargs,
            )
        except OutcomeAlreadyReportedError:
            # The backend just proved an outcome exists — never auto-FAILED on exit.
            self._outcome_reported = True
            raise
        self._record_entity_version(response)
        self._outcome_reported = True

    def amend_outcome(
        self,
        reason: Optional[str] = None,
        *,
        expected_entity_version: Optional[int] = None,
        execution_status: Optional[str] = None,
        outcome_type: Optional[str] = None,
        outcome_value: Optional[float] = None,
        outcome_currency: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        reported_by: Optional[str] = None,
        outcome_reason: Optional[str] = None,
        metrics: Optional[Sequence[Mapping[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Amend a previously-reported outcome (PATCH; outcomes are amendable).

        Returns the parsed job resource as a dict (the backend's JobResource
        JSON; this SDK does not define a typed JobResource class).

        ``reason`` is this amendment's audit justification and stays the first
        positional argument. Omit it only as an API-key caller: the backend then
        records an automated correction reason derived from the source. A session
        caller must still supply one — the backend rejects a session amendment
        without a reason — and a blank string is rejected here either way.

        ``expected_entity_version`` is the optimistic lock. It defaults to the
        version this instance recorded from its last report or amendment
        (``entity_version``) and is omitted from the request entirely when no
        version is known, which is the pre-lock last-write-wins behavior. To
        amend without locking, use a fresh ``JobContext.attach()`` handle, which
        has recorded nothing. Every outcome call replaces the recorded version
        with the one its response reports, so chained amendments each lock
        against the previous one — and a response that carries no version
        (bodiless, or an older backend) leaves the next amendment unlocked
        rather than locked to a token the backend has already moved past.

        On ``OutcomeAmendConflictError`` another writer changed the outcome
        first. The exception carries ``current_entity_version``, the version the
        backend actually holds, and this handle records it, so recovery is:
        re-check the outcome you intend to write (``get_outcome_history`` shows
        what the other writer changed), then re-issue this call — passing that
        version explicitly, or simply through this same handle. When
        ``current_entity_version`` is ``None`` the conflict body carried no
        version, and it takes a job read outside this SDK
        (``GET /v2/api/jobs/{agenticJobId}``) to obtain one; history rows carry a
        ``sequence``, not an entity version. The SDK never auto-retries an
        amendment — PATCH-amend is not idempotent, so a blind retry can append a
        duplicate history row.

        ``outcome_reason`` (wire field ``outcomeReason``) is the prescribed field
        for the outcome's business failure explanation — not to be confused with
        ``reason``, which is this amendment's audit justification. Omit it to
        leave the stored value untouched; pass an empty string to clear it. The
        ingest API caps it at 2048 characters and rejects longer values with a
        400.

        ``metrics`` appends declared per-job metric facts as part of this
        amendment, with the same entry shape and the same server-side rules as
        ``report_outcome``: the metric must be declared PER_JOB on the job
        type's economics contract (BACK-3078), ``quality_rate`` is range-checked
        to 0..1 server-side, and the facts are append-only — amending never
        replaces a fact recorded earlier.

        Raises:
            ValueError: blank ``reason``, invalid ``execution_status``, a
                malformed ``metrics`` entry, or a metering (``rev_mk_``) API key.
            OutcomeReportingError: no API key, or team_id unresolvable.
            OutcomeNotReportedError: the job has no outcome yet (backend 422).
            OutcomeAmendConflictError: concurrent amendment or a stale
                ``expected_entity_version`` (backend 409) — refetch, take the
                current version, and retry with it; the SDK does not auto-retry.
        """
        # Any explicit attempt means the user owns the outcome; auto-FAILED
        # must not second-guess it — set before validation on purpose.
        self._outcome_attempted = True
        if reason is not None and (not isinstance(reason, str) or not reason.strip()):
            # Omitting the key is the deliberate way to let the backend fill in
            # an automated reason; a blank string is a caller bug, and a session
            # caller would get a 422 for it anyway.
            raise ValueError("reason must be a non-blank string, or omitted entirely")
        if execution_status is not None and execution_status not in _VALID_EXECUTION_STATUSES:
            raise ValueError(
                f"execution_status must be one of {sorted(_VALID_EXECUTION_STATUSES)}, "
                f"got {execution_status!r}"
            )
        normalized_metrics = normalize_outcome_metrics(metrics)
        # Resolved and checked before any HTTP: the backend rejects a negative
        # version with a 400, which is a confusing answer to a caller bug.
        version = _coerce_expected_entity_version(
            expected_entity_version
            if expected_entity_version is not None
            else self._entity_version
        )

        api_key = self._resolve_api_key()
        team_id = self._resolve_team_id(api_key)

        payload: Dict[str, Any] = {}
        if reason is not None:
            payload["reason"] = reason
        if version is not None:
            payload["expectedEntityVersion"] = version
        if execution_status is not None:
            payload["executionStatus"] = execution_status
        self._add_optional_outcome_fields(
            payload,
            outcome_type=outcome_type,
            outcome_value=outcome_value,
            outcome_currency=outcome_currency,
            metadata=metadata,
            reported_by=reported_by,
            outcome_reason=outcome_reason,
            metrics=normalized_metrics,
        )

        try:
            response = amend_outcome_request(
                self._http(),
                self._base_url,
                self.job_id,
                payload,
                team_id=team_id,
                api_key=api_key,
                **self._retry_kwargs,
            )
        except OutcomeAmendConflictError as conflict:
            # The conflict reports the version the backend actually holds, so
            # record it (``None`` when it carries none, leaving the retry
            # unlocked): the version we were holding is provably wrong, and a
            # caller who decides the amendment still applies can retry through
            # this same handle instead of threading the value back by hand.
            self._entity_version = conflict.current_entity_version
            raise
        # A successful amendment proves an outcome exists — never auto-FAILED on exit.
        self._outcome_reported = True
        self._record_entity_version(response)
        try:
            return response.json()
        except Exception:  # noqa: BLE001 — 2xx with a non-JSON body
            return {}

    def append_outcome_metrics(self, entries: Sequence[Mapping[str, Any]]) -> None:
        """Append declared per-job metric facts to this job after the fact.

        For facts that are only measurable once the outcome is already
        reported — a graded quality sample, a downstream correction. Each entry
        is ``{"key": ..., "value": ...}`` plus the optional ``provenance``
        (MEASURED | SELF_REPORTED | DERIVED | ATTESTED), ``recordedBy``,
        ``source``, ``reason`` and ``recordedAt``, forwarded exactly as
        supplied so the server's own defaults (SELF_REPORTED, the calling
        principal, ``api``) apply to whatever was omitted.

        The metric has to be declared as PER_JOB on the job type's economics
        contract first (BACK-3078); a fact for an undeclared key is rejected
        with a 400, as is a ``quality_rate`` outside 0..1, which the server
        range-checks. The endpoint is append-only and the SDK does not dedupe:
        the server owns fact identity, and dropping a repeated measurement
        client-side would lose real data.

        Appending facts is not reporting an outcome, so this call leaves the
        outcome state of the context alone — a block that then raises still
        auto-reports FAILED on exit.

        The recorded ``entity_version`` survives this call. The
        "a versionless success clears the token" rule that ``report_outcome``
        and ``amend_outcome`` follow is about calls that mutate the Job row and
        so advance its version; appending facts is not one — it writes only the
        job's metric facts and answers a bodiless 201, leaving the version this
        handle already holds still current. Clearing it would leave a following
        ``amend_outcome`` unlocked and re-open the lost update that lock exists
        to catch. A response that does carry a version is adopted, so a backend
        that starts sending one needs no change here.

        Requires the same write-scope key (``rev_sk_``) and team id as
        ``report_outcome``. Works on a ``JobContext.attach()`` handle, which is
        the usual way to record a fact from a later process.

        Raises:
            ValueError: no entries, a malformed entry, or a metering
                (``rev_mk_``) API key.
            OutcomeReportingError: no API key available, or team_id
                unresolvable.
        """
        # require_entries=True raises for both ``None`` and an empty sequence,
        # so the normalizer cannot answer ``None`` here; ``or []`` only satisfies
        # its Optional return type.
        normalized = normalize_outcome_metrics(entries, require_entries=True) or []
        api_key = self._resolve_api_key()
        team_id = self._resolve_team_id(api_key)
        response = append_outcome_metrics_request(
            self._http(),
            self._base_url,
            self.job_id,
            normalized,
            team_id=team_id,
            api_key=api_key,
            **self._retry_kwargs,
        )
        # Deliberately not ``_record_entity_version``: that one replaces the
        # token unconditionally, which is right for report/amend because those
        # mutate the Job row and advance its version, leaving anything held from
        # before them stale. This call does not touch the Job row (it appends
        # metric facts) and answers a bodiless 201, so the held version is still
        # the job's current one — dropping it would silently unlock the next
        # amendment. Adopt a version only if a response actually carries one.
        version = parse_entity_version(response)
        if version is not None:
            self._entity_version = version

    # ------------------------------------------------------------------ internals

    def _record_entity_version(self, response) -> None:
        """Replace the recorded ``entityVersion`` with the one this response reports.

        Unconditional, including down to ``None``. A successful report or
        amendment advances the job's version, so a version held over from before
        it is guaranteed stale — keeping it (when the response is bodiless, or
        comes from a backend that sends no version) would make the next
        amendment send a token the backend cannot match and turn it into a false
        conflict. Dropping to ``None`` instead leaves the next amendment
        unlocked, which is the behavior on a backend without the field anyway.
        """
        self._entity_version = parse_entity_version(response)

    @staticmethod
    def _add_optional_outcome_fields(
        payload: Dict[str, Any],
        *,
        outcome_type: Optional[str] = None,
        outcome_value: Optional[float] = None,
        outcome_currency: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        reported_by: Optional[str] = None,
        outcome_reason: Optional[str] = None,
        metrics: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Add the optional outcome fields the POST and PATCH bodies share.

        The single place shared outcome fields are assembled — a new field that
        both bodies accept belongs here, not in the two callers. Fields only one
        verb accepts (``reason`` and ``expectedEntityVersion`` on the PATCH) stay
        with that caller.

        Every field is tested with ``is not None`` rather than truthiness: the
        amendment contract distinguishes an omitted key (leave the stored value
        untouched) from an empty string (clear it), and ``0.0`` is a legitimate
        outcome value.
        """
        if outcome_type is not None:
            payload["outcomeType"] = outcome_type
        if outcome_value is not None:
            payload["outcomeValue"] = float(outcome_value)
        if outcome_currency is not None:
            payload["outcomeCurrency"] = outcome_currency
        if metadata is not None:
            payload["metadata"] = json.dumps(metadata)
        if reported_by is not None:
            payload["reportedBy"] = reported_by
        if outcome_reason is not None:
            payload["outcomeReason"] = outcome_reason
        if metrics is not None:
            # Already shape-checked by ``normalize_outcome_metrics``; the
            # entries go out as the caller wrote them.
            payload["metrics"] = metrics

    def _http(self) -> httpx.Client:
        if self._http_client is None:
            self._http_client = httpx.Client(timeout=60.0)
        return self._http_client

    def _resolve_api_key(self) -> str:
        key = resolve_write_api_key(self._api_key)
        if not key:
            raise OutcomeReportingError(
                "No API key available for outcome reporting: pass api_key= or set "
                "REVENIUM_WRITE_API_KEY (REVENIUM_OUTCOME_API_KEY is a deprecated "
                "fallback; write-scope rev_sk_ key required)."
            )
        return validate_outcome_key(key)

    def _resolve_team_id(self, api_key: str) -> str:
        if self._resolved_team_id:
            return self._resolved_team_id
        team_id = resolve_team_id(
            self._team_id, api_key, self._http(), self._base_url, use_env=True
        )
        if not team_id:
            raise OutcomeReportingError(
                "Could not resolve team_id for outcome reporting: pass team_id= "
                "or set REVENIUM_TEAM_ID."
            )
        self._resolved_team_id = team_id
        return team_id

    def _auto_report_failed(self, exc: BaseException) -> None:
        if not isinstance(exc, Exception):
            # KeyboardInterrupt/SystemExit: never delay interpreter shutdown with
            # a blocking outcome POST, and never risk replacing the original.
            return
        try:
            self.report_outcome(
                execution_status="FAILED",
                metadata={"error": str(exc), "errorType": exc.__class__.__name__},
                outcome_reason=str(exc)[:_OUTCOME_REASON_MAX_LEN],
            )
        except Exception as report_exc:  # noqa: BLE001 — must never mask the user's exception
            logger.warning(
                "Auto-report of FAILED outcome for job %s failed: %s",
                self.job_id, report_exc,
            )

    def _prepare_auto_report(self, exc: BaseException) -> Optional[Callable[[], None]]:
        """Build a self-contained auto-FAILED reporter for the async exit path.

        Everything the report needs is captured here, on the caller's context,
        so the returned callable touches no instance state at all: a worker left
        running by a cancellation cannot corrupt a later run of this instance,
        and it closes the transferred client itself.

        Returns ``None`` when there is nothing to run (non-``Exception`` exit, or
        the API key cannot be resolved) — key/config failures stay
        swallowed-and-logged, as on the sync path.
        """
        if not isinstance(exc, Exception):
            # KeyboardInterrupt/SystemExit: never delay interpreter shutdown with
            # a blocking outcome POST, and never risk replacing the original.
            return None
        try:
            api_key = self._resolve_api_key()
        except Exception as prep_exc:  # noqa: BLE001 — best effort, never masks the user's exception
            logger.warning(
                "Auto-report of FAILED outcome for job %s failed: %s",
                self.job_id, prep_exc,
            )
            return None

        client, owned = self._http(), self._owns_http_client
        job_id, base_url = self.job_id, self._base_url
        retry_kwargs = dict(self._retry_kwargs)
        # Already-resolved team wins; otherwise the worker resolves it, off the
        # event loop, because that lookup can itself be a blocking request.
        team_id = self._resolved_team_id or self._team_id
        payload = {
            "executionStatus": "FAILED",
            "metadata": json.dumps(
                {"error": str(exc), "errorType": exc.__class__.__name__}
            ),
            "outcomeReason": str(exc)[:_OUTCOME_REASON_MAX_LEN],
        }

        def _report() -> None:
            try:
                resolved_team_id = resolve_team_id(
                    team_id, api_key, client, base_url, use_env=True
                )
                if not resolved_team_id:
                    raise OutcomeReportingError(
                        "Could not resolve team_id for outcome reporting: pass "
                        "team_id= or set REVENIUM_TEAM_ID."
                    )
                # The response's entityVersion is deliberately dropped here: this
                # worker writes no instance state (see the docstring), and the
                # instance may already have been re-entered for a different run
                # by the time this lands.
                report_outcome_request(
                    client,
                    base_url,
                    job_id,
                    payload,
                    team_id=resolved_team_id,
                    api_key=api_key,
                    **retry_kwargs,
                )
            except Exception as report_exc:  # noqa: BLE001 — must never mask the user's exception
                logger.warning(
                    "Auto-report of FAILED outcome for job %s failed: %s",
                    job_id, report_exc,
                )
            finally:
                if owned:
                    client.close()

        return _report
