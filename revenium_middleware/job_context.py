"""Public job-context surface for agentic workflows (BACK-777 Phase 2).

``JobContext`` scopes all AI completions inside its block to one agentic job
(via the ``_core.context`` contextvar that every provider middleware already
reads) and reports the job's business outcome to Revenium.

Nesting semantics are replace-not-merge: a nested ``JobContext`` is a
different job and does not inherit the outer job's name/type/version; exiting
the inner context restores the outer job's fields via token reset.

Outcome reporting requires a write-scope API key (``rev_sk_``) — resolution:
explicit ``api_key`` > ``REVENIUM_OUTCOME_API_KEY`` > ``REVENIUM_METERING_API_KEY``,
failing fast on a metering key (``rev_mk_``).
"""

import json
import logging
import os
import threading
from typing import Any, Callable, Dict, Optional

import httpx

# 3.8 support: asyncio.to_thread only exists on Python 3.9+, so use the backport.
from ._metering._utils._sync import _asyncio_to_thread
from ._core.config import Config
from ._core.context import _agentic_job_context, set_agentic_job_fields
from ._core.exceptions import OutcomeAlreadyReportedError, OutcomeReportingError
from ._core.outcomes import (
    amend_outcome_request,
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
    ) -> None:
        """Report the job's terminal business outcome.

        ``outcome_reason`` (wire field ``outcomeReason``) is the prescribed field
        for the business explanation of a FAILED or CANCELLED job — pass it here
        instead of encoding the reason inside ``metadata``. The ingest API caps
        it at 2048 characters and rejects longer values with a 400.

        Raises:
            ValueError: invalid ``execution_status``, ``outcome_value`` without
                ``outcome_type``, or a metering (``rev_mk_``) API key.
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
        )

        try:
            report_outcome_request(
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
        self._outcome_reported = True

    def amend_outcome(
        self,
        reason: str,
        *,
        execution_status: Optional[str] = None,
        outcome_type: Optional[str] = None,
        outcome_value: Optional[float] = None,
        outcome_currency: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        reported_by: Optional[str] = None,
        outcome_reason: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Amend a previously-reported outcome (PATCH; outcomes are amendable).

        Returns the parsed job resource as a dict (the backend's JobResource
        JSON; this SDK does not define a typed JobResource class).

        ``outcome_reason`` (wire field ``outcomeReason``) is the prescribed field
        for the outcome's business failure explanation — not to be confused with
        ``reason``, which is this amendment's audit justification. Omit it to
        leave the stored value untouched; pass an empty string to clear it. The
        ingest API caps it at 2048 characters and rejects longer values with a
        400.

        Raises:
            ValueError: blank ``reason``, invalid ``execution_status``, or a
                metering (``rev_mk_``) API key.
            OutcomeReportingError: no API key, or team_id unresolvable.
            OutcomeNotReportedError: the job has no outcome yet (backend 422).
            OutcomeAmendConflictError: concurrent amendment (backend 409) —
                refetch with ``get_outcome_history`` and retry; the SDK does
                not auto-retry.
        """
        # Any explicit attempt means the user owns the outcome; auto-FAILED
        # must not second-guess it — set before validation on purpose.
        self._outcome_attempted = True
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("reason must be a non-blank string")
        if execution_status is not None and execution_status not in _VALID_EXECUTION_STATUSES:
            raise ValueError(
                f"execution_status must be one of {sorted(_VALID_EXECUTION_STATUSES)}, "
                f"got {execution_status!r}"
            )

        api_key = self._resolve_api_key()
        team_id = self._resolve_team_id(api_key)

        payload: Dict[str, Any] = {"reason": reason}
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
        )

        response = amend_outcome_request(
            self._http(),
            self._base_url,
            self.job_id,
            payload,
            team_id=team_id,
            api_key=api_key,
            **self._retry_kwargs,
        )
        # A successful amendment proves an outcome exists — never auto-FAILED on exit.
        self._outcome_reported = True
        try:
            return response.json()
        except Exception:  # noqa: BLE001 — 2xx with a non-JSON body
            return {}

    # ------------------------------------------------------------------ internals

    @staticmethod
    def _add_optional_outcome_fields(
        payload: Dict[str, Any],
        *,
        outcome_type: Optional[str],
        outcome_value: Optional[float],
        outcome_currency: Optional[str],
        metadata: Optional[Dict[str, Any]],
        reported_by: Optional[str],
        outcome_reason: Optional[str],
    ) -> None:
        """Add the optional outcome fields the POST and PATCH bodies share.

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

    def _http(self) -> httpx.Client:
        if self._http_client is None:
            self._http_client = httpx.Client(timeout=60.0)
        return self._http_client

    def _resolve_api_key(self) -> str:
        key = (
            self._api_key
            or os.getenv(Config.ENV_REVENIUM_OUTCOME_API_KEY)
            or os.getenv(Config.ENV_REVENIUM_API_KEY)
        )
        if not key:
            raise OutcomeReportingError(
                "No API key available for outcome reporting: pass api_key= or set "
                "REVENIUM_OUTCOME_API_KEY (write-scope rev_sk_ key required)."
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
