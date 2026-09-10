"""
Core exceptions shared across all Revenium middleware providers.

The unified SDK ships ``BudgetExceededError`` from ``_core`` so every
provider subpackage (openai, anthropic, google, …) raises the same exception
type and downstream callers can ``except BudgetExceededError`` once.
"""

from typing import Optional, Union


class BudgetExceededError(Exception):
    """Raised when a Revenium enforcement rule blocks the outbound request.

    Inherits directly from ``Exception`` (not from any middleware-error base)
    so per-provider ``handle_exception_safely`` decorators never swallow it —
    enforcement must always reach the caller.
    """

    def __init__(
        self,
        message: str,
        rule_name: Optional[str] = None,
        current_value: Optional[float] = None,
        threshold: Optional[float] = None,
        resets_at: Optional[str] = None,
        rule_id: Optional[Union[str, int]] = None,
    ):
        super().__init__(message)
        self.message = message
        self.rule_name = rule_name
        self.current_value = current_value
        self.threshold = threshold
        self.resets_at = resets_at
        self.rule_id = rule_id


# Deprecated alias preserved for backward compatibility. The exception was
# renamed in v0.1.5 to align with the Go and Node SDKs and the backend
# `BudgetExceededException`. Existing code that does
# `except ReveniumCostLimitExceeded:` continues to catch the new exception
# unchanged. Plan to remove in a future major release.
ReveniumCostLimitExceeded = BudgetExceededError


class OutcomeReportingError(Exception):
    """Raised when an agentic job outcome cannot be reported.

    Configuration failures (unresolvable team_id, missing API key) raise this
    class directly;
    backend-state conditions raise the subclasses.
    """


class OutcomeAlreadyReportedError(OutcomeReportingError):
    """The job already has an outcome (backend 409 with amendment guidance).

    Callers can inspect ``reported_at`` / ``amendment_count`` and decide to
    amend (``amend_outcome``).
    """

    def __init__(
        self,
        message: str,
        reported_at: Optional[str] = None,
        amendment_count: Optional[int] = None,
    ):
        super().__init__(message)
        self.reported_at = reported_at
        self.amendment_count = amendment_count


class OutcomeNotReportedError(OutcomeReportingError):
    """Amendment attempted on a job with no outcome yet (backend 422).

    Report the initial outcome with ``report_outcome`` first.
    """


class OutcomeAmendConflictError(OutcomeReportingError):
    """Concurrent amendment changed the outcome row (backend 409, optimistic lock).

    Raised on any amendment conflict the backend reports, including one on an
    amendment that carried no ``expected_entity_version`` at all — a versionless
    amendment racing another writer still conflicts. When the amendment did
    carry a version, the conflict also means the backend no longer holds it.

    ``current_entity_version`` is the version the backend does hold, read off
    the conflict body, and it is what makes the conflict recoverable: re-check
    the outcome you intend to write (``get_outcome_history`` shows what the
    other writer changed), then re-issue the amendment with
    ``expected_entity_version`` set to it. It is ``None`` when the body carries
    no version, in which case the version has to come from a job read outside
    this SDK (``GET /v2/api/jobs/{agenticJobId}``) — history rows carry a
    ``sequence``, not an entity version. The SDK does not auto-retry:
    PATCH-amend is not idempotent, so a blind retry can append a duplicate
    history row.
    """

    def __init__(self, message: str, current_entity_version: Optional[int] = None):
        super().__init__(message)
        self.current_entity_version = current_entity_version
