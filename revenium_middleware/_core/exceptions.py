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
