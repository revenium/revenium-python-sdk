"""
Core exceptions shared across all Revenium middleware providers.

The unified SDK ships ``ReveniumCostLimitExceeded`` from ``_core`` so every
provider subpackage (openai, anthropic, google, …) raises the same exception
type and downstream callers can ``except ReveniumCostLimitExceeded`` once.
"""

from typing import Optional, Union


class ReveniumCostLimitExceeded(Exception):
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
