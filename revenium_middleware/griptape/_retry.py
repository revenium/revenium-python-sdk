"""Retry settings shared by every Revenium Griptape driver."""
from revenium_middleware._core.exceptions import BudgetExceededError


def stop_retrying_budget_errors(driver) -> None:
    """Make a spending-cap refusal final: a retry would only wait and meet the same cap.

    Keeps every exception type the driver already ignores, Griptape's defaults
    and the caller's own, and adds ``BudgetExceededError`` once.
    """
    if BudgetExceededError not in driver.ignored_exception_types:
        driver.ignored_exception_types = (*driver.ignored_exception_types, BudgetExceededError)
