"""What the provider wrappers record for a model call the provider refused or broke off."""

import hashlib
import logging
import sys
import uuid
from typing import Any, Dict, NamedTuple, Optional

from revenium_middleware._core.context import _MAX_IDEMPOTENCY_KEY_LENGTH, get_idempotency_key
from revenium_middleware._core.error_text import reportable_exception_text
from revenium_middleware._core.exceptions import BudgetExceededError
from revenium_middleware._core.wire_guards import STORED_LENGTHS

logger = logging.getLogger("revenium_middleware")

ERROR_STOP_REASON = "ERROR"
_MAX_ERROR_REASON_LENGTH = STORED_LENGTHS["error_reason"]
_STATUS_CODE_ATTRIBUTE = object()


class CallFailure(NamedTuple):
    """The error reason, Idempotency-Key and error code of one failed call's record."""

    error_reason: str
    idempotency_key: str
    error_code: Optional[int] = None

    def record_fields(self) -> Dict[str, Any]:
        """The fields this failure adds to its usage record; a call with no
        HTTP error status sends no error code at all."""
        fields: Dict[str, Any] = {"error_reason": self.error_reason}
        if self.error_code is not None:
            fields["error_code"] = self.error_code
        return fields


def _chain(error: Optional[BaseException], excluded=frozenset()):
    """``error`` and every exception it was raised from or while handling."""
    pending = [error]
    seen = set(excluded)
    while pending:
        current = pending.pop()
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))
        yield current
        pending.extend((current.__cause__, current.__context__))


def raised_from_any(error: BaseException, outer: Optional[BaseException], error_types) -> bool:
    """Whether ``error`` or an exception it was raised from is one of
    ``error_types``, leaving out ``outer``, the call's ``exception_in_flight()``."""
    earlier = {id(e) for e in _chain(outer)}
    return any(isinstance(e, error_types) for e in _chain(error, earlier))


def _caused_by_spending_cap(error: BaseException, outer: Optional[BaseException] = None) -> bool:
    return raised_from_any(error, outer, BudgetExceededError)


def is_spending_cap_refusal(error: BaseException) -> bool:
    """Whether a spending cap refused the call that raised ``error``."""
    return _caused_by_spending_cap(error)


def exception_in_flight() -> Optional[BaseException]:
    """The exception the caller is handling as a call starts: a refusal in
    its chain belongs to an earlier call, not to this one."""
    return sys.exc_info()[1]


def _is_http_status(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and 100 <= value <= 599


def _error_reason(error: BaseException, status=_STATUS_CODE_ATTRIBUTE) -> str:
    if status is _STATUS_CODE_ATTRIBUTE:
        status = getattr(error, "status_code", None)
    elif not _is_http_status(status):
        status = None
    suffix = f" (HTTP {status})" if isinstance(status, int) else ""
    return reportable_exception_text(error, _MAX_ERROR_REASON_LENGTH - len(suffix)) + suffix


def _error_code(error: BaseException, status=_STATUS_CODE_ATTRIBUTE) -> Optional[int]:
    """The HTTP error status the provider answered with, or None. A status
    below 400, such as the 200 of a stream that broke off, is no error code."""
    if status is _STATUS_CODE_ATTRIBUTE:
        status = getattr(error, "status_code", None)
    return status if _is_http_status(status) and status >= 400 else None


def _error_idempotency_key() -> str:
    """A key of its own, so a failed call never takes the key of the
    caller's idempotency block that its successful rerun must get."""
    suffix = uuid.uuid4().hex
    block_key = get_idempotency_key()
    if block_key is None:
        return f"err:{suffix}"
    key = f"{block_key}:err:{suffix[:8]}"
    if len(key) <= _MAX_IDEMPOTENCY_KEY_LENGTH:
        return key
    return f"{hashlib.sha256(block_key.encode()).hexdigest()}:err:{suffix[:8]}"


def failure_of(error: BaseException, outer: Optional[BaseException] = None,
               status=_STATUS_CODE_ATTRIBUTE) -> Optional[CallFailure]:
    """The record fields for a call that raised ``error``, or None when a
    spending cap refused the call, which is never recorded as a failure.
    ``outer`` is the call's ``exception_in_flight()``. ``status`` is the HTTP
    status for a provider whose errors do not carry it in ``status_code``;
    a value outside 100 to 599 names no status. An error status is also the
    record's error code."""
    try:
        if _caused_by_spending_cap(error, outer):
            return None
        return CallFailure(_error_reason(error, status), _error_idempotency_key(), _error_code(error, status))
    except Exception as e:
        logger.warning("Could not describe a failed call for its usage record: %s", e)
        return None


def failed_transaction_id() -> str:
    """A fresh id: a shared one would collide in Revenium's transaction dedup."""
    return str(uuid.uuid4())
