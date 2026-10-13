"""Keep a usage record that one optional field would otherwise lose.

Revenium refuses a whole completion record with a 400 when one optional field
breaks its rule, and the SDK never resends a 400, so the call's tokens and cost
are lost. Each value that breaks its rule is left off the record with a WARNING
instead; the rules mirror the metering API's own checks.

Revenium also answers 201 for a record whose text tag is longer than its
column, then never stores it. Each such tag is cut to the stored length with a
WARNING.
"""

import logging
import math
import re
from typing import Any, Callable, Dict, Mapping

logger = logging.getLogger("revenium_middleware")

__all__ = ["STORED_LENGTHS", "guard_wire_fields", "stored_subscriber_value"]

_SLUG = re.compile(r"[A-Za-z0-9_-]+")
_SOURCE_SLUG = re.compile(r"[A-Za-z0-9._:-]+")
_SURFACE_SLUG = re.compile(r"[A-Za-z0-9._-]+")
_DECIMAL = re.compile(r"-?[0-9]+(\.[0-9]+)?")
_WHOLE_NUMBER = re.compile(r"-?[0-9]+")
_SLUG_CHARACTERS = "letters, digits, hyphens or underscores"
_SOURCE_CHARACTERS = "letters, digits, dots, colons, hyphens or underscores"
_SURFACE_CHARACTERS = "letters, digits, dots, hyphens or underscores"
_MAX_PROMPT_LENGTH = 2147483647
# Stored as numeric(5,2): 999.995 rounds up to 1000.00, which overflows the column.
_QUALITY_SCORE_LIMIT = 999.995
_PRICING_TIERS = ("STANDARD", "BATCH")
STORED_LENGTHS = {
    "transaction_name": 255,
    "task_type": 255,
    "agent": 255,
    "subscription_id": 255,
    "parent_transaction_id": 255,
    "environment": 255,
    "region": 255,
    "trace_id": 255,
    "operation_subtype": 255,
    "error_reason": 255,
    # Revenium also creates "Rating Aggregation for Product <name>" in a
    # 255-character column, so 31 characters of the name's room are taken.
    "product_name": 224,
}
_SUBSCRIBER_STORED_LENGTHS = {"id": 255, "email": 255}
_DROPPED_MESSAGE = "Usage field %s%s left off the record: it %s. The call is still recorded."
_CUT_MESSAGE = "Usage field %s%s is %d characters; Revenium stores %d, so it was shortened."


class _Malformed(ValueError):
    """The value breaks its field's rule; the message names the rule."""


def _slug(pattern: "re.Pattern[str]", max_length: int, characters: str) -> Callable[[Any], Any]:
    def check(value: Any) -> Any:
        if isinstance(value, int) and not isinstance(value, bool):
            text = str(value)
        elif isinstance(value, str):
            text = value
        else:
            raise _Malformed("must be text")
        if len(text) > max_length or not pattern.fullmatch(text):
            raise _Malformed(f"must be 1 to {max_length} {characters}")
        return value

    return check


def _as_number(value: Any) -> float:
    if isinstance(value, bool):
        raise _Malformed("must be a number")
    if isinstance(value, (int, float)):
        number = value
    elif isinstance(value, str) and _DECIMAL.fullmatch(value):
        number = float(value)
    else:
        raise _Malformed("must be a number")
    if isinstance(number, float) and not math.isfinite(number):
        raise _Malformed("must be a finite number")
    return number


def _finite_number(value: Any) -> Any:
    _as_number(value)
    return value


def _prompt_length(value: Any) -> Any:
    number = _as_number(value)
    fractional = isinstance(number, float) and not number.is_integer()
    if fractional or (isinstance(value, str) and not _WHOLE_NUMBER.fullmatch(value)):
        raise _Malformed("must be a whole number")
    if not 0 <= number <= _MAX_PROMPT_LENGTH:
        raise _Malformed(f"must be between 0 and {_MAX_PROMPT_LENGTH}")
    return value


def _quality_score(value: Any) -> Any:
    if not abs(_as_number(value)) < _QUALITY_SCORE_LIMIT:
        raise _Malformed("must be a number between -999.99 and 999.99")
    return value


def _pricing_tier(value: Any) -> Any:
    if isinstance(value, str) and value.upper() in _PRICING_TIERS:
        return value.upper()
    raise _Malformed("must be STANDARD or BATCH")


def _scalar(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple, set, frozenset)):
        raise _Malformed("must be text, not a collection")
    return value


_SHARED_RULES: Dict[str, Callable[[Any], Any]] = {
    "pricing_tier": _pricing_tier,
    **{field: _scalar for field in STORED_LENGTHS},
}
# Audio sends its own ``speed``, the text-to-speech rate as a number, so the
# completion-only rules never apply to another operation.
_RULES_BY_OPERATION: Dict[str, Dict[str, Callable[[Any], Any]]] = {
    "completion": {
        **_SHARED_RULES,
        "effort": _slug(_SLUG, 16, _SLUG_CHARACTERS),
        "speed": _slug(_SLUG, 16, _SLUG_CHARACTERS),
        "query_source": _slug(_SOURCE_SLUG, 128, _SOURCE_CHARACTERS),
        "subagent_type": _slug(_SOURCE_SLUG, 128, _SOURCE_CHARACTERS),
        "client_surface": _slug(_SURFACE_SLUG, 64, _SURFACE_CHARACTERS),
        "prompt_id": _slug(_SLUG, 64, _SLUG_CHARACTERS),
        "prompt_length": _prompt_length,
        "response_quality_score": _quality_score,
        "cost_multiplier": _finite_number,
    },
}
_SUBSCRIBER_RULES: Dict[str, Callable[[Any], Any]] = {field: _scalar for field in _SUBSCRIBER_STORED_LENGTHS}


def _drop_malformed(record: Mapping[str, Any], rules: Mapping[str, Callable[[Any], Any]], prefix: str) -> Dict[str, Any]:
    checked = dict(record)
    for field, rule in rules.items():
        if checked.get(field) is None:
            continue
        try:
            checked[field] = rule(checked[field])
        except _Malformed as problem:
            del checked[field]
            logger.warning(_DROPPED_MESSAGE, prefix, field, problem)
    return checked


def stored_subscriber_value(field: str, value: Any) -> Any:
    """``value`` as Revenium stores a subscriber ``field`` (``id`` or ``email``), without logging."""
    limit = _SUBSCRIBER_STORED_LENGTHS[field]
    if isinstance(value, str) and len(value) > limit:
        return value[:limit]
    return value


def _cut_overlong(record: Mapping[str, Any], lengths: Mapping[str, int], prefix: str) -> Dict[str, Any]:
    cut = dict(record)
    for field, limit in lengths.items():
        value = cut.get(field)
        if isinstance(value, str) and len(value) > limit:
            cut[field] = value[:limit]
            logger.warning(_CUT_MESSAGE, prefix, field, len(value), limit)
    return cut


_Pass = Callable[[Mapping[str, Any], Mapping[str, Any], str], Dict[str, Any]]


def _guard_pass(guard: _Pass, args: Mapping[str, Any], fields: Mapping[str, Any],
                subscriber_fields: Mapping[str, Any]) -> Dict[str, Any]:
    """Run one pass over the record and its nested subscriber, copying both."""
    checked = guard(args, fields, "")
    subscriber = checked.get("subscriber")
    if isinstance(subscriber, dict):
        checked["subscriber"] = guard(subscriber, subscriber_fields, "subscriber.")
    return checked


def guard_wire_fields(args: Mapping[str, Any], operation: str) -> Dict[str, Any]:
    """A copy of one ``operation`` record's keyword arguments that Revenium will accept.

    Each optional value that breaks its metering API rule is removed, with one
    WARNING naming the field and the rule (never the value). A lower-case
    pricing tier is sent upper-case. A text tag longer than Revenium stores is
    then cut to that length, with one WARNING naming the field and the lengths.
    Absent fields and None values are left as they are, and neither ``args``
    nor its nested subscriber is ever changed.
    """
    rules = _RULES_BY_OPERATION.get(operation, _SHARED_RULES)
    kept = _guard_pass(_drop_malformed, args, rules, _SUBSCRIBER_RULES)
    return _guard_pass(_cut_overlong, kept, STORED_LENGTHS, _SUBSCRIBER_STORED_LENGTHS)
