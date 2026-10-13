"""Turn a metering record into values JSON can write, so one odd value never costs the record.

The HTTP client writes the body with ``allow_nan=False`` and strict UTF-8, so
a UUID, Decimal, date, bytes, NaN, a self-referencing dict or half an emoji
anywhere in a record raises before anything is sent, and the record is lost.
Imports only the standard library, so the tool-event decorator can import it
at module level.
"""
import datetime
import decimal
import enum
import functools
import logging
import math
import numbers
import re
import uuid
from collections.abc import Mapping
from typing import Any, Dict, FrozenSet

logger = logging.getLogger("revenium_middleware")

_MAX_DEPTH = 32
_MAX_VALUES_CHECKED = 10_000
_SURROGATE = re.compile("[\ud800-\udfff]")
_PLAIN_FIELD_NAME = re.compile("[A-Za-z_]{1,40}")
_HIDDEN_KEY = "<key>"
_REPLACEMENT_CHARACTER = "�"
_DROP = object()
_NO_PLAIN_FORM = object()

_NOT_FINITE = "held a number that is NaN or infinite and was left off"
_HALF_EMOJI = "held text with half of an emoji; the broken character was replaced"
_NOT_UTF8 = "held bytes that are not UTF-8 text; the unreadable bytes were replaced"
_SELF_REFERENCE = "refers to itself and was left off"
_SAME_KEY = "held two keys JSON writes as the same text; the earlier one was left off, as a JSON reader would"
_TOO_DEEP = f"is nested more than {_MAX_DEPTH} levels deep, too deep to check"
_TOO_LARGE = f"holds more than {_MAX_VALUES_CHECKED} values, too many to check"


class _CheckAbandoned(Exception):
    """The record is too big to check; it is sent as given, as it was before cleaning existed."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def _text_without_surrogates(text: str) -> str:
    """Rejoin an emoji split into its two halves; replace a lone half with U+FFFD."""
    if _SURROGATE.search(text) is None:
        return text
    return text.encode("utf-16", "surrogatepass").decode("utf-16", "replace")


@functools.lru_cache(maxsize=256)
def _warn_once(subject: str, path: str, reason: str) -> None:
    logger.warning(
        "%s: field %s %s; the rest of the record is still sent.",
        _text_without_surrogates(subject),
        _text_without_surrogates(path),
        reason,
    )


@functools.lru_cache(maxsize=256)
def _warn_once_sent_as_given(subject: str, reason: str) -> None:
    logger.warning(
        "%s: %s; sending it as given.",
        _text_without_surrogates(subject),
        reason,
    )


def _json_key(key: Any) -> str:
    """The text ``json.dumps`` writes for a key it accepts; any other key as its own text."""
    if isinstance(key, str):
        return key
    if isinstance(key, bool) or key is None:
        return {True: "true", False: "false", None: "null"}[key]
    if isinstance(key, int):
        return int.__repr__(key)
    if isinstance(key, float) and math.isfinite(key):
        return float.__repr__(key)
    return str(key)


def _shown_key(key: str) -> str:
    """A caller's key as it may appear in a log line: a plain field name, else a placeholder.

    Keys can be email addresses or credentials, so anything beyond letters and
    underscores is hidden.
    """
    return key if _PLAIN_FIELD_NAME.fullmatch(key) else _HIDDEN_KEY


def _plain_form(value: Any) -> Any:
    """A pydantic model's JSON form or a numpy-style scalar's plain value; else ``_NO_PLAIN_FORM``."""
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        try:
            dumped = model_dump(mode="json")
        except Exception:
            dumped = None
        if isinstance(dumped, Mapping):
            return dumped
    item = getattr(value, "item", None)
    if callable(item):
        try:
            plain = item()
        except Exception:
            plain = None
        if isinstance(plain, (bool, int, float)):
            return plain
    return _NO_PLAIN_FORM


class _RecordWalk:
    """One pass over a record, building a cleaned copy and never touching the original."""

    def __init__(self, subject: str) -> None:
        self.subject = subject
        self.containers_on_path = set()
        self.values_checked = 0

    def drop(self, path: str, reason: str) -> Any:
        _warn_once(self.subject, path, reason)
        return _DROP

    def clean(self, value: Any, path: str, depth: int) -> Any:
        if value is None or isinstance(value, (bool, int)):
            return value
        if isinstance(value, float):
            return self.finite(value, path)
        if isinstance(value, str):
            return self.text(value, path)
        if isinstance(value, (Mapping, list, tuple, set, frozenset)):
            return self.container(value, path, depth)
        return self.scalar(value, path, depth)

    def finite(self, number: float, path: str) -> Any:
        return number if math.isfinite(number) else self.drop(path, _NOT_FINITE)

    def text(self, text: str, path: str) -> str:
        cleaned = _text_without_surrogates(text)
        if cleaned is not text and cleaned.count(_REPLACEMENT_CHARACTER) > text.count(_REPLACEMENT_CHARACTER):
            _warn_once(self.subject, path, _HALF_EMOJI)
        return cleaned

    def scalar(self, value: Any, path: str, depth: int) -> Any:
        if isinstance(value, decimal.Decimal):
            return self.decimal_number(value, path)
        if isinstance(value, enum.Enum):
            return self.clean(value.value, path, depth + 1)
        if isinstance(value, uuid.UUID):
            return str(value)
        if isinstance(value, (datetime.date, datetime.time)):
            return value.isoformat()
        if isinstance(value, (bytes, bytearray, memoryview)):
            return self.bytes_text(bytes(value), path)
        if isinstance(value, numbers.Integral):
            return int(value)
        if isinstance(value, numbers.Real):
            return self.finite(float(value), path)
        return self.object_text(value, path, depth)

    def decimal_number(self, number: decimal.Decimal, path: str) -> Any:
        if not number.is_finite():
            return self.drop(path, _NOT_FINITE)
        if number == number.to_integral_value():
            return int(number)
        return self.finite(float(number), path)

    def bytes_text(self, raw: bytes, path: str) -> str:
        try:
            return raw.decode("utf-8")
        except UnicodeDecodeError:
            _warn_once(self.subject, path, _NOT_UTF8)
            return raw.decode("utf-8", "replace")

    def object_text(self, value: Any, path: str, depth: int) -> Any:
        """A model or numpy-style scalar by its plain form; anything else by its text."""
        plain = _plain_form(value)
        if plain is not _NO_PLAIN_FORM:
            return self.clean(plain, path, depth + 1)
        type_name = type(value).__name__
        try:
            text = str(value)
        except Exception:
            return self.drop(path, f"held a {type_name} that could not be turned into text and was left off")
        _warn_once(self.subject, path, f"held a {type_name} that is not JSON, so it was sent as its text")
        return self.text(text, path)

    def container(self, value: Any, path: str, depth: int) -> Any:
        if id(value) in self.containers_on_path:
            return self.drop(path, _SELF_REFERENCE)
        if depth >= _MAX_DEPTH:
            raise _CheckAbandoned(_TOO_DEEP)
        self.containers_on_path.add(id(value))
        try:
            if isinstance(value, Mapping):
                return self.mapping(value, path, depth + 1)
            return self.sequence(value, path, depth + 1)
        finally:
            self.containers_on_path.discard(id(value))

    def mapping(self, value: Mapping, path: str, depth: int) -> Dict[str, Any]:
        cleaned = {}
        for key, item in value.items():
            text_key = _json_key(key)
            item_path = f"{path}.{_shown_key(text_key)}"
            cleaned_item = self.nested(item, item_path, depth)
            if cleaned_item is _DROP:
                continue
            cleaned_key = self.text(text_key, item_path)
            if cleaned_key in cleaned:
                _warn_once(self.subject, item_path, _SAME_KEY)
            cleaned[cleaned_key] = cleaned_item
        return cleaned

    def sequence(self, value: Any, path: str, depth: int) -> list:
        item_path = f"{path}[]"
        cleaned = []
        for item in value:
            cleaned_item = self.nested(item, item_path, depth)
            cleaned.append(None if cleaned_item is _DROP else cleaned_item)
        return cleaned

    def nested(self, value: Any, path: str, depth: int) -> Any:
        self.values_checked += 1
        if self.values_checked > _MAX_VALUES_CHECKED:
            raise _CheckAbandoned(_TOO_LARGE)
        return self.clean(value, path, depth)


def json_safe_record(
    record: Mapping, subject: str, verbatim_keys: FrozenSet[str] = frozenset()
) -> Dict[str, Any]:
    """A copy of ``record`` that JSON can write; ``record`` and everything in it are left unchanged.

    A UUID, date or bytes value becomes text, a Decimal or numpy number a plain
    number, a pydantic model its JSON form, a set or tuple a list. A value that cannot be sent (NaN, infinity,
    a container that holds itself) is left off. Keys in ``verbatim_keys`` are
    copied as given. Anything dropped or altered is warned about once per
    ``subject`` and field, never with the value, which can hold a secret. If
    the record is too large or too deeply nested to check, or the check itself
    fails, the record is returned as given, so cleaning never costs a record
    that would have been sent before.
    """
    try:
        walk = _RecordWalk(subject)
        walk.containers_on_path.add(id(record))
        cleaned = {}
        for key, value in record.items():
            if key in verbatim_keys:
                cleaned[key] = value
                continue
            cleaned_value = walk.clean(value, str(key), 0)
            if cleaned_value is not _DROP:
                cleaned[key] = cleaned_value
        return cleaned
    except _CheckAbandoned as abandoned:
        _warn_once_sent_as_given(subject, abandoned.reason)
    except Exception as exc:
        _warn_once_sent_as_given(subject, f"could not be checked for values JSON cannot write ({type(exc).__name__})")
    return dict(record)
