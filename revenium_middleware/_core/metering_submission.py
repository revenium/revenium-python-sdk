"""Centralized submission of AI metering events with Idempotency-Key injection."""
from __future__ import annotations

import logging
import uuid
from typing import Any, Dict, Optional

from revenium_middleware._core.context import (
    IDEMPOTENCY_KEY_RULE,
    _is_sendable_idempotency_key,
    next_idempotency_key,
)
from revenium_middleware._core.delivery_circuit import get_circuit
from revenium_middleware._core.json_safe import json_safe_record
from revenium_middleware._core.metering import current_client
from revenium_middleware._core import metering_buffer
from revenium_middleware._core.metering_status import (
    record_metering_error,
    record_metering_success,
)
from revenium_middleware._core.prompt_extraction import sanitize_credentials
from revenium_middleware._core.wire_guards import guard_wire_fields
from revenium_middleware._metering._types import NotGiven

logger = logging.getLogger("revenium_middleware")


_AI_OPERATIONS = frozenset({"completion", "image", "video", "audio"})
_PROMPT_FIELDS = ("system_prompt", "input_messages", "output_response")
# Client options rather than body fields: an httpx.Timeout must reach the client as itself.
_NOT_BODY_KEYS = frozenset({"extra_headers", "extra_query", "timeout"})
_SHARED_REQUIRED_ARGS = frozenset(
    {"model", "provider", "request_duration", "request_time", "response_time", "transaction_id"}
)
_REQUIRED_ARGS = {
    "completion": _SHARED_REQUIRED_ARGS | {
        "completion_start_time", "cost_type", "input_token_count", "is_streamed",
        "output_token_count", "stop_reason", "total_token_count",
    },
    "image": _SHARED_REQUIRED_ARGS | {"requested_image_count", "actual_image_count"},
    "video": _SHARED_REQUIRED_ARGS | {"duration_seconds"},
    "audio": _SHARED_REQUIRED_ARGS,
}


def _json_safe_args(operation: str, args: Dict[str, Any]) -> Dict[str, Any]:
    """A copy of ``args`` JSON can write, keeping every argument ``create_<operation>`` requires.

    A required argument the cleaner would leave off is sent as given instead:
    dropping it would fail the call on a missing argument rather than on its value.
    """
    verbatim = _NOT_BODY_KEYS | {key for key, value in args.items() if isinstance(value, NotGiven)}
    cleaned = json_safe_record(args, "Usage record", verbatim)
    for key in _REQUIRED_ARGS[operation]:
        if key in args and key not in cleaned:
            cleaned[key] = args[key]
    return cleaned


def _mask_prompt_fields(merged_args: Dict[str, Any]) -> None:
    """Mask credentials in the captured prompt fields of a record about to be sent.

    If masking fails, the prompt fields are dropped rather than sent unmasked,
    and the usage record still goes out.
    """
    try:
        masked = {
            field: sanitize_credentials(merged_args[field])
            for field in _PROMPT_FIELDS
            if isinstance(merged_args.get(field), str)
        }
    except Exception:
        logger.warning("Could not mask credentials in captured prompt text; sending the record without it")
        for field in _PROMPT_FIELDS:
            merged_args.pop(field, None)
        return
    merged_args.update(masked)


def _accepted_fields(operation: str, args: Dict[str, Any]) -> Dict[str, Any]:
    """The record's fields without values Revenium would refuse the whole record for.

    If the check itself fails, the fields are sent as given rather than losing
    the record.
    """
    try:
        return guard_wire_fields(args, operation)
    except Exception:
        logger.warning("Could not check usage fields; sending them as given")
        return args


def _replay_payload(operation: str, merged_args: Dict[str, Any], client: Any) -> Dict[str, Any]:
    return {
        "operation": operation,
        "args": merged_args,
        "api_key": client.api_key,
        "base_url": str(client.base_url),
    }


def _choose_idempotency_key(explicit_key: Optional[str]) -> str:
    """The explicit key, else the current block's next key, else a UUID v4; a key the header cannot carry becomes a UUID v4."""
    if explicit_key is not None:
        if explicit_key == "":
            raise ValueError(
                "idempotency_key must be a non-empty string; "
                "the backend rejects empty keys"
            )
        key = explicit_key
    else:
        ctx_key = next_idempotency_key()
        key = ctx_key if ctx_key is not None else str(uuid.uuid4())
    if _is_sendable_idempotency_key(key):
        return key
    logger.warning(
        "Idempotency key %s; sending this usage record under a generated key instead",
        IDEMPOTENCY_KEY_RULE,
    )
    return str(uuid.uuid4())


def submit_ai_event(
    operation: str,
    args: Dict[str, Any],
    idempotency_key: Optional[str] = None,
) -> Any:
    """Submit an AI metering event with an Idempotency-Key header.

    Args:
        operation: One of "completion", "image", "video", "audio".
        args: Keyword arguments forwarded to client.ai.create_<operation>.
        idempotency_key: Optional explicit override. If None, the function
            checks the current contextvar (set via the idempotency_key()
            context manager or set_idempotency_key());
            if that is also None, a UUID v4 is generated. A key that is not
            1 to 255 printable ASCII characters without leading or trailing
            spaces is replaced by a UUID v4, with one warning.

    Returns:
        Whatever client.ai.create_<operation> returns. Returns None when
        the metering client is not configured (no API key), when delivery
        failed with a retryable error and the event was placed in the
        store-and-forward buffer for automatic replay, or when it runs under
        ``delivery_deferred_to_buffer`` and the event was buffered unsent.

    Raises:
        ValueError: if operation is not one of the recognized AI operations.
    """
    client = current_client()
    if client is None:
        return None

    if operation not in _AI_OPERATIONS:
        raise ValueError(
            f"Unknown AI metering operation {operation!r}; "
            f"expected one of {sorted(_AI_OPERATIONS)}"
        )

    key = _choose_idempotency_key(idempotency_key)

    existing_headers = args.get("extra_headers") or {}
    if any(k.lower() == "idempotency-key" for k in existing_headers):
        raise ValueError(
            "Pass Idempotency-Key via the idempotency_key parameter, "
            "not via extra_headers; the wrapper owns this header"
        )
    merged_args = {
        **_accepted_fields(operation, _json_safe_args(operation, args)),
        "extra_headers": {**existing_headers, "Idempotency-Key": key},
    }
    _mask_prompt_fields(merged_args)

    if metering_buffer.is_delivery_deferred_to_buffer():
        metering_buffer.buffer_deferred_event("ai", _replay_payload(operation, merged_args, client))
        return None

    method = getattr(client.ai, f"create_{operation}")
    try:
        result = method(**merged_args)
    except Exception as exc:
        record_metering_error(exc, operation=operation)
        if metering_buffer.is_retryable_failure(exc):
            get_circuit().record_failure()
            # Retries are exhausted inside the client; keep the event (with
            # its frozen Idempotency-Key) for background replay instead of
            # discarding it.
            logger.error(
                "Metering %s event delivery failed (%s); "
                "event buffered for automatic retry",
                operation,
                exc,
            )
            metering_buffer.get_buffer().push("ai", _replay_payload(operation, merged_args, client))
            return None
        if metering_buffer.warn_if_redirected(exc):
            raise
        logger.error(
            "Metering %s event delivery failed permanently: %s "
            "(inputTokenCount=%s, outputTokenCount=%s, totalTokenCount=%s)",
            operation,
            exc,
            args.get("input_token_count"),
            args.get("output_token_count"),
            args.get("total_token_count"),
        )
        raise
    _record_delivered()
    return result


def _record_delivered() -> None:
    record_metering_success()
    if get_circuit().record_success():
        metering_buffer.request_replay()
