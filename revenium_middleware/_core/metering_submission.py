"""Centralized submission of AI metering events with Idempotency-Key injection."""
from __future__ import annotations

import uuid
from typing import Any, Dict, Optional

from revenium_middleware._core.context import get_idempotency_key
from revenium_middleware._core.metering import client


_AI_OPERATIONS = frozenset({"completion", "image", "video", "audio"})


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
            if that is also None, a UUID v4 is generated.

    Returns:
        Whatever client.ai.create_<operation> returns. Returns None when
        the metering client is not configured (no API key).

    Raises:
        ValueError: if operation is not one of the recognized AI operations.
    """
    if client is None:
        return None

    if operation not in _AI_OPERATIONS:
        raise ValueError(
            f"Unknown AI metering operation {operation!r}; "
            f"expected one of {sorted(_AI_OPERATIONS)}"
        )

    if idempotency_key is not None:
        if idempotency_key == "":
            raise ValueError(
                "idempotency_key must be a non-empty string; "
                "the backend rejects empty keys"
            )
        key = idempotency_key
    else:
        ctx_key = get_idempotency_key()
        key = ctx_key if ctx_key is not None else str(uuid.uuid4())

    existing_headers = args.get("extra_headers") or {}
    if any(k.lower() == "idempotency-key" for k in existing_headers):
        raise ValueError(
            "Pass Idempotency-Key via the idempotency_key parameter, "
            "not via extra_headers; the wrapper owns this header"
        )
    merged_args = {
        **args,
        "extra_headers": {**existing_headers, "Idempotency-Key": key},
    }

    method = getattr(client.ai, f"create_{operation}")
    return method(**merged_args)
