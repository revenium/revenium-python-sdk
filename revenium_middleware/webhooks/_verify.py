from __future__ import annotations

import hashlib
import hmac
import time

_MAX_SIGNATURES = 10  # defense-in-depth: more than sufficient for secret rotation overlap (normally 1-2 entries)


def verify_signature(
    payload: bytes,
    signature_header: str,
    timestamp_header: str,
    secrets: list[str],
    tolerance_seconds: float = 300,
) -> bool:
    """Return True if any signature on the header matches any secret; False otherwise.

    All malformed input (wrong types, unparseable timestamp, no sha256= entries,
    empty secrets list) returns False. The function never raises.
    """
    if not secrets:
        return False
    if not all(isinstance(s, str) for s in secrets):
        return False
    if not isinstance(payload, (bytes, bytearray)):
        return False
    if not isinstance(signature_header, str) or not isinstance(timestamp_header, str):
        return False

    ts = timestamp_header.strip()
    if not _is_timestamp_within_tolerance(ts, tolerance_seconds):
        return False

    received_digests = _parse_signature_header(signature_header)
    if not received_digests:
        return False

    signed_payload = f"{ts}.".encode("utf-8") + bytes(payload)
    for secret in secrets:
        expected = hmac.new(
            secret.encode("utf-8"),
            signed_payload,
            hashlib.sha256,
        ).hexdigest()
        for received in received_digests:
            if hmac.compare_digest(expected, received):
                return True
    return False


def _parse_signature_header(header: str | None) -> list[str]:
    """Return hex digests of every `sha256=` entry in a comma-separated header."""
    if not header:
        return []
    digests: list[str] = []
    for entry in header.split(","):
        entry = entry.strip()
        if entry.startswith("sha256="):
            digest = entry[len("sha256="):].lower()
            if digest:
                digests.append(digest)
                if len(digests) >= _MAX_SIGNATURES:
                    break
    return digests


def _is_timestamp_within_tolerance(ts: str, tolerance_seconds: float) -> bool:
    """Return True if `ts` parses as int and |now - ts| <= tolerance_seconds."""
    try:
        signed_at = int(ts)
    except (TypeError, ValueError):
        return False
    return abs(time.time() - signed_at) <= tolerance_seconds
