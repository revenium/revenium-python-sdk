import hashlib
import hmac

import pytest

from revenium_middleware.webhooks import verify_signature


SECRET = "test_secret_value"
NOW = 1_716_750_000  # fixed unix epoch used by frozen clock


def _sign(payload: bytes, timestamp: str, secret: str) -> str:
    return hmac.new(
        secret.encode("utf-8"),
        f"{timestamp}.".encode("utf-8") + payload,
        hashlib.sha256,
    ).hexdigest()


@pytest.fixture(autouse=True)
def _frozen_time(monkeypatch):
    monkeypatch.setattr(
        "revenium_middleware.webhooks._verify.time.time",
        lambda: float(NOW),
    )


def test_single_secret_happy_path():
    payload = b'{"event":"test"}'
    ts = str(NOW)
    sig = _sign(payload, ts, SECRET)
    assert verify_signature(
        payload=payload,
        signature_header=f"sha256={sig}",
        timestamp_header=ts,
        secrets=[SECRET],
    ) is True


def test_secrets_with_non_str_item_returns_false():
    """Defense: non-str items in secrets (e.g., None) return False without raising."""
    payload = b"body"
    ts = str(NOW)
    sig = _sign(payload, ts, SECRET)
    assert verify_signature(payload, f"sha256={sig}", ts, [None]) is False
    assert verify_signature(payload, f"sha256={sig}", ts, [SECRET, None]) is False


def test_empty_string_secret_returns_false():
    """Security: empty-string secret produces a degenerate HMAC with a zero-byte key.
    Any caller who knows the key is empty can forge valid signatures trivially, so
    reject empty strings at the type guard."""
    payload = b"body"
    ts = str(NOW)
    sig_with_empty_key = hmac.new(b"", f"{ts}.".encode("utf-8") + payload, hashlib.sha256).hexdigest()
    assert verify_signature(payload, f"sha256={sig_with_empty_key}", ts, [""]) is False
    assert verify_signature(payload, f"sha256={sig_with_empty_key}", ts, [SECRET, ""]) is False


def test_rotation_overlap_both_signatures_match():
    payload = b'{"event":"test"}'
    ts = str(NOW)
    secret_old = "secret_old"
    secret_new = "secret_new"
    sig_old = _sign(payload, ts, secret_old)
    sig_new = _sign(payload, ts, secret_new)
    header = f"sha256={sig_old}, sha256={sig_new}"
    assert verify_signature(payload, header, ts, [secret_old, secret_new]) is True


def test_rotation_overlap_only_old_secret_matches():
    """Client knows only the old secret; new one is unknown."""
    payload = b'{"event":"test"}'
    ts = str(NOW)
    secret_old = "secret_old"
    secret_new = "secret_new"
    sig_old = _sign(payload, ts, secret_old)
    sig_new = _sign(payload, ts, secret_new)
    header = f"sha256={sig_old}, sha256={sig_new}"
    assert verify_signature(payload, header, ts, [secret_old]) is True


def test_rotation_overlap_order_independent():
    """Match must succeed regardless of secret order."""
    payload = b'{"event":"test"}'
    ts = str(NOW)
    secret_a = "secret_a"
    secret_b = "secret_b"
    sig_a = _sign(payload, ts, secret_a)
    sig_b = _sign(payload, ts, secret_b)
    header = f"sha256={sig_a}, sha256={sig_b}"
    assert verify_signature(payload, header, ts, [secret_a, secret_b]) is True
    assert verify_signature(payload, header, ts, [secret_b, secret_a]) is True


def test_timestamp_in_past_outside_tolerance():
    payload = b"body"
    past_ts = str(NOW - 301)
    sig = _sign(payload, past_ts, SECRET)
    assert verify_signature(payload, f"sha256={sig}", past_ts, [SECRET]) is False


def test_timestamp_in_future_outside_tolerance():
    payload = b"body"
    future_ts = str(NOW + 301)
    sig = _sign(payload, future_ts, SECRET)
    assert verify_signature(payload, f"sha256={sig}", future_ts, [SECRET]) is False


def test_timestamp_at_boundary_passes():
    payload = b"body"
    boundary_ts = str(NOW - 300)
    sig = _sign(payload, boundary_ts, SECRET)
    assert verify_signature(payload, f"sha256={sig}", boundary_ts, [SECRET]) is True


def test_timestamp_non_numeric_returns_false():
    assert verify_signature(b"body", "sha256=deadbeef", "not-a-number", [SECRET]) is False


@pytest.mark.parametrize("header", ["", "foo", "md5=abc123", ",", "sha256="])
def test_malformed_signature_header_returns_false(header):
    """No valid sha256= entries → False, even with valid secrets and timestamp."""
    payload = b"body"
    ts = str(NOW)
    assert verify_signature(payload, header, ts, [SECRET]) is False


def test_none_signature_header_returns_false():
    payload = b"body"
    ts = str(NOW)
    assert verify_signature(payload, None, ts, [SECRET]) is False


def test_none_timestamp_header_returns_false():
    payload = b"body"
    assert verify_signature(payload, "sha256=deadbeef", None, [SECRET]) is False


def test_empty_secrets_returns_false():
    payload = b"body"
    ts = str(NOW)
    sig = _sign(payload, ts, SECRET)
    assert verify_signature(payload, f"sha256={sig}", ts, []) is False


def test_secret_mismatch_returns_false():
    payload = b"body"
    ts = str(NOW)
    sig = _sign(payload, ts, "wrong_secret")
    assert verify_signature(payload, f"sha256={sig}", ts, [SECRET]) is False


def test_payload_as_str_returns_false():
    """Defense: payload must be bytes, not str. Returns False without raising."""
    ts = str(NOW)
    sig = _sign(b"hello", ts, SECRET)
    assert verify_signature("hello", f"sha256={sig}", ts, [SECRET]) is False


def test_payload_bytes_with_unicode():
    """UTF-8 multibyte payload round-trips correctly through HMAC."""
    payload = "olá mundo 你好".encode("utf-8")
    ts = str(NOW)
    sig = _sign(payload, ts, SECRET)
    assert verify_signature(payload, f"sha256={sig}", ts, [SECRET]) is True


def test_uppercase_hex_signature_verifies():
    """Server sending uppercase hex digest still verifies — hex comparison is case-insensitive."""
    payload = b"body"
    ts = str(NOW)
    sig_lower = _sign(payload, ts, SECRET)
    assert verify_signature(payload, f"sha256={sig_lower.upper()}", ts, [SECRET]) is True
