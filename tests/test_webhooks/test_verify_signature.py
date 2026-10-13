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


@pytest.mark.parametrize("signing_position", ["first", "second"])
def test_rotation_accepts_a_header_signed_only_with_one_listed_secret(signing_position):
    payload = b'{"event":"test"}'
    ts = str(NOW)
    secrets = ["secret_old", "secret_new"]
    signing_secret = secrets[0] if signing_position == "first" else secrets[1]
    header = f"sha256={_sign(payload, ts, signing_secret)}"
    assert verify_signature(payload, header, ts, secrets) is True


def test_a_matching_signature_after_a_non_matching_one_is_accepted():
    payload = b'{"event":"test"}'
    ts = str(NOW)
    header = f"sha256={'0' * 64}, sha256={_sign(payload, ts, SECRET)}"
    assert verify_signature(payload, header, ts, [SECRET]) is True


def test_a_header_signed_with_neither_listed_secret_is_rejected():
    payload = b'{"event":"test"}'
    ts = str(NOW)
    header = f"sha256={_sign(payload, ts, 'secret_other')}"
    assert verify_signature(payload, header, ts, ["secret_old", "secret_new"]) is False


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


_DISTINCT_SECRET_CHARACTERS = sorted(set(SECRET))


@pytest.mark.parametrize("character", _DISTINCT_SECRET_CHARACTERS)
def test_signature_made_with_one_character_of_a_plain_string_secret_is_rejected(character):
    """BACK-4404: a plain-string secret must not be split into one secret per character."""
    payload = b'{"event":"forged"}'
    ts = str(NOW)
    header = f"sha256={_sign(payload, ts, character)}"
    assert verify_signature(payload, header, ts, SECRET) is False


def test_a_header_carrying_one_signature_per_character_is_rejected():
    payload = b'{"event":"forged"}'
    ts = str(NOW)
    characters = _DISTINCT_SECRET_CHARACTERS[:10]
    header = ", ".join(f"sha256={_sign(payload, ts, c)}" for c in characters)
    assert verify_signature(payload, header, ts, SECRET) is False


def test_plain_string_secrets_returns_false_without_raising():
    """A genuine signature with the secret passed bare still returns False.

    Guards against the wrong fix of wrapping a plain string in a list, which
    would return True here; the docstring promises False for wrong types.
    """
    payload = b'{"event":"test"}'
    ts = str(NOW)
    header = f"sha256={_sign(payload, ts, SECRET)}"
    assert verify_signature(payload, header, ts, SECRET) is False


@pytest.mark.parametrize(
    "secrets",
    ["secret", "", None, b"x", 5, 3.0, {"k": "v"}, ["a", 1]],
    ids=["str", "empty-str", "none", "bytes", "int", "float", "dict", "mixed-list"],
)
def test_wrong_secrets_types_return_false_without_raising(secrets):
    payload = b'{"event":"test"}'
    ts = str(NOW)
    header = f"sha256={_sign(payload, ts, SECRET)}, sha256={_sign(payload, ts, 'k')}"
    assert verify_signature(payload, header, ts, secrets) is False


def test_a_tuple_of_secrets_still_verifies():
    payload = b'{"event":"test"}'
    ts = str(NOW)
    header = f"sha256={_sign(payload, ts, SECRET)}"
    assert verify_signature(payload, header, ts, (SECRET,)) is True
