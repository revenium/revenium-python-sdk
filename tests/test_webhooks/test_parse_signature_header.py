import pytest

from revenium_middleware.webhooks._verify import _parse_signature_header


def test_single_value():
    assert _parse_signature_header("sha256=abc123") == ["abc123"]


def test_multi_value_with_space():
    assert _parse_signature_header("sha256=A, sha256=B") == ["a", "b"]


def test_multi_value_no_space():
    assert _parse_signature_header("sha256=A,sha256=B") == ["a", "b"]


def test_multi_value_double_space():
    assert _parse_signature_header("sha256=A,  sha256=B") == ["a", "b"]


def test_leading_and_trailing_whitespace():
    assert _parse_signature_header(" sha256=abc ") == ["abc"]


def test_filters_non_sha256_entries():
    assert _parse_signature_header("md5=X, sha256=A") == ["a"]


def test_filters_unknown_algorithms_only():
    assert _parse_signature_header("sha1=X, sha256=A, sha512=Y") == ["a"]


def test_filters_empty_digest_after_prefix():
    """sha256= with no digest is filtered out, not returned as empty string."""
    assert _parse_signature_header("sha256=") == []
    assert _parse_signature_header("sha256=, sha256=A") == ["a"]


@pytest.mark.parametrize("header", ["", ",", ",,", "   "])
def test_empty_or_trivial_returns_empty_list(header):
    assert _parse_signature_header(header) == []


def test_none_returns_empty_list():
    assert _parse_signature_header(None) == []


def test_normalizes_uppercase_hex_to_lowercase():
    """Uppercase hex digests are normalized to lowercase so hmac.compare_digest matches."""
    assert _parse_signature_header("sha256=DEADBEEF") == ["deadbeef"]
    assert _parse_signature_header("sha256=AbCdEf123") == ["abcdef123"]


def test_mixed_case_in_multi_value_header():
    """Each entry in a multi-value header is normalized independently."""
    assert _parse_signature_header("sha256=ABC, sha256=def") == ["abc", "def"]


def test_caps_at_max_signatures():
    """Parser caps at _MAX_SIGNATURES to prevent load amplification on attacker-controlled headers."""
    from revenium_middleware.webhooks._verify import _MAX_SIGNATURES

    # Build a header with 2x the cap; parser must return exactly _MAX_SIGNATURES entries
    entries = ", ".join(f"sha256={i:064x}" for i in range(_MAX_SIGNATURES * 2))
    result = _parse_signature_header(entries)
    assert len(result) == _MAX_SIGNATURES
    # First entries kept (cap stops collection, not selection)
    assert result == [f"{i:064x}" for i in range(_MAX_SIGNATURES)]


def test_empty_entries_dont_count_toward_cap():
    """Empty `sha256=` entries are skipped without consuming cap slots."""
    from revenium_middleware.webhooks._verify import _MAX_SIGNATURES

    # 5 empty entries followed by enough valid entries to fill the cap exactly
    empties = ["sha256=" for _ in range(5)]
    valids = [f"sha256={i:064x}" for i in range(_MAX_SIGNATURES)]
    header = ", ".join(empties + valids)
    result = _parse_signature_header(header)
    assert len(result) == _MAX_SIGNATURES
