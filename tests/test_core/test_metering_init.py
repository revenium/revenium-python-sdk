"""Tests for the metering client bootstrap (`_build_metering_client`).

Tests call the builder directly to keep module-level state untouched — earlier
versions used ``importlib.reload`` and caused state-leakage failures in
``tests/test_metering.py``.
"""

import pytest

from revenium_middleware._core.metering import _build_metering_client


def test_unset_api_key_returns_none():
    assert _build_metering_client(None, None) is None


def test_empty_api_key_returns_none():
    assert _build_metering_client("", None) is None


def test_invalid_prefix_raises():
    with pytest.raises(ValueError, match='should start with "hak_" or "rev_"'):
        _build_metering_client("invalid_no_prefix", None)


def test_hak_prefix_initializes_client():
    client = _build_metering_client("hak_test123abc", None)
    assert client is not None


def test_rev_prefix_initializes_client():
    client = _build_metering_client("rev_mk_test123abc", None)
    assert client is not None
