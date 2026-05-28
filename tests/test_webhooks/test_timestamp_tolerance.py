import pytest

from revenium_middleware.webhooks._verify import _is_timestamp_within_tolerance


NOW = 1_716_750_000


@pytest.fixture(autouse=True)
def _frozen_time(monkeypatch):
    monkeypatch.setattr(
        "revenium_middleware.webhooks._verify.time.time",
        lambda: float(NOW),
    )


def test_at_exact_now():
    assert _is_timestamp_within_tolerance(str(NOW), 300) is True


def test_within_window_past():
    assert _is_timestamp_within_tolerance(str(NOW - 299), 300) is True


def test_within_window_future():
    assert _is_timestamp_within_tolerance(str(NOW + 299), 300) is True


def test_boundary_past_passes():
    assert _is_timestamp_within_tolerance(str(NOW - 300), 300) is True


def test_boundary_future_passes():
    assert _is_timestamp_within_tolerance(str(NOW + 300), 300) is True


def test_just_outside_past_fails():
    assert _is_timestamp_within_tolerance(str(NOW - 301), 300) is False


def test_just_outside_future_fails():
    assert _is_timestamp_within_tolerance(str(NOW + 301), 300) is False


def test_non_numeric_returns_false():
    assert _is_timestamp_within_tolerance("not-a-number", 300) is False


def test_empty_string_returns_false():
    assert _is_timestamp_within_tolerance("", 300) is False


def test_custom_tolerance_value():
    """Caller-supplied tolerance is respected (not hard-coded to 300)."""
    assert _is_timestamp_within_tolerance(str(NOW - 60), 30) is False
    assert _is_timestamp_within_tolerance(str(NOW - 60), 60) is True
