"""A trace type or trace name that is not a string is left off, never raised on (BACK-4449 N6)."""
import logging

import pytest

from revenium_middleware._core.trace_fields import validate_trace_name, validate_trace_type


@pytest.mark.parametrize("validate, value", [
    pytest.param(validate_trace_type, 5, id="trace-type-int"),
    pytest.param(validate_trace_type, ["call-type"], id="trace-type-list"),
    pytest.param(validate_trace_name, 7, id="trace-name-int"),
    pytest.param(validate_trace_name, {"name": "Call Trace"}, id="trace-name-dict"),
])
def test_a_non_string_value_is_left_off_with_one_warning_naming_its_type(caplog, validate, value):
    with caplog.at_level(logging.WARNING):
        assert validate(value) is None

    [warning] = [record for record in caplog.records if record.levelno == logging.WARNING]
    assert type(value).__name__ in warning.getMessage()
    assert repr(value) not in warning.getMessage()
