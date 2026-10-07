from revenium_middleware._core.config import is_selective_metering_enabled
from revenium_middleware._core.context import is_inside_decorated_function


def metering_skipped() -> bool:
    """True when selective metering is on and the call is outside a @revenium_meter function."""
    return is_selective_metering_enabled() and not is_inside_decorated_function()
