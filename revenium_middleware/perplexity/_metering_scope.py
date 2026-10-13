from revenium_middleware._core.config import is_selective_metering_enabled
from revenium_middleware._core.context import is_inside_decorated_function
from revenium_middleware._core.selective_metering import reported_if_unmetered
from revenium_middleware._core.perplexity_usage_metadata import take_usage_metadata  # noqa: F401


def metering_skipped() -> bool:
    """True when selective metering is on and the call is outside a @revenium_meter function."""
    return reported_if_unmetered(is_selective_metering_enabled() and not is_inside_decorated_function())
