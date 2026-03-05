"""
Core functionality for Revenium middleware.

This subpackage contains the foundational components shared across all
provider-specific middleware implementations.
"""

from .metering import run_async_in_thread, shutdown_event, client
from .context import (
    is_inside_decorated_function,
    get_function_metadata,
    set_decorated_context,
    clear_decorated_context,
    get_injected_metadata,
    set_injected_metadata,
    clear_injected_metadata,
    merge_metadata,
)
from .decorators import revenium_meter, revenium_metadata, track_usage
from .config import is_selective_metering_enabled
from .trace_fields import (
    get_environment,
    get_region,
    get_credential_alias,
    get_trace_type,
    get_trace_name,
    get_parent_transaction_id,
    get_transaction_name,
    get_retry_number,
    validate_trace_type,
    validate_trace_name,
)

__all__ = [
    # Metering
    "client",
    "run_async_in_thread",
    "shutdown_event",
    # Decorators
    "revenium_meter",
    "revenium_metadata",
    "track_usage",
    # Context management
    "is_inside_decorated_function",
    "get_function_metadata",
    "set_decorated_context",
    "clear_decorated_context",
    "get_injected_metadata",
    "set_injected_metadata",
    "clear_injected_metadata",
    "merge_metadata",
    # Config
    "is_selective_metering_enabled",
    # Trace fields
    "get_environment",
    "get_region",
    "get_credential_alias",
    "get_trace_type",
    "get_trace_name",
    "get_parent_transaction_id",
    "get_transaction_name",
    "get_retry_number",
    "validate_trace_type",
    "validate_trace_name",
]
