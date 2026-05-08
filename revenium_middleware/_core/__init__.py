"""
Core functionality for Revenium middleware.

This subpackage contains the foundational components shared across all
provider-specific middleware implementations.
"""

from .metering import run_async_in_thread, shutdown_event, client
from .exceptions import ReveniumCostLimitExceeded
from .enforcement import check_enforcement, is_circuit_breaker_enabled, stop_polling
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
from .fields import (
    extract_field_with_fallback,
    extract_with_aliases,
    extract_organization_name,
    extract_product_name,
    extract_org_and_product,
    extract_common_metadata,
    extract_agentic_job_fields,
    merge_extra_body,
)
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
    # Enforcement / circuit breaker
    "ReveniumCostLimitExceeded",
    "check_enforcement",
    "is_circuit_breaker_enabled",
    "stop_polling",
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
    # Field extraction
    "extract_field_with_fallback",
    "extract_with_aliases",
    "extract_organization_name",
    "extract_product_name",
    "extract_org_and_product",
    "extract_common_metadata",
    "extract_agentic_job_fields",
    "merge_extra_body",
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
