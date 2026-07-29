"""
Trace visualization fields for Perplexity AI API.

This module handles extraction and validation of trace visualization fields
for distributed tracing and analytics.

All functions are imported from _core.trace_fields. Perplexity has no
provider-specific trace field functions.
"""

from revenium_middleware._core.trace_fields import (  # noqa: F401 — re-exported
    TRACE_TYPE_MAX_LENGTH,
    TRACE_NAME_MAX_LENGTH,
    TICKET_ID_MAX_LENGTH,
    TRACE_TYPE_PATTERN,
    get_environment,
    get_region,
    get_credential_alias,
    get_trace_type,
    get_trace_name,
    get_parent_transaction_id,
    get_transaction_name,
    get_retry_number,
    get_ticket_id,
    validate_trace_type,
    validate_trace_name,
    validate_ticket_id,
)
