"""
Trace visualization field capture and validation.

This module provides functions to capture trace visualization fields from
environment variables and validate them according to the specification.

Shared functions are imported from _core.trace_fields. This module retains
only the LiteLLM-specific detect_operation_type function.
"""

from typing import Optional, Dict, Any

from revenium_middleware._core.trace_fields import (  # noqa: F401 — re-exported
    TRACE_TYPE_MAX_LENGTH,
    TRACE_NAME_MAX_LENGTH,
    TRACE_TYPE_PATTERN,
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


def detect_operation_type(method_name: str, request_body: Optional[Dict[str, Any]] = None) -> str:
    """
    Auto-detect operation type from method name and request body.

    Args:
        method_name: The name of the method being called
        request_body: Optional request body to check for tools/functions

    Returns:
        Operation type: 'CHAT', 'EMBED', or 'TOOL_CALL'
    """
    # Check for embeddings
    if 'embed' in method_name.lower():
        return 'EMBED'

    # Check for tool/function calls in request body
    if request_body:
        if request_body.get('tools') or request_body.get('functions'):
            return 'TOOL_CALL'

    # Default to CHAT for completion/generation operations
    return 'CHAT'
