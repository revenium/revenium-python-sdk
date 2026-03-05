"""
Trace visualization field capture and validation.

This module provides functions to capture trace visualization fields from
environment variables and validate them according to the specification.

Shared functions are imported from _core.trace_fields. This module retains
only the Ollama-specific detect_operation_type function.
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


def detect_operation_type(
    endpoint: str,
    request_body: Optional[Dict[str, Any]] = None
) -> str:
    """
    Auto-detect operation type from endpoint and request.

    Args:
        endpoint: API endpoint (e.g., 'chat', 'generate', 'embeddings')
        request_body: Optional request body to check for tools

    Returns:
        Operation type string ('CHAT', 'GENERATE', 'EMBED', 'TOOL_CALL')
    """
    request_body = request_body or {}

    # Chat endpoint
    if endpoint == 'chat':
        # Check for tools in request
        has_tools = request_body.get('tools')
        if has_tools:
            return 'TOOL_CALL'
        return 'CHAT'

    # Generate endpoint
    if endpoint == 'generate':
        return 'GENERATE'

    # Embeddings endpoint
    if endpoint in ('embeddings', 'embed'):
        return 'EMBED'

    # Default fallback
    return 'CHAT'
