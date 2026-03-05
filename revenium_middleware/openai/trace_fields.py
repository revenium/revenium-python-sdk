"""
Trace visualization field capture and validation.

This module provides functions to capture trace visualization fields from
environment variables and validate them according to the specification.

Shared functions are imported from _core.trace_fields. This module retains
only the OpenAI-specific detect_operation_type function.
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
    provider: str,
    endpoint: str,
    request_body: Optional[Dict[str, Any]] = None
) -> Dict[str, Optional[str]]:
    """
    Auto-detect operation type and subtype from provider, endpoint,
    and request.

    Args:
        provider: Provider name (e.g., 'openai', 'azure_openai') or Provider enum
        endpoint: API endpoint (e.g., '/chat/completions', '/embeddings')
        request_body: Optional request body to check for tools/functions

    Returns:
        Dictionary with 'operationType' and 'operationSubtype' keys
    """
    # Handle Provider enum or string
    if hasattr(provider, 'name'):
        # It's a Provider enum, get the name (e.g., 'OPENAI', 'AZURE_OPENAI')
        provider_str = provider.name
    else:
        provider_str = str(provider)

    provider_lower = provider_str.lower()
    request_body = request_body or {}

    # OpenAI and Azure OpenAI
    if provider_lower in ('openai', 'azure_openai', 'azure'):
        # Chat completions
        is_chat = (
            'chat/completions' in endpoint or
            endpoint.endswith('/chat/completions')
        )
        if is_chat:
            # Check for tools or functions
            has_tools = (
                request_body.get('tools') or
                request_body.get('functions')
            )
            if has_tools:
                return {
                    'operationType': 'TOOL_CALL',
                    'operationSubtype': 'function_call'
                }
            return {
                'operationType': 'CHAT',
                'operationSubtype': None
            }

        # Embeddings
        if 'embeddings' in endpoint or endpoint.endswith('/embeddings'):
            return {
                'operationType': 'EMBED',
                'operationSubtype': None
            }

        # Moderations
        if 'moderations' in endpoint or endpoint.endswith('/moderations'):
            return {
                'operationType': 'MODERATION',
                'operationSubtype': None
            }

    # Default fallback
    return {
        'operationType': 'CHAT',
        'operationSubtype': None
    }
