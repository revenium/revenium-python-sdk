"""
Vertex AI SDK middleware for Revenium.

This module provides middleware for the native Vertex AI SDK (vertexai package),
offering enhanced features like comprehensive token counting and local tokenization.
"""

import logging

logger = logging.getLogger(__name__)

# Import provider (no SDK dependency)
from . import provider

from .provider import (
    detect_provider,
    get_provider_metadata,
    validate_vertex_ai_configuration,
    get_vertex_ai_config,
    is_vertex_ai_available,
)

# Conditionally import middleware (requires vertexai SDK)
try:
    import vertexai  # noqa: F401
    from . import middleware
    from .middleware import (
        extract_vertex_ai_usage_data,
        extract_vertex_ai_generation_tokens,
        extract_vertex_ai_embedding_tokens,
        create_vertex_ai_metering_call,
        handle_vertex_ai_streaming_response,
    )
except ImportError as e:
    from revenium_middleware._core.load_diagnostics import log_middleware_load_failure
    log_middleware_load_failure("Vertex AI", e, required_packages=("vertexai",))
    middleware = None  # type: ignore
    extract_vertex_ai_usage_data = None  # type: ignore
    extract_vertex_ai_generation_tokens = None  # type: ignore
    extract_vertex_ai_embedding_tokens = None  # type: ignore
    create_vertex_ai_metering_call = None  # type: ignore
    handle_vertex_ai_streaming_response = None  # type: ignore

__all__ = [
    # Middleware functions
    "extract_vertex_ai_usage_data",
    "extract_vertex_ai_generation_tokens",
    "extract_vertex_ai_embedding_tokens",
    "create_vertex_ai_metering_call",
    "handle_vertex_ai_streaming_response",
    # Provider functions
    "detect_provider",
    "get_provider_metadata",
    "validate_vertex_ai_configuration",
    "get_vertex_ai_config",
    "is_vertex_ai_available",
]
