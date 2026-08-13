"""
Google AI SDK middleware for Revenium.

This module provides middleware for the Google AI SDK (google-genai package),
supporting both Gemini Developer API and Vertex AI endpoints through the
unified google-genai interface.
"""

import logging

logger = logging.getLogger(__name__)

# Import provider (no SDK dependency)
from . import provider

from .provider import (
    detect_provider,
    get_provider_metadata,
    is_vertex_ai_endpoint,
    get_or_detect_provider,
    reset_provider_cache,
    GoogleAIEndpoint,
)

# Conditionally import middleware (requires google-genai SDK)
try:
    import google.genai  # noqa: F401
    from . import middleware
    from .middleware import (
        extract_google_ai_usage_data,
        create_google_ai_metering_call,
        handle_streaming_response,
    )
except ImportError as e:
    from revenium_middleware._core.load_diagnostics import log_middleware_load_failure
    log_middleware_load_failure("Google AI", e, required_packages=("google.genai",))
    middleware = None  # type: ignore
    extract_google_ai_usage_data = None  # type: ignore
    create_google_ai_metering_call = None  # type: ignore
    handle_streaming_response = None  # type: ignore

__all__ = [
    # Middleware functions
    "extract_google_ai_usage_data",
    "create_google_ai_metering_call",
    "handle_streaming_response",
    # Provider functions
    "detect_provider",
    "get_provider_metadata",
    "is_vertex_ai_endpoint",
    "get_or_detect_provider",
    "reset_provider_cache",
    "GoogleAIEndpoint",
]
