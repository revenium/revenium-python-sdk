"""
Provider detection and configuration for Google AI SDK support.

This module handles detection of Gemini Developer API vs Vertex AI endpoints
when using the Google AI SDK (google-genai package). Both endpoints report
as "Google" provider for unified analytics.

Migrated from the original provider.py with updates for the new architecture.
"""

import os
import logging
from enum import Enum, auto
from typing import Optional, Any

from ..common import ProviderMetadata

logger = logging.getLogger("revenium_middleware.extension")


class GoogleAIEndpoint(Enum):
    """Google AI SDK endpoint types."""

    GEMINI_DEVELOPER_API = auto()
    VERTEX_AI = auto()


def detect_provider(client: Optional[Any] = None) -> GoogleAIEndpoint:
    """
    Detect which Google AI endpoint is being used with the Google AI SDK.

    Detection priority:
    1. Client configuration (vertexai parameter) - most reliable
    2. Environment variables (GOOGLE_GENAI_USE_VERTEXAI)
    3. Check for project/location vs API key configuration
    4. Default to Gemini Developer API

    Args:
        client: Google GenAI client instance

    Returns:
        GoogleAIEndpoint enum indicating detected endpoint
    """
    logger.debug("Detecting Google AI SDK endpoint...")

    # 1. Check client configuration first (most reliable)
    if client and hasattr(client, "_vertexai") and client._vertexai:
        logger.debug("Vertex AI endpoint detected via client configuration")
        return GoogleAIEndpoint.VERTEX_AI

    # 2. Check environment variable
    if os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "").lower() in ("true", "1", "yes"):
        logger.debug(
            "Vertex AI endpoint detected via GOOGLE_GENAI_USE_VERTEXAI environment variable"
        )
        return GoogleAIEndpoint.VERTEX_AI

    # 3. Check for Vertex AI configuration (project + location)
    if os.getenv("GOOGLE_CLOUD_PROJECT") and os.getenv("GOOGLE_CLOUD_LOCATION"):
        logger.debug(
            "Vertex AI endpoint detected via GOOGLE_CLOUD_PROJECT and GOOGLE_CLOUD_LOCATION"
        )
        return GoogleAIEndpoint.VERTEX_AI

    # 4. Check if we have API key (Gemini Developer API)
    if os.getenv("GOOGLE_API_KEY"):
        logger.debug("Gemini Developer API endpoint detected via GOOGLE_API_KEY")
        return GoogleAIEndpoint.GEMINI_DEVELOPER_API

    # 5. Default to Gemini Developer API
    logger.debug("Defaulting to Gemini Developer API endpoint")
    return GoogleAIEndpoint.GEMINI_DEVELOPER_API


def get_provider_metadata() -> ProviderMetadata:
    """
    Get provider metadata for Google AI SDK usage records.

    Returns:
        ProviderMetadata with standardized "Google" provider name
    """
    # Google AI SDK always uses the same provider metadata
    return ProviderMetadata.for_google_ai_sdk()


def is_vertex_ai_endpoint(endpoint: GoogleAIEndpoint) -> bool:
    """
    Check if the endpoint is Vertex AI.

    Args:
        endpoint: Endpoint to check

    Returns:
        True if Vertex AI endpoint, False otherwise
    """
    return endpoint == GoogleAIEndpoint.VERTEX_AI


# Global endpoint cache to avoid repeated detection
_detected_endpoint: Optional[GoogleAIEndpoint] = None
_endpoint_detection_attempted: bool = False


def get_or_detect_provider(
    client: Optional[Any] = None, force_redetect: bool = False
) -> GoogleAIEndpoint:
    """
    Get cached endpoint or detect if not already done.

    This provides lazy loading - detection only happens when needed and is cached.

    Args:
        client: Google GenAI client instance
        force_redetect: Force re-detection even if cached

    Returns:
        Detected endpoint
    """
    global _detected_endpoint, _endpoint_detection_attempted

    if force_redetect or not _endpoint_detection_attempted:
        _detected_endpoint = detect_provider(client)
        _endpoint_detection_attempted = True
        logger.debug(
            f"Google AI SDK endpoint detection completed: {_detected_endpoint}"
        )

    return _detected_endpoint


def reset_provider_cache():
    """Reset endpoint detection cache. Useful for testing."""
    global _detected_endpoint, _endpoint_detection_attempted
    _detected_endpoint = None
    _endpoint_detection_attempted = False
