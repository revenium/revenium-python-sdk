"""
Provider detection and configuration for Vertex AI SDK support.

This module handles provider detection for the native Vertex AI SDK (vertexai package).
All Vertex AI SDK usage reports as "Google" provider for unified analytics.
"""

import os
import logging
from ..common import ProviderMetadata

logger = logging.getLogger("revenium_middleware.extension")


def detect_provider() -> str:
    """
    Detect Vertex AI provider configuration.

    For the native Vertex AI SDK, we always return "vertex_ai" since
    this module only handles the vertexai package.

    Returns:
        Always returns "vertex_ai" for this module
    """
    logger.debug("Vertex AI SDK provider detected")
    return "vertex_ai"


def get_provider_metadata() -> ProviderMetadata:
    """
    Get provider metadata for Vertex AI SDK usage records.

    Returns standardized "Google" provider name for unified analytics.

    Returns:
        ProviderMetadata with "Google" provider name
    """
    return ProviderMetadata.for_vertex_ai_sdk()


def validate_vertex_ai_configuration() -> bool:
    """
    Validate that Vertex AI is properly configured.

    Checks for required environment variables and configuration.

    Returns:
        True if Vertex AI is properly configured, False otherwise
    """
    # Check for required Vertex AI configuration
    project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
    location = os.getenv("GOOGLE_CLOUD_LOCATION")

    if not project_id:
        logger.warning(
            "GOOGLE_CLOUD_PROJECT environment variable not set for Vertex AI"
        )
        return False

    if not location:
        logger.warning(
            "GOOGLE_CLOUD_LOCATION environment variable not set for Vertex AI"
        )
        return False

    logger.debug(
        f"Vertex AI configuration validated: project={project_id}, location={location}"
    )
    return True


def get_vertex_ai_config() -> dict:
    """
    Get Vertex AI configuration from environment variables.

    Returns:
        Dictionary with Vertex AI configuration
    """
    return {
        "project_id": os.getenv("GOOGLE_CLOUD_PROJECT"),
        "location": os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1"),
        "credentials_path": os.getenv("GOOGLE_APPLICATION_CREDENTIALS"),
    }


def is_vertex_ai_available() -> bool:
    """
    Check if Vertex AI SDK is available and properly configured.

    Returns:
        True if Vertex AI SDK is available and configured, False otherwise
    """
    try:
        import vertexai

        return validate_vertex_ai_configuration()
    except ImportError:
        logger.debug("Vertex AI SDK not available")
        return False
