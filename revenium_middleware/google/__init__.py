"""
Revenium middleware for Google AI services.

This package provides middleware to track and meter usage of Google AI services
including Google AI SDK (google-genai) and Vertex AI SDK (vertexai).

The middleware automatically wraps API calls to capture usage metrics and send
them to Revenium for tracking and billing purposes.

Usage:
    Simply import this package before using Google AI services:

    import revenium_middleware.google
    from google import genai  # or import vertexai

    # Your existing code works unchanged
    client = genai.Client(api_key="your-key")
    response = client.models.generate_content(...)
"""

import logging

from revenium_middleware._core.config import env_flag_enabled


# Set up logging
logger = logging.getLogger(__name__)

# Check if verbose startup logging is enabled
_verbose_startup = env_flag_enabled("REVENIUM_VERBOSE_STARTUP")

if _verbose_startup:
    logger.info("Revenium middleware initialization starting")

# Import common utilities (always available)
from .common import utils

# Import and activate middleware for available SDKs
try:
    # Try to import Google AI SDK and activate middleware
    if _verbose_startup:
        logger.debug("Attempting to import google.genai")
    import google.genai

    if _verbose_startup:
        logger.debug("google.genai imported successfully, importing middleware")
    from .google_ai import middleware as google_ai_middleware

    logger.info("Google AI SDK middleware activated")
except ImportError as e:
    from revenium_middleware._core.load_diagnostics import log_middleware_load_failure
    log_middleware_load_failure("Google AI", e, required_packages=("google.genai",))

try:
    # Try to import Vertex AI SDK and activate middleware
    if _verbose_startup:
        logger.debug("Attempting to import vertexai")
    import vertexai

    if _verbose_startup:
        logger.debug("vertexai imported successfully, importing middleware")
    from .vertex_ai import middleware as vertex_ai_middleware

    logger.info("Vertex AI SDK middleware activated")
except ImportError as e:
    from revenium_middleware._core.load_diagnostics import log_middleware_load_failure
    log_middleware_load_failure("Vertex AI", e, required_packages=("vertexai",))

active_sdks = []
if "google.genai" in globals():
    active_sdks.append("google_ai")
if "vertexai" in globals():
    active_sdks.append("vertex_ai")

logger.info(
    "Revenium middleware activated for: %s",
    ", ".join(active_sdks) if active_sdks else "none",
)

__all__ = ["utils"]
