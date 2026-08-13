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
import os


# Configure logging based on REVENIUM_LOG_LEVEL environment variable
def _configure_logging():
    """Configure logging for the Revenium middleware."""
    log_level_str = os.getenv("REVENIUM_LOG_LEVEL", "INFO").upper()

    # Map string levels to logging constants
    level_mapping = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }

    log_level = level_mapping.get(log_level_str, logging.INFO)

    # Configure the revenium middleware logger
    revenium_logger = logging.getLogger("revenium_middleware")
    revenium_logger.setLevel(log_level)

    # Only add handler if none exists to avoid duplicate logs
    if not revenium_logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        revenium_logger.addHandler(handler)

    return log_level


# Configure logging
_log_level = _configure_logging()

# Set up logging
logger = logging.getLogger(__name__)

# Check if verbose startup logging is enabled
_verbose_startup = os.getenv("REVENIUM_VERBOSE_STARTUP", "").lower() in (
    "true",
    "1",
    "yes",
)

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
