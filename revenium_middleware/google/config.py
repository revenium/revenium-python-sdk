"""
Configuration for Revenium Google middleware.

Extends shared core config with Google-specific settings.
"""

import logging
import os

from revenium_middleware._core.config import (  # noqa: F401
    Config as _CoreConfig,
    SecurityConfig,
    SummaryFormat,
    get_config_value,
    is_debug_enabled,
    get_timeout_config,
)


def _parse_max_prompt_length() -> int:
    """Parse MAX_PROMPT_LENGTH from env var with safe fallback."""
    default_value = 10000
    env_value = os.getenv("REVENIUM_MAX_PROMPT_LENGTH", str(default_value))
    try:
        parsed = int(env_value)
        if parsed <= 0:
            logger = logging.getLogger("revenium_middleware.config")
            logger.warning(
                f"REVENIUM_MAX_PROMPT_LENGTH must be positive, got {parsed}. "
                f"Using default: {default_value}"
            )
            return default_value
        if parsed > 1_000_000:
            logger = logging.getLogger("revenium_middleware.config")
            logger.warning(
                f"REVENIUM_MAX_PROMPT_LENGTH too large ({parsed}), "
                f"capping at 1,000,000 characters"
            )
            return 1_000_000
        return parsed
    except (ValueError, TypeError) as e:
        logger = logging.getLogger("revenium_middleware.config")
        logger.warning(
            f"Invalid REVENIUM_MAX_PROMPT_LENGTH value '{env_value}': {e}. "
            f"Using default: {default_value}"
        )
        return default_value


class Config(_CoreConfig):
    """Google-specific configuration extending core config."""

    # Provider-specific environment variables
    ENV_GOOGLE_API_KEY: str = "GOOGLE_API_KEY"
    ENV_GOOGLE_CLOUD_PROJECT: str = "GOOGLE_CLOUD_PROJECT"
    ENV_GOOGLE_APPLICATION_CREDENTIALS: str = "GOOGLE_APPLICATION_CREDENTIALS"

    # Region environment variables
    ENV_GCP_REGION: str = "GCP_REGION"
    ENV_GOOGLE_CLOUD_REGION: str = "GOOGLE_CLOUD_REGION"

    # Google overrides MAX_PROMPT_LENGTH with env-var-configurable parser (default 10000)
    ENV_REVENIUM_MAX_PROMPT_LENGTH: str = "REVENIUM_MAX_PROMPT_LENGTH"
    MAX_PROMPT_LENGTH: int = _parse_max_prompt_length()
