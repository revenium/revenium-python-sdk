"""
Configuration for Revenium Google middleware.

Extends shared core config with Google-specific settings.
"""

from revenium_middleware._core.config import (  # noqa: F401
    Config as _CoreConfig,
    DEFAULT_MAX_PROMPT_LENGTH,
    SecurityConfig,
    get_config_value,
    is_debug_enabled,
    get_timeout_config,
    read_prompt_length,
)

_MAX_PROMPT_LENGTH_CEILING = 1_000_000
_ENV_REVENIUM_MAX_PROMPT_LENGTH = "REVENIUM_MAX_PROMPT_LENGTH"


def _max_prompt_length() -> int:
    """REVENIUM_MAX_PROMPT_SIZE, else the Gemini-only REVENIUM_MAX_PROMPT_LENGTH, else 50,000, the core default."""
    gemini_length = read_prompt_length(_ENV_REVENIUM_MAX_PROMPT_LENGTH, DEFAULT_MAX_PROMPT_LENGTH)
    return min(read_prompt_length(_CoreConfig.ENV_REVENIUM_MAX_PROMPT_SIZE, gemini_length),
               _MAX_PROMPT_LENGTH_CEILING)


class Config(_CoreConfig):
    """Google-specific configuration extending core config."""

    # Provider-specific environment variables
    ENV_GOOGLE_API_KEY: str = "GOOGLE_API_KEY"
    ENV_GOOGLE_CLOUD_PROJECT: str = "GOOGLE_CLOUD_PROJECT"
    ENV_GOOGLE_APPLICATION_CREDENTIALS: str = "GOOGLE_APPLICATION_CREDENTIALS"

    # Region environment variables
    ENV_GCP_REGION: str = "GCP_REGION"
    ENV_GOOGLE_CLOUD_REGION: str = "GOOGLE_CLOUD_REGION"

    ENV_REVENIUM_MAX_PROMPT_LENGTH: str = _ENV_REVENIUM_MAX_PROMPT_LENGTH
    MAX_PROMPT_LENGTH: int = _max_prompt_length()
