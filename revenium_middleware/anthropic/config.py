"""
Configuration for Revenium Anthropic middleware.

Extends shared core config with Anthropic-specific settings.
"""

from revenium_middleware._core.config import (  # noqa: F401
    Config as _CoreConfig,
    SecurityConfig,
    get_config_value,
    is_debug_enabled,
    get_timeout_config,
    get_team_id,
    get_base_url,
)


class Config(_CoreConfig):
    """Anthropic-specific configuration extending core config."""

    # Provider-specific environment variables
    ENV_ANTHROPIC_API_KEY: str = "ANTHROPIC_API_KEY"

    # Region environment variables
    ENV_AWS_REGION: str = "AWS_REGION"
    ENV_AWS_DEFAULT_REGION: str = "AWS_DEFAULT_REGION"
