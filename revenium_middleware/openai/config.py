"""
Configuration for Revenium OpenAI middleware.

Extends shared core config with OpenAI and Azure-specific settings.
"""

from revenium_middleware._core.config import (  # noqa: F401
    Config as _CoreConfig,
    SecurityConfig,
    get_config_value,
    is_debug_enabled,
    get_timeout_config as _core_get_timeout_config,
    get_team_id,
    get_base_url,
)


class Config(_CoreConfig):
    """OpenAI-specific configuration extending core config."""

    # Azure API settings
    AZURE_API_VERSION_DEFAULT: str = "2024-10-21"
    AZURE_MODEL_RESOLUTION_TIMEOUT: float = 5.0

    # Provider-specific environment variables
    ENV_OPENAI_API_KEY: str = "OPENAI_API_KEY"
    ENV_AZURE_OPENAI_ENDPOINT: str = "AZURE_OPENAI_ENDPOINT"
    ENV_AZURE_OPENAI_API_KEY: str = "AZURE_OPENAI_API_KEY"

    # Region environment variables
    ENV_AWS_REGION: str = "AWS_REGION"
    ENV_AWS_DEFAULT_REGION: str = "AWS_DEFAULT_REGION"
    ENV_AZURE_REGION: str = "AZURE_REGION"
    ENV_GCP_REGION: str = "GCP_REGION"
    ENV_GOOGLE_CLOUD_REGION: str = "GOOGLE_CLOUD_REGION"


def get_timeout_config() -> dict:
    """Get all timeout-related configuration, including Azure."""
    config = _core_get_timeout_config()
    config['azure_model_resolution'] = Config.AZURE_MODEL_RESOLUTION_TIMEOUT
    return config
