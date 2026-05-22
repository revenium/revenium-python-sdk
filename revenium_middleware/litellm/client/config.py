"""
Configuration for Revenium LiteLLM middleware.

Imports shared config from core and adds LiteLLM-specific settings.
"""

import os
from typing import Optional

from revenium_middleware._core.config import (  # noqa: F401
    Config as _CoreConfig,
    get_team_id,
    get_base_url,
)

# Re-export environment variable names for backward compatibility
ENV_REVENIUM_TEAM_ID = _CoreConfig.ENV_REVENIUM_TEAM_ID
ENV_REVENIUM_METERING_BASE_URL = _CoreConfig.ENV_REVENIUM_BASE_URL
ENV_REVENIUM_METERING_API_KEY = _CoreConfig.ENV_REVENIUM_API_KEY

DEFAULT_BASE_URL = _CoreConfig.DEFAULT_BASE_URL


def get_api_key() -> Optional[str]:
    """Get the Revenium API key from environment."""
    return os.getenv(ENV_REVENIUM_METERING_API_KEY)


__all__ = [
    'ENV_REVENIUM_TEAM_ID',
    'ENV_REVENIUM_METERING_BASE_URL',
    'ENV_REVENIUM_METERING_API_KEY',
    'DEFAULT_BASE_URL',
    'get_team_id',
    'get_base_url',
    'get_api_key',
]
