"""
Configuration for Revenium LiteLLM middleware.

Imports shared config from core and adds LiteLLM-specific settings.
"""

import os
from typing import Optional

from revenium_middleware._core.config import (  # noqa: F401
    Config as _CoreConfig,
    SummaryFormat,
    parse_print_summary_value,
    get_print_summary_config,
    get_team_id,
    get_base_url,
)

# Re-export environment variable names for backward compatibility
ENV_REVENIUM_PRINT_SUMMARY = _CoreConfig.ENV_REVENIUM_PRINT_SUMMARY
ENV_REVENIUM_TEAM_ID = _CoreConfig.ENV_REVENIUM_TEAM_ID
ENV_REVENIUM_METERING_BASE_URL = _CoreConfig.ENV_REVENIUM_BASE_URL
ENV_REVENIUM_METERING_API_KEY = _CoreConfig.ENV_REVENIUM_API_KEY

# Summary settings (LiteLLM uses a shorter retry delay)
SUMMARY_RETRY_ATTEMPTS = _CoreConfig.SUMMARY_RETRY_ATTEMPTS
SUMMARY_RETRY_DELAY: float = 1.0  # LiteLLM uses 1.0s vs core's 2.0s
SUMMARY_API_TIMEOUT = _CoreConfig.SUMMARY_API_TIMEOUT

DEFAULT_BASE_URL = _CoreConfig.DEFAULT_BASE_URL


def get_api_key() -> Optional[str]:
    """Get the Revenium API key from environment."""
    return os.getenv(ENV_REVENIUM_METERING_API_KEY)


__all__ = [
    'ENV_REVENIUM_PRINT_SUMMARY',
    'ENV_REVENIUM_TEAM_ID',
    'ENV_REVENIUM_METERING_BASE_URL',
    'ENV_REVENIUM_METERING_API_KEY',
    'SUMMARY_RETRY_ATTEMPTS',
    'SUMMARY_RETRY_DELAY',
    'SUMMARY_API_TIMEOUT',
    'DEFAULT_BASE_URL',
    'SummaryFormat',
    'parse_print_summary_value',
    'get_print_summary_config',
    'get_team_id',
    'get_base_url',
    'get_api_key',
]
