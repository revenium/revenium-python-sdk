"""
Shared configuration constants and settings for Revenium middleware.

This module is the single source of truth for configuration values shared
across all provider middlewares. Provider-specific config files extend
this via class inheritance and re-export symbols for backward compatibility.
"""

import logging
import os
from typing import Set, Literal, Union, Optional

logger = logging.getLogger(__name__)

# Type alias for summary format
SummaryFormat = Literal["human", "json"]


class Config:
    """Shared configuration constants for all Revenium middleware providers."""

    # Threading and async timeouts
    THREAD_JOIN_TIMEOUT: float = 5.0
    API_REQUEST_TIMEOUT: float = 30.0
    BACKGROUND_THREAD_TIMEOUT: float = 5.0

    # Cache settings
    PROVIDER_CACHE_TTL: int = 3600  # 1 hour
    MODEL_CACHE_TTL: int = 3600     # 1 hour

    # Logging and security
    MAX_LOG_STRING_LENGTH: int = 100
    MAX_SANITIZATION_DEPTH: int = 3

    # Stream processing
    STREAM_CHUNK_BUFFER_SIZE: int = 1000
    STREAM_TIMEOUT: float = 30.0

    # Retry and resilience
    MAX_RETRY_ATTEMPTS: int = 3
    RETRY_BACKOFF_FACTOR: float = 1.5
    INITIAL_RETRY_DELAY: float = 0.1

    # Core environment variable names
    ENV_REVENIUM_API_KEY: str = "REVENIUM_METERING_API_KEY"
    ENV_LOG_LEVEL: str = "REVENIUM_LOG_LEVEL"

    # Trace visualization environment variables
    ENV_REVENIUM_ENVIRONMENT: str = "REVENIUM_ENVIRONMENT"
    ENV_ENVIRONMENT: str = "ENVIRONMENT"
    ENV_DEPLOYMENT_ENV: str = "DEPLOYMENT_ENV"
    ENV_REVENIUM_REGION: str = "REVENIUM_REGION"
    ENV_REVENIUM_CREDENTIAL_ALIAS: str = "REVENIUM_CREDENTIAL_ALIAS"
    ENV_REVENIUM_TRACE_TYPE: str = "REVENIUM_TRACE_TYPE"
    ENV_REVENIUM_TRACE_NAME: str = "REVENIUM_TRACE_NAME"
    ENV_REVENIUM_PARENT_TRANSACTION_ID: str = "REVENIUM_PARENT_TRANSACTION_ID"
    ENV_REVENIUM_TRANSACTION_NAME: str = "REVENIUM_TRANSACTION_NAME"
    ENV_REVENIUM_RETRY_NUMBER: str = "REVENIUM_RETRY_NUMBER"

    # Prompt capture settings
    ENV_REVENIUM_CAPTURE_PROMPTS: str = "REVENIUM_CAPTURE_PROMPTS"
    CAPTURE_PROMPTS: bool = os.getenv("REVENIUM_CAPTURE_PROMPTS", "false").lower() in ("true", "1", "yes", "on")
    MAX_PROMPT_LENGTH: int = 50_000  # Maximum characters per prompt field

    # Terminal summary settings
    ENV_REVENIUM_PRINT_SUMMARY: str = "REVENIUM_PRINT_SUMMARY"
    ENV_REVENIUM_TEAM_ID: str = "REVENIUM_TEAM_ID"
    ENV_REVENIUM_BASE_URL: str = "REVENIUM_METERING_BASE_URL"
    SUMMARY_RETRY_ATTEMPTS: int = 3
    SUMMARY_RETRY_DELAY: float = 2.0
    SUMMARY_API_TIMEOUT: float = 5.0
    DEFAULT_BASE_URL: str = "https://api.revenium.ai"

    # Enforcement / circuit breaker — see _core/enforcement.py.
    # Taxonomy locked in BACK-1154 docs; keep in sync across SDKs.
    ENV_REVENIUM_BYPASS: str = "REVENIUM_BYPASS"
    ENV_CIRCUIT_BREAKER_ENABLED: str = "REVENIUM_CIRCUIT_BREAKER_ENABLED"
    ENV_REVENIUM_ENFORCEMENT_BASE_URL: str = "REVENIUM_ENFORCEMENT_BASE_URL"
    ENV_REVENIUM_CB_POLL_INTERVAL_SECONDS: str = "REVENIUM_CB_POLL_INTERVAL_SECONDS"
    ENV_REVENIUM_CB_FAIL_MODE: str = "REVENIUM_CB_FAIL_MODE"
    ENV_REVENIUM_CACHE_DIR: str = "REVENIUM_CACHE_DIR"


class SecurityConfig:
    """Security-related configuration shared across all providers."""

    SENSITIVE_FIELDS: Set[str] = {
        'api_key', 'subscriber_credential', 'subscriber_email', 'messages',
        'input', 'content', 'prompt', 'text', 'data', 'authorization',
        'x-api-key', 'bearer', 'token', 'password', 'secret', 'key',
        'credential', 'auth', 'private',
        'system_prompt', 'input_messages', 'output_response'
    }

    SENSITIVE_PATTERNS: Set[str] = {
        'sk-', 'pk-', 'Bearer ', 'Basic ', 'Token '
    }


def is_selective_metering_enabled() -> bool:
    """
    Check if selective metering is enabled.

    When enabled, only functions decorated with @revenium_meter will be metered.
    When disabled (default), all API calls are metered automatically.

    The setting is controlled by the REVENIUM_SELECTIVE_METERING environment variable.
    Accepted values for enabled: "true", "1", "yes", "on" (case-insensitive)
    """
    env_value = os.environ.get("REVENIUM_SELECTIVE_METERING", "false").lower()
    return env_value in ("true", "1", "yes", "on")


def get_config_value(key: str, default: any = None) -> any:
    """Get configuration value from environment or use default."""
    return os.getenv(key, default)


def is_debug_enabled() -> bool:
    """Check if debug logging is enabled."""
    log_level = get_config_value(Config.ENV_LOG_LEVEL, "INFO").upper()
    return log_level == "DEBUG"


def get_timeout_config() -> dict:
    """Get base timeout-related configuration."""
    return {
        'thread_join': Config.THREAD_JOIN_TIMEOUT,
        'api_request': Config.API_REQUEST_TIMEOUT,
        'background_thread': Config.BACKGROUND_THREAD_TIMEOUT,
        'stream': Config.STREAM_TIMEOUT,
    }


def parse_print_summary_value(value: Optional[str]) -> Union[bool, SummaryFormat]:
    """
    Parse REVENIUM_PRINT_SUMMARY environment variable value.

    Returns:
        False if disabled, 'human' or 'json' if enabled
    """
    if value is None:
        return False

    value_lower = value.lower().strip()

    if value_lower in ('false', '0', 'no', 'off', 'disabled', ''):
        return False
    elif value_lower in ('true', '1', 'yes', 'on', 'enabled', 'human'):
        return 'human'
    elif value_lower == 'json':
        return 'json'
    else:
        logger.warning(
            f"Invalid REVENIUM_PRINT_SUMMARY value '{value}'. "
            f"Expected 'true', 'human', 'json', or 'false'. Defaulting to disabled."
        )
        return False


def get_print_summary_config() -> Union[bool, SummaryFormat]:
    """Get print summary configuration from environment."""
    value = get_config_value(Config.ENV_REVENIUM_PRINT_SUMMARY)
    return parse_print_summary_value(value)


def get_team_id() -> Optional[str]:
    """Get Revenium team ID from environment."""
    return get_config_value(Config.ENV_REVENIUM_TEAM_ID)


def get_base_url() -> str:
    """Get Revenium base URL from environment (defaults to https://api.revenium.ai)."""
    return get_config_value(Config.ENV_REVENIUM_BASE_URL, Config.DEFAULT_BASE_URL)
