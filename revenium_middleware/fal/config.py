"""
Configuration constants and settings for Revenium fal.ai middleware.
"""

import os
from importlib.metadata import version, PackageNotFoundError
from typing import Set


def get_package_version() -> str:
    """Get the package version dynamically."""
    try:
        return version("revenium-python-sdk")
    except PackageNotFoundError:
        return "0.1.1"


def get_middleware_source() -> str:
    """Get the middleware source identifier with version."""
    return f"revenium-python-sdk-fal@{get_package_version()}"


class Config:
    """Configuration constants for Revenium fal.ai middleware."""

    # Threading and async timeouts
    THREAD_JOIN_TIMEOUT: float = 5.0
    API_REQUEST_TIMEOUT: float = 120.0
    BACKGROUND_THREAD_TIMEOUT: float = 5.0

    # Stream processing
    STREAM_CHUNK_BUFFER_SIZE: int = 1000
    STREAM_TIMEOUT: float = 120.0

    # Environment variable names
    ENV_FAL_KEY: str = "FAL_KEY"
    ENV_REVENIUM_API_KEY: str = "REVENIUM_METERING_API_KEY"
    ENV_LOG_LEVEL: str = "REVENIUM_LOG_LEVEL"

    # Provider identification
    PROVIDER: str = "fal_ai"
    MODEL_SOURCE: str = "fal_ai"


class SecurityConfig:
    """Security-related configuration."""

    SENSITIVE_FIELDS: Set[str] = {
        "api_key",
        "fal_key",
        "authorization",
        "bearer",
        "token",
        "password",
        "secret",
        "key",
        "credential",
    }
