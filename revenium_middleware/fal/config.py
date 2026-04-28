"""
Configuration constants and settings for Revenium fal.ai middleware.
"""

import os
from importlib.metadata import version, PackageNotFoundError
from typing import Set

from revenium_middleware._core.config import Config as _CoreConfig
from revenium_middleware._core.config import SecurityConfig as _CoreSecurityConfig


def get_package_version() -> str:
    try:
        return version("revenium-python-sdk")
    except PackageNotFoundError:
        return "0.1.2"


def get_middleware_source() -> str:
    return f"revenium-python-sdk-fal@{get_package_version()}"


class Config(_CoreConfig):
    API_REQUEST_TIMEOUT: float = 120.0
    STREAM_TIMEOUT: float = 120.0
    ENV_FAL_KEY: str = "FAL_KEY"
    PROVIDER: str = "fal_ai"
    MODEL_SOURCE: str = "fal_ai"


class SecurityConfig(_CoreSecurityConfig):
    SENSITIVE_FIELDS: Set[str] = _CoreSecurityConfig.SENSITIVE_FIELDS | {
        "fal_key",
    }
