"""
Trace visualization field capture and media type detection for fal.ai middleware.

This module provides functions to detect media types from fal.ai application names,
normalize model names to LiteLLM format, and capture trace visualization fields.
"""

import os
import re
import logging
from typing import Optional, Dict, Any

from revenium_middleware._core.trace_fields import (  # noqa: F401 — re-exported
    TICKET_ID_MAX_LENGTH,
    get_ticket_id,
    validate_ticket_id,
)

logger = logging.getLogger(__name__)

# Environment variable names
ENV_REVENIUM_ENVIRONMENT = "REVENIUM_ENVIRONMENT"
ENV_ENVIRONMENT = "ENVIRONMENT"
ENV_DEPLOYMENT_ENV = "DEPLOYMENT_ENV"

ENV_REVENIUM_REGION = "REVENIUM_REGION"
ENV_AWS_REGION = "AWS_REGION"
ENV_AWS_DEFAULT_REGION = "AWS_DEFAULT_REGION"
ENV_AZURE_REGION = "AZURE_REGION"
ENV_GCP_REGION = "GCP_REGION"
ENV_GOOGLE_CLOUD_REGION = "GOOGLE_CLOUD_REGION"

ENV_REVENIUM_CREDENTIAL_ALIAS = "REVENIUM_CREDENTIAL_ALIAS"
ENV_REVENIUM_TRACE_TYPE = "REVENIUM_TRACE_TYPE"
ENV_REVENIUM_TRACE_NAME = "REVENIUM_TRACE_NAME"
ENV_REVENIUM_PARENT_TRANSACTION_ID = "REVENIUM_PARENT_TRANSACTION_ID"
ENV_REVENIUM_TRANSACTION_NAME = "REVENIUM_TRANSACTION_NAME"
ENV_REVENIUM_RETRY_NUMBER = "REVENIUM_RETRY_NUMBER"

# Validation constants
TRACE_TYPE_MAX_LENGTH = 128
TRACE_NAME_MAX_LENGTH = 256
TRACE_TYPE_PATTERN = re.compile(r"^[a-zA-Z0-9_-]+$")

# Media type detection patterns
IMAGE_KEYWORDS = [
    "flux",
    "stable-diffusion",
    "sdxl",
    "dalle",
    "midjourney",
    "imagen",
    "image",
    "img",
    "photo",
    "picture",
    "illustration",
    "art",
    "draw",
    "paint",
    "controlnet",
    "inpaint",
    "outpaint",
    "upscale",
    "face",
    "portrait",
    "remove-background",
    "rembg",
]

VIDEO_KEYWORDS = [
    "video",
    "animate",
    "animation",
    "motion",
    "movie",
    "film",
    "clip",
    "deforum",
    "text-to-video",
    "img-to-video",
    "luma",
    "runway",
    "pika",
    "kling",
    "cogvideo",
    "svd",
    "stable-video",
]

AUDIO_KEYWORDS = [
    "audio",
    "sound",
    "music",
    "voice",
    "speech",
    "tts",
    "text-to-speech",
    "whisper",
    "transcribe",
    "stt",
    "speech-to-text",
    "bark",
    "elevenlabs",
    "coqui",
    "tortoise",
]

# LiteLLM naming constants
LITELLM_PREFIX = "fal_ai/"
FAL_ENDPOINT_PREFIX = "fal-ai/"


def get_environment() -> Optional[str]:
    """Get deployment environment from environment variables."""
    return (
        os.getenv(ENV_REVENIUM_ENVIRONMENT)
        or os.getenv(ENV_ENVIRONMENT)
        or os.getenv(ENV_DEPLOYMENT_ENV)
    )


def get_region() -> Optional[str]:
    """Get cloud region from environment variables."""
    region = os.getenv(ENV_REVENIUM_REGION)
    if region:
        return region
    region = os.getenv(ENV_AWS_REGION) or os.getenv(ENV_AWS_DEFAULT_REGION)
    if region:
        return region
    region = os.getenv(ENV_AZURE_REGION)
    if region:
        return region
    region = os.getenv(ENV_GCP_REGION) or os.getenv(ENV_GOOGLE_CLOUD_REGION)
    if region:
        return region
    return None


def get_credential_alias() -> Optional[str]:
    """Get credential alias from environment variables."""
    return os.getenv(ENV_REVENIUM_CREDENTIAL_ALIAS)


def get_trace_type() -> Optional[str]:
    """Get and validate trace type from environment variables."""
    trace_type = os.getenv(ENV_REVENIUM_TRACE_TYPE)
    if trace_type:
        return validate_trace_type(trace_type)
    return None


def get_trace_name() -> Optional[str]:
    """Get and validate trace name from environment variables."""
    trace_name = os.getenv(ENV_REVENIUM_TRACE_NAME)
    if trace_name:
        return validate_trace_name(trace_name)
    return None


def get_parent_transaction_id() -> Optional[str]:
    """Get parent transaction ID from environment variables."""
    return os.getenv(ENV_REVENIUM_PARENT_TRANSACTION_ID)


def get_transaction_name(usage_metadata: Optional[Dict[str, Any]] = None) -> Optional[str]:
    """Get transaction name with fallback to task_type."""
    transaction_name = os.getenv(ENV_REVENIUM_TRANSACTION_NAME)
    if transaction_name:
        return transaction_name
    if usage_metadata:
        transaction_name = usage_metadata.get("transactionName") or usage_metadata.get(
            "transaction_name"
        )
        if transaction_name:
            return transaction_name
        task_type = usage_metadata.get("task_type") or usage_metadata.get("taskType")
        if task_type:
            return task_type
    return None


def get_retry_number() -> int:
    """Get retry number from environment variables."""
    try:
        return int(os.getenv(ENV_REVENIUM_RETRY_NUMBER, "0"))
    except ValueError:
        logger.warning("Invalid REVENIUM_RETRY_NUMBER value, defaulting to 0")
        return 0


def validate_trace_type(trace_type: str) -> Optional[str]:
    """Validate trace type format and length."""
    if not trace_type:
        return None
    if len(trace_type) > TRACE_TYPE_MAX_LENGTH:
        logger.warning(
            "traceType exceeds maximum length of %d characters. Field will be omitted.",
            TRACE_TYPE_MAX_LENGTH,
        )
        return None
    if not TRACE_TYPE_PATTERN.match(trace_type):
        logger.warning(
            "traceType contains invalid characters. Field will be omitted."
        )
        return None
    return trace_type


def validate_trace_name(trace_name: str) -> Optional[str]:
    """Validate trace name length and truncate if needed."""
    if not trace_name:
        return None
    if len(trace_name) > TRACE_NAME_MAX_LENGTH:
        logger.warning(
            "traceName exceeds maximum length of %d characters. Truncating.",
            TRACE_NAME_MAX_LENGTH,
        )
        return trace_name[:TRACE_NAME_MAX_LENGTH]
    return trace_name


def detect_media_type(application: str) -> str:
    """
    Detect the media type (image, video, audio) from the fal.ai application name.

    Args:
        application: The fal.ai application/model ID (e.g., "fal-ai/flux/dev")

    Returns:
        Media type string: 'image', 'video', 'audio', or 'generation' (default)
    """
    if not application:
        return "generation"

    app_lower = application.lower()

    # Check for video keywords first (more specific)
    for keyword in VIDEO_KEYWORDS:
        if keyword in app_lower:
            return "video"

    # Check for audio keywords
    for keyword in AUDIO_KEYWORDS:
        if keyword in app_lower:
            return "audio"

    # Check for image keywords
    for keyword in IMAGE_KEYWORDS:
        if keyword in app_lower:
            return "image"

    # Default to image since most fal.ai models are image generators
    return "image"


def normalize_model_name(model: str) -> str:
    """
    Normalize the model name to LiteLLM naming convention: "fal_ai/{fal_endpoint_id}".

    This matches the format stored in the AIModel table after LiteLLM sync,
    enabling the pricing dimension lookup query to find matching pricing entries.

    Examples:
        "flux/dev"                 -> "fal_ai/fal-ai/flux/dev"
        "fal-ai/flux/dev"          -> "fal_ai/fal-ai/flux/dev"
        "fal_ai/fal-ai/flux/dev"   -> "fal_ai/fal-ai/flux/dev" (already correct)
        "fal_ai/flux/dev"          -> "fal_ai/fal-ai/flux/dev" (missing inner segment)

    Args:
        model: The raw model/application name from fal.ai

    Returns:
        Normalized model name in LiteLLM format
    """
    if not model:
        return model

    # Already in full LiteLLM format (fal_ai/fal-ai/...)
    if model.startswith(LITELLM_PREFIX + FAL_ENDPOINT_PREFIX):
        return model

    # Has fal_ai/ prefix but missing fal-ai/ segment (e.g., "fal_ai/flux/dev")
    if model.startswith(LITELLM_PREFIX):
        remainder = model[len(LITELLM_PREFIX):]
        logger.warning(
            "Model name '%s' has 'fal_ai/' prefix but missing 'fal-ai/' segment. Auto-normalizing.",
            model,
        )
        return LITELLM_PREFIX + FAL_ENDPOINT_PREFIX + remainder

    # Has fal-ai/ endpoint prefix but not litellm prefix — prepend fal_ai/
    if model.startswith(FAL_ENDPOINT_PREFIX):
        return LITELLM_PREFIX + model

    # Bare model name (e.g., "flux/dev") — add both prefixes
    logger.warning(
        "Model name '%s' is missing 'fal-ai/' prefix. Auto-normalizing to '%s%s%s'",
        model,
        LITELLM_PREFIX,
        FAL_ENDPOINT_PREFIX,
        model,
    )
    return LITELLM_PREFIX + FAL_ENDPOINT_PREFIX + model
