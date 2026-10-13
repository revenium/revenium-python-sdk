"""
Trace visualization field capture and media type detection for fal.ai middleware.

This module provides functions to detect media types from fal.ai application names,
normalize model names to LiteLLM format, and capture trace visualization fields.
"""

import re
import logging
from typing import Any

from revenium_middleware._core.trace_fields import (  # noqa: F401 — re-exported
    TICKET_ID_MAX_LENGTH,
    AGENT_VERSION_MAX_LENGTH,
    get_environment,
    get_region,
    get_credential_alias,
    get_trace_type,
    get_trace_name,
    get_parent_transaction_id,
    get_transaction_name,
    get_retry_number,
    get_ticket_id,
    get_agent_version,
    validate_trace_type,
    validate_trace_name,
    validate_ticket_id,
    validate_agent_version,
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

LANGUAGE_MODEL_ENDPOINT_PATTERN = re.compile(r"openrouter|llm|text-generation", re.IGNORECASE)
RESULT_MEDIA_KEYS = ("images", "image", "video", "audio", "audio_url")

INPUT_TOKEN_KEYS = ("prompt_tokens", "input_tokens")
OUTPUT_TOKEN_KEYS = ("completion_tokens", "output_tokens")
TOTAL_TOKEN_KEY = "total_tokens"
TOKEN_USAGE_KEYS = INPUT_TOKEN_KEYS + OUTPUT_TOKEN_KEYS + (TOTAL_TOKEN_KEY,)

# LiteLLM naming constants
LITELLM_PREFIX = "fal_ai/"
FAL_ENDPOINT_PREFIX = "fal-ai/"


def _result_has_media(result: Any) -> bool:
    return isinstance(result, dict) and any(result.get(key) for key in RESULT_MEDIA_KEYS)


def _is_positive_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0


def _result_reports_token_usage(result: Any) -> bool:
    usage = result.get("usage") if isinstance(result, dict) else None
    if not isinstance(usage, dict):
        return False
    return any(_is_positive_number(usage.get(key)) for key in TOKEN_USAGE_KEYS)


def detect_media_type(application: str, result: Any = None) -> str:
    """
    Detect the record type (chat, image, video, audio) of a fal.ai call.

    Args:
        application: The fal.ai application/model ID (e.g., "fal-ai/flux/dev")
        result: The call's result, when there is one. Only a dict is inspected.

    Returns:
        Media type string: 'chat', 'image', 'video', 'audio', or 'generation'
        (no application name)
    """
    if not application:
        return "generation"

    app_lower = application.lower()
    has_media = _result_has_media(result)

    if LANGUAGE_MODEL_ENDPOINT_PATTERN.search(application) and not has_media:
        return "chat"

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

    # Token usage is trusted only here, for an endpoint no keyword list claims:
    # image and audio endpoints can report usage too and keep their own record.
    if not has_media and _result_reports_token_usage(result):
        return "chat"

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
