"""
Trace visualization field capture and validation.

This module provides functions to capture trace visualization fields from
environment variables and validate them according to the specification.

Shared functions are imported from _core.trace_fields. This module retains
only Google-specific functions: detect_vision_content and detect_operation_type.
"""

import logging
from typing import Optional, Dict, Any, Union, Literal

from revenium_middleware._core.config import (
    Config as _CoreConfig,
    get_team_id,
    get_base_url,
)
from revenium_middleware._core.trace_fields import (  # noqa: F401 — re-exported
    TRACE_TYPE_MAX_LENGTH,
    TRACE_NAME_MAX_LENGTH,
    TICKET_ID_MAX_LENGTH,
    TRACE_TYPE_PATTERN,
    ENV_REVENIUM_ENVIRONMENT,
    ENV_ENVIRONMENT,
    ENV_DEPLOYMENT_ENV,
    ENV_REVENIUM_REGION,
    ENV_AWS_REGION,
    ENV_AWS_DEFAULT_REGION,
    ENV_AZURE_REGION,
    ENV_GCP_REGION,
    ENV_GOOGLE_CLOUD_REGION,
    ENV_REVENIUM_CREDENTIAL_ALIAS,
    ENV_REVENIUM_TRACE_TYPE,
    ENV_REVENIUM_TRACE_NAME,
    ENV_REVENIUM_PARENT_TRANSACTION_ID,
    ENV_REVENIUM_TRANSACTION_NAME,
    ENV_REVENIUM_RETRY_NUMBER,
    ENV_REVENIUM_TICKET_ID,
    get_environment,
    get_region,
    get_credential_alias,
    get_trace_type,
    get_trace_name,
    get_parent_transaction_id,
    get_transaction_name,
    get_retry_number,
    get_ticket_id,
    validate_trace_type,
    validate_trace_name,
    validate_ticket_id,
)

logger = logging.getLogger(__name__)

# Revenium API targeting (re-exported from core)
ENV_REVENIUM_TEAM_ID = _CoreConfig.ENV_REVENIUM_TEAM_ID
ENV_REVENIUM_METERING_BASE_URL = _CoreConfig.ENV_REVENIUM_BASE_URL
DEFAULT_BASE_URL = _CoreConfig.DEFAULT_BASE_URL


def detect_vision_content(contents: Any = None) -> bool:
    """
    Detect if contents contain vision/image content.

    Google AI content formats that indicate vision input:
    - Part with inline_data containing image/* or video/* MIME type
    - Part with file_data containing image/* or video/* MIME type
    - PIL Image objects passed directly
    - google.genai types.Part with inline_data/file_data

    Args:
        contents: The contents parameter from generate_content() call.
            Can be a string, list, Part, or Content object.

    Returns:
        True if any content contains image/video data, False otherwise.
    """
    if contents is None:
        return False

    # Normalize to a list for uniform processing
    if not isinstance(contents, list):
        contents = [contents]

    for item in contents:
        if _item_has_vision_content(item, _depth=0):
            return True

    return False


_MAX_VISION_RECURSION_DEPTH = 10


def _item_has_vision_content(item: Any, _depth: int = 0) -> bool:
    """Check if a single content item contains vision data."""
    if item is None or _depth > _MAX_VISION_RECURSION_DEPTH:
        return False

    # Check for PIL Image objects
    try:
        import PIL.Image
        if isinstance(item, PIL.Image.Image):
            return True
    except ImportError:
        pass

    # Check for dict-based content (raw API format)
    if isinstance(item, dict):
        # Check inline_data
        inline_data = item.get("inline_data")
        if inline_data and isinstance(inline_data, dict):
            mime_type = inline_data.get("mime_type", "")
            if mime_type.startswith("image/") or mime_type.startswith("video/"):
                return True

        # Check file_data
        file_data = item.get("file_data")
        if file_data and isinstance(file_data, dict):
            mime_type = file_data.get("mime_type", "")
            if mime_type.startswith("image/") or mime_type.startswith("video/"):
                return True

        # Check parts list within a Content dict
        parts = item.get("parts")
        if parts and isinstance(parts, list):
            for part in parts:
                if _item_has_vision_content(part, _depth + 1):
                    return True

        # Check type field (e.g., {"type": "image", ...})
        if item.get("type") in ("image", "video"):
            return True

        return False

    # Check for google.genai types (Part objects with inline_data/file_data attributes)
    if hasattr(item, "inline_data") and item.inline_data is not None:
        mime_type = getattr(item.inline_data, "mime_type", "") or ""
        if mime_type.startswith("image/") or mime_type.startswith("video/"):
            return True

    if hasattr(item, "file_data") and item.file_data is not None:
        mime_type = getattr(item.file_data, "mime_type", "") or ""
        if mime_type.startswith("image/") or mime_type.startswith("video/"):
            return True

    # Check Content objects with parts
    if hasattr(item, "parts") and item.parts:
        parts = item.parts if isinstance(item.parts, list) else [item.parts]
        for part in parts:
            if _item_has_vision_content(part, _depth + 1):
                return True

    # Check for bytes (raw image data passed directly)
    if isinstance(item, bytes) and len(item) > 0:
        # Check for common image magic bytes
        if (item[:4] == b'\x89PNG' or  # PNG
                item[:2] == b'\xff\xd8' or  # JPEG
                item[:4] == b'GIF8' or  # GIF
                (len(item) >= 12 and item[:4] == b'RIFF' and item[8:12] == b'WEBP')):  # WebP
            return True

    return False


def detect_operation_type(
    method_name: str,
    request_body: Optional[Dict[str, Any]] = None
) -> str:
    """
    Auto-detect operation type from method name and request.

    Args:
        method_name: API method name (e.g., 'generate_content', 'embed_content')
        request_body: Optional request body to check for tools

    Returns:
        Operation type string ('CHAT', 'GENERATE', 'EMBED', 'TOOL_CALL')
    """
    request_body = request_body or {}

    # Embeddings
    if 'embed' in method_name.lower():
        return 'EMBED'

    # Content generation (chat/generate)
    if 'generate' in method_name.lower() or 'chat' in method_name.lower():
        # Check for tools in request
        has_tools = request_body.get('tools')
        if has_tools:
            return 'TOOL_CALL'
        return 'CHAT'

    # Default fallback
    return 'CHAT'


# get_team_id, get_base_url are imported from revenium_middleware._core.config
# at the top of this file
