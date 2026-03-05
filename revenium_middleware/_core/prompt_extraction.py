"""
Shared prompt extraction utilities.

This module provides prompt extraction functions shared across all provider
middlewares. Provider-specific prompt_extractor modules import from here.
"""

import logging
from typing import Dict, Any, Optional

from .config import Config

logger = logging.getLogger("revenium_middleware.extension")


def extract_streaming_response_content(
    accumulated_content: str,
    prompts_truncated: bool = False,
    max_prompt_length: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Extract output response content from accumulated streaming chunks.

    Args:
        accumulated_content: Accumulated text from streaming response
        prompts_truncated: Whether prompts were already truncated (from request)
        max_prompt_length: Maximum prompt length (defaults to Config.MAX_PROMPT_LENGTH)

    Returns:
        Dict containing:
            - outputResponse: String or None (assistant response content)
            - promptsTruncated: Boolean (True if any field was truncated)
    """
    if max_prompt_length is None:
        max_prompt_length = Config.MAX_PROMPT_LENGTH

    output_response = accumulated_content if accumulated_content else None
    was_truncated = prompts_truncated

    # Apply truncation - keep total at max_prompt_length
    if output_response and len(output_response) > max_prompt_length:
        marker = "...[TRUNCATED]"
        marker_len = len(marker)
        truncate_at = max_prompt_length - marker_len
        output_response = output_response[:truncate_at] + marker
        was_truncated = True
        logger.debug(
            f"Streaming output response truncated to "
            f"{max_prompt_length} characters"
        )

    return {
        'outputResponse': output_response,
        'promptsTruncated': was_truncated
    }
