"""
Prompt extraction utilities for capturing AI prompts and responses.

This module provides functions to extract and truncate prompts from Anthropic API
requests and responses for optional storage in Revenium analytics.
"""

import json
import logging
from typing import Dict, Any, Optional, List, Tuple

from .config import Config
from revenium_middleware._core.config import is_capture_prompts_enabled
from revenium_middleware._core.prompt_extraction import (  # noqa: F401  re-exported
    extract_streaming_response_content,
    serialize_messages_within_limit,
)

logger = logging.getLogger("revenium_middleware.extension")


def extract_prompts_from_request(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract system prompt and input messages from Anthropic API request.
    
    Args:
        kwargs: The kwargs dict passed to the Anthropic API call
        
    Returns:
        Dict containing:
            - systemPrompt: String or None (system message content)
            - inputMessages: JSON string or None (user/assistant messages)
            - promptsTruncated: Boolean (True if any field was truncated)
    """
    # Extract system prompt (Anthropic has separate 'system' parameter)
    system_prompt = kwargs.get('system')
    messages = kwargs.get('messages', [])
    prompts_truncated = False
    
    # Apply truncation to system prompt
    if system_prompt:
        if isinstance(system_prompt, list):
            # System can be an array of text blocks
            try:
                system_prompt = json.dumps(system_prompt, ensure_ascii=False)
            except (TypeError, ValueError) as e:
                logger.warning(f"Failed to serialize system prompt to JSON: {e}")
                system_prompt = str(system_prompt)
        
        if len(system_prompt) > Config.MAX_PROMPT_LENGTH:
            marker = "...[TRUNCATED]"
            marker_len = len(marker)
            truncate_at = Config.MAX_PROMPT_LENGTH - marker_len
            system_prompt = system_prompt[:truncate_at] + marker
            prompts_truncated = True
            logger.debug(f"System prompt truncated to {Config.MAX_PROMPT_LENGTH} characters")
    
    input_messages = None
    if messages:
        dict_messages = [msg for msg in messages if isinstance(msg, dict)]
        input_messages, input_truncated = serialize_messages_within_limit(dict_messages, Config.MAX_PROMPT_LENGTH)
        prompts_truncated = prompts_truncated or input_truncated

    return {
        'systemPrompt': system_prompt,
        'inputMessages': input_messages,
        'promptsTruncated': prompts_truncated
    }


def extract_response_content(response: Any, prompts_truncated: bool = False) -> Dict[str, Any]:
    """
    Extract output response content from Anthropic API response.
    
    Args:
        response: Anthropic API response object (Message)
        prompts_truncated: Whether prompts were already truncated (from request)
        
    Returns:
        Dict containing:
            - outputResponse: String or None (assistant response content)
            - promptsTruncated: Boolean (True if any field was truncated)
    """
    output_response = None
    was_truncated = prompts_truncated
    
    try:
        # Extract content from response.content (array of content blocks)
        if hasattr(response, 'content') and response.content:
            content_blocks = []

            for block in response.content:
                if hasattr(block, 'type'):
                    if block.type == 'text' and hasattr(block, 'text'):
                        content_blocks.append(block.text)
                    else:
                        # Handle other content types (tool_use, etc.)
                        try:
                            content_blocks.append(json.dumps(block.__dict__, ensure_ascii=False))
                        except Exception:
                            content_blocks.append(str(block))

            # Join all content blocks
            if content_blocks:
                output_response = '\n'.join(content_blocks)

                # Apply truncation
                if len(output_response) > Config.MAX_PROMPT_LENGTH:
                    marker = "...[TRUNCATED]"
                    marker_len = len(marker)
                    truncate_at = Config.MAX_PROMPT_LENGTH - marker_len
                    output_response = output_response[:truncate_at] + marker
                    was_truncated = True
                    logger.debug(
                        f"Output response truncated to "
                        f"{Config.MAX_PROMPT_LENGTH} characters"
                    )
    except Exception as e:
        logger.warning(f"Failed to extract response content: {e}")
        output_response = None

    return {
        'outputResponse': output_response,
        'promptsTruncated': was_truncated
    }


def extract_prompt_data_if_enabled(
    request_body: Optional[Dict[str, Any]],
    response: Any = None,
    accumulated_content: str = None
) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[bool]]:
    """
    Extract prompt data if capture is enabled.

    Args:
        request_body: Request body containing messages and system prompt
        response: API response object (for non-streaming)
        accumulated_content: Accumulated streaming content (for streaming)

    Returns:
        Tuple of (system_prompt, input_messages, output_response, prompts_truncated)
    """
    if not is_capture_prompts_enabled() or not request_body:
        return None, None, None, None

    # Extract prompts from request
    prompt_data = extract_prompts_from_request(request_body)
    system_prompt = prompt_data.get('systemPrompt')
    input_messages = prompt_data.get('inputMessages')
    prompts_truncated = prompt_data.get('promptsTruncated', False)

    # Extract response content
    if accumulated_content is not None:
        # Streaming response
        response_data = extract_streaming_response_content(
            accumulated_content, prompts_truncated
        )
    elif response is not None:
        # Non-streaming response
        response_data = extract_response_content(response, prompts_truncated)
    else:
        response_data = {'outputResponse': None, 'promptsTruncated': prompts_truncated}

    output_response = response_data.get('outputResponse')
    prompts_truncated = response_data.get('promptsTruncated', prompts_truncated)

    logger.debug(
        f"Prompt capture - system_prompt: {bool(system_prompt)}, "
        f"input_messages: {bool(input_messages)}, "
        f"output_response: {bool(output_response)}, "
        f"truncated: {prompts_truncated}"
    )

    return system_prompt, input_messages, output_response, prompts_truncated


def captured_prompt_fields(
    request_body: Optional[Dict[str, Any]],
    response: Any = None,
    accumulated_content: Optional[str] = None
) -> Dict[str, Any]:
    """The prompt capture fields for a usage record, empty when capture is off.

    Fields with no value are left out, so a record built with capture off is
    exactly the record built before capture existed.
    """
    try:
        system_prompt, input_messages, output_response, prompts_truncated = extract_prompt_data_if_enabled(
            request_body, response=response, accumulated_content=accumulated_content
        )
    except Exception as exc:  # noqa: BLE001 - capture must never cost the usage record
        logger.warning("Prompt capture skipped: %s", type(exc).__name__)
        return {}
    fields = {
        "system_prompt": system_prompt,
        "input_messages": input_messages,
        "output_response": output_response,
        "prompts_truncated": prompts_truncated,
    }
    return {name: value for name, value in fields.items() if value is not None}

