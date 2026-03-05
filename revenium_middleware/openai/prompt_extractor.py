"""
Prompt extraction utilities for capturing AI prompts and responses.

This module provides functions to extract and truncate prompts from OpenAI API
requests and responses for optional storage in Revenium analytics.
"""

import json
import logging
from typing import Dict, Any, Optional, List

from .config import Config
from revenium_middleware._core.prompt_extraction import (  # noqa: F401 — re-exported
    extract_streaming_response_content,
)

logger = logging.getLogger("revenium_middleware.extension")


def extract_prompts_from_request(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract system prompt and input messages from OpenAI API request.
    
    Args:
        kwargs: The kwargs dict passed to the OpenAI API call
        
    Returns:
        Dict containing:
            - systemPrompt: String or None (system message content)
            - inputMessages: JSON string or None (non-system messages)
            - promptsTruncated: Boolean (True if any field was truncated)
    """
    messages = kwargs.get('messages', [])
    
    if not messages:
        return {
            'systemPrompt': None,
            'inputMessages': None,
            'promptsTruncated': False
        }
    
    system_prompt = None
    user_messages = []
    prompts_truncated = False
    
    # Separate system message from other messages
    for msg in messages:
        if not isinstance(msg, dict):
            continue
            
        role = msg.get('role')
        if role == 'system':
            # Extract system prompt (take first system message if multiple)
            if system_prompt is None:
                content = msg.get('content', '')
                if isinstance(content, str):
                    system_prompt = content
                elif isinstance(content, list):
                    # Handle content as array (multimodal)
                    system_prompt = json.dumps(content)
        else:
            # Collect all non-system messages
            user_messages.append(msg)
    
    # Apply truncation to system prompt
    # Keep total length at MAX_PROMPT_LENGTH by subtracting marker length
    if system_prompt and len(system_prompt) > Config.MAX_PROMPT_LENGTH:
        marker = "...[TRUNCATED]"
        marker_len = len(marker)
        truncate_at = Config.MAX_PROMPT_LENGTH - marker_len
        system_prompt = system_prompt[:truncate_at] + marker
        prompts_truncated = True
        logger.debug(f"System prompt truncated to {Config.MAX_PROMPT_LENGTH} characters")
    
    # Convert user messages to JSON string
    input_messages = None
    if user_messages:
        try:
            # First, truncate individual message contents to avoid invalid JSON
            # when the total exceeds the limit
            marker = "...[TRUNCATED]"
            marker_len = len(marker)
            truncated_messages = []
            for msg in user_messages:
                truncated_msg = msg.copy()
                content = msg.get('content', '')
                if isinstance(content, str) and len(content) > Config.MAX_PROMPT_LENGTH // 2:
                    # Truncate individual messages to half the limit to be safe
                    truncate_at = (Config.MAX_PROMPT_LENGTH // 2) - marker_len
                    truncated_msg['content'] = content[:truncate_at] + marker
                    prompts_truncated = True
                truncated_messages.append(truncated_msg)

            input_messages = json.dumps(truncated_messages, ensure_ascii=False)

            # Apply final truncation if still too long (keeps valid JSON structure)
            if len(input_messages) > Config.MAX_PROMPT_LENGTH:
                truncate_at = Config.MAX_PROMPT_LENGTH - marker_len
                input_messages = input_messages[:truncate_at] + marker
                prompts_truncated = True
                logger.debug(
                    f"Input messages truncated to "
                    f"{Config.MAX_PROMPT_LENGTH} characters"
                )
        except (TypeError, ValueError) as e:
            logger.warning(f"Failed to serialize input messages to JSON: {e}")
            input_messages = None
    
    return {
        'systemPrompt': system_prompt,
        'inputMessages': input_messages,
        'promptsTruncated': prompts_truncated
    }


def extract_response_content(response: Any, prompts_truncated: bool = False) -> Dict[str, Any]:
    """
    Extract output response content from OpenAI API response.
    
    Args:
        response: OpenAI API response object (ChatCompletion or similar)
        prompts_truncated: Whether prompts were already truncated (from request)
        
    Returns:
        Dict containing:
            - outputResponse: String or None (assistant response content)
            - promptsTruncated: Boolean (True if any field was truncated)
    """
    output_response = None
    was_truncated = prompts_truncated
    
    try:
        # Extract content from response.choices[0].message.content
        if hasattr(response, 'choices') and response.choices and len(response.choices) > 0:
            first_choice = response.choices[0]
            
            if hasattr(first_choice, 'message') and hasattr(first_choice.message, 'content'):
                content = first_choice.message.content
                
                if content:
                    if isinstance(content, str):
                        output_response = content
                    else:
                        # Handle non-string content (e.g., structured output)
                        output_response = json.dumps(content, ensure_ascii=False)
                    
                    # Apply truncation - keep total at MAX_PROMPT_LENGTH
                    if len(output_response) > Config.MAX_PROMPT_LENGTH:
                        marker = "...[TRUNCATED]"
                        marker_len = len(marker)
                        truncate_at = Config.MAX_PROMPT_LENGTH - marker_len
                        output_response = (
                            output_response[:truncate_at]
                            + marker
                        )
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



# extract_streaming_response_content is imported from _core.prompt_extraction

