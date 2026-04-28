"""
Prompt extraction utilities for capturing AI prompts and responses.

This module provides functions to extract and truncate prompts from Anthropic API
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
    
    # Convert messages to JSON string
    input_messages = None
    if messages:
        try:
            # First, truncate individual message contents to avoid invalid JSON
            marker = "...[TRUNCATED]"
            marker_len = len(marker)
            truncated_messages = []
            
            for msg in messages:
                if not isinstance(msg, dict):
                    continue
                    
                truncated_msg = msg.copy()
                content = msg.get('content', '')
                
                # Handle content as string
                if isinstance(content, str) and len(content) > Config.MAX_PROMPT_LENGTH // 2:
                    truncate_at = (Config.MAX_PROMPT_LENGTH // 2) - marker_len
                    truncated_msg['content'] = content[:truncate_at] + marker
                    prompts_truncated = True
                # Handle content as array (multimodal - text, images, etc.)
                elif isinstance(content, list):
                    truncated_content = []
                    for block in content:
                        if isinstance(block, dict):
                            truncated_block = block.copy()
                            # Truncate text blocks
                            if block.get('type') == 'text':
                                text = block.get('text', '')
                                if len(text) > Config.MAX_PROMPT_LENGTH // 2:
                                    truncate_at = (Config.MAX_PROMPT_LENGTH // 2) - marker_len
                                    truncated_block['text'] = text[:truncate_at] + marker
                                    prompts_truncated = True
                            truncated_content.append(truncated_block)
                        else:
                            truncated_content.append(block)
                    truncated_msg['content'] = truncated_content
                
                truncated_messages.append(truncated_msg)

            input_messages = json.dumps(truncated_messages, ensure_ascii=False)

            # Apply final truncation if still too long
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



# extract_streaming_response_content is imported from _core.prompt_extraction

