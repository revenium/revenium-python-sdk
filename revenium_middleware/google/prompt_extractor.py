"""
Prompt extraction utilities for Google Gemini API.

This module provides functions to extract system instructions, input messages,
and output responses from Google Gemini API requests and responses.

Google Gemini API Format:
- system_instruction: Separate field with parts array
- contents: Array of content objects with role and parts
- parts: Array of text/data objects within each content
"""

import json
import logging
from typing import Dict, Any, Optional, Tuple

from .config import Config
from revenium_middleware._core.prompt_extraction import (
    extract_streaming_response_content as _core_extract_streaming_response_content,
)

logger = logging.getLogger("revenium_middleware.extension")


def _sanitize_content_dict(content: Dict[str, Any]) -> Dict[str, Any]:
    """
    Sanitize a content dict to remove binary data and non-JSON-serializable values.

    Args:
        content: Content dict that may contain inline_data or other binary fields

    Returns:
        Sanitized content dict safe for JSON serialization
    """
    sanitized = {'role': content.get('role', 'user'), 'parts': []}

    for part in content.get('parts', []):
        if isinstance(part, dict):
            # Check for text content
            if 'text' in part:
                sanitized['parts'].append({'text': part['text']})
            # Check for binary data fields and replace with placeholder
            elif 'inline_data' in part:
                sanitized['parts'].append({'inline_data': '[BINARY_DATA]'})
            elif 'file_data' in part:
                sanitized['parts'].append({'file_data': '[FILE_DATA]'})
            elif 'function_call' in part:
                # Keep function calls but sanitize them
                sanitized['parts'].append({'function_call': '[FUNCTION_CALL]'})
            elif 'function_response' in part:
                sanitized['parts'].append({'function_response': '[FUNCTION_RESPONSE]'})
            else:
                # Unknown part type - include only safe string representation
                sanitized['parts'].append({'unknown': '[UNKNOWN_PART_TYPE]'})
        else:
            # Non-dict part - skip it
            logger.debug(f"Skipping non-dict part in content: {type(part)}")

    return sanitized


def extract_prompts_from_request(
    kwargs: Dict[str, Any],
    config: Optional[Any] = None,
    args: Optional[Tuple] = None
) -> Dict[str, Any]:
    """
    Extract system instruction and input contents from Google Gemini API request.

    Args:
        kwargs: The kwargs dict passed to the Google Gemini API call
        config: Optional GenerateContentConfig object (for google-genai SDK)
        args: Optional positional arguments tuple (first arg is usually contents)

    Returns:
        Dict containing:
            - systemPrompt: String or None (system instruction content)
            - inputMessages: JSON string or None (contents array)
            - promptsTruncated: Boolean (True if any field was truncated)
    """
    system_prompt = None
    input_messages = None
    prompts_truncated = False
    marker = "...[TRUNCATED]"
    marker_len = len(marker)

    # Extract system instruction
    # Can be in config object (google-genai) or kwargs (both SDKs)
    system_instruction = None

    # Try config object first (google-genai SDK)
    if config and hasattr(config, 'system_instruction'):
        system_instruction = config.system_instruction

    # Try kwargs (both SDKs support this)
    if not system_instruction and 'system_instruction' in kwargs:
        system_instruction = kwargs.get('system_instruction')

    # Extract text from system_instruction
    if system_instruction:
        try:
            if isinstance(system_instruction, str):
                system_prompt = system_instruction
            elif hasattr(system_instruction, 'parts'):
                # system_instruction is a Content object with parts
                parts_text = []
                for part in system_instruction.parts:
                    if hasattr(part, 'text') and part.text:
                        parts_text.append(part.text)
                    elif isinstance(part, dict) and 'text' in part:
                        parts_text.append(part['text'])
                if parts_text:
                    system_prompt = '\n'.join(parts_text)
            elif isinstance(system_instruction, dict) and 'parts' in system_instruction:
                # system_instruction is a dict with parts array
                parts_text = []
                for part in system_instruction['parts']:
                    if isinstance(part, dict) and 'text' in part:
                        parts_text.append(part['text'])
                if parts_text:
                    system_prompt = '\n'.join(parts_text)

            # Apply truncation to system prompt
            if system_prompt and len(system_prompt) > Config.MAX_PROMPT_LENGTH:
                truncate_at = Config.MAX_PROMPT_LENGTH - marker_len
                system_prompt = system_prompt[:truncate_at] + marker
                prompts_truncated = True
                logger.debug(
                    f"System instruction truncated to {Config.MAX_PROMPT_LENGTH} characters"
                )
        except Exception as e:
            logger.warning(f"Failed to extract system instruction: {e}")
            system_prompt = None

    # Extract contents (input messages)
    # Try kwargs first, then fall back to positional args
    contents = kwargs.get('contents')
    if not contents and args and len(args) > 0:
        # First positional argument is typically the contents
        contents = args[0]
    if contents:
        try:
            # Normalize contents to list of dicts
            normalized_contents = []
            
            # Handle different input formats
            if isinstance(contents, str):
                # Simple string input
                normalized_contents.append({
                    'role': 'user',
                    'parts': [{'text': contents}]
                })
            elif isinstance(contents, list):
                for content in contents:
                    if isinstance(content, str):
                        # String in list
                        normalized_contents.append({
                            'role': 'user',
                            'parts': [{'text': content}]
                        })
                    elif isinstance(content, dict):
                        # Already a dict - sanitize it to prevent binary data leakage
                        normalized_contents.append(_sanitize_content_dict(content))
                    elif hasattr(content, 'role') and hasattr(content, 'parts'):
                        # Content object - convert to dict
                        content_dict = {'role': content.role, 'parts': []}
                        for part in content.parts:
                            if hasattr(part, 'text') and part.text:
                                content_dict['parts'].append({'text': part.text})
                            elif hasattr(part, 'inline_data'):
                                # Skip binary data, just note it exists
                                content_dict['parts'].append({'inline_data': '[BINARY_DATA]'})
                            elif hasattr(part, 'file_data'):
                                content_dict['parts'].append({'file_data': '[FILE_DATA]'})
                        normalized_contents.append(content_dict)
            elif hasattr(contents, 'role') and hasattr(contents, 'parts'):
                # Single Content object
                content_dict = {'role': contents.role, 'parts': []}
                for part in contents.parts:
                    if hasattr(part, 'text') and part.text:
                        content_dict['parts'].append({'text': part.text})
                    elif hasattr(part, 'inline_data'):
                        content_dict['parts'].append({'inline_data': '[BINARY_DATA]'})
                normalized_contents.append(content_dict)
            
            # Truncate if needed
            if normalized_contents:
                # Serialize to JSON first to check size
                input_messages = json.dumps(normalized_contents, ensure_ascii=False)

                # Apply truncation if too long
                if len(input_messages) > Config.MAX_PROMPT_LENGTH:
                    # Truncate at the content level to maintain valid JSON
                    # Strategy: Truncate text within parts, then re-serialize
                    truncated_contents = []
                    remaining_length = Config.MAX_PROMPT_LENGTH - marker_len - 100  # Reserve space for JSON structure and marker
                    current_length = 0

                    for content in normalized_contents:
                        if current_length >= remaining_length:
                            break

                        truncated_content = {'role': content.get('role', 'user'), 'parts': []}
                        for part in content.get('parts', []):
                            if current_length >= remaining_length:
                                break

                            if 'text' in part:
                                text = part['text']
                                available = remaining_length - current_length
                                if len(text) > available:
                                    # Truncate this text part
                                    truncated_content['parts'].append({'text': text[:available] + '...'})
                                    current_length += available
                                    break
                                else:
                                    truncated_content['parts'].append(part)
                                    current_length += len(text)
                            else:
                                # Non-text part (e.g., binary data marker)
                                truncated_content['parts'].append(part)
                                current_length += 20  # Estimate for non-text parts

                        if truncated_content['parts']:
                            truncated_contents.append(truncated_content)

                    # Re-serialize the truncated content and add marker as a note in the last message
                    # This keeps the JSON valid while indicating truncation
                    if truncated_contents:
                        # Add truncation marker as a special part in the last message
                        truncated_contents[-1]['parts'].append({'text': marker})

                    input_messages = json.dumps(truncated_contents, ensure_ascii=False)

                    # Verify the serialized length and trim further if needed
                    # (JSON escaping/structure can cause the serialized output to exceed the budget)
                    if len(input_messages) > Config.MAX_PROMPT_LENGTH:
                        # Instead of hard truncating (which can produce invalid JSON),
                        # return a minimal valid JSON structure with truncation notice
                        truncation_notice = {
                            "role": "user",
                            "parts": [{"text": f"[Content truncated - exceeded {Config.MAX_PROMPT_LENGTH} character limit]"}]
                        }
                        input_messages = json.dumps([truncation_notice], ensure_ascii=False)
                        logger.warning(
                            f"Serialized JSON exceeded limit after content truncation "
                            f"({len(input_messages)} > {Config.MAX_PROMPT_LENGTH}). "
                            f"Returning minimal valid JSON structure with truncation notice."
                        )

                    prompts_truncated = True
                    logger.debug(
                        f"Input contents truncated to {len(input_messages)} characters (valid JSON maintained)"
                    )
        except (TypeError, ValueError) as e:
            logger.warning(f"Failed to serialize input contents to JSON: {e}")
            input_messages = None
    
    return {
        'systemPrompt': system_prompt,
        'inputMessages': input_messages,
        'promptsTruncated': prompts_truncated
    }


def extract_response_content(response: Any, prompts_truncated: bool = False) -> Dict[str, Any]:
    """
    Extract output response content from Google Gemini API response.

    Args:
        response: Google Gemini API response object (GenerateContentResponse)
        prompts_truncated: Whether prompts were already truncated (from request)

    Returns:
        Dict containing:
            - outputResponse: String or None (model response content)
            - promptsTruncated: Boolean (True if any field was truncated)
    """
    output_response = None
    was_truncated = prompts_truncated
    marker = "...[TRUNCATED]"
    marker_len = len(marker)

    try:
        # Extract content from response.candidates[0].content.parts
        if hasattr(response, 'candidates') and response.candidates and len(response.candidates) > 0:
            first_candidate = response.candidates[0]

            if hasattr(first_candidate, 'content') and first_candidate.content:
                content = first_candidate.content

                if hasattr(content, 'parts') and content.parts:
                    # Extract text from all parts
                    text_parts = []
                    for part in content.parts:
                        if hasattr(part, 'text') and part.text:
                            text_parts.append(part.text)

                    if text_parts:
                        output_response = '\n'.join(text_parts)

                        # Apply truncation
                        if len(output_response) > Config.MAX_PROMPT_LENGTH:
                            truncate_at = Config.MAX_PROMPT_LENGTH - marker_len
                            output_response = output_response[:truncate_at] + marker
                            was_truncated = True
                            logger.debug(
                                f"Output response truncated to {Config.MAX_PROMPT_LENGTH} characters"
                            )

        # Fallback: try response.text property
        if not output_response and hasattr(response, 'text'):
            output_response = response.text

            if output_response and len(output_response) > Config.MAX_PROMPT_LENGTH:
                truncate_at = Config.MAX_PROMPT_LENGTH - marker_len
                output_response = output_response[:truncate_at] + marker
                was_truncated = True
                logger.debug(
                    f"Output response (from .text) truncated to {Config.MAX_PROMPT_LENGTH} characters"
                )
    except Exception as e:
        logger.warning(f"Failed to extract response content: {e}")
        output_response = None

    return {
        'outputResponse': output_response,
        'promptsTruncated': was_truncated
    }



def extract_streaming_response_content(accumulated_content: str, prompts_truncated: bool = False) -> Dict[str, Any]:
    """Extract streaming response content using Google's MAX_PROMPT_LENGTH."""
    return _core_extract_streaming_response_content(
        accumulated_content, prompts_truncated, max_prompt_length=Config.MAX_PROMPT_LENGTH
    )


def extract_prompt_data_if_enabled(
    kwargs: Dict[str, Any],
    args: Optional[Tuple] = None,
    config: Optional[Any] = None,
    response: Any = None,
    accumulated_content: str = None
) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[bool]]:
    """
    Extract prompt data if capture is enabled.

    This is a shared helper function used by both Google AI and Vertex AI middleware.

    Args:
        kwargs: Request kwargs containing contents and system_instruction
        args: Optional positional arguments (first arg is usually contents)
        config: Optional config object (for Google AI SDK)
        response: API response object (for non-streaming)
        accumulated_content: Accumulated streaming content (for streaming)

    Returns:
        Tuple of (system_prompt, input_messages, output_response, prompts_truncated)
    """
    if not Config.CAPTURE_PROMPTS:
        return None, None, None, None

    try:
        # Extract request data (system instruction and contents)
        request_data = extract_prompts_from_request(kwargs, config=config, args=args)
        system_prompt = request_data.get('systemPrompt')
        input_messages = request_data.get('inputMessages')
        prompts_truncated = request_data.get('promptsTruncated', False)

        # Extract response data
        output_response = None
        if response is not None:
            # Non-streaming response
            response_data = extract_response_content(response, prompts_truncated)
            output_response = response_data.get('outputResponse')
            prompts_truncated = response_data.get('promptsTruncated', prompts_truncated)
        elif accumulated_content is not None:
            # Streaming response
            response_data = extract_streaming_response_content(accumulated_content, prompts_truncated)
            output_response = response_data.get('outputResponse')
            prompts_truncated = response_data.get('promptsTruncated', prompts_truncated)

        logger.debug(
            f"Prompt capture - system_prompt: {bool(system_prompt)}, "
            f"input_messages: {bool(input_messages)}, "
            f"output_response: {bool(output_response)}, "
            f"truncated: {prompts_truncated}"
        )

        return system_prompt, input_messages, output_response, prompts_truncated
    except Exception as e:
        logger.warning(f"Failed to extract prompt data: {e}")
        return None, None, None, None

