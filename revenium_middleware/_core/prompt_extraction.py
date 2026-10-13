"""
Shared prompt extraction utilities.

This module provides prompt extraction functions shared across all provider
middlewares. Provider-specific prompt_extractor modules import from here.
"""

import base64
import binascii
import json
import logging
import re
from typing import Dict, Any, Optional, Tuple

from .config import Config, is_capture_prompts_enabled
from .secret_shapes import (
    SECRET_SHAPES,
    SIGNED_URL_PARAM,
    URL_USERINFO_PASSWORD,
    is_password_label,
    is_secret_label,
    mask_jwts,
    mask_secret_shapes,
)

logger = logging.getLogger("revenium_middleware.extension")


REDACTED = "***REDACTED***"
_REDACTED_JWT = "***REDACTED_JWT***"
_SECRET_VALUE = r"[a-zA-Z0-9_\-.+/=]{20,}"
# JSON escapes are consumed whole so a truncated, unparseable JSON value is
# masked past its escaped quotes; the match never ends on an escaped quote or
# a trailing JSON delimiter, so the surrounding structure is kept.
_PASSWORD_VALUE = r"(?:\\.|[^\"'\s\\]){8,}(?<![,{}\[\]])(?<!\\\")(?<!\\')"
_PARSED_PASSWORD_VALUE = r"(?s).{8,}"
# A key and its separator, with an optional plain or JSON-escaped quote on
# either side, so only the value is replaced and serialized JSON stays valid.
_QUOTE = r"(?:\\?[\"'])?"

_SECRET_VALUE_RE = re.compile(_SECRET_VALUE)
_PASSWORD_VALUE_RE = re.compile(_PASSWORD_VALUE)
_PARSED_PASSWORD_VALUE_RE = re.compile(_PARSED_PASSWORD_VALUE)
# A label starts where a run starts, or right after a serialized "\n", "\r"
# or "\t" escape, whose letter is never read as part of the label.
_LABEL = re.compile(
    rf"((?:(?<![A-Za-z0-9_-])(?!(?<=\\)[nrt])|(?<=\\[nrt]))([A-Za-z0-9_-]{{1,128}}){_QUOTE}\s*[:=]\s*{_QUOTE})"
)
_BEARER = re.compile(r"(?i)(Bearer\s+)[a-zA-Z0-9_\-.+/=]{8,}")
_BASIC = re.compile(r"(?i)((?<![A-Za-z0-9_](?<!\\[nrt]))Basic\s+)([A-Za-z0-9+/]{8,}={0,2})(?![A-Za-z0-9+/=])")


def _text_value_pattern(label: str) -> "Optional[re.Pattern[str]]":
    if is_password_label(label):
        return _PASSWORD_VALUE_RE
    if is_secret_label(label):
        return _SECRET_VALUE_RE
    return None


def _mask_labelled_values(text: str) -> str:
    """Mask the value after every label ending in a secret word (``AWS_SECRET_ACCESS_KEY=...``)."""
    out = []
    position = 0
    label = _LABEL.search(text)
    while label:
        value_pattern = _text_value_pattern(label.group(2))
        value = value_pattern.match(text, label.end()) if value_pattern else None
        resume = value.end() if value else label.end()
        out.append(text[position:label.end()] + (REDACTED if value else text[label.end():resume]))
        position = resume
        label = _LABEL.search(text, position)
    out.append(text[position:])
    return "".join(out)


def _mask_basic_credentials(match: "re.Match[str]") -> str:
    try:
        decoded = base64.b64decode(match.group(2), validate=True)
    except (binascii.Error, ValueError):
        return match.group(0)
    return match.group(1) + REDACTED if b":" in decoded else match.group(0)


def _is_secret_value(key: str, value: str) -> bool:
    if is_password_label(key):
        return bool(_PARSED_PASSWORD_VALUE_RE.match(value))
    return is_secret_label(key) and bool(_SECRET_VALUE_RE.match(value))


def _mask_parsed(node: Any, key: Optional[str] = None) -> Any:
    if isinstance(node, dict):
        return {_sanitize_text(child_key): _mask_parsed(value, child_key) for child_key, value in node.items()}
    if isinstance(node, list):
        return [_mask_parsed(item, key) for item in node]
    if isinstance(node, bool) or node is None:
        return node
    if key is not None and _is_secret_value(key, str(node)):
        return REDACTED
    if isinstance(node, str):
        return sanitize_credentials(node)
    return node


def _parse_json_container(text: str) -> Any:
    if not text.lstrip().startswith(("{", "[")):
        return None
    try:
        parsed = json.loads(text)
    except ValueError:
        return None
    return parsed if isinstance(parsed, (dict, list)) else None


def sanitize_credentials(text: str) -> str:
    """Mask API keys, tokens and passwords in captured prompt or reply text.

    Serialized JSON is masked value by value and re-serialized, so it stays valid.
    """
    parsed = _parse_json_container(text)
    if parsed is not None:
        masked = _mask_parsed(parsed)
        return text if masked == parsed else json.dumps(masked, ensure_ascii=False)
    return _sanitize_text(text)


def _sanitize_text(text: str) -> str:
    text = mask_secret_shapes(text, SECRET_SHAPES, lambda kept: kept + REDACTED)
    text = mask_jwts(text, _REDACTED_JWT)
    text = _BEARER.sub(r"\g<1>" + REDACTED, text)
    text = _BASIC.sub(_mask_basic_credentials, text)
    text = SIGNED_URL_PARAM.sub(r"\g<1>" + REDACTED, text)
    text = URL_USERINFO_PASSWORD.sub(r"\g<1>" + REDACTED + r"\g<2>", text)
    return _mask_labelled_values(text)


class ReplyCollector:
    """Collects a streamed reply, holding no more than ``Config.MAX_PROMPT_LENGTH + 1`` characters.

    The one character past the limit is what lets
    ``extract_streaming_response_content`` still see the reply as over-long and mark it truncated.
    """

    def __init__(self):
        self._parts = []
        self._held = 0

    def add(self, text: str) -> None:
        room = Config.MAX_PROMPT_LENGTH + 1 - self._held
        if room <= 0 or not text:
            return
        kept = text[:room]
        self._parts.append(kept)
        self._held += len(kept)

    def text(self) -> str:
        return "".join(self._parts)


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


TRUNCATION_MARKER = "...[TRUNCATED]"
_MAX_SEARCH_STEPS = 40


def serialize_messages_within_limit(messages: list, max_length: int) -> Tuple[Optional[str], bool]:
    """Serialize ``messages`` to JSON text of at most ``max_length`` characters.

    Returns ``(json_text, was_cut)``. Messages that fit are returned exactly as
    sent. Otherwise the longest strings are shortened first, each ending in the
    truncation marker, so short turns stay whole and the text always parses. If
    that is not enough, the oldest messages are left out and the newest kept.
    The result is valid JSON of at most ``max_length`` characters for any
    ``max_length`` of 2 or more; below that it is ``"[]"``. The caller's
    messages are never changed. ``(None, False)`` if they cannot be serialized.
    """
    try:
        whole = _dump(messages)
        if len(whole) <= max_length:
            return whole, False
        text = _cut_to_fit(messages, max_length)
        if text is None:
            newest = _newest_messages_that_fit(messages, max_length)
            text = _cut_to_fit(newest, max_length) if newest else "[]"
    except (TypeError, ValueError) as e:
        logger.warning(f"Failed to serialize input messages to JSON: {e}")
        return None, False
    logger.debug(f"Input messages truncated to {len(text)} characters")
    return text, True


def _dump(node: Any) -> str:
    return json.dumps(node, ensure_ascii=False)


def _cut_strings(node: Any, keep: int) -> Any:
    """A copy of ``node`` where every string longer than ``keep`` is cut to ``keep`` characters, marker included."""
    if isinstance(node, str):
        return node[:keep - len(TRUNCATION_MARKER)] + TRUNCATION_MARKER if len(node) > keep else node
    if isinstance(node, dict):
        return {key: _cut_strings(value, keep) for key, value in node.items()}
    if isinstance(node, (list, tuple)):
        return [_cut_strings(value, keep) for value in node]
    return node


def _longest_string(node: Any) -> int:
    if isinstance(node, str):
        return len(node)
    if isinstance(node, dict):
        return max(map(_longest_string, node.values()), default=0)
    if isinstance(node, (list, tuple)):
        return max(map(_longest_string, node), default=0)
    return 0


def _cut_to_fit(messages: list, max_length: int) -> Optional[str]:
    """The serialization with the largest common string length that fits, or None if even the shortest does not."""
    floor = len(TRUNCATION_MARKER)
    if len(_dump(_cut_strings(messages, floor))) > max_length:
        return None
    fits, too_long = floor, max(floor, min(_longest_string(messages) + 1, max_length + 1))
    for _ in range(_MAX_SEARCH_STEPS):
        if too_long - fits <= 1:
            break
        middle = (fits + too_long) // 2
        if len(_dump(_cut_strings(messages, middle))) <= max_length:
            fits = middle
        else:
            too_long = middle
    return _dump(_cut_strings(messages, fits))


def _newest_messages_that_fit(messages: list, max_length: int) -> list:
    """The longest run of the newest messages that fits once every string is cut to the marker."""
    floor = len(TRUNCATION_MARKER)
    total = 2
    kept = 0
    for message in reversed(messages):
        total += len(_dump(_cut_strings(message, floor))) + (2 if kept else 0)
        if total > max_length:
            break
        kept += 1
    return messages[len(messages) - kept:]


def _truncate_system_prompt(system_prompt: Optional[str]) -> Tuple[Optional[str], bool]:
    """Cut a system prompt to ``Config.MAX_PROMPT_LENGTH``, marker included."""
    if not system_prompt or len(system_prompt) <= Config.MAX_PROMPT_LENGTH:
        return system_prompt, False
    marker = "...[TRUNCATED]"
    truncate_at = Config.MAX_PROMPT_LENGTH - len(marker)
    logger.debug(f"System prompt truncated to {Config.MAX_PROMPT_LENGTH} characters")
    return system_prompt[:truncate_at] + marker, True


def _serialize_input_messages(messages: list) -> Tuple[Optional[str], bool]:
    """The messages as a JSON string within ``Config.MAX_PROMPT_LENGTH``, and whether any was cut."""
    if not messages:
        return None, False
    return serialize_messages_within_limit(messages, Config.MAX_PROMPT_LENGTH)


def _message_content_text(content: Any) -> Optional[str]:
    """A message's content as text: a string as is, a content-part list as JSON."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return json.dumps(content)
    return None


def _as_input_item(item: Any) -> Optional[Dict[str, Any]]:
    """A Responses input item as a dict; output items passed back are openai models."""
    if isinstance(item, dict):
        return item
    to_dict = getattr(item, 'to_dict', None)
    return to_dict(mode="json") if callable(to_dict) else None


def _responses_input_items(request_input: Any) -> list:
    if isinstance(request_input, str):
        return [{'role': 'user', 'content': request_input}]
    if isinstance(request_input, list):
        return [item for item in map(_as_input_item, request_input) if item is not None]
    return []


def _is_responses_request(kwargs: Dict[str, Any]) -> bool:
    # The openai client passes omit or NOT_GIVEN for a field the caller left
    # out, so only a real string or list counts as given.
    return isinstance(kwargs.get('instructions'), str) or isinstance(kwargs.get('input'), (str, list))


def _extract_responses_prompts(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """System prompt and input of a Responses API request.

    The system prompt is ``instructions``, else the first ``system`` or
    ``developer`` input item, which is then left out of the input messages.
    """
    input_items = _responses_input_items(kwargs.get('input'))
    instructions = kwargs.get('instructions')
    system_prompt = instructions if isinstance(instructions, str) else None
    if system_prompt is None:
        system_item = next((item for item in input_items
                            if item.get('role') in ('system', 'developer')
                            and _message_content_text(item.get('content')) is not None), None)
        if system_item is not None:
            system_prompt = _message_content_text(system_item.get('content'))
            input_items = [item for item in input_items if item is not system_item]

    system_prompt, system_truncated = _truncate_system_prompt(system_prompt)
    input_messages, input_truncated = _serialize_input_messages(input_items)
    return {
        'systemPrompt': system_prompt,
        'inputMessages': input_messages,
        'promptsTruncated': system_truncated or input_truncated
    }


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

    if not messages and _is_responses_request(kwargs):
        return _extract_responses_prompts(kwargs)

    if not messages:
        return {
            'systemPrompt': None,
            'inputMessages': None,
            'promptsTruncated': False
        }

    system_prompt = None
    user_messages = []

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

    system_prompt, system_truncated = _truncate_system_prompt(system_prompt)
    input_messages, input_truncated = _serialize_input_messages(user_messages)

    return {
        'systemPrompt': system_prompt,
        'inputMessages': input_messages,
        'promptsTruncated': system_truncated or input_truncated
    }


def responses_output_text(response: Any) -> str:
    """The text of a Responses API reply's message items, joined as ``Response.output_text`` does.

    Read by attribute, so it works on the openai ``Response`` model and on the
    attribute view of a streamed event's ``response``.
    """
    output = getattr(response, 'output', None)
    if not isinstance(output, list):
        return ""
    texts = []
    for item in output:
        if getattr(item, 'type', None) != 'message':
            continue
        content = getattr(item, 'content', None)
        for part in content if isinstance(content, list) else ():
            text = getattr(part, 'text', None)
            if getattr(part, 'type', None) == 'output_text' and isinstance(text, str):
                texts.append(text)
    return "".join(texts)


def _reply_content(response: Any) -> Any:
    """The reply of a chat completion (``choices[0].message.content``) or of a Responses call."""
    if hasattr(response, 'choices') and response.choices and len(response.choices) > 0:
        first_choice = response.choices[0]
        if hasattr(first_choice, 'message') and hasattr(first_choice.message, 'content'):
            return first_choice.message.content
        return None
    return responses_output_text(response)


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
        content = _reply_content(response)

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


def extract_prompt_data_if_enabled(
    request_body: Optional[Dict[str, Any]],
    response: Any = None,
    accumulated_content: str = None,
    request_prompts: Optional[Dict[str, Any]] = None
) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[bool]]:
    """
    Extract prompt data if capture is enabled.

    Args:
        request_body: Request body containing messages
        response: API response object (for non-streaming)
        accumulated_content: Accumulated streaming content (for streaming)
        request_prompts: Prompts already extracted from the request with
            ``extract_prompts_from_request``, used instead of ``request_body``

    Returns:
        Tuple of (system_prompt, input_messages, output_response, prompts_truncated)
    """
    if not is_capture_prompts_enabled() or not (request_body or request_prompts):
        return None, None, None, None

    # Extract prompts from request
    prompt_data = request_prompts if request_prompts is not None else extract_prompts_from_request(request_body)
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
