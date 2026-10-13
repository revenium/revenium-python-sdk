"""The prompt and reply of a LangChain call the callback records, in the
shape prompt capture reads on every other path.

Reads LangChain messages by their ``type`` and attributes only, so it works
with every langchain-core version the SDK supports.
"""

import logging
from typing import Any, Dict, List, Optional

from revenium_middleware._core.config import is_capture_prompts_enabled
from revenium_middleware._core.prompt_extraction import extract_prompts_from_request

logger = logging.getLogger("revenium_middleware.langchain")

_ROLES = {"system": "system", "human": "user", "ai": "assistant", "tool": "tool", "function": "function"}
_CLASS_PREFIXES = {"System": "system", "Human": "human", "AI": "ai", "Tool": "tool", "Function": "function",
                   "Chat": "chat"}
_CLASS_STYLE_TYPES = {
    f"{prefix}{suffix}": message_type
    for prefix, message_type in _CLASS_PREFIXES.items()
    for suffix in ("Message", "MessageChunk")
}


def _role_of(message: Any) -> str:
    message_type = _CLASS_STYLE_TYPES.get(message.type, message.type)
    return _ROLES.get(message_type) or getattr(message, "role", None) or "user"


def _as_request_message(message: Any) -> Dict[str, Any]:
    if not (hasattr(message, "type") and hasattr(message, "content")):
        return {"role": "user", "content": str(message)}
    request_message = {"role": _role_of(message), "content": message.content}
    tool_call_id = getattr(message, "tool_call_id", None)
    if isinstance(tool_call_id, str):
        request_message["tool_call_id"] = tool_call_id
    return request_message


def _request_messages(request_input: List[Any], text_model: bool) -> List[Dict[str, Any]]:
    if text_model:
        return [{"role": "user", "content": prompt} for prompt in request_input]
    messages = []
    for message_list in request_input:
        batch = message_list if isinstance(message_list, list) else [message_list]
        messages.extend(_as_request_message(message) for message in batch)
    return messages


def captured_request_prompts(request_input: List[Any], text_model: bool) -> Optional[Dict[str, Any]]:
    """The capture fields of a run's prompt, or None with capture off.

    Only these bounded strings are kept for the run, never the messages. A
    prompt it cannot read is recorded without its prompt rather than failing
    the call.
    """
    if not is_capture_prompts_enabled():
        return None
    try:
        messages = _request_messages(request_input, text_model)
        return extract_prompts_from_request({"messages": messages}) if messages else None
    except Exception as e:
        logger.warning("Prompt capture could not read this LangChain prompt (%s); recording it without its prompt",
                       type(e).__name__)
        return None


def reply_text(response: Any) -> str:
    """The text of the run's first generation, or "" when it has none."""
    try:
        text = response.generations[0][0].text
    except (AttributeError, IndexError, KeyError, TypeError):
        return ""
    return text if isinstance(text, str) else ""
