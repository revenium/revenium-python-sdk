"""Responses API payloads in the shapes the API streams them.

Id, model and usage are nested under ``event.response``; no stream event
carries them at the top level. ``event_payloads`` is the wire form (for an SSE
body) and ``typed_events`` validates the same sequence into the ``openai``
event classes, so a fixture that drifts from the real schema fails loudly.
"""
from openai.types.responses import (
    ResponseCompletedEvent,
    ResponseContentPartAddedEvent,
    ResponseContentPartDoneEvent,
    ResponseCreatedEvent,
    ResponseFailedEvent,
    ResponseIncompleteEvent,
    ResponseOutputItemAddedEvent,
    ResponseOutputItemDoneEvent,
    ResponseTextDeltaEvent,
    ResponseTextDoneEvent,
)

RESPONSE_ID = "resp_stream_test"
MODEL = "o4-mini"

_EVENT_CLASSES = {
    "response.created": ResponseCreatedEvent,
    "response.output_item.added": ResponseOutputItemAddedEvent,
    "response.content_part.added": ResponseContentPartAddedEvent,
    "response.output_text.delta": ResponseTextDeltaEvent,
    "response.output_text.done": ResponseTextDoneEvent,
    "response.content_part.done": ResponseContentPartDoneEvent,
    "response.output_item.done": ResponseOutputItemDoneEvent,
    "response.completed": ResponseCompletedEvent,
    "response.incomplete": ResponseIncompleteEvent,
    "response.failed": ResponseFailedEvent,
}

TERMINAL_STATUS = {
    "response.completed": "completed",
    "response.incomplete": "incomplete",
    "response.failed": "failed",
}


def usage_payload(input_tokens=80, output_tokens=20, cached_tokens=0, reasoning_tokens=0):
    return {
        "input_tokens": input_tokens,
        "input_tokens_details": {"cached_tokens": cached_tokens, "cache_write_tokens": 0},
        "output_tokens": output_tokens,
        "output_tokens_details": {"reasoning_tokens": reasoning_tokens},
        "total_tokens": input_tokens + output_tokens,
    }


def _message_item(status, text):
    content = [{"type": "output_text", "text": text, "annotations": []}] if text is not None else []
    return {"id": "msg_1", "type": "message", "role": "assistant", "status": status, "content": content}


def response_payload(status="completed", usage=None, with_output=True):
    payload = {
        "id": RESPONSE_ID,
        "object": "response",
        "created_at": 1,
        "model": MODEL,
        "status": status,
        "output": [_message_item("completed", "hi")] if with_output else [],
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
        "usage": usage,
    }
    if status == "incomplete":
        payload["incomplete_details"] = {"reason": "max_output_tokens"}
    if status == "failed":
        payload["error"] = {"code": "server_error", "message": "upstream failure"}
    return payload


def event_payloads(terminal="response.completed", usage=None):
    """``response.created`` through the terminal event, which carries ``usage``."""
    usage = usage_payload() if usage is None else usage
    text_part = {"type": "output_text", "text": "", "annotations": []}
    text_ref = {"output_index": 0, "content_index": 0, "item_id": "msg_1"}
    events = [
        {"type": "response.created",
         "response": response_payload(status="in_progress", with_output=False)},
        {"type": "response.output_item.added", "output_index": 0,
         "item": _message_item("in_progress", None)},
        {"type": "response.content_part.added", **text_ref, "part": text_part},
        {"type": "response.output_text.delta", **text_ref, "delta": "hi", "logprobs": []},
        {"type": "response.output_text.done", **text_ref, "text": "hi", "logprobs": []},
        {"type": "response.content_part.done", **text_ref, "part": {**text_part, "text": "hi"}},
        {"type": "response.output_item.done", "output_index": 0,
         "item": _message_item("completed", "hi")},
        {"type": terminal,
         "response": response_payload(status=TERMINAL_STATUS[terminal], usage=usage)},
    ]
    return [{**event, "sequence_number": i} for i, event in enumerate(events)]


def typed_events(terminal="response.completed", usage=None):
    return [_EVENT_CLASSES[e["type"]].model_validate(e) for e in event_payloads(terminal, usage)]
