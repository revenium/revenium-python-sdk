"""A real langchain-mistralai ChatMistralAI whose HTTP clients answer from a stub.

ChatMistralAI calls Mistral through its own httpx clients, which no Revenium
wrap patches, so only the LangChain callback sees these calls. Both clients
are given: one left out would be built against api.mistral.ai.
"""
import json

import httpx

MISTRAL_MODEL = "mistral-small-latest"
INPUT_TOKENS = 7
OUTPUT_TOKENS = 3
ANSWER = "hello"

_USAGE = {"prompt_tokens": INPUT_TOKENS, "completion_tokens": OUTPUT_TOKENS,
          "total_tokens": INPUT_TOKENS + OUTPUT_TOKENS}
_BASE = {"id": "cmpl-mistral", "created": 1, "model": MISTRAL_MODEL}


def _completion():
    return {**_BASE, "object": "chat.completion",
            "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": ANSWER}}],
            "usage": _USAGE}


def _chunks():
    # Every Mistral chunk carries a choice; the usage rides on the last one.
    chunks = [
        {**_BASE, "object": "chat.completion.chunk",
         "choices": [{"index": 0, "delta": {"role": "assistant", "content": ANSWER}, "finish_reason": None}]},
        {**_BASE, "object": "chat.completion.chunk",
         "choices": [{"index": 0, "delta": {"content": ""}, "finish_reason": "stop"}], "usage": _USAGE},
    ]
    return "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks).encode() + b"data: [DONE]\n\n"


def mistral_handler(during_request=None):
    def handler(request):
        if during_request is not None:
            during_request()
        if json.loads(request.content or b"{}").get("stream"):
            return httpx.Response(200, content=_chunks(), headers={"content-type": "text/event-stream"})
        return httpx.Response(200, json=_completion())
    return handler


def chat_mistralai(callbacks, during_request=None):
    from langchain_mistralai import ChatMistralAI

    transport = httpx.MockTransport(mistral_handler(during_request))
    base_url = "http://mistral.test/v1"
    return ChatMistralAI(model=MISTRAL_MODEL, api_key="stub", callbacks=callbacks,
                         client=httpx.Client(base_url=base_url, transport=transport),
                         async_client=httpx.AsyncClient(base_url=base_url, transport=transport))
