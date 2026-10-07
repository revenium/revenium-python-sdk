"""Stubbed provider calls for the entry-point matrix, one per executable manifest row.

This module must not import revenium_middleware: the differential run executes it
in a fresh interpreter without our middleware to attribute any crash.
"""
import asyncio
import contextlib
import itertools
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx

CALLS = {}

OPENAI_MODEL = "gpt-4o-mini"
ANTHROPIC_MODEL = "claude-sonnet-4-5"
GEMINI_MODEL = "gemini-2.0-flash"
GEMINI_EMBED_MODEL = "text-embedding-004"
IMAGEN_MODEL = "imagen-3.0-generate-002"
OLLAMA_MODEL = "llama3"
LITELLM_MODEL = "gpt-4o-mini"
PERPLEXITY_MODEL = "sonar"
FAL_APP = "fal-ai/flux/schnell"

INPUT_TOKENS = 11
OUTPUT_TOKENS = 7
CACHE_READ_TOKENS = 3
REASONING_TOKENS = 2
USER_MESSAGES = [{"role": "user", "content": "hi"}]


def call(row_id):
    def register(fn):
        if row_id in CALLS:
            raise ValueError(f"duplicate call for {row_id}")
        CALLS[row_id] = fn
        return fn
    return register


def run_call(row_id):
    result = CALLS[row_id]()
    if asyncio.iscoroutine(result):
        asyncio.run(result)


def _sse(events):
    return "".join(
        (f"event: {name}\n" if name else "") + f"data: {json.dumps(data)}\n\n"
        for name, data in events
    ).encode()


# --- OpenAI (and Perplexity through the OpenAI-compatible client) ---------------

_OPENAI_USAGE = {"prompt_tokens": INPUT_TOKENS, "completion_tokens": OUTPUT_TOKENS,
                 "total_tokens": INPUT_TOKENS + OUTPUT_TOKENS,
                 "prompt_tokens_details": {"cached_tokens": CACHE_READ_TOKENS},
                 "completion_tokens_details": {"reasoning_tokens": REASONING_TOKENS}}


_PERPLEXITY_USAGE = {"prompt_tokens": INPUT_TOKENS, "completion_tokens": OUTPUT_TOKENS,
                     "total_tokens": INPUT_TOKENS + OUTPUT_TOKENS}


def _chat_completion(model, usage):
    return {"id": "chatcmpl-1", "object": "chat.completion", "created": 1, "model": model,
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": "{\"a\": 1}"}}],
            "usage": usage}


def _chat_chunks(model, usage):
    base = {"id": "chatcmpl-1", "object": "chat.completion.chunk", "created": 1, "model": model}
    chunks = [
        {**base, "choices": [{"index": 0, "delta": {"role": "assistant", "content": "{\"a\": 1}"},
                              "finish_reason": None}]},
        {**base, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
        {**base, "choices": [], "usage": usage},
    ]
    return _sse([(None, c) for c in chunks]) + b"data: [DONE]\n\n"


_RESPONSE_USAGE = {"input_tokens": INPUT_TOKENS, "output_tokens": OUTPUT_TOKENS,
                   "total_tokens": INPUT_TOKENS + OUTPUT_TOKENS,
                   "input_tokens_details": {"cached_tokens": CACHE_READ_TOKENS},
                   "output_tokens_details": {"reasoning_tokens": REASONING_TOKENS}}
_OUTPUT_TEXT = {"type": "output_text", "text": "{\"a\": 1}", "annotations": []}
_MESSAGE_ITEM = {"type": "message", "id": "msg_1", "role": "assistant", "status": "completed",
                 "content": [_OUTPUT_TEXT]}


def _response(status="completed", output=None, usage=_RESPONSE_USAGE):
    return {"id": "resp_1", "object": "response", "created_at": 1, "model": OPENAI_MODEL,
            "status": status, "output": [_MESSAGE_ITEM] if output is None else output,
            "parallel_tool_calls": True, "tool_choice": "auto", "tools": [], "usage": usage}


def _response_events():
    in_progress = _response(status="in_progress", output=[], usage=None)
    item_added = {**_MESSAGE_ITEM, "status": "in_progress", "content": []}
    events = [
        {"type": "response.created", "response": in_progress},
        {"type": "response.in_progress", "response": in_progress},
        {"type": "response.output_item.added", "output_index": 0, "item": item_added},
        {"type": "response.content_part.added", "item_id": "msg_1", "output_index": 0,
         "content_index": 0, "part": {**_OUTPUT_TEXT, "text": ""}},
        {"type": "response.output_text.delta", "item_id": "msg_1", "output_index": 0,
         "content_index": 0, "delta": _OUTPUT_TEXT["text"], "logprobs": []},
        {"type": "response.output_text.done", "item_id": "msg_1", "output_index": 0,
         "content_index": 0, "text": _OUTPUT_TEXT["text"], "logprobs": []},
        {"type": "response.content_part.done", "item_id": "msg_1", "output_index": 0,
         "content_index": 0, "part": _OUTPUT_TEXT},
        {"type": "response.output_item.done", "output_index": 0, "item": _MESSAGE_ITEM},
        {"type": "response.completed", "response": _response()},
    ]
    for number, event in enumerate(events):
        event["sequence_number"] = number
    return _sse([(e["type"], e) for e in events])


_EMBEDDING = {"object": "list", "model": "text-embedding-3-small",
              "data": [{"object": "embedding", "index": 0, "embedding": [0.1, 0.2]}],
              "usage": {"prompt_tokens": INPUT_TOKENS, "total_tokens": INPUT_TOKENS}}


def _openai_compatible_handler(model, usage=_OPENAI_USAGE):
    def handler(request):
        body = json.loads(request.content or b"{}")
        path = request.url.path
        streamed = bool(body.get("stream"))
        if path.endswith("/chat/completions"):
            if streamed:
                return httpx.Response(200, content=_chat_chunks(model, usage),
                                      headers={"content-type": "text/event-stream"})
            return httpx.Response(200, json=_chat_completion(model, usage))
        if path.endswith("/responses"):
            if streamed:
                return httpx.Response(200, content=_response_events(),
                                      headers={"content-type": "text/event-stream"})
            return httpx.Response(200, json=_response())
        if path.endswith("/embeddings"):
            return httpx.Response(200, json=_EMBEDDING)
        return httpx.Response(404, json={"error": path})
    return handler


def _openai(base_url=None, model=OPENAI_MODEL, usage=_OPENAI_USAGE):
    import openai
    transport = httpx.MockTransport(_openai_compatible_handler(model, usage))
    return openai.OpenAI(api_key="stub", base_url=base_url, http_client=httpx.Client(transport=transport))


def _async_openai(base_url=None, model=OPENAI_MODEL, usage=_OPENAI_USAGE):
    import openai
    transport = httpx.MockTransport(_openai_compatible_handler(model, usage))
    return openai.AsyncOpenAI(api_key="stub", base_url=base_url,
                              http_client=httpx.AsyncClient(transport=transport))


def _chat_kwargs(model=OPENAI_MODEL):
    return {"model": model, "messages": USER_MESSAGES}


def _structured_output():
    """The Pydantic model the stubbed ``{"a": 1}`` completion text parses into."""
    import pydantic

    class Structured(pydantic.BaseModel):
        a: int

    return Structured


@call("openai.chat.create.sync")
def _():
    _openai().chat.completions.create(**_chat_kwargs())


@call("openai.chat.create.sync.stream")
def _():
    for _chunk in _openai().chat.completions.create(**_chat_kwargs(), stream=True):
        pass


@call("openai.chat.create.async")
async def _():
    await _async_openai().chat.completions.create(**_chat_kwargs())


@call("openai.chat.create.async.stream")
async def _():
    async for _chunk in await _async_openai().chat.completions.create(**_chat_kwargs(), stream=True):
        pass


@call("openai.chat.stream_helper.sync")
def _():
    with _openai().chat.completions.stream(**_chat_kwargs()) as stream:
        for _event in stream:
            pass


@call("openai.chat.stream_helper.async")
async def _():
    async with _async_openai().chat.completions.stream(**_chat_kwargs()) as stream:
        async for _event in stream:
            pass


@call("openai.chat.parse.sync")
def _():
    _openai().chat.completions.parse(**_chat_kwargs(), response_format=_structured_output())


@call("openai.chat.parse.async")
async def _():
    await _async_openai().chat.completions.parse(**_chat_kwargs(), response_format=_structured_output())


@call("openai.responses.create.sync")
def _():
    _openai().responses.create(model=OPENAI_MODEL, input="hi")


@call("openai.responses.create.sync.stream")
def _():
    for _event in _openai().responses.create(model=OPENAI_MODEL, input="hi", stream=True):
        pass


@call("openai.responses.create.async")
async def _():
    await _async_openai().responses.create(model=OPENAI_MODEL, input="hi")


@call("openai.responses.create.async.stream")
async def _():
    async for _event in await _async_openai().responses.create(model=OPENAI_MODEL, input="hi", stream=True):
        pass


@call("openai.responses.stream_helper.sync")
def _():
    with _openai().responses.stream(model=OPENAI_MODEL, input="hi") as stream:
        for _event in stream:
            pass


@call("openai.responses.stream_helper.async")
async def _():
    async with _async_openai().responses.stream(model=OPENAI_MODEL, input="hi") as stream:
        async for _event in stream:
            pass


@call("openai.responses.parse.sync")
def _():
    _openai().responses.parse(model=OPENAI_MODEL, input="hi", text_format=_structured_output())


@call("openai.responses.parse.async")
async def _():
    await _async_openai().responses.parse(model=OPENAI_MODEL, input="hi", text_format=_structured_output())


@call("openai.embeddings.create.sync")
def _():
    _openai().embeddings.create(model="text-embedding-3-small", input="hi")


@call("openai.embeddings.create.async")
async def _():
    await _async_openai().embeddings.create(model="text-embedding-3-small", input="hi")


# --- Anthropic ---------------------------------------------------------------

_ANTHROPIC_MESSAGE = {"id": "msg_1", "type": "message", "role": "assistant", "model": ANTHROPIC_MODEL,
                      "content": [{"type": "text", "text": "hi"}], "stop_reason": "end_turn",
                      "stop_sequence": None,
                      "usage": {"input_tokens": INPUT_TOKENS, "output_tokens": OUTPUT_TOKENS,
                                "cache_read_input_tokens": CACHE_READ_TOKENS}}


def _anthropic_events():
    start = {**_ANTHROPIC_MESSAGE, "content": [], "stop_reason": None,
             "usage": {"input_tokens": INPUT_TOKENS, "output_tokens": 1,
                       "cache_read_input_tokens": CACHE_READ_TOKENS}}
    return _sse([
        ("message_start", {"type": "message_start", "message": start}),
        ("content_block_start", {"type": "content_block_start", "index": 0,
                                 "content_block": {"type": "text", "text": ""}}),
        ("content_block_delta", {"type": "content_block_delta", "index": 0,
                                 "delta": {"type": "text_delta", "text": "hi"}}),
        ("content_block_stop", {"type": "content_block_stop", "index": 0}),
        ("message_delta", {"type": "message_delta",
                           "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                           "usage": {"output_tokens": OUTPUT_TOKENS}}),
        ("message_stop", {"type": "message_stop"}),
    ])


def _anthropic_handler(request):
    import httpx2
    if json.loads(request.content or b"{}").get("stream"):
        return httpx2.Response(200, content=_anthropic_events(), headers={"content-type": "text/event-stream"})
    return httpx2.Response(200, json=_ANTHROPIC_MESSAGE)


def _anthropic():
    import anthropic
    import httpx2
    return anthropic.Anthropic(api_key="stub", http_client=httpx2.Client(transport=httpx2.MockTransport(_anthropic_handler)))


def _async_anthropic():
    import anthropic
    import httpx2
    transport = httpx2.MockTransport(_anthropic_handler)
    return anthropic.AsyncAnthropic(api_key="stub", http_client=httpx2.AsyncClient(transport=transport))


_ANTHROPIC_KWARGS = {"model": ANTHROPIC_MODEL, "max_tokens": 16, "messages": USER_MESSAGES}


def _drain_sync(manager):
    with manager as stream:
        for _event in stream:
            pass


async def _drain_async(manager):
    async with manager as stream:
        async for _event in stream:
            pass


@call("anthropic.messages.create.sync")
def _():
    _anthropic().messages.create(**_ANTHROPIC_KWARGS)


@call("anthropic.messages.create.sync.stream")
def _():
    for _event in _anthropic().messages.create(**_ANTHROPIC_KWARGS, stream=True):
        pass


@call("anthropic.messages.create.async")
async def _():
    await _async_anthropic().messages.create(**_ANTHROPIC_KWARGS)


@call("anthropic.messages.create.async.stream")
async def _():
    async for _event in await _async_anthropic().messages.create(**_ANTHROPIC_KWARGS, stream=True):
        pass


@call("anthropic.messages.stream_helper.sync")
def _():
    _drain_sync(_anthropic().messages.stream(**_ANTHROPIC_KWARGS))


@call("anthropic.messages.stream_helper.async")
async def _():
    await _drain_async(_async_anthropic().messages.stream(**_ANTHROPIC_KWARGS))


@call("anthropic.messages.parse.sync")
def _():
    _anthropic().messages.parse(**_ANTHROPIC_KWARGS)


@call("anthropic.messages.parse.async")
async def _():
    await _async_anthropic().messages.parse(**_ANTHROPIC_KWARGS)


@call("anthropic.beta.messages.create.sync")
def _():
    _anthropic().beta.messages.create(**_ANTHROPIC_KWARGS)


@call("anthropic.beta.messages.create.sync.stream")
def _():
    for _event in _anthropic().beta.messages.create(**_ANTHROPIC_KWARGS, stream=True):
        pass


@call("anthropic.beta.messages.create.async")
async def _():
    await _async_anthropic().beta.messages.create(**_ANTHROPIC_KWARGS)


@call("anthropic.beta.messages.create.async.stream")
async def _():
    async for _event in await _async_anthropic().beta.messages.create(**_ANTHROPIC_KWARGS, stream=True):
        pass


@call("anthropic.beta.messages.stream_helper.sync")
def _():
    _drain_sync(_anthropic().beta.messages.stream(**_ANTHROPIC_KWARGS))


@call("anthropic.beta.messages.stream_helper.async")
async def _():
    await _drain_async(_async_anthropic().beta.messages.stream(**_ANTHROPIC_KWARGS))


@call("anthropic.beta.messages.parse.sync")
def _():
    _anthropic().beta.messages.parse(**_ANTHROPIC_KWARGS)


@call("anthropic.beta.messages.parse.async")
async def _():
    await _async_anthropic().beta.messages.parse(**_ANTHROPIC_KWARGS)


# --- google-genai (Gemini API) --------------------------------------------------

_GEMINI_RESPONSE = {
    "candidates": [{"content": {"role": "model", "parts": [{"text": "hi"}]}, "finishReason": "STOP", "index": 0}],
    "usageMetadata": {"promptTokenCount": INPUT_TOKENS, "candidatesTokenCount": OUTPUT_TOKENS,
                      "cachedContentTokenCount": CACHE_READ_TOKENS,
                      "totalTokenCount": INPUT_TOKENS + OUTPUT_TOKENS},
    "modelVersion": GEMINI_MODEL, "responseId": "r1"}
_GEMINI_EMBEDDING = {"embeddings": [{"values": [0.1, 0.2]}]}
_GEMINI_IMAGES = {"predictions": [{"bytesBase64Encoded": "aGk=", "mimeType": "image/png"}]}


class _GeminiHandler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        self.rfile.read(int(self.headers.get("content-length") or 0))
        if "streamGenerateContent" in self.path:
            body, content_type = _sse([(None, _GEMINI_RESPONSE)]), "text/event-stream"
        elif "embedContent" in self.path or "batchEmbedContents" in self.path:
            body, content_type = json.dumps(_GEMINI_EMBEDDING).encode(), "application/json"
        elif ":predict" in self.path:
            body, content_type = json.dumps(_GEMINI_IMAGES).encode(), "application/json"
        else:
            body, content_type = json.dumps(_GEMINI_RESPONSE).encode(), "application/json"
        self.send_response(200)
        self.send_header("content-type", content_type)
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


_gemini_base_url = []
_gemini_lock = threading.Lock()
_gemini_clients = []


def _gemini_server_url():
    with _gemini_lock:
        if not _gemini_base_url:
            server = ThreadingHTTPServer(("127.0.0.1", 0), _GeminiHandler)
            threading.Thread(target=server.serve_forever, daemon=True).start()
            _gemini_base_url.append(f"http://127.0.0.1:{server.server_address[1]}")
        return _gemini_base_url[0]


def _genai(**client_kwargs):
    from google import genai
    from google.genai import types
    client = genai.Client(api_key="stub", http_options=types.HttpOptions(base_url=_gemini_server_url()),
                          **client_kwargs)
    # genai closes the transport when the client is garbage collected, which can
    # happen mid-call for an unreferenced temporary client.
    _gemini_clients.append(client)
    return client


def _genai_vertex_express():
    """google-genai serves generate_images only in Vertex mode; an API key selects express mode, no ADC needed."""
    return _genai(vertexai=True)


@call("genai.models.generate_content.sync")
def _():
    _genai().models.generate_content(model=GEMINI_MODEL, contents="hi")


@call("genai.models.generate_content_stream.sync")
def _():
    for _chunk in _genai().models.generate_content_stream(model=GEMINI_MODEL, contents="hi"):
        pass


@call("genai.models.embed_content.sync")
def _():
    _genai().models.embed_content(model=GEMINI_EMBED_MODEL, contents="hi")


@call("genai.chats.send_message.sync")
def _():
    _genai().chats.create(model=GEMINI_MODEL).send_message("hi")


@call("genai.chats.send_message_stream.sync")
def _():
    for _chunk in _genai().chats.create(model=GEMINI_MODEL).send_message_stream("hi"):
        pass


@call("genai.models.generate_images.sync")
def _():
    _genai_vertex_express().models.generate_images(model=IMAGEN_MODEL, prompt="a cat")


@call("genai.aio.models.generate_content.async")
async def _():
    await _genai().aio.models.generate_content(model=GEMINI_MODEL, contents="hi")


@call("genai.aio.models.generate_content_stream.async")
async def _():
    async for _chunk in await _genai().aio.models.generate_content_stream(model=GEMINI_MODEL, contents="hi"):
        pass


@call("genai.aio.models.embed_content.async")
async def _():
    await _genai().aio.models.embed_content(model=GEMINI_EMBED_MODEL, contents="hi")


@call("genai.aio.chats.send_message.async")
async def _():
    await _genai().aio.chats.create(model=GEMINI_MODEL).send_message("hi")


@call("genai.aio.chats.send_message_stream.async")
async def _():
    async for _chunk in await _genai().aio.chats.create(model=GEMINI_MODEL).send_message_stream("hi"):
        pass


@call("genai.aio.models.generate_images.async")
async def _():
    await _genai_vertex_express().aio.models.generate_images(model=IMAGEN_MODEL, prompt="a cat")


# --- Ollama ------------------------------------------------------------------

OLLAMA_HOST = "http://127.0.0.1:11434"
OLLAMA_SELF_HOSTED = "http://ollama.internal:8080"
_OLLAMA_COUNTS = {"prompt_eval_count": INPUT_TOKENS, "eval_count": OUTPUT_TOKENS}


def _ollama_chunks(key, final_value):
    head = {"model": OLLAMA_MODEL, "created_at": "2026-01-01T00:00:00Z", "done": False}
    last = {**head, "done": True, "done_reason": "stop", **_OLLAMA_COUNTS}
    if key == "message":
        head["message"] = {"role": "assistant", "content": final_value}
        last["message"] = {"role": "assistant", "content": ""}
    else:
        head[key] = final_value
        last[key] = ""
    return (json.dumps(head) + "\n" + json.dumps(last) + "\n").encode()


def _ollama_handler(request):
    body = json.loads(request.content or b"{}")
    path = request.url.path
    streamed = body.get("stream", True) is not False
    done = {"model": OLLAMA_MODEL, "created_at": "2026-01-01T00:00:00Z", "done": True, "done_reason": "stop",
            **_OLLAMA_COUNTS}
    if path == "/api/chat":
        if streamed:
            return httpx.Response(200, content=_ollama_chunks("message", "hi"))
        return httpx.Response(200, json={**done, "message": {"role": "assistant", "content": "hi"}})
    if path == "/api/generate":
        if streamed:
            return httpx.Response(200, content=_ollama_chunks("response", "hi"))
        return httpx.Response(200, json={**done, "response": "hi"})
    if path == "/api/embed":
        return httpx.Response(200, json={"model": OLLAMA_MODEL, "embeddings": [[0.1, 0.2]],
                                         "prompt_eval_count": INPUT_TOKENS})
    if path == "/api/embeddings":
        return httpx.Response(200, json={"embedding": [0.1, 0.2]})
    return httpx.Response(404, json={"error": path})


@contextlib.contextmanager
def _ollama_default_client():
    import ollama
    default = ollama._client
    original = default._client
    default._client = httpx.Client(transport=httpx.MockTransport(_ollama_handler), base_url=OLLAMA_HOST)
    try:
        yield ollama
    finally:
        default._client = original


def _ollama_client():
    import ollama
    return ollama.Client(host=OLLAMA_SELF_HOSTED, transport=httpx.MockTransport(_ollama_handler))


def _ollama_async_client():
    import ollama
    return ollama.AsyncClient(host=OLLAMA_SELF_HOSTED, transport=httpx.MockTransport(_ollama_handler))


@call("ollama.chat.sync")
def _():
    with _ollama_default_client() as ollama:
        ollama.chat(model=OLLAMA_MODEL, messages=USER_MESSAGES)


@call("ollama.chat.sync.stream")
def _():
    with _ollama_default_client() as ollama:
        for _chunk in ollama.chat(model=OLLAMA_MODEL, messages=USER_MESSAGES, stream=True):
            pass


@call("ollama.generate.sync")
def _():
    with _ollama_default_client() as ollama:
        ollama.generate(model=OLLAMA_MODEL, prompt="hi")


@call("ollama.embed.sync")
def _():
    with _ollama_default_client() as ollama:
        ollama.embed(model=OLLAMA_MODEL, input="hi")


@call("ollama.embeddings.sync")
def _():
    with _ollama_default_client() as ollama:
        ollama.embeddings(model=OLLAMA_MODEL, prompt="hi")


@call("ollama.client.chat.sync")
def _():
    _ollama_client().chat(model=OLLAMA_MODEL, messages=USER_MESSAGES)


@call("ollama.client.chat.sync.stream")
def _():
    for _chunk in _ollama_client().chat(model=OLLAMA_MODEL, messages=USER_MESSAGES, stream=True):
        pass


@call("ollama.client.generate.sync")
def _():
    _ollama_client().generate(model=OLLAMA_MODEL, prompt="hi")


@call("ollama.client.generate.sync.stream")
def _():
    for _chunk in _ollama_client().generate(model=OLLAMA_MODEL, prompt="hi", stream=True):
        pass


@call("ollama.client.embed.sync")
def _():
    _ollama_client().embed(model=OLLAMA_MODEL, input="hi")


@call("ollama.async_client.chat.async")
async def _():
    await _ollama_async_client().chat(model=OLLAMA_MODEL, messages=USER_MESSAGES)


@call("ollama.async_client.chat.async.stream")
async def _():
    async for _chunk in await _ollama_async_client().chat(model=OLLAMA_MODEL, messages=USER_MESSAGES, stream=True):
        pass


@call("ollama.async_client.generate.async")
async def _():
    await _ollama_async_client().generate(model=OLLAMA_MODEL, prompt="hi")


@call("ollama.async_client.generate.async.stream")
async def _():
    async for _chunk in await _ollama_async_client().generate(model=OLLAMA_MODEL, prompt="hi", stream=True):
        pass


@call("ollama.async_client.embed.async")
async def _():
    await _ollama_async_client().embed(model=OLLAMA_MODEL, input="hi")


@call("ollama.async_client.embeddings.async")
async def _():
    await _ollama_async_client().embeddings(model=OLLAMA_MODEL, prompt="hi")


# --- LiteLLM -----------------------------------------------------------------

def _litellm_kwargs():
    return {"model": LITELLM_MODEL, "messages": USER_MESSAGES, "mock_response": "hello world"}


@call("litellm.completion.sync")
def _():
    import litellm
    litellm.completion(**_litellm_kwargs())


@call("litellm.completion.sync.stream_with_usage")
def _():
    import litellm
    for _chunk in litellm.completion(**_litellm_kwargs(), stream=True, stream_options={"include_usage": True}):
        pass


@call("litellm.completion.sync.stream_without_usage")
def _():
    import litellm
    for _chunk in litellm.completion(**_litellm_kwargs(), stream=True):
        pass


@call("litellm.acompletion.async")
async def _():
    import litellm
    await litellm.acompletion(**_litellm_kwargs())


@call("litellm.acompletion.async.stream_with_usage")
async def _():
    import litellm
    stream = await litellm.acompletion(**_litellm_kwargs(), stream=True, stream_options={"include_usage": True})
    async for _chunk in stream:
        pass


@call("litellm.acompletion.async.stream_without_usage")
async def _():
    import litellm
    stream = await litellm.acompletion(**_litellm_kwargs(), stream=True)
    async for _chunk in stream:
        pass


@call("litellm.batch_completion.sync")
def _():
    import litellm
    litellm.batch_completion(model=LITELLM_MODEL, messages=[USER_MESSAGES], mock_response="hello world")


def _litellm_text_kwargs():
    return {"model": LITELLM_MODEL, "prompt": "hello", "mock_response": "hello world"}


@call("litellm.text_completion.sync")
def _():
    import litellm
    litellm.text_completion(**_litellm_text_kwargs())


@call("litellm.text_completion.sync.stream_without_usage")
def _():
    import litellm
    for _chunk in litellm.text_completion(**_litellm_text_kwargs(), stream=True):
        pass


@call("litellm.atext_completion.async")
async def _():
    import litellm
    await litellm.atext_completion(**_litellm_text_kwargs())


@call("litellm.completion_with_retries.sync")
def _():
    import litellm
    litellm.completion_with_retries(**_litellm_kwargs())


@call("litellm.completion_with_retries.sync.over_wrapped_completion")
def _():
    import litellm
    litellm.completion_with_retries(**_litellm_kwargs(), original_function=litellm.completion)


@call("litellm.acompletion_with_retries.async")
async def _():
    import litellm
    await litellm.acompletion_with_retries(**_litellm_kwargs())


def _litellm_embedding_kwargs():
    return {"model": "text-embedding-3-small", "input": ["hello"], "mock_response": [0.1, 0.2]}


@call("litellm.embedding.sync")
def _():
    import litellm
    litellm.embedding(**_litellm_embedding_kwargs())


@call("litellm.aembedding.async")
async def _():
    import litellm
    await litellm.aembedding(**_litellm_embedding_kwargs())


# --- Perplexity ----------------------------------------------------------------

PERPLEXITY_BASE_URL = "https://api.perplexity.ai"


def _perplexity_handler():
    return _openai_compatible_handler(PERPLEXITY_MODEL, _PERPLEXITY_USAGE)


def _perplexity_native():
    import perplexity
    transport = httpx.MockTransport(_perplexity_handler())
    return perplexity.Perplexity(api_key="stub", http_client=httpx.Client(transport=transport))


def _async_perplexity_native():
    import perplexity
    transport = httpx.MockTransport(_perplexity_handler())
    return perplexity.AsyncPerplexity(api_key="stub", http_client=httpx.AsyncClient(transport=transport))


@call("perplexity.native.chat.create.sync")
def _():
    _perplexity_native().chat.completions.create(model=PERPLEXITY_MODEL, messages=USER_MESSAGES)


@call("perplexity.native.chat.create.sync.stream")
def _():
    for _chunk in _perplexity_native().chat.completions.create(model=PERPLEXITY_MODEL, messages=USER_MESSAGES,
                                                               stream=True):
        pass


@call("perplexity.native.chat.create.async")
async def _():
    await _async_perplexity_native().chat.completions.create(model=PERPLEXITY_MODEL, messages=USER_MESSAGES)


@call("perplexity.native.chat.create.async.stream")
async def _():
    stream = await _async_perplexity_native().chat.completions.create(model=PERPLEXITY_MODEL,
                                                                      messages=USER_MESSAGES, stream=True)
    async for _chunk in stream:
        pass


@call("perplexity.openai_compatible.chat.create.sync")
def _():
    client = _openai(base_url=PERPLEXITY_BASE_URL, model=PERPLEXITY_MODEL, usage=_PERPLEXITY_USAGE)
    client.chat.completions.create(**_chat_kwargs(PERPLEXITY_MODEL))


@call("perplexity.openai_compatible.chat.create.sync.stream")
def _():
    client = _openai(base_url=PERPLEXITY_BASE_URL, model=PERPLEXITY_MODEL, usage=_PERPLEXITY_USAGE)
    for _chunk in client.chat.completions.create(**_chat_kwargs(PERPLEXITY_MODEL), stream=True):
        pass


@call("perplexity.openai_compatible.chat.create.async")
async def _():
    client = _async_openai(base_url=PERPLEXITY_BASE_URL, model=PERPLEXITY_MODEL, usage=_PERPLEXITY_USAGE)
    await client.chat.completions.create(**_chat_kwargs(PERPLEXITY_MODEL))


@call("perplexity.openai_compatible.chat.create.async.stream")
async def _():
    client = _async_openai(base_url=PERPLEXITY_BASE_URL, model=PERPLEXITY_MODEL, usage=_PERPLEXITY_USAGE)
    stream = await client.chat.completions.create(**_chat_kwargs(PERPLEXITY_MODEL), stream=True)
    async for _chunk in stream:
        pass


# --- fal -----------------------------------------------------------------------

_FAL_RESULT = {"images": [{"url": "https://stub/y.png", "width": 1024, "height": 1024,
                           "content_type": "image/png"}], "seed": 1, "request_id": "req-1"}
_FAL_STREAM_EVENTS = [{"status": "IN_PROGRESS"}, _FAL_RESULT]
_FAL_REQUESTS_URL = "https://queue.fal.run/" + FAL_APP + "/requests/"
_FAL_ARGUMENTS = {"prompt": "cat"}
_FAL_POLLS_BEFORE_COMPLETION = 1
_fal_request_ids = itertools.count(1)
_fal_status_polls = {}


def _fal_submitted():
    request_id = f"req-{next(_fal_request_ids)}"
    base = _FAL_REQUESTS_URL + request_id
    return {"request_id": request_id, "response_url": base, "status_url": base + "/status",
            "cancel_url": base + "/cancel"}


def _fal_status(request_id):
    polls = _fal_status_polls[request_id] = _fal_status_polls.get(request_id, 0) + 1
    if polls <= _FAL_POLLS_BEFORE_COMPLETION:
        return {"status": "IN_PROGRESS", "logs": []}
    return {"status": "COMPLETED", "logs": [], "metrics": {}}


def _fal_queue(request):
    if request.method == "POST":
        return _fal_submitted()
    request_id = request.url.path.rpartition("/requests/")[2].split("/")[0]
    if request.url.path.endswith("/status"):
        return _fal_status(request_id)
    return {**_FAL_RESULT, "request_id": request_id}


def _fal_handler(request):
    headers = {"x-fal-request-id": "req-1"}
    if request.url.host == "queue.fal.run":
        return httpx.Response(200, headers=headers, json=_fal_queue(request))
    if request.url.path.endswith("/stream"):
        return httpx.Response(200, headers={**headers, "content-type": "text/event-stream"},
                              content=_sse([(None, event) for event in _FAL_STREAM_EVENTS]))
    return httpx.Response(200, headers=headers, json=_FAL_RESULT)


@contextlib.contextmanager
def _fal_key():
    previous = os.environ.get("FAL_KEY")
    os.environ["FAL_KEY"] = "stub:stub"
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("FAL_KEY", None)
        else:
            os.environ["FAL_KEY"] = previous


@contextlib.contextmanager
def _stubbed_http(fal_client_object, http_client):
    cached = fal_client_object.__dict__
    had_client = "_client" in cached
    previous = cached.get("_client")
    cached["_client"] = http_client
    try:
        yield fal_client_object
    finally:
        if had_client:
            cached["_client"] = previous
        else:
            cached.pop("_client", None)


def _fal_sync_http():
    return httpx.Client(transport=httpx.MockTransport(_fal_handler))


class _Awaited:
    """fal's async client caches ``_client`` as an awaitable, not the client itself."""

    def __init__(self, value):
        self._value = value

    def __await__(self):
        if False:
            yield
        return self._value


def _fal_async_http():
    return _Awaited(httpx.AsyncClient(transport=httpx.MockTransport(_fal_handler)))


@call("fal.run.sync")
def _():
    import fal_client
    with _fal_key(), _stubbed_http(fal_client.sync_client, _fal_sync_http()):
        fal_client.run(FAL_APP, arguments=_FAL_ARGUMENTS)


@call("fal.run_async.async")
async def _():
    import fal_client
    with _fal_key(), _stubbed_http(fal_client.async_client, _fal_async_http()):
        await fal_client.run_async(FAL_APP, arguments=_FAL_ARGUMENTS)


@call("fal.sync_client.run.sync")
def _():
    from fal_client import SyncClient
    with _stubbed_http(SyncClient(key="stub:stub"), _fal_sync_http()) as client:
        client.run(FAL_APP, arguments=_FAL_ARGUMENTS)


@call("fal.async_client.run.async")
async def _():
    from fal_client import AsyncClient
    with _stubbed_http(AsyncClient(key="stub:stub"), _fal_async_http()) as client:
        await client.run(FAL_APP, arguments=_FAL_ARGUMENTS)


@call("fal.submit.sync")
def _():
    import fal_client
    with _fal_key(), _stubbed_http(fal_client.sync_client, _fal_sync_http()):
        fal_client.submit(FAL_APP, arguments=_FAL_ARGUMENTS).get()


@call("fal.submit_async.async")
async def _():
    import fal_client
    with _fal_key(), _stubbed_http(fal_client.async_client, _fal_async_http()):
        handle = await fal_client.submit_async(FAL_APP, arguments=_FAL_ARGUMENTS)
        await handle.get()


@call("fal.submit.status.result.sync")
def _():
    import fal_client
    with _fal_key(), _stubbed_http(fal_client.sync_client, _fal_sync_http()):
        request_id = fal_client.submit(FAL_APP, arguments=_FAL_ARGUMENTS).request_id
        fal_client.status(FAL_APP, request_id)
        fal_client.result(FAL_APP, request_id)


@call("fal.submit_async.status.result.async")
async def _():
    import fal_client
    with _fal_key(), _stubbed_http(fal_client.async_client, _fal_async_http()):
        request_id = (await fal_client.submit_async(FAL_APP, arguments=_FAL_ARGUMENTS)).request_id
        await fal_client.status_async(FAL_APP, request_id)
        await fal_client.result_async(FAL_APP, request_id)


@call("fal.sync_client.submit.sync")
def _():
    from fal_client import SyncClient
    with _stubbed_http(SyncClient(key="stub:stub"), _fal_sync_http()) as client:
        client.submit(FAL_APP, arguments=_FAL_ARGUMENTS).get()


@call("fal.async_client.submit.async")
async def _():
    from fal_client import AsyncClient
    with _stubbed_http(AsyncClient(key="stub:stub"), _fal_async_http()) as client:
        await (await client.submit(FAL_APP, arguments=_FAL_ARGUMENTS)).get()


@call("fal.subscribe.sync")
def _():
    import fal_client
    with _fal_key(), _stubbed_http(fal_client.sync_client, _fal_sync_http()):
        fal_client.subscribe(FAL_APP, arguments=_FAL_ARGUMENTS)


@call("fal.subscribe_async.async")
async def _():
    import fal_client
    with _fal_key(), _stubbed_http(fal_client.async_client, _fal_async_http()):
        await fal_client.subscribe_async(FAL_APP, arguments=_FAL_ARGUMENTS)


@call("fal.sync_client.subscribe.sync")
def _():
    from fal_client import SyncClient
    with _stubbed_http(SyncClient(key="stub:stub"), _fal_sync_http()) as client:
        client.subscribe(FAL_APP, arguments=_FAL_ARGUMENTS)


@call("fal.async_client.subscribe.async")
async def _():
    from fal_client import AsyncClient
    with _stubbed_http(AsyncClient(key="stub:stub"), _fal_async_http()) as client:
        await client.subscribe(FAL_APP, arguments=_FAL_ARGUMENTS)


@call("fal.stream.sync.stream")
def _():
    import fal_client
    with _fal_key(), _stubbed_http(fal_client.sync_client, _fal_sync_http()):
        list(fal_client.stream(FAL_APP, arguments=_FAL_ARGUMENTS))


@call("fal.stream_async.async.stream")
async def _():
    import fal_client
    with _fal_key(), _stubbed_http(fal_client.async_client, _fal_async_http()):
        [event async for event in fal_client.stream_async(FAL_APP, arguments=_FAL_ARGUMENTS)]


@call("fal.sync_client.stream.sync.stream")
def _():
    from fal_client import SyncClient
    with _stubbed_http(SyncClient(key="stub:stub"), _fal_sync_http()) as client:
        list(client.stream(FAL_APP, arguments=_FAL_ARGUMENTS))


@call("fal.async_client.stream.async.stream")
async def _():
    from fal_client import AsyncClient
    with _stubbed_http(AsyncClient(key="stub:stub"), _fal_async_http()) as client:
        [event async for event in client.stream(FAL_APP, arguments=_FAL_ARGUMENTS)]


# --- Vertex AI -----------------------------------------------------------------

VERTEX_PROJECT = "stub-project"
VERTEX_LOCATION = "us-central1"
VERTEX_EMBED_MODEL = "text-embedding-004"
VERTEX_GA = "vertexai.generative_models"
VERTEX_PREVIEW = "vertexai.preview.generative_models"
# The GA classes speak aiplatform v1, whose UsageMetadata has no cached-token
# field in google-cloud-aiplatform 1.71; the preview classes speak v1beta1.
_VERTEX_GAPIC_TYPES = {VERTEX_GA: "google.cloud.aiplatform_v1.types",
                       VERTEX_PREVIEW: "google.cloud.aiplatform_v1beta1.types"}


def _vertex_chunk(module_path, text, final=True):
    import importlib
    types = importlib.import_module(_VERTEX_GAPIC_TYPES[module_path])
    content = types.Content(role="model", parts=[types.Part(text=text)])
    if not final:
        return types.GenerateContentResponse(candidates=[types.Candidate(index=0, content=content)],
                                             model_version=GEMINI_MODEL)
    usage = {"prompt_token_count": INPUT_TOKENS, "candidates_token_count": OUTPUT_TOKENS,
             "total_token_count": INPUT_TOKENS + OUTPUT_TOKENS}
    if module_path == VERTEX_PREVIEW:
        usage["cached_content_token_count"] = CACHE_READ_TOKENS
    candidate = types.Candidate(index=0, content=content, finish_reason=types.Candidate.FinishReason.STOP)
    return types.GenerateContentResponse(candidates=[candidate], model_version=GEMINI_MODEL,
                                         usage_metadata=types.GenerateContentResponse.UsageMetadata(**usage))


def _vertex_response(module_path):
    return _vertex_chunk(module_path, "hi")


def _vertex_chunks(module_path):
    return [_vertex_chunk(module_path, "h", final=False), _vertex_chunk(module_path, "i")]


async def _async_iterate(items):
    for item in items:
        yield item


class _VertexPredictionClient:
    def __init__(self, module_path):
        self._module_path = module_path

    def generate_content(self, request, **kwargs):
        return _vertex_response(self._module_path)

    def stream_generate_content(self, request, **kwargs):
        return iter(_vertex_chunks(self._module_path))


class _VertexPredictionAsyncClient(_VertexPredictionClient):
    async def generate_content(self, request, **kwargs):
        return _vertex_response(self._module_path)

    async def stream_generate_content(self, request, **kwargs):
        return _async_iterate(_vertex_chunks(self._module_path))


def _vertex_predict_response():
    from google.cloud.aiplatform_v1.types import PredictResponse
    embedding = {"values": [0.1, 0.2], "statistics": {"token_count": INPUT_TOKENS, "truncated": False}}
    return PredictResponse.from_json(json.dumps({"predictions": [{"embeddings": embedding}]}))


class _VertexEndpointClient:
    def predict(self, endpoint, instances, parameters, timeout=None):
        return _vertex_predict_response()


class _VertexEndpointAsyncClient:
    async def predict(self, endpoint, instances, parameters, timeout=None):
        return _vertex_predict_response()


def _vertex_init():
    import vertexai
    from google.auth.credentials import AnonymousCredentials
    vertexai.init(project=VERTEX_PROJECT, location=VERTEX_LOCATION, credentials=AnonymousCredentials())


# The SDK builds its gapic clients lazily and reuses whatever it finds cached
# under these attributes, so the request still goes through the SDK's own
# request building and response parsing, and never reaches the network.
def vertex_model(module_path=VERTEX_GA):
    import importlib
    _vertex_init()
    model = importlib.import_module(module_path).GenerativeModel(GEMINI_MODEL)
    model._prediction_client_value = _VertexPredictionClient(module_path)
    model._prediction_async_client_value = _VertexPredictionAsyncClient(module_path)
    return model


def vertex_embedding_model():
    from vertexai.language_models import TextEmbeddingModel
    _vertex_init()
    endpoint = f"projects/{VERTEX_PROJECT}/locations/{VERTEX_LOCATION}/publishers/google/models/{VERTEX_EMBED_MODEL}"
    model = TextEmbeddingModel(model_id=VERTEX_EMBED_MODEL, endpoint_name=endpoint)
    model._endpoint._prediction_client_value = _VertexEndpointClient()
    model._endpoint._prediction_async_client_value = _VertexEndpointAsyncClient()
    return model


@call("vertex.generative_model.generate_content.sync")
def _():
    vertex_model().generate_content("hi")


@call("vertex.generative_model.generate_content.sync.stream")
def _():
    for _chunk in vertex_model().generate_content("hi", stream=True):
        pass


@call("vertex.generative_model.generate_content_async")
async def _():
    await vertex_model().generate_content_async("hi")


@call("vertex.generative_model.generate_content_async.stream")
async def _():
    async for _chunk in await vertex_model().generate_content_async("hi", stream=True):
        pass


@call("vertex.preview.generative_model.generate_content_async")
async def _():
    await vertex_model(VERTEX_PREVIEW).generate_content_async("hi")


@call("vertex.text_embedding_model.get_embeddings.sync")
def _():
    vertex_embedding_model().get_embeddings(["hi"])


@call("vertex.text_embedding_model.get_embeddings_async")
async def _():
    await vertex_embedding_model().get_embeddings_async(["hi"])


@call("vertex.chat_session.send_message")
def _():
    vertex_model().start_chat().send_message("hi")


@call("vertex.chat_session.send_message.stream")
def _():
    for _chunk in vertex_model().start_chat().send_message("hi", stream=True):
        pass


@call("vertex.chat_session.send_message_async")
async def _():
    await vertex_model().start_chat().send_message_async("hi")


@call("vertex.chat_session.send_message_async.stream")
async def _():
    async for _chunk in await vertex_model().start_chat().send_message_async("hi", stream=True):
        pass


@call("vertex.preview.chat_session.send_message")
def _():
    vertex_model(VERTEX_PREVIEW).start_chat().send_message("hi")


def run_isolated(call_ids, middleware_modules):
    """Run calls in this fresh interpreter, importing only ``middleware_modules``.

    With no middleware the result is the differential baseline. With middleware,
    payloads are recorded at the metering client, below ``submit_ai_event``.
    """
    recorder = _install_recorder(middleware_modules) if middleware_modules else None
    outcomes = {}
    for call_id in call_ids:
        if recorder is not None:
            recorder.reset_mock()
        try:
            run_call(call_id)
            error = None
        except Exception as exc:  # noqa: BLE001
            error = f"{type(exc).__name__}: {exc}"[:500]
        payloads = []
        if recorder is not None:
            wait_for_metering()
            payloads = recorded_payloads(recorder)
        outcomes[call_id] = {"error": error, "payloads": payloads}
    return outcomes


PAYLOAD_FIELDS = ("model", "provider", "operation_type", "input_token_count",
                  "output_token_count", "cache_read_token_count", "reasoning_token_count",
                  "actual_image_count", "requested_image_count")
OPERATIONS = ("completion", "image", "video", "audio")


def recorded_payloads(recording_client):
    payloads = []
    for operation in OPERATIONS:
        for recorded in getattr(recording_client.ai, f"create_{operation}").call_args_list:
            fields = {name: recorded.kwargs[name] for name in PAYLOAD_FIELDS if name in recorded.kwargs}
            payloads.append({"operation": operation, **fields})
    return payloads


def wait_for_metering(timeout=10.0):
    from revenium_middleware._core.metering_pool import wait_until_idle
    wait_until_idle(timeout)


def _install_recorder(middleware_modules):
    import importlib
    from unittest.mock import MagicMock, patch

    recorder = MagicMock()
    patch("revenium_middleware._core.metering.client", recorder).start()
    for module in middleware_modules:
        importlib.import_module(module)
    return recorder


def main(argv):
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--middleware", default="", help="comma-separated modules to import first")
    parser.add_argument("--calls", default="", help="comma-separated call ids; default all")
    options = parser.parse_args(argv)
    middleware = [m for m in options.middleware.split(",") if m]
    call_ids = [c for c in options.calls.split(",") if c] or list(CALLS)
    outcomes = run_isolated(call_ids, middleware)
    loaded = sorted(m for m in sys.modules if m.startswith("revenium_middleware"))
    print(json.dumps({"revenium_modules": loaded, "outcomes": outcomes}, default=str))


if __name__ == "__main__":
    main(sys.argv[1:])
