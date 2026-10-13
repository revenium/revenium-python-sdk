"""Perplexity answers carrying ``usage.cost``, for ChatPerplexity and the Perplexity SDK.

Run as a module (``python -m entry_points.perplexity_cost <scenario>`` with
``tests`` on the path) it drives one scenario in a process that never imports
``revenium_middleware.perplexity``, which the test process always has, and
prints the records it sent as JSON.
"""
import asyncio
import json
import os
import sys

import httpx

PERPLEXITY_MODEL = "sonar"
PERPLEXITY_BASE_URL = "https://api.perplexity.ai"
INPUT_COST = 0.000002
OUTPUT_COST = 0.000003
REQUEST_FEE = 0.005
TOTAL_COST = 0.005005
ANSWER = "hi"
BUILT_IN_TOOLS = [{"type": "web_search"}]
BATCH_COSTS = {"first prompt": (0.011, 11), "second prompt": (0.022, 22), "third prompt": (0.033, 33)}

_CHAT_USAGE = {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5,
               "cost": {"input_tokens_cost": INPUT_COST, "output_tokens_cost": OUTPUT_COST,
                        "request_cost": REQUEST_FEE, "total_cost": TOTAL_COST}}
_CHAT_BODY = {"id": "pplx-chat-0001", "object": "chat.completion", "created": 1760000000, "model": PERPLEXITY_MODEL,
              "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": ANSWER}}],
              "usage": _CHAT_USAGE}
_RESPONSES_BODY = {
    "id": "resp_0001", "object": "response", "created_at": 1760000000, "model": PERPLEXITY_MODEL,
    "status": "completed",
    "output": [{"type": "message", "id": "msg_1", "role": "assistant", "status": "completed",
                "content": [{"type": "output_text", "text": ANSWER, "annotations": []}]}],
    "usage": {"input_tokens": 2, "output_tokens": 3, "total_tokens": 5,
              "cost": {"currency": "USD", "input_cost": INPUT_COST, "output_cost": OUTPUT_COST,
                       "tool_calls_cost": REQUEST_FEE, "total_cost": TOTAL_COST}},
}


_UNCHARGED = {"input_tokens_cost": 0, "output_tokens_cost": 0, "request_cost": 0, "total_cost": 0}


def _cumulative_usage(prompt_tokens, completion_tokens, cost):
    return {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens, "cost": cost}


def _chat_chunks(prompt_tokens=2, cost=_CHAT_USAGE["cost"], chat_id="pplx-chat-0002"):
    """Perplexity's stream: every chunk carries the usage so far, and only the last one the charge."""
    base = {"id": chat_id, "object": "chat.completion.chunk", "created": 1760000000, "model": PERPLEXITY_MODEL}
    chunks = [
        {**base, "choices": [{"index": 0, "delta": {"role": "assistant", "content": ANSWER[0]}, "finish_reason": None}],
         "usage": _cumulative_usage(prompt_tokens, 1, _UNCHARGED)},
        {**base, "choices": [{"index": 0, "delta": {"content": ANSWER[1:]}, "finish_reason": None}],
         "usage": _cumulative_usage(prompt_tokens, 2, _UNCHARGED)},
        {**base, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
         "usage": _cumulative_usage(prompt_tokens, 3, cost)},
    ]
    return _sse(chunks)


def _sse(events):
    return "".join(f"data: {json.dumps(event)}\n\n" for event in events).encode() + b"data: [DONE]\n\n"


def _responses_events():
    return _sse([
        {"type": "response.output_text.delta", "sequence_number": 1, "item_id": "msg_1", "output_index": 0,
         "content_index": 0, "delta": ANSWER},
        {"type": "response.completed", "sequence_number": 2, "response": _RESPONSES_BODY},
    ])


def _batched_chat_body(prompt):
    total_cost, prompt_tokens = BATCH_COSTS[prompt]
    usage = {**_CHAT_USAGE, "prompt_tokens": prompt_tokens, "total_tokens": prompt_tokens + 3,
             "cost": {**_CHAT_USAGE["cost"], "total_cost": total_cost}}
    return {**_CHAT_BODY, "id": f"pplx-chat-{prompt_tokens}", "usage": usage}


def _batched_chat_chunks(prompt):
    total_cost, prompt_tokens = BATCH_COSTS[prompt]
    return _chat_chunks(prompt_tokens, {**_CHAT_USAGE["cost"], "total_cost": total_cost},
                        f"pplx-chat-{prompt_tokens}")


def _event_stream(content):
    return httpx.Response(200, content=content, headers={"content-type": "text/event-stream"})


def _handler(request):
    body = json.loads(request.content or b"{}")
    streamed = bool(body.get("stream"))
    prompt = (body.get("messages") or [{}])[-1].get("content")
    if prompt in BATCH_COSTS:
        if streamed:
            return _event_stream(_batched_chat_chunks(prompt))
        return httpx.Response(200, json=_batched_chat_body(prompt))
    if request.url.path.endswith("/responses"):
        if streamed:
            return _event_stream(_responses_events())
        return httpx.Response(200, json=_RESPONSES_BODY)
    if streamed:
        return _event_stream(_chat_chunks())
    return httpx.Response(200, json=_CHAT_BODY)


def perplexity_clients():
    import perplexity

    transport = httpx.MockTransport(_handler)
    return (perplexity.Perplexity(api_key="stub", http_client=httpx.Client(transport=transport)),
            perplexity.AsyncPerplexity(api_key="stub", http_client=httpx.AsyncClient(transport=transport)))


def chat_perplexity(callbacks):
    from langchain_perplexity import ChatPerplexity

    client, async_client = perplexity_clients()
    return ChatPerplexity(model=PERPLEXITY_MODEL, api_key="stub", client=client, async_client=async_client,
                          callbacks=callbacks)


def openai_clients_at(base_url):
    """OpenAI clients pointed at ``base_url``, answered with Perplexity's bodies."""
    import openai

    transport = httpx.MockTransport(_handler)
    return (openai.OpenAI(base_url=base_url, api_key="stub", max_retries=0,
                          http_client=httpx.Client(transport=transport)),
            openai.AsyncOpenAI(base_url=base_url, api_key="stub", max_retries=0,
                               http_client=httpx.AsyncClient(transport=transport)))


def _openai_clients_at_perplexity():
    return openai_clients_at(PERPLEXITY_BASE_URL)


def _chat_openai_given_the_base_url(**fields):
    from langchain_openai import ChatOpenAI

    transport = httpx.MockTransport(_handler)
    return ChatOpenAI(base_url=PERPLEXITY_BASE_URL, http_client=httpx.Client(transport=transport),
                      http_async_client=httpx.AsyncClient(transport=transport), **fields)


def _chat_openai_on_the_openai_base_url_variable(**fields):
    """The OpenAI SDK reads OPENAI_BASE_URL itself, so LangChain never sees the URL."""
    from langchain_openai import ChatOpenAI

    os.environ["OPENAI_BASE_URL"] = PERPLEXITY_BASE_URL
    transport = httpx.MockTransport(_handler)
    return ChatOpenAI(http_client=httpx.Client(transport=transport),
                      http_async_client=httpx.AsyncClient(transport=transport), **fields)


def _chat_openai_on_a_callers_client(**fields):
    from langchain_openai import ChatOpenAI

    client, async_client = _openai_clients_at_perplexity()
    return ChatOpenAI(root_client=client, client=client.chat.completions, root_async_client=async_client,
                      async_client=async_client.chat.completions, **fields)


_CHAT_OPENAI_BASE_URL_SOURCES = {
    "argument": _chat_openai_given_the_base_url,
    "env": _chat_openai_on_the_openai_base_url_variable,
    "root_client": _chat_openai_on_a_callers_client,
}


def chat_openai_at_perplexity(callbacks, base_url_source="argument", **fields):
    """LangChain's ChatOpenAI calling Perplexity, with the base URL given the way ``base_url_source`` names."""
    return _CHAT_OPENAI_BASE_URL_SOURCES[base_url_source](
        model=PERPLEXITY_MODEL, api_key="stub", max_retries=0, callbacks=callbacks, **fields)


def recorded_costs(recording_client):
    return [{name: entry.kwargs.get(name) for name in
             ("provider", "model_source", "transaction_id", "total_cost", "input_token_cost", "output_token_cost",
              "input_token_count", "output_token_count")}
            for entry in recording_client.ai.create_completion.call_args_list]


def _callback_scenario(call):
    def run():
        from revenium_middleware.openai.langchain import ReveniumCallbackHandler

        call(chat_perplexity([ReveniumCallbackHandler()]))
    return run


def _chat_openai_scenario(call, base_url_source="argument", **fields):
    def run():
        from revenium_middleware.openai.langchain import ReveniumCallbackHandler

        call(chat_openai_at_perplexity([ReveniumCallbackHandler()], base_url_source, **fields))
    return run


def _direct_chat_after_both_imports():
    import revenium_middleware.openai.langchain  # noqa: F401
    import revenium_middleware.perplexity  # noqa: F401
    from revenium_middleware._core.patch_registry import is_patched

    integration = "perplexity.resources.chat.completions.CompletionsResource.create"
    if not is_patched(integration):
        raise SystemExit(f"{integration} was not installed")
    client, _ = perplexity_clients()
    client.chat.completions.create(model=PERPLEXITY_MODEL, messages=[{"role": "user", "content": "hi"}])


def _chat_openai_stream_with_both_integrations(first, then):
    """Imports the two integrations in the order given, so either one's wrap can be the outer one."""
    def run():
        import importlib

        importlib.import_module(first)
        importlib.import_module(then)
        from revenium_middleware.openai.langchain import ReveniumCallbackHandler

        list(chat_openai_at_perplexity([ReveniumCallbackHandler()]).stream("hi"))
    return run


async def _drain(chunks):
    return [chunk async for chunk in chunks]


def _batched_prompts():
    from langchain_core.messages import HumanMessage

    return [[HumanMessage(content=prompt)] for prompt in BATCH_COSTS]


def _cached_batch_scenario(cached_prompt, batch):
    """Warms LangChain's cache with ``cached_prompt`` alone, then runs the whole batch through ``batch``."""
    def call(model):
        from langchain_core.caches import InMemoryCache
        from langchain_core.globals import set_llm_cache
        from langchain_core.messages import HumanMessage

        set_llm_cache(InMemoryCache())
        batch(model, [[HumanMessage(content=cached_prompt)]])
        batch(model, _batched_prompts())
    return _callback_scenario(call)


def _generate(model, prompts):
    return model.generate(prompts)


def _agenerate(model, prompts):
    return asyncio.run(model.agenerate(prompts))


SCENARIOS = {
    "callback-chat": _callback_scenario(lambda model: model.invoke("hi")),
    "callback-chat-ainvoke": _callback_scenario(lambda model: asyncio.run(model.ainvoke("hi"))),
    "callback-chat-stream": _callback_scenario(lambda model: list(model.stream("hi"))),
    "callback-chat-astream": _callback_scenario(lambda model: asyncio.run(_drain(model.astream("hi")))),
    "callback-built-in-tool": _callback_scenario(lambda model: model.invoke("hi", tools=BUILT_IN_TOOLS)),
    "callback-built-in-tool-ainvoke": _callback_scenario(
        lambda model: asyncio.run(model.ainvoke("hi", tools=BUILT_IN_TOOLS))),
    "callback-built-in-tool-stream": _callback_scenario(
        lambda model: list(model.stream("hi", tools=BUILT_IN_TOOLS))),
    "callback-built-in-tool-astream": _callback_scenario(
        lambda model: asyncio.run(_drain(model.astream("hi", tools=BUILT_IN_TOOLS)))),
    "callback-batched-generate": _callback_scenario(lambda model: model.generate(_batched_prompts())),
    "callback-batched-agenerate": _callback_scenario(
        lambda model: asyncio.run(model.agenerate(_batched_prompts()))),
    "callback-cached-batched-generate": _cached_batch_scenario("first prompt", _generate),
    "callback-cached-batched-agenerate": _cached_batch_scenario("first prompt", _agenerate),
    "callback-middle-cached-batched-generate": _cached_batch_scenario("second prompt", _generate),
    "callback-middle-cached-batched-agenerate": _cached_batch_scenario("second prompt", _agenerate),
    "callback-chat-openai": _chat_openai_scenario(lambda model: model.invoke("hi")),
    "callback-chat-openai-ainvoke": _chat_openai_scenario(lambda model: asyncio.run(model.ainvoke("hi"))),
    "callback-chat-openai-stream": _chat_openai_scenario(lambda model: list(model.stream("hi"))),
    "callback-chat-openai-astream": _chat_openai_scenario(lambda model: asyncio.run(_drain(model.astream("hi")))),
    "callback-chat-openai-stream-env": _chat_openai_scenario(lambda model: list(model.stream("hi")), "env"),
    "callback-chat-openai-stream-root-client": _chat_openai_scenario(
        lambda model: list(model.stream("hi")), "root_client"),
    "callback-chat-openai-streaming-invoke": _chat_openai_scenario(lambda model: model.invoke("hi"), streaming=True),
    "callback-chat-openai-streaming-ainvoke": _chat_openai_scenario(
        lambda model: asyncio.run(model.ainvoke("hi")), streaming=True),
    "callback-chat-openai-stream-raw-response": _chat_openai_scenario(
        lambda model: list(model.stream("hi")), include_response_headers=True),
    "callback-chat-openai-streaming-batch": _chat_openai_scenario(
        lambda model: model.batch(_batched_prompts()), streaming=True),
    "callback-chat-openai-env": _chat_openai_scenario(lambda model: model.invoke("hi"), "env"),
    "callback-chat-openai-root-client": _chat_openai_scenario(lambda model: model.invoke("hi"), "root_client"),
    "integration-imported-after-callback": _direct_chat_after_both_imports,
    "chat-openai-stream-perplexity-integration-first": _chat_openai_stream_with_both_integrations(
        "revenium_middleware.perplexity", "revenium_middleware.openai"),
    "chat-openai-stream-openai-integration-first": _chat_openai_stream_with_both_integrations(
        "revenium_middleware.openai", "revenium_middleware.perplexity"),
}


def main(scenario):
    from unittest.mock import MagicMock, patch

    recorder = MagicMock()
    with patch("revenium_middleware._core.metering.client", recorder):
        SCENARIOS[scenario]()
        from revenium_middleware._core.metering_pool import wait_until_idle
        wait_until_idle(10.0)
    print(json.dumps(recorded_costs(recorder)))


if __name__ == "__main__":
    main(sys.argv[1])
