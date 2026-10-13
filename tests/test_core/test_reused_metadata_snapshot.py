"""A call is recorded with the metadata it was made with, not what the caller's dict holds later (BACK-4372).

A batch job that keeps one ``subscriber`` dict and fills it in before each
call had every call filed under the next customer in the list: the SDK copied
only the top level of ``usage_metadata``, and the record, built on the
metering thread after the call returned, read the nested dict the caller had
already changed. Every record queued by these tests is held until the whole
loop has run, the worst case, so the result does not depend on thread timing.
"""
import copy
from types import SimpleNamespace

import pytest

from revenium_middleware._core import metering_pool
from revenium_middleware._core.context import injected_metadata_scope, merge_metadata, snapshot_metadata

from .test_metering_key_switch import (  # noqa: F401  keys is a fixture
    anthropic_kwargs, anthropic_messages, drain, gemini_chunk, keys, ollama_chunks, openai_chunks,
    openai_completion, responses_events,
)
from .test_tool_metering_dispatch import endpoint  # noqa: F401  a fixture

CALLS = 20


@pytest.fixture
def held_pool(monkeypatch):
    """Hold every queued metering task until ``release()``, then deliver them all."""
    held = []
    submit = metering_pool.MeteringWorkerPool.submit
    monkeypatch.setattr(metering_pool.MeteringWorkerPool, "submit", lambda pool, task: held.append((pool, task)))

    def release():
        monkeypatch.setattr(metering_pool.MeteringWorkerPool, "submit", submit)
        for pool, task in held:
            submit(pool, task)
        held.clear()
        drain()

    return SimpleNamespace(release=release)


# --- one call per site, with the metadata it is given --------------------------

def openai_chat_call(usage_metadata):
    from revenium_middleware.openai import middleware as mw
    mw.create_wrapper(lambda *a, **k: openai_completion("CustomerA"), SimpleNamespace(_client=None), (),
                      {"model": "gpt-4o-mini", "messages": [], "usage_metadata": usage_metadata})


def openai_chat_stream_call(usage_metadata):
    from revenium_middleware.openai import middleware as mw
    list(mw.create_wrapper(lambda *a, **k: iter(openai_chunks()), SimpleNamespace(_client=None), (),
                           {"model": "gpt-4o-mini", "messages": [], "stream": True, "usage_metadata": usage_metadata}))


def openai_responses_call(usage_metadata):
    from revenium_middleware.openai import middleware as mw
    list(mw.responses_create_wrapper(lambda *a, **k: iter(responses_events()), SimpleNamespace(_client=None), (),
                                     {"model": "o4-mini", "stream": True, "usage_metadata": usage_metadata}))


def litellm_call(usage_metadata):
    pytest.importorskip("litellm")
    from revenium_middleware.litellm.client import middleware as litellm_mw
    usage = SimpleNamespace(prompt_tokens=3, completion_tokens=5, total_tokens=8)
    chunks = [SimpleNamespace(usage=None, choices=[], model="gpt-4o-mini", id="chatcmpl-reuse"),
              SimpleNamespace(usage=usage, choices=[], model="gpt-4o-mini", id="chatcmpl-reuse")]
    list(litellm_mw.completion_wrapper(lambda *a, **k: iter(chunks), None, (),
                                       {"model": "gpt-4o-mini", "messages": [], "stream": True,
                                        "usage_metadata": usage_metadata}))


def perplexity_sdk_call(usage_metadata):
    pytest.importorskip("perplexity")
    from revenium_middleware.perplexity import perplexity_sdk
    usage = SimpleNamespace(prompt_tokens=3, completion_tokens=5, total_tokens=8)
    chunks = [SimpleNamespace(usage=None, choices=[SimpleNamespace(finish_reason=None)]),
              SimpleNamespace(usage=usage, choices=[SimpleNamespace(finish_reason="stop")])]
    list(perplexity_sdk.perplexity_create_wrapper(lambda *a, **k: iter(chunks), None, (),
                                                  {"model": "sonar", "messages": [], "stream": True,
                                                   "usage_metadata": usage_metadata}))


def gemini_call(usage_metadata):
    from revenium_middleware.google.google_ai import middleware as genai_mw
    chunks = [gemini_chunk("Hel"), gemini_chunk("lo", total=8)]
    list(genai_mw.generate_content_stream_wrapper(lambda *a, **k: iter(chunks), SimpleNamespace(_api_client=None),
                                                  (), {"model": "gemini-2.5-flash", "contents": "hi",
                                                       "usage_metadata": usage_metadata}))


def vertex_call(usage_metadata):
    pytest.importorskip("vertexai")
    from revenium_middleware.google.vertex_ai import middleware as vertex_mw
    chunks = [gemini_chunk("Hel"), gemini_chunk("lo", total=8)]
    list(vertex_mw.generate_content_wrapper_impl(lambda *a, **k: iter(chunks),
                                                 SimpleNamespace(_model_name="gemini-2.5-flash"),
                                                 ("hi",), {"stream": True, "usage_metadata": usage_metadata}))


def ollama_call(usage_metadata):
    pytest.importorskip("ollama")
    from revenium_middleware.ollama import middleware as ollama_mw
    generator = (chunk for chunk in ollama_chunks())
    list(ollama_mw.sync_client_wrapper("chat")(lambda *a, **k: generator, None, (),
                                                {"model": "llama3", "messages": [], "stream": True,
                                                 "usage_metadata": usage_metadata}))


def fal_call(usage_metadata):
    pytest.importorskip("fal_client")
    import contextvars
    from revenium_middleware.fal._call import FalCall
    from revenium_middleware.fal.middleware import _metered_events
    events = [{"output": "h", "partial": True},
              {"output": "hi", "partial": False, "usage": {"prompt_tokens": 3, "completion_tokens": 5}}]
    call = FalCall.from_request(("openrouter/router", {"prompt": "hi"}), {"usage_metadata": usage_metadata})
    list(_metered_events(iter(events), call, contextvars.copy_context()))


def anthropic_call(usage_metadata):
    list(anthropic_messages(async_client=False).create(**dict(anthropic_kwargs(stream=True),
                                                              usage_metadata=usage_metadata)))


def converse_response():
    return {
        "ResponseMetadata": {"RequestId": "req-reuse"},
        "usage": {"inputTokens": 20, "outputTokens": 9, "totalTokens": 29},
        "stopReason": "end_turn",
        "output": {"message": {"role": "assistant", "content": []}},
    }


def bedrock_transport_call(usage_metadata):
    from revenium_middleware.anthropic import bedrock_transport
    from revenium_middleware.anthropic.middleware import usage_context
    instance = SimpleNamespace(meta=SimpleNamespace(
        service_model=SimpleNamespace(service_name="bedrock-runtime"), region_name="us-east-1"))
    token = usage_context.set(usage_metadata)
    try:
        bedrock_transport._make_api_call_wrapper(
            lambda *a, **k: converse_response(), instance,
            ("Converse", {"modelId": "us.anthropic.claude-3-5-sonnet-20241022-v2:0"}), {})
    finally:
        usage_context.reset(token)


def langchain_handler_call(usage_metadata):
    pytest.importorskip("langchain_core")
    import uuid
    from langchain_core.messages import AIMessage
    from langchain_core.outputs import ChatGeneration, LLMResult
    from revenium_middleware.openai.langchain.unified_handler import UnifiedReveniumCallbackHandler

    handler = UnifiedReveniumCallbackHandler(usage_metadata=usage_metadata)
    run_id = uuid.uuid4()
    serialized = {"id": ["langchain", "chat_models", "ollama", "ChatOllama"], "kwargs": {"model": "llama3"}}
    handler.on_llm_start(serialized, ["hi"], run_id=run_id, invocation_params={"model": "llama3"})
    message = AIMessage(content="hello", usage_metadata={"input_tokens": 3, "output_tokens": 5, "total_tokens": 8})
    handler.on_llm_end(LLMResult(generations=[[ChatGeneration(message=message)]],
                                 llm_output={"model_name": "llama3"}), run_id=run_id)


# --- each site as a caller that reuses one subscriber dict ----------------------
#
# A site takes the shared ``subscriber`` dict and returns ``call(trace_id)``, one
# metered call made the way that integration is used.

def per_call_metadata(make_call):
    def site(subscriber):
        return lambda trace_id: make_call({"subscriber": subscriber, "trace_id": trace_id})
    return site


def fresh_dict_site(subscriber):
    return lambda trace_id: openai_chat_call({"subscriber": dict(subscriber), "trace_id": trace_id})


def revenium_metadata_site(subscriber):
    from revenium_middleware._core.decorators import revenium_metadata

    @revenium_metadata(subscriber=subscriber)
    def handle(trace_id):
        openai_chat_call({"trace_id": trace_id})
    return handle


def griptape_driver_site(subscriber):
    pytest.importorskip("griptape")
    from griptape.common.prompt_stack.prompt_stack import PromptStack
    from revenium_middleware.griptape.openai_driver import ReveniumOpenAiDriver
    from revenium_middleware.openai import middleware as mw

    driver = ReveniumOpenAiDriver(model="gpt-4o-mini", api_key="test-key", usage_metadata={"subscriber": subscriber})

    def call(trace_id):
        driver.usage_metadata["trace_id"] = trace_id
        prompt_stack = PromptStack()
        prompt_stack.add_user_message("hi")
        params = driver._base_params(prompt_stack)
        params.pop("stream", None)
        mw.create_wrapper(lambda *a, **k: openai_completion("CustomerA"), SimpleNamespace(_client=None), (), params)
    return call


SITES = {
    "openai_chat": per_call_metadata(openai_chat_call),
    "openai_chat_stream": per_call_metadata(openai_chat_stream_call),
    "openai_responses": per_call_metadata(openai_responses_call),
    "litellm": per_call_metadata(litellm_call),
    "perplexity_sdk": per_call_metadata(perplexity_sdk_call),
    "gemini": per_call_metadata(gemini_call),
    "vertex": per_call_metadata(vertex_call),
    "ollama": per_call_metadata(ollama_call),
    "fal": per_call_metadata(fal_call),
    "revenium_metadata": revenium_metadata_site,
    "langchain_handler": per_call_metadata(langchain_handler_call),
    "griptape_driver": griptape_driver_site,
    "anthropic": per_call_metadata(anthropic_call),
    "bedrock_transport": per_call_metadata(bedrock_transport_call),
    "fresh_dict": fresh_dict_site,
}


def recorded_subscribers(client):
    """``(trace id, subscriber id)`` of every usage record ``client`` was asked to deliver."""
    return sorted(
        (kwargs.get("trace_id"), (kwargs.get("subscriber") or {}).get("id"))
        for name, _args, kwargs in client.ai.method_calls
        if name.startswith("create_")
    )


def expected_subscribers():
    return sorted((f"call-{i}", f"user-{i}") for i in range(CALLS))


@pytest.fixture
def bedrock_transport_on(monkeypatch):
    monkeypatch.setenv("REVENIUM_BEDROCK_TRANSPORT", "1")


def test_merged_metadata_does_not_share_nested_dicts_with_the_caller():
    caller = {"subscriber": {"id": "user-0", "credential": {"name": "key-0"}}, "tags": ["a"]}
    before = copy.deepcopy(caller)

    merged = merge_metadata(caller)

    assert caller == before
    caller["subscriber"]["id"] = "user-1"
    caller["subscriber"]["credential"]["name"] = "key-1"
    caller["tags"].append("b")
    assert merged == before

    scoped_subscriber = {"id": "user-0", "credential": {"name": "key-0"}}
    with injected_metadata_scope({"subscriber": scoped_subscriber}):
        scoped_subscriber["id"] = "user-1"
        scoped_subscriber["credential"]["name"] = "key-1"
        assert merge_metadata({"trace_id": "t"})["subscriber"] == {"id": "user-0", "credential": {"name": "key-0"}}

    looped = {"id": "user-0"}
    looped["self"] = looped
    merged_loop = merge_metadata({"subscriber": looped})
    assert merged_loop["subscriber"]["id"] == "user-0"


class UnreadableDict(dict):
    def items(self):
        raise RuntimeError("dictionary changed size during iteration")

    def __iter__(self):
        raise RuntimeError("dictionary changed size during iteration")


def test_snapshot_metadata_never_raises_on_unreadable_or_very_deep_metadata():
    unreadable = UnreadableDict(id="user-0")
    assert snapshot_metadata({"subscriber": unreadable})["subscriber"] is unreadable
    twice = snapshot_metadata({"subscriber": unreadable, "owner": unreadable})
    assert twice["subscriber"] is unreadable
    assert twice["owner"] is unreadable

    deep = leaf = {}
    for _ in range(2000):
        leaf["next"] = {}
        leaf = leaf["next"]
    copied = snapshot_metadata({"deep": deep})
    assert copied["deep"] is not deep

    lock = object()
    assert snapshot_metadata({"client": lock, "pair": (1, [2])}) == {"client": lock, "pair": (1, [2])}
    assert snapshot_metadata({"client": lock})["client"] is lock


def test_snapshot_metadata_copies_a_tuple_reused_inside_the_metadata_once():
    shared = (["tag"],)
    copied = snapshot_metadata({"pair": (shared, shared)})
    assert copied["pair"][0] is copied["pair"][1]
    assert copied["pair"] == ((["tag"],), (["tag"],))


@pytest.mark.parametrize("site", sorted(SITES))
def test_a_reused_subscriber_dict_records_each_call_under_its_own_user(site, keys, held_pool, bedrock_transport_on):
    subscriber = {"id": None, "email": None}
    call = SITES[site](subscriber)

    for i in range(CALLS):
        subscriber["id"] = f"user-{i}"
        subscriber["email"] = f"user-{i}@example.com"
        call(f"call-{i}")
    held_pool.release()

    assert recorded_subscribers(keys.a) == expected_subscribers()


def test_report_tool_call_in_a_loop_records_each_event_with_the_values_it_was_called_with(endpoint, held_pool):
    from revenium_middleware._metering.decorator import report_tool_call

    shared = {"step": None, "subscriber": {"id": None}}
    for i in range(CALLS):
        shared["step"] = i
        shared["subscriber"]["id"] = f"user-{i}"
        report_tool_call(tool_id="reused-metadata-tool", duration_ms=5, success=True, usage_metadata=shared)
    held_pool.release()

    sent = sorted((call.json["usageMetadata"]["step"], call.json["usageMetadata"]["subscriber"]["id"])
                  for call in endpoint.calls)
    assert sent == [(i, f"user-{i}") for i in range(CALLS)]


@pytest.mark.parametrize("site", sorted(SITES))
def test_the_sdk_never_changes_the_callers_metadata_dict(site, keys, bedrock_transport_on):
    subscriber = {"id": "user-0", "email": "user-0@example.com", "credential": {"name": "key", "value": "v"}}
    before = copy.deepcopy(subscriber)

    SITES[site](subscriber)("call-0")
    drain()

    assert subscriber == before
