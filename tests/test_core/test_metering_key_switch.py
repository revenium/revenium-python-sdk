"""A record goes out under the Revenium key of the call it belongs to (BACK-4257).

A service that meters several of its own customers switches keys between
them. A stream, a Veo video or a LangChain callback run that started under
customer A's key and finished after a switch to customer B's key was sent
under B's key, and so was a call on another thread. Streams keep the key that
was current when they started; ``metering_key()`` gives each thread or asyncio
task its own key.
"""
import asyncio
import contextlib
import contextvars
import datetime
import gc
import json
import os
import threading
import urllib.request
import uuid
from collections import OrderedDict
from types import SimpleNamespace
from unittest.mock import ANY, MagicMock, patch

import httpx
import pytest

import revenium_middleware
from revenium_middleware._core import metering, metering_pool, tool_event_transport
from revenium_middleware._metering.context import context

NOW = datetime.datetime.now(datetime.timezone.utc)
KEY_A = "hak_key_a_fake"
KEY_B = "hak_key_b_fake"
KEY_C = "hak_key_c_fake"
KEY_P = "hak_key_process_fake"
BASE_URL = "https://api.test.revenium.ai/meter/"


def fake_client(api_key, base_url=BASE_URL):
    client = MagicMock(name=api_key)
    client.api_key = api_key
    client.base_url = base_url
    return client


def records(client):
    """The usage records ``client`` was asked to deliver, as (operation, organization name)."""
    return [
        (name, kwargs.get("organization_name"))
        for name, _args, kwargs in client.ai.method_calls
        if name.startswith("create_")
    ]


def drain():
    assert metering_pool.drain(5) == 0


class FakeClients:
    """Stands in for ``metering._new_client``: one fake client per (key, base URL)."""

    def __init__(self):
        self.built = {}

    def __call__(self, api_key, base_url=None, http_client=None):
        return self.client(api_key, base_url or BASE_URL)

    def client(self, api_key, base_url=BASE_URL):
        if (api_key, base_url) not in self.built:
            self.built[(api_key, base_url)] = fake_client(api_key, base_url)
        return self.built[(api_key, base_url)]


@pytest.fixture
def keys(monkeypatch):
    """Customer A's key is current; ``keys.switch()`` makes customer B's key current, as
    ``initialize_metering(api_key=B)`` does."""
    factory = FakeClients()
    monkeypatch.setattr(metering, "_new_client", factory)
    monkeypatch.setattr(metering, "_scoped_clients", OrderedDict(), raising=False)
    a, b = fake_client(KEY_A), fake_client(KEY_B)
    monkeypatch.setattr(metering, "client", a)
    return SimpleNamespace(a=a, b=b, factory=factory, switch=lambda: monkeypatch.setattr(metering, "client", b))


def assert_delivered_under_a_only(keys):
    drain()
    delivered = {"under key A": len(records(keys.a)), "under key B": len(records(keys.b))}
    assert delivered == {"under key A": 1, "under key B": 0}


# --- OpenAI chat streams -----------------------------------------------------

def openai_chunks():
    def chunk(content, usage=None):
        return SimpleNamespace(
            id="chatcmpl-switch", model="gpt-4o-mini", system_fingerprint=None, usage=usage,
            choices=[SimpleNamespace(index=0, delta=SimpleNamespace(content=content, role="assistant"),
                                     finish_reason="stop" if usage else None)],
        )
    usage = SimpleNamespace(prompt_tokens=3, completion_tokens=5, total_tokens=8,
                            prompt_tokens_details=None, completion_tokens_details=None)
    return [chunk("Hel"), chunk("lo"), chunk("", usage)]


def drive_openai_stream(switch):
    from revenium_middleware.openai import middleware as mw
    stream = mw.handle_streaming_response(iter(openai_chunks()), NOW, {"organizationName": "CustomerA"}, None, {})
    next(stream)
    switch()
    list(stream)


def drive_async_openai_stream(switch):
    from revenium_middleware.openai import middleware as mw

    async def chunks():
        for chunk in openai_chunks():
            yield chunk

    async def consume():
        stream = mw._wrap_async_stream(chunks(), NOW, {"organizationName": "CustomerA"})
        await stream.__anext__()
        switch()
        async for _ in stream:
            pass

    asyncio.run(consume())


def test_a_stream_started_under_key_a_is_delivered_under_key_a_after_a_switch(keys):
    drive_openai_stream(keys.switch)

    drain()
    assert records(keys.a) == [("create_completion", "CustomerA")]
    assert records(keys.b) == []


def test_an_async_openai_stream_keeps_its_key_across_a_switch(keys):
    drive_async_openai_stream(keys.switch)

    drain()
    assert records(keys.a) == [("create_completion", "CustomerA")]
    assert records(keys.b) == []


# --- Veo -----------------------------------------------------------------------

def test_a_veo_video_polled_after_a_switch_is_delivered_under_the_submit_key(keys):
    from revenium_middleware.google.google_ai import middleware as genai_mw

    name = f"operations/veo-{uuid.uuid4()}"
    submitted = SimpleNamespace(name=name, done=False)
    genai_mw._meter_generate_videos(
        submitted, None, (), {"model": "veo-3.0-generate-001"}, {"organizationName": "CustomerA"}, NOW)
    keys.switch()
    finished = SimpleNamespace(name=name, done=True, error=None, response=None,
                               result=SimpleNamespace(generated_videos=[object()]))
    genai_mw._meter_finished_video(finished, polled_name=name)

    drain()
    assert records(keys.a) == [("create_video", "CustomerA")]
    assert records(keys.b) == []


# --- every other stream site ----------------------------------------------------

def drive_responses_stream(switch):
    from revenium_middleware.openai import middleware as mw
    usage = SimpleNamespace(input_tokens=3, output_tokens=5, total_tokens=8,
                            input_tokens_details=None, output_tokens_details=None)

    def event(kind, status, usage=None):
        return SimpleNamespace(type=kind, response=SimpleNamespace(
            id="resp_switch", model="o4-mini", usage=usage, status=status, incomplete_details=None))

    stream = mw.handle_streaming_responses(
        iter([event("response.created", "in_progress"), event("response.completed", "completed", usage)]),
        NOW, {"organizationName": "CustomerA"})
    next(stream)
    switch()
    list(stream)


def drive_transcription_stream(switch):
    from revenium_middleware.openai import audio

    events = [
        SimpleNamespace(type="transcript.text.delta", delta="Hel"),
        SimpleNamespace(type=audio.TRANSCRIPT_DONE_EVENT, text="Hello",
                        usage=SimpleNamespace(type="tokens", input_tokens=3, output_tokens=5)),
    ]
    wrapper = audio._audio_wrapper(audio.TRANSCRIPTION)
    stream = wrapper(lambda *a, **k: iter(events), SimpleNamespace(_client=None), (),
                     {"model": "gpt-4o-transcribe", "file": b"audio", "stream": True,
                      "usage_metadata": {"organizationName": "CustomerA"}})
    next(stream)
    switch()
    list(stream)


def anthropic_events():
    from anthropic.types import (
        Message, RawContentBlockDeltaEvent, RawContentBlockStartEvent, RawContentBlockStopEvent,
        RawMessageDeltaEvent, RawMessageStartEvent, RawMessageStopEvent, TextBlock, TextDelta, Usage,
    )
    started = Message(id="msg_switch", content=[], model="claude-haiku-4-5", role="assistant", stop_reason=None,
                      stop_sequence=None, type="message", usage=Usage(input_tokens=3, output_tokens=1))
    return [
        RawMessageStartEvent(type="message_start", message=started),
        RawContentBlockStartEvent(type="content_block_start", index=0, content_block=TextBlock(type="text", text="")),
        RawContentBlockDeltaEvent(type="content_block_delta", index=0, delta=TextDelta(type="text_delta", text="Hi")),
        RawContentBlockStopEvent(type="content_block_stop", index=0),
        RawMessageDeltaEvent(type="message_delta", delta={"stop_reason": "end_turn", "stop_sequence": None},
                             usage={"output_tokens": 5}),
        RawMessageStopEvent(type="message_stop"),
    ]


class SyncRawStream:
    """Stands in for anthropic.Stream at the messages resource's network seam."""

    def __init__(self, events):
        self._events = events

    def __iter__(self):
        return iter(self._events)

    def close(self):
        pass


class AsyncRawStream:
    def __init__(self, events):
        self._events = events

    async def __aiter__(self):
        for event in self._events:
            yield event

    async def close(self):
        pass


def anthropic_kwargs(**extra):
    return {"model": "claude-haiku-4-5", "max_tokens": 16, "messages": [{"role": "user", "content": "hi"}],
            "usage_metadata": {"organizationName": "CustomerA"}, **extra}


def anthropic_messages(async_client, before_reply=lambda: None):
    anthropic = pytest.importorskip("anthropic")
    import revenium_middleware.anthropic  # noqa: F401  installs the SDK patches
    client = anthropic.AsyncAnthropic(api_key="test-key") if async_client else anthropic.Anthropic(api_key="test-key")
    messages = client.messages
    raw = AsyncRawStream(anthropic_events()) if async_client else SyncRawStream(anthropic_events())
    if async_client:
        async def fake_post(*args, **kwargs):
            before_reply()
            return raw
    else:
        def fake_post(*args, **kwargs):
            before_reply()
            return raw
    messages._post = fake_post
    return messages


def drive_anthropic_raw_stream(switch):
    stream = anthropic_messages(async_client=False).create(**anthropic_kwargs(stream=True))
    iterator = iter(stream)
    next(iterator)
    switch()
    list(iterator)


def drive_async_anthropic_raw_stream(switch):
    messages = anthropic_messages(async_client=True)

    async def consume():
        stream = await messages.create(**anthropic_kwargs(stream=True))
        iterator = stream.__aiter__()
        await iterator.__anext__()
        switch()
        async for _ in iterator:
            pass

    asyncio.run(consume())


def drive_anthropic_stream_helper(switch):
    with anthropic_messages(async_client=False).stream(**anthropic_kwargs()) as stream:
        iterator = iter(stream)
        next(iterator)
        switch()
        list(iterator)


def drive_async_anthropic_stream_helper(switch):
    messages = anthropic_messages(async_client=True)

    async def consume():
        async with messages.stream(**anthropic_kwargs()) as stream:
            iterator = stream.__aiter__()
            await iterator.__anext__()
            switch()
            async for _ in iterator:
                pass

    asyncio.run(consume())


def drive_bedrock_stream_wrapper(switch):
    from revenium_middleware.anthropic import bedrock_adapter

    chunks = iter(["Hel", "lo"])

    class Texts:
        input_tokens, output_tokens, cache_creation_tokens, cache_read_tokens, cache_creation = 3, 5, 0, 0, None

        def __iter__(self):
            return self

        def __next__(self):
            return next(chunks)

    with patch.object(bedrock_adapter, "bedrock_invoke_stream", return_value=Texts()):
        wrapper = bedrock_adapter.BedrockStreamWrapper(
            "claude-3-haiku", {}, region="us-east-1", usage_metadata={"organizationName": "CustomerA"})
        with wrapper as stream:
            text = stream.text_stream
            next(text)
            switch()
            list(text)


def enqueue_one_record(organization_name):
    """What a provider's metering callback does: queue one usage record."""
    from revenium_middleware._core.metering_submission import submit_ai_event

    async def send():
        submit_ai_event("completion", {"organization_name": organization_name})

    return lambda *_: metering.run_async_in_thread(send())


def drive_bedrock_transport_stream(switch):
    from revenium_middleware.anthropic.bedrock_transport import _MeteredEventStream

    stream = iter(_MeteredEventStream(iter([{"chunk": 1}, {"chunk": 2}]), lambda event: None,
                                      enqueue_one_record("CustomerA")))
    next(stream)
    switch()
    list(stream)


def gemini_chunk(text, total=None):
    usage = SimpleNamespace(prompt_token_count=3, candidates_token_count=5, total_token_count=8,
                            cached_content_token_count=0, thoughts_token_count=0) if total else None
    return SimpleNamespace(
        model_version="gemini-2.5-flash", text=text, usage_metadata=usage,
        candidates=[SimpleNamespace(finish_reason="STOP" if total else None,
                                    content=SimpleNamespace(parts=[SimpleNamespace(text=text)]))],
    )


def drive_gemini_stream(switch):
    from revenium_middleware.google.google_ai import middleware as genai_mw
    stream = genai_mw.handle_streaming_response(
        iter([gemini_chunk("Hel"), gemini_chunk("lo", total=8)]), NOW, {"organizationName": "CustomerA"})
    next(stream)
    switch()
    list(stream)


def drive_vertex_stream(switch):
    pytest.importorskip("vertexai")
    from revenium_middleware.google.vertex_ai import middleware as vertex_mw
    stream = vertex_mw.handle_vertex_ai_streaming_response(
        iter([gemini_chunk("Hel"), gemini_chunk("lo", total=8)]), NOW, {"organizationName": "CustomerA"},
        "gemini-2.5-flash")
    next(stream)
    switch()
    list(stream)


def drive_litellm_stream(switch):
    pytest.importorskip("litellm")
    from revenium_middleware.litellm.client import middleware as litellm_mw
    usage = SimpleNamespace(prompt_tokens=3, completion_tokens=5, total_tokens=8)
    chunks = [SimpleNamespace(usage=None, choices=[], model="gpt-4o-mini", id="chatcmpl-switch"),
              SimpleNamespace(usage=usage, choices=[], model="gpt-4o-mini", id="chatcmpl-switch")]
    stream = litellm_mw.handle_streaming_response(iter(chunks), NOW, {"organizationName": "CustomerA"})
    next(stream)
    switch()
    list(stream)


def drive_ollama_stream(switch):
    pytest.importorskip("ollama")
    from revenium_middleware.ollama import middleware as ollama_mw
    chunks = [SimpleNamespace(model="llama3", message=SimpleNamespace(content="Hel"), done=False),
              SimpleNamespace(model="llama3", message=SimpleNamespace(content="lo"), done=True,
                              prompt_eval_count=3, eval_count=5, done_reason="stop")]
    stream = ollama_mw.handle_streaming_response(
        iter(chunks), NOW, {"organizationName": "CustomerA"}, "txn-switch", "chat", {"model": "llama3"})
    next(stream)
    switch()
    list(stream)


def drive_perplexity_stream(switch):
    from revenium_middleware.perplexity.streaming import MeteredStream, StreamMeter
    stream = MeteredStream(iter(["Hel", "lo"]), StreamMeter(enqueue_one_record("CustomerA")))
    next(stream)
    switch()
    list(stream)


def drive_fal_stream(switch):
    pytest.importorskip("fal_client")
    from revenium_middleware.fal._call import FalCall
    from revenium_middleware.fal.middleware import _metered_events
    events = [{"output": "h", "partial": True},
              {"output": "hi", "partial": False, "usage": {"prompt_tokens": 3, "completion_tokens": 5}}]
    call = FalCall.from_request(("openrouter/router", {"prompt": "hi"}),
                                {"usage_metadata": {"organizationName": "CustomerA"}})
    stream = _metered_events(iter(events), call, contextvars.copy_context())
    next(stream)
    switch()
    list(stream)


def drive_langchain_callback(switch):
    pytest.importorskip("langchain_core")
    from langchain_core.outputs import ChatGeneration, LLMResult
    from langchain_core.messages import AIMessage
    from revenium_middleware.openai.langchain.unified_handler import UnifiedReveniumCallbackHandler

    handler = UnifiedReveniumCallbackHandler(usage_metadata={"organizationName": "CustomerA"})
    run_id = uuid.uuid4()
    serialized = {"id": ["langchain", "chat_models", "ollama", "ChatOllama"], "kwargs": {"model": "llama3"}}
    handler.on_llm_start(serialized, ["hi"], run_id=run_id, invocation_params={"model": "llama3"})
    switch()
    message = AIMessage(content="hello", usage_metadata={"input_tokens": 3, "output_tokens": 5, "total_tokens": 8})
    handler.on_llm_end(LLMResult(generations=[[ChatGeneration(message=message)]],
                                 llm_output={"model_name": "llama3"}), run_id=run_id)


STREAM_SITES = {
    "openai_chat": drive_openai_stream,
    "openai_async_chat": drive_async_openai_stream,
    "openai_responses": drive_responses_stream,
    "openai_transcription": drive_transcription_stream,
    "anthropic_raw_stream": drive_anthropic_raw_stream,
    "anthropic_async_raw_stream": drive_async_anthropic_raw_stream,
    "anthropic_stream_helper": drive_anthropic_stream_helper,
    "anthropic_async_stream_helper": drive_async_anthropic_stream_helper,
    "anthropic_bedrock_stream": drive_bedrock_stream_wrapper,
    "anthropic_bedrock_transport": drive_bedrock_transport_stream,
    "gemini": drive_gemini_stream,
    "vertex": drive_vertex_stream,
    "litellm": drive_litellm_stream,
    "ollama": drive_ollama_stream,
    "perplexity": drive_perplexity_stream,
    "fal": drive_fal_stream,
    "langchain_callback": drive_langchain_callback,
}


@pytest.mark.parametrize("site", sorted(STREAM_SITES))
def test_every_stream_site_keeps_its_key_across_a_switch(site, keys):
    STREAM_SITES[site](keys.switch)

    assert_delivered_under_a_only(keys)


# --- a switch while the provider request is still in flight ---------------------

def customer_a(**kwargs):
    return {"usage_metadata": {"organizationName": "CustomerA"}, **kwargs}


def returning_after(switch, result):
    def wrapped(*args, **kwargs):
        switch()
        return result
    return wrapped


def awaiting_after(switch, result):
    async def wrapped(*args, **kwargs):
        switch()
        return result
    return wrapped


async def async_iter(items):
    for item in items:
        yield item


async def consume_async(stream):
    async for _ in stream:
        pass


def responses_events():
    usage = SimpleNamespace(input_tokens=3, output_tokens=5, total_tokens=8,
                            input_tokens_details=None, output_tokens_details=None)

    def event(kind, status, usage=None):
        return SimpleNamespace(type=kind, response=SimpleNamespace(
            id="resp_switch", model="o4-mini", usage=usage, status=status, incomplete_details=None))

    return [event("response.created", "in_progress"), event("response.completed", "completed", usage)]


def request_openai_stream(switch):
    from revenium_middleware.openai import middleware as mw
    list(mw.create_wrapper(returning_after(switch, iter(openai_chunks())), SimpleNamespace(_client=None), (),
                           customer_a(model="gpt-4o-mini", messages=[], stream=True)))


def request_async_openai_stream(switch):
    from revenium_middleware.openai import middleware as mw

    async def call():
        stream = await mw.async_create_wrapper(
            awaiting_after(switch, async_iter(openai_chunks())), SimpleNamespace(_client=None), (),
            customer_a(model="gpt-4o-mini", messages=[], stream=True))
        await consume_async(stream)

    asyncio.run(call())


def request_responses_stream(switch):
    from revenium_middleware.openai import middleware as mw
    list(mw.responses_create_wrapper(returning_after(switch, iter(responses_events())),
                                     SimpleNamespace(_client=None), (), customer_a(model="o4-mini", stream=True)))


def request_async_responses_stream(switch):
    from revenium_middleware.openai import middleware as mw

    async def call():
        stream = await mw.async_responses_create_wrapper(
            awaiting_after(switch, async_iter(responses_events())), SimpleNamespace(_client=None), (),
            customer_a(model="o4-mini", stream=True))
        await consume_async(stream)

    asyncio.run(call())


def request_anthropic_raw_stream(switch):
    list(anthropic_messages(async_client=False, before_reply=switch).create(**anthropic_kwargs(stream=True)))


def request_async_anthropic_raw_stream(switch):
    messages = anthropic_messages(async_client=True, before_reply=switch)

    async def call():
        await consume_async(await messages.create(**anthropic_kwargs(stream=True)))

    asyncio.run(call())


def request_veo_video(switch):
    from revenium_middleware.google.google_ai import middleware as genai_mw
    name = f"operations/veo-{uuid.uuid4()}"
    genai_mw.generate_videos_wrapper(returning_after(switch, SimpleNamespace(name=name, done=False)), None, (),
                                     customer_a(model="veo-3.0-generate-001", prompt="a cat"))
    finished = SimpleNamespace(name=name, done=True, error=None, response=None,
                               result=SimpleNamespace(generated_videos=[object()]))
    genai_mw.operations_get_wrapper(lambda *a, **k: finished, None, (), {"operation": name})


def request_gemini_stream(switch):
    from revenium_middleware.google.google_ai import middleware as genai_mw
    chunks = iter([gemini_chunk("Hel"), gemini_chunk("lo", total=8)])
    list(genai_mw.generate_content_stream_wrapper(returning_after(switch, chunks), SimpleNamespace(_api_client=None),
                                                  (), customer_a(model="gemini-2.5-flash", contents="hi")))


def request_async_gemini_stream(switch):
    from revenium_middleware.google.google_ai import middleware as genai_mw
    chunks = async_iter([gemini_chunk("Hel"), gemini_chunk("lo", total=8)])

    async def call():
        await consume_async(await genai_mw.async_generate_content_stream_wrapper(
            awaiting_after(switch, chunks), SimpleNamespace(_api_client=None), (),
            customer_a(model="gemini-2.5-flash", contents="hi")))

    asyncio.run(call())


def request_vertex_stream(switch):
    pytest.importorskip("vertexai")
    from revenium_middleware.google.vertex_ai import middleware as vertex_mw
    chunks = iter([gemini_chunk("Hel"), gemini_chunk("lo", total=8)])
    list(vertex_mw.generate_content_wrapper_impl(returning_after(switch, chunks),
                                                 SimpleNamespace(_model_name="gemini-2.5-flash"),
                                                 ("hi",), customer_a(stream=True)))


def request_litellm_stream(switch):
    pytest.importorskip("litellm")
    from revenium_middleware.litellm.client import middleware as litellm_mw
    usage = SimpleNamespace(prompt_tokens=3, completion_tokens=5, total_tokens=8)
    chunks = iter([SimpleNamespace(usage=None, choices=[], model="gpt-4o-mini", id="chatcmpl-switch"),
                   SimpleNamespace(usage=usage, choices=[], model="gpt-4o-mini", id="chatcmpl-switch")])
    list(litellm_mw.completion_wrapper(returning_after(switch, chunks), None, (),
                                       customer_a(model="gpt-4o-mini", messages=[], stream=True)))


def ollama_chunks():
    return [SimpleNamespace(model="llama3", message=SimpleNamespace(content="Hel"), done=False),
            SimpleNamespace(model="llama3", message=SimpleNamespace(content="lo"), done=True,
                            prompt_eval_count=3, eval_count=5, done_reason="stop")]


def request_ollama_stream(switch):
    pytest.importorskip("ollama")
    from revenium_middleware.ollama import middleware as ollama_mw
    generator = (chunk for chunk in ollama_chunks())
    list(ollama_mw.sync_client_wrapper("chat")(returning_after(switch, generator), None, (),
                                                customer_a(model="llama3", messages=[], stream=True)))


def request_async_ollama_stream(switch):
    pytest.importorskip("ollama")
    from revenium_middleware.ollama import middleware as ollama_mw

    async def call():
        await consume_async(await ollama_mw.async_client_wrapper("chat")(
            awaiting_after(switch, async_iter(ollama_chunks())), None, (),
            customer_a(model="llama3", messages=[], stream=True)))

    asyncio.run(call())


def request_perplexity_stream(switch):
    pytest.importorskip("perplexity")
    from revenium_middleware.perplexity import perplexity_sdk
    usage = SimpleNamespace(prompt_tokens=3, completion_tokens=5, total_tokens=8)
    chunks = iter([SimpleNamespace(usage=None, choices=[SimpleNamespace(finish_reason=None)]),
                   SimpleNamespace(usage=usage, choices=[SimpleNamespace(finish_reason="stop")])])
    list(perplexity_sdk.perplexity_create_wrapper(returning_after(switch, chunks), None, (),
                                                  customer_a(model="sonar", messages=[], stream=True)))


def request_transcription_stream(switch):
    from revenium_middleware.openai import audio
    events = [
        SimpleNamespace(type="transcript.text.delta", delta="Hel"),
        SimpleNamespace(type=audio.TRANSCRIPT_DONE_EVENT, text="Hello",
                        usage=SimpleNamespace(type="tokens", input_tokens=3, output_tokens=5)),
    ]
    list(audio._audio_wrapper(audio.TRANSCRIPTION)(
        returning_after(switch, iter(events)), SimpleNamespace(_client=None), (),
        customer_a(model="gpt-4o-transcribe", file=b"audio", stream=True)))


def request_bedrock_transport_stream(switch, monkeypatch):
    from revenium_middleware.anthropic import bedrock_transport
    monkeypatch.setattr(bedrock_transport, "_metered_model_id", lambda instance, args: "anthropic.claude-3-haiku")
    monkeypatch.setattr(bedrock_transport, "_emit_completion",
                        lambda **fields: enqueue_one_record("CustomerA")())
    response = {"body": iter([{"chunk": 1}, {"chunk": 2}])}
    observed = bedrock_transport._make_api_call_wrapper(
        returning_after(switch, response), SimpleNamespace(meta=SimpleNamespace(region_name="us-east-1")),
        ("InvokeModelWithResponseStream", {"modelId": "anthropic.claude-3-haiku"}), {})
    list(observed["body"])


IN_FLIGHT_SITES = {
    "openai_chat": request_openai_stream,
    "openai_async_chat": request_async_openai_stream,
    "openai_responses": request_responses_stream,
    "openai_async_responses": request_async_responses_stream,
    "openai_transcription": request_transcription_stream,
    "anthropic_raw_stream": request_anthropic_raw_stream,
    "anthropic_async_raw_stream": request_async_anthropic_raw_stream,
    "veo": request_veo_video,
    "gemini": request_gemini_stream,
    "gemini_async": request_async_gemini_stream,
    "vertex": request_vertex_stream,
    "litellm": request_litellm_stream,
    "ollama": request_ollama_stream,
    "ollama_async": request_async_ollama_stream,
    "perplexity": request_perplexity_stream,
}


@pytest.mark.parametrize("site", sorted(IN_FLIGHT_SITES))
def test_a_switch_while_the_provider_request_is_in_flight_keeps_the_key_of_the_call(site, keys):
    IN_FLIGHT_SITES[site](keys.switch)

    assert_delivered_under_a_only(keys)


def test_a_bedrock_transport_stream_keeps_its_key_across_a_switch_in_flight(keys, monkeypatch):
    request_bedrock_transport_stream(keys.switch, monkeypatch)

    assert_delivered_under_a_only(keys)


# --- metering_key() --------------------------------------------------------------

def openai_completion(organization_name):
    from openai.types.chat import ChatCompletion
    return ChatCompletion.model_validate({
        "id": f"chatcmpl-{organization_name}", "object": "chat.completion", "created": 1, "model": "gpt-4o-mini",
        "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "hi"}}],
        "usage": {"prompt_tokens": 3, "completion_tokens": 5, "total_tokens": 8},
    })


def call_openai_in_its_own_scope(key, organization_name, inside_the_call):
    """One customer's thread: a chat completion through the OpenAI wrapper, inside ``metering_key(key)``."""
    from revenium_middleware.openai import middleware as mw

    def wrapped(*args, **kwargs):
        inside_the_call()
        return openai_completion(organization_name)

    with revenium_middleware.metering_key(key):
        mw.create_wrapper(wrapped, SimpleNamespace(_client=None), (),
                          {"model": "gpt-4o-mini", "messages": [], "usage_metadata": {"organizationName": organization_name}})


def run_threads(*targets):
    threads = [threading.Thread(target=target) for target in targets]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(10)


@pytest.mark.parametrize("process_switch_inside_the_call", [False, True], ids=["overlapping", "process_switch"])
def test_two_threads_with_their_own_metering_key_deliver_under_their_own_key(keys, process_switch_inside_the_call):
    both_inside = threading.Barrier(2, timeout=5)

    def customer_b_inside_the_call():
        both_inside.wait()
        if process_switch_inside_the_call:
            metering.initialize_metering(api_key=KEY_P)

    run_threads(
        lambda: call_openai_in_its_own_scope(KEY_A, "CustomerA", both_inside.wait),
        lambda: call_openai_in_its_own_scope(KEY_B, "CustomerB", customer_b_inside_the_call),
    )

    drain()
    assert records(keys.factory.client(KEY_A)) == [("create_completion", "CustomerA")]
    assert records(keys.factory.client(KEY_B)) == [("create_completion", "CustomerB")]
    assert records(keys.a) == []
    assert records(keys.factory.client(KEY_P)) == []


def test_metering_key_without_a_base_url_uses_the_base_url_set_in_code(keys, monkeypatch):
    monkeypatch.setattr(metering, "client", fake_client(KEY_P, "https://eu.revenium.example/meter/"))

    with revenium_middleware.metering_key(KEY_A):
        scoped = metering.get_client()

    assert (scoped.api_key, scoped.base_url) == (KEY_A, "https://eu.revenium.example/meter/")


def test_metering_key_with_a_base_url_uses_it(keys):
    with revenium_middleware.metering_key(KEY_A, base_url="https://us.revenium.example/meter/"):
        scoped = metering.get_client()

    assert (scoped.api_key, scoped.base_url) == (KEY_A, "https://us.revenium.example/meter/")


@pytest.mark.parametrize("bad_key", [None, "", "sk-not-a-revenium-key"], ids=["none", "empty", "malformed"])
def test_metering_key_rejects_a_missing_or_malformed_key(keys, bad_key):
    with pytest.raises(ValueError):
        with revenium_middleware.metering_key(bad_key):
            pass


def test_metering_key_scope_ends_with_its_block(keys):
    with revenium_middleware.metering_key(KEY_B):
        pass

    assert metering.get_client() is keys.a


def test_tool_events_inside_metering_key_use_its_key(keys, monkeypatch):
    from revenium_middleware._metering import decorator

    monkeypatch.setattr(decorator, "_metering_url", None)
    monkeypatch.setattr(decorator, "_api_key", None)
    with revenium_middleware.metering_key(KEY_B, base_url="https://customer-b.example"):
        url, key = decorator._resolve_endpoint()
    assert (url, key) == ("https://customer-b.example/meter/v2/tool/events", KEY_B)

    monkeypatch.setattr(decorator, "_api_key", "hak_configured_for_tool_events_fake")
    with revenium_middleware.metering_key(KEY_B, base_url="https://customer-b.example"):
        url, key = decorator._resolve_endpoint()
    assert (url, key) == ("https://api.test.revenium.ai/meter/v2/tool/events", "hak_configured_for_tool_events_fake")


def test_calls_made_entirely_before_or_after_a_switch_keep_their_key(keys):
    from revenium_middleware.openai import middleware as mw

    def chat(organization_name):
        mw.create_wrapper(lambda *a, **k: openai_completion(organization_name), SimpleNamespace(_client=None), (),
                          {"model": "gpt-4o-mini", "messages": [], "usage_metadata": {"organizationName": organization_name}})
        drain()

    chat("CustomerA")
    keys.switch()
    chat("CustomerB")

    assert records(keys.a) == [("create_completion", "CustomerA")]
    assert records(keys.b) == [("create_completion", "CustomerB")]


# --- a non-streamed call whose record is queued when the caller reads the body ----

CUSTOMER_A = {"organizationName": "CustomerA"}


async def resolved(value):
    return await value if asyncio.iscoroutine(value) or asyncio.isfuture(value) else value


def anthropic_stub():
    pytest.importorskip("anthropic")
    pytest.importorskip("httpx2")
    from test_anthropic.test_raw_response import REQUEST, StubAnthropic
    return StubAnthropic(), dict(REQUEST, usage_metadata=CUSTOMER_A)


def perplexity_stub():
    pytest.importorskip("perplexity")
    pytest.importorskip("openai")
    from test_perplexity.test_raw_response import StubPerplexity, _request
    return StubPerplexity("chunked"), dict(_request(False), usage_metadata=CUSTOMER_A)


def read_anthropic_streaming_response(switch, beta=False):
    stub, request = anthropic_stub()
    client = stub.sync_client()
    messages = client.beta.messages if beta else client.messages
    with messages.with_streaming_response.create(**request) as response:
        switch()
        response.parse()


def read_async_anthropic_streaming_response(switch):
    stub, request = anthropic_stub()

    async def call():
        async with stub.async_client().messages.with_streaming_response.create(**request) as response:
            switch()
            await response.parse()

    asyncio.run(call())


def read_perplexity_streaming_response(kind):
    def read(switch):
        stub, request = perplexity_stub()
        with stub.client(kind, "sync").chat.completions.with_streaming_response.create(**request) as response:
            switch()
            response.parse()
    return read


def read_async_perplexity_streaming_response(kind):
    def read(switch):
        stub, request = perplexity_stub()

        async def call():
            create = stub.client(kind, "async").chat.completions.with_streaming_response.create
            async with create(**request) as response:
                switch()
                await response.parse()

        asyncio.run(call())
    return read


def read_perplexity_raw_response(kind, mode):
    def read(switch):
        stub, request = perplexity_stub()
        create = stub.client(kind, mode).chat.completions.with_raw_response.create

        async def call():
            response = await create(**request)
            switch()
            await resolved(response.parse())

        if mode == "async":
            asyncio.run(call())
            return
        response = create(**request)
        switch()
        response.parse()
    return read


BODY_READ_SITES = {
    "anthropic_streaming_response": read_anthropic_streaming_response,
    "anthropic_beta_streaming_response": lambda switch: read_anthropic_streaming_response(switch, beta=True),
    "anthropic_async_streaming_response": read_async_anthropic_streaming_response,
    "perplexity_streaming_response": read_perplexity_streaming_response("native"),
    "perplexity_async_streaming_response": read_async_perplexity_streaming_response("native"),
    "perplexity_openai_client_streaming_response": read_perplexity_streaming_response("openai_compatible"),
    "perplexity_openai_client_async_streaming_response": read_async_perplexity_streaming_response("openai_compatible"),
}

# The SDK reads a non-streamed with_raw_response body before it returns, so
# these were already recorded under the key of the call.
RAW_RESPONSE_SITES = {
    f"perplexity_{kind}_raw_response_{mode}": read_perplexity_raw_response(kind, mode)
    for kind in ("native", "openai_compatible") for mode in ("sync", "async")
}


@pytest.mark.parametrize("site", sorted(BODY_READ_SITES))
def test_a_non_streamed_body_read_after_a_switch_keeps_the_key_of_the_call(site, keys):
    BODY_READ_SITES[site](keys.switch)

    assert_delivered_under_a_only(keys)


@pytest.mark.parametrize("site", sorted(RAW_RESPONSE_SITES))
def test_a_non_streamed_raw_response_parsed_after_a_switch_keeps_the_key_of_the_call(site, keys):
    RAW_RESPONSE_SITES[site](keys.switch)

    assert_delivered_under_a_only(keys)


# --- a call that outlives its metering_key() block, with no process-wide key ------

@pytest.fixture
def only_scoped_keys(keys, monkeypatch):
    monkeypatch.setattr(metering, "client", None)
    monkeypatch.delenv("REVENIUM_METERING_API_KEY", raising=False)
    return keys


def records_under(keys, api_key):
    return [record for (key, _url), client in keys.factory.built.items() if key == api_key
            for record in records(client)]


def drive_fal_queued_result(leave_block):
    pytest.importorskip("fal_client")
    from revenium_middleware.fal import middleware as fal_mw
    submitted = SimpleNamespace(request_id="rq-outlives-scope")
    handle = fal_mw.submit_wrapper(lambda *a, **k: submitted, None, ("fal-ai/flux/dev", {"prompt": "hi"}),
                                   {"usage_metadata": {"organizationName": "CustomerA"}})
    leave_block()
    handle.response_url = "https://queue.fal.run/fal-ai/flux/requests/rq-outlives-scope"
    fal_mw.result_wrapper(lambda *a, **k: {"images": [{"url": "https://fal.example/a.png"}]}, handle, (), {})


def request_perplexity_openai_client_stream(leave_block):
    from revenium_middleware.perplexity import middleware as perplexity_mw
    usage = SimpleNamespace(prompt_tokens=3, completion_tokens=5, total_tokens=8)
    chunks = iter([SimpleNamespace(usage=None, choices=[SimpleNamespace(finish_reason=None)]),
                   SimpleNamespace(usage=usage, choices=[SimpleNamespace(finish_reason="stop")])])
    instance = SimpleNamespace(_client=SimpleNamespace(base_url="https://api.perplexity.ai"))
    list(perplexity_mw.create_wrapper(returning_after(leave_block, chunks), instance, (),
                                      customer_a(model="sonar", messages=[], stream=True)))


OUTLIVING_SITES = {
    "litellm_stream": drive_litellm_stream,
    "ollama_stream": drive_ollama_stream,
    "perplexity_openai_client_stream": request_perplexity_openai_client_stream,
    "fal_queued_result": drive_fal_queued_result,
    "anthropic_streaming_response_body": read_anthropic_streaming_response,
    "perplexity_streaming_response_body": read_perplexity_streaming_response("native"),
}


@pytest.mark.parametrize("site", sorted(OUTLIVING_SITES))
def test_a_call_finishing_after_its_metering_key_block_is_still_delivered_under_that_key(site, only_scoped_keys):
    with contextlib.ExitStack() as scope:
        scope.enter_context(revenium_middleware.metering_key(KEY_A))
        OUTLIVING_SITES[site](scope.close)

    drain()
    assert records_under(only_scoped_keys, KEY_A) == [(ANY, "CustomerA")]


def async_body_read_in_its_own_block(open_response, read):
    """``metering_key()`` is opened inside the coroutine: a ContextVar token
    cannot be reset from another context."""
    async def call():
        with contextlib.ExitStack() as scope:
            scope.enter_context(revenium_middleware.metering_key(KEY_A))
            async with open_response() as response:
                scope.close()
                await read(response)

    asyncio.run(call())


def read_async_anthropic_streaming_response_outside_its_block():
    stub, request = anthropic_stub()
    async_body_read_in_its_own_block(
        lambda: stub.async_client().messages.with_streaming_response.create(**request),
        lambda response: response.parse())


def read_async_perplexity_streaming_response_outside_its_block():
    stub, request = perplexity_stub()
    async_body_read_in_its_own_block(
        lambda: stub.client("native", "async").chat.completions.with_streaming_response.create(**request),
        lambda response: response.parse())


ASYNC_OUTLIVING_SITES = {
    "anthropic_async_streaming_response_body": read_async_anthropic_streaming_response_outside_its_block,
    "perplexity_async_streaming_response_body": read_async_perplexity_streaming_response_outside_its_block,
}


@pytest.mark.parametrize("site", sorted(ASYNC_OUTLIVING_SITES))
def test_an_async_body_read_after_its_metering_key_block_is_still_delivered_under_that_key(site, only_scoped_keys):
    ASYNC_OUTLIVING_SITES[site]()

    drain()
    assert records_under(only_scoped_keys, KEY_A) == [(ANY, "CustomerA")]


@pytest.mark.parametrize("site", sorted(BODY_READ_SITES))
def test_a_body_read_call_started_with_no_key_is_not_recorded_under_a_key_set_before_the_read(
        site, only_scoped_keys):
    """As with a stream started while metering was off."""
    BODY_READ_SITES[site](only_scoped_keys.switch)

    drain()
    assert records(only_scoped_keys.b) == []


# --- a streamed tool whose stream crosses metering_key() blocks (BACK-4306) ----

class ToolEventRecorder:
    """Stands in for the shared tool-event HTTP client, keeping (key, organization, success) per event."""

    def __init__(self):
        self.events = []

    def post(self, url, headers=None, json=None, timeout=None):
        self.events.append((headers["x-api-key"], json.get("organizationName"), json.get("success")))
        return SimpleNamespace(raise_for_status=lambda: None)


@pytest.fixture
def tool_event_keys(keys, monkeypatch):
    """Tool events resolve their key from the metering clients only: no configure() override hides it."""
    from revenium_middleware._metering import decorator

    monkeypatch.setattr(decorator, "_api_key", None)
    monkeypatch.setattr(decorator, "_metering_url", None)
    recorder = ToolEventRecorder()
    monkeypatch.setattr(tool_event_transport, "_client", recorder)
    return recorder


def search_tool(kind, fail_after_first=False):
    if kind == "sync":
        @revenium_middleware.meter_tool("search-tool")
        def search():
            yield "first"
            if fail_after_first:
                raise RuntimeError("search backend went away")
            yield "second"
    else:
        @revenium_middleware.meter_tool("search-tool")
        async def search():
            yield "first"
            if fail_after_first:
                raise RuntimeError("search backend went away")
            yield "second"
    return search()


def key_block(key):
    return revenium_middleware.metering_key(key) if key else contextlib.nullcontext()


def read_first_item_under_key_b(stream):
    with context(organization_name="customer-B"), revenium_middleware.metering_key(KEY_B):
        return next(stream)


async def read_first_item_under_key_b_async(stream):
    with context(organization_name="customer-B"), revenium_middleware.metering_key(KEY_B):
        return await stream.__anext__()


@pytest.mark.parametrize("kind", ["sync", "async"])
@pytest.mark.parametrize("rest_key", [KEY_C, None], ids=["rest-in-another-block", "rest-after-the-block"])
def test_a_streamed_tool_is_delivered_under_the_key_active_at_its_first_item(kind, rest_key, tool_event_keys):
    stream = search_tool(kind)
    if kind == "sync":
        assert read_first_item_under_key_b(stream) == "first"
        with key_block(rest_key):
            assert list(stream) == ["second"]
    else:
        async def read():
            assert await read_first_item_under_key_b_async(stream) == "first"
            with key_block(rest_key):
                assert [item async for item in stream] == ["second"]

        asyncio.run(read())

    drain()
    assert tool_event_keys.events == [(KEY_B, "customer-B", True)]


def end_sync_stream(stream, how):
    with revenium_middleware.metering_key(KEY_C):
        if how == "exception-mid-stream":
            with pytest.raises(RuntimeError):
                next(stream)
        else:
            stream.close()


def end_async_stream(stream, how):
    async def end():
        await read_first_item_under_key_b_async(stream)
        with revenium_middleware.metering_key(KEY_C):
            if how == "exception-mid-stream":
                with pytest.raises(RuntimeError):
                    await stream.__anext__()
            else:
                await stream.aclose()

    asyncio.run(end())


@pytest.mark.parametrize("kind", ["sync", "async"])
@pytest.mark.parametrize("how, succeeded", [("exception-mid-stream", False), ("closed-early", True)],
                         ids=["exception-mid-stream", "closed-early"])
def test_a_streamed_tool_that_fails_or_is_closed_early_keeps_its_first_item_key(kind, how, succeeded, tool_event_keys):
    stream = search_tool(kind, fail_after_first=how == "exception-mid-stream")
    if kind == "sync":
        read_first_item_under_key_b(stream)
        end_sync_stream(stream, how)
    else:
        end_async_stream(stream, how)

    drain()
    assert tool_event_keys.events == [(KEY_B, "customer-B", succeeded)]


def set_a_key_later(only_scoped_keys, monkeypatch, how):
    if how == "metering-key-block":
        return revenium_middleware.metering_key(KEY_C)
    monkeypatch.setattr(metering, "client", only_scoped_keys.b)
    return contextlib.nullcontext()


@pytest.mark.parametrize("kind", ["sync", "async"])
@pytest.mark.parametrize("how", ["metering-key-block", "process-key-set"])
def test_a_streamed_tool_started_with_no_key_is_not_sent_under_a_key_set_before_it_ends(
        kind, how, tool_event_keys, only_scoped_keys, monkeypatch):
    stream = search_tool(kind)
    if kind == "sync":
        assert next(stream) == "first"
        with set_a_key_later(only_scoped_keys, monkeypatch, how):
            assert list(stream) == ["second"]
    else:
        async def read():
            assert await stream.__anext__() == "first"
            with set_a_key_later(only_scoped_keys, monkeypatch, how):
                assert [item async for item in stream] == ["second"]

        asyncio.run(read())

    drain()
    assert tool_event_keys.events == []


# --- many customer keys share one connection pool (BACK-4377) -------------------

class ReveniumRecorder:
    """Stands in for Revenium at the HTTP transport: keeps (key, organization, cookie) per record sent."""

    def __init__(self):
        self.records = []
        self.set_cookie_for = None

    def handle_request(self, transport, request):
        body = json.loads(request.read() or b"{}")
        api_key = request.headers.get("x-api-key")
        self.records.append((api_key, body.get("organizationName"), request.headers.get("cookie")))
        headers = {"set-cookie": "AWSALB=abc; Path=/"} if api_key == self.set_cookie_for else {}
        return httpx.Response(200, json={}, headers=headers)


class ClientConstructions:
    """Counts every ``httpx.Client`` built, on any thread."""

    def __init__(self, monkeypatch):
        self.count = 0
        original = httpx.Client.__init__

        def counting_init(client, *args, **kwargs):
            self.count += 1
            original(client, *args, **kwargs)

        monkeypatch.setattr(httpx.Client, "__init__", counting_init)

    def reset(self):
        self.count = 0


@pytest.fixture
def real_scoped_clients(monkeypatch):
    """Real ``metering_key()`` clients, with Revenium replaced by a recorder at the HTTP transport."""
    monkeypatch.setattr(metering, "_scoped_clients", OrderedDict())
    monkeypatch.setattr(metering, "_scoped_clients_lock", threading.Lock())
    monkeypatch.setattr(metering, "_scoped_http_client", None, raising=False)
    monkeypatch.setattr(metering, "client", None)
    monkeypatch.delenv("REVENIUM_METERING_API_KEY", raising=False)
    recorder = ReveniumRecorder()
    monkeypatch.setattr(httpx.HTTPTransport, "handle_request",
                        lambda transport, request: recorder.handle_request(transport, request))
    return SimpleNamespace(recorder=recorder, constructions=ClientConstructions(monkeypatch))


def customer_key(index):
    return f"hak_customer_{index:04d}_fake"


def connection_pool_under(key):
    with revenium_middleware.metering_key(key):
        return metering.get_client()._client


def chat_under(key, organization_name):
    call_openai_in_its_own_scope(key, organization_name, lambda: None)


def stream_started_under(key, organization_name):
    from revenium_middleware.openai import middleware as mw
    with revenium_middleware.metering_key(key):
        stream = mw.handle_streaming_response(
            iter(openai_chunks()), NOW, {"organizationName": organization_name}, None, {})
        next(stream)
    return stream


def sent(recorder):
    return sorted((key, organization) for key, organization, _cookie in recorder.records)


def test_entering_metering_key_for_a_new_key_opens_no_new_connection_pool(real_scoped_clients):
    shared = connection_pool_under(KEY_A)
    real_scoped_clients.constructions.reset()

    pools = [connection_pool_under(customer_key(i)) for i in range(50)]

    assert [pool is shared for pool in pools] == [True] * 50
    assert real_scoped_clients.constructions.count == 0


def test_keys_past_the_client_cap_reuse_the_shared_pool_and_never_close_it(real_scoped_clients, monkeypatch):
    monkeypatch.setattr(metering, "MAX_SCOPED_CLIENTS", 4)
    shared = connection_pool_under(KEY_A)
    stream = stream_started_under(KEY_B, "CustomerB")
    real_scoped_clients.constructions.reset()

    for _round in range(3):
        for i in range(10):
            connection_pool_under(customer_key(i))

    assert real_scoped_clients.constructions.count == 0
    assert len(metering._scoped_clients) <= 4
    assert KEY_B not in {key for key, _base_url in metering._scoped_clients}
    list(stream)
    drain()
    assert sent(real_scoped_clients.recorder) == [(KEY_B, "CustomerB")]

    del stream
    gc.collect()
    assert shared.is_closed is False
    chat_under(customer_key(99), "Customer99")
    drain()
    assert sent(real_scoped_clients.recorder) == sorted([(KEY_B, "CustomerB"), (customer_key(99), "Customer99")])


def test_a_stream_finishing_after_a_key_switch_goes_out_with_its_own_key_over_the_shared_pool(real_scoped_clients):
    stream = stream_started_under(KEY_A, "CustomerA")
    with revenium_middleware.metering_key(KEY_B):
        list(stream)
    chat_under(KEY_B, "CustomerB")
    drain()

    assert sent(real_scoped_clients.recorder) == [(KEY_A, "CustomerA"), (KEY_B, "CustomerB")]
    assert connection_pool_under(KEY_A) is connection_pool_under(KEY_B)
    assert real_scoped_clients.constructions.count == 1


def test_closing_one_customer_client_does_not_stop_delivery_for_other_keys(real_scoped_clients):
    with revenium_middleware.metering_key(KEY_A):
        metering.get_client().close()
        with metering.get_client():
            pass

    chat_under(KEY_B, "CustomerB")
    drain()

    assert sent(real_scoped_clients.recorder) == [(KEY_B, "CustomerB")]
    assert connection_pool_under(KEY_A) is connection_pool_under(KEY_B)
    assert connection_pool_under(KEY_B).is_closed is False


@pytest.mark.skipif(not hasattr(os, "fork"), reason="os.fork is unavailable")
def test_a_forked_child_gets_its_own_shared_pool(real_scoped_clients, monkeypatch):
    # macOS reads system proxy settings through CoreFoundation, which crashes
    # in the child of a multi-threaded fork; environment proxies are enough here.
    monkeypatch.setattr(httpx._utils, "getproxies", urllib.request.getproxies_environment)
    parent_pool = connection_pool_under(KEY_A)
    assert connection_pool_under(KEY_B) is parent_pool
    parent_wrappers = {id(wrapper) for wrapper in metering._scoped_clients.values()}

    reader, writer = os.pipe()
    pid = os.fork()
    if pid == 0:
        try:
            os.close(reader)
            child_pool = connection_pool_under(KEY_C)
            report = {
                "shares_parent_pool": child_pool is parent_pool,
                "kept_parent_wrappers": any(id(w) in parent_wrappers for w in metering._scoped_clients.values()),
            }
            os.write(writer, json.dumps(report).encode())
        finally:
            os._exit(0)
    os.close(writer)
    with os.fdopen(reader) as child_output:
        report = json.loads(child_output.read())
    os.waitpid(pid, 0)

    assert report == {"shares_parent_pool": False, "kept_parent_wrappers": False}
    assert parent_pool.is_closed is False


def _report_from_forked_child(monkeypatch, describe_child):
    # macOS reads system proxy settings through CoreFoundation, which crashes
    # in the child of a multi-threaded fork; environment proxies are enough here.
    monkeypatch.setattr(httpx._utils, "getproxies", urllib.request.getproxies_environment)
    reader, writer = os.pipe()
    pid = os.fork()
    if pid == 0:
        try:
            os.close(reader)
            os.write(writer, json.dumps(describe_child()).encode())
        finally:
            os._exit(0)
    os.close(writer)
    with os.fdopen(reader) as child_output:
        report = json.loads(child_output.read())
    os.waitpid(pid, 0)
    return report


@pytest.mark.skipif(not hasattr(os, "fork"), reason="os.fork is unavailable")
def test_a_child_forked_inside_a_metering_key_block_does_not_use_the_parent_pool(real_scoped_clients, monkeypatch):
    with revenium_middleware.metering_key(KEY_A):
        parent_pool = metering.get_client()._client

        def describe_child():
            child_client = metering.get_client()
            return {"shares_parent_pool": child_client._client is parent_pool, "api_key": child_client.api_key}

        report = _report_from_forked_child(monkeypatch, describe_child)

    assert report == {"shares_parent_pool": False, "api_key": KEY_A}
    assert parent_pool.is_closed is False


@pytest.mark.skipif(not hasattr(os, "fork"), reason="os.fork is unavailable")
def test_a_child_forked_inside_recorded_under_does_not_use_the_parent_pool(real_scoped_clients, monkeypatch):
    with revenium_middleware.metering_key(KEY_A):
        pinned = metering.get_client()
    parent_pool = pinned._client

    with metering.recorded_under(pinned):

        def describe_child():
            child_client = metering.current_client()
            return {"shares_parent_pool": child_client._client is parent_pool, "api_key": child_client.api_key}

        report = _report_from_forked_child(monkeypatch, describe_child)

    assert report == {"shares_parent_pool": False, "api_key": KEY_A}
    assert parent_pool.is_closed is False


def test_a_cookie_set_for_one_customer_key_is_not_sent_with_another(real_scoped_clients):
    real_scoped_clients.recorder.set_cookie_for = KEY_A

    chat_under(KEY_A, "CustomerA")
    drain()
    chat_under(KEY_B, "CustomerB")
    drain()

    assert real_scoped_clients.recorder.records == [(KEY_A, "CustomerA", None), (KEY_B, "CustomerB", None)]
