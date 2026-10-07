"""Every OpenAI chat and Responses entry point, driven through the real
``openai`` client over ``httpx.MockTransport``, meters exactly once and leaves
what the caller sees unchanged.

The baseline for "unchanged" is the same call with the wrappers passing
through untouched (selective metering on, outside any decorated function).
"""
import asyncio
import json
import logging
import threading
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import patch

import httpx
import pytest
import wrapt
from openai import NOT_GIVEN, AsyncOpenAI, OpenAI, Omit
from openai.resources.responses import AsyncResponses, Responses

from revenium_middleware.openai import middleware as mw

from .responses_stream_events import MODEL, event_payloads, response_payload, usage_payload

RESPONSES_USAGE = usage_payload(input_tokens=11, output_tokens=7, cached_tokens=3, reasoning_tokens=2)
CHAT_USAGE = {
    "prompt_tokens": 11,
    "completion_tokens": 7,
    "total_tokens": 18,
    "prompt_tokens_details": {"cached_tokens": 3},
    "completion_tokens_details": {"reasoning_tokens": 2},
}
CHAT_REQUEST = {"model": MODEL, "messages": [{"role": "user", "content": "hi"}]}
RESPONSES_REQUEST = {"model": MODEL, "input": "hi"}
MISSING_USAGE_WARNING = "No usage data found in streaming Responses API response!"
METERED_FIELDS = (
    "model", "input_token_count", "output_token_count", "total_token_count",
    "cache_read_token_count", "reasoning_token_count",
)


def _sse(events, done_marker):
    lines = []
    for event in events:
        if "type" in event:
            lines.append(f"event: {event['type']}\n")
        lines.append(f"data: {json.dumps(event)}\n\n")
    if done_marker:
        lines.append("data: [DONE]\n\n")
    return "".join(lines).encode()


def _chat_chunks(include_usage):
    base = {"id": "chatcmpl-1", "object": "chat.completion.chunk", "created": 1, "model": MODEL}
    chunks = [
        {**base, "choices": [{"index": 0, "delta": {"role": "assistant", "content": "hi"},
                              "finish_reason": None}]},
        {**base, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
    ]
    if include_usage:
        chunks.append({**base, "choices": [], "usage": CHAT_USAGE})
    return chunks


class FakeOpenAI:
    """Serves chat and Responses calls and records what each request asked for."""

    def __init__(self, responses_terminal="response.completed"):
        self.requests = []
        self._responses_terminal = responses_terminal

    def __call__(self, request):
        body = json.loads(request.content or b"{}")
        self.requests.append(body)
        sse = {"content-type": "text/event-stream"}
        if request.url.path.endswith("/chat/completions"):
            if body.get("stream"):
                include_usage = bool((body.get("stream_options") or {}).get("include_usage"))
                return httpx.Response(200, headers=sse, content=_sse(_chat_chunks(include_usage), True))
            return httpx.Response(200, json={
                "id": "chatcmpl-1", "object": "chat.completion", "created": 1, "model": MODEL,
                "usage": CHAT_USAGE,
                "choices": [{"index": 0, "finish_reason": "stop",
                             "message": {"role": "assistant", "content": "hi"}}],
            })
        if request.url.path.endswith("/responses"):
            if body.get("stream"):
                events = event_payloads(self._responses_terminal, RESPONSES_USAGE)
                return httpx.Response(200, headers=sse, content=_sse(events, False))
            return httpx.Response(200, json=response_payload(usage=RESPONSES_USAGE))
        return httpx.Response(404, json={"error": "unrouted"})

    def sync_client(self):
        return OpenAI(api_key="sk-test", base_url="https://api.openai.test/v1",
                      http_client=httpx.Client(transport=httpx.MockTransport(self)))

    def async_client(self):
        async def handler(request):
            await request.aread()
            return self(request)
        return AsyncOpenAI(api_key="sk-test", base_url="https://api.openai.test/v1",
                           http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))


def _run_coro_to_completion(coro):
    errors = []

    def target():
        try:
            asyncio.run(coro)
        except Exception as exc:  # surfaced to the test below
            errors.append(exc)

    thread = threading.Thread(target=target)
    thread.start()
    thread.join()
    if errors:
        raise errors[0]
    return thread


@pytest.fixture
def payloads():
    recorded = []

    def record(kind, args):
        recorded.append(args)
        return SimpleNamespace(id="evt-test")

    with patch.object(mw, "get_client", lambda: object()), \
         patch.object(mw, "submit_ai_event", side_effect=record), \
         patch.object(mw, "run_async_in_thread", side_effect=_run_coro_to_completion):
        yield recorded


# ---- calls: each returns what the caller saw ------------------------------

def _chat_kwargs(stream_options):
    return CHAT_REQUEST if stream_options is None else {**CHAT_REQUEST, "stream_options": stream_options}


def _responses_kwargs(stream_options):
    return RESPONSES_REQUEST if stream_options is None else {**RESPONSES_REQUEST, "stream_options": stream_options}


def sync_chat_create(fake, _options):
    return [fake.sync_client().chat.completions.create(**CHAT_REQUEST)]


def sync_chat_create_stream(fake, options):
    return list(fake.sync_client().chat.completions.create(**_chat_kwargs(options), stream=True))


def sync_chat_stream(fake, options):
    with fake.sync_client().chat.completions.stream(**_chat_kwargs(options)) as stream:
        return [event.type for event in stream]


def async_chat_create(fake, _options):
    async def go():
        return [await fake.async_client().chat.completions.create(**CHAT_REQUEST)]
    return asyncio.run(go())


def async_chat_create_stream(fake, options):
    async def go():
        stream = await fake.async_client().chat.completions.create(**_chat_kwargs(options), stream=True)
        return [chunk async for chunk in stream]
    return asyncio.run(go())


def async_chat_stream(fake, options):
    async def go():
        async with fake.async_client().chat.completions.stream(**_chat_kwargs(options)) as stream:
            return [event.type async for event in stream]
    return asyncio.run(go())


def sync_responses_create(fake, _options):
    return [fake.sync_client().responses.create(**RESPONSES_REQUEST)]


def sync_responses_create_stream(fake, options):
    return [event.type for event in
            fake.sync_client().responses.create(**_responses_kwargs(options), stream=True)]


def sync_responses_stream(fake, options):
    with fake.sync_client().responses.stream(**_responses_kwargs(options)) as stream:
        events = [event.type for event in stream]
        assert stream.get_final_response().usage.total_tokens == RESPONSES_USAGE["total_tokens"]
        return events


def async_responses_create(fake, _options):
    async def go():
        return [await fake.async_client().responses.create(**RESPONSES_REQUEST)]
    return asyncio.run(go())


def async_responses_create_stream(fake, options):
    async def go():
        stream = await fake.async_client().responses.create(**_responses_kwargs(options), stream=True)
        return [event.type async for event in stream]
    return asyncio.run(go())


def async_responses_stream(fake, options):
    async def go():
        async with fake.async_client().responses.stream(**_responses_kwargs(options)) as stream:
            events = [event.type async for event in stream]
            final = await stream.get_final_response()
            assert final.usage.total_tokens == RESPONSES_USAGE["total_tokens"]
            return events
    return asyncio.run(go())


@dataclass(frozen=True)
class Case:
    call: object
    options: object = None

    def __str__(self):
        suffix = "" if self.options is None else "-caller_stream_options"
        return f"{self.call.__name__}{suffix}"


CHAT_OPTIONS = {"include_usage": True}
RESPONSES_OPTIONS = {"include_obfuscation": False}
NON_STREAMED = [sync_chat_create, async_chat_create, sync_responses_create, async_responses_create]
STREAMED_CHAT = [sync_chat_create_stream, sync_chat_stream, async_chat_create_stream, async_chat_stream]
STREAMED_RESPONSES = [sync_responses_create_stream, sync_responses_stream,
                      async_responses_create_stream, async_responses_stream]

# stream_options only means something on a streamed call, so the non-streamed
# calls run once, without it.
CASES = (
    [Case(call) for call in NON_STREAMED]
    + [Case(call, options) for call in STREAMED_CHAT for options in (None, CHAT_OPTIONS)]
    + [Case(call, options) for call in STREAMED_RESPONSES for options in (None, RESPONSES_OPTIONS)]
)


def _caller_view(items):
    return [item.model_dump() if hasattr(item, "model_dump") else item for item in items]


@pytest.mark.parametrize("case", CASES, ids=str)
def test_entry_point_meters_once_and_caller_view_matches_baseline(case, payloads, caplog):
    with patch.object(mw, "is_selective_metering_enabled", return_value=True):
        baseline = case.call(FakeOpenAI(), case.options)
    assert payloads == []

    with caplog.at_level(logging.WARNING, logger="revenium_middleware.extension"):
        metered = case.call(FakeOpenAI(), case.options)

    assert len(payloads) == 1
    assert _caller_view(metered) == _caller_view(baseline)
    assert MISSING_USAGE_WARNING not in caplog.text


@pytest.mark.parametrize("call", STREAMED_CHAT, ids=lambda c: c.__name__)
@pytest.mark.parametrize("options", [None, CHAT_OPTIONS, {"include_obfuscation": False}],
                         ids=["no_options", "include_usage", "other_options"])
def test_chat_stream_requests_usage_and_keeps_caller_options(call, options, payloads):
    fake = FakeOpenAI()

    call(fake, options)

    sent = fake.requests[0]["stream_options"]
    assert sent["include_usage"] is True
    for key, value in (options or {}).items():
        assert sent[key] == value
    assert payloads[0]["total_token_count"] == CHAT_USAGE["total_tokens"]


@pytest.mark.parametrize("call", [sync_chat_create_stream, async_chat_create_stream],
                         ids=lambda c: c.__name__)
def test_injected_usage_chunk_is_visible_only_when_requested(call, payloads):
    hidden = call(FakeOpenAI(), None)
    requested = call(FakeOpenAI(), CHAT_OPTIONS)

    assert not any(chunk.usage for chunk in hidden)
    assert [chunk.usage.total_tokens for chunk in requested if chunk.usage] == [CHAT_USAGE["total_tokens"]]


def _metered_values(payload):
    return {field: payload[field] for field in METERED_FIELDS}


@pytest.mark.parametrize("call", STREAMED_RESPONSES + [async_responses_create], ids=lambda c: c.__name__)
def test_responses_event_matches_the_non_streamed_call(call, payloads):
    sync_responses_create(FakeOpenAI(), None)
    call(FakeOpenAI(), None)

    non_streamed, other = payloads
    assert _metered_values(other) == _metered_values(non_streamed)
    assert other["model"] == MODEL
    assert other["cache_read_token_count"] == 3
    assert other["reasoning_token_count"] == 2
    assert other["is_streamed"] is (call is not async_responses_create)


@pytest.mark.parametrize("terminal", ["response.incomplete", "response.failed"])
@pytest.mark.parametrize("call", [sync_responses_create_stream, async_responses_create_stream],
                         ids=lambda c: c.__name__)
def test_responses_stream_meters_usage_from_non_completed_terminal_event(call, terminal, payloads):
    call(FakeOpenAI(responses_terminal=terminal), None)

    assert len(payloads) == 1
    assert payloads[0]["total_token_count"] == RESPONSES_USAGE["total_tokens"]
    assert payloads[0]["model"] == MODEL


@pytest.mark.parametrize("placeholder", [Omit(), NOT_GIVEN, None], ids=["omit", "not_given", "none"])
def test_placeholder_stream_options_count_as_not_supplied(placeholder):
    forced, caller_requested_usage = mw._with_forced_usage_reporting(
        {"stream": True, "stream_options": placeholder})

    assert forced["stream_options"] == {"include_usage": True}
    assert caller_requested_usage is False


def test_forcing_usage_does_not_mutate_the_caller_options():
    caller_options = {"include_obfuscation": False}

    forced, caller_requested_usage = mw._with_forced_usage_reporting({"stream_options": caller_options})

    assert forced["stream_options"] == {"include_obfuscation": False, "include_usage": True}
    assert caller_options == {"include_obfuscation": False}
    assert caller_requested_usage is False


def test_async_responses_create_is_wrapped():
    assert isinstance(AsyncResponses.__dict__["create"], wrapt.FunctionWrapper)
    assert isinstance(Responses.__dict__["create"], wrapt.FunctionWrapper)
