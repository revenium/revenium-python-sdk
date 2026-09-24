"""``async with client.messages.stream(...)`` must produce a metering record.

``AsyncMessages.stream`` was never patched. The manager it returns yields an
``AsyncMessageStream`` of its own and never passes through
``AsyncMessages.create``, so every async streaming Anthropic call -- the common
shape in agentic servers -- was billed by the provider and invisible to
Revenium: no record, no error, no signal.

These tests drive the real, patched Anthropic SDK entry point and replace only
the network call, at the messages resource's ``_post`` seam. The whole
streaming path runs for real (``AsyncMessageStreamManager``,
``AsyncMessageStream``, ``MessageStream``, the SDK's own event accumulation),
so what is asserted is the shape a customer actually gets rather than a
hand-built stand-in.
"""
import asyncio
import os
import threading

import pytest
from freezegun import freeze_time
from unittest.mock import MagicMock, patch

anthropic = pytest.importorskip("anthropic")

from anthropic.types import (  # noqa: E402
    Message,
    RawContentBlockDeltaEvent,
    RawContentBlockStartEvent,
    RawContentBlockStopEvent,
    RawMessageDeltaEvent,
    RawMessageStartEvent,
    RawMessageStopEvent,
    TextBlock,
    TextDelta,
    Usage,
)

import revenium_middleware.anthropic  # noqa: F401,E402  installs the SDK patches
from revenium_middleware._core.patch_registry import is_patched  # noqa: E402
from revenium_middleware.anthropic import middleware as mw  # noqa: E402

MODEL = "claude-3-5-sonnet-20241022"
MESSAGE_ID = "msg_async_stream"
FOUNDRY_RESOURCE = "example-resource"

requires_foundry_client = pytest.mark.skipif(
    not hasattr(anthropic, "AsyncAnthropicFoundry"),
    reason="Installed anthropic SDK predates the Foundry client classes"
)


def raw_events(text_chunks=("Hello ", "world"), output_tokens=5):
    """The raw SSE event sequence the Anthropic API sends for one completion."""
    started = Message(
        id=MESSAGE_ID,
        content=[],
        model=MODEL,
        role="assistant",
        stop_reason=None,
        stop_sequence=None,
        type="message",
        usage=Usage(
            input_tokens=13,
            output_tokens=1,
            cache_creation_input_tokens=4,
            cache_read_input_tokens=7,
        ),
    )

    events = [
        RawMessageStartEvent(type="message_start", message=started),
        RawContentBlockStartEvent(type="content_block_start", index=0,
                                  content_block=TextBlock(type="text", text="")),
    ]
    for chunk in text_chunks:
        events.append(
            RawContentBlockDeltaEvent(type="content_block_delta", index=0,
                                      delta=TextDelta(type="text_delta", text=chunk))
        )
    events += [
        RawContentBlockStopEvent(type="content_block_stop", index=0),
        RawMessageDeltaEvent(type="message_delta",
                             delta={"stop_reason": "end_turn", "stop_sequence": None},
                             usage={"output_tokens": output_tokens}),
        RawMessageStopEvent(type="message_stop"),
    ]
    return events


class FakeRawStream:
    """Stands in for anthropic.AsyncStream: async-iterable and closeable.

    Closing mid-flight makes further iteration raise, the way a closed httpx
    response body does -- that is what makes an abandoned stream unable to
    produce a final message.
    """

    def __init__(self, events, fail_after=None):
        self._events = list(events)
        self._fail_after = fail_after
        self.closed = False

    async def __aiter__(self):
        for index, event in enumerate(self._events):
            if self.closed:
                raise RuntimeError("stream closed before completion")
            if self._fail_after is not None and index == self._fail_after:
                raise RuntimeError("upstream stream failed")
            yield event

    async def close(self):
        self.closed = True


class SyncFakeRawStream:
    """The sync twin of FakeRawStream, standing in for anthropic.Stream."""

    def __init__(self, events, fail_after=None):
        self._events = list(events)
        self._fail_after = fail_after
        self.closed = False

    def __iter__(self):
        for index, event in enumerate(self._events):
            if self.closed:
                raise RuntimeError("stream closed before completion")
            if self._fail_after is not None and index == self._fail_after:
                raise RuntimeError("upstream stream failed")
            yield event

    def close(self):
        self.closed = True


def run_metering_inline(coro_func, *args, **kwargs):
    """Run the metering coroutine to completion on a dedicated thread.

    Mirrors production, where _safe_run_async_in_thread dispatches to a thread,
    while keeping the assertions synchronous.
    """
    thread = threading.Thread(target=lambda: asyncio.run(coro_func(*args, **kwargs)))
    thread.start()
    thread.join(timeout=10)
    return thread


def call_kwargs(with_metadata=True, extra=None):
    kwargs = {
        "model": MODEL,
        "max_tokens": 16,
        "messages": [{"role": "user", "content": "hi"}],
    }
    if with_metadata:
        kwargs["usage_metadata"] = {"trace_id": "trace-async-stream"}
    kwargs.update(extra or {})
    return kwargs


def capture_metering(payloads):
    """Intercept the metering dispatch and run it inline on this thread."""
    return patch.multiple(
        mw,
        submit_ai_event=MagicMock(side_effect=lambda operation, args: payloads.append(args)),
        _safe_run_async_in_thread=MagicMock(side_effect=run_metering_inline),
        _get_thread_safe_client=MagicMock(return_value=MagicMock()),
    )


def install_network_seam(client, raw_stream):
    """Replace the one call that would reach Anthropic, and return the resource.

    Patched on the resource rather than the client because the resource binds
    client.post once, at construction (AsyncAnthropicBedrock builds its
    messages resource eagerly in __init__). Everything above this seam is the
    SDK's real streaming code.
    """
    messages = client.messages
    if isinstance(raw_stream, FakeRawStream):
        async def fake_post(*args, **kwargs):
            return raw_stream
    else:
        def fake_post(*args, **kwargs):
            return raw_stream

    messages._post = fake_post
    return messages


def stream_payloads(consume, client=None, events=None, fail_after=None,
                    expect_error=None, request_kwargs=None, with_metadata=True):
    """Drive one async streaming scenario and return the metering payloads.

    ``consume`` stands in for the customer's code inside the ``async with``
    block; it is awaited with the object the SDK handed back.
    """
    client = client if client is not None else anthropic.AsyncAnthropic(api_key="test-key")
    messages = install_network_seam(
        client,
        FakeRawStream(raw_events() if events is None else events, fail_after=fail_after),
    )
    kwargs = call_kwargs(with_metadata, request_kwargs)

    async def scenario():
        async with messages.stream(**kwargs) as stream:
            await consume(stream)

    payloads = []
    with capture_metering(payloads):
        if expect_error is None:
            asyncio.run(scenario())
        else:
            with pytest.raises(expect_error):
                asyncio.run(scenario())

    return payloads


def sync_stream_payloads(consume, client=None, events=None, fail_after=None,
                         expect_error=None, with_metadata=True, request_kwargs=None):
    """The sync twin of stream_payloads, for the parity assertions."""
    client = client if client is not None else anthropic.Anthropic(api_key="test-key")
    messages = install_network_seam(
        client,
        SyncFakeRawStream(raw_events() if events is None else events, fail_after=fail_after),
    )
    kwargs = call_kwargs(with_metadata, request_kwargs)

    def scenario():
        with messages.stream(**kwargs) as stream:
            consume(stream)

    payloads = []
    with capture_metering(payloads):
        if expect_error is None:
            scenario()
        else:
            with pytest.raises(expect_error):
                scenario()

    return payloads


async def consume_events(stream):
    async for _event in stream:
        pass


async def consume_text(stream):
    return "".join([chunk async for chunk in stream.text_stream])


async def consume_nothing(stream):
    return None


def test_async_messages_stream_is_patched():
    """The registration itself: the surface used to be missing entirely."""
    assert is_patched("anthropic.resources.messages.messages.AsyncMessages.stream")


def test_the_sdk_hands_back_the_metering_wrapper():
    captured = {}

    async def capture(stream):
        captured["stream"] = stream

    stream_payloads(capture)

    assert isinstance(captured["stream"], mw._AsyncMessageStreamMetering)


def test_consuming_the_events_meters_the_completion():
    payloads = stream_payloads(consume_events)

    assert len(payloads) == 1
    payload = payloads[0]
    assert payload["provider"] == "ANTHROPIC"
    assert payload["model_source"] == "ANTHROPIC"
    assert payload["model"] == MODEL
    assert payload["is_streamed"] is True
    assert payload["transaction_id"] == MESSAGE_ID
    assert payload["input_token_count"] == 13
    assert payload["output_token_count"] == 5
    assert payload["total_token_count"] == 18
    assert payload["stop_reason"] == "END"
    assert payload["trace_id"] == "trace-async-stream"


def test_iterating_the_text_stream_meters_the_completion():
    text = {}

    async def consume(stream):
        # The delay makes the sub-millisecond first token measurable: the
        # payload carries whole milliseconds, so an uninstrumented text_stream
        # would report 0 here rather than the real wait.
        await asyncio.sleep(0.01)
        text["value"] = await consume_text(stream)

    payloads = stream_payloads(consume)

    assert text["value"] == "Hello world"
    assert len(payloads) == 1
    assert payloads[0]["output_token_count"] == 5
    assert payloads[0]["time_to_first_token"] >= 10


def test_awaiting_the_final_message_meters_the_completion():
    final = {}

    async def consume(stream):
        final["message"] = await stream.get_final_message()

    payloads = stream_payloads(consume)

    assert final["message"].id == MESSAGE_ID
    assert len(payloads) == 1
    assert payloads[0]["input_token_count"] == 13


def test_consuming_and_then_awaiting_the_final_message_meters_once():
    """Both idioms in one block must not bill the customer twice."""
    async def consume(stream):
        await consume_events(stream)
        await stream.get_final_message()

    assert len(stream_payloads(consume)) == 1


def test_a_stream_with_no_revenium_metadata_is_still_metered():
    """The plainest async stream there is -- the call that used to vanish.

    Nothing Revenium-specific is passed, so before the fix the block ran to
    completion and no metering record was created at all: no error, no signal,
    just spend the platform never saw.
    """
    payloads = stream_payloads(consume_events, with_metadata=False)

    assert len(payloads) == 1
    assert payloads[0]["provider"] == "ANTHROPIC"
    assert payloads[0]["input_token_count"] == 13
    assert payloads[0]["output_token_count"] == 5


def test_cache_token_counts_are_forwarded():
    payload = stream_payloads(consume_events)[0]

    assert payload["cache_creation_token_count"] == 4
    assert payload["cache_read_token_count"] == 7


def test_leaving_the_block_without_reading_anything_reports_nothing():
    """Nothing was read off the wire, so there is nothing to report.

    Not even ``message_start`` arrived, and the response is closed by the time
    finalisation runs, so draining it here would do I/O the caller declined.
    A stream the caller did read from is a different case -- see below.
    """
    assert stream_payloads(consume_nothing) == []


def test_breaking_out_of_the_events_reports_the_partial_usage():
    """A stream broken off part-way still spent the tokens it delivered.

    Finalisation used to ask for the final message, which a closed stream
    cannot produce, and swallowed the error -- so a partially consumed,
    already-billed request produced no metering event at all (Greptile P1).
    """
    async def consume(stream):
        async for _event in stream:
            break

    payloads = stream_payloads(consume)

    assert len(payloads) == 1
    assert payloads[0]["transaction_id"] == MESSAGE_ID
    # message_start's counts: the input in full, the output so far.
    assert payloads[0]["input_token_count"] == 13
    assert payloads[0]["output_token_count"] == 1
    assert payloads[0]["is_streamed"] is True


def test_breaking_out_of_the_text_stream_reports_the_partial_usage():
    async def consume(stream):
        async for _chunk in stream.text_stream:
            break

    payloads = stream_payloads(consume)

    assert len(payloads) == 1
    assert payloads[0]["input_token_count"] == 13
    assert payloads[0]["output_token_count"] == 1


def test_sync_breaking_out_of_the_events_reports_the_partial_usage():
    """Parity: the sync wrapper had the same hole and gets the same fallback.

    This changes sync behaviour -- an interrupted ``with
    client.messages.stream(...)`` block used to report nothing.
    """
    def consume(stream):
        for _event in stream:
            break

    payloads = sync_stream_payloads(consume)

    assert len(payloads) == 1
    assert payloads[0]["input_token_count"] == 13
    assert payloads[0]["output_token_count"] == 1


def test_sync_full_consumption_still_meters_the_whole_completion():
    """The shared finalisation must not have changed the sync happy path."""
    def consume(stream):
        for _event in stream:
            pass

    payloads = sync_stream_payloads(consume)

    assert len(payloads) == 1
    assert payloads[0]["provider"] == "ANTHROPIC"
    assert payloads[0]["input_token_count"] == 13
    assert payloads[0]["output_token_count"] == 5
    assert payloads[0]["is_streamed"] is True


def test_an_exception_mid_stream_propagates_and_reports_the_partial_usage():
    """The error reaches the caller and the tokens already spent are still billed.

    The stream dies after ``message_start``, so the snapshot the SDK kept is
    what gets metered: input tokens in full and the output tokens counted so
    far. Same outcome as the sync wrapper on the same failure.
    """
    payloads = stream_payloads(consume_events, fail_after=2, expect_error=RuntimeError)

    assert len(payloads) == 1
    assert payloads[0]["input_token_count"] == 13
    assert payloads[0]["output_token_count"] == 1


def test_unknown_attributes_are_forwarded_to_the_sdk_stream():
    """The wrapper must not amputate the stream's own surface."""
    snapshot = {}

    async def consume(stream):
        await consume_events(stream)
        snapshot["id"] = stream.current_message_snapshot.id

    stream_payloads(consume)

    assert snapshot["id"] == MESSAGE_ID


def test_selective_metering_leaves_the_sdk_manager_untouched(monkeypatch):
    monkeypatch.setenv("REVENIUM_SELECTIVE_METERING", "true")
    captured = {}

    async def capture(stream):
        captured["stream"] = stream

    payloads = stream_payloads(capture)

    assert payloads == []
    assert isinstance(captured["stream"], anthropic.lib.streaming.AsyncMessageStream)


def test_captured_prompts_include_the_streamed_output(monkeypatch):
    monkeypatch.setenv("REVENIUM_CAPTURE_PROMPTS", "true")

    async def consume(stream):
        await consume_text(stream)

    payload = stream_payloads(consume)[0]

    assert "Hello world" in payload["output_response"]
    assert payload["input_messages"] is not None


def test_captured_prompts_survive_the_get_final_message_idiom(monkeypatch):
    """Capture must not depend on which consumption style the caller picked.

    ``get_final_message()`` reads the underlying SDK stream directly and never
    passes through the wrapper's iterators, so nothing accumulates there; the
    text is taken off the finalised message instead.
    """
    monkeypatch.setenv("REVENIUM_CAPTURE_PROMPTS", "true")

    async def consume(stream):
        await stream.get_final_message()

    payload = stream_payloads(consume)[0]

    assert "Hello world" in payload["output_response"]


def test_sync_captured_prompts_survive_the_get_final_message_idiom(monkeypatch):
    monkeypatch.setenv("REVENIUM_CAPTURE_PROMPTS", "true")

    def consume(stream):
        stream.get_final_message()

    payload = sync_stream_payloads(consume)[0]

    assert "Hello world" in payload["output_response"]


def test_partial_iteration_then_get_final_message_captures_the_whole_text():
    """Mixed consumption must not truncate the captured output.

    Reading a couple of chunks and then letting `get_final_message()` drain
    the rest is a legitimate idiom, and the rest of the stream is drained on
    the SDK stream directly -- so a chunk-by-chunk accumulator stops growing
    exactly where the manual loop stopped, and used to report "Hello " as the
    complete response.
    """
    async def consume(stream):
        async for chunk in stream.text_stream:
            assert chunk == "Hello "
            break
        await stream.get_final_message()

    with patch.dict(os.environ, {"REVENIUM_CAPTURE_PROMPTS": "true"}):
        payload = stream_payloads(consume)[0]

    assert payload["output_response"] == "Hello world"
    assert payload["output_token_count"] == 5


def test_sync_partial_iteration_then_get_final_message_captures_the_whole_text():
    def consume(stream):
        for chunk in stream.text_stream:
            assert chunk == "Hello "
            break
        stream.get_final_message()

    with patch.dict(os.environ, {"REVENIUM_CAPTURE_PROMPTS": "true"}):
        payload = sync_stream_payloads(consume)[0]

    assert payload["output_response"] == "Hello world"
    assert payload["output_token_count"] == 5


def test_completion_start_time_is_the_first_token_not_the_stream_end():
    """The completion started when the first token landed, not at the end.

    Reporting the finish time as the start understates first-token latency on
    every streamed completion. Mirrors the same fix on the raw-stream path.
    """
    with freeze_time("2023-11-14T22:13:20Z") as frozen:
        async def consume(stream):
            frozen.tick(42)
            await consume_text(stream)
            frozen.tick(8)

        payload = stream_payloads(consume)[0]

    assert payload["completion_start_time"] == "2023-11-14T22:14:02Z"
    assert payload["response_time"] == "2023-11-14T22:14:10Z"
    assert payload["time_to_first_token"] == 42000
    assert payload["request_duration"] == 50000


def test_sync_completion_start_time_is_the_first_token_not_the_stream_end():
    with freeze_time("2023-11-14T22:13:20Z") as frozen:
        def consume(stream):
            frozen.tick(42)
            for _chunk in stream.text_stream:
                pass
            frozen.tick(8)

        payload = sync_stream_payloads(consume)[0]

    assert payload["completion_start_time"] == "2023-11-14T22:14:02Z"
    assert payload["time_to_first_token"] == 42000


def test_completion_start_time_falls_back_to_the_end_when_no_token_was_seen():
    """No text was read, so there is no first-token moment to report."""
    async def consume(stream):
        await stream.get_final_message()

    payload = stream_payloads(consume)[0]

    assert payload["completion_start_time"] == payload["response_time"]
    assert payload["time_to_first_token"] == 0


@requires_foundry_client
def test_foundry_async_stream_carries_the_foundry_label(monkeypatch):
    monkeypatch.delenv("REVENIUM_BEDROCK_DISABLE", raising=False)
    client = anthropic.AsyncAnthropicFoundry(api_key="test-key", resource=FOUNDRY_RESOURCE)

    payload = stream_payloads(consume_events, client=client)[0]

    assert payload["provider"] == "Foundry"
    assert payload["model_source"] == "ANTHROPIC"
    assert payload["output_token_count"] == 5


@requires_foundry_client
def test_foundry_label_survives_bedrock_being_disabled(monkeypatch):
    monkeypatch.setenv("REVENIUM_BEDROCK_DISABLE", "1")
    client = anthropic.AsyncAnthropicFoundry(api_key="test-key", resource=FOUNDRY_RESOURCE)

    payload = stream_payloads(consume_events, client=client)[0]

    assert payload["provider"] == "Foundry"


def test_bedrock_async_stream_is_attributed_to_aws_without_rerouting(monkeypatch):
    """An async Bedrock stream stays on the SDK transport and is labelled AWS.

    The sync wrapper's Bedrock fast path is a blocking boto3 call and cannot
    serve an async context manager, so this path must not hand the request to
    the Bedrock adapter -- exactly the choice ``AsyncMessages.create`` makes.
    """
    monkeypatch.delenv("REVENIUM_BEDROCK_DISABLE", raising=False)
    client = anthropic.AsyncAnthropicBedrock(aws_region="us-east-1",
                                             aws_access_key="test-access-key",
                                             aws_secret_key="test-secret-key")

    with patch.object(mw, "_handle_bedrock_stream_request") as bedrock_handler:
        payload = stream_payloads(consume_events, client=client)[0]

    bedrock_handler.assert_not_called()
    assert payload["provider"] == "AWS"
    assert payload["model_source"] == "ANTHROPIC"


AGENT_VERSION_METADATA = {"trace_id": "trace-async-stream", "agent_version": "1.4.2"}


def test_async_stream_forwards_the_agent_version():
    """Both stream wrappers meter through the shared helper, which was written
    before ``agent_version`` existed and so had no forwarding of its own.

    Every other Anthropic metering call site carries the field; a stream that
    dropped it would attribute the spend to no release at all, and silently --
    the event still delivers.
    """
    payload = stream_payloads(
        consume_events,
        request_kwargs={"usage_metadata": dict(AGENT_VERSION_METADATA)},
    )[0]

    assert payload["agent_version"] == "1.4.2"


def test_sync_stream_forwards_the_agent_version():
    payload = sync_stream_payloads(
        lambda stream: [None for _event in stream],
        request_kwargs={"usage_metadata": dict(AGENT_VERSION_METADATA)},
    )[0]

    assert payload["agent_version"] == "1.4.2"


def test_stream_without_an_agent_version_sends_the_key_as_none():
    """Parity with the create wrappers: the key is always present on the
    payload, carrying None when the caller set no version."""
    payload = stream_payloads(consume_events)[0]

    assert payload["agent_version"] is None
