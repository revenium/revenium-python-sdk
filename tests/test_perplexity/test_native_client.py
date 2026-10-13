"""The native Perplexity client is metered exactly once per call (BACK-3607)."""
import asyncio
import copy
import itertools
import json
import re
import uuid
from unittest.mock import MagicMock, patch

import httpx
import perplexity
import pytest
from freezegun import freeze_time

import revenium_middleware.perplexity  # noqa: F401
from revenium_middleware import idempotency_key
from entry_points import calls

MESSAGES = [{"role": "user", "content": "hi"}]


@pytest.fixture
def recorder():
    recording_client = MagicMock()
    with patch("revenium_middleware._core.metering.client", recording_client):
        yield recording_client


def _payloads(recording_client):
    calls.wait_for_metering()
    return calls.recorded_payloads(recording_client)


def _assert_one_perplexity_payload(recording_client):
    payloads = _payloads(recording_client)
    assert len(payloads) == 1, payloads
    payload = payloads[0]
    assert payload["provider"] == "PERPLEXITY"
    assert payload["model"] == calls.PERPLEXITY_MODEL
    assert (payload["input_token_count"], payload["output_token_count"]) == (calls.INPUT_TOKENS, calls.OUTPUT_TOKENS)
    return recording_client.ai.create_completion.call_args.kwargs


class _RecordingHandler:
    def __init__(self):
        self.bodies = []
        self._respond = calls._perplexity_handler()

    def __call__(self, request):
        self.bodies.append(json.loads(request.content or b"{}"))
        return self._respond(request)


def _client(handler=None):
    transport = httpx.MockTransport(handler or calls._perplexity_handler())
    return perplexity.Perplexity(api_key="stub", http_client=httpx.Client(transport=transport))


def _async_client():
    transport = httpx.MockTransport(calls._perplexity_handler())
    return perplexity.AsyncPerplexity(api_key="stub", http_client=httpx.AsyncClient(transport=transport))


def test_sync_create_is_metered_once(recorder):
    _client().chat.completions.create(model=calls.PERPLEXITY_MODEL, messages=MESSAGES)

    sent = _assert_one_perplexity_payload(recorder)
    assert sent["is_streamed"] is False


def test_async_create_is_metered_once(recorder):
    asyncio.run(_async_client().chat.completions.create(model=calls.PERPLEXITY_MODEL, messages=MESSAGES))

    _assert_one_perplexity_payload(recorder)


def test_usage_metadata_is_taken_out_of_extra_body_before_the_request(recorder):
    handler = _RecordingHandler()

    _client(handler).chat.completions.create(model=calls.PERPLEXITY_MODEL, messages=MESSAGES,
                                             extra_body={"usage_metadata": {"trace_id": "t-1"}})

    assert "usage_metadata" not in handler.bodies[0]
    assert _assert_one_perplexity_payload(recorder)["trace_id"] == "t-1"


def test_a_stream_used_as_a_context_manager_is_metered_once_from_its_usage_chunk(recorder):
    create = _client().chat.completions.create
    with create(model=calls.PERPLEXITY_MODEL, messages=MESSAGES, stream=True) as stream:
        chunks = list(stream)

    assert len(chunks) == 3
    sent = _assert_one_perplexity_payload(recorder)
    assert sent["is_streamed"] is True
    assert sent["stop_reason"] == "END"


def test_a_stream_closed_early_is_metered_once(recorder):
    stream = _client().chat.completions.create(model=calls.PERPLEXITY_MODEL, messages=MESSAGES, stream=True)
    next(stream)
    stream.close()
    stream.close()

    payloads = _payloads(recorder)
    assert len(payloads) == 1, payloads


def test_the_wrapped_stream_still_exposes_the_http_response(recorder):
    stream = _client().chat.completions.create(model=calls.PERPLEXITY_MODEL, messages=MESSAGES, stream=True)

    assert isinstance(stream.response, httpx.Response)
    stream.close()


def test_an_async_stream_used_as_a_context_manager_is_metered_once(recorder):
    async def consume():
        create = _async_client().chat.completions.create
        stream = await create(model=calls.PERPLEXITY_MODEL, messages=MESSAGES, stream=True)
        async with stream:
            return [chunk async for chunk in stream]

    assert len(asyncio.run(consume())) == 3
    assert _assert_one_perplexity_payload(recorder)["is_streamed"] is True


def test_a_stream_that_yields_nothing_is_not_metered(recorder):
    stream = _client().chat.completions.create(model=calls.PERPLEXITY_MODEL, messages=MESSAGES, stream=True)
    stream.close()

    assert _payloads(recorder) == []


def _async_recording_client(handler):
    transport = httpx.MockTransport(handler)
    return perplexity.AsyncPerplexity(api_key="stub", http_client=httpx.AsyncClient(transport=transport))


def _create_sync(handler, **kwargs):
    _client(handler).chat.completions.create(model=calls.PERPLEXITY_MODEL, messages=MESSAGES, **kwargs)


def _create_async(handler, **kwargs):
    create = _async_recording_client(handler).chat.completions.create
    asyncio.run(create(model=calls.PERPLEXITY_MODEL, messages=MESSAGES, **kwargs))


def _create_stream(handler, **kwargs):
    create = _client(handler).chat.completions.create
    with create(model=calls.PERPLEXITY_MODEL, messages=MESSAGES, stream=True, **kwargs) as stream:
        list(stream)


_METADATA = {"organizationName": "AcmeCorp", "subscriber": {"email": "dev@acme.example"}}


@pytest.mark.parametrize("create", [_create_sync, _create_async, _create_stream], ids=["sync", "async", "stream"])
@pytest.mark.parametrize("metadata_kwargs", [
    pytest.param({"usage_metadata": _METADATA}, id="top_level"),
    pytest.param({"extra_body": {"usage_metadata": _METADATA}}, id="extra_body"),
])
def test_selective_pass_through_strips_usage_metadata_from_the_request(create, metadata_kwargs, recorder, monkeypatch):
    """Selective metering on, outside @revenium_meter: not metered, and the
    caller's Revenium metadata never reaches Perplexity (BACK-3991)."""
    monkeypatch.setenv("REVENIUM_SELECTIVE_METERING", "true")
    handler = _RecordingHandler()

    create(handler, **copy.deepcopy(metadata_kwargs))

    assert len(handler.bodies) == 1
    assert "usage_metadata" not in handler.bodies[0]
    assert _payloads(recorder) == []


def test_top_level_usage_metadata_is_taken_out_before_the_request(recorder):
    """The README passes ``usage_metadata=`` straight to ``create``; the
    native client accepts it as the OpenAI client does (BACK-4041)."""
    handler = _RecordingHandler()

    _create_sync(handler, usage_metadata={"trace_id": "t-1"})

    assert "usage_metadata" not in handler.bodies[0]
    assert _assert_one_perplexity_payload(recorder)["trace_id"] == "t-1"


def test_async_create_accepts_top_level_usage_metadata(recorder):
    handler = _RecordingHandler()

    _create_async(handler, usage_metadata={"trace_id": "t-async"})

    assert "usage_metadata" not in handler.bodies[0]
    assert _assert_one_perplexity_payload(recorder)["trace_id"] == "t-async"


def test_a_stream_with_top_level_usage_metadata_is_metered_once_and_attributed(recorder):
    handler = _RecordingHandler()

    _create_stream(handler, usage_metadata={"trace_id": "t-stream", "task_type": "creative_writing"})

    assert "usage_metadata" not in handler.bodies[0]
    sent = _assert_one_perplexity_payload(recorder)
    assert sent["is_streamed"] is True
    assert (sent["trace_id"], sent["task_type"]) == ("t-stream", "creative_writing")


def test_top_level_usage_metadata_wins_over_extra_body(recorder):
    handler = _RecordingHandler()

    _create_sync(handler, usage_metadata={"trace_id": "from-keyword"},
                 extra_body={"usage_metadata": {"trace_id": "from-extra-body", "task_type": "kept"}})

    assert "usage_metadata" not in handler.bodies[0]
    sent = _assert_one_perplexity_payload(recorder)
    assert (sent["trace_id"], sent["task_type"]) == ("from-keyword", "kept")


MINTED_UUIDS = 1000


@pytest.fixture
def sequential_uuids(monkeypatch):
    counter = itertools.count(1)
    monkeypatch.setattr(uuid, "uuid4", lambda: uuid.UUID(int=next(counter)))


def _transaction_ids(recording_client):
    calls.wait_for_metering()
    return [entry.kwargs["transaction_id"] for entry in recording_client.ai.create_completion.call_args_list]


def test_record_uses_the_perplexity_response_id(recorder):
    _client().chat.completions.create(model=calls.PERPLEXITY_MODEL, messages=MESSAGES)

    assert _transaction_ids(recorder) == ["chatcmpl-1"]


def test_same_instant_streams_get_distinct_transaction_ids(recorder, sequential_uuids):
    with freeze_time("2026-10-08 12:00:00"):
        for _ in range(2):
            for _chunk in _client().chat.completions.create(model=calls.PERPLEXITY_MODEL, messages=MESSAGES,
                                                            stream=True):
                pass

    ids = _transaction_ids(recorder)
    assert len(set(ids)) == 2
    assert all(re.match(r"^perplexity-sdk-[0-9a-f]{32}$", transaction_id) for transaction_id in ids)
    assert set(ids) <= {f"perplexity-sdk-{uuid.UUID(int=i).hex}" for i in range(1, MINTED_UUIDS)}


def _refuses(request):
    return httpx.Response(429, json={"error": {"message": "rate limited", "type": "rate_limit_error"}})


def _refused_sync(streamed):
    client = perplexity.Perplexity(api_key="stub", max_retries=0,
                                   http_client=httpx.Client(transport=httpx.MockTransport(_refuses)))
    with idempotency_key("caller-block"):
        client.chat.completions.create(model=calls.PERPLEXITY_MODEL, messages=MESSAGES, stream=streamed)


def _refused_async(streamed):
    client = perplexity.AsyncPerplexity(api_key="stub", max_retries=0,
                                        http_client=httpx.AsyncClient(transport=httpx.MockTransport(_refuses)))

    async def call():
        with idempotency_key("caller-block"):
            await client.chat.completions.create(model=calls.PERPLEXITY_MODEL, messages=MESSAGES, stream=streamed)
    asyncio.run(call())


@pytest.mark.parametrize("streamed", [False, True], ids=["create", "stream"])
@pytest.mark.parametrize("call", [_refused_sync, _refused_async], ids=["sync", "async"])
def test_a_refused_native_call_sends_one_error_record(recorder, call, streamed):
    """A call Perplexity refuses is recorded once, as an ERROR, and the caller still gets its error (BACK-4335)."""
    with pytest.raises(perplexity.RateLimitError):
        call(streamed)

    calls.wait_for_metering()
    assert recorder.ai.create_completion.call_count == 1
    sent = recorder.ai.create_completion.call_args.kwargs
    assert sent["provider"] == "PERPLEXITY"
    assert sent["model"] == calls.PERPLEXITY_MODEL
    assert sent["stop_reason"] == "ERROR"
    assert "RateLimitError (HTTP 429)" in sent["error_reason"]
    assert sent.get("error_code") == 429
    assert (sent["input_token_count"], sent["output_token_count"], sent["total_token_count"]) == (0, 0, 0)
    assert sent["is_streamed"] is streamed
    assert sent["extra_headers"]["Idempotency-Key"] != "caller-block"
