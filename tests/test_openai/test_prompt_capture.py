"""
Tests for prompt capture functionality.
"""

import asyncio
import copy
import json
import openai
import pytest
from unittest.mock import Mock, patch
from openai.types.responses import Response
from revenium_middleware._core.config import Config as CoreConfig
from revenium_middleware._core.prompt_extraction import sanitize_credentials
from revenium_middleware.openai.config import Config
from revenium_middleware.openai.prompt_extractor import (
    extract_prompts_from_request,
    extract_response_content,
    extract_streaming_response_content
)

MAX_PROMPT_LENGTH = Config.MAX_PROMPT_LENGTH
TRUNCATION_MARKER = "...[TRUNCATED]"


def _output_text(text):
    return {"type": "output_text", "text": text, "annotations": []}


def _output_message(item_id, content):
    return {"id": item_id, "type": "message", "role": "assistant", "status": "completed", "content": content}


def _responses_reply(output):
    return Response.model_validate({
        "id": "resp_1", "object": "response", "created_at": 1, "model": "o4-mini", "status": "completed",
        "parallel_tool_calls": True, "tool_choice": "auto", "tools": [], "output": output,
    })


class TestExtractPromptsFromRequest:
    """Test prompt extraction from OpenAI API requests."""
    
    def test_extract_system_and_user_messages(self):
        """Test extracting system prompt and user messages."""
        kwargs = {
            'messages': [
                {'role': 'system', 'content': 'You are a helpful assistant.'},
                {'role': 'user', 'content': 'Hello!'},
                {'role': 'assistant', 'content': 'Hi there!'},
                {'role': 'user', 'content': 'How are you?'}
            ]
        }
        
        result = extract_prompts_from_request(kwargs)
        
        assert result['systemPrompt'] == 'You are a helpful assistant.'
        assert result['promptsTruncated'] is False
        
        # Parse input messages
        input_messages = json.loads(result['inputMessages'])
        assert len(input_messages) == 3
        assert input_messages[0]['role'] == 'user'
        assert input_messages[0]['content'] == 'Hello!'
    
    def test_no_system_message(self):
        """Test when there's no system message."""
        kwargs = {
            'messages': [
                {'role': 'user', 'content': 'Hello!'}
            ]
        }
        
        result = extract_prompts_from_request(kwargs)
        
        assert result['systemPrompt'] is None
        assert result['inputMessages'] is not None
        assert result['promptsTruncated'] is False
    
    def test_empty_messages(self):
        """Test with empty messages list."""
        kwargs = {'messages': []}
        
        result = extract_prompts_from_request(kwargs)
        
        assert result['systemPrompt'] is None
        assert result['inputMessages'] is None
        assert result['promptsTruncated'] is False
    
    def test_truncation_system_prompt(self):
        """Test truncation of long system prompt."""
        long_content = 'A' * (MAX_PROMPT_LENGTH + 1000)
        kwargs = {
            'messages': [
                {'role': 'system', 'content': long_content},
                {'role': 'user', 'content': 'Hello!'}
            ]
        }

        result = extract_prompts_from_request(kwargs)

        # Should be truncated to exactly MAX_PROMPT_LENGTH (marker included)
        assert len(result['systemPrompt']) == MAX_PROMPT_LENGTH
        assert result['systemPrompt'].endswith(TRUNCATION_MARKER)
        assert result['promptsTruncated'] is True
    
    def test_truncation_input_messages(self):
        """Test truncation of long input messages."""
        # Create a large message that will exceed limit when JSON serialized
        large_message = 'B' * (MAX_PROMPT_LENGTH // 2)
        kwargs = {
            'messages': [
                {'role': 'user', 'content': large_message},
                {'role': 'user', 'content': large_message}
            ]
        }

        result = extract_prompts_from_request(kwargs)

        # The list is cut inside its texts, so it still parses and fits the limit
        stored = json.loads(result['inputMessages'])
        assert len(result['inputMessages']) <= MAX_PROMPT_LENGTH
        assert all(message['content'].endswith(TRUNCATION_MARKER) for message in stored)
        assert result['promptsTruncated'] is True

    def test_multimodal_content(self):
        """Test handling of multimodal content (arrays)."""
        kwargs = {
            'messages': [
                {
                    'role': 'system',
                    'content': [
                        {'type': 'text', 'text': 'You are helpful'},
                        {'type': 'image_url', 'image_url': {'url': 'http://example.com/img.jpg'}}
                    ]
                },
                {'role': 'user', 'content': 'What is this?'}
            ]
        }
        
        result = extract_prompts_from_request(kwargs)
        
        # System prompt should be JSON string for multimodal
        assert result['systemPrompt'] is not None
        assert isinstance(result['systemPrompt'], str)
        # Should be valid JSON
        json.loads(result['systemPrompt'])

    def test_responses_input_is_truncated_like_chat(self):
        """BACK-4381: a long Responses ``input`` string is cut as a chat message is.

        Only a ``str`` input is guaranteed to stay valid JSON after the cut; a list
        item whose long text is not a ``content`` string falls through to the
        whole-string cut, as a multimodal chat message does.
        """
        result = extract_prompts_from_request({'model': 'o4-mini', 'input': 'B' * (MAX_PROMPT_LENGTH + 1000)})

        assert result['inputMessages'] is not None
        assert TRUNCATION_MARKER in result['inputMessages']
        assert result['promptsTruncated'] is True
        assert len(result['inputMessages']) <= MAX_PROMPT_LENGTH
        assert json.loads(result['inputMessages'])[0]['role'] == 'user'

    def test_omitted_responses_fields_are_ignored(self):
        """BACK-4381: ``responses.stream()`` passes ``instructions=omit`` when the caller set none."""
        for placeholder in (openai.omit, openai.NOT_GIVEN):
            result = extract_prompts_from_request({'model': 'o4-mini', 'instructions': placeholder, 'input': 'hi'})

            assert result['systemPrompt'] is None


class TestExtractResponseContent:
    """Test response content extraction."""
    
    def test_extract_simple_response(self):
        """Test extracting simple text response."""
        response = Mock()
        response.choices = [Mock()]
        response.choices[0].message = Mock()
        response.choices[0].message.content = 'Hello, how can I help you?'
        
        result = extract_response_content(response)
        
        assert result['outputResponse'] == 'Hello, how can I help you?'
        assert result['promptsTruncated'] is False
    
    def test_extract_long_response(self):
        """Test truncation of long response."""
        long_content = 'C' * (MAX_PROMPT_LENGTH + 1000)
        response = Mock()
        response.choices = [Mock()]
        response.choices[0].message = Mock()
        response.choices[0].message.content = long_content

        result = extract_response_content(response)

        # Should be truncated to exactly MAX_PROMPT_LENGTH (marker included)
        assert len(result['outputResponse']) == MAX_PROMPT_LENGTH
        assert result['outputResponse'].endswith(TRUNCATION_MARKER)
        assert result['promptsTruncated'] is True
    
    def test_no_content(self):
        """Test when response has no content."""
        response = Mock()
        response.choices = [Mock()]
        response.choices[0].message = Mock()
        response.choices[0].message.content = None
        
        result = extract_response_content(response)
        
        assert result['outputResponse'] is None
        assert result['promptsTruncated'] is False
    
    def test_preserve_existing_truncation_flag(self):
        """Test that existing truncation flag is preserved."""
        response = Mock()
        response.choices = [Mock()]
        response.choices[0].message = Mock()
        response.choices[0].message.content = 'Short response'
        
        result = extract_response_content(response, prompts_truncated=True)

        assert result['promptsTruncated'] is True

    def test_responses_output_text_edge_shapes(self):
        """BACK-4381: a Responses reply is the text of its message items, joined as ``Response.output_text``."""
        reasoning = {"id": "rs_1", "type": "reasoning", "summary": []}
        tool_call = {"id": "fc_1", "type": "function_call", "call_id": "call_1", "name": "lookup",
                     "arguments": "{}", "status": "completed"}
        first = _output_message("msg_1", [_output_text("Hel")])
        second = _output_message("msg_2", [_output_text("lo"), {"type": "refusal", "refusal": "no"}])

        mixed = extract_response_content(_responses_reply([reasoning, tool_call, first, second]))
        tool_call_only = extract_response_content(_responses_reply([tool_call]))

        assert mixed['outputResponse'] == "Hello"
        assert tool_call_only['outputResponse'] is None


class TestExtractStreamingResponseContent:
    """Test streaming response content extraction."""

    def test_extract_streaming_content(self):
        """Test extracting accumulated streaming content."""
        accumulated = 'This is a streaming response.'

        result = extract_streaming_response_content(accumulated)

        assert result['outputResponse'] == accumulated
        assert result['promptsTruncated'] is False

    def test_truncate_long_streaming_content(self):
        """Test truncation of long streaming content."""
        long_content = 'D' * (MAX_PROMPT_LENGTH + 1000)

        result = extract_streaming_response_content(long_content)

        # Should be truncated to exactly MAX_PROMPT_LENGTH (marker included)
        assert len(result['outputResponse']) == MAX_PROMPT_LENGTH
        assert result['outputResponse'].endswith(TRUNCATION_MARKER)
        assert result['promptsTruncated'] is True


class TestPromptCaptureIntegration:
    """Integration tests for prompt capture with middleware."""

    def test_prompt_capture_disabled_by_default(self):
        from revenium_middleware._core.config import is_capture_prompts_enabled

        assert is_capture_prompts_enabled() is False

    @patch.dict('os.environ', {'REVENIUM_CAPTURE_PROMPTS': 'true'})
    def test_prompt_capture_enabled_via_env(self):
        from revenium_middleware._core.config import is_capture_prompts_enabled

        assert is_capture_prompts_enabled() is True

    @patch.dict('os.environ', {'REVENIUM_CAPTURE_PROMPTS': 'false'})
    def test_prompt_capture_disabled_via_env(self):
        from revenium_middleware._core.config import is_capture_prompts_enabled

        assert is_capture_prompts_enabled() is False

    def test_max_prompt_length_constant(self):
        """Test that MAX_PROMPT_LENGTH is set correctly."""
        from revenium_middleware.openai.config import Config

        assert Config.MAX_PROMPT_LENGTH == 50_000



def test_streamed_chat_capture_masks_credentials(monkeypatch, mock_revenium_client):
    """BACK-4237: a streamed OpenAI chat sends its captured prompt with credentials masked."""
    from revenium_middleware.openai import middleware as mw
    from .test_openai_sdk_entry_points import FakeOpenAI, _run_coro_to_completion

    monkeypatch.setenv("REVENIUM_CAPTURE_PROMPTS", "true")
    monkeypatch.setattr(mw, "run_async_in_thread", _run_coro_to_completion)
    messages = [
        {"role": "system", "content": "Authorization: Bearer FAKEBEARERTOKENFAKEBEARERTOKEN0002"},
        {"role": "user", "content": "My key is sk-proj-FAKE" + "0" * 48 + " and password: FAKEpassw0rd!"},
    ]

    client = FakeOpenAI().sync_client()
    list(client.chat.completions.create(model="gpt-4o-mini", messages=messages, stream=True))

    mock_revenium_client.ai.create_completion.assert_called_once()
    sent = mock_revenium_client.ai.create_completion.call_args.kwargs
    for field in ("system_prompt", "input_messages"):
        assert "FAKE" not in sent[field], (field, sent[field])
        assert "***REDACTED***" in sent[field], (field, sent[field])
    assert json.loads(sent["input_messages"])


INSTRUCTIONS = "Answer tersely."
QUESTION = "What is 2+2?"
STUB_REPLY = "hi"
RESPONSES_CAPTURE_REQUEST = {"model": "o4-mini", "instructions": INSTRUCTIONS, "input": QUESTION}
CHAT_CAPTURE_REQUEST = {"model": "gpt-4o-mini", "messages": [
    {"role": "system", "content": INSTRUCTIONS}, {"role": "user", "content": QUESTION}]}
CAPTURED_FIELDS = ("system_prompt", "input_messages", "output_response")


@pytest.fixture
def capture_on(monkeypatch):
    from revenium_middleware.openai import middleware as mw
    from .test_background_responses import _run_coro_to_completion

    monkeypatch.setenv("REVENIUM_CAPTURE_PROMPTS", "true")
    monkeypatch.setattr(mw, "run_async_in_thread", _run_coro_to_completion)


def _sent_record(mock_revenium_client):
    mock_revenium_client.ai.create_completion.assert_called_once()
    return mock_revenium_client.ai.create_completion.call_args.kwargs


def _fake_openai():
    from .test_openai_sdk_entry_points import FakeOpenAI

    return FakeOpenAI()


def _background_client(fake):
    from .test_background_responses import SyncDriver

    return SyncDriver(fake).client


def _drain_async(make_stream):
    async def drain():
        stream = await make_stream()
        return [event async for event in stream]
    return asyncio.run(drain())


def responses_sync_create(request):
    _fake_openai().sync_client().responses.create(**request)


def responses_async_create(request):
    asyncio.run(_fake_openai().async_client().responses.create(**request))


def responses_sync_raw_parse(request):
    _fake_openai().sync_client().responses.with_raw_response.create(**request).parse()


def responses_async_raw_parse(request):
    async def call():
        raw = await _fake_openai().async_client().responses.with_raw_response.create(**request)
        return raw.parse()
    asyncio.run(call())


def responses_sync_create_stream(request):
    list(_fake_openai().sync_client().responses.create(**request, stream=True))


def responses_async_create_stream(request):
    _drain_async(lambda: _fake_openai().async_client().responses.create(**request, stream=True))


def responses_sync_stream_helper(request):
    with _fake_openai().sync_client().responses.stream(**request) as stream:
        list(stream)


def responses_async_stream_helper(request):
    async def call():
        async with _fake_openai().async_client().responses.stream(**request) as stream:
            return [event async for event in stream]
    asyncio.run(call())


def responses_sync_raw_stream(request):
    list(_fake_openai().sync_client().responses.with_raw_response.create(**request, stream=True).parse())


def responses_async_raw_stream(request):
    async def raw_stream():
        raw = await _fake_openai().async_client().responses.with_raw_response.create(**request, stream=True)
        return raw.parse()
    _drain_async(raw_stream)


def responses_background(request):
    from .test_background_responses import BackgroundJobFake

    client = _background_client(BackgroundJobFake(polls_until_done=1))
    queued = client.responses.create(background=True, **request)
    client.responses.retrieve(queued.id)


def responses_background_resumed_stream(request):
    from .test_background_responses import INTERRUPT_AFTER, BackgroundJobFake

    fake = BackgroundJobFake()
    client = _background_client(fake)
    stream = client.responses.create(background=True, stream=True, **request)
    for event in stream:
        if event.sequence_number == INTERRUPT_AFTER:
            break
    stream.close()
    list(client.responses.retrieve(fake.response_id, stream=True, starting_after=INTERRUPT_AFTER))


RESPONSES_CALLS = [
    pytest.param(responses_sync_create, id="sync-create"),
    pytest.param(responses_async_create, id="async-create"),
    pytest.param(responses_sync_raw_parse, id="sync-with_raw_response-parse"),
    pytest.param(responses_async_raw_parse, id="async-with_raw_response-parse"),
    pytest.param(responses_sync_create_stream, id="sync-create-stream"),
    pytest.param(responses_async_create_stream, id="async-create-stream"),
    pytest.param(responses_sync_stream_helper, id="sync-stream-helper"),
    pytest.param(responses_async_stream_helper, id="async-stream-helper"),
    pytest.param(responses_sync_raw_stream, id="sync-raw-stream"),
    pytest.param(responses_async_raw_stream, id="async-raw-stream"),
    pytest.param(responses_background, id="background"),
    pytest.param(responses_background_resumed_stream, id="background-resumed-stream"),
]


@pytest.mark.parametrize("call", RESPONSES_CALLS)
def test_responses_call_captures_instructions_input_and_reply(call, capture_on, mock_revenium_client):
    """BACK-4381: a Responses call sends its instructions, input and reply, in every call shape."""
    call(RESPONSES_CAPTURE_REQUEST)

    sent = _sent_record(mock_revenium_client)
    assert sent.get("system_prompt") == INSTRUCTIONS
    assert json.loads(sent.get("input_messages") or "null") == [{"role": "user", "content": QUESTION}]
    assert sent.get("output_response") == STUB_REPLY


def test_responses_stream_reply_not_double_counted(capture_on, mock_revenium_client):
    """BACK-4381: the streamed reply arrives as deltas and again in ``response.completed``; it is sent once."""
    responses_sync_create_stream(RESPONSES_CAPTURE_REQUEST)

    assert _sent_record(mock_revenium_client).get("output_response") == STUB_REPLY


def test_responses_list_input_is_captured_as_given(capture_on, mock_revenium_client):
    """BACK-4381: a ``developer`` input item is the system prompt; the other items are the input."""
    user_item = {"role": "user", "content": [{"type": "input_text", "text": "Hi"}]}

    responses_sync_create({"model": "o4-mini", "input": [{"role": "developer", "content": "Be brief."}, user_item]})

    sent = _sent_record(mock_revenium_client)
    assert sent.get("system_prompt") == "Be brief."
    assert json.loads(sent.get("input_messages") or "null") == [user_item]


def test_responses_call_with_tools_keeps_its_operation_subtype(capture_on, mock_revenium_client):
    """BACK-4381: capturing a Responses call with tools leaves its operation subtype as it was."""
    responses_sync_create({**RESPONSES_CAPTURE_REQUEST, "tools": [{"type": "web_search"}]})

    sent = _sent_record(mock_revenium_client)
    assert sent.get("operation_subtype") is None
    assert sent.get("system_prompt") == INSTRUCTIONS


def chat_sync_raw_parse(request):
    _fake_openai().sync_client().chat.completions.with_raw_response.create(**request).parse()


def chat_async_raw_parse(request):
    async def call():
        raw = await _fake_openai().async_client().chat.completions.with_raw_response.create(**request)
        return raw.parse()
    asyncio.run(call())


def chat_sync_streaming_response_parse(request):
    with _fake_openai().sync_client().chat.completions.with_streaming_response.create(**request) as response:
        response.parse()


@pytest.mark.parametrize("call", [
    pytest.param(chat_sync_raw_parse, id="sync-with_raw_response-parse"),
    pytest.param(chat_async_raw_parse, id="async-with_raw_response-parse"),
    pytest.param(chat_sync_streaming_response_parse, id="sync-with_streaming_response-parse"),
])
def test_raw_response_chat_captures_the_reply(call, capture_on, mock_revenium_client):
    """BACK-4381: a raw-response chat call, the form LangChain uses, sends its reply as a plain call does."""
    call(CHAT_CAPTURE_REQUEST)

    sent = _sent_record(mock_revenium_client)
    assert sent.get("system_prompt") == INSTRUCTIONS
    assert json.loads(sent.get("input_messages") or "null") == [{"role": "user", "content": QUESTION}]
    assert sent.get("output_response") == STUB_REPLY


def chat_async_create_stream(request):
    _drain_async(lambda: _fake_openai().async_client().chat.completions.create(**request, stream=True))


def chat_async_raw_stream(request):
    async def raw_stream():
        raw = await _fake_openai().async_client().chat.completions.with_raw_response.create(**request, stream=True)
        return raw.parse()
    _drain_async(raw_stream)


@pytest.mark.parametrize("call", [
    pytest.param(chat_async_create_stream, id="async-create-stream"),
    pytest.param(chat_async_raw_stream, id="async-with_raw_response-stream"),
])
def test_async_streamed_chat_captures_the_reply(call, capture_on, mock_revenium_client):
    """BACK-4381: an async streamed chat call sends its reply, as a sync one does."""
    call(CHAT_CAPTURE_REQUEST)

    assert _sent_record(mock_revenium_client).get("output_response") == STUB_REPLY


def test_responses_capture_masks_credentials(capture_on, mock_revenium_client):
    """BACK-4381: Responses instructions and input are sent with credentials masked."""
    responses_sync_create({
        "model": "o4-mini",
        "instructions": "Authorization: Bearer FAKEBEARERTOKENFAKEBEARERTOKEN0002",
        "input": "My key is sk-proj-FAKE" + "0" * 48,
    })

    sent = _sent_record(mock_revenium_client)
    for field in ("system_prompt", "input_messages"):
        assert isinstance(sent.get(field), str), (field, sent.get(field))
        assert "FAKE" not in sent[field], (field, sent[field])
        assert "***REDACTED***" in sent[field], (field, sent[field])


def test_capture_off_holds_no_responses_request(monkeypatch, mock_revenium_client):
    """BACK-4381: with capture off, a waiting background job keeps no request text and sends none."""
    from revenium_middleware.openai import middleware as mw
    from .test_background_responses import BackgroundJobFake, _run_coro_to_completion

    monkeypatch.delenv("REVENIUM_CAPTURE_PROMPTS", raising=False)
    monkeypatch.setattr(mw, "run_async_in_thread", _run_coro_to_completion)
    client = _background_client(BackgroundJobFake(polls_until_done=1))

    queued = client.responses.create(background=True, **RESPONSES_CAPTURE_REQUEST)
    pending = mw._pending_background_responses._pending[queued.id]
    assert getattr(pending, "request_prompts", None) is None
    assert getattr(pending, "request_body", None) is None

    client.responses.retrieve(queued.id)
    sent = _sent_record(mock_revenium_client)
    for field in CAPTURED_FIELDS:
        assert field not in sent, (field, sent[field])


def test_a_responses_request_capture_cannot_read_still_goes_through(capture_on, mock_revenium_client, monkeypatch):
    """BACK-4381: a Responses request the prompt capture cannot read never fails the call or its usage record.

    The prompt is read before the call is sent. A failing read is forced here:
    a ``developer`` item holding an openai model fails ``json.dumps`` when this
    module runs alone, but not in the full suite.
    """
    from revenium_middleware.openai import middleware as mw

    def unreadable(_request):
        raise TypeError("Object of type ResponseInputText is not JSON serializable")

    monkeypatch.setattr(mw, "extract_prompts_from_request", unreadable)

    responses_sync_create(RESPONSES_CAPTURE_REQUEST)

    sent = _sent_record(mock_revenium_client)
    assert sent["input_token_count"] == 11
    for field in CAPTURED_FIELDS:
        assert field not in sent, (field, sent[field])


SMALL_LIMIT = 100
LONG_REPLY_PIECE = "x" * 40
LONG_REPLY_PIECES = 10


def _assert_cut_to_the_limit(sent):
    assert len(sent["output_response"]) == SMALL_LIMIT
    assert sent["output_response"].endswith("...[TRUNCATED]")
    assert sent["prompts_truncated"] is True


@pytest.mark.parametrize("terminal_carries_text", [
    pytest.param(False, id="deltas-only"),
    pytest.param(True, id="final-text"),
])
def test_responses_stream_reply_over_the_limit_is_cut_and_marked(
        terminal_carries_text, capture_on, mock_revenium_client, monkeypatch):
    """A Responses reply streamed in pieces far over the limit is sent cut to it, marked truncated."""
    from . import test_openai_sdk_entry_points as entry_points

    real_event_payloads = entry_points.event_payloads
    long_text = LONG_REPLY_PIECE * LONG_REPLY_PIECES

    def long_reply_payloads(terminal="response.completed", usage=None):
        events = real_event_payloads(terminal, usage)
        delta = next(event for event in events if event["type"] == "response.output_text.delta")
        pieces = [{**delta, "delta": LONG_REPLY_PIECE} for _ in range(LONG_REPLY_PIECES)]
        events = [piece for event in events for piece in (pieces if event is delta else [event])]
        last = events[-1]
        output = [{**last["response"]["output"][0], "content": [
            {"type": "output_text", "text": long_text, "annotations": []}]}] if terminal_carries_text else []
        events[-1] = {**last, "response": {**last["response"], "output": output}}
        return [{**event, "sequence_number": i} for i, event in enumerate(events)]

    monkeypatch.setattr(entry_points, "event_payloads", long_reply_payloads)
    monkeypatch.setattr(CoreConfig, "MAX_PROMPT_LENGTH", SMALL_LIMIT)

    responses_sync_create_stream(RESPONSES_CAPTURE_REQUEST)

    _assert_cut_to_the_limit(_sent_record(mock_revenium_client))


def test_async_streamed_chat_reply_over_the_limit_is_cut_and_marked(capture_on, mock_revenium_client, monkeypatch):
    """An async chat reply streamed in pieces far over the limit is sent cut to it, marked truncated."""
    from .test_openai_sdk_entry_points import FakeOpenAI

    monkeypatch.setattr(CoreConfig, "MAX_PROMPT_LENGTH", SMALL_LIMIT)
    fake = FakeOpenAI(first_content=LONG_REPLY_PIECE * LONG_REPLY_PIECES)

    _drain_async(lambda: fake.async_client().chat.completions.create(**CHAT_CAPTURE_REQUEST, stream=True))

    _assert_cut_to_the_limit(_sent_record(mock_revenium_client))


def test_reply_collector_never_holds_more_than_one_character_past_the_limit(monkeypatch):
    from revenium_middleware._core.prompt_extraction import ReplyCollector

    monkeypatch.setattr(CoreConfig, "MAX_PROMPT_LENGTH", 10)
    collector = ReplyCollector()
    for piece in ("abcd", "efgh", "ijkl", "mnop", ""):
        collector.add(piece)
        assert len(collector.text()) <= 11

    assert collector.text() == "abcdefghijk"
LIMIT = 50_000


@pytest.fixture
def limit_50000(monkeypatch):
    monkeypatch.setattr("revenium_middleware._core.prompt_extraction.Config.MAX_PROMPT_LENGTH", LIMIT)


def _user(content):
    return {'role': 'user', 'content': content}


def _extract_messages(*messages):
    return extract_prompts_from_request({'messages': list(messages)})


@pytest.mark.usefixtures('limit_50000')
class TestMessageListLimit:
    """input_messages is whole when it fits and valid JSON within the limit when it does not."""

    def test_a_single_30000_character_message_is_kept_whole(self):
        result = _extract_messages(_user('D' * 30_000))

        assert json.loads(result['inputMessages']) == [_user('D' * 30_000)]
        assert result['promptsTruncated'] is False

    def test_an_overflowing_message_list_is_valid_json_within_the_limit(self):
        result = _extract_messages(*[_user(letter * 20_000) for letter in 'ABCD'])

        stored = json.loads(result['inputMessages'])
        assert len(result['inputMessages']) <= LIMIT
        assert len(stored) == 4
        assert all(message['content'].endswith(TRUNCATION_MARKER) for message in stored)
        assert result['promptsTruncated'] is True

    def test_only_the_longest_message_is_cut(self):
        result = _extract_messages(_user('hi'), _user('D' * 60_000), _user('thanks'))

        stored = json.loads(result['inputMessages'])
        assert [stored[0]['content'], stored[2]['content']] == ['hi', 'thanks']
        assert stored[1]['content'].endswith(TRUNCATION_MARKER)
        assert LIMIT - 100 < len(result['inputMessages']) <= LIMIT
        assert result['promptsTruncated'] is True

    def test_a_long_text_part_is_cut_and_the_image_part_kept(self):
        image = {'type': 'image_url', 'image_url': {'url': 'https://example.com/cat.png'}}
        result = _extract_messages(_user([{'type': 'text', 'text': 'T' * 60_000}, image]))

        stored = json.loads(result['inputMessages'])
        text_part, image_part = stored[0]['content']
        assert len(result['inputMessages']) <= LIMIT
        assert text_part['text'].endswith(TRUNCATION_MARKER)
        assert image_part == image
        assert result['promptsTruncated'] is True

    def test_escaped_text_never_pushes_the_list_over_the_limit(self):
        result = _extract_messages(_user('"\n' * 20_000))

        assert isinstance(json.loads(result['inputMessages']), list)
        assert len(result['inputMessages']) <= LIMIT
        assert result['promptsTruncated'] is True

    def test_too_many_messages_keep_the_newest(self, monkeypatch):
        monkeypatch.setattr("revenium_middleware._core.prompt_extraction.Config.MAX_PROMPT_LENGTH", 1_000)
        sent = [_user(f'message {number}') for number in range(2_000)]

        result = _extract_messages(*sent)

        stored = json.loads(result['inputMessages'])
        assert len(result['inputMessages']) <= 1_000
        assert stored[-1] == sent[-1]
        assert result['promptsTruncated'] is True

    def test_the_newest_message_that_fits_is_kept_whole_after_older_ones_are_dropped(self, monkeypatch):
        monkeypatch.setattr("revenium_middleware._core.prompt_extraction.Config.MAX_PROMPT_LENGTH", 200)
        older = _user([{'type': 'text', 'text': 'T' * 100} for _ in range(10)])
        newest = _user('N' * 30)

        result = _extract_messages(older, newest)

        assert json.loads(result['inputMessages']) == [newest]
        assert result['promptsTruncated'] is True

    def test_capturing_a_long_request_leaves_the_request_unchanged(self):
        """Guard: the request is still being sent to the provider."""
        image = {'type': 'image_url', 'image_url': {'url': 'https://example.com/cat.png'}}
        messages = [_user('D' * 60_000), _user([{'type': 'text', 'text': 'T' * 60_000}, image])]
        before = copy.deepcopy(messages)

        _extract_messages(*messages)

        assert messages == before

    def test_an_overflowing_request_with_a_key_is_stored_parseable_and_masked(self):
        key = 'sk-' + 'a' * 48
        result = _extract_messages(_user(f'my key is {key}'), *[_user(letter * 30_000) for letter in 'ABC'])

        stored = sanitize_credentials(result['inputMessages'])

        messages = json.loads(stored)
        assert key not in stored
        assert '***REDACTED***' in messages[0]['content']
