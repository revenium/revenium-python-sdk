"""Prompt capture on LiteLLM client calls (BACK-4045).

With ``REVENIUM_CAPTURE_PROMPTS`` on, a LiteLLM chat row carries the system
prompt, the other input messages and the reply, with the same values and the
same truncation as the OpenAI integration records for the same messages.
"""
import asyncio
import json
import time

import pytest

pytest.importorskip("litellm")

import litellm  # noqa: E402

from revenium_middleware._core import metering_pool  # noqa: E402
from revenium_middleware._core.config import Config  # noqa: E402
from revenium_middleware.litellm.client import middleware  # noqa: E402,F401

MODEL = "gpt-4o-mini"
SYSTEM_PROMPT = "You are terse."
USER_MESSAGES = [{"role": "user", "content": "hi there"}]
MESSAGES = [{"role": "system", "content": SYSTEM_PROMPT}, *USER_MESSAGES]
REPLY = "hello world"
PROMPT_FIELDS = ("system_prompt", "input_messages", "output_response", "prompts_truncated")
# Long enough for the metering pool on a loaded CI runner; a passing run
# returns as soon as the row lands.
ROW_TIMEOUT_SECONDS = 15.0


def _completion():
    litellm.completion(model=MODEL, messages=MESSAGES, mock_response=REPLY)


def _acompletion():
    asyncio.run(litellm.acompletion(model=MODEL, messages=MESSAGES, mock_response=REPLY))


def _streamed_completion():
    list(litellm.completion(model=MODEL, messages=MESSAGES, mock_response=REPLY, stream=True))


def _streamed_acompletion():
    async def consume():
        stream = await litellm.acompletion(model=MODEL, messages=MESSAGES, mock_response=REPLY, stream=True)
        return [chunk async for chunk in stream]

    asyncio.run(consume())


CHAT_CALLS = [
    pytest.param(_completion, id="completion"),
    pytest.param(_acompletion, id="acompletion"),
    pytest.param(_streamed_completion, id="completion-streamed"),
    pytest.param(_streamed_acompletion, id="acompletion-streamed"),
]


def _the_one_row(mock_revenium_client):
    deadline = time.monotonic() + ROW_TIMEOUT_SECONDS
    while time.monotonic() < deadline and not mock_revenium_client.ai.create_completion.called:
        metering_pool.wait_until_idle(0.5)
        time.sleep(0.05)
    metering_pool.wait_until_idle(ROW_TIMEOUT_SECONDS)
    calls = mock_revenium_client.ai.create_completion.call_args_list
    assert len(calls) == 1, calls
    return calls[0].kwargs


@pytest.fixture
def capture_on(monkeypatch):
    monkeypatch.setenv("REVENIUM_CAPTURE_PROMPTS", "true")


@pytest.fixture
def capture_off(monkeypatch):
    monkeypatch.delenv("REVENIUM_CAPTURE_PROMPTS", raising=False)


@pytest.mark.parametrize("call", CHAT_CALLS)
def test_capture_on_records_system_prompt_input_and_output(call, capture_on, mock_revenium_client):
    call()

    row = _the_one_row(mock_revenium_client)
    assert {field: row.get(field) for field in PROMPT_FIELDS} == {
        "system_prompt": SYSTEM_PROMPT,
        "input_messages": json.dumps(USER_MESSAGES, ensure_ascii=False),
        "output_response": REPLY,
        "prompts_truncated": False,
    }


def test_a_single_30000_character_message_is_recorded_whole(capture_on, mock_revenium_client, monkeypatch):
    monkeypatch.setattr("revenium_middleware._core.prompt_extraction.Config.MAX_PROMPT_LENGTH", 50_000)
    document = [{"role": "user", "content": "D" * 30_000}]

    litellm.completion(model=MODEL, messages=[{"role": "system", "content": SYSTEM_PROMPT}, *document],
                       mock_response=REPLY)

    row = _the_one_row(mock_revenium_client)
    assert json.loads(row["input_messages"]) == document
    assert not row.get("prompts_truncated")


@pytest.mark.parametrize("call", CHAT_CALLS)
def test_capture_off_records_no_prompt_fields(call, capture_off, mock_revenium_client):
    call()

    row = _the_one_row(mock_revenium_client)
    assert [field for field in PROMPT_FIELDS if field in row] == []


def test_an_embedding_records_no_prompt_fields(capture_on, mock_revenium_client):
    litellm.embedding(model="text-embedding-3-small", input=["hi there"], mock_response=[0.1, 0.2])

    row = _the_one_row(mock_revenium_client)
    assert [field for field in PROMPT_FIELDS if field in row] == []


LONG_REPLY = "the quick brown fox jumps over the lazy dog"


def _long_reply_row(mock_revenium_client, stream):
    response = litellm.completion(model=MODEL, messages=MESSAGES, mock_response=LONG_REPLY, stream=stream)
    if stream:
        list(response)
    row = _the_one_row(mock_revenium_client)
    mock_revenium_client.ai.create_completion.reset_mock()
    return row


def test_a_streamed_reply_over_the_limit_is_truncated_like_a_whole_reply(capture_on, mock_revenium_client,
                                                                          monkeypatch):
    monkeypatch.setattr(Config, "MAX_PROMPT_LENGTH", 20)

    whole = _long_reply_row(mock_revenium_client, stream=False)
    streamed = _long_reply_row(mock_revenium_client, stream=True)

    assert streamed["output_response"] == whole["output_response"]
    assert streamed["output_response"].endswith("...[TRUNCATED]")
    assert streamed["prompts_truncated"] is True


def test_a_streamed_reply_keeps_one_character_past_the_limit_and_releases_it(monkeypatch):
    monkeypatch.setattr(Config, "MAX_PROMPT_LENGTH", 5)
    reply = middleware._ReplyCollector()

    for chunk in ("abc", "defg", "hij"):
        reply.add(chunk)

    assert reply.take() == "abcdef"
    assert reply.take() == ""


# BACK-4237: credentials in the prompt and the reply are masked on every chat form.
FAKE_BEARER = "FAKEBEARERTOKENFAKEBEARERTOKEN0002"
FAKE_PROJECT_KEY = "sk-proj-FAKE" + "0" * 48
FAKE_ANTHROPIC_KEY = "sk-ant-FAKE" + "3" * 24
SECRET_MESSAGES = [
    {"role": "system", "content": f"Authorization: Bearer {FAKE_BEARER}"},
    {"role": "user", "content": f"My key is {FAKE_PROJECT_KEY} and password: FAKEpassw0rd!"},
]
SECRET_REPLY = f"Noted, your other key is {FAKE_ANTHROPIC_KEY}"


def _completion_replying(reply):
    litellm.completion(model=MODEL, messages=SECRET_MESSAGES, mock_response=reply)


def _acompletion_replying(reply):
    asyncio.run(litellm.acompletion(model=MODEL, messages=SECRET_MESSAGES, mock_response=reply))


def _streamed_completion_replying(reply):
    list(litellm.completion(model=MODEL, messages=SECRET_MESSAGES, mock_response=reply, stream=True))


def _streamed_acompletion_replying(reply):
    async def consume():
        stream = await litellm.acompletion(model=MODEL, messages=SECRET_MESSAGES, mock_response=reply, stream=True)
        return [chunk async for chunk in stream]

    asyncio.run(consume())


CHAT_CALLS_REPLYING = [
    pytest.param(_completion_replying, id="completion"),
    pytest.param(_acompletion_replying, id="acompletion"),
    pytest.param(_streamed_completion_replying, id="completion-streamed"),
    pytest.param(_streamed_acompletion_replying, id="acompletion-streamed"),
]


@pytest.mark.parametrize("call", CHAT_CALLS_REPLYING)
def test_capture_on_masks_credentials_in_prompt_and_reply(call, capture_on, mock_revenium_client):
    call(SECRET_REPLY)

    row = _the_one_row(mock_revenium_client)
    for field in ("system_prompt", "input_messages", "output_response"):
        assert "FAKE" not in row[field], (field, row[field])
        assert "***REDACTED***" in row[field], (field, row[field])
    assert json.loads(row["input_messages"])[0]["role"] == "user"
