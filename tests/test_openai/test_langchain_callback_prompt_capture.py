"""With prompt capture on, a LangChain call the callback records sends its
prompt and reply (BACK-4448).

The callback records the calls no Revenium client patch meters: ChatMistralAI,
ChatCohere and other partner chat models, and text-completion models. Imports
only langchain-core, langchain-mistralai's stub and the handler module, so it
runs wherever the dev extra is installed. Stand-in models report an
``ls_provider`` with no transport, so no client patch claims their calls.
"""
import asyncio
import json
import uuid
from unittest.mock import MagicMock, patch

import pytest

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.language_models.llms import LLM
from langchain_core.messages import (
    AIMessage, AIMessageChunk, HumanMessage, SystemMessage, SystemMessageChunk, ToolMessage,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult, Generation, LLMResult

from revenium_middleware._core.call_ownership import _published_in_context, reset_claimed_response_ids
from revenium_middleware._core.prompt_extraction import extract_prompts_from_request
from revenium_middleware.openai.langchain.unified_handler import UnifiedReveniumCallbackHandler
from entry_points import calls, mistral

SYSTEM = "You are a terse assistant."
QUESTION = "Say hi."
REPLY = "the reply"
TEXT_PROMPT = "text prompt"
INPUT_TOKENS = 13
OUTPUT_TOKENS = 5
CAP = 50_000
TRUNCATED = "...[TRUNCATED]"
PROMPT_FIELDS = ("system_prompt", "input_messages", "output_response", "prompts_truncated")
TIME_FIELDS = ("request_time", "response_time", "completion_start_time", "request_duration")
PER_RECORD_FIELDS = ("transaction_id", "extra_headers")


@pytest.fixture(autouse=True)
def _isolated(monkeypatch):
    monkeypatch.setenv("REVENIUM_CAPTURE_PROMPTS", "true")
    monkeypatch.delenv("REVENIUM_SELECTIVE_METERING", raising=False)
    reset_claimed_response_ids()
    publications = _published_in_context.set(())
    yield
    _published_in_context.reset(publications)
    reset_claimed_response_ids()


@pytest.fixture
def records():
    recording_client = MagicMock()

    def recorded():
        calls.wait_for_metering()
        return [entry.kwargs for entry in recording_client.ai.create_completion.call_args_list]

    with patch("revenium_middleware._core.metering.client", recording_client):
        yield recorded


def _usage():
    return {"input_tokens": INPUT_TOKENS, "output_tokens": OUTPUT_TOKENS,
            "total_tokens": INPUT_TOKENS + OUTPUT_TOKENS}


class StandIn(BaseChatModel):
    """A callback-only chat model whose streamed reply comes in two chunks."""

    ls_provider: str = "mistral"
    model: str = "stand-in-small"
    reply: str = REPLY
    tool_calls: list = []

    @property
    def _llm_type(self):
        return "callback-stand-in"

    def _get_ls_params(self, stop=None, **kwargs):
        return {"ls_provider": self.ls_provider, "ls_model_name": self.model, "ls_model_type": "chat"}

    def _reply_to(self, messages):
        return self.reply

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        message = AIMessage(content=self._reply_to(messages), tool_calls=self.tool_calls, usage_metadata=_usage())
        return ChatResult(generations=[ChatGeneration(message=message)])

    def _stream(self, messages, stop=None, run_manager=None, **kwargs):
        reply = self._reply_to(messages)
        yield ChatGenerationChunk(message=AIMessageChunk(content=reply[:4]))
        yield ChatGenerationChunk(message=AIMessageChunk(content=reply[4:], usage_metadata=_usage()))


class EchoStandIn(StandIn):
    def _reply_to(self, messages):
        return f"echo {messages[-1].content}"


class FailingStandIn(StandIn):
    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        raise RuntimeError("provider down")


class TextStandIn(LLM):
    """A callback-only text-completion model."""

    @property
    def _llm_type(self):
        return "callback-text-stand-in"

    def _get_ls_params(self, stop=None, **kwargs):
        return {"ls_provider": "mistral", "ls_model_name": "stand-in-text", "ls_model_type": "llm"}

    def _call(self, prompt, stop=None, run_manager=None, **kwargs):
        return REPLY

    def _generate(self, prompts, stop=None, run_manager=None, **kwargs):
        return LLMResult(generations=[[Generation(text=REPLY)] for _ in prompts],
                         llm_output={"token_usage": {"prompt_tokens": INPUT_TOKENS,
                                                     "completion_tokens": OUTPUT_TOKENS,
                                                     "total_tokens": INPUT_TOKENS + OUTPUT_TOKENS}})

    async def _agenerate(self, prompts, stop=None, run_manager=None, **kwargs):
        return self._generate(prompts, stop=stop, **kwargs)


def invoke(model, prompt):
    model.invoke(prompt)


def ainvoke(model, prompt):
    asyncio.run(model.ainvoke(prompt))


def stream(model, prompt):
    list(model.stream(prompt))


def astream(model, prompt):
    async def call():
        return [chunk async for chunk in model.astream(prompt)]
    asyncio.run(call())


def _user_messages(*contents):
    return json.dumps([{"role": "user", "content": content} for content in contents], ensure_ascii=False)


def _chat(model_class=StandIn, **fields):
    return model_class(callbacks=[UnifiedReveniumCallbackHandler()], **fields)


def _only(sent):
    assert len(sent) == 1, sent
    return sent[0]


def _start_chat_run(handler, messages):
    run_id = uuid.uuid4()
    handler.on_chat_model_start({"id": ["stand_in", "ChatStandIn"]}, [messages],
                                run_id=run_id, metadata={"ls_provider": "mistral"})
    return run_id


def _llm_result(text):
    message = AIMessage(content=text, usage_metadata=_usage())
    return LLMResult(generations=[[ChatGeneration(message=message)]])


@pytest.mark.parametrize("call", [invoke, ainvoke, stream, astream])
def test_a_callback_only_chat_call_sends_its_system_prompt_input_and_reply(records, call):
    call(_chat(), [SystemMessage(SYSTEM), HumanMessage(QUESTION)])

    record = _only(records())
    assert record.get("system_prompt") == SYSTEM
    assert record.get("input_messages") == _user_messages(QUESTION)
    assert record.get("output_response") == REPLY
    assert record.get("prompts_truncated") is False
    if call in (stream, astream):
        assert record.get("is_streamed") is True


@pytest.mark.parametrize("call", [invoke, ainvoke])
def test_a_callback_only_text_completion_call_sends_its_prompt_and_reply(records, call):
    call(TextStandIn(callbacks=[UnifiedReveniumCallbackHandler()]), TEXT_PROMPT)

    record = _only(records())
    assert record.get("system_prompt") is None
    assert record.get("input_messages") == _user_messages(TEXT_PROMPT)
    assert record.get("output_response") == REPLY


def test_a_callback_record_labelled_openai_with_no_transport_claim_carries_capture(records):
    invoke(_chat(ls_provider="openai"), [SystemMessage(SYSTEM), HumanMessage(QUESTION)])

    record = _only(records())
    assert record["provider"] == "OPENAI"
    assert record.get("system_prompt") == SYSTEM
    assert record.get("input_messages") == _user_messages(QUESTION)
    assert record.get("output_response") == REPLY


def test_capture_off_sends_and_holds_no_prompt(records, monkeypatch):
    monkeypatch.delenv("REVENIUM_CAPTURE_PROMPTS", raising=False)
    invoke(_chat(), [SystemMessage(SYSTEM), HumanMessage(QUESTION)])

    record = _only(records())
    assert [record.get(field) for field in PROMPT_FIELDS] == [None] * len(PROMPT_FIELDS)

    handler = UnifiedReveniumCallbackHandler()
    run_id = _start_chat_run(handler, [SystemMessage(SYSTEM), HumanMessage(QUESTION)])
    assert handler._active_runs[run_id]["prompts"] is None


def test_a_selectively_unmetered_run_holds_no_prompt(records, monkeypatch):
    monkeypatch.setenv("REVENIUM_SELECTIVE_METERING", "true")
    handler = UnifiedReveniumCallbackHandler()

    run_id = _start_chat_run(handler, [SystemMessage(SYSTEM), HumanMessage(QUESTION)])
    assert handler._active_runs[run_id]["prompts"] is None

    handler.on_llm_end(_llm_result(REPLY), run_id=run_id)
    assert records() == []


def test_callback_captured_fields_are_cut_at_the_cap_and_flagged(records):
    long_system = "s" * 60_000
    invoke(_chat(reply="r" * 60_000), [SystemMessage(long_system), HumanMessage(QUESTION)])

    record = _only(records())
    assert len(record["system_prompt"]) == CAP and record["system_prompt"].endswith(TRUNCATED)
    assert len(record["output_response"]) == CAP and record["output_response"].endswith(TRUNCATED)
    assert record["prompts_truncated"] is True
    same_request_through_openai_path = extract_prompts_from_request({"messages": [
        {"role": "system", "content": long_system}, {"role": "user", "content": QUESTION}]})
    assert record["system_prompt"] == same_request_through_openai_path["systemPrompt"]
    assert record["input_messages"] == same_request_through_openai_path["inputMessages"]


def test_credentials_in_callback_captured_fields_are_masked(records):
    key = "sk-" + "a" * 30
    secret_text = f"my key {key} password: hunter2-long"
    invoke(_chat(reply=secret_text), [HumanMessage(secret_text)])

    record = _only(records())
    for field in ("input_messages", "output_response"):
        assert "sk-***REDACTED***" in record[field]
        assert "password: ***REDACTED***" in record[field]
        assert key not in record[field]
        assert "hunter2-long" not in record[field]


def test_a_reply_with_no_text_never_sends_the_placeholder_reply(records):
    model = _chat(reply="", tool_calls=[{"name": "f", "args": {}, "id": "c1"}])
    invoke(model, [HumanMessage(QUESTION)])

    record = _only(records())
    assert record.get("output_response") is None
    assert not any("Mock response" in str(value) for value in record.values())


def test_an_ended_or_failed_run_keeps_no_prompt(records):
    handler = UnifiedReveniumCallbackHandler()
    StandIn(callbacks=[handler]).invoke([HumanMessage(QUESTION)])
    assert handler._active_runs == {}

    with pytest.raises(RuntimeError):
        FailingStandIn(callbacks=[handler]).invoke([HumanMessage(QUESTION)])
    assert handler._active_runs == {}


def test_chunk_messages_in_history_keep_their_roles(records):
    history = [SystemMessageChunk(content=SYSTEM), HumanMessage(QUESTION), AIMessageChunk(content="a"),
               HumanMessage("and again?")]
    invoke(_chat(), history)

    record = _only(records())
    assert record.get("system_prompt") == SYSTEM
    assert [message["role"] for message in json.loads(record["input_messages"])] == ["user", "assistant", "user"]


def test_a_tool_message_keeps_its_tool_call_id(records):
    history = [HumanMessage(QUESTION), AIMessage(content="", tool_calls=[{"name": "f", "args": {}, "id": "c1"}]),
               ToolMessage(content="42", tool_call_id="c1")]
    invoke(_chat(), history)

    sent = json.loads(_only(records())["input_messages"])
    assert {"role": "tool", "content": "42", "tool_call_id": "c1"} in sent


def test_capture_changes_only_the_prompt_fields(records, monkeypatch):
    tools = [{"type": "function", "function": {"name": "f", "parameters": {}}}]
    dropped = set(PROMPT_FIELDS) | set(PER_RECORD_FIELDS) | set(TIME_FIELDS)

    def sent_without_capture_fields():
        _chat().invoke([SystemMessage(SYSTEM), HumanMessage(QUESTION)], tools=tools)
        return {name: value for name, value in records()[-1].items() if name not in dropped}

    monkeypatch.delenv("REVENIUM_CAPTURE_PROMPTS", raising=False)
    capture_off = sent_without_capture_fields()
    monkeypatch.setenv("REVENIUM_CAPTURE_PROMPTS", "true")
    capture_on = sent_without_capture_fields()

    assert records()[-1].get("system_prompt") == SYSTEM
    assert capture_on == capture_off


@pytest.mark.parametrize("call", [invoke, ainvoke, stream])
def test_a_chat_mistralai_call_sends_its_prompt_and_reply(records, call):
    call(mistral.chat_mistralai([UnifiedReveniumCallbackHandler()]), [SystemMessage(SYSTEM), HumanMessage(QUESTION)])

    record = _only(records())
    assert record["provider"] == "MISTRAL"
    assert record.get("system_prompt") == SYSTEM
    assert record.get("input_messages") == _user_messages(QUESTION)
    assert record.get("output_response") == mistral.ANSWER


def test_concurrent_callback_runs_keep_their_own_prompts(records):
    model = _chat(EchoStandIn)

    async def both():
        await asyncio.gather(model.ainvoke([HumanMessage("first")]), model.ainvoke([HumanMessage("second")]))

    asyncio.run(both())

    sent = records()
    assert len(sent) == 2, sent
    pairs = sorted((record["input_messages"], record["output_response"]) for record in sent)
    assert pairs == [(_user_messages("first"), "echo first"), (_user_messages("second"), "echo second")]
