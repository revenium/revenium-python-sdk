"""A LangChain Ollama model with the Revenium callback is metered once, by
the Ollama transport wrap (BACK-3604 on top of BACK-3582), and that record
keeps the attribution given to the callback (BACK-3913).

ChatOllama calls ``ollama.Client``/``AsyncClient``, which the Ollama
middleware now meters, so the callback must stand down. ``OllamaClientChatModel``
reproduces ChatOllama's call shape (every call streams through the Ollama
client and reports each chunk to ``on_llm_new_token``) so this runs where
langchain-ollama is not installed; the real ChatOllama runs where it is.
"""
import asyncio
import os
from unittest.mock import patch

import httpx
import ollama
import pytest

pytest.importorskip("langchain")

from langchain_core.language_models.chat_models import BaseChatModel  # noqa: E402
from langchain_core.messages import AIMessage, AIMessageChunk  # noqa: E402
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult  # noqa: E402

from revenium_middleware.openai.langchain import ReveniumCallbackHandler  # noqa: E402

from .test_client_wraps import (  # noqa: E402,F401  records and stub are fixtures
    INPUT_TOKENS, MODEL, OUTPUT_TOKENS, SELF_HOSTED, StubOllama, _run_now, records, stub,
)


def _chunk(part):
    usage = None
    if part.done:
        usage = {"input_tokens": part.prompt_eval_count, "output_tokens": part.eval_count,
                 "total_tokens": part.prompt_eval_count + part.eval_count}
    return ChatGenerationChunk(message=AIMessageChunk(content=part.message.content, usage_metadata=usage))


class OllamaClientChatModel(BaseChatModel):
    transport: httpx.MockTransport

    model_config = {"arbitrary_types_allowed": True}

    @property
    def _llm_type(self):
        return "chat-ollama"

    def _get_ls_params(self, stop=None, **kwargs):
        return {"ls_provider": "ollama", "ls_model_name": MODEL, "ls_model_type": "chat"}

    def _request(self, messages):
        return dict(model=MODEL, messages=[{"role": "user", "content": m.content} for m in messages], stream=True)

    def _stream(self, messages, stop=None, run_manager=None, **kwargs):
        client = ollama.Client(host=SELF_HOSTED, transport=self.transport)
        for part in client.chat(**self._request(messages)):
            chunk = _chunk(part)
            if run_manager:
                run_manager.on_llm_new_token(chunk.text, chunk=chunk)
            yield chunk

    async def _astream(self, messages, stop=None, run_manager=None, **kwargs):
        client = ollama.AsyncClient(host=SELF_HOSTED, transport=self.transport)
        async for part in await client.chat(**self._request(messages)):
            chunk = _chunk(part)
            if run_manager:
                await run_manager.on_llm_new_token(chunk.text, chunk=chunk)
            yield chunk

    @staticmethod
    def _result(chunks):
        final = chunks[0]
        for chunk in chunks[1:]:
            final += chunk
        message = AIMessage(content=final.text, usage_metadata=final.message.usage_metadata)
        return ChatResult(generations=[ChatGeneration(message=message)])

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        return self._result(list(self._stream(messages, stop, run_manager)))

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        return self._result([chunk async for chunk in self._astream(messages, stop, run_manager)])


class CallbackOnlyOllamaModel(OllamaClientChatModel):
    """An Ollama-labelled model that never reaches the Ollama client."""

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        message = AIMessage(content="hi", usage_metadata={"input_tokens": INPUT_TOKENS, "output_tokens": OUTPUT_TOKENS,
                                                          "total_tokens": INPUT_TOKENS + OUTPUT_TOKENS})
        return ChatResult(generations=[ChatGeneration(message=message)])


@pytest.fixture
def all_records(records):
    def record(operation, payload, *args, **kwargs):
        records.append(payload)

    with patch.dict(os.environ, {"REVENIUM_METERING_API_KEY": "hak_test_langchain_ollama"}), \
            patch("revenium_middleware.openai.middleware.submit_ai_event", side_effect=record), \
            patch("revenium_middleware.openai.middleware.run_async_in_thread", side_effect=_run_now):
        yield records


def invoke(model):
    return model.invoke("hi").text


def ainvoke(model):
    return asyncio.run(model.ainvoke("hi")).text


def stream(model):
    return "".join(chunk.text for chunk in model.stream("hi"))


def astream(model):
    async def call():
        return "".join([chunk.text async for chunk in model.astream("hi")])
    return asyncio.run(call())


MODES = [(invoke, False), (ainvoke, False), (stream, True), (astream, True)]


def assert_one_transport_record(stub, records):
    assert stub.hosts == ["ollama.internal"]
    assert len(records) == 1, records
    record = records[0]
    assert record["provider"] == "OLLAMA"
    assert record["transaction_id"].startswith("ollama-"), "the record must come from the transport, not the callback"
    assert record["input_token_count"] == INPUT_TOKENS
    assert record["output_token_count"] == OUTPUT_TOKENS
    assert record["is_streamed"] is True


@pytest.mark.parametrize("call, _streamed", MODES, ids=[mode.__name__ for mode, _ in MODES])
def test_a_chatollama_shaped_model_is_metered_once_by_the_transport(stub, all_records, call, _streamed):
    model = OllamaClientChatModel(transport=httpx.MockTransport(stub.respond), callbacks=[ReveniumCallbackHandler()])
    assert call(model) == "hi"
    assert_one_transport_record(stub, all_records)


@pytest.mark.parametrize("call, _streamed", MODES, ids=[mode.__name__ for mode, _ in MODES])
def test_the_real_chatollama_is_metered_once_by_the_transport(stub, all_records, call, _streamed):
    langchain_ollama = pytest.importorskip("langchain_ollama")
    model = langchain_ollama.ChatOllama(model=MODEL, base_url=SELF_HOSTED,
                                        client_kwargs={"transport": httpx.MockTransport(stub.respond)},
                                        callbacks=[ReveniumCallbackHandler()])
    assert call(model) == "hi"
    assert_one_transport_record(stub, all_records)


def test_an_ollama_model_the_transport_never_sees_keeps_its_callback_record(stub, all_records):
    model = CallbackOnlyOllamaModel(transport=httpx.MockTransport(stub.respond), callbacks=[ReveniumCallbackHandler()])
    assert invoke(model) == "hi"
    assert stub.hosts == []
    assert len(all_records) == 1
    assert all_records[0]["provider"] == "OLLAMA"
    assert all_records[0]["transaction_id"].startswith("langchain-")


@pytest.mark.parametrize("call, _streamed", MODES, ids=[mode.__name__ for mode, _ in MODES])
def test_the_transport_record_carries_the_callback_attribution(stub, all_records, call, _streamed):
    handler = ReveniumCallbackHandler(usage_metadata={"organizationName": "org-callback", "traceId": "trace-callback"})
    model = OllamaClientChatModel(transport=httpx.MockTransport(stub.respond), callbacks=[handler])
    assert call(model) == "hi"
    assert_one_transport_record(stub, all_records)
    assert all_records[0]["organization_name"] == "org-callback"
    assert all_records[0]["trace_id"] == "trace-callback"
