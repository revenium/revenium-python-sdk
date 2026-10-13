"""A LangChain call to every provider the README lists is recorded once,
under that provider, with the attribution given to the callback (BACK-3989).

Imports only langchain-core and the handler module, so it runs wherever the
dev extra is installed. Stand-in chat models report the ``ls_provider`` the
partner packages report and, for vendors the SDK wraps (Gemini, LiteLLM,
native Perplexity), call the real wrapped client over a stubbed transport.
"""
import asyncio
import contextvars
import gc
import importlib.util
import itertools
import re
import uuid
from unittest.mock import MagicMock, patch

import pytest

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult, LLMResult
from langchain_core.runnables import Runnable, RunnableLambda
from langchain_core.runnables.config import ensure_config

import revenium_middleware.google  # noqa: F401  installs the google-genai patches
import revenium_middleware.litellm  # noqa: F401  installs the LiteLLM client patches
import revenium_middleware.perplexity  # noqa: F401  installs the Perplexity patches
from revenium_middleware._core.call_ownership import (
    ANTHROPIC, BEDROCK, OPENAI, PERPLEXITY, _published_in_context, claim_call_for_transport,
    reset_claimed_response_ids, take_callback_metadata,
)
from revenium_middleware import revenium_meter, revenium_metadata
from revenium_middleware._core.exceptions import BudgetExceededError
from revenium_middleware.openai.langchain import _model_call, attach_to, unified_handler
from revenium_middleware.openai.langchain.unified_handler import UnifiedReveniumCallbackHandler
from entry_points import calls, mistral

INPUT_TOKENS = 13
OUTPUT_TOKENS = 5

CALLBACK_METADATA = {
    "subscriber": {"id": "sub-callback", "email": "callback@example.com"},
    "organizationName": "org-callback",
    "traceId": "trace-callback",
}


@pytest.fixture(autouse=True)
def _isolated(monkeypatch):
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


def attributed_handler():
    return UnifiedReveniumCallbackHandler(usage_metadata=CALLBACK_METADATA)


def _usage(input_tokens, output_tokens):
    return {"input_tokens": input_tokens, "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens}


def _result(text, served_model, input_tokens, output_tokens):
    message = AIMessage(content=text, response_metadata={"model_name": served_model},
                        usage_metadata=_usage(input_tokens, output_tokens))
    return ChatResult(generations=[ChatGeneration(message=message)])


class StandIn(BaseChatModel):
    """A callback-only chat model: no Revenium transport wrap sees its calls."""

    ls_provider: str = "cohere"
    model: str = "command-r"

    @property
    def _llm_type(self):
        return f"{self.ls_provider}-stand-in"

    def _get_ls_params(self, stop=None, **kwargs):
        return {"ls_provider": self.ls_provider, "ls_model_name": self.model, "ls_model_type": "chat"}

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        return _result("hi", self.model, INPUT_TOKENS, OUTPUT_TOKENS)


VERTEX_CACHE_READ = 1000


class VertexClaudeStandIn(StandIn):
    """How langchain-google-vertexai's ChatAnthropicVertex streams: the cache
    read pooled into input_tokens, as its copy of _create_usage_metadata does."""

    ls_provider: str = "anthropicvertex"
    model: str = "claude-sonnet-4-5"

    def _stream(self, messages, stop=None, run_manager=None, **kwargs):
        pooled_input = INPUT_TOKENS + VERTEX_CACHE_READ
        yield ChatGenerationChunk(message=AIMessageChunk(
            content="hi", response_metadata={"model_name": self.model},
            usage_metadata={"input_tokens": pooled_input, "output_tokens": OUTPUT_TOKENS,
                            "total_tokens": pooled_input + OUTPUT_TOKENS,
                            "input_token_details": {"cache_read": VERTEX_CACHE_READ, "cache_creation": 0}}))


# ---- Stand-ins that call the real wrapped vendor clients ----

class GeminiStandIn(StandIn):
    """How langchain-google-genai reaches google-genai: ``models.generate_content``."""

    ls_provider: str = "google_genai"
    model: str = calls.GEMINI_MODEL

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        return self._to_result(calls._genai().models.generate_content(model=self.model, contents="hi"))

    def _stream(self, messages, stop=None, run_manager=None, **kwargs):
        for chunk in calls._genai().models.generate_content_stream(model=self.model, contents="hi"):
            yield self._to_chunk(chunk)

    def _to_result(self, response):
        usage = response.usage_metadata
        return _result(response.text, self.model, usage.prompt_token_count, usage.candidates_token_count)

    def _to_chunk(self, chunk):
        usage = chunk.usage_metadata
        return ChatGenerationChunk(message=AIMessageChunk(
            content=chunk.text or "",
            usage_metadata=_usage(usage.prompt_token_count, usage.candidates_token_count) if usage else None))


class GeminiNativeAsync(GeminiStandIn):
    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        return self._to_result(await calls._genai().aio.models.generate_content(model=self.model, contents="hi"))


LITELLM_KWARGS = {"model": calls.LITELLM_MODEL, "messages": [{"role": "user", "content": "hi"}],
                  "mock_response": "hi"}


def _field(obj, name):
    """perplexityai on Python 3.10 leaves ``choices`` entries as plain dicts."""
    return obj.get(name) if isinstance(obj, dict) else getattr(obj, name, None)


def _openai_shaped_result(response, model):
    usage = response.usage
    content = _field(_field(response.choices[0], "message"), "content")
    return _result(content, model, usage.prompt_tokens, usage.completion_tokens)


def _openai_shaped_chunk(chunk):
    usage = getattr(chunk, "usage", None)
    choices = getattr(chunk, "choices", None) or []
    text = (_field(_field(choices[0], "delta"), "content") or "") if choices else ""
    return ChatGenerationChunk(message=AIMessageChunk(
        content=text, usage_metadata=_usage(usage.prompt_tokens, usage.completion_tokens) if usage else None))


class LiteLLMStandIn(StandIn):
    """How langchain-litellm reaches LiteLLM: ``litellm.completion``."""

    ls_provider: str = "litellm"
    model: str = calls.LITELLM_MODEL

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        import litellm
        return _openai_shaped_result(litellm.completion(**LITELLM_KWARGS), self.model)

    def _stream(self, messages, stop=None, run_manager=None, **kwargs):
        import litellm
        for chunk in litellm.completion(**LITELLM_KWARGS, stream=True, stream_options={"include_usage": True}):
            yield _openai_shaped_chunk(chunk)


class LiteLLMNativeAsync(LiteLLMStandIn):
    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        import litellm
        return _openai_shaped_result(await litellm.acompletion(**LITELLM_KWARGS), self.model)


PERPLEXITY_KWARGS = {"model": calls.PERPLEXITY_MODEL, "messages": [{"role": "user", "content": "hi"}]}


class PerplexityStandIn(StandIn):
    """How langchain-perplexity reaches Perplexity: the native client. ChatPerplexity
    sets no ``ls_provider``, so langchain-core derives "perplexity" from the class name."""

    ls_provider: str = "perplexity"
    model: str = calls.PERPLEXITY_MODEL

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        return _openai_shaped_result(
            calls._perplexity_native().chat.completions.create(**PERPLEXITY_KWARGS), self.model)

    def _stream(self, messages, stop=None, run_manager=None, **kwargs):
        for chunk in calls._perplexity_native().chat.completions.create(**PERPLEXITY_KWARGS, stream=True):
            yield _openai_shaped_chunk(chunk)


class PerplexityNativeAsync(PerplexityStandIn):
    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        response = await calls._async_perplexity_native().chat.completions.create(**PERPLEXITY_KWARGS)
        return _openai_shaped_result(response, self.model)


class ChatPerplexity(BaseChatModel):
    """Like langchain-perplexity's: sets no ls_provider, so langchain-core derives it from the class name."""

    model: str = "sonar"

    @property
    def _llm_type(self):
        return "perplexity-chat"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        return _result("hi", self.model, INPUT_TOKENS, OUTPUT_TOKENS)


class ChatLiteLLM(ChatPerplexity):
    """Like langchain-litellm's: sets no ls_provider, so langchain-core derives it from the class name."""

    model: str = "gpt-4o-mini"

    @property
    def _llm_type(self):
        return "litellm-chat"


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


def assert_callback_attribution(record):
    assert record["organization_name"] == "org-callback"
    assert record["trace_id"] == "trace-callback"
    assert (record.get("subscriber") or {}).get("id") == "sub-callback"


class TestProviderLabels:
    @pytest.mark.parametrize("ls_provider, model, provider, model_source", [
        pytest.param("openai", "gpt-4o-mini", "OPENAI", "OPENAI", id="openai"),
        pytest.param("azure", "gpt-4o-mini", "Azure", "OPENAI", id="azure"),
        pytest.param("anthropic", "claude-haiku-4-5", "ANTHROPIC", "ANTHROPIC", id="anthropic"),
        pytest.param("ollama", "llama3", "OLLAMA", "OLLAMA", id="ollama"),
        pytest.param("google_genai", "gemini-2.5-flash", "Google", "GOOGLE", id="gemini"),
        # ChatGoogleGenerativeAI(vertexai=True) still reports google_genai.
        pytest.param("google_genai", "gemini-2.5-flash", "Google", "GOOGLE", id="gemini-on-vertexai"),
        pytest.param("google_vertexai", "gemini-2.5-flash", "Google", "GOOGLE", id="chat-vertexai"),
        pytest.param("amazon_bedrock", "anthropic.claude-3-haiku-20240307-v1:0", "AWS", "ANTHROPIC",
                     id="bedrock-claude"),
        pytest.param("amazon_bedrock", "us.anthropic.claude-3-5-haiku-20241022-v1:0", "AWS", "ANTHROPIC",
                     id="bedrock-claude-inference-profile"),
        pytest.param("amazon_bedrock", "amazon.nova-lite-v1:0", "AWS", "AMAZON", id="bedrock-nova"),
        pytest.param("amazon_bedrock", "us.meta.llama3-1-8b-instruct-v1:0", "AWS", "META", id="bedrock-llama"),
        pytest.param("anthropic-bedrock", "anthropic.claude-3-haiku-20240307-v1:0", "AWS", "ANTHROPIC",
                     id="chat-anthropic-bedrock"),
        pytest.param("cohere", "command-r", "COHERE", "COHERE", id="cohere"),
        pytest.param("huggingface", "HuggingFaceH4/zephyr-7b-beta", "HUGGINGFACE", "HUGGINGFACE",
                     id="huggingface"),
        pytest.param("litellm", "gpt-4o-mini", "LITELLM", "LITELLM", id="litellm"),
        pytest.param("perplexity", "sonar", "PERPLEXITY", "PERPLEXITY", id="perplexity"),
        # The Anthropic wrap labels an AnthropicVertex call ANTHROPIC as well.
        pytest.param("anthropicvertex", "claude-sonnet-4-5", "ANTHROPIC", "ANTHROPIC", id="chat-anthropic-vertex"),
    ])
    def test_each_readme_provider_is_recorded_under_its_own_label(
            self, records, ls_provider, model, provider, model_source):
        assert invoke(StandIn(ls_provider=ls_provider, model=model, callbacks=[attributed_handler()])) == "hi"
        sent = records()
        assert len(sent) == 1, sent
        assert (sent[0]["provider"], sent[0]["model_source"]) == (provider, model_source)
        assert sent[0]["model"] == model
        assert sent[0]["transaction_id"].startswith("langchain-")
        assert_callback_attribution(sent[0])

    def test_a_streamed_claude_on_vertex_call_is_labelled_anthropic_and_counts_the_cache_once(self, records):
        """BACK-4222: ChatAnthropicVertex pools the cache read into a streamed input_tokens."""
        assert stream(VertexClaudeStandIn(callbacks=[attributed_handler()])) == "hi"
        sent = records()
        assert len(sent) == 1, sent
        assert (sent[0]["provider"], sent[0]["model_source"]) == ("ANTHROPIC", "ANTHROPIC")
        assert sent[0]["input_token_count"] == INPUT_TOKENS
        assert sent[0]["cache_read_token_count"] == VERTEX_CACHE_READ
        assert sent[0]["is_streamed"] is True

    @pytest.mark.parametrize("model_class, provider", [
        pytest.param(ChatPerplexity, "PERPLEXITY", id="perplexity"),
        pytest.param(ChatLiteLLM, "LITELLM", id="litellm"),
    ])
    def test_a_provider_langchain_core_names_from_the_class_is_recorded_under_its_label(
            self, records, model_class, provider):
        assert invoke(model_class(callbacks=[attributed_handler()])) == "hi"
        sent = records()
        assert len(sent) == 1, sent
        assert (sent[0]["provider"], sent[0]["model_source"]) == (provider, provider)
        assert_callback_attribution(sent[0])

    def test_a_chat_mistralai_call_is_recorded_under_mistral(self, records):
        """BACK-4258: a real ChatMistralAI call is recorded once under Mistral, not OpenAI."""
        assert invoke(mistral.chat_mistralai([attributed_handler()])) == mistral.ANSWER
        sent = records()
        assert len(sent) == 1, sent
        assert (sent[0]["provider"], sent[0]["model_source"]) == ("MISTRAL", "MISTRAL")
        assert sent[0]["transaction_id"].startswith("langchain-")
        assert (sent[0]["input_token_count"], sent[0]["output_token_count"]) == (
            mistral.INPUT_TOKENS, mistral.OUTPUT_TOKENS)
        assert_callback_attribution(sent[0])

    def test_an_openai_claim_during_a_chat_mistralai_call_does_not_hide_its_record(self, records):
        """BACK-4258: an OpenAI call metered while ChatMistralAI waits on Mistral does not stand in
        for the Mistral call, which no transport wrap meters."""
        model = mistral.chat_mistralai([attributed_handler()], during_request=lambda: claim_call_for_transport(OPENAI))
        assert invoke(model) == mistral.ANSWER
        assert [(record["provider"], record["model_source"]) for record in records()] == [("MISTRAL", "MISTRAL")]


VENDORS = [
    pytest.param(GeminiStandIn, GeminiNativeAsync, "Google", id="gemini"),
    pytest.param(LiteLLMStandIn, LiteLLMNativeAsync, "LITELLM", id="litellm"),
    pytest.param(PerplexityStandIn, PerplexityNativeAsync, "PERPLEXITY", id="perplexity-native"),
]


class TestWrappedVendorCalls:
    @pytest.mark.parametrize("model_class, native_async_class, provider", VENDORS)
    @pytest.mark.parametrize("mode", ["invoke", "ainvoke-gather-child", "ainvoke-executor-copy", "stream"])
    def test_a_wrapped_vendor_call_is_recorded_once_with_the_callback_attribution(
            self, records, model_class, native_async_class, provider, mode):
        build = native_async_class if mode == "ainvoke-gather-child" else model_class
        call = {"invoke": invoke, "stream": stream}.get(mode, ainvoke)
        assert call(build(callbacks=[attributed_handler()]))
        sent = records()
        assert len(sent) == 1, sent
        assert not sent[0]["transaction_id"].startswith("langchain-")
        assert sent[0]["provider"] == provider
        assert_callback_attribution(sent[0])

    def test_a_wrap_with_no_metadata_on_ainvoke_is_recorded_once(self, records):
        assert ainvoke(GeminiNativeAsync(callbacks=[UnifiedReveniumCallbackHandler()])) == "hi"
        sent = records()
        assert len(sent) == 1, sent
        assert sent[0]["provider"] == "Google"
        assert not sent[0]["transaction_id"].startswith("langchain-")

    def test_a_call_selective_metering_leaves_unmetered_sends_no_record(self, records, monkeypatch):
        """BACK-4320: neither the transport nor the callback records a call outside a metered function."""
        monkeypatch.setenv("REVENIUM_SELECTIVE_METERING", "true")
        assert invoke(GeminiStandIn(callbacks=[attributed_handler()])) == "hi"
        assert records() == []

    def test_a_failed_google_submission_keeps_the_callback_record(self, records):
        """BACK-3989: a claim made after a failed submission would lose the call's usage."""
        with patch("revenium_middleware.google.google_ai.middleware.create_metering_call",
                   side_effect=RuntimeError("submission failed")):
            assert invoke(GeminiStandIn(callbacks=[attributed_handler()])) == "hi"
        sent = records()
        assert len(sent) == 1, sent
        assert sent[0]["transaction_id"].startswith("langchain-")
        assert sent[0]["provider"] == "Google"
        assert_callback_attribution(sent[0])

    def test_a_callback_only_call_and_a_metered_call_in_one_task_are_both_recorded(self, records):
        async def one_task():
            await StandIn(ls_provider="cohere", callbacks=[attributed_handler()]).ainvoke("hi")
            await GeminiNativeAsync(callbacks=[attributed_handler()]).ainvoke("hi")
            await StandIn(ls_provider="cohere", callbacks=[attributed_handler()]).ainvoke("hi")

        asyncio.run(one_task())
        sent = records()
        assert sorted((r["provider"], r["transaction_id"].startswith("langchain-")) for r in sent) == [
            ("COHERE", True), ("COHERE", True), ("Google", False)], sent

    def test_a_batched_agenerate_gives_one_record_per_message_list(self, records):
        model = GeminiNativeAsync(callbacks=[attributed_handler()])
        asyncio.run(model.agenerate([[HumanMessage("a")], [HumanMessage("b")]]))
        sent = records()
        assert len(sent) == 2, sent
        assert all(not r["transaction_id"].startswith("langchain-") for r in sent)
        assert all(r["provider"] == "Google" for r in sent)


def _llm_result():
    message = AIMessage(content="hi", usage_metadata=_usage(INPUT_TOKENS, OUTPUT_TOKENS))
    return LLMResult(generations=[[ChatGeneration(message=message)]])


def _start_run(handler, ls_provider):
    run_id = uuid.uuid4()
    handler.on_chat_model_start({"id": ["stand_in", "ChatStandIn"]}, [[HumanMessage("hi")]],
                                run_id=run_id, metadata={"ls_provider": ls_provider})
    return run_id


def _in_gather_child(transport_calls):
    """Runs ``transport_calls`` the way ``agenerate`` runs a model call: in a
    child task created by ``asyncio.gather``, in a copy of this context."""
    async def child():
        transport_calls()

    async def parent():
        await asyncio.gather(child())

    contextvars.copy_context().run(asyncio.run, parent())


class TestCallOwnershipAcrossTasks:
    def test_a_claim_inside_ainvoke_is_seen_by_the_callback(self, records):
        handler = attributed_handler()
        run_id = _start_run(handler, "google_genai")

        def gemini_transport():
            take_callback_metadata("google")
            claim_call_for_transport("google")

        _in_gather_child(gemini_transport)
        handler.on_llm_end(_llm_result(), run_id=run_id)
        assert records() == []

    def test_a_nested_wrap_does_not_erase_the_outer_claim(self, records):
        handler = attributed_handler()
        run_id = _start_run(handler, "litellm")

        def litellm_over_the_openai_client():
            take_callback_metadata("litellm")
            take_callback_metadata(OPENAI)
            claim_call_for_transport("litellm")

        _in_gather_child(litellm_over_the_openai_client)
        handler.on_llm_end(_llm_result(), run_id=run_id)
        assert records() == []


CALLBACK_RECORDS = 20


def test_callback_records_get_full_random_transaction_ids(records, monkeypatch):
    counter = itertools.count(1)
    monkeypatch.setattr(uuid, "uuid4", lambda: uuid.UUID(int=next(counter)))

    for _ in range(CALLBACK_RECORDS):
        assert invoke(StandIn(callbacks=[attributed_handler()])) == "hi"

    ids = [record["transaction_id"] for record in records()]
    assert len(set(ids)) == CALLBACK_RECORDS
    assert all(re.match(r"^langchain-[0-9a-f]{32}$", transaction_id) for transaction_id in ids)
    assert set(ids) <= {f"langchain-{uuid.UUID(int=i).hex}" for i in range(1, 10_000)}


class StopStandIn(StandIn):
    """A callback-only chat model reporting its stop reason where the given vendor's LangChain class does."""

    response_metadata: dict = {}
    generation_info: dict = {}
    llm_output: dict = {}

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        message = AIMessage(content="hi", response_metadata={"model_name": self.model, **self.response_metadata},
                            usage_metadata=_usage(INPUT_TOKENS, OUTPUT_TOKENS))
        generation = ChatGeneration(message=message, generation_info=self.generation_info or None)
        return ChatResult(generations=[generation], llm_output=self.llm_output or None)

    def _stream(self, messages, stop=None, run_manager=None, **kwargs):
        yield ChatGenerationChunk(message=AIMessageChunk(content="h", response_metadata={"model_name": self.model}))
        yield ChatGenerationChunk(message=AIMessageChunk(
            content="i", response_metadata=self.response_metadata, usage_metadata=_usage(INPUT_TOKENS, OUTPUT_TOKENS)))


def _stop_stand_in(ls_provider, model="claude-haiku-4-5", **shape):
    return StopStandIn(ls_provider=ls_provider, model=model, callbacks=[attributed_handler()], **shape)


def _sent_stop_reason(records, call, model):
    assert call(model) == "hi"
    sent = records()
    assert len(sent) == 1, sent
    return sent[0]["stop_reason"]


TRUNCATED_SHAPES = [
    pytest.param(invoke, "anthropic", "claude-haiku-4-5", {"response_metadata": {"stop_reason": "max_tokens"}},
                 id="anthropic"),
    pytest.param(invoke, "anthropicvertex", "claude-sonnet-4-5", {"llm_output": {"stop_reason": "max_tokens"}},
                 id="vertex-claude-invoke"),
    pytest.param(stream, "anthropicvertex", "claude-sonnet-4-5", {"response_metadata": {"stop_reason": "max_tokens"}},
                 id="vertex-claude-stream"),
    pytest.param(invoke, "google_vertexai", "gemini-2.5-flash", {"generation_info": {"finish_reason": "MAX_TOKENS"}},
                 id="vertex-gemini"),
    pytest.param(invoke, "amazon_bedrock", "anthropic.claude-3-haiku-20240307-v1:0",
                 {"llm_output": {"stop_reason": "max_tokens"}}, id="chat-bedrock"),
    pytest.param(invoke, "amazon_bedrock", "us.anthropic.claude-haiku-4-5-20251001-v1:0",
                 {"response_metadata": {"stopReason": "max_tokens", "model_provider": "bedrock_converse"}},
                 id="bedrock-converse"),
    pytest.param(invoke, "cohere", "command-r", {"generation_info": {"finish_reason": "MAX_TOKENS"}}, id="cohere"),
    pytest.param(invoke, "huggingface", "HuggingFaceH4/zephyr-7b-beta", {"generation_info": {"finish_reason": "length"}},
                 id="huggingface"),
    pytest.param(invoke, "ollama", "llama3", {"generation_info": {"done_reason": "length"}}, id="ollama"),
]


class TestStopReasons:
    """BACK-4226: a callback record carries the stop reason the vendor's LangChain class reported."""

    @pytest.mark.parametrize("call, ls_provider, model, shape", TRUNCATED_SHAPES)
    def test_a_truncated_answer_is_recorded_with_the_token_limit_stop_reason(
            self, records, call, ls_provider, model, shape):
        assert _sent_stop_reason(records, call, _stop_stand_in(ls_provider, model, **shape)) == "TOKEN_LIMIT"

    @pytest.mark.parametrize("shape", [
        pytest.param({"response_metadata": {"stop_reason": "end_turn"}}, id="anthropic-end-turn"),
        pytest.param({"generation_info": {"finish_reason": "STOP"}}, id="gemini-stop"),
        pytest.param({"generation_info": {"finish_reason": "COMPLETE"}}, id="cohere-complete"),
        pytest.param({"generation_info": {"finish_reason": "stop"}}, id="huggingface-stop"),
        pytest.param({}, id="no-stop-reason"),
    ])
    def test_a_normal_finish_is_still_recorded_as_end(self, records, shape):
        assert _sent_stop_reason(records, invoke, _stop_stand_in("cohere", **shape)) == "END"

    def test_a_tool_call_finish_matches_the_anthropic_wrap(self, records):
        shape = {"response_metadata": {"stop_reason": "tool_use"}}
        assert _sent_stop_reason(records, invoke, _stop_stand_in("anthropic", **shape)) == "END_SEQUENCE"

    def test_a_given_stop_reason_overrides_the_responses_finish_reason(self, records):
        import datetime
        from types import SimpleNamespace
        from revenium_middleware.openai.middleware import OperationType, create_metering_call

        response = SimpleNamespace(
            id="chatcmpl-stop-reason", model="gpt-4o-mini", system_fingerprint=None,
            usage=SimpleNamespace(prompt_tokens=INPUT_TOKENS, completion_tokens=OUTPUT_TOKENS,
                                  total_tokens=INPUT_TOKENS + OUTPUT_TOKENS),
            choices=[SimpleNamespace(finish_reason="length")],
        )

        create_metering_call(response, OperationType.CHAT, datetime.datetime.now(datetime.timezone.utc), {},
                             stop_reason="END")

        sent = records()
        assert [record["stop_reason"] for record in sent] == ["END"], sent


DECISIONS_MODEL = "gpt-6-luna"
DECISIONS_SERVED_MODEL = "gpt-6-luna-2026-10-01"


def _decision(input_tokens=120, output_tokens=4, answers=True):
    from types import SimpleNamespace

    response = SimpleNamespace(model=DECISIONS_SERVED_MODEL,
                               usage=SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens))
    if answers:
        response.answers = {}
    return response


class DecisionsStandIn(Runnable):
    """Reports to callbacks exactly as langchain-openai's OpenAIDecisions does:
    a chain run with run_type "llm", the ls_* metadata, and usage on the output."""

    def __init__(self, response=None, error=None):
        self.response = response if response is not None else _decision()
        self.error = error

    def _traced_config(self, config):
        config = ensure_config(config)
        config["metadata"] = {**(config.get("metadata") or {}), "ls_provider": "openai",
                              "ls_model_name": DECISIONS_MODEL, "ls_model_type": "chat"}
        return config

    def _decide(self, request):
        if self.error is not None:
            raise self.error
        return self.response

    async def _adecide(self, request):
        return self._decide(request)

    def invoke(self, input, config=None, **kwargs):
        return self._call_with_config(self._decide, input, self._traced_config(config), run_type="llm")

    async def ainvoke(self, input, config=None, **kwargs):
        return await self._acall_with_config(self._adecide, input, self._traced_config(config), run_type="llm")


class TestDecisionsRuns:
    """BACK-4236: an OpenAI Decisions request made through LangChain is recorded once."""

    def test_a_decisions_run_is_recorded_once_with_its_token_counts(self, records):
        DecisionsStandIn().invoke({"input": "hi"}, config={"callbacks": [attributed_handler()]})

        sent = records()
        assert len(sent) == 1, sent
        assert (sent[0]["input_token_count"], sent[0]["output_token_count"]) == (120, 4)
        assert sent[0]["model"] == DECISIONS_SERVED_MODEL
        assert sent[0]["provider"] == "OPENAI"
        assert_callback_attribution(sent[0])

    def test_an_async_decisions_run_is_recorded_once(self, records):
        asyncio.run(DecisionsStandIn().ainvoke({"input": "hi"}, config={"callbacks": [attributed_handler()]}))

        sent = records()
        assert len(sent) == 1, sent
        assert (sent[0]["input_token_count"], sent[0]["output_token_count"]) == (120, 4)

    @pytest.mark.parametrize("run", [
        pytest.param(lambda config: DecisionsStandIn().invoke({"input": "hi"}, config=config), id="invoke"),
        pytest.param(lambda config: asyncio.run(DecisionsStandIn().ainvoke({"input": "hi"}, config=config)),
                     id="ainvoke"),
    ])
    def test_a_decisions_run_outside_a_metered_function_sends_no_record(self, records, monkeypatch, run):
        """BACK-4320: selective metering leaves a Decisions run outside a @revenium_meter function unrecorded."""
        monkeypatch.setenv("REVENIUM_SELECTIVE_METERING", "true")
        handler = attributed_handler()

        run({"callbacks": [handler]})

        assert records() == []
        assert handler._active_runs == {}

    def test_a_decisions_run_inside_a_metered_function_is_recorded_once(self, records, monkeypatch):
        monkeypatch.setenv("REVENIUM_SELECTIVE_METERING", "true")

        @revenium_meter()
        def handle_request():
            DecisionsStandIn().invoke({"input": "hi"}, config={"callbacks": [attributed_handler()]})

        handle_request()

        sent = records()
        assert len(sent) == 1, sent
        assert (sent[0]["input_token_count"], sent[0]["output_token_count"]) == (120, 4)

    def test_a_decisions_run_inside_revenium_metadata_keeps_the_request_fields(self, records):
        handler = UnifiedReveniumCallbackHandler(usage_metadata={"organizationName": "org-callback"})

        @revenium_metadata(subscriber={"id": "sub-request"})
        def handle_request():
            DecisionsStandIn().invoke({"input": "hi"}, config={"callbacks": [handler]})

        handle_request()

        sent = records()
        assert len(sent) == 1, sent
        assert (sent[0].get("subscriber") or {}).get("id") == "sub-request"
        assert sent[0]["organization_name"] == "org-callback"

    def test_an_ordinary_chain_run_sends_no_record(self, records):
        handler = attributed_handler()
        RunnableLambda(lambda request: _decision()).invoke({"input": "hi"}, config={"callbacks": [handler]})

        assert records() == []
        assert handler._active_runs == {}

    def test_an_llm_typed_chain_without_decision_shape_sends_no_record(self, records):
        handler = attributed_handler()
        DecisionsStandIn(response=_decision(answers=False)).invoke({"input": "hi"}, config={"callbacks": [handler]})

        assert records() == []
        assert handler._active_runs == {}

    def test_a_decisions_run_with_no_usage_is_recorded_with_zero_tokens(self, records):
        DecisionsStandIn(response=_decision(None, None)).invoke(
            {"input": "hi"}, config={"callbacks": [attributed_handler()]})

        sent = records()
        assert len(sent) == 1, sent
        assert (sent[0]["input_token_count"], sent[0]["output_token_count"]) == (0, 0)

    def test_a_failed_decisions_run_leaves_no_tracked_run(self, records):
        handler = attributed_handler()
        with pytest.raises(RuntimeError, match="decisions refused"):
            DecisionsStandIn(error=RuntimeError("decisions refused")).invoke(
                {"input": "hi"}, config={"callbacks": [handler]})

        assert records() == []
        assert handler._active_runs == {}

    def test_router_middleware_decisions_and_routed_chat_are_both_recorded(self, records):
        httpx = pytest.importorskip("httpx")
        langchain_openai = pytest.importorskip("langchain_openai")
        decisions = pytest.importorskip("langchain_openai.decisions")
        agents = pytest.importorskip("langchain.agents")
        middleware = pytest.importorskip("langchain_openai.middleware")
        import json

        upstream_paths = []

        def answer(request):
            upstream_paths.append(request.url.path)
            if request.url.path.endswith("/decisions"):
                return httpx.Response(200, json={
                    "model": DECISIONS_SERVED_MODEL,
                    "answers": [{"name": "model_route", "type": "choice", "choice": "fast", "confidence": 0.9,
                                 "probabilities": [{"value": "fast", "probability": 0.95},
                                                   {"value": "deep", "probability": 0.05}]}],
                    "usage": {"input_tokens": 120, "output_tokens": 4},
                })
            body = json.loads(request.content)
            return httpx.Response(200, json={
                "id": f"chatcmpl-{uuid.uuid4().hex}", "object": "chat.completion", "created": 1,
                "model": body["model"],
                "choices": [{"index": 0, "finish_reason": "stop",
                             "message": {"role": "assistant", "content": "routed"}}],
                "usage": {"prompt_tokens": INPUT_TOKENS, "completion_tokens": OUTPUT_TOKENS,
                          "total_tokens": INPUT_TOKENS + OUTPUT_TOKENS},
            })

        def client():
            return dict(api_key="sk-test", base_url="https://api.openai.test/v1", max_retries=0,
                        http_client=httpx.Client(transport=httpx.MockTransport(answer)))

        def chat(model):
            return langchain_openai.ChatOpenAI(model=model, **client())

        router = middleware.OpenAIModelRouterMiddleware(
            choices={"fast": middleware.ModelChoice(model=chat("gpt-4o-mini"), criteria="Simple tasks."),
                     "deep": middleware.ModelChoice(model=chat("gpt-4o"), criteria="Hard tasks.")},
            instructions="Pick the cheapest model that fits.",
            model=decisions.OpenAIDecisions(model=DECISIONS_MODEL, **client()),
        )
        agent = agents.create_agent(chat("gpt-4o"), middleware=[router])

        agent.invoke({"messages": [HumanMessage("hi")]}, config={"callbacks": [attributed_handler()]})

        assert upstream_paths == ["/v1/decisions", "/v1/chat/completions"]
        sent = records()
        assert sorted(record["model"] for record in sent) == sorted([DECISIONS_SERVED_MODEL, "gpt-4o-mini"]), sent
        for record in sent:
            assert_callback_attribution(record)


_OPENAI_CHAT_WRAP = "openai.resources.chat.completions.Completions.create"
_ANTHROPIC_WRAP = "anthropic.resources.messages.messages.Messages.create"
_needs_langchain_aws = pytest.mark.skipif(importlib.util.find_spec("langchain_aws") is None,
                                          reason="the dev-langchain-aws extra is not installed")


def _labels(provider, model_source):
    return {"provider": provider, "model_source": model_source}


LANGCHAIN_CLASSES = [
    pytest.param("langchain_openai", "ChatOpenAI", {"model": "gpt-4o-mini"}, "openai",
                 {"labels": _labels("OPENAI", "OPENAI"), "scope": frozenset({OPENAI}),
                  "cap_wraps": (_OPENAI_CHAT_WRAP,)}, id="openai"),
    pytest.param("langchain_openai", "AzureChatOpenAI", {"model": "gpt-4o-mini"}, "azure",
                 {"labels": _labels("Azure", "OPENAI"), "scope": frozenset({OPENAI}),
                  "cap_wraps": (_OPENAI_CHAT_WRAP,)}, id="azure"),
    pytest.param("langchain_anthropic", "ChatAnthropic", {"model": "claude-haiku-4-5"}, "anthropic",
                 {"labels": _labels("ANTHROPIC", "ANTHROPIC"), "scope": frozenset({ANTHROPIC}),
                  "cap_wraps": (_ANTHROPIC_WRAP,)}, id="anthropic"),
    pytest.param("langchain_mistralai", "ChatMistralAI", {"model": mistral.MISTRAL_MODEL}, "mistral",
                 {"labels": _labels("MISTRAL", "MISTRAL"), "scope": frozenset(), "cap_wraps": None}, id="mistral"),
    pytest.param("langchain_perplexity", "ChatPerplexity", {"model": "sonar"}, "perplexity",
                 {"labels": _labels("PERPLEXITY", "PERPLEXITY"), "scope": frozenset({PERPLEXITY, OPENAI}),
                  "cap_wraps": (f"perplexity:{_OPENAI_CHAT_WRAP}",
                                "perplexity.resources.chat.completions.CompletionsResource.create",
                                "perplexity.resources.responses.responses.ResponsesResource.create")},
                 id="perplexity"),
    pytest.param("langchain_cohere", "ChatCohere", {"model": "command-r"}, "cohere",
                 {"labels": _labels("COHERE", "COHERE"), "scope": frozenset(), "cap_wraps": None}, id="cohere"),
    pytest.param("langchain_huggingface", "ChatHuggingFace", {"model_id": "HuggingFaceH4/zephyr-7b-beta"},
                 "huggingface",
                 {"labels": _labels("HUGGINGFACE", "HUGGINGFACE"), "scope": frozenset(), "cap_wraps": None},
                 id="huggingface"),
    pytest.param("langchain_aws", "ChatBedrock", {"model_id": "anthropic.claude-3-haiku-20240307-v1:0"},
                 "amazon_bedrock",
                 {"labels": _labels("AWS", "ANTHROPIC"), "scope": frozenset({BEDROCK}), "cap_wraps": None},
                 marks=_needs_langchain_aws, id="bedrock"),
    pytest.param("langchain_aws", "ChatBedrockConverse", {"model": "anthropic.claude-3-haiku-20240307-v1:0"},
                 "amazon_bedrock",
                 {"labels": _labels("AWS", "ANTHROPIC"), "scope": frozenset({BEDROCK}), "cap_wraps": None},
                 marks=_needs_langchain_aws, id="bedrock-converse"),
    pytest.param("langchain_aws", "ChatAnthropicBedrock", {"model": "anthropic.claude-3-haiku-20240307-v1:0"},
                 "anthropic-bedrock",
                 {"labels": _labels("AWS", "ANTHROPIC"), "scope": frozenset({ANTHROPIC}),
                  "cap_wraps": (_ANTHROPIC_WRAP,)}, marks=_needs_langchain_aws, id="anthropic-bedrock"),
    pytest.param("langchain_aws", "ChatAnthropicMantle", {"model": "claude-haiku-4-5"}, "anthropic-mantle",
                 {"cap_wraps": (_ANTHROPIC_WRAP,)}, marks=_needs_langchain_aws, id="anthropic-mantle"),
    pytest.param("langchain_aws", "ChatOpenAIMantle", {"model": "gpt-oss-120b"}, "openai-mantle",
                 {"cap_wraps": (_OPENAI_CHAT_WRAP,)}, marks=_needs_langchain_aws, id="openai-mantle"),
]


@pytest.mark.parametrize("package, class_name, fields, ls_provider, relied_on", LANGCHAIN_CLASSES)
def test_each_langchain_class_reports_a_provider_the_sdk_lists(package, class_name, fields, ls_provider, relied_on):
    """BACK-4258: the SDK listed ChatMistralAI as "mistralai" while the package reports "mistral".
    Each partner class CI installs is read without validators, so nothing reaches the network."""
    model_class = getattr(importlib.import_module(package), class_name)
    assert model_class.model_construct(**fields)._get_ls_params()["ls_provider"] == ls_provider
    listed = {
        "labels": _model_call._PROVIDER_LABELS.get(ls_provider),
        "scope": _model_call._TRANSPORT_SCOPES.get(ls_provider),
        "cap_wraps": _model_call._CAP_CHECKING_WRAPS.get(ls_provider),
    }
    assert {table: listed[table] for table in relied_on} == relied_on


STARTUP_ORGANIZATION = {"organizationName": "org-startup"}
REQUEST_PERSON = {"id": "user-42", "email": "u42@example.com"}

CALLBACK_RECORDED_MODELS = [
    pytest.param(lambda: mistral.chat_mistralai(None), mistral.ANSWER, id="chat-mistralai"),
    pytest.param(StandIn, "hi", id="cohere-stand-in"),
]


def _attached_at_startup(build, **usage_metadata):
    model = build()
    attach_to(model, usage_metadata=usage_metadata or STARTUP_ORGANIZATION)
    return model


def _stream_started_inside_revenium_metadata(model):
    @revenium_metadata(subscriber=REQUEST_PERSON)
    def start():
        chunks = model.stream("hi")
        return next(chunks).text, chunks

    first, rest = start()
    return first + "".join(chunk.text for chunk in rest)


def _astream_started_inside_revenium_metadata(model):
    @revenium_metadata(subscriber=REQUEST_PERSON)
    async def start():
        chunks = model.astream("hi").__aiter__()
        return (await chunks.__anext__()).text, chunks

    async def drain():
        first, rest = await start()
        return first + "".join([chunk.text async for chunk in rest])

    return asyncio.run(drain())


PER_CALL_PERSON = {"id": "per-call"}


def _per_call_handler():
    return UnifiedReveniumCallbackHandler(usage_metadata={"subscriber": PER_CALL_PERSON})


def _revenium_handlers(binding):
    return [handler for handler in [*binding.bound.callbacks, *binding.config["callbacks"]]
            if isinstance(handler, UnifiedReveniumCallbackHandler)]


class FailingStandIn(StandIn):
    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        raise RuntimeError("provider refused")


class TestRequestMetadataOnCallbackRecords:
    """BACK-4316: a model attached once at startup, with the person named per
    request by @revenium_metadata, as the README shows."""

    @pytest.mark.parametrize("call", [invoke, ainvoke, stream, astream])
    @pytest.mark.parametrize("build, answer", CALLBACK_RECORDED_MODELS)
    def test_a_callback_recorded_call_inside_revenium_metadata_keeps_the_request_person(
            self, records, build, answer, call):
        model = _attached_at_startup(build)

        @revenium_metadata(subscriber=REQUEST_PERSON)
        def handle_request():
            return call(model)

        assert handle_request() == answer
        sent = records()
        assert len(sent) == 1, sent
        assert sent[0].get("subscriber") == REQUEST_PERSON
        assert sent[0]["organization_name"] == "org-startup"
        assert sent[0]["transaction_id"].startswith("langchain-")

    @pytest.mark.parametrize("call", [_stream_started_inside_revenium_metadata,
                                      _astream_started_inside_revenium_metadata])
    def test_a_stream_started_inside_revenium_metadata_and_drained_outside_keeps_the_request_person(
            self, records, call):
        assert call(_attached_at_startup(lambda: mistral.chat_mistralai(None))) == mistral.ANSWER
        sent = records()
        assert len(sent) == 1, sent
        assert sent[0].get("subscriber") == REQUEST_PERSON
        assert sent[0]["organization_name"] == "org-startup"

    def test_the_attached_handlers_field_wins_over_the_decorators_as_on_chat_openai(self, records):
        model = _attached_at_startup(lambda: mistral.chat_mistralai(None))

        @revenium_metadata(organization_name="org-decorator", subscriber=REQUEST_PERSON)
        def handle_request():
            return invoke(model)

        assert handle_request() == mistral.ANSWER
        sent = records()
        assert len(sent) == 1, sent
        assert sent[0]["organization_name"] == "org-startup"
        assert sent[0].get("subscriber") == REQUEST_PERSON

    @pytest.mark.parametrize("call", [invoke, ainvoke, stream, astream])
    def test_an_attached_and_a_per_call_handler_record_a_callback_recorded_call_once(self, records, call):
        """BACK-4318."""
        model = _attached_at_startup(lambda: mistral.chat_mistralai(None)).with_config(callbacks=[_per_call_handler()])
        assert call(model) == mistral.ANSWER
        sent = records()
        assert len(sent) == 1, sent
        assert sent[0]["organization_name"] == "org-startup"
        assert not sent[0].get("subscriber")
        assert (sent[0]["input_token_count"], sent[0]["output_token_count"]) == (
            mistral.INPUT_TOKENS, mistral.OUTPUT_TOKENS)

    def test_a_sibling_handler_records_the_call_when_the_first_handoff_fails(self, records):
        model = _attached_at_startup(lambda: mistral.chat_mistralai(None)).with_config(callbacks=[_per_call_handler()])
        handlers = _revenium_handlers(model)
        original = handlers[0]._create_metering_call
        handoffs = itertools.count()

        def fail_the_first_handoff(*args, **kwargs):
            if next(handoffs) == 0:
                raise RuntimeError("handoff failed")
            return original(*args, **kwargs)

        for handler in handlers:
            handler._create_metering_call = fail_the_first_handoff
        assert invoke(model) == mistral.ANSWER
        assert len(records()) == 1

    def test_a_sibling_handler_records_the_call_when_the_first_handoff_is_skipped(self, records):
        model = _attached_at_startup(lambda: mistral.chat_mistralai(None)).with_config(callbacks=[_per_call_handler()])
        handlers = _revenium_handlers(model)
        original = handlers[0]._create_metering_call
        handoffs = itertools.count()

        def skip_the_first_handoff(*args, **kwargs):
            if next(handoffs) == 0:
                return None
            return original(*args, **kwargs)

        for handler in handlers:
            handler._create_metering_call = skip_the_first_handoff
        assert invoke(model) == mistral.ANSWER
        assert len(records()) == 1

    def test_a_per_call_handler_is_used_when_the_attached_one_has_no_fields(self, records):
        model = mistral.chat_mistralai(None)
        attach_to(model)
        assert invoke(model.with_config(callbacks=[_per_call_handler()])) == mistral.ANSWER
        sent = records()
        assert len(sent) == 1, sent
        assert sent[0].get("subscriber") == PER_CALL_PERSON

    def test_a_batch_with_two_handlers_records_each_prompt_once(self, records):
        model = _attached_at_startup(lambda: mistral.chat_mistralai(None)).with_config(callbacks=[_per_call_handler()])
        assert [answer.text for answer in model.batch(["a", "b", "c"])] == [mistral.ANSWER] * 3
        sent = records()
        assert len(sent) == 3, sent
        assert [record["organization_name"] for record in sent] == ["org-startup"] * 3

    def test_three_revenium_handlers_on_one_call_record_it_once(self, records):
        model = _attached_at_startup(lambda: mistral.chat_mistralai(None)).with_config(callbacks=[_per_call_handler()])
        chain = RunnableLambda(lambda question: question) | model
        outer = UnifiedReveniumCallbackHandler(usage_metadata={"traceId": "trace-outer"})
        assert chain.invoke("hi", config={"callbacks": [outer]}).text == mistral.ANSWER
        sent = records()
        assert len(sent) == 1, sent
        assert sent[0]["organization_name"] == "org-startup"

    def test_two_calls_in_a_row_with_two_handlers_record_two(self, records):
        model = _attached_at_startup(lambda: mistral.chat_mistralai(None)).with_config(callbacks=[_per_call_handler()])
        assert [invoke(model), invoke(model)] == [mistral.ANSWER] * 2
        assert len(records()) == 2

    def test_a_refused_or_failed_run_leaves_no_group_behind(self, records):
        def refuse_the_startup_organization(usage_metadata, **_keywords):
            if usage_metadata.get("organizationName") == "org-startup":
                raise BudgetExceededError(message="budget breached", rule_name="startup-org-cap")

        def assert_nothing_left_behind(model):
            gc.collect()
            assert [handler._active_runs for handler in _revenium_handlers(model)] == [{}, {}]
            assert len(unified_handler._run_groups) == 0

        refused = _attached_at_startup(lambda: mistral.chat_mistralai(None)).with_config(
            callbacks=[_per_call_handler()])
        with patch.object(unified_handler, "check_enforcement", side_effect=refuse_the_startup_organization):
            for _ in range(2):
                with pytest.raises(BudgetExceededError):
                    refused.invoke("hi")
                assert_nothing_left_behind(refused)

        failing = _attached_at_startup(FailingStandIn).with_config(callbacks=[_per_call_handler()])
        with pytest.raises(RuntimeError):
            failing.invoke("hi")
        assert_nothing_left_behind(failing)
        assert records() == []
