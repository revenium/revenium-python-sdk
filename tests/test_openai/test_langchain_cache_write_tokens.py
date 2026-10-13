"""The LangChain callback records for ChatOpenAI and AzureChatOpenAI take
OpenAI cache writes out of the input count and report them as cache-creation
tokens (BACK-3998). Kept apart from the OpenAI entry-point cases so an install
without langchain-openai still runs those.
"""
import time
from types import SimpleNamespace

import pytest

from .test_cache_write_tokens import (  # noqa: F401  served_usage and served_model are fixtures
    GATEWAY_CLAUDE_MODEL,
    LITELLM_CLAUDE_CHAT_USAGE,
    PROMPT,
    WRITING_CHAT_USAGE,
    WRITTEN,
    served_model,
    served_usage,
    the_one_record,
)
from .test_openai_sdk_entry_points import FakeOpenAI, payloads, sync_chat_create  # noqa: F401  payloads is a fixture

langchain_openai = pytest.importorskip("langchain_openai")

from langchain_core.outputs import LLMResult  # noqa: E402
from langchain_openai.chat_models.base import _create_usage_metadata  # noqa: E402

from revenium_middleware._core.metering import current_client  # noqa: E402
from revenium_middleware.openai.langchain.unified_handler import UnifiedReveniumCallbackHandler  # noqa: E402

LANGCHAIN_MODEL = "gpt-5.6-terra"


def _run_info(ls_provider, class_name="ChatOpenAI", is_streaming=False, model=LANGCHAIN_MODEL):
    return {
        "serialized": {"id": ["langchain", "chat_models", "openai", class_name],
                       "kwargs": {"model_name": model}},
        "invocation_params": {},
        "run_metadata": {"ls_provider": ls_provider} if ls_provider else {},
        "usage_metadata": {},
        "start_time": time.time(),
        "metering_client": current_client(),
        "is_streaming": is_streaming,
    }


def _meter_through_the_callback(response, run_info):
    handler = UnifiedReveniumCallbackHandler()
    usage_data = handler._extract_usage_from_response(response)
    handler._process_metering_call(usage_data, run_info)


def _usage_metadata_only_result():
    """What a streamed or Responses API ChatOpenAI call leaves on the result:
    no ``llm_output``, only the message's ``usage_metadata``, built by
    langchain-openai from OpenAI's raw usage."""
    usage_metadata = _create_usage_metadata(WRITING_CHAT_USAGE)
    return SimpleNamespace(generations=[[SimpleNamespace(message=SimpleNamespace(usage_metadata=usage_metadata))]])


def test_langchain_non_streamed_openai_record_takes_cache_writes_out_of_input(payloads):
    chat_result = langchain_openai.ChatOpenAI(model=LANGCHAIN_MODEL, api_key="sk-test")._create_chat_result({
        "id": "chatcmpl-langchain", "object": "chat.completion", "created": 1, "model": LANGCHAIN_MODEL,
        "usage": WRITING_CHAT_USAGE,
        "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "hi"}}],
    })
    llm_result = LLMResult(generations=[chat_result.generations], llm_output=chat_result.llm_output)

    _meter_through_the_callback(llm_result, _run_info("openai"))

    record = the_one_record(payloads)
    assert record["input_token_count"] == PROMPT - WRITTEN
    assert record["cache_creation_token_count"] == WRITTEN


@pytest.mark.parametrize("ls_provider, class_name", [("openai", "ChatOpenAI"), ("azure", "AzureChatOpenAI")],
                         ids=["openai", "azure"])
def test_langchain_streamed_or_responses_openai_record_takes_cache_writes_out_of_input(
        ls_provider, class_name, payloads):
    assert _usage_metadata_only_result().generations[0][0].message.usage_metadata["input_token_details"][
        "cache_creation"] == WRITTEN

    _meter_through_the_callback(_usage_metadata_only_result(), _run_info(ls_provider, class_name, True))

    record = the_one_record(payloads)
    assert record["input_token_count"] == PROMPT - WRITTEN
    assert record["cache_creation_token_count"] == WRITTEN


def test_langchain_record_for_an_unlabelled_provider_is_unchanged(payloads):
    """Regression guard that locks in a known pre-existing double bill: with no
    provider label the callback cannot tell what ``cache_creation`` means, so
    the write stays in input as well as in the creation count."""
    run_info = _run_info(None, class_name="SomeChatModel")

    _meter_through_the_callback(_usage_metadata_only_result(), run_info)

    record = the_one_record(payloads)
    assert record["input_token_count"] == PROMPT
    assert record["cache_creation_token_count"] == WRITTEN


def litellm_claude_chat_usage(read, write, uncached=10, output=20):
    """LiteLLM's chat usage for a Claude call, in the shape of LITELLM_CLAUDE_CHAT_USAGE."""
    prompt = uncached + read + write
    return {
        "cache_creation_input_tokens": write, "cache_read_input_tokens": read, "completion_tokens": output,
        "completion_tokens_details": {"reasoning_tokens": 0, "text_tokens": output}, "prompt_tokens": prompt,
        "prompt_tokens_details": {"cache_creation_tokens": write, "cache_write_tokens": write, "cached_tokens": read,
                                  "text_tokens": uncached},
        "total_tokens": prompt + output,
    }


def test_the_usage_builder_reproduces_the_recorded_gateway_body():
    assert litellm_claude_chat_usage(read=2000, write=500) == LITELLM_CLAUDE_CHAT_USAGE


def _non_streamed_result(usage):
    chat_result = langchain_openai.ChatOpenAI(model=GATEWAY_CLAUDE_MODEL, api_key="sk-test")._create_chat_result({
        "id": "chatcmpl-gateway", "object": "chat.completion", "created": 1, "model": GATEWAY_CLAUDE_MODEL,
        "usage": usage,
        "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "hi"}}],
    })
    return LLMResult(generations=[chat_result.generations], llm_output=chat_result.llm_output)


def _usage_metadata_only(usage):
    usage_metadata = _create_usage_metadata(usage)
    return SimpleNamespace(generations=[[SimpleNamespace(message=SimpleNamespace(usage_metadata=usage_metadata))]])


GATEWAY_CACHE_CASES = [(2000, 500), (2000, 0), (0, 500)]
COMPARED_FIELDS = ("input_token_count", "cache_read_token_count", "cache_creation_token_count",
                   "output_token_count")


def _counts(record):
    return {field: record[field] for field in COMPARED_FIELDS}


@pytest.mark.parametrize("read, write", GATEWAY_CACHE_CASES, ids=lambda v: str(v))
@pytest.mark.parametrize("langchain_result", [_non_streamed_result, _usage_metadata_only],
                         ids=["non_streamed", "usage_metadata_only"])
def test_langchain_chatopenai_gateway_claude_record_matches_the_openai_client(
        read, write, langchain_result, payloads, served_usage, served_model):
    """BACK-4240: ChatOpenAI pointed at a LiteLLM gateway serving Claude must
    record the same counts as the OpenAI client does for the same response."""
    usage = litellm_claude_chat_usage(read, write)
    served_usage(chat=usage)
    served_model(GATEWAY_CLAUDE_MODEL)
    sync_chat_create(FakeOpenAI(), None)
    client_record = the_one_record(payloads)
    payloads.clear()

    _meter_through_the_callback(langchain_result(usage), _run_info("openai", model=GATEWAY_CLAUDE_MODEL))

    callback_record = the_one_record(payloads)
    assert _counts(callback_record) == _counts(client_record)
    assert callback_record["input_token_count"] == 10
