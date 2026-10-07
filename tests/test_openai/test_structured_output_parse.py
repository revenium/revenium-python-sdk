"""OpenAI structured-output ``parse()`` calls meter exactly once (BACK-3609).

``chat.completions.parse`` and ``responses.parse`` post their request
themselves instead of calling ``create()``, so they carry their own wraps. The
real ``openai`` client runs over ``httpx.MockTransport``; no call leaves the
process.
"""
import asyncio
import contextlib
import json
import logging
import threading
from types import SimpleNamespace
from unittest.mock import patch

import httpx
import openai
import pydantic
import pytest
import wrapt
from openai import AsyncOpenAI, OpenAI
from openai.resources.chat.completions import AsyncCompletions, Completions
from openai.resources.responses import AsyncResponses, Responses

from revenium_middleware._core.call_ownership import (
    OPENAI, claimed_by_transport, reset_claimed_response_ids, transport_claim_mark,
)
from revenium_middleware.openai import middleware as mw

MODEL = "gpt-4o-mini"
CHAT_ID = "chatcmpl-parse-1"
RESPONSE_ID = "resp_parse_1"
INPUT_TOKENS, OUTPUT_TOKENS, CACHE_READ_TOKENS, REASONING_TOKENS = 11, 7, 3, 2
STRUCTURED_TEXT = json.dumps({"city": "Paris", "degrees": 21})
MESSAGES = [{"role": "user", "content": "weather in Paris?"}]

CHAT_USAGE = {
    "prompt_tokens": INPUT_TOKENS, "completion_tokens": OUTPUT_TOKENS,
    "total_tokens": INPUT_TOKENS + OUTPUT_TOKENS,
    "prompt_tokens_details": {"cached_tokens": CACHE_READ_TOKENS},
    "completion_tokens_details": {"reasoning_tokens": REASONING_TOKENS},
}
RESPONSES_USAGE = {
    "input_tokens": INPUT_TOKENS, "output_tokens": OUTPUT_TOKENS,
    "total_tokens": INPUT_TOKENS + OUTPUT_TOKENS,
    "input_tokens_details": {"cached_tokens": CACHE_READ_TOKENS},
    "output_tokens_details": {"reasoning_tokens": REASONING_TOKENS},
}


class Weather(pydantic.BaseModel):
    city: str
    degrees: int


def _chat_completion(text, finish_reason):
    return {"id": CHAT_ID, "object": "chat.completion", "created": 1, "model": MODEL,
            "choices": [{"index": 0, "finish_reason": finish_reason,
                         "message": {"role": "assistant", "content": text}}],
            "usage": CHAT_USAGE}


def _response(text):
    message = {"type": "message", "id": "msg_1", "role": "assistant", "status": "completed",
               "content": [{"type": "output_text", "text": text, "annotations": []}]}
    return {"id": RESPONSE_ID, "object": "response", "created_at": 1, "model": MODEL,
            "status": "completed", "output": [message], "parallel_tool_calls": True,
            "tool_choice": "auto", "tools": [], "usage": RESPONSES_USAGE}


class FakeOpenAI:
    def __init__(self, base_url="https://api.openai.test/v1"):
        self.base_url = base_url
        self.requests = []
        self.fail_next = False
        self.text = STRUCTURED_TEXT
        self.finish_reason = "stop"

    def __call__(self, request):
        self.requests.append(json.loads(request.content or b"{}"))
        if self.fail_next:
            self.fail_next = False
            return httpx.Response(400, json={"error": {"message": "bad request", "type": "invalid_request"}})
        if request.url.path.endswith("/chat/completions"):
            return httpx.Response(200, json=_chat_completion(self.text, self.finish_reason))
        if request.url.path.endswith("/responses"):
            return httpx.Response(200, json=_response(self.text))
        return httpx.Response(404, json={"error": "unrouted"})

    def sync_client(self):
        return OpenAI(api_key="sk-test", base_url=self.base_url, max_retries=0,
                      http_client=httpx.Client(transport=httpx.MockTransport(self)))

    def async_client(self):
        async def handler(request):
            await request.aread()
            return self(request)
        return AsyncOpenAI(api_key="sk-test", base_url=self.base_url, max_retries=0,
                           http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))


def _run_coro_to_completion(coro):
    thread = threading.Thread(target=lambda: asyncio.run(coro))
    thread.start()
    thread.join()
    return thread


@pytest.fixture
def payloads(monkeypatch):
    monkeypatch.setenv("REVENIUM_METERING_API_KEY", "hak_test_parse")
    recorded = []

    def record(kind, args):
        recorded.append(args)
        return SimpleNamespace(id="evt-test")

    reset_claimed_response_ids()
    with patch.object(mw, "get_client", lambda: object()), \
         patch.object(mw, "submit_ai_event", side_effect=record), \
         patch.object(mw, "run_async_in_thread", side_effect=_run_coro_to_completion):
        yield recorded
    reset_claimed_response_ids()


@pytest.fixture
def fake():
    return FakeOpenAI()


def sync_chat_parse(fake):
    completion = fake.sync_client().chat.completions.parse(model=MODEL, messages=MESSAGES,
                                                            response_format=Weather)
    return completion.choices[0].message.parsed


def async_chat_parse(fake):
    async def go():
        return await fake.async_client().chat.completions.parse(model=MODEL, messages=MESSAGES,
                                                                 response_format=Weather)
    return asyncio.run(go()).choices[0].message.parsed


def sync_responses_parse(fake):
    return fake.sync_client().responses.parse(model=MODEL, input="weather in Paris?",
                                              text_format=Weather).output_parsed


def async_responses_parse(fake):
    async def go():
        return await fake.async_client().responses.parse(model=MODEL, input="weather in Paris?",
                                                         text_format=Weather)
    return asyncio.run(go()).output_parsed


def sync_chat_create(fake):
    return fake.sync_client().chat.completions.create(model=MODEL, messages=MESSAGES)


def sync_responses_create(fake):
    return fake.sync_client().responses.create(model=MODEL, input="hi")


PARSE_CALLS = [
    pytest.param(sync_chat_parse, CHAT_ID, id="chat-sync"),
    pytest.param(async_chat_parse, CHAT_ID, id="chat-async"),
    pytest.param(sync_responses_parse, RESPONSE_ID, id="responses-sync"),
    pytest.param(async_responses_parse, RESPONSE_ID, id="responses-async"),
]


def assert_one_exact_payload(payloads, response_id):
    assert len(payloads) == 1, payloads
    payload = payloads[0]
    assert payload["transaction_id"] == response_id
    assert payload["model"] == MODEL
    assert payload["provider"] == "OPENAI"
    assert payload["is_streamed"] is False
    assert payload["input_token_count"] == INPUT_TOKENS
    assert payload["output_token_count"] == OUTPUT_TOKENS
    assert payload["total_token_count"] == INPUT_TOKENS + OUTPUT_TOKENS
    assert payload["cache_read_token_count"] == CACHE_READ_TOKENS
    assert payload["reasoning_token_count"] == REASONING_TOKENS


class TestParseMetersOnce:
    @pytest.mark.parametrize("parse_call, response_id", PARSE_CALLS)
    def test_one_payload_with_the_stub_usage(self, fake, payloads, parse_call, response_id):
        assert parse_call(fake) == Weather(city="Paris", degrees=21)
        assert len(fake.requests) == 1
        assert_one_exact_payload(payloads, response_id)

    @pytest.mark.parametrize("parse_call, response_id", PARSE_CALLS)
    def test_the_call_is_claimed_for_the_transport(self, fake, payloads, parse_call, response_id):
        mark = transport_claim_mark()
        parse_call(fake)
        assert claimed_by_transport(mark, {OPENAI}, [response_id])

    def test_raw_response_parse_meters_once(self, fake, payloads):
        raw = fake.sync_client().chat.completions.with_raw_response.parse(
            model=MODEL, messages=MESSAGES, response_format=Weather)
        assert raw.parse().choices[0].message.parsed == Weather(city="Paris", degrees=21)
        assert_one_exact_payload(payloads, CHAT_ID)

    def test_usage_metadata_is_not_sent_upstream(self, fake, payloads):
        fake.sync_client().chat.completions.parse(model=MODEL, messages=MESSAGES, response_format=Weather,
                                                   usage_metadata={"trace_id": "trace-parse"})
        assert "usage_metadata" not in fake.requests[0]
        assert len(payloads) == 1

    @pytest.mark.parametrize("parse_call, _response_id", PARSE_CALLS)
    def test_selective_metering_passes_through(self, fake, payloads, parse_call, _response_id):
        with patch.object(mw, "is_selective_metering_enabled", return_value=True):
            assert parse_call(fake) == Weather(city="Paris", degrees=21)
        assert payloads == []


class TestCreateStillMetersAroundParse:
    @pytest.mark.parametrize("create_call, response_id", [
        pytest.param(sync_chat_create, CHAT_ID, id="chat"),
        pytest.param(sync_responses_create, RESPONSE_ID, id="responses"),
    ])
    def test_create_after_parse_meters_once(self, fake, payloads, create_call, response_id):
        sync_chat_parse(fake)
        payloads.clear()
        create_call(fake)
        assert_one_exact_payload(payloads, response_id)

    def test_create_after_a_failed_parse_meters_once(self, fake, payloads, caplog):
        fake.fail_next = True
        with pytest.raises(openai.BadRequestError):
            sync_chat_parse(fake)
        assert payloads == []
        assert UNRECORDED_WARNING not in caplog.text
        sync_chat_create(fake)
        assert_one_exact_payload(payloads, CHAT_ID)


UNRECORDED_WARNING = "so the call's usage was not recorded"


class TestParseThatFailsAfterTheProviderAnswered:
    @pytest.mark.parametrize("parse_call", [
        pytest.param(sync_chat_parse, id="sync"),
        pytest.param(async_chat_parse, id="async"),
    ])
    def test_length_limit_is_metered_and_still_raised(self, fake, payloads, parse_call):
        fake.finish_reason = "length"
        with pytest.raises(openai.LengthFinishReasonError):
            parse_call(fake)
        assert_one_exact_payload(payloads, CHAT_ID)

    def test_length_limit_is_claimed_for_the_transport(self, fake, payloads):
        fake.finish_reason = "length"
        mark = transport_claim_mark()
        with pytest.raises(openai.LengthFinishReasonError):
            sync_chat_parse(fake)
        assert claimed_by_transport(mark, {OPENAI}, [CHAT_ID])

    @pytest.mark.parametrize("parse_call", [
        pytest.param(sync_chat_parse, id="chat-sync"),
        pytest.param(async_chat_parse, id="chat-async"),
        pytest.param(sync_responses_parse, id="responses-sync"),
        pytest.param(async_responses_parse, id="responses-async"),
    ])
    def test_validation_error_warns_and_is_still_raised(self, fake, payloads, caplog, parse_call):
        fake.text = json.dumps({"city": "Paris"})
        with caplog.at_level(logging.WARNING, logger="revenium_middleware.extension"):
            with pytest.raises(pydantic.ValidationError):
                parse_call(fake)
        assert payloads == []
        warnings = [r for r in caplog.records if UNRECORDED_WARNING in r.getMessage()]
        assert len(warnings) == 1
        assert warnings[0].levelno == logging.WARNING
        assert "ValidationError" in warnings[0].getMessage()

    def test_content_filter_warns_and_is_still_raised(self, fake, payloads, caplog):
        fake.finish_reason = "content_filter"
        with caplog.at_level(logging.WARNING, logger="revenium_middleware.extension"):
            with pytest.raises(openai.ContentFilterFinishReasonError):
                sync_chat_parse(fake)
        assert payloads == []
        assert UNRECORDED_WARNING in caplog.text

    def test_a_metering_failure_does_not_replace_the_callers_exception(self, fake, payloads):
        fake.finish_reason = "length"
        with patch.object(mw, "_meter_owned_response", side_effect=RuntimeError("metering broke")):
            with pytest.raises(openai.LengthFinishReasonError):
                sync_chat_parse(fake)


PERPLEXITY_BASE_URL = "https://api.perplexity.ai"


class TestPerplexityBoundParse:
    """The Perplexity middleware wraps create() only, so a Perplexity-bound
    parse() through the OpenAI client is enforced and metered here."""

    @pytest.fixture
    def perplexity(self):
        return FakeOpenAI(base_url=PERPLEXITY_BASE_URL)

    @pytest.mark.parametrize("parse_call", [
        pytest.param(sync_chat_parse, id="sync"),
        pytest.param(async_chat_parse, id="async"),
    ])
    def test_one_perplexity_record_after_the_enforcement_check(self, perplexity, payloads, parse_call):
        with patch.object(mw, "check_enforcement", wraps=mw.check_enforcement) as enforcement:
            assert parse_call(perplexity) == Weather(city="Paris", degrees=21)
        enforcement.assert_called_once()
        assert len(perplexity.requests) == 1
        assert len(payloads) == 1
        payload = payloads[0]
        assert payload["provider"] == "PERPLEXITY"
        assert payload["model_source"] == "PERPLEXITY"
        assert payload["transaction_id"] == CHAT_ID
        assert payload["input_token_count"] == INPUT_TOKENS
        assert payload["output_token_count"] == OUTPUT_TOKENS

    def test_a_blocked_budget_stops_the_call_before_perplexity(self, perplexity, payloads):
        with patch.object(mw, "check_enforcement", side_effect=mw.BudgetExceededError("blocked")):
            with pytest.raises(mw.BudgetExceededError):
                sync_chat_parse(perplexity)
        assert perplexity.requests == []
        assert payloads == []

    def test_openai_bound_parse_keeps_the_openai_label(self, fake, payloads):
        sync_chat_parse(fake)
        assert payloads[0]["provider"] == "OPENAI"
        assert payloads[0]["model_source"] == "OPENAI"


def _delegating_parse(format_kwarg):
    """``parse()`` as a future openai release might write it: through ``create()``."""
    def parse(self, **kwargs):
        kwargs.pop(format_kwarg)
        return self.create(**kwargs)
    return parse


def _async_delegating_parse(format_kwarg):
    async def parse(self, **kwargs):
        kwargs.pop(format_kwarg)
        return await self.create(**kwargs)
    return parse


@pytest.fixture
def parse_delegates_to_create(monkeypatch):
    for cls, delegate, wrapper in [
        (Completions, _delegating_parse("response_format"), mw.chat_parse_wrapper),
        (AsyncCompletions, _async_delegating_parse("response_format"), mw.async_chat_parse_wrapper),
        (Responses, _delegating_parse("text_format"), mw.responses_parse_wrapper),
        (AsyncResponses, _async_delegating_parse("text_format"), mw.async_responses_parse_wrapper),
    ]:
        monkeypatch.setattr(cls, "parse", wrapt.FunctionWrapper(delegate, wrapper))


def _delegated_calls():
    def chat_sync(fake):
        fake.sync_client().chat.completions.parse(model=MODEL, messages=MESSAGES, response_format=Weather)

    def chat_async(fake):
        async def go():
            await fake.async_client().chat.completions.parse(model=MODEL, messages=MESSAGES,
                                                              response_format=Weather)
        asyncio.run(go())

    def responses_sync(fake):
        fake.sync_client().responses.parse(model=MODEL, input="hi", text_format=Weather)

    def responses_async(fake):
        async def go():
            await fake.async_client().responses.parse(model=MODEL, input="hi", text_format=Weather)
        asyncio.run(go())

    return [
        pytest.param(chat_sync, CHAT_ID, id="chat-sync"),
        pytest.param(chat_async, CHAT_ID, id="chat-async"),
        pytest.param(responses_sync, RESPONSE_ID, id="responses-sync"),
        pytest.param(responses_async, RESPONSE_ID, id="responses-async"),
    ]


@pytest.mark.usefixtures("parse_delegates_to_create")
class TestParseThatDelegatesToCreate:
    @pytest.mark.parametrize("call, response_id", _delegated_calls())
    def test_one_payload(self, fake, payloads, call, response_id):
        call(fake)
        assert len(fake.requests) == 1
        assert_one_exact_payload(payloads, response_id)

    @pytest.mark.parametrize("call, _response_id", _delegated_calls())
    def test_without_the_guard_both_wraps_would_meter(self, fake, payloads, call, _response_id):
        with patch.object(mw, "_metered_by_parse", contextlib.nullcontext):
            call(fake)
        assert len(payloads) == 2


class TestLangChainStructuredOutput:
    """``with_structured_output`` reaches OpenAI through ``with_raw_response.parse``;
    the transport record must be the only one."""

    @pytest.fixture
    def langchain_openai(self, monkeypatch, fake):
        module = pytest.importorskip("langchain_openai")
        monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
        monkeypatch.setattr(httpx.HTTPTransport, "handle_request", lambda self, request: fake(request))

        async def handle_async(self, request):
            await request.aread()
            return fake(request)

        monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", handle_async)
        return module

    @staticmethod
    def _structured(langchain_openai, **kwargs):
        from revenium_middleware.openai.langchain import ReveniumCallbackHandler

        model = langchain_openai.ChatOpenAI(model=MODEL, api_key="sk-test", max_retries=0,
                                            callbacks=[ReveniumCallbackHandler()], **kwargs)
        return model.with_structured_output(Weather, method="json_schema")

    @pytest.mark.parametrize("use_responses_api, response_id", [
        pytest.param(False, CHAT_ID, id="chat"),
        pytest.param(True, RESPONSE_ID, id="responses"),
    ])
    def test_invoke_sends_one_transport_record(self, langchain_openai, fake, payloads,
                                               use_responses_api, response_id):
        chain = self._structured(langchain_openai, use_responses_api=use_responses_api)
        assert chain.invoke("weather in Paris?") == Weather(city="Paris", degrees=21)
        assert len(fake.requests) == 1
        assert_one_exact_payload(payloads, response_id)

    @pytest.mark.parametrize("use_responses_api, response_id", [
        pytest.param(False, CHAT_ID, id="chat"),
        pytest.param(True, RESPONSE_ID, id="responses"),
    ])
    def test_ainvoke_sends_one_transport_record(self, langchain_openai, fake, payloads,
                                                use_responses_api, response_id):
        chain = self._structured(langchain_openai, use_responses_api=use_responses_api)
        assert asyncio.run(chain.ainvoke("weather in Paris?")) == Weather(city="Paris", degrees=21)
        assert len(fake.requests) == 1
        assert_one_exact_payload(payloads, response_id)


class TestRegistrationOnOlderOpenAI:
    def test_an_attribute_the_installed_openai_lacks_is_skipped(self):
        assert mw._openai_defines("openai.resources.chat.completions", "Completions.parse")
        assert not mw._openai_defines("openai.resources.chat.completions", "Completions.no_such_method")
        assert not mw._openai_defines("openai.resources.no_such_module", "Completions.parse")
