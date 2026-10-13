"""A LangChain ChatOpenAI call sends the reply it got when prompt capture is on (BACK-4381).

Real langchain-openai runs over the stubbed transports of the exactly-once
tests, so no call leaves the process. These tests live in their own module
because importing that stub skips the module when anthropic, httpx2 or
langchain-anthropic is missing, and the other prompt capture tests must not be
skipped with it.
"""
import asyncio
import json
import threading
import uuid
from unittest.mock import MagicMock, patch

import pytest

langchain_openai = pytest.importorskip("langchain_openai")

from langchain_core.messages import HumanMessage, SystemMessage  # noqa: E402
from pydantic import BaseModel  # noqa: E402

import revenium_middleware.openai  # noqa: E402,F401  installs the OpenAI patches
from revenium_middleware._core.call_ownership import _published_in_context, reset_claimed_response_ids  # noqa: E402
from revenium_middleware.openai.langchain import wrap  # noqa: E402

from .openai_transport import answer_every_transport, http_library_of  # noqa: E402
from .responses_stream_events import response_payload, usage_payload  # noqa: E402
from .test_langchain_exactly_once import (  # noqa: E402
    INPUT_TOKENS, OPENAI_ALIAS, OPENAI_DATED, OUTPUT_TOKENS, StubOpenAI, _run_now,
)

SYSTEM = "You are a terse assistant."
QUESTION = "Say hi."
MESSAGES = [SystemMessage(SYSTEM), HumanMessage(QUESTION)]
STRUCTURED_REPLY = '{"answer": "hi"}'


class Answer(BaseModel):
    answer: str


class StubOpenAIWithResponses(StubOpenAI):
    """Also answers a non-streamed Responses call, which LangChain's invoke makes with use_responses_api."""

    def respond(self, request):
        if request.url.path.endswith("/responses") and not json.loads(request.content or b"{}").get("stream"):
            return self._responses_reply(request)
        return super().respond(request)

    def _responses_reply(self, request):
        response_id = f"resp_{uuid.uuid4().hex}"
        with self._lock:
            self.ids.append(response_id)
        payload = response_payload(usage=usage_payload(input_tokens=INPUT_TOKENS, output_tokens=OUTPUT_TOKENS))
        payload.update(id=response_id, model=OPENAI_DATED)
        response = http_library_of(request).Response(200, json=payload)
        response.request = request
        return response


@pytest.fixture(autouse=True)
def _isolated(monkeypatch):
    monkeypatch.setenv("REVENIUM_METERING_API_KEY", "hak_test_langchain")
    monkeypatch.setenv("REVENIUM_CAPTURE_PROMPTS", "true")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    reset_claimed_response_ids()
    publications = _published_in_context.set(())
    yield
    _published_in_context.reset(publications)
    reset_claimed_response_ids()


@pytest.fixture
def records():
    captured = []
    lock = threading.Lock()

    def record(operation, payload, *args, **kwargs):
        with lock:
            captured.append(payload)
        return MagicMock(id="recorded", status_code=201)

    with patch("revenium_middleware.openai.middleware.submit_ai_event", side_effect=record), \
            patch("revenium_middleware.openai.middleware.run_async_in_thread", side_effect=_run_now):
        yield captured


@pytest.fixture
def openai_stub(monkeypatch):
    stub = StubOpenAIWithResponses()
    answer_every_transport(monkeypatch, stub.respond)
    return stub


def invoke(model):
    model.invoke(MESSAGES)


def ainvoke(model):
    asyncio.run(model.ainvoke(MESSAGES))


def astream(model):
    async def call():
        return [chunk async for chunk in model.astream(MESSAGES)]
    asyncio.run(call())


def structured_invoke(model):
    model.with_structured_output(Answer).invoke(MESSAGES)


@pytest.mark.parametrize("call, settings, reply", [
    pytest.param(invoke, {}, "hi", id="invoke"),
    pytest.param(ainvoke, {}, "hi", id="ainvoke"),
    pytest.param(astream, {}, "hi", id="astream"),
    pytest.param(invoke, {"use_responses_api": True}, "hi", id="invoke-responses-api"),
    pytest.param(astream, {"use_responses_api": True}, "hi", id="astream-responses-api"),
    pytest.param(structured_invoke, {}, STRUCTURED_REPLY, id="with_structured_output-invoke"),
])
def test_langchain_chat_openai_captures_the_reply(openai_stub, records, call, settings, reply):
    """BACK-4381: a wrapped ChatOpenAI call sends its system prompt and the reply it got."""
    openai_stub.content = reply
    model = wrap(langchain_openai.ChatOpenAI(model=OPENAI_ALIAS, api_key="sk-test", max_retries=0, **settings))

    call(model)

    assert len(openai_stub.ids) == 1
    assert len(records) == 1, records
    assert not records[0]["transaction_id"].startswith("langchain-")
    assert records[0].get("system_prompt") == SYSTEM
    assert [(message["role"], message["content"]) for message in json.loads(records[0]["input_messages"])] == [
        ("user", QUESTION)]
    assert records[0].get("output_response") == reply
