"""ReveniumCrewWrapper on CrewAI 1.x native providers (BACK-4000).

CrewAI 1.x calls OpenAI, Anthropic and Gemini through their own SDKs instead
of LiteLLM, so the crew's records come from the provider middlewares. These
tests run a real crew through the real vendor SDKs against a loopback stub and
read the records the provider middleware sends.
"""
import asyncio
import json
import logging
import os
import threading
from http.server import ThreadingHTTPServer

import pytest

for _name, _value in (
    ("CREWAI_DISABLE_TELEMETRY", "true"),
    ("OTEL_SDK_DISABLED", "true"),
    ("CREWAI_TRACING_ENABLED", "false"),
    ("CREWAI_TESTING", "true"),
):
    os.environ.setdefault(_name, _value)

crewai = pytest.importorskip("crewai")
openai = pytest.importorskip("openai")
pytest.importorskip("anthropic")
pytest.importorskip("google.genai")

import revenium_middleware.anthropic  # noqa: E402,F401
import revenium_middleware.google.google_ai  # noqa: E402,F401
import revenium_middleware.litellm.client.middleware  # noqa: E402,F401
import revenium_middleware.openai  # noqa: E402,F401
import revenium_middleware.litellm.client.integrations.crewai as crewai_mod  # noqa: E402
from revenium_middleware import revenium_metadata  # noqa: E402
from revenium_middleware._core.context import get_injected_metadata  # noqa: E402
from revenium_middleware._core.metering_pool import wait_until_idle  # noqa: E402
from revenium_middleware.litellm.client.context import metadata_context  # noqa: E402
from revenium_middleware.litellm.client.integrations.crewai import ReveniumCrewWrapper  # noqa: E402

from . import fake_openai_upstream  # noqa: E402

ATTRIBUTION = {"organization_id": "AcmeCorp", "subscription_id": "sub-1", "product_id": "Platinum"}


def _anthropic_message(body):
    return {
        "id": "msg_1", "type": "message", "role": "assistant", "model": body.get("model"),
        "content": [{"type": "text", "text": "billing"}], "stop_reason": "end_turn",
        "stop_sequence": None, "usage": {"input_tokens": 11, "output_tokens": 7},
    }


def _gemini_response():
    return {
        "candidates": [{"content": {"role": "model", "parts": [{"text": "billing"}]}, "finishReason": "STOP"}],
        "usageMetadata": {"promptTokenCount": 11, "candidatesTokenCount": 7, "totalTokenCount": 18},
        "modelVersion": "gemini-2.5-flash",
    }


_COUNT_TOKENS_PATH = "/messages/count_tokens"


def _generation_paths():
    """The requests that generate text: CrewAI 1.15.29 preflights an Anthropic call with a free token count."""
    return [path for path in _ProviderHandler.paths if not path.endswith(_COUNT_TOKENS_PATH)]


class _ProviderHandler(fake_openai_upstream._Handler):
    """OpenAI shapes from the shared fake upstream, plus Anthropic and Gemini."""

    paths = []

    def do_POST(self):
        type(self).paths.append(self.path)
        if self.path.endswith(_COUNT_TOKENS_PATH) or self.path.endswith("/messages") \
                or "generateContent" in self.path:
            length = int(self.headers.get("content-length", 0))
            body = json.loads(self.rfile.read(length) or b"{}")
            if self.path.endswith(_COUNT_TOKENS_PATH):
                self._send("application/json", json.dumps({"input_tokens": 11}).encode())
                return
            reply = _anthropic_message(body) if self.path.endswith("/messages") else _gemini_response()
            self._send("application/json", json.dumps(reply).encode())
            return
        super().do_POST()


@pytest.fixture
def stub(monkeypatch, tmp_path):
    _ProviderHandler.paths = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), _ProviderHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = "http://127.0.0.1:{}".format(server.server_address[1])
    monkeypatch.setenv("CREWAI_STORAGE_DIR", str(tmp_path))
    monkeypatch.setenv("GOOGLE_GEMINI_BASE_URL", base)
    yield base
    server.shutdown()
    server.server_close()


def _openai_llm(stub, **kwargs):
    return crewai.LLM(model="gpt-4o-mini", base_url=stub + "/v1", api_key="sk-test", **kwargs)


def _anthropic_llm(stub):
    return crewai.LLM(model="anthropic/claude-3-5-haiku-latest", base_url=stub, api_key="sk-ant-test")


def _gemini_llm(stub):
    return crewai.LLM(model="gemini/gemini-2.5-flash", api_key="g-test", base_url=stub)


def _wrapper(llm, **crew_kwargs):
    agent = crewai.Agent(role="Triage", goal="triage", backstory="b", llm=llm, allow_delegation=False)
    task = crewai.Task(description="Classify: refund please.", expected_output="one word", agent=agent)
    return ReveniumCrewWrapper(agents=[agent], tasks=[task], agentic_job_id="job-42", **ATTRIBUTION, **crew_kwargs)


def _records(mock_revenium_client):
    assert wait_until_idle(10)
    return [call.kwargs for call in mock_revenium_client.ai.create_completion.call_args_list]


def _assert_crew_attribution(record, wrapper):
    assert record.get("organization_name") == "AcmeCorp"
    assert record.get("product_name") == "Platinum"
    assert record.get("subscription_id") == "sub-1"
    assert record.get("trace_id") == wrapper.trace_id
    assert (record.get("extra_body") or {}).get("agenticJobId") == "job-42"


def _direct_openai_call(stub):
    client = openai.OpenAI(base_url=stub + "/v1", api_key="sk-test")
    client.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": "hi"}])


def test_openai_native_crew_records_one_attributed_call(stub, mock_revenium_client):
    llm = _openai_llm(stub)
    assert type(llm).__name__ == "OpenAICompletion"
    wrapper = _wrapper(llm)

    wrapper.kickoff()

    records = _records(mock_revenium_client)
    assert _ProviderHandler.paths == ["/v1/chat/completions"]
    assert len(records) == 1
    assert records[0].get("provider") == "OPENAI"
    _assert_crew_attribution(records[0], wrapper)


def test_openai_responses_api_crew_records_one_attributed_call(stub, mock_revenium_client):
    llm = _openai_llm(stub, api="responses")
    assert type(llm).__name__ == "OpenAICompletion"
    wrapper = _wrapper(llm)

    wrapper.kickoff()

    records = _records(mock_revenium_client)
    assert _ProviderHandler.paths == ["/v1/responses"]
    assert len(records) == 1
    assert records[0].get("provider") == "OPENAI"
    _assert_crew_attribution(records[0], wrapper)


def test_anthropic_native_crew_records_one_attributed_call(stub, mock_revenium_client):
    llm = _anthropic_llm(stub)
    assert type(llm).__name__ == "AnthropicCompletion"
    wrapper = _wrapper(llm)

    wrapper.kickoff()

    records = _records(mock_revenium_client)
    assert _generation_paths() == ["/v1/messages"]
    assert len(records) == 1
    assert records[0].get("provider") == "ANTHROPIC"
    _assert_crew_attribution(records[0], wrapper)


def test_gemini_native_crew_records_one_call(stub, mock_revenium_client):
    llm = _gemini_llm(stub)
    assert type(llm).__name__ == "GeminiCompletion"
    wrapper = _wrapper(llm)

    wrapper.kickoff()

    records = _records(mock_revenium_client)
    assert len(_ProviderHandler.paths) == 1
    assert "generateContent" in _ProviderHandler.paths[0]
    assert len(records) == 1


def test_gemini_native_crew_carries_crew_attribution(stub, mock_revenium_client):
    wrapper = _wrapper(_gemini_llm(stub))

    wrapper.kickoff()

    records = _records(mock_revenium_client)
    assert len(records) == 1
    _assert_crew_attribution(records[0], wrapper)


def test_kickoff_async_attributes_native_call(stub, mock_revenium_client):
    wrapper = _wrapper(_openai_llm(stub))

    asyncio.run(wrapper.kickoff_async())

    records = _records(mock_revenium_client)
    assert len(records) == 1
    _assert_crew_attribution(records[0], wrapper)


def test_litellm_routed_crew_records_once_with_openai_middleware_loaded(stub, mock_revenium_client):
    llm = _openai_llm(stub, is_litellm=True)
    assert type(llm).__name__ == "LLM"
    wrapper = _wrapper(llm)

    wrapper.kickoff()

    records = _records(mock_revenium_client)
    assert _ProviderHandler.paths == ["/v1/chat/completions"]
    assert len(records) == 1
    assert records[0].get("provider") == "LITELLM"
    assert records[0].get("organization_name") == "AcmeCorp"
    assert records[0].get("trace_id") == wrapper.trace_id


def test_attribution_ends_with_kickoff(stub, mock_revenium_client):
    _wrapper(_openai_llm(stub)).kickoff()

    _direct_openai_call(stub)

    records = _records(mock_revenium_client)
    assert len(records) == 2
    assert records[1].get("organization_name") is None
    assert records[1].get("product_name") is None


def test_kickoff_warns_once_when_native_provider_middleware_missing(stub, monkeypatch, caplog):
    monkeypatch.setattr(crewai_mod, "is_patched", lambda patch_key: False, raising=False)
    monkeypatch.setattr(crewai_mod, "_warned_llm_classes", set(), raising=False)
    wrapper = _wrapper(_openai_llm(stub))

    with caplog.at_level(logging.WARNING, logger="revenium_middleware.crewai"):
        wrapper.kickoff()
        wrapper.kickoff()

    warnings = [r.getMessage() for r in caplog.records
                if r.name == "revenium_middleware.crewai" and r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "OpenAICompletion" in warnings[0]
    assert "revenium_middleware.openai" in warnings[0]


def test_streaming_crew_attributes_native_call(stub, mock_revenium_client):
    wrapper = _wrapper(_openai_llm(stub), stream=True)

    streaming = wrapper.kickoff()
    assert type(streaming).__name__ == "CrewStreamingOutput"
    for _chunk in streaming:
        pass

    assert streaming.result is not None
    records = _records(mock_revenium_client)
    assert len(records) == 1
    _assert_crew_attribution(records[0], wrapper)


def test_async_streaming_crew_attributes_native_call(stub, mock_revenium_client):
    wrapper = _wrapper(_openai_llm(stub), stream=True)

    async def consume():
        streaming = await wrapper.kickoff_async()
        assert type(streaming).__name__ == "CrewStreamingOutput"
        async for _chunk in streaming:
            pass
        return streaming

    streaming = asyncio.run(consume())

    assert streaming.result is not None
    records = _records(mock_revenium_client)
    assert len(records) == 1
    _assert_crew_attribution(records[0], wrapper)


def test_outer_revenium_metadata_survives_kickoff(stub, mock_revenium_client):
    wrapper = _wrapper(_openai_llm(stub))

    @revenium_metadata(task_type="outer-task")
    def run():
        wrapper.kickoff()
        _direct_openai_call(stub)

    run()

    crew_record, direct_record = _records(mock_revenium_client)
    assert crew_record.get("task_type") == "outer-task"
    _assert_crew_attribution(crew_record, wrapper)
    assert direct_record.get("task_type") == "outer-task"
    assert direct_record.get("organization_name") is None


def test_attribution_resets_when_kickoff_raises(stub, monkeypatch, mock_revenium_client):
    wrapper = _wrapper(_openai_llm(stub))
    seen_inside = {}

    def failing_kickoff(self, inputs=None, **kwargs):
        seen_inside.update(get_injected_metadata() or {})
        raise RuntimeError("crew failed")

    monkeypatch.setattr(type(wrapper._crew), "kickoff", failing_kickoff)
    injected_before = get_injected_metadata()
    litellm_before = metadata_context.get()

    with pytest.raises(RuntimeError, match="crew failed"):
        wrapper.kickoff()

    assert seen_inside.get("organization_name") == "AcmeCorp"
    assert get_injected_metadata() == injected_before
    assert metadata_context.get() == litellm_before
    _direct_openai_call(stub)
    records = _records(mock_revenium_client)
    assert len(records) == 1
    assert records[0].get("organization_name") is None


def _crew_org_ids():
    return (get_injected_metadata() or {}).get("organization_name"), metadata_context.get().get("organization_name")


class _StreamSteps:
    """Two chunks, then the end of the stream or an error, noting both
    contexts each time the wrapper asks for the next chunk."""

    def __init__(self, fail_at_end=False):
        self.seen_inside = []
        self._chunks = iter(["a", "b"])
        self._fail_at_end = fail_at_end

    def _step(self):
        self.seen_inside.append(_crew_org_ids())
        chunk = next(self._chunks, None)
        if chunk is not None:
            return chunk
        if self._fail_at_end:
            raise RuntimeError("stream failed")
        raise StopIteration

    def __next__(self):
        return self._step()

    async def __anext__(self):
        try:
            return self._step()
        except StopIteration:
            raise StopAsyncIteration


@pytest.mark.parametrize("fail_at_end", [False, True])
def test_stream_steps_restore_both_contexts_between_chunks(stub, fail_at_end):
    wrapper = _wrapper(_openai_llm(stub))
    steps = _StreamSteps(fail_at_end=fail_at_end)
    seen_between = []

    def consume():
        for _chunk in wrapper._scoped_steps(steps):
            seen_between.append(_crew_org_ids())

    if fail_at_end:
        with pytest.raises(RuntimeError, match="stream failed"):
            consume()
    else:
        consume()

    assert steps.seen_inside == [("AcmeCorp", "AcmeCorp")] * 3
    assert seen_between == [(None, None)] * 2
    assert _crew_org_ids() == (None, None)


@pytest.mark.parametrize("fail_at_end", [False, True])
def test_async_stream_steps_restore_both_contexts_between_chunks(stub, fail_at_end):
    wrapper = _wrapper(_openai_llm(stub))
    steps = _StreamSteps(fail_at_end=fail_at_end)
    seen_between = []

    async def consume():
        async for _chunk in wrapper._scoped_async_steps(steps):
            seen_between.append(_crew_org_ids())

    async def consume_and_report_after():
        try:
            await consume()
        finally:
            seen_between.append(_crew_org_ids())

    if fail_at_end:
        with pytest.raises(RuntimeError, match="stream failed"):
            asyncio.run(consume_and_report_after())
    else:
        asyncio.run(consume_and_report_after())

    assert steps.seen_inside == [("AcmeCorp", "AcmeCorp")] * 3
    assert seen_between == [(None, None)] * 3


def test_repeated_kickoff_keeps_the_task_callback(stub, mock_revenium_client):
    """BACK-4262: reusing one wrapper never stacks a layer on the task's callback."""
    calls = []

    def customer_cb(output):
        calls.append(output)

    agent = crewai.Agent(role="Triage", goal="triage", backstory="b", llm=_openai_llm(stub), allow_delegation=False)
    task = crewai.Task(
        description="Classify: refund please.", expected_output="one word", agent=agent, callback=customer_cb,
    )
    wrapper = ReveniumCrewWrapper(agents=[agent], tasks=[task], **ATTRIBUTION)

    for _ in range(3):
        wrapper.kickoff()
        assert task.callback is customer_cb

    assert len(calls) == 3
    assert len(_records(mock_revenium_client)) == 3
