"""OpenAI-client Responses API calls to Perplexity with the README's import (BACK-4312).

The README sets Perplexity up with ``import revenium_middleware.perplexity`` alone. Each
configuration runs in a fresh interpreter so the integrations imported are exactly the
ones the test names, and reports per call what was recorded, cap-checked and sent.
"""
import datetime
import json
import os
import pathlib
import subprocess
import sys
import textwrap
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from entry_points import calls
from entry_points.caps import breached_rule, scoped
from entry_points.perplexity_cost import PERPLEXITY_MODEL, TOTAL_COST
from revenium_middleware.perplexity.middleware import send_metering_data
from revenium_middleware.perplexity.provider import Provider
from revenium_middleware.perplexity.streaming import StreamMeter

PERPLEXITY_ONLY = ["revenium_middleware.perplexity"]
BOTH_IMPORT_ORDERS = [["revenium_middleware.perplexity", "revenium_middleware.openai"],
                      ["revenium_middleware.openai", "revenium_middleware.perplexity"]]

PERPLEXITY_CALLS = ("sync", "async", "stream", "async_stream", "raw", "raw_stream",
                    "helper", "async_helper", "parse", "async_parse")
STREAMED_CALLS = {"stream", "async_stream", "raw_stream", "helper", "async_helper"}
OPENAI_CALLS = ("openai_sync", "openai_stream")

_RESPONSES_PROBE = textwrap.dedent("""
    import asyncio, importlib, json, sys
    from unittest.mock import MagicMock, patch

    import httpx
    import openai
    import pytest

    from entry_points.caps import record_cap_checks, seed_rules
    from entry_points.perplexity_cost import (
        ANSWER, PERPLEXITY_BASE_URL, PERPLEXITY_MODEL, _RESPONSES_BODY, _sse, openai_clients_at,
    )
    from entry_points.stub_requests import count_provider_requests

    OPENAI_BASE_URL = "https://api.openai.com/v1"
    REQUEST = {"model": PERPLEXITY_MODEL, "input": "hi"}


    def _helper_events():
        in_progress = {**_RESPONSES_BODY, "status": "in_progress", "output": [], "usage": None}
        message = {"type": "message", "id": "msg_1", "role": "assistant", "status": "in_progress", "content": []}
        empty_text = {"type": "output_text", "text": "", "annotations": []}
        return _sse([
            {"type": "response.created", "sequence_number": 0, "response": in_progress},
            {"type": "response.output_item.added", "sequence_number": 1, "output_index": 0, "item": message},
            {"type": "response.content_part.added", "sequence_number": 2, "item_id": "msg_1", "output_index": 0,
             "content_index": 0, "part": empty_text},
            {"type": "response.output_text.delta", "sequence_number": 3, "item_id": "msg_1", "output_index": 0,
             "content_index": 0, "delta": ANSWER},
            {"type": "response.completed", "sequence_number": 4, "response": _RESPONSES_BODY},
        ])


    def _helper_clients():
        def handler(request):
            return httpx.Response(200, content=_helper_events(), headers={"content-type": "text/event-stream"})
        transport = httpx.MockTransport(handler)
        return (openai.OpenAI(base_url=PERPLEXITY_BASE_URL, api_key="stub", max_retries=0,
                              http_client=httpx.Client(transport=transport)),
                openai.AsyncOpenAI(base_url=PERPLEXITY_BASE_URL, api_key="stub", max_retries=0,
                                   http_client=httpx.AsyncClient(transport=transport)))


    def _sync_client(base_url=PERPLEXITY_BASE_URL):
        return openai_clients_at(base_url)[0]


    def _async_client():
        return openai_clients_at(PERPLEXITY_BASE_URL)[1]


    async def _async_drain(stream):
        return [event async for event in stream]


    async def _async_stream():
        await _async_drain(await _async_client().responses.create(stream=True, **REQUEST))


    async def _async_helper():
        async with _helper_clients()[1].responses.stream(**REQUEST) as stream:
            await _async_drain(stream)


    def _sync_helper():
        with _helper_clients()[0].responses.stream(**REQUEST) as stream:
            list(stream)


    CALLS = {
        "sync": lambda: _sync_client().responses.create(**REQUEST),
        "async": lambda: asyncio.run(_async_client().responses.create(**REQUEST)),
        "stream": lambda: list(_sync_client().responses.create(stream=True, **REQUEST)),
        "async_stream": lambda: asyncio.run(_async_stream()),
        "raw": lambda: _sync_client().responses.with_raw_response.create(**REQUEST).parse(),
        "raw_stream": lambda: list(
            _sync_client().responses.with_raw_response.create(stream=True, **REQUEST).parse()),
        "helper": _sync_helper,
        "async_helper": lambda: asyncio.run(_async_helper()),
        "parse": lambda: _sync_client().responses.parse(**REQUEST),
        "async_parse": lambda: asyncio.run(_async_client().responses.parse(**REQUEST)),
        "openai_sync": lambda: _sync_client(OPENAI_BASE_URL).responses.create(**REQUEST),
        "openai_stream": lambda: list(_sync_client(OPENAI_BASE_URL).responses.create(stream=True, **REQUEST)),
    }
    RECORD_FIELDS = ("provider", "model", "model_source", "transaction_id", "total_cost",
                     "input_token_count", "output_token_count", "is_streamed")

    config = json.loads(sys.argv[1])
    monkeypatch = pytest.MonkeyPatch()
    seed_rules(monkeypatch, *config["rules"])
    checks = record_cap_checks(monkeypatch)
    requests = count_provider_requests(monkeypatch)
    recorder = MagicMock()
    with patch("revenium_middleware._core.metering.client", recorder):
        for module in config["imports"]:
            importlib.import_module(module)
        from revenium_middleware._core.metering_pool import wait_until_idle

        sent = recorder.ai.create_completion.call_args_list
        outcomes = {}
        for name in config["calls"]:
            records_before, checks_before, requests_before = len(sent), len(checks), requests["httpx"]
            try:
                CALLS[name]()
                error = None
            except Exception as exc:
                error = type(exc).__name__
            wait_until_idle(10.0)
            outcomes[name] = {
                "error": error,
                "records": [{field: entry.kwargs.get(field) for field in RECORD_FIELDS}
                            for entry in sent[records_before:]],
                "cap_checks": [[check.provider, check.model] for check in checks[checks_before:]],
                "provider_requests": requests["httpx"] - requests_before,
            }
    print(json.dumps({"outcomes": outcomes,
                      "openai_middleware_loaded": "revenium_middleware.openai.middleware" in sys.modules}))
""")


def _run_responses_calls(imports, calls, rules=()):
    env = {**os.environ, "PYTHONPATH": str(pathlib.Path(__file__).resolve().parents[1]),
           "REVENIUM_METERING_API_KEY": "hak_test_perplexity_responses",
           "REVENIUM_METERING_BASE_URL": "http://127.0.0.1:9"}
    config = json.dumps({"imports": imports, "calls": list(calls), "rules": list(rules)})
    completed = subprocess.run([sys.executable, "-c", _RESPONSES_PROBE, config],
                               capture_output=True, text=True, timeout=300, env=env)
    assert completed.returncode == 0, completed.stderr[-4000:]
    return json.loads(completed.stdout.strip().splitlines()[-1])


def _perplexity_record(call, model_source=None):
    return {"provider": "PERPLEXITY", "model": PERPLEXITY_MODEL, "model_source": model_source,
            "total_cost": TOTAL_COST, "input_token_count": 2, "output_token_count": 3,
            "is_streamed": call in STREAMED_CALLS}


def _without_transaction_id(record):
    return {field: value for field, value in record.items() if field != "transaction_id"}


def test_a_responses_call_to_perplexity_is_recorded_with_only_the_perplexity_integration():
    result = _run_responses_calls(PERPLEXITY_ONLY, PERPLEXITY_CALLS)

    assert result["openai_middleware_loaded"] is False
    for call, outcome in result["outcomes"].items():
        assert outcome["error"] is None, call
        assert [_without_transaction_id(record) for record in outcome["records"]] == [_perplexity_record(call)], call
        if call not in STREAMED_CALLS:
            assert outcome["records"][0]["transaction_id"] == "resp_0001", call


def test_a_breached_perplexity_cap_refuses_a_responses_call_with_only_the_perplexity_integration():
    rule = breached_rule(4312, scoped("PROVIDER", "IS", "PERPLEXITY"))

    result = _run_responses_calls(PERPLEXITY_ONLY, PERPLEXITY_CALLS, [rule])

    assert result["openai_middleware_loaded"] is False
    assert result["outcomes"] == {
        call: {"error": "BudgetExceededError", "records": [], "cap_checks": [["PERPLEXITY", PERPLEXITY_MODEL]],
               "provider_requests": 0}
        for call in PERPLEXITY_CALLS}


def test_a_responses_call_to_openai_is_untouched_with_only_the_perplexity_integration():
    result = _run_responses_calls(PERPLEXITY_ONLY, OPENAI_CALLS, [breached_rule(4312)])

    assert result["openai_middleware_loaded"] is False
    assert result["outcomes"] == {
        call: {"error": None, "records": [], "cap_checks": [], "provider_requests": 1} for call in OPENAI_CALLS}


@pytest.mark.parametrize("imports", BOTH_IMPORT_ORDERS, ids=["perplexity_first", "openai_first"])
def test_a_responses_call_to_perplexity_with_both_integrations_is_recorded_and_checked_once(imports):
    allowing_rule = breached_rule(4312, scoped("AGENT", "IS", "nobody"))

    result = _run_responses_calls(imports, PERPLEXITY_CALLS + OPENAI_CALLS, [allowing_rule])

    for call in PERPLEXITY_CALLS:
        outcome = result["outcomes"][call]
        assert outcome["error"] is None, call
        assert outcome["provider_requests"] == 1, call
        assert outcome["cap_checks"] == [["PERPLEXITY", PERPLEXITY_MODEL]], call
        assert [_without_transaction_id(record) for record in outcome["records"]] == [
            _perplexity_record(call, model_source="PERPLEXITY")], call
    for call in OPENAI_CALLS:
        outcome = result["outcomes"][call]
        assert outcome["error"] is None, call
        assert outcome["provider_requests"] == 1, call
        assert [check[0] for check in outcome["cap_checks"]] == ["OPENAI"], call
        assert [record["provider"] for record in outcome["records"]] == ["OPENAI"], call


OPENAI_RESPONSES_KEYS_PROBE = r"""
import json
import revenium_middleware.openai
import revenium_middleware.perplexity
from revenium_middleware._core import patch_registry
from revenium_middleware.perplexity import middleware as mw

print(json.dumps([patch_registry.is_patched(key) for key in (
    mw.OPENAI_RESPONSES_CREATE, mw.OPENAI_ASYNC_RESPONSES_CREATE,
    mw.OPENAI_RESPONSES_PARSE, mw.OPENAI_ASYNC_RESPONSES_PARSE)]))
"""


def test_the_openai_middleware_names_the_responses_keys_the_perplexity_middleware_defers_to():
    env = {**os.environ, "REVENIUM_METERING_API_KEY": "hak_test_perplexity_responses",
           "REVENIUM_METERING_BASE_URL": "http://127.0.0.1:9"}
    completed = subprocess.run([sys.executable, "-c", OPENAI_RESPONSES_KEYS_PROBE],
                               capture_output=True, text=True, env=env, timeout=120)

    assert completed.returncode == 0, completed.stderr[-4000:]
    assert json.loads(completed.stdout.strip().splitlines()[-1]) == [True, True, True, True]


def _metered_usage(chunks):
    records = []
    meter = StreamMeter(records.append)
    for chunk in chunks:
        meter.observe(chunk)
    meter.meter()
    return [record.usage for record in records]


def test_a_chat_chunk_without_usage_keeps_the_usage_seen_earlier_in_the_stream():
    usage = SimpleNamespace(prompt_tokens=2, completion_tokens=3)
    chunks = [SimpleNamespace(usage=usage, choices=[]), MagicMock(usage=None, choices=[])]

    assert _metered_usage(chunks) == [usage]


def test_a_completed_responses_event_supplies_the_stream_usage():
    usage = SimpleNamespace(input_tokens=2, output_tokens=3)
    events = [SimpleNamespace(type="response.output_text.delta", delta="hi"),
              SimpleNamespace(type="response.completed", response=SimpleNamespace(usage=usage))]

    assert _metered_usage(events) == [usage]


@pytest.fixture
def recorder():
    recording_client = MagicMock()
    with patch("revenium_middleware._core.metering.client", recording_client):
        yield recording_client


def _submitted_stop_reason(recording_client, response):
    send_metering_data(response, datetime.datetime.now(datetime.timezone.utc), {}, PERPLEXITY_MODEL,
                       Provider.PERPLEXITY)
    calls.wait_for_metering()
    recording_client.ai.create_completion.assert_called_once()
    return recording_client.ai.create_completion.call_args.kwargs["stop_reason"]


def _stream_summary(*events):
    summaries = []
    meter = StreamMeter(summaries.append)
    for event in events:
        meter.observe(event)
    meter.meter()
    return summaries[0]


def _terminal_event(event_type, status, incomplete_reason=None):
    details = SimpleNamespace(reason=incomplete_reason) if incomplete_reason else None
    usage = SimpleNamespace(input_tokens=2, output_tokens=3, total_tokens=5)
    return SimpleNamespace(type=event_type, response=SimpleNamespace(
        id="resp_1", usage=usage, status=status, incomplete_details=details))


@pytest.mark.parametrize("event, stop_reason", [
    (_terminal_event("response.incomplete", "incomplete", "max_output_tokens"), "TOKEN_LIMIT"),
    (_terminal_event("response.failed", "failed"), "ERROR"),
    (_terminal_event("response.completed", "completed"), "END"),
], ids=["incomplete_max_output_tokens", "failed", "completed"])
def test_a_terminal_responses_event_records_its_stop_reason(recorder, event, stop_reason):
    summary = _stream_summary(SimpleNamespace(type="response.output_text.delta", delta="hi"), event)

    assert _submitted_stop_reason(recorder, summary) == stop_reason


def test_a_non_streamed_incomplete_responses_object_records_token_limit(recorder):
    response = _terminal_event("response", "incomplete", "max_output_tokens").response

    assert _submitted_stop_reason(recorder, response) == "TOKEN_LIMIT"
