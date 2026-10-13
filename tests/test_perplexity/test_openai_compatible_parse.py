"""A structured-output ``parse()`` sent to Perplexity through the OpenAI client
is recorded once, with Perplexity's reported cost, whichever integrations are
imported and in whichever order (BACK-4247).

Each configuration runs in a fresh interpreter, since importing an
integration patches the OpenAI client for the rest of the process. Records are
captured at the metering client; no call leaves the process.
"""
import asyncio
import contextlib
import json
import os
import pathlib
import subprocess
import sys
import textwrap

import httpx
import pydantic
import pytest
import wrapt
from openai import AsyncOpenAI, OpenAI
from openai.resources.chat.completions import AsyncCompletions, Completions

from revenium_middleware.perplexity import middleware as perplexity_middleware

TESTS_DIR = pathlib.Path(__file__).resolve().parents[1]
PERPLEXITY_REPORTED_COST = {"input_tokens_cost": 0.00011, "output_tokens_cost": 1e-05, "request_cost": 0.0025,
                            "search_queries_cost": 0.005, "total_cost": 0.00762}
PERPLEXITY_EXPECTED_COST = {"input_token_cost": 0.00011, "output_token_cost": 1e-05, "total_cost": 0.00762}
RECORDED_FIELDS = ("provider", "model", "input_token_count", "output_token_count", "transaction_id",
                   *PERPLEXITY_EXPECTED_COST)

_SCRIPT = textwrap.dedent("""
    import asyncio, json, sys
    import httpx
    from entry_points import calls

    modules, cost, fields = sys.argv[1].split(","), json.loads(sys.argv[2]), json.loads(sys.argv[3])
    recorder = calls._install_recorder(modules)
    import openai

    def handler_for(finish_reason):
        def handler(request):
            return httpx.Response(200, json={
                "id": "chatcmpl-4247", "object": "chat.completion", "created": 1, "model": calls.PERPLEXITY_MODEL,
                "choices": [{"index": 0, "finish_reason": finish_reason,
                             "message": {"role": "assistant", "content": '{"a": 1}'}}],
                "usage": {**calls._PERPLEXITY_USAGE, "cost": cost}})
        return handler

    def sync_parse(finish_reason):
        client = openai.OpenAI(api_key="stub", base_url=calls.PERPLEXITY_BASE_URL,
                               http_client=httpx.Client(transport=httpx.MockTransport(handler_for(finish_reason))))
        client.chat.completions.parse(**calls._chat_kwargs(calls.PERPLEXITY_MODEL),
                                      response_format=calls._structured_output())

    def async_parse(finish_reason):
        async def handler(request):
            await request.aread()
            return handler_for(finish_reason)(request)
        client = openai.AsyncOpenAI(api_key="stub", base_url=calls.PERPLEXITY_BASE_URL,
                                    http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
        asyncio.run(client.chat.completions.parse(**calls._chat_kwargs(calls.PERPLEXITY_MODEL),
                                                  response_format=calls._structured_output()))

    outcomes = {}
    for mode, parse in (("sync", sync_parse), ("async", async_parse)):
        for finish_reason in ("stop", "length"):
            recorder.reset_mock()
            try:
                parse(finish_reason)
                error = None
            except Exception as exc:
                error = type(exc).__name__
            calls.wait_for_metering()
            records = [{name: recorded.kwargs.get(name) for name in fields}
                       for recorded in recorder.ai.create_completion.call_args_list]
            outcomes[f"{mode}-{finish_reason}"] = {"error": error, "records": records}
    print(json.dumps(outcomes))
""")

CONFIGURATIONS = {
    "perplexity_only": ("revenium_middleware.perplexity",),
    "perplexity_then_openai": ("revenium_middleware.perplexity", "revenium_middleware.openai"),
    "openai_then_perplexity": ("revenium_middleware.openai", "revenium_middleware.perplexity"),
}
MODES = ["sync", "async"]


def _run(modules):
    env = {**os.environ, "PYTHONPATH": str(TESTS_DIR), "REVENIUM_METERING_API_KEY": "hak_test_perplexity_parse",
           "REVENIUM_METERING_BASE_URL": "http://127.0.0.1:9"}
    completed = subprocess.run(
        [sys.executable, "-c", _SCRIPT, ",".join(modules), json.dumps(PERPLEXITY_REPORTED_COST),
         json.dumps(RECORDED_FIELDS)],
        capture_output=True, text=True, timeout=300, env=env)
    assert completed.returncode == 0, completed.stderr[-4000:]
    return json.loads(completed.stdout.strip().splitlines()[-1])


@pytest.fixture(scope="module")
def outcomes():
    cache = {}

    def outcome(configuration, scenario):
        if configuration not in cache:
            cache[configuration] = _run(CONFIGURATIONS[configuration])
        return cache[configuration][scenario]
    return outcome


def _the_one_perplexity_record(outcome):
    assert len(outcome["records"]) == 1, outcome
    record = outcome["records"][0]
    assert record["provider"] == "PERPLEXITY"
    assert record["transaction_id"] == "chatcmpl-4247"
    return record


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("configuration", CONFIGURATIONS)
def test_a_parse_sends_one_perplexity_record_with_the_reported_cost(outcomes, configuration, mode):
    outcome = outcomes(configuration, f"{mode}-stop")

    assert outcome["error"] is None
    record = _the_one_perplexity_record(outcome)
    assert {field: record[field] for field in PERPLEXITY_EXPECTED_COST} == pytest.approx(PERPLEXITY_EXPECTED_COST)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("configuration", CONFIGURATIONS)
def test_a_parse_cut_off_by_the_length_limit_is_recorded_and_still_raised(outcomes, configuration, mode):
    outcome = outcomes(configuration, f"{mode}-length")

    assert outcome["error"] == "LengthFinishReasonError"
    _the_one_perplexity_record(outcome)


class Answer(pydantic.BaseModel):
    a: int


def _answer(request):
    return httpx.Response(200, json={
        "id": "chatcmpl-4247", "object": "chat.completion", "created": 1, "model": "sonar",
        "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": '{"a": 1}'}}],
        "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18}})


async def _async_answer(request):
    await request.aread()
    return _answer(request)


def _delegating_parse(self, **kwargs):
    """``parse()`` as a future openai release might write it: through ``create()``."""
    kwargs.pop("response_format")
    return self.create(**kwargs)


async def _async_delegating_parse(self, **kwargs):
    kwargs.pop("response_format")
    return await self.create(**kwargs)


def _sync_delegated(provider_requests):
    client = OpenAI(api_key="stub", base_url="https://api.perplexity.ai", max_retries=0,
                    http_client=httpx.Client(transport=httpx.MockTransport(provider_requests(_answer))))
    client.chat.completions.parse(model="sonar", messages=[{"role": "user", "content": "hi"}], response_format=Answer)


def _async_delegated(provider_requests):
    client = AsyncOpenAI(api_key="stub", base_url="https://api.perplexity.ai", max_retries=0,
                         http_client=httpx.AsyncClient(transport=httpx.MockTransport(provider_requests(_async_answer))))
    asyncio.run(client.chat.completions.parse(model="sonar", messages=[{"role": "user", "content": "hi"}],
                                              response_format=Answer))


class TestParseThatDelegatesToCreate:
    """If an openai release routed parse() through create(), the Perplexity
    parse and create wraps would still send one record, not two."""

    @pytest.fixture(autouse=True)
    def parse_delegates_to_create(self, monkeypatch):
        monkeypatch.setattr(Completions, "parse",
                            wrapt.FunctionWrapper(_delegating_parse, perplexity_middleware.parse_wrapper))
        monkeypatch.setattr(AsyncCompletions, "parse",
                            wrapt.FunctionWrapper(_async_delegating_parse, perplexity_middleware.async_parse_wrapper))

    @pytest.fixture
    def records(self, monkeypatch):
        sent = []
        monkeypatch.setattr(perplexity_middleware, "send_metering_data", lambda response, *args, **kwargs:
                            sent.append(response.id))
        return sent

    @pytest.fixture
    def requests(self):
        served = []

        def counted(answer):
            def handler(request):
                served.append(request.url.path)
                return answer(request)
            return handler
        counted.served = served
        return counted

    @pytest.mark.parametrize("call", [_sync_delegated, _async_delegated], ids=["sync", "async"])
    def test_one_record(self, records, requests, call):
        call(requests)

        assert requests.served == ["/chat/completions"]
        assert records == ["chatcmpl-4247"]

    @pytest.mark.parametrize("call", [_sync_delegated, _async_delegated], ids=["sync", "async"])
    def test_without_the_guard_both_wraps_would_record(self, records, requests, call, monkeypatch):
        monkeypatch.setattr(perplexity_middleware, "_metered_by_parse", contextlib.nullcontext)

        call(requests)

        assert records == ["chatcmpl-4247", "chatcmpl-4247"]


def _parse_that_cannot_validate(self, **kwargs):
    Answer.model_validate({"a": "not a number"})


async def _async_parse_that_cannot_validate(self, **kwargs):
    Answer.model_validate({"a": "not a number"})


class TestParseThatFailsToValidate:
    @pytest.fixture(autouse=True)
    def parse_raises_validation_error(self, monkeypatch):
        monkeypatch.setattr(Completions, "parse",
                            wrapt.FunctionWrapper(_parse_that_cannot_validate, perplexity_middleware.parse_wrapper))
        monkeypatch.setattr(AsyncCompletions, "parse",
                            wrapt.FunctionWrapper(_async_parse_that_cannot_validate,
                                                  perplexity_middleware.async_parse_wrapper))

    @pytest.mark.parametrize("call", [_sync_delegated, _async_delegated], ids=["sync", "async"])
    def test_the_unrecorded_call_is_warned_about_as_perplexity(self, caplog, call):
        with caplog.at_level("WARNING"), pytest.raises(pydantic.ValidationError):
            call(lambda answer: answer)

        warnings = [record for record in caplog.records if record.levelname == "WARNING"]
        assert [record.name for record in warnings] == ["revenium_middleware.perplexity"]
        assert warnings[0].getMessage().startswith("Perplexity answered and billed")
        assert "OpenAI" not in warnings[0].getMessage()
