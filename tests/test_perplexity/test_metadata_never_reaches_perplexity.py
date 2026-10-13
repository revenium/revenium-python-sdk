"""Revenium metadata never reaches Perplexity, in either form the README offers (BACK-4369).

Callers pass Revenium metadata as ``usage_metadata=`` or as
``extra_body={"usage_metadata": ...}``. On every Perplexity entry point the
SDK sees, the request Perplexity receives must hold neither, and a recorded
call must carry the metadata for its record and its spending-cap check.

Each import configuration runs once, in a fresh interpreter, with every
scenario below; the tests read its outcomes. Perplexity is an httpx
MockTransport that keeps every request body. The legacy OpenAI-client
``completions.create`` is not covered: Perplexity has no text-completion
endpoint, so it is not a Perplexity entry point.
"""
import json
import os
import pathlib
import subprocess
import sys
import textwrap

import pytest

from entry_points.caps import breached_rule, scoped

SUBSCRIBER_EMAIL = "jane.doe@example.com"
SUBSCRIBER_ID = "sub-4369"
CREDENTIAL_VALUE = "made-up-credential-4369"
ORGANIZATION = "AcmeCorp4369"

PERPLEXITY_ONLY = "perplexity_only"
PERPLEXITY_FIRST = "perplexity_first"
OPENAI_FIRST = "openai_first"
OPENAI_ONLY = "openai_only"
IMPORTS = {
    PERPLEXITY_ONLY: ["revenium_middleware.perplexity"],
    PERPLEXITY_FIRST: ["revenium_middleware.perplexity", "revenium_middleware.openai"],
    OPENAI_FIRST: ["revenium_middleware.openai", "revenium_middleware.perplexity"],
    OPENAI_ONLY: ["revenium_middleware.openai"],
}
BOTH_ORDERS = [PERPLEXITY_FIRST, OPENAI_FIRST]
WITH_THE_PERPLEXITY_INTEGRATION = [PERPLEXITY_ONLY, PERPLEXITY_FIRST, OPENAI_FIRST]
WITH_THE_OPENAI_INTEGRATION = [PERPLEXITY_FIRST, OPENAI_FIRST, OPENAI_ONLY]

OPENAI_CLIENT_CALLS = (
    "responses_create", "async_responses_create", "responses_stream_true", "responses_stream_helper",
    "async_responses_stream_helper", "responses_parse", "async_responses_parse", "responses_raw",
    "async_responses_raw", "embeddings", "async_embeddings", "chat_create", "chat_parse", "async_chat_parse",
)
NATIVE_UNMETERED_CALLS = (
    "native_search", "async_native_search", "native_embeddings", "async_native_embeddings",
    "native_contextualized_embeddings", "async_native_contextualized_embeddings",
    "native_async_chat", "async_native_async_chat", "native_browser_session", "async_native_browser_session",
)
NATIVE_CALLS = ("native_chat", "async_native_chat", "native_responses", "async_native_responses") + \
    NATIVE_UNMETERED_CALLS
FORMS = ("keyword", "extra_body")
OPENAI_ONLY_CALLS = ("responses_create", "responses_parse", "chat_parse", "embeddings")
RECORDED_RESPONSES_CALLS = ("responses_create", "async_responses_create", "responses_stream_true")
# The OpenAI client's stream() helpers take no extra keywords; only the openai
# integration wraps them, so without it the keyword form is refused before any request.
REFUSED_WITHOUT_THE_OPENAI_INTEGRATION = {"responses_stream_helper", "async_responses_stream_helper"}

_PROBE = textwrap.dedent("""
    import asyncio, copy, importlib, json, sys
    from unittest.mock import MagicMock, patch

    import httpx
    import pytest

    from entry_points.caps import record_cap_checks, seed_rules
    from entry_points.perplexity_cost import (
        ANSWER, PERPLEXITY_BASE_URL, PERPLEXITY_MODEL, _CHAT_BODY, _RESPONSES_BODY, _sse,
    )

    OPENAI_BASE_URL = "https://api.openai.com/v1"
    SETUP = json.loads(sys.argv[1])
    META = SETUP["metadata"]
    BODIES = []


    def _responses_events():
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


    EMBEDDINGS_BODY = {"object": "list", "model": "pplx-embed-v1-0.6b",
                       "data": [{"object": "embedding", "index": 0, "embedding": [0.1]}],
                       "usage": {"prompt_tokens": 2, "total_tokens": 2}}
    OTHER_BODY = {"id": "x", "results": [], "status": "CREATED", "model": PERPLEXITY_MODEL, "created_at": 1760000000}


    def handler(request):
        body = request.content.decode() if request.content else ""
        BODIES.append(body)
        path = request.url.path
        if path.endswith("/responses"):
            if json.loads(body or "{}").get("stream"):
                return httpx.Response(200, content=_responses_events(), headers={"content-type": "text/event-stream"})
            return httpx.Response(200, json=_RESPONSES_BODY)
        if path.endswith("/chat/completions") and "/async/" not in path:
            return httpx.Response(200, json=_CHAT_BODY)
        if path.endswith("/embeddings"):
            return httpx.Response(200, json=EMBEDDINGS_BODY)
        return httpx.Response(200, json=OTHER_BODY)


    def openai_client(base_url=PERPLEXITY_BASE_URL):
        import openai
        return openai.OpenAI(base_url=base_url, api_key="stub", max_retries=0,
                             http_client=httpx.Client(transport=httpx.MockTransport(handler)))


    def async_openai_client():
        import openai
        return openai.AsyncOpenAI(base_url=PERPLEXITY_BASE_URL, api_key="stub", max_retries=0,
                                  http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))


    def native_client():
        import perplexity
        return perplexity.Perplexity(api_key="stub", max_retries=0,
                                     http_client=httpx.Client(transport=httpx.MockTransport(handler)))


    def async_native_client():
        import perplexity
        return perplexity.AsyncPerplexity(api_key="stub", max_retries=0,
                                          http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))


    MESSAGES = [{"role": "user", "content": "hi"}]
    RESPONSES = {"model": PERPLEXITY_MODEL, "input": "hi"}
    CHAT = {"model": PERPLEXITY_MODEL, "messages": MESSAGES}
    EMBEDDINGS = {"model": "pplx-embed-v1-0.6b", "input": "hi"}
    CONTEXTUALIZED = {"model": "pplx-embed-context-v1-0.6b", "input": [["hi"]]}
    ASYNC_CHAT = {"request": CHAT}


    def run(coroutine):
        return asyncio.run(coroutine)


    def sync_stream_helper(**metadata):
        with openai_client().responses.stream(**RESPONSES, **metadata) as stream:
            list(stream)


    async def async_stream_helper(**metadata):
        async with async_openai_client().responses.stream(**RESPONSES, **metadata) as stream:
            [event async for event in stream]


    CALLS = {
        "responses_create": lambda m: openai_client().responses.create(**RESPONSES, **m),
        "async_responses_create": lambda m: run(async_openai_client().responses.create(**RESPONSES, **m)),
        "responses_stream_true": lambda m: list(openai_client().responses.create(stream=True, **RESPONSES, **m)),
        "responses_stream_helper": lambda m: sync_stream_helper(**m),
        "async_responses_stream_helper": lambda m: run(async_stream_helper(**m)),
        "responses_parse": lambda m: openai_client().responses.parse(**RESPONSES, **m),
        "async_responses_parse": lambda m: run(async_openai_client().responses.parse(**RESPONSES, **m)),
        "responses_raw": lambda m: openai_client().responses.with_raw_response.create(**RESPONSES, **m).parse(),
        "async_responses_raw": lambda m: run(
            async_openai_client().responses.with_raw_response.create(**RESPONSES, **m)),
        "embeddings": lambda m: openai_client().embeddings.create(**EMBEDDINGS, **m),
        "async_embeddings": lambda m: run(async_openai_client().embeddings.create(**EMBEDDINGS, **m)),
        "chat_create": lambda m: openai_client().chat.completions.create(**CHAT, **m),
        "chat_parse": lambda m: openai_client().chat.completions.parse(**CHAT, **m),
        "async_chat_parse": lambda m: run(async_openai_client().chat.completions.parse(**CHAT, **m)),
        "native_chat": lambda m: native_client().chat.completions.create(**CHAT, **m),
        "async_native_chat": lambda m: run(async_native_client().chat.completions.create(**CHAT, **m)),
        "native_responses": lambda m: native_client().responses.create(**RESPONSES, **m),
        "async_native_responses": lambda m: run(async_native_client().responses.create(**RESPONSES, **m)),
        "native_search": lambda m: native_client().search.create(query="hi", **m),
        "async_native_search": lambda m: run(async_native_client().search.create(query="hi", **m)),
        "native_embeddings": lambda m: native_client().embeddings.create(**EMBEDDINGS, **m),
        "async_native_embeddings": lambda m: run(async_native_client().embeddings.create(**EMBEDDINGS, **m)),
        "native_contextualized_embeddings": lambda m: native_client().contextualized_embeddings.create(
            **CONTEXTUALIZED, **m),
        "async_native_contextualized_embeddings": lambda m: run(
            async_native_client().contextualized_embeddings.create(**CONTEXTUALIZED, **m)),
        "native_async_chat": lambda m: native_client().async_.chat.completions.create(**ASYNC_CHAT, **m),
        "async_native_async_chat": lambda m: run(
            async_native_client().async_.chat.completions.create(**ASYNC_CHAT, **m)),
        "native_browser_session": lambda m: native_client().browser.sessions.create(**m),
        "async_native_browser_session": lambda m: run(async_native_client().browser.sessions.create(**m)),
        "plain_openai_responses_create": lambda m: openai_client("https://api.openai.com/v1").responses.create(
            **RESPONSES, **m),
    }


    def metadata_kwargs(form, metadata):
        if form == "keyword":
            return {"usage_metadata": copy.deepcopy(metadata)}
        return {"extra_body": {"usage_metadata": copy.deepcopy(metadata)}}


    def record_of(entry):
        subscriber = entry.kwargs.get("subscriber") or {}
        if not isinstance(subscriber, dict):
            subscriber = {key: getattr(subscriber, key, None) for key in ("id", "email")}
        return {"provider": entry.kwargs.get("provider"), "organization_name": entry.kwargs.get("organization_name"),
                "subscriber_id": subscriber.get("id"), "subscriber_email": subscriber.get("email")}


    monkeypatch = pytest.MonkeyPatch()
    checks = record_cap_checks(monkeypatch)
    recorder = MagicMock()
    with patch("revenium_middleware._core.metering.client", recorder):
        for module in SETUP["imports"]:
            importlib.import_module(module)
        from revenium_middleware._core.metering_pool import wait_until_idle

        sent = recorder.ai.create_completion.call_args_list
        outcomes = {}
        for scenario in SETUP["scenarios"]:
            scope = pytest.MonkeyPatch()
            scope.delenv("REVENIUM_CIRCUIT_BREAKER_ENABLED", raising=False)
            scope.delenv("REVENIUM_SELECTIVE_METERING", raising=False)
            if scenario["rules"]:
                seed_rules(scope, *scenario["rules"])
            if scenario["selective"]:
                scope.setenv("REVENIUM_SELECTIVE_METERING", "true")
            records_before, checks_before, bodies_before = len(sent), len(checks), len(BODIES)
            errors = []
            shared = scenario.get("shared_kwargs")
            for kwargs in ([shared] * scenario["repeat"] if shared is not None
                           else [metadata_kwargs(scenario["form"], META)] * scenario["repeat"]):
                try:
                    CALLS[scenario["call"]](kwargs)
                    errors.append(None)
                except Exception as exc:
                    errors.append(type(exc).__name__)
            wait_until_idle(10.0)
            scope.undo()
            outcomes[scenario["name"]] = {
                "errors": errors,
                "bodies": BODIES[bodies_before:],
                "records": [record_of(entry) for entry in sent[records_before:]],
                "cap_checks": len(checks) - checks_before,
                "shared_kwargs_after": shared,
            }
    print(json.dumps(outcomes))
""")

METADATA = {
    "subscriber": {"id": SUBSCRIBER_ID, "email": SUBSCRIBER_EMAIL,
                   "credential": {"name": "prod-key", "value": CREDENTIAL_VALUE}},
    "organizationName": ORGANIZATION,
}
BREACHED_FOR_THE_SUBSCRIBER = breached_rule(4369, scoped("SUBSCRIBER", "IS", SUBSCRIBER_EMAIL))
ALLOWING = breached_rule(4369, scoped("AGENT", "IS", "nobody"))


def _scenario(name, call, form="extra_body", rules=(), selective=False, repeat=1, **extra):
    return {"name": name, "call": call, "form": form, "rules": list(rules), "selective": selective,
            "repeat": repeat, **extra}


def _scenarios():
    scenarios = [_scenario(f"leak:{form}:{call}", call, form)
                 for form in FORMS for call in OPENAI_CLIENT_CALLS + NATIVE_CALLS]
    scenarios += [_scenario(f"cap:{call}", call, rules=[BREACHED_FOR_THE_SUBSCRIBER])
                  for call in RECORDED_RESPONSES_CALLS]
    scenarios += [_scenario(f"selective_cap:{call}", call, rules=[BREACHED_FOR_THE_SUBSCRIBER], selective=True)
                  for call in ("responses_create", "responses_parse", "embeddings")]
    scenarios += [_scenario(f"selective:{call}", call, selective=True) for call in ("responses_create", "embeddings")]
    scenarios += [_scenario(f"native_guard:{call}", call, rules=[ALLOWING]) for call in NATIVE_UNMETERED_CALLS]
    scenarios.append(_scenario("plain_openai", "plain_openai_responses_create"))
    conflicting = {"usage_metadata": {"organizationName": "K"},
                   "extra_body": {"usage_metadata": {**METADATA, "organizationName": "E"}}}
    scenarios.append(_scenario("keyword_wins:create", "responses_create", shared_kwargs=conflicting))
    scenarios.append(_scenario("keyword_wins:stream_helper", "responses_stream_helper", shared_kwargs=conflicting))
    scenarios.append(_scenario("reused_extra_body", "responses_create", repeat=2,
                               shared_kwargs={"extra_body": {"usage_metadata": METADATA}}))
    return scenarios


def _run_probe(config):
    env = {**os.environ, "PYTHONPATH": str(pathlib.Path(__file__).resolve().parents[1]),
           "REVENIUM_METERING_API_KEY": "hak_test_perplexity_metadata",
           "REVENIUM_METERING_BASE_URL": "http://127.0.0.1:9"}
    setup = json.dumps({"imports": IMPORTS[config], "metadata": METADATA, "scenarios": _scenarios()})
    completed = subprocess.run([sys.executable, "-c", _PROBE, setup],
                               capture_output=True, text=True, timeout=600, env=env)
    assert completed.returncode == 0, completed.stderr[-4000:]
    return json.loads(completed.stdout.strip().splitlines()[-1])


_OUTCOMES = {}


@pytest.fixture(scope="module")
def outcomes():
    def of(config):
        if config not in _OUTCOMES:
            _OUTCOMES[config] = _run_probe(config)
        return _OUTCOMES[config]
    return of


def _leaks(body):
    return [marker for marker in ("usage_metadata", SUBSCRIBER_EMAIL, CREDENTIAL_VALUE) if marker in body]


def _assert_clean(outcome, label):
    assert outcome["bodies"], f"{label}: no request reached Perplexity"
    for body in outcome["bodies"]:
        assert _leaks(body) == [], f"{label} sent {_leaks(body)} to Perplexity"


def _assert_recorded_with_the_subscriber(outcome, label, organization=ORGANIZATION):
    assert outcome["errors"] == [None] * len(outcome["errors"]), label
    assert len(outcome["records"]) == len(outcome["errors"]), label
    for record in outcome["records"]:
        assert record["subscriber_email"] == SUBSCRIBER_EMAIL, label
        assert record["subscriber_id"] == SUBSCRIBER_ID, label
        assert record["organization_name"] == organization, label


@pytest.mark.parametrize("form", FORMS)
@pytest.mark.parametrize("config", WITH_THE_PERPLEXITY_INTEGRATION)
def test_no_perplexity_request_carries_revenium_metadata(outcomes, config, form):
    result = outcomes(config)

    for call in OPENAI_CLIENT_CALLS + NATIVE_CALLS:
        outcome = result[f"leak:{form}:{call}"]
        label = f"{config} {form} {call}"
        if form == "keyword" and config == PERPLEXITY_ONLY and call in REFUSED_WITHOUT_THE_OPENAI_INTEGRATION:
            assert outcome["errors"] == ["TypeError"], label
            assert outcome["bodies"] == [], label
            continue
        assert outcome["errors"] == [None], f"{label}: {outcome['errors']}"
        _assert_clean(outcome, label)


@pytest.mark.parametrize("form", ["extra_body"])
def test_openai_only_perplexity_bound_calls_carry_no_revenium_metadata(outcomes, form):
    result = outcomes(OPENAI_ONLY)

    for call in OPENAI_ONLY_CALLS:
        outcome = result[f"leak:{form}:{call}"]
        assert outcome["errors"] == [None], call
        _assert_clean(outcome, call)


@pytest.mark.parametrize("config", list(IMPORTS))
def test_a_responses_call_with_metadata_in_extra_body_is_recorded_with_its_subscriber_and_organization(
        outcomes, config):
    result = outcomes(config)

    for call in RECORDED_RESPONSES_CALLS:
        outcome = result[f"leak:extra_body:{call}"]
        _assert_recorded_with_the_subscriber(outcome, call)
        assert [record["provider"] for record in outcome["records"]] == ["PERPLEXITY"], call


@pytest.mark.parametrize("config", list(IMPORTS))
def test_a_breached_subscriber_cap_refuses_a_responses_call_with_metadata_in_extra_body(outcomes, config):
    result = outcomes(config)

    for call in RECORDED_RESPONSES_CALLS:
        outcome = result[f"cap:{call}"]
        assert outcome["errors"] == ["BudgetExceededError"], call
        assert outcome["records"] == [], call
        assert outcome["bodies"] == [], call


@pytest.mark.parametrize("config", WITH_THE_OPENAI_INTEGRATION)
def test_a_breached_subscriber_cap_refuses_a_selectively_unmetered_responses_call_with_metadata_in_extra_body(
        outcomes, config):
    result = outcomes(config)

    for call in ("responses_create", "responses_parse", "embeddings"):
        outcome = result[f"selective_cap:{call}"]
        assert outcome["errors"] == ["BudgetExceededError"], call
        assert outcome["records"] == [], call
        assert outcome["bodies"] == [], call


@pytest.mark.parametrize("config", BOTH_ORDERS)
def test_openai_client_embeddings_to_perplexity_record_the_extra_body_subscriber(outcomes, config):
    result = outcomes(config)

    for call in ("embeddings", "async_embeddings"):
        _assert_recorded_with_the_subscriber(result[f"leak:extra_body:{call}"], call)


@pytest.mark.parametrize("config", BOTH_ORDERS)
def test_selectively_unmetered_perplexity_calls_carry_no_revenium_metadata(outcomes, config):
    result = outcomes(config)

    for call in ("responses_create", "embeddings"):
        outcome = result[f"selective:{call}"]
        assert outcome["errors"] == [None], call
        assert outcome["records"] == [], call
        _assert_clean(outcome, call)


@pytest.mark.parametrize("config", list(IMPORTS))
def test_keyword_metadata_wins_over_extra_body_on_perplexity_responses(outcomes, config):
    result = outcomes(config)

    scenarios = ["keyword_wins:create"]
    if config != PERPLEXITY_ONLY:
        scenarios.append("keyword_wins:stream_helper")
    for scenario in scenarios:
        outcome = result[scenario]
        _assert_recorded_with_the_subscriber(outcome, scenario, organization="K")
        _assert_clean(outcome, scenario)


@pytest.mark.parametrize("config", list(IMPORTS))
def test_a_reused_extra_body_dict_keeps_its_metadata_across_perplexity_responses_calls(outcomes, config):
    outcome = outcomes(config)["reused_extra_body"]

    _assert_recorded_with_the_subscriber(outcome, "reused_extra_body")
    assert len(outcome["records"]) == 2
    _assert_clean(outcome, "reused_extra_body")
    assert outcome["shared_kwargs_after"] == {"extra_body": {"usage_metadata": METADATA}}


def test_native_calls_the_sdk_does_not_meter_send_no_record(outcomes):
    result = outcomes(PERPLEXITY_ONLY)

    for call in NATIVE_UNMETERED_CALLS:
        outcome = result[f"native_guard:{call}"]
        assert outcome["errors"] == [None], call
        assert outcome["records"] == [], call
        assert outcome["cap_checks"] == 0, call


@pytest.mark.parametrize("config", BOTH_ORDERS)
def test_a_plain_openai_responses_call_keeps_extra_body_untouched(outcomes, config):
    outcome = outcomes(config)["plain_openai"]

    assert outcome["errors"] == [None]
    assert [json.loads(body) for body in outcome["bodies"]] == [
        {"model": "sonar", "input": "hi", "usage_metadata": METADATA}]
