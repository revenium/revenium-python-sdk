"""Provider paths the entry-point matrix does not reach still check a breached cap first (BACK-4190).

The Bedrock fast path, calls selective metering leaves unmetered, and the
paths that only exist behind a flag are each driven here with one breached
rule seeded in the enforcement cache. A refused call raises
``BudgetExceededError`` before the provider is contacted. On the async
surfaces the check also never fetches rules on, or holds, the caller's loop.
"""
import asyncio
import io
import json
import os
import pathlib
import subprocess
import sys
import textwrap
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import anthropic
import pytest

from entry_points import calls
from entry_points.caps import breached_rule, record_cap_checks, scoped, seed_rules
from entry_points.instrumented import import_all_middleware
from entry_points.stub_requests import count_provider_requests

from .enforcement_stub_server import (
    FAILURE_MODES, EnforcementStub, drive_on_loop, point_enforcement_at, shut_down, wait_until,
)

import_all_middleware()

from revenium_middleware._core import call_ownership, enforcement  # noqa: E402
from revenium_middleware._core.decorators import revenium_metadata  # noqa: E402
from revenium_middleware._core.exceptions import BudgetExceededError  # noqa: E402
from revenium_middleware.anthropic import middleware as anthropic_middleware  # noqa: E402
from revenium_middleware.anthropic.provider import Provider  # noqa: E402

RULE_ID = 4190
REQUESTED_CLAUDE = "claude-3-sonnet-20240229"
FAL_MODEL = "fal_ai/" + calls.FAL_APP


def cap(*filters, action="BLOCK"):
    return breached_rule(RULE_ID, *filters, action=action)


@pytest.fixture
def seed(monkeypatch):
    return lambda *rules: seed_rules(monkeypatch, *rules)


def middleware_fn(obj):
    return getattr(obj, "_self_wrapper", obj)


@pytest.fixture
def provider_requests(monkeypatch):
    return count_provider_requests(monkeypatch)


@pytest.fixture
def selective_metering(monkeypatch):
    monkeypatch.setenv("REVENIUM_SELECTIVE_METERING", "true")


@pytest.fixture
def released_stub_response_id():
    # The stub answers every Responses call with the same id, which the metered call claims; a later test reuses it.
    yield
    call_ownership.reset_claimed_response_ids()


BEDROCK_FAST_PATHS = [
    pytest.param(anthropic_middleware.create_wrapper, "_handle_bedrock_request", id="create"),
    pytest.param(anthropic_middleware.stream_wrapper, "_handle_bedrock_stream_request", id="stream"),
]


@pytest.mark.parametrize("wrapper, handler", BEDROCK_FAST_PATHS)
def test_anthropic_bedrock_fast_path_is_refused_before_boto3(wrapper, handler, seed):
    instance = SimpleNamespace(_client=SimpleNamespace(aws_region="us-east-1"))
    wrapped = MagicMock()

    def call():
        middleware_fn(wrapper)(wrapped, instance, (), {"model": REQUESTED_CLAUDE, "max_tokens": 5,
                                                       "messages": calls.USER_MESSAGES})

    with patch.object(anthropic_middleware, "detect_provider", return_value=Provider.BEDROCK), \
            patch.object(anthropic_middleware, handler) as bedrock:
        seed(cap(scoped("PROVIDER", "IS", "AWS"), scoped("MODEL", "IS", REQUESTED_CLAUDE)))
        with pytest.raises(BudgetExceededError):
            call()
        bedrock.assert_not_called()

        seed(cap(scoped("PROVIDER", "IS", "ANTHROPIC")))
        call()
        bedrock.assert_called_once()
    wrapped.assert_not_called()


def test_anthropic_foundry_call_is_checked_under_the_foundry_label(seed):
    wrapped = MagicMock(return_value=SimpleNamespace(id="msg_1", usage=None))

    def call():
        middleware_fn(anthropic_middleware.create_wrapper)(
            wrapped, SimpleNamespace(_client=None), (), {"model": "claude-sonnet-4-5", "max_tokens": 5,
                                                         "messages": calls.USER_MESSAGES})

    with patch.object(anthropic_middleware, "detect_provider", return_value=Provider.FOUNDRY), \
            patch.object(anthropic_middleware, "_meter_message"):
        seed(cap(scoped("PROVIDER", "IS", "Foundry")))
        with pytest.raises(BudgetExceededError):
            call()
        wrapped.assert_not_called()

        seed(cap(scoped("PROVIDER", "IS_NOT", "Foundry")))
        call()
    wrapped.assert_called_once()


SELECTIVELY_UNMETERED = [
    pytest.param("anthropic.messages.create.sync", "ANTHROPIC", calls.ANTHROPIC_MODEL, id="anthropic"),
    pytest.param("genai.models.generate_content.sync", "Google", calls.GEMINI_MODEL, id="genai"),
    pytest.param("ollama.client.chat.sync", "OLLAMA", calls.OLLAMA_MODEL, id="ollama"),
    pytest.param("perplexity.openai_compatible.chat.create.sync", "PERPLEXITY", calls.PERPLEXITY_MODEL,
                 id="perplexity_openai"),
    pytest.param("perplexity.native.chat.create.sync", "PERPLEXITY", calls.PERPLEXITY_MODEL, id="perplexity_native"),
    pytest.param("fal.run.sync", "fal_ai", FAL_MODEL, id="fal"),
]


@pytest.mark.parametrize("call_id, provider, model", SELECTIVELY_UNMETERED)
def test_selectively_unmetered_calls_are_still_refused(call_id, provider, model, selective_metering,
                                                      seed, provider_requests, mock_revenium_client):
    seed(cap(scoped("PROVIDER", "IS", provider), scoped("MODEL", "IS", model)))
    with pytest.raises(BudgetExceededError):
        calls.run_call(call_id)
    assert sum(provider_requests.values()) == 0

    seed(cap(scoped("PROVIDER", "IS_NOT", provider)))
    calls.run_call(call_id)
    assert sum(provider_requests.values()) >= 1
    calls.wait_for_metering()
    assert calls.recorded_payloads(mock_revenium_client) == []


CLAUDE_ON_BEDROCK = "us.anthropic.claude-3-5-sonnet-20241022-v2:0"


@pytest.mark.parametrize("operation", ["InvokeModel", "Converse"])
def test_bedrock_transport_refuses_a_raw_boto3_call(operation, monkeypatch, seed):
    from revenium_middleware.anthropic import bedrock_transport

    monkeypatch.setenv("REVENIUM_BEDROCK_TRANSPORT", "1")
    instance = SimpleNamespace(meta=SimpleNamespace(service_model=SimpleNamespace(service_name="bedrock-runtime"),
                                                    region_name="us-east-1"))
    boto3_call = MagicMock(return_value={"ResponseMetadata": {"RequestId": "req-1"}})

    def call():
        bedrock_transport._make_api_call_wrapper(boto3_call, instance, (operation, {"modelId": CLAUDE_ON_BEDROCK}), {})

    with patch.object(bedrock_transport, "_observe_response"):
        seed(cap(scoped("PROVIDER", "IS", "AWS"), scoped("MODEL", "IS", CLAUDE_ON_BEDROCK)))
        with pytest.raises(BudgetExceededError):
            call()
        boto3_call.assert_not_called()

        seed(cap(scoped("PROVIDER", "IS_NOT", "AWS")))
        call()
    boto3_call.assert_called_once()


def _invoke_model_answer():
    from botocore.response import StreamingBody

    raw = json.dumps({"id": "msg_1", "stop_reason": "end_turn", "usage": {"input_tokens": 3, "output_tokens": 2}})
    return {"ResponseMetadata": {"RequestId": "req-1"}, "body": StreamingBody(io.BytesIO(raw.encode()), len(raw))}


def _converse_answer():
    return {"ResponseMetadata": {"RequestId": "req-1"}, "stopReason": "end_turn",
            "usage": {"inputTokens": 3, "outputTokens": 2, "totalTokens": 5},
            "output": {"message": {"role": "assistant", "content": []}}}


@pytest.mark.parametrize("operation, answer", [
    pytest.param("InvokeModel", _invoke_model_answer, id="InvokeModel"),
    pytest.param("Converse", _converse_answer, id="Converse"),
])
def test_bedrock_transport_refuses_a_selectively_unmetered_raw_boto3_call(
        operation, answer, selective_metering, monkeypatch, seed):
    """BACK-4320: a raw boto3 call outside a @revenium_meter function is still capped under the
    LangChain attribution, then passed through unrecorded, leaving that attribution for the record."""
    from revenium_middleware._core.call_ownership import (
        BEDROCK, _published_in_context, peek_callback_metadata, publish_callback_metadata,
    )
    from revenium_middleware.anthropic import bedrock_transport

    monkeypatch.setenv("REVENIUM_BEDROCK_TRANSPORT", "1")
    instance = SimpleNamespace(meta=SimpleNamespace(service_model=SimpleNamespace(service_name="bedrock-runtime"),
                                                    region_name="us-east-1"))
    boto3_call = MagicMock(side_effect=lambda *args, **kwargs: answer())

    def call():
        bedrock_transport._make_api_call_wrapper(boto3_call, instance, (operation, {"modelId": CLAUDE_ON_BEDROCK}), {})

    publications = _published_in_context.set(())
    try:
        publish_callback_metadata({"agent": "langchain-agent"}, providers={BEDROCK})
        with patch.object(bedrock_transport, "_emit_completion") as emit:
            seed(cap(scoped("PROVIDER", "IS", "AWS"), scoped("MODEL", "IS", CLAUDE_ON_BEDROCK)))
            with pytest.raises(BudgetExceededError):
                call()
            boto3_call.assert_not_called()

            seed(cap(scoped("PROVIDER", "IS_NOT", "AWS")))
            call()
        boto3_call.assert_called_once()
        emit.assert_not_called()
        assert peek_callback_metadata(BEDROCK) == {"agent": "langchain-agent"}
    finally:
        _published_in_context.reset(publications)


LANGCHAIN_ATTRIBUTED_PERPLEXITY = [
    pytest.param("perplexity.openai_compatible.chat.create.sync", "openai", id="openai_compatible"),
    pytest.param("perplexity.native.chat.create.sync", "perplexity", id="native"),
]


@pytest.mark.parametrize("call_id, transport", LANGCHAIN_ATTRIBUTED_PERPLEXITY)
def test_an_unmetered_perplexity_call_reads_the_langchain_attribution_without_taking_it(
        call_id, transport, selective_metering, seed, provider_requests):
    from revenium_middleware._core.call_ownership import peek_callback_metadata, publish_callback_metadata

    publish_callback_metadata({"agent": "langchain-agent"}, providers={transport})
    seed(cap(scoped("AGENT", "IS", "langchain-agent")))

    with pytest.raises(BudgetExceededError):
        calls.run_call(call_id)
    assert sum(provider_requests.values()) == 0
    assert peek_callback_metadata(transport) == {"agent": "langchain-agent"}


def _perplexity_bound():
    return SimpleNamespace(_client=SimpleNamespace(base_url="https://api.perplexity.ai"))


def _run_wrapper(wrapper, wrapped, kwargs):
    result = wrapper(wrapped, _perplexity_bound(), (), kwargs)
    if asyncio.iscoroutine(result):
        return asyncio.run(result)
    return result


PERPLEXITY_COMPATIBLE_WRAPPERS = [pytest.param("create_wrapper", id="sync"),
                                  pytest.param("async_create_wrapper", id="async")]


@pytest.mark.parametrize("wrapper_name", PERPLEXITY_COMPATIBLE_WRAPPERS)
def test_perplexity_compatible_call_is_refused_without_the_openai_middleware(wrapper_name, seed, monkeypatch):
    """The OpenAI-compatible Perplexity wrapper on its own, as when only
    revenium_middleware.perplexity is imported (the matrix runs that row in a subprocess)."""
    from revenium_middleware.perplexity import middleware as perplexity_middleware

    monkeypatch.setattr(perplexity_middleware, "meter_call", lambda call, response, *rest: response)
    wrapped = AsyncMock() if wrapper_name.startswith("async") else MagicMock()
    wrapper = getattr(perplexity_middleware, wrapper_name)

    seed(cap(scoped("PROVIDER", "IS", "PERPLEXITY"), scoped("MODEL", "IS", calls.PERPLEXITY_MODEL)))
    with pytest.raises(BudgetExceededError):
        _run_wrapper(wrapper, wrapped, {"model": calls.PERPLEXITY_MODEL, "messages": calls.USER_MESSAGES})
    wrapped.assert_not_called()

    seed(cap(scoped("PROVIDER", "IS_NOT", "PERPLEXITY")))
    _run_wrapper(wrapper, wrapped, {"model": calls.PERPLEXITY_MODEL, "messages": calls.USER_MESSAGES})
    wrapped.assert_called_once()


@pytest.mark.parametrize("wrapper_name", PERPLEXITY_COMPATIBLE_WRAPPERS)
def test_perplexity_call_owned_by_an_outer_wrapper_is_checked_and_passed_on_intact(wrapper_name, seed, monkeypatch):
    from revenium_middleware.perplexity import middleware as perplexity_middleware

    monkeypatch.setattr(perplexity_middleware, "call_owned_by_outer_wrapper", lambda: True)
    wrapped = AsyncMock() if wrapper_name.startswith("async") else MagicMock()
    wrapper = getattr(perplexity_middleware, wrapper_name)

    def kwargs():
        return {"model": calls.PERPLEXITY_MODEL, "messages": calls.USER_MESSAGES,
                "usage_metadata": {"agent": "outer-agent"}}

    seed(cap(scoped("AGENT", "IS", "outer-agent"), scoped("PROVIDER", "IS", "PERPLEXITY")))
    with pytest.raises(BudgetExceededError):
        _run_wrapper(wrapper, wrapped, kwargs())
    wrapped.assert_not_called()

    seed(cap(scoped("AGENT", "IS_NOT", "outer-agent")))
    _run_wrapper(wrapper, wrapped, kwargs())
    assert wrapped.call_args.kwargs["usage_metadata"] == {"agent": "outer-agent"}


PERPLEXITY_PARSE_CALLS = ("perplexity.openai_compatible.chat.parse.sync", "perplexity.openai_compatible.chat.parse.async")
_PERPLEXITY_ONLY_PARSE_UNDER_A_CAP = textwrap.dedent("""
    import json, sys
    import pytest
    from entry_points import calls
    from entry_points.caps import breached_rule, scoped, seed_rules
    from entry_points.stub_requests import count_provider_requests
    import revenium_middleware.perplexity

    monkeypatch = pytest.MonkeyPatch()
    seed_rules(monkeypatch, breached_rule(4247, scoped("PROVIDER", "IS", "PERPLEXITY")))
    requests = count_provider_requests(monkeypatch)
    outcomes = {}
    for call_id in sys.argv[1].split(","):
        before = requests["httpx"]
        try:
            calls.run_call(call_id)
            error = None
        except Exception as exc:
            error = type(exc).__name__
        outcomes[call_id] = {"error": error, "provider_requests": requests["httpx"] - before}
    print(json.dumps({"outcomes": outcomes,
                      "openai_middleware_loaded": "revenium_middleware.openai.middleware" in sys.modules}))
""")


def test_perplexity_parse_is_refused_with_only_the_perplexity_middleware():
    """The README setup imports only revenium_middleware.perplexity (BACK-4247)."""
    env = {**os.environ, "PYTHONPATH": str(pathlib.Path(__file__).resolve().parents[1]),
           "REVENIUM_METERING_API_KEY": "hak_test_perplexity_parse_cap",
           "REVENIUM_METERING_BASE_URL": "http://127.0.0.1:9"}
    completed = subprocess.run(
        [sys.executable, "-c", _PERPLEXITY_ONLY_PARSE_UNDER_A_CAP, ",".join(PERPLEXITY_PARSE_CALLS)],
        capture_output=True, text=True, timeout=300, env=env)
    assert completed.returncode == 0, completed.stderr[-4000:]
    result = json.loads(completed.stdout.strip().splitlines()[-1])

    assert result["openai_middleware_loaded"] is False
    assert result["outcomes"] == {call_id: {"error": "BudgetExceededError", "provider_requests": 0}
                                  for call_id in PERPLEXITY_PARSE_CALLS}


def _fal_subscribe(**options):
    import fal_client
    with calls._fal_key(), calls._stubbed_http(fal_client.sync_client, calls._fal_sync_http()):
        fal_client.subscribe(calls.FAL_APP, arguments=dict(calls._FAL_ARGUMENTS), **options)


def _fal_subscribe_async():
    import fal_client

    async def subscribe():
        with calls._fal_key(), calls._stubbed_http(fal_client.async_client, calls._fal_async_http()):
            await fal_client.subscribe_async(calls.FAL_APP, arguments=dict(calls._FAL_ARGUMENTS))

    asyncio.run(subscribe())


FAL_SUBSCRIBES = [
    pytest.param(_fal_subscribe, id="sync"),
    pytest.param(lambda: _fal_subscribe(client_timeout=30), id="sync_client_timeout"),
    pytest.param(_fal_subscribe_async, id="async"),
]


@pytest.mark.parametrize("subscribe", FAL_SUBSCRIBES)
def test_fal_subscribe_is_checked_once(subscribe, seed, monkeypatch):
    """subscribe submits through the wrapped submit, on fal's executor thread when
    client_timeout is set; the job is checked once, with the caller's identity."""
    seed(cap(scoped("AGENT", "IS_NOT", "fal-agent")))
    checks = record_cap_checks(monkeypatch)

    @revenium_metadata(agent="fal-agent")
    def decorated():
        subscribe()

    decorated()
    assert [(check.model, check.provider, check.usage_metadata.get("agent")) for check in checks] == [
        (FAL_MODEL, "fal_ai", "fal-agent")]


OTHER_FAL_APP = "fal-ai/other"


def _sync_subscribe(direct_submit):
    from revenium_middleware.fal import middleware as fal_middleware
    inner_submit = MagicMock(return_value=SimpleNamespace(request_id="req-subscribed"))

    def fal_subscribe(application, arguments, **options):
        direct_submit()
        middleware_fn(fal_middleware.submit_wrapper)(inner_submit, None, (application, arguments), {})

    return lambda arguments: middleware_fn(fal_middleware.subscribe_wrapper)(
        fal_subscribe, None, (calls.FAL_APP, arguments), {}), inner_submit


def _async_subscribe(direct_submit):
    from revenium_middleware.fal import middleware as fal_middleware
    inner_submit = AsyncMock(return_value=SimpleNamespace(request_id="req-subscribed"))

    async def fal_subscribe(application, arguments, **options):
        await direct_submit()
        await middleware_fn(fal_middleware.submit_async_wrapper)(inner_submit, None, (application, arguments), {})

    return lambda arguments: asyncio.run(middleware_fn(fal_middleware.subscribe_async_wrapper)(
        fal_subscribe, None, (calls.FAL_APP,), {"arguments": arguments})), inner_submit


def _direct_sync_submit(provider_submit, arguments, refusals):
    from revenium_middleware.fal import middleware as fal_middleware

    def submit():
        try:
            middleware_fn(fal_middleware.submit_wrapper)(
                provider_submit, None, (OTHER_FAL_APP, arguments), {"usage_metadata": {"agent": "direct-agent"}})
        except BudgetExceededError as error:
            refusals.append(error)
    return submit


def _direct_async_submit(provider_submit, arguments, refusals):
    from revenium_middleware.fal import middleware as fal_middleware

    async def submit():
        try:
            await middleware_fn(fal_middleware.submit_async_wrapper)(
                provider_submit, None, (OTHER_FAL_APP, arguments), {"usage_metadata": {"agent": "direct-agent"}})
        except BudgetExceededError as error:
            refusals.append(error)
    return submit


@pytest.mark.parametrize("make_subscribe, make_direct_submit, provider_submit", [
    pytest.param(_sync_subscribe, _direct_sync_submit, MagicMock, id="sync"),
    pytest.param(_async_subscribe, _direct_async_submit, AsyncMock, id="async"),
])
def test_fal_direct_submit_reusing_a_subscribes_arguments_runs_its_own_check(
        make_subscribe, make_direct_submit, provider_submit, seed, monkeypatch):
    """A direct submit made with the caller's dict while subscribe's own submit is
    still pending cannot take subscribe's already-checked mark."""
    seed(cap(scoped("MODEL", "IS", "fal_ai/" + OTHER_FAL_APP)))
    checks = record_cap_checks(monkeypatch)
    arguments = dict(calls._FAL_ARGUMENTS)
    direct_provider_submit, refusals = provider_submit(), []
    subscribe, inner_submit = make_subscribe(make_direct_submit(direct_provider_submit, arguments, refusals))

    subscribe(arguments)

    assert len(refusals) == 1
    direct_provider_submit.assert_not_called()
    inner_submit.assert_called_once()
    assert [(check.model, check.usage_metadata.get("agent")) for check in checks] == [
        (FAL_MODEL, None), ("fal_ai/" + OTHER_FAL_APP, "direct-agent")]


LANGCHAIN_CALLBACK_WARNING = "Error in UnifiedReveniumCallbackHandler.on_chat_model_start callback"


def _langchain_handler(agent):
    from revenium_middleware.openai.langchain import ReveniumCallbackHandler
    return ReveniumCallbackHandler(usage_metadata={"agent": agent})


def _with_ls_provider(model_class, ls_provider):
    class Relabelled(model_class):
        def _get_ls_params(self, stop=None, **kwargs):
            params = super()._get_ls_params(stop=stop, **kwargs)
            params["ls_provider"] = ls_provider
            return params

    return Relabelled


@pytest.mark.parametrize("invoke", [
    pytest.param(lambda model: model.invoke("hi"), id="invoke"),
    pytest.param(lambda model: asyncio.run(model.ainvoke("hi")), id="ainvoke"),
])
def test_langchain_callback_refuses_a_model_no_wrap_covers(invoke, seed, caplog):
    from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
    from langchain_core.messages import AIMessage
    from revenium_middleware._core import call_ownership

    answers = iter([AIMessage(content="hello")])
    model = _with_ls_provider(GenericFakeChatModel, "cohere")(messages=answers,
                                                              callbacks=[_langchain_handler("cohere-agent")])
    seed(cap(scoped("PROVIDER", "IS", "COHERE"), scoped("AGENT", "IS", "cohere-agent")))
    published_before = call_ownership._open_publications()

    with pytest.raises(BudgetExceededError):
        invoke(model)
    assert next(answers).content == "hello", "the model was called"
    assert call_ownership._open_publications() == published_before
    assert any(record.levelname == "WARNING" and LANGCHAIN_CALLBACK_WARNING in record.getMessage()
               for record in caplog.records)

    answers = iter([AIMessage(content="hello")])
    model.messages = answers
    seed(cap(scoped("PROVIDER", "IS_NOT", "COHERE")))
    assert invoke(model).content == "hello"


def _chat_openai(model_class, agent):
    import httpx
    return model_class(model=calls.OPENAI_MODEL, api_key="stub", max_retries=0, callbacks=[_langchain_handler(agent)],
                       http_client=httpx.Client(transport=httpx.MockTransport(
                           calls._openai_compatible_handler(calls.OPENAI_MODEL))))


def test_langchain_callback_leaves_the_check_to_an_installed_transport(seed, monkeypatch):
    from langchain_openai import ChatOpenAI

    seed(cap(scoped("AGENT", "IS", "nobody")))
    checks = record_cap_checks(monkeypatch)

    _chat_openai(ChatOpenAI, "openai-agent").invoke("hi")

    assert [(check.provider, check.usage_metadata.get("agent")) for check in checks] == [("OPENAI", "openai-agent")]


OWN_SDK_VENDORS = ["groq", "fireworks", "mistral", "a-vendor-revenium-does-not-know"]


@pytest.mark.parametrize("ls_provider", OWN_SDK_VENDORS)
@pytest.mark.parametrize("invoke", [
    pytest.param(lambda model: model.invoke("hi"), id="invoke"),
    pytest.param(lambda model: asyncio.run(model.ainvoke("hi")), id="ainvoke"),
])
def test_langchain_callback_refuses_a_vendor_on_its_own_sdk(invoke, ls_provider, seed):
    """BACK-4220: ChatGroq, ChatFireworks and ChatMistralAI call their own SDKs,
    which no Revenium wrap checks, and so does a vendor Revenium has never heard of."""
    from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
    from langchain_core.messages import AIMessage

    answers = iter([AIMessage(content="hello")])
    model = _with_ls_provider(GenericFakeChatModel, ls_provider)(
        messages=answers, callbacks=[_langchain_handler("own-sdk-agent")])
    seed(cap(scoped("AGENT", "IS", "own-sdk-agent")))

    with pytest.raises(BudgetExceededError):
        invoke(model)
    assert next(answers).content == "hello", "the model was called"


@pytest.mark.parametrize("ls_provider, judged_as", [
    ("groq", "OPENAI"), ("mistral", "MISTRAL"), ("a-vendor-revenium-does-not-know", "OPENAI"),
])
def test_an_unlisted_vendor_is_not_refused_by_another_providers_cap(ls_provider, judged_as, seed, monkeypatch):
    """BACK-4220, BACK-4258: the callback checks under the label the record is sent with."""
    from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
    from langchain_core.messages import AIMessage

    model = _with_ls_provider(GenericFakeChatModel, ls_provider)(
        messages=iter([AIMessage(content="hello")]), callbacks=[_langchain_handler("own-sdk-agent")])
    seed(cap(scoped("PROVIDER", "IS", "ANTHROPIC")))
    checks = record_cap_checks(monkeypatch)

    assert model.invoke("hi").content == "hello"
    assert [(check.provider, check.usage_metadata.get("agent")) for check in checks] == [(judged_as, "own-sdk-agent")]


MISTRAL_CAPS = [
    pytest.param(cap(), True, id="team_cap"),
    pytest.param(cap(scoped("PROVIDER", "IS", "MISTRAL")), True, id="mistral_cap"),
    pytest.param(cap(scoped("PROVIDER", "IS", "OPENAI")), False, id="openai_cap"),
]
LANGCHAIN_CALLS = [
    pytest.param(lambda model: model.invoke("hi").text, id="invoke"),
    pytest.param(lambda model: asyncio.run(model.ainvoke("hi")).text, id="ainvoke"),
    pytest.param(lambda model: "".join(chunk.text for chunk in model.stream("hi")), id="stream"),
]


@pytest.mark.parametrize("rule, refused", MISTRAL_CAPS)
@pytest.mark.parametrize("call", LANGCHAIN_CALLS)
def test_chat_mistralai_is_judged_as_mistral_by_caps(call, rule, refused, seed, provider_requests):
    """BACK-4258: a real ChatMistralAI call is refused by a team cap and a Mistral cap before it
    reaches Mistral, and an OpenAI cap does not refuse it."""
    from entry_points.mistral import ANSWER, chat_mistralai

    model = chat_mistralai([_langchain_handler("mistral-agent")])
    seed(rule)

    if refused:
        with pytest.raises(BudgetExceededError):
            call(model)
        assert provider_requests["httpx"] == 0
    else:
        assert call(model) == ANSWER
        assert provider_requests["httpx"] == 1


@pytest.mark.parametrize("call", LANGCHAIN_CALLS)
def test_a_selectively_unmetered_chat_mistralai_call_is_still_judged_as_mistral_by_caps(
        call, selective_metering, seed, provider_requests, mock_revenium_client):
    """BACK-4320: a ChatMistralAI call outside a @revenium_meter function is refused by a Mistral cap
    before it reaches Mistral, and an allowed one is answered without a record."""
    from entry_points.mistral import ANSWER, chat_mistralai

    model = chat_mistralai([_langchain_handler("mistral-agent")])
    seed(cap(scoped("PROVIDER", "IS", "MISTRAL")))
    with pytest.raises(BudgetExceededError):
        call(model)
    assert provider_requests["httpx"] == 0

    seed(cap(scoped("PROVIDER", "IS", "OPENAI")))
    assert call(model) == ANSWER
    assert provider_requests["httpx"] == 1
    calls.wait_for_metering()
    assert calls.recorded_payloads(mock_revenium_client) == []


def test_a_selectively_unmetered_chat_openai_call_is_refused_by_a_cap_on_the_callback_attribution(
        selective_metering, seed, provider_requests):
    """BACK-4320: the callback still publishes its attribution for a call outside a @revenium_meter
    function, so the OpenAI transport refuses it under a cap on that attribution."""
    from langchain_openai import ChatOpenAI

    seed(cap(scoped("AGENT", "IS", "lc-agent")))
    with pytest.raises(BudgetExceededError):
        _chat_openai(ChatOpenAI, "lc-agent").invoke("hi")
    assert sum(provider_requests.values()) == 0


@pytest.mark.parametrize("ls_provider", ["deepseek", "xai", "together", "cerebras", "upstage", "openai-mantle"])
def test_a_vendor_on_the_openai_client_is_checked_once(ls_provider, seed, monkeypatch):
    from langchain_openai import ChatOpenAI

    seed(cap(scoped("PROVIDER", "IS", "ANTHROPIC")))
    checks = record_cap_checks(monkeypatch)

    _chat_openai(_with_ls_provider(ChatOpenAI, ls_provider), "vendor-agent").invoke("hi")

    assert [(check.provider, check.usage_metadata.get("agent")) for check in checks] == [("OPENAI", "vendor-agent")]


def test_an_unknown_vendor_on_the_openai_client_is_checked_twice_with_one_verdict(seed, monkeypatch):
    """BACK-4220: the callback cannot tell an unknown vendor's client, so it checks,
    and the OpenAI wrap checks again under the same label."""
    from langchain_openai import ChatOpenAI

    seed(cap(scoped("PROVIDER", "IS", "ANTHROPIC")))
    checks = record_cap_checks(monkeypatch)

    _chat_openai(_with_ls_provider(ChatOpenAI, "a-vendor-revenium-does-not-know"), "vendor-agent").invoke("hi")

    assert [(check.provider, check.usage_metadata.get("agent")) for check in checks] == [
        ("OPENAI", "vendor-agent"), ("OPENAI", "vendor-agent")]


def test_chat_anthropic_with_the_anthropic_wrap_is_checked_once(seed, monkeypatch):
    from langchain_anthropic import ChatAnthropic

    model = ChatAnthropic(model=calls.ANTHROPIC_MODEL, api_key="stub", max_tokens=5, max_retries=0,
                          callbacks=[_langchain_handler("claude-agent")])
    model.__dict__["_client"] = calls._anthropic()
    seed(cap(scoped("AGENT", "IS", "nobody")))
    checks = record_cap_checks(monkeypatch)

    model.invoke("hi")

    assert [(check.provider, check.usage_metadata.get("agent")) for check in checks] == [("ANTHROPIC", "claude-agent")]


TEXT_MODEL = "gpt-3.5-turbo-instruct"
TEXT_COMPLETION_USAGE = {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}


def _text_completion(text, finish_reason, usage=None):
    body = {"id": "cmpl-stub", "object": "text_completion", "created": 1, "model": TEXT_MODEL,
            "choices": [{"index": 0, "text": text, "logprobs": None, "finish_reason": finish_reason}]}
    if usage:
        body["usage"] = usage
    return body


def _text_completion_handler(request):
    import httpx
    if not request.url.path.endswith("/completions") or request.url.path.endswith("/chat/completions"):
        return httpx.Response(404, json={"error": request.url.path})
    if not json.loads(request.content or b"{}").get("stream"):
        return httpx.Response(200, json=_text_completion("hi", "stop", TEXT_COMPLETION_USAGE))
    chunks = [_text_completion("hi", None), _text_completion("", "stop")]
    body = "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks) + "data: [DONE]\n\n"
    return httpx.Response(200, content=body.encode(), headers={"content-type": "text/event-stream"})


def _text_completion_clients():
    import httpx
    return dict(http_client=httpx.Client(transport=httpx.MockTransport(_text_completion_handler)),
                http_async_client=httpx.AsyncClient(transport=httpx.MockTransport(_text_completion_handler)))


def _openai_text_model(agent):
    from langchain_openai import OpenAI
    return OpenAI(model=TEXT_MODEL, api_key="stub", max_retries=0, callbacks=[_langchain_handler(agent)],
                  **_text_completion_clients())


def _azure_text_model(agent):
    from langchain_openai import AzureOpenAI
    return AzureOpenAI(model=TEXT_MODEL, azure_deployment="instruct-dep", api_version="2024-10-21",
                       azure_endpoint="https://azure.stub.invalid", api_key="stub", max_retries=0,
                       callbacks=[_langchain_handler(agent)], **_text_completion_clients())


async def _joined_astream(model):
    return "".join([chunk async for chunk in model.astream("hi")])


TEXT_MODEL_CLASSES = [
    pytest.param(_openai_text_model, id="openai"),
    pytest.param(_azure_text_model, id="azure"),
]
TEXT_MODEL_CALLS = [
    pytest.param(lambda model: model.invoke("hi"), id="invoke"),
    pytest.param(lambda model: asyncio.run(model.ainvoke("hi")), id="ainvoke"),
    pytest.param(lambda model: "".join(model.stream("hi")), id="stream"),
    pytest.param(lambda model: asyncio.run(_joined_astream(model)), id="astream"),
]


@pytest.mark.parametrize("call", TEXT_MODEL_CALLS)
@pytest.mark.parametrize("build", TEXT_MODEL_CLASSES)
def test_langchain_text_completion_class_is_refused_before_the_provider(build, call, seed, provider_requests):
    """BACK-4348: langchain_openai.OpenAI and AzureOpenAI call the legacy completions
    endpoint, which no Revenium wrap checks, so the callback must refuse them."""
    model = build("text-agent")
    seed(cap())

    with pytest.raises(BudgetExceededError):
        call(model)
    assert sum(provider_requests.values()) == 0


@pytest.mark.parametrize("build, label", [
    pytest.param(_openai_text_model, "OPENAI", id="openai"),
    pytest.param(_azure_text_model, "Azure", id="azure"),
])
def test_langchain_text_completion_class_is_checked_once_under_its_own_label(build, label, seed, monkeypatch):
    seed(cap(scoped("AGENT", "IS", "nobody")))
    checks = record_cap_checks(monkeypatch)

    assert build("text-agent").invoke("hi") == "hi"

    assert [(check.provider, check.usage_metadata.get("agent")) for check in checks] == [(label, "text-agent")]


def test_a_custom_text_model_without_ls_model_type_is_refused(seed):
    """A text model that reports only its provider is still judged as a text model:
    LangChain starts it with on_llm_start, never on_chat_model_start."""
    from langchain_core.language_models.llms import LLM

    answered = []

    class OpenAILabelledTextModel(LLM):
        @property
        def _llm_type(self):
            return "openai-labelled-text-model"

        def _get_ls_params(self, stop=None, **kwargs):
            return {"ls_provider": "openai"}

        def _call(self, prompt, stop=None, run_manager=None, **kwargs):
            answered.append(prompt)
            return "hello"

    model = OpenAILabelledTextModel(callbacks=[_langchain_handler("custom-text-agent")])
    seed(cap())

    with pytest.raises(BudgetExceededError):
        model.invoke("hi")
    assert answered == []


def test_chat_openai_under_a_breached_cap_is_refused_after_one_check(seed, monkeypatch, provider_requests):
    from langchain_openai import ChatOpenAI

    seed(cap())
    checks = record_cap_checks(monkeypatch)

    with pytest.raises(BudgetExceededError):
        _chat_openai(ChatOpenAI, "openai-agent").invoke("hi")
    assert len(checks) == 1
    assert sum(provider_requests.values()) == 0


def test_anthropic_text_model_is_checked_once_by_its_wrap(seed, monkeypatch):
    """AnthropicLLM calls client.messages.create, which the Anthropic wrap checks,
    so the callback leaves the check to it."""
    from langchain_anthropic.llms import AnthropicLLM

    model = AnthropicLLM(model=calls.ANTHROPIC_MODEL, api_key="stub", max_tokens=5, max_retries=0,
                         callbacks=[_langchain_handler("claude-text-agent")])
    model.client = calls._anthropic()
    seed(cap(scoped("AGENT", "IS", "nobody")))
    checks = record_cap_checks(monkeypatch)

    model.invoke("hi")

    assert [(check.provider, check.usage_metadata.get("agent")) for check in checks] == [
        ("ANTHROPIC", "claude-text-agent")]


OPENAI_ONLY_PERPLEXITY_PROBE = r"""
import json, os, sys
sys.path.insert(0, sys.argv[1])
import httpx
import revenium_middleware.openai  # the only Revenium middleware this process imports
from langchain_openai import ChatOpenAI
from entry_points import calls
from entry_points.caps import breached_rule, record_cap_checks, seed_rules
from revenium_middleware._core import patch_registry
from revenium_middleware._core.exceptions import BudgetExceededError
from revenium_middleware.openai.langchain import ReveniumCallbackHandler


class Patcher:
    setattr = staticmethod(setattr)

    def setenv(self, name, value):
        os.environ[name] = value

    def delenv(self, name, raising=True):
        os.environ.pop(name, None)


seed_rules(Patcher(), breached_rule(4252))
checks = record_cap_checks(Patcher())
answer = calls._openai_compatible_handler(calls.PERPLEXITY_MODEL)
provider_requests = []


def handler(request):
    provider_requests.append(request.url.path)
    return answer(request)


model = ChatOpenAI(model=calls.PERPLEXITY_MODEL, base_url="https://api.perplexity.ai", api_key="stub", max_retries=0,
                   callbacks=[ReveniumCallbackHandler(usage_metadata={"agent": "perplexity-agent"})],
                   http_client=httpx.Client(transport=httpx.MockTransport(handler)))
try:
    model.invoke("hi")
    refused = False
except BudgetExceededError:
    refused = True
print(json.dumps({"refused": refused, "provider_requests": len(provider_requests),
                  "perplexity_wrapped": any(key.startswith("perplexity:") for key in patch_registry._patched),
                  "checks": [[check.provider, check.usage_metadata.get("agent")] for check in checks]}))
"""


def test_langchain_chat_openai_pointed_at_perplexity_is_refused_without_the_perplexity_middleware():
    """BACK-4252: run in a fresh interpreter that imports only revenium_middleware.openai,
    as a customer who never imported the Perplexity middleware would."""
    import json
    import os
    import pathlib
    import subprocess
    import sys

    tests_dir = str(pathlib.Path(__file__).resolve().parents[1])
    env = {**os.environ, "REVENIUM_METERING_API_KEY": "hak_stub_provider_paths",
           "REVENIUM_METERING_BASE_URL": "http://127.0.0.1:9"}
    completed = subprocess.run([sys.executable, "-c", OPENAI_ONLY_PERPLEXITY_PROBE, tests_dir],
                               capture_output=True, text=True, env=env, timeout=120)

    assert completed.returncode == 0, completed.stderr[-4000:]
    outcome = json.loads(completed.stdout.strip().splitlines()[-1])
    assert outcome == {"refused": True, "provider_requests": 0, "perplexity_wrapped": False,
                       "checks": [["PERPLEXITY", "perplexity-agent"]]}


PERPLEXITY_WRAP_KEYS_PROBE = r"""
import json
import revenium_middleware.openai
import revenium_middleware.perplexity
from revenium_middleware._core import patch_registry
from revenium_middleware.openai import middleware as mw

print(json.dumps([patch_registry.is_patched(mw._PERPLEXITY_CHAT_WRAP),
                  patch_registry.is_patched(mw._PERPLEXITY_ASYNC_CHAT_WRAP)]))
"""


def test_the_openai_middleware_names_the_keys_the_perplexity_middleware_registers():
    import json
    import os
    import subprocess
    import sys

    env = {**os.environ, "REVENIUM_METERING_API_KEY": "hak_stub_provider_paths",
           "REVENIUM_METERING_BASE_URL": "http://127.0.0.1:9"}
    completed = subprocess.run([sys.executable, "-c", PERPLEXITY_WRAP_KEYS_PROBE],
                               capture_output=True, text=True, env=env, timeout=120)

    assert completed.returncode == 0, completed.stderr[-4000:]
    assert json.loads(completed.stdout.strip().splitlines()[-1]) == [True, True]


PERPLEXITY_CHAT_WRAPS = ("perplexity:openai.resources.chat.completions.Completions.create",
                         "perplexity:openai.resources.chat.completions.AsyncCompletions.create")


def test_an_allowed_perplexity_bound_call_with_both_middlewares_is_recorded_once(
        seed, monkeypatch, provider_requests, mock_revenium_client):
    from revenium_middleware._core import patch_registry

    monkeypatch.setattr(patch_registry, "_patched", patch_registry._patched | set(PERPLEXITY_CHAT_WRAPS))
    seed(cap(scoped("AGENT", "IS", "nobody")))
    checks = record_cap_checks(monkeypatch)

    calls.run_call("perplexity.openai_compatible.chat.create.sync")
    calls.wait_for_metering()

    assert sum(provider_requests.values()) == 1
    assert len(calls.recorded_payloads(mock_revenium_client)) == 1
    assert len(checks) <= 1


def _batched_built_in_tool_generate(model):
    from langchain_core.messages import HumanMessage
    from entry_points.perplexity_cost import BUILT_IN_TOOLS

    return model.generate([[HumanMessage(content="first")], [HumanMessage(content="second")]], tools=BUILT_IN_TOOLS)


def _built_in_tool(call):
    from entry_points.perplexity_cost import BUILT_IN_TOOLS

    return lambda model: call(model, BUILT_IN_TOOLS)


async def _drain(chunks):
    return [chunk async for chunk in chunks]


CHAT_PERPLEXITY_CALLS = [
    pytest.param(_built_in_tool(lambda model, tools: model.invoke("hi", tools=tools)), id="invoke"),
    pytest.param(_built_in_tool(lambda model, tools: asyncio.run(model.ainvoke("hi", tools=tools))), id="ainvoke"),
    pytest.param(_built_in_tool(lambda model, tools: list(model.stream("hi", tools=tools))), id="stream"),
    pytest.param(_built_in_tool(lambda model, tools: asyncio.run(_drain(model.astream("hi", tools=tools)))),
                 id="astream"),
    pytest.param(_batched_built_in_tool_generate, id="batched-generate"),
    pytest.param(lambda model: model.invoke("hi"), id="plain-chat"),
]


@pytest.mark.parametrize("call", CHAT_PERPLEXITY_CALLS)
def test_a_chat_perplexity_built_in_tool_call_is_refused_with_both_integrations_imported(
        call, seed, monkeypatch, provider_requests):
    """BACK-4283: langchain-perplexity sends a built-in-tool call to Perplexity's Responses API,
    and the callback leaves the check to the Perplexity integration once it is imported."""
    from entry_points.perplexity_cost import chat_perplexity

    model = chat_perplexity([_langchain_handler("pplx-agent")])
    seed(cap(scoped("AGENT", "IS", "pplx-agent")))
    checks = record_cap_checks(monkeypatch)

    with pytest.raises(BudgetExceededError):
        call(model)

    assert provider_requests["httpx"] == 0
    assert [(check.provider, check.usage_metadata.get("agent")) for check in checks] == [("PERPLEXITY", "pplx-agent")]


def _responses_call(base_url, client_index=0):
    from entry_points.perplexity_cost import PERPLEXITY_MODEL, openai_clients_at

    client = openai_clients_at(base_url)[client_index]
    return lambda: client.responses.create(model=PERPLEXITY_MODEL, input="hi")


def _sync_responses_call(base_url):
    return _responses_call(base_url)


def _async_responses_call(base_url):
    create = _responses_call(base_url, client_index=1)
    return lambda: asyncio.run(create())


def _chat_openai_responses_call(_base_url):
    from entry_points.perplexity_cost import chat_openai_at_perplexity

    model = chat_openai_at_perplexity([_langchain_handler("pplx-agent")], use_responses_api=True)
    return lambda: model.invoke("hi")


RESPONSES_API_CALLS = [
    pytest.param(_sync_responses_call, False, id="create"),
    pytest.param(_async_responses_call, False, id="async-create"),
    pytest.param(_chat_openai_responses_call, False, id="chat-openai-invoke"),
    pytest.param(_sync_responses_call, True, id="selectively-unmetered-create"),
]


@pytest.mark.parametrize("make_call, selectively_metered", RESPONSES_API_CALLS)
def test_a_responses_api_call_to_perplexity_is_refused_by_a_perplexity_cap(
        make_call, selectively_metered, seed, monkeypatch, provider_requests):
    """BACK-4293: the Responses API reaches Perplexity through the OpenAI client too."""
    from entry_points.perplexity_cost import PERPLEXITY_BASE_URL

    if selectively_metered:
        monkeypatch.setenv("REVENIUM_SELECTIVE_METERING", "true")
    call = make_call(PERPLEXITY_BASE_URL)
    seed(cap(scoped("PROVIDER", "IS", "PERPLEXITY")))
    checks = record_cap_checks(monkeypatch)

    with pytest.raises(BudgetExceededError):
        call()

    assert provider_requests["httpx"] == 0
    assert [check.provider for check in checks] == ["PERPLEXITY"]


def test_a_responses_api_call_to_openai_is_not_refused_by_a_perplexity_cap(
        seed, monkeypatch, provider_requests, mock_revenium_client, released_stub_response_id):
    call = _sync_responses_call("https://api.openai.com/v1")
    seed(cap(scoped("PROVIDER", "IS", "PERPLEXITY")))
    checks = record_cap_checks(monkeypatch)

    call()

    assert provider_requests["httpx"] == 1
    assert [check.provider for check in checks] == ["OPENAI"]


CALLBACK_ONLY_BUILT_IN_TOOL_PROBE = r"""
import asyncio, json, os, sys
sys.path.insert(0, sys.argv[1])
import httpx
from revenium_middleware.openai.langchain import ReveniumCallbackHandler  # the only Revenium import
from entry_points.caps import breached_rule, record_cap_checks, scoped, seed_rules
from entry_points.perplexity_cost import BUILT_IN_TOOLS, chat_perplexity
from entry_points.stub_requests import count_provider_requests
from revenium_middleware._core import patch_registry
from revenium_middleware._core.exceptions import BudgetExceededError


class Patcher:
    setattr = staticmethod(setattr)

    def setenv(self, name, value):
        os.environ[name] = value

    def delenv(self, name, raising=True):
        os.environ.pop(name, None)


async def drain(chunks):
    return [chunk async for chunk in chunks]


CALLS = {
    "invoke": lambda model: model.invoke("hi", tools=BUILT_IN_TOOLS),
    "ainvoke": lambda model: asyncio.run(model.ainvoke("hi", tools=BUILT_IN_TOOLS)),
    "stream": lambda model: list(model.stream("hi", tools=BUILT_IN_TOOLS)),
    "astream": lambda model: asyncio.run(drain(model.astream("hi", tools=BUILT_IN_TOOLS))),
}
outcomes = {}
for name, call in CALLS.items():
    patcher = Patcher()
    seed_rules(patcher, breached_rule(4283, scoped("AGENT", "IS", "pplx-agent")))
    checks = record_cap_checks(patcher)
    provider_requests = count_provider_requests(patcher)
    model = chat_perplexity([ReveniumCallbackHandler(usage_metadata={"agent": "pplx-agent"})])
    try:
        call(model)
        refused = False
    except BudgetExceededError:
        refused = True
    outcomes[name] = {"refused": refused, "provider_requests": provider_requests["httpx"],
                      "checks": [[check.provider, check.usage_metadata.get("agent")] for check in checks]}
outcomes["perplexity_wrapped"] = any(key.startswith("perplexity") for key in patch_registry._patched)
print(json.dumps(outcomes))
"""


def test_a_chat_perplexity_built_in_tool_call_is_refused_once_with_only_the_callback():
    """BACK-4283 guard: a fresh interpreter that never imports revenium_middleware.perplexity,
    so the callback makes the one check itself."""
    tests_dir = str(pathlib.Path(__file__).resolve().parents[1])
    env = {**os.environ, "REVENIUM_METERING_API_KEY": "hak_stub_provider_paths",
           "REVENIUM_METERING_BASE_URL": "http://127.0.0.1:9"}
    completed = subprocess.run([sys.executable, "-c", CALLBACK_ONLY_BUILT_IN_TOOL_PROBE, tests_dir],
                               capture_output=True, text=True, env=env, timeout=120)

    assert completed.returncode == 0, completed.stderr[-4000:]
    outcomes = json.loads(completed.stdout.strip().splitlines()[-1])
    assert outcomes.pop("perplexity_wrapped") is False
    refused_once = {"refused": True, "provider_requests": 0, "checks": [["PERPLEXITY", "pplx-agent"]]}
    assert outcomes == {name: refused_once for name in ("invoke", "ainvoke", "stream", "astream")}


ANTHROPIC_WRAP_KEYS = ("anthropic.resources.messages.messages.Messages.create",
                       "anthropic.resources.messages.messages.AsyncMessages.create")


def test_langchain_callback_checks_claude_when_the_anthropic_wrap_is_not_installed(seed, monkeypatch):
    """As in a process that never imported revenium_middleware.anthropic: the
    Anthropic client is unwrapped and the patch registry does not list it."""
    import anthropic
    from langchain_anthropic import ChatAnthropic
    from revenium_middleware._core import patch_registry

    monkeypatch.setattr(patch_registry, "_patched", patch_registry._patched - set(ANTHROPIC_WRAP_KEYS))
    anthropic_create = MagicMock()
    monkeypatch.setattr(anthropic.resources.messages.messages.Messages, "create", anthropic_create)
    model = ChatAnthropic(model=calls.ANTHROPIC_MODEL, api_key="stub", max_tokens=5, max_retries=0,
                          callbacks=[_langchain_handler("claude-agent")])
    seed(cap(scoped("PROVIDER", "IS", "ANTHROPIC"), scoped("MODEL", "IS", calls.ANTHROPIC_MODEL)))

    with pytest.raises(BudgetExceededError):
        model.invoke("hi")
    anthropic_create.assert_not_called()


@pytest.mark.parametrize("invoke", [
    pytest.param(lambda model: model.invoke("hi"), id="invoke"),
    pytest.param(lambda model: asyncio.run(model.ainvoke("hi")), id="ainvoke"),
])
def test_langchain_callback_checks_claude_on_vertex_when_the_anthropic_wrap_is_not_installed(
        invoke, seed, monkeypatch):
    """BACK-4222: ChatAnthropicVertex calls AnthropicVertex, which only the Anthropic wrap checks."""
    from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
    from langchain_core.messages import AIMessage
    from revenium_middleware._core import patch_registry

    monkeypatch.setattr(patch_registry, "_patched", patch_registry._patched - set(ANTHROPIC_WRAP_KEYS))
    answers = iter([AIMessage(content="hello")])
    model = _with_ls_provider(GenericFakeChatModel, "anthropicvertex")(
        messages=answers, callbacks=[_langchain_handler("vertex-claude-agent")])
    seed(cap(scoped("PROVIDER", "IS", "ANTHROPIC"), scoped("AGENT", "IS", "vertex-claude-agent")))
    checks = record_cap_checks(monkeypatch)

    with pytest.raises(BudgetExceededError):
        invoke(model)
    assert next(answers).content == "hello", "the model was called"
    assert [check.provider for check in checks] == ["ANTHROPIC"]

    answers = iter([AIMessage(content="hello")])
    model.messages = answers
    seed(cap(scoped("PROVIDER", "IS_NOT", "ANTHROPIC")))
    assert invoke(model).content == "hello"


def _claude_stand_in(ls_provider, make_client, agent, transport_handler=calls._anthropic_handler):
    """A LangChain Claude class over an Anthropic SDK client, as ChatAnthropicVertex
    and ChatAnthropicMantle reach their vendors: ``client.messages.create``."""
    import httpx2
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.messages import AIMessage
    from langchain_core.outputs import ChatGeneration, ChatResult

    class ClaudeStandIn(BaseChatModel):
        @property
        def _llm_type(self):
            return ls_provider + "-stand-in"

        def _get_ls_params(self, stop=None, **kwargs):
            return {"ls_provider": ls_provider, "ls_model_name": calls.ANTHROPIC_MODEL, "ls_model_type": "chat"}

        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            with make_client(httpx2.Client(transport=httpx2.MockTransport(transport_handler))) as client:
                message = client.messages.create(model=calls.ANTHROPIC_MODEL, max_tokens=5,
                                                 messages=calls.USER_MESSAGES)
            return ChatResult(generations=[ChatGeneration(message=AIMessage(
                content=message.content[0].text, response_metadata={"id": message.id, "model_name": message.model},
                usage_metadata={"input_tokens": message.usage.input_tokens,
                                "output_tokens": message.usage.output_tokens,
                                "total_tokens": message.usage.input_tokens + message.usage.output_tokens}))])

    return ClaudeStandIn(callbacks=[_langchain_handler(agent)])


def _claude_on_vertex_stand_in(agent):
    import anthropic

    return _claude_stand_in("anthropicvertex", lambda http_client: anthropic.AnthropicVertex(
        region="us-east5", project_id="p", access_token="stub", max_retries=0, http_client=http_client), agent)


def test_claude_on_vertex_through_the_anthropic_wrap_is_checked_once(seed, monkeypatch):
    """BACK-4222: with revenium_middleware.anthropic imported, the wrap checks and
    meters the call, and the callback does neither."""
    seed(cap(scoped("AGENT", "IS", "nobody")))
    checks = record_cap_checks(monkeypatch)
    recording_client = MagicMock()

    with patch("revenium_middleware._core.metering.client", recording_client):
        _claude_on_vertex_stand_in("vertex-claude-agent").invoke("hi")
        calls.wait_for_metering()

    assert [(check.provider, check.usage_metadata.get("agent")) for check in checks] == [
        ("ANTHROPIC", "vertex-claude-agent")]
    sent = [entry.kwargs for entry in recording_client.ai.create_completion.call_args_list]
    assert [record["provider"] for record in sent] == ["ANTHROPIC"], sent


def test_chat_anthropic_mantle_is_refused_by_the_anthropic_wrap_before_the_request(seed, monkeypatch):
    """ChatAnthropicMantle calls a real ``anthropic.AnthropicBedrockMantle``; the
    Anthropic wrap refuses it and the callback leaves the check to the wrap."""
    import anthropic

    requests = []

    def record_request(request):
        requests.append(request)
        return calls._anthropic_handler(request)

    model = _claude_stand_in("anthropic-mantle", lambda http_client: anthropic.AnthropicBedrockMantle(
        aws_region="us-east-1", api_key="stub", max_retries=0, http_client=http_client),
        "mantle-agent", transport_handler=record_request)
    seed(cap(scoped("AGENT", "IS", "mantle-agent")))
    checks = record_cap_checks(monkeypatch)

    with pytest.raises(BudgetExceededError):
        model.invoke("hi")

    assert requests == [], "the request reached Bedrock Mantle"
    assert [(check.provider, check.usage_metadata.get("agent")) for check in checks] == [
        ("ANTHROPIC", "mantle-agent")]


ASYNC_SURFACES = [
    pytest.param("anthropic.messages.create.async", id="anthropic_create"),
    pytest.param("anthropic.messages.stream_helper.async", id="anthropic_stream"),
    pytest.param("genai.aio.models.generate_content.async", id="genai_aio"),
    pytest.param("ollama.async_client.chat.async", id="ollama_async"),
    pytest.param("perplexity.native.chat.create.async", id="perplexity_native_async"),
    pytest.param("perplexity.openai_compatible.chat.create.async", id="perplexity_openai_async"),
    pytest.param("fal.run_async.async", id="fal_run_async"),
]
POLL_THREAD = "revenium-enforcement-poll"


@pytest.mark.parametrize("call_id", ASYNC_SURFACES)
def test_async_provider_checks_never_fetch_rules_on_the_callers_loop(call_id, monkeypatch):
    """A stale cache wakes the poller thread; the check on the caller's loop never fetches itself."""
    fetch_threads = []

    def record_fetch():
        fetch_threads.append(threading.current_thread().name)

    monkeypatch.setenv("REVENIUM_CIRCUIT_BREAKER_ENABLED", "true")
    monkeypatch.delenv("REVENIUM_BYPASS", raising=False)
    monkeypatch.delenv("REVENIUM_CB_FAIL_MODE", raising=False)
    monkeypatch.setattr(enforcement, "_load_cache_from_disk", lambda: None)
    monkeypatch.setattr(enforcement, "_fetch_rules", record_fetch)
    monkeypatch.setattr(enforcement, "_cached_rules", [])
    monkeypatch.setattr(enforcement, "_cache_initialized", True)
    monkeypatch.setattr(enforcement, "_cache_timestamp", 0.0)
    monkeypatch.setattr(enforcement, "_refresh_cooldown_until", 0.0)
    enforcement._poll_wakeup.clear()

    calls.run_call(call_id)

    assert wait_until(lambda: fetch_threads), "the check never woke the poller"
    assert set(fetch_threads) == {POLL_THREAD}


MAX_LOOP_LAG_SECONDS = 0.25


def _fal_run_async():
    from fal_client import AsyncClient
    client = AsyncClient(key="stub:stub")
    client.__dict__["_client"] = calls._fal_async_http()
    return lambda: client.run(calls.FAL_APP, arguments=calls._FAL_ARGUMENTS)


def _openai_compatible_perplexity():
    client = calls._async_openai(base_url=calls.PERPLEXITY_BASE_URL, model=calls.PERPLEXITY_MODEL,
                                 usage=calls._PERPLEXITY_USAGE)
    return lambda: client.chat.completions.create(**calls._chat_kwargs(calls.PERPLEXITY_MODEL))


def _anthropic_create():
    client = calls._async_anthropic()
    return lambda: client.messages.create(**calls._ANTHROPIC_KWARGS)


def _anthropic_stream():
    client = calls._async_anthropic()
    return lambda: calls._drain_async(client.messages.stream(**calls._ANTHROPIC_KWARGS))


def _genai_aio():
    client = calls._genai()
    return lambda: client.aio.models.generate_content(model=calls.GEMINI_MODEL, contents="hi")


def _ollama_async():
    client = calls._ollama_async_client()
    return lambda: client.chat(model=calls.OLLAMA_MODEL, messages=calls.USER_MESSAGES)


def _perplexity_native_async():
    client = calls._async_perplexity_native()
    return lambda: client.chat.completions.create(model=calls.PERPLEXITY_MODEL, messages=calls.USER_MESSAGES)


# Each surface's client is built once, so the loop lag measures the call and
# not the provider SDK's client construction, which blocks on its own.
ASYNC_CLIENT_CALLS = [
    pytest.param(_anthropic_create, id="anthropic_create"),
    pytest.param(_anthropic_stream, id="anthropic_stream"),
    pytest.param(_genai_aio, id="genai_aio"),
    pytest.param(_ollama_async, id="ollama_async"),
    pytest.param(_perplexity_native_async, id="perplexity_native_async"),
    pytest.param(_openai_compatible_perplexity, id="perplexity_openai_async"),
    pytest.param(_fal_run_async, id="fal_run_async"),
]


def _fal_stream_async():
    from fal_client import AsyncClient
    client = AsyncClient(key="stub:stub")
    client.__dict__["_client"] = calls._fal_async_http()

    async def drain(events):
        async for _event in events:
            pass

    return lambda: drain(client.stream(calls.FAL_APP, arguments=calls._FAL_ARGUMENTS))


CALLS_CREATED_BEFORE_THEY_ARE_AWAITED = [
    pytest.param(_anthropic_create, id="anthropic_create"),
    pytest.param(_ollama_async, id="ollama_async"),
    pytest.param(_perplexity_native_async, id="perplexity_native_async"),
    pytest.param(_openai_compatible_perplexity, id="perplexity_openai_async"),
    pytest.param(_genai_aio, id="genai_aio"),
    pytest.param(_fal_run_async, id="fal_run_async"),
    pytest.param(_fal_stream_async, id="fal_stream_async"),
]


@pytest.mark.parametrize("make_call", CALLS_CREATED_BEFORE_THEY_ARE_AWAITED)
@pytest.mark.parametrize("selective", [False, True], ids=["metered", "selectively_unmetered"])
def test_an_async_call_created_before_a_cap_breaches_is_refused_when_awaited(
        make_call, selective, seed, provider_requests, monkeypatch):
    if selective:
        monkeypatch.setenv("REVENIUM_SELECTIVE_METERING", "true")
    seed(cap(scoped("PROVIDER", "IS", "NOBODY")))

    async def run():
        pending = make_call()()
        seed(cap())
        await pending

    with pytest.raises(BudgetExceededError):
        asyncio.run(run())
    assert sum(provider_requests.values()) == 0


@pytest.mark.parametrize("make_call", ASYNC_CLIENT_CALLS)
@pytest.mark.parametrize("mode", FAILURE_MODES)
def test_async_provider_calls_never_hold_the_loop_while_revenium_is_down(make_call, mode, monkeypatch):
    stub = EnforcementStub(mode)
    point_enforcement_at(monkeypatch, stub)
    errors = []

    async def run():
        provider_call = make_call()

        async def call():
            try:
                await provider_call()
            except Exception as error:  # collected so the assertion below names it
                errors.append(error)

        return await drive_on_loop(call, requests=15, rate=30)

    try:
        report = asyncio.run(run())
    finally:
        shut_down(stub)

    assert errors == []
    assert report.max_lag < MAX_LOOP_LAG_SECONDS, (
        f"the call held the loop {report.max_lag:.3f}s with the rules endpoint {mode}")


def _foundry():
    import httpx2
    return anthropic.AnthropicFoundry(api_key="stub", resource="example-resource",
                                      http_client=httpx2.Client(transport=httpx2.MockTransport(calls._anthropic_handler)))


requires_foundry_client = pytest.mark.skipif(
    not hasattr(anthropic, "AnthropicFoundry"), reason="Installed anthropic SDK predates the Foundry client classes")

ANTHROPIC_PROVIDER_CAPS = [
    pytest.param(scoped("PROVIDER", "IS", "ANTHROPIC"), True, id="is_anthropic"),
    pytest.param({"dimension": "PROVIDER", "operator": "IN", "value": "ANTHROPIC", "values": ["ANTHROPIC"]}, True,
                 id="in_anthropic"),
    pytest.param(scoped("PROVIDER", "IS_NOT", "ANTHROPIC"), False, id="is_not_anthropic"),
]
FOUNDRY_PROVIDER_CAPS = ANTHROPIC_PROVIDER_CAPS + [
    pytest.param(scoped("PROVIDER", "IS", "FOUNDRY"), True, id="is_foundry"),
    pytest.param(scoped("PROVIDER", "IS_NOT", "FOUNDRY"), False, id="is_not_foundry"),
]


def _assert_judged(make_client, refused, provider_requests):
    if refused:
        with pytest.raises(BudgetExceededError):
            make_client().messages.create(**calls._ANTHROPIC_KWARGS)
        assert provider_requests["httpx2"] == 0
    else:
        make_client().messages.create(**calls._ANTHROPIC_KWARGS)
        assert provider_requests["httpx2"] == 1


@requires_foundry_client
@pytest.mark.parametrize("rule_filter, refused", FOUNDRY_PROVIDER_CAPS)
def test_a_foundry_client_is_judged_as_anthropic_by_provider_caps(rule_filter, refused, seed, provider_requests):
    seed(cap(rule_filter))
    with patch.object(anthropic_middleware, "_meter_message"):
        _assert_judged(_foundry, refused, provider_requests)


@requires_foundry_client
def test_a_selectively_unmetered_foundry_call_is_judged_as_anthropic_by_provider_caps(
        selective_metering, seed, provider_requests):
    seed(cap(scoped("PROVIDER", "IS", "ANTHROPIC")))
    _assert_judged(_foundry, True, provider_requests)


@pytest.mark.parametrize("rule_filter, refused", ANTHROPIC_PROVIDER_CAPS)
def test_a_direct_anthropic_client_is_judged_as_before_by_anthropic_provider_caps(
        rule_filter, refused, seed, provider_requests):
    seed(cap(rule_filter))
    with patch.object(anthropic_middleware, "_meter_message"):
        _assert_judged(calls._anthropic, refused, provider_requests)


GEMINI_FLASH_LATEST = "gemini-flash-latest"


def _genai_generate_content(client):
    return client.models.generate_content(model=GEMINI_FLASH_LATEST, contents="hi")


def _genai_generate_content_stream(client):
    return client.models.generate_content_stream(model=GEMINI_FLASH_LATEST, contents="hi")


def _genai_aio_generate_content(client):
    return asyncio.run(client.aio.models.generate_content(model=GEMINI_FLASH_LATEST, contents="hi"))


@pytest.mark.parametrize("call", [
    pytest.param(_genai_generate_content, id="sync"),
    pytest.param(_genai_generate_content_stream, id="stream"),
    pytest.param(_genai_aio_generate_content, id="async"),
])
def test_a_gemini_latest_alias_is_refused_before_google_by_a_cap_on_its_served_model(call, seed, provider_requests):
    """Google records ``gemini-flash-latest`` as the Flash model it serves, so
    a breached cap on that model stops the alias before Google (BACK-4351)."""
    client = calls._genai()
    seed(cap(scoped("MODEL", "IS", "gemini-3.8-flash")))

    with pytest.raises(BudgetExceededError):
        call(client)

    assert provider_requests["gemini"] == 0


XAI_BASE_URL = "https://api.x.ai/v1"


def _xai_chat(client, model):
    return client.chat.completions.create(model=model, messages=calls.USER_MESSAGES)


def _xai_chat_stream(client, model):
    for _chunk in client.chat.completions.create(model=model, messages=calls.USER_MESSAGES, stream=True):
        pass


def _xai_responses(client, model):
    return client.responses.create(model=model, input="hi")


@pytest.mark.parametrize("call, requested, capped", [
    pytest.param(_xai_chat, "grok-4.20", "grok-4.20-0309-reasoning", id="chat"),
    pytest.param(_xai_chat_stream, "grok-4.20", "grok-4.20-0309-reasoning", id="stream"),
    pytest.param(_xai_responses, "grok-4.20", "grok-4.20-0309-reasoning", id="responses"),
    pytest.param(_xai_chat, "grok-code-fast-1", "grok-build-0.1", id="code-fast"),
])
def test_an_xai_short_name_is_refused_before_xai_by_a_cap_on_its_served_model(
        call, requested, capped, seed, provider_requests, released_stub_response_id):
    """xAI records ``grok-4.20`` as ``grok-4.20-0309-reasoning`` and
    ``grok-code-fast-1`` as ``grok-build-0.1``, so a breached cap on the
    recorded model stops the short name before xAI (BACK-4352)."""
    client = calls._openai(base_url=XAI_BASE_URL, model=capped)
    seed(cap(scoped("MODEL", "IS", capped)))

    with pytest.raises(BudgetExceededError):
        call(client, requested)

    assert provider_requests["httpx"] == 0
