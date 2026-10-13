"""BACK-4259, BACK-4280: a LangChain call to Perplexity is recorded once with the charge Perplexity reported.

langchain-perplexity keeps only token counts, so Perplexity's flat per-request
fee reaches the record only through the Perplexity SDK call itself. This test
process has the Perplexity integration imported; the callback-only scenarios
run in a child process that never imports it.
"""
import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import revenium_middleware.perplexity  # noqa: F401  installs the Perplexity integration
from revenium_middleware import revenium_meter
from revenium_middleware._core import call_ownership
from revenium_middleware._core.call_ownership import OPENAI
from revenium_middleware.openai.langchain import _model_call
from revenium_middleware.openai.langchain.unified_handler import UnifiedReveniumCallbackHandler
from entry_points import calls, perplexity_cost
from entry_points.perplexity_cost import BATCH_COSTS, INPUT_COST, OUTPUT_COST, TOTAL_COST

TESTS_DIR = Path(__file__).resolve().parents[1]
REPORTED_COST = {"total_cost": TOTAL_COST, "input_token_cost": INPUT_COST, "output_token_cost": OUTPUT_COST}
CHAT_COMPLETION_PARSED_BY_THE_PERPLEXITY_SDK = pytest.mark.skipif(
    sys.version_info < (3, 11),
    reason="On Python 3.10 the Perplexity SDK leaves the forward reference in CompletionResponseOutput.choices "
           "unresolved, so a non-streamed chat completion keeps dict choices and ChatPerplexity fails reading "
           ".message before any record is made")


def _parsed_chat(scenario):
    return pytest.param(scenario, marks=CHAT_COMPLETION_PARSED_BY_THE_PERPLEXITY_SDK)


def _costs(record):
    return {name: record[name] for name in REPORTED_COST}


def _run_without_the_integration(scenario):
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, [str(TESTS_DIR), os.environ.get("PYTHONPATH")])),
           "REVENIUM_METERING_API_KEY": os.environ.get("REVENIUM_METERING_API_KEY") or "hak_test_dummy_key",
           "REVENIUM_CIRCUIT_BREAKER_ENABLED": "false"}
    completed = subprocess.run([sys.executable, "-m", "entry_points.perplexity_cost", scenario], env=env,
                               cwd=TESTS_DIR, capture_output=True, text=True, timeout=120)
    assert completed.returncode == 0, completed.stderr
    return json.loads(completed.stdout.strip().splitlines()[-1])


@pytest.mark.parametrize("scenario", [
    _parsed_chat("callback-chat"), _parsed_chat("callback-chat-ainvoke"), "callback-chat-stream", "callback-chat-astream",
    "callback-built-in-tool", "callback-built-in-tool-ainvoke", "callback-built-in-tool-stream",
    "callback-built-in-tool-astream",
])
def test_a_callback_only_chat_perplexity_call_carries_the_reported_cost(scenario):
    records = _run_without_the_integration(scenario)
    assert len(records) == 1, records
    assert records[0]["provider"] == "PERPLEXITY"
    assert records[0]["transaction_id"].startswith("langchain-")
    assert records[0]["total_cost"] == TOTAL_COST
    assert (records[0]["input_token_count"], records[0]["output_token_count"]) == (2, 3)


@CHAT_COMPLETION_PARSED_BY_THE_PERPLEXITY_SDK
@pytest.mark.parametrize("scenario", ["callback-batched-generate", "callback-batched-agenerate"])
def test_each_batched_chat_perplexity_run_carries_the_cost_reported_for_its_own_prompt(scenario):
    records = _run_without_the_integration(scenario)
    assert len(records) == len(BATCH_COSTS), records
    assert all(record["transaction_id"].startswith("langchain-") for record in records), records
    assert sorted((record["input_token_count"], record["total_cost"]) for record in records) == sorted(
        (prompt_tokens, total_cost) for total_cost, prompt_tokens in BATCH_COSTS.values())


def _tokens_and_cost(records):
    """Each record's (input tokens, reported total cost), ordered by tokens and then cost, unpriced first."""
    pairs = [(record["input_token_count"], record["total_cost"]) for record in records]
    return sorted(pairs, key=lambda pair: (pair[0], -1 if pair[1] is None else pair[1]))


@CHAT_COMPLETION_PARSED_BY_THE_PERPLEXITY_SDK
@pytest.mark.parametrize("scenario, expected", [
    # Each row holds the warm-up call that filled the cache, carrying its own cost, and the three batched runs.
    pytest.param("callback-cached-batched-generate", [(11, None), (11, 0.011), (22, 0.022), (33, 0.033)],
                 id="cached-batched-generate"),
    pytest.param("callback-cached-batched-agenerate", [(11, None), (11, 0.011), (22, 0.022), (33, 0.033)],
                 id="cached-batched-agenerate"),
    pytest.param("callback-middle-cached-batched-generate", [(11, 0.011), (22, None), (22, 0.022), (33, 0.033)],
                 id="middle-cached-batched-generate"),
    pytest.param("callback-middle-cached-batched-agenerate", [(11, 0.011), (22, None), (22, 0.022), (33, 0.033)],
                 id="middle-cached-batched-agenerate"),
])
def test_a_prompt_answered_from_the_langchain_cache_takes_no_perplexity_cost(scenario, expected):
    """BACK-4282: the cached run never reaches Perplexity, so it must not take the next real call's cost."""
    records = _run_without_the_integration(scenario)
    assert _tokens_and_cost(records) == expected, records


def test_a_built_in_tool_call_with_the_integration_imported_carries_the_reported_cost(mock_revenium_client):
    model = perplexity_cost.chat_perplexity([UnifiedReveniumCallbackHandler()])
    assert model.invoke("hi", tools=perplexity_cost.BUILT_IN_TOOLS).text == perplexity_cost.ANSWER
    calls.wait_for_metering()
    records = perplexity_cost.recorded_costs(mock_revenium_client)
    assert len(records) == 1, records
    assert records[0]["transaction_id"].startswith("langchain-")
    assert _costs(records[0]) == REPORTED_COST


@pytest.mark.parametrize("stream", [pytest.param(False, id="chat", marks=CHAT_COMPLETION_PARSED_BY_THE_PERPLEXITY_SDK),
                                    pytest.param(True, id="chat-stream")])
def test_a_chat_perplexity_chat_call_with_the_integration_imported_is_recorded_once_by_it(
        stream, mock_revenium_client):
    model = perplexity_cost.chat_perplexity([UnifiedReveniumCallbackHandler()])
    if stream:
        list(model.stream("hi"))
    else:
        model.invoke("hi")
    calls.wait_for_metering()
    records = perplexity_cost.recorded_costs(mock_revenium_client)
    assert len(records) == 1, records
    assert not records[0]["transaction_id"].startswith("langchain-")
    assert _costs(records[0]) == REPORTED_COST


def test_a_direct_perplexity_call_keeps_its_reported_cost(mock_revenium_client):
    client, _ = perplexity_cost.perplexity_clients()
    client.chat.completions.create(model=perplexity_cost.PERPLEXITY_MODEL, messages=calls.USER_MESSAGES)
    calls.wait_for_metering()
    records = perplexity_cost.recorded_costs(mock_revenium_client)
    assert [(record["transaction_id"], _costs(record)) for record in records] == [("pplx-chat-0001", REPORTED_COST)]


def test_the_perplexity_integration_installs_when_imported_after_the_langchain_callback():
    records = _run_without_the_integration("integration-imported-after-callback")
    assert [(record["transaction_id"], _costs(record)) for record in records] == [("pplx-chat-0001", REPORTED_COST)]


def test_only_the_perplexity_integration_wraps_count_as_checking_caps_for_chat_perplexity():
    assert _model_call._CAP_CHECKING_WRAPS["perplexity"] == (
        "perplexity:openai.resources.chat.completions.Completions.create",
        "perplexity.resources.chat.completions.CompletionsResource.create",
        "perplexity.resources.responses.responses.ResponsesResource.create",
    )


@pytest.mark.parametrize("scenario", [
    pytest.param("callback-chat-openai", id="invoke"),
    pytest.param("callback-chat-openai-ainvoke", id="ainvoke"),
    pytest.param("callback-chat-openai-env", id="openai-base-url-variable"),
    pytest.param("callback-chat-openai-root-client", id="callers-client"),
    pytest.param("callback-chat-openai-stream", id="stream"),
    pytest.param("callback-chat-openai-astream", id="astream"),
    pytest.param("callback-chat-openai-stream-env", id="stream-openai-base-url-variable"),
    pytest.param("callback-chat-openai-stream-root-client", id="stream-callers-client"),
    pytest.param("callback-chat-openai-streaming-invoke", id="streaming-invoke"),
    pytest.param("callback-chat-openai-streaming-ainvoke", id="streaming-ainvoke"),
    pytest.param("callback-chat-openai-stream-raw-response", id="stream-raw-response"),
])
def test_a_callback_only_chat_openai_call_to_perplexity_is_recorded_as_perplexity_with_its_cost(scenario):
    """BACK-4280: ChatOpenAI reports ls_provider "openai" whatever its base URL.

    BACK-4292: langchain-openai keeps only token counts from a stream, so a
    streamed call's cost is read from the last chunk Perplexity sent. The
    interim chunks carry a zero charge, which must not win.
    """
    records = _run_without_the_integration(scenario)
    assert len(records) == 1, records
    assert records[0]["provider"] == "PERPLEXITY"
    assert records[0]["transaction_id"].startswith("langchain-")
    assert _costs(records[0]) == REPORTED_COST


def _final_tokens(record):
    return record["input_token_count"], record["output_token_count"]


def test_each_streamed_batched_chat_openai_run_to_perplexity_keeps_the_cost_and_tokens_of_its_own_stream():
    """BACK-4292, BACK-4294: three streams in one batch end with three different charges and token counts."""
    records = _run_without_the_integration("callback-chat-openai-streaming-batch")
    assert len(records) == len(BATCH_COSTS), records
    assert all(record["provider"] == "PERPLEXITY" for record in records), records
    recorded = {(*_final_tokens(record), record["total_cost"]) for record in records}
    assert recorded == {(prompt_tokens, 3, total_cost) for total_cost, prompt_tokens in BATCH_COSTS.values()}


@pytest.mark.parametrize("scenario", [
    pytest.param("callback-chat-openai-stream", id="stream"),
    pytest.param("callback-chat-openai-astream", id="astream"),
    pytest.param("callback-chat-openai-streaming-invoke", id="streaming-invoke"),
    pytest.param("callback-chat-openai-streaming-ainvoke", id="streaming-ainvoke"),
    pytest.param("callback-chat-openai-stream-raw-response", id="stream-raw-response"),
])
def test_a_streamed_chat_openai_call_to_perplexity_keeps_perplexitys_final_token_counts(scenario):
    """BACK-4294: Perplexity repeats the usage so far on every chunk, and
    langchain-openai adds the chunks up, so only the last chunk's usage is the call's."""
    records = _run_without_the_integration(scenario)
    assert len(records) == 1, records
    assert _final_tokens(records[0]) == (2, 3)


@pytest.mark.parametrize("stream", [pytest.param(False, id="invoke"), pytest.param(True, id="stream")])
def test_a_chat_openai_call_to_perplexity_with_the_integration_imported_is_recorded_once(
        stream, mock_revenium_client):
    model = perplexity_cost.chat_openai_at_perplexity([UnifiedReveniumCallbackHandler()])
    if stream:
        list(model.stream("hi"))
    else:
        model.invoke("hi")
    calls.wait_for_metering()
    records = perplexity_cost.recorded_costs(mock_revenium_client)
    assert len(records) == 1, records
    assert records[0]["provider"] == "PERPLEXITY"
    assert not records[0]["transaction_id"].startswith("langchain-")


@pytest.mark.parametrize("scenario", [
    pytest.param("chat-openai-stream-perplexity-integration-first", id="perplexity-imported-first"),
    pytest.param("chat-openai-stream-openai-integration-first", id="openai-imported-first"),
])
def test_a_chat_openai_stream_to_perplexity_is_recorded_once_whichever_integration_is_imported_first(scenario):
    records = _run_without_the_integration(scenario)
    assert len(records) == 1, records
    assert records[0]["provider"] == "PERPLEXITY"
    assert not records[0]["transaction_id"].startswith("langchain-")
    assert _costs(records[0]) == REPORTED_COST


def test_a_chat_openai_call_to_perplexity_outside_a_metered_function_sends_no_record(
        monkeypatch, mock_revenium_client):
    """BACK-4320: with selective metering on, neither the Perplexity integration nor the callback
    records a call made outside a @revenium_meter function."""
    # The stub answers every call with the same response id, which an earlier test's transport record claimed.
    call_ownership.reset_claimed_response_ids()
    monkeypatch.setenv("REVENIUM_SELECTIVE_METERING", "true")
    model = perplexity_cost.chat_openai_at_perplexity([UnifiedReveniumCallbackHandler()])
    assert model.invoke("hi").text == perplexity_cost.ANSWER
    calls.wait_for_metering()
    assert perplexity_cost.recorded_costs(mock_revenium_client) == []


def test_a_chat_openai_call_to_perplexity_inside_a_metered_function_is_recorded_once_as_perplexity(
        monkeypatch, mock_revenium_client):
    """BACK-4280, BACK-4320: inside a @revenium_meter function the Perplexity integration records the
    call with Perplexity's cost, and the callback does not record it again."""
    call_ownership.reset_claimed_response_ids()
    monkeypatch.setenv("REVENIUM_SELECTIVE_METERING", "true")
    model = perplexity_cost.chat_openai_at_perplexity([UnifiedReveniumCallbackHandler()])
    revenium_meter()(model.invoke)("hi")
    calls.wait_for_metering()
    records = perplexity_cost.recorded_costs(mock_revenium_client)
    assert len(records) == 1, records
    assert records[0]["provider"] == "PERPLEXITY"
    assert not records[0]["transaction_id"].startswith("langchain-")
    assert records[0]["total_cost"] == TOTAL_COST


@pytest.mark.parametrize("base_url, provider", [
    pytest.param(None, "OPENAI", id="no-base-url"),
    pytest.param("https://api.openai.com/v1", "OPENAI", id="openai"),
    pytest.param(perplexity_cost.PERPLEXITY_BASE_URL, "PERPLEXITY", id="perplexity"),
])
def test_chat_openai_is_labelled_by_the_base_url_it_calls(base_url, provider):
    from langchain_core.load import dumpd
    from langchain_openai import ChatOpenAI

    serialized = dumpd(ChatOpenAI(model=perplexity_cost.PERPLEXITY_MODEL, api_key="stub", base_url=base_url))
    metadata = {"ls_provider": "openai"}
    labels = {"provider": provider, "model_source": provider}
    assert _model_call.provider_metadata_for(serialized, {}, metadata) == labels
    assert _model_call.transport_scope_for(serialized, {}, metadata) == frozenset({OPENAI})


OPENAI_BASE_URL = "https://api.openai.com/v1"


def _responses_create(base_url, stream=False):
    def create():
        client, _ = perplexity_cost.openai_clients_at(base_url)
        response = client.responses.create(model=perplexity_cost.PERPLEXITY_MODEL, input="hi", stream=stream)
        return list(response) if stream else response
    return create


def _async_responses_create(base_url, stream=False):
    async def create():
        _, client = perplexity_cost.openai_clients_at(base_url)
        response = await client.responses.create(model=perplexity_cost.PERPLEXITY_MODEL, input="hi", stream=stream)
        if stream:
            return await _drain(response)
        return response
    return lambda: asyncio.run(create())


async def _drain(events):
    return [event async for event in events]


def _chat_openai_on_the_responses_api(call):
    def run():
        model = perplexity_cost.chat_openai_at_perplexity([UnifiedReveniumCallbackHandler()], use_responses_api=True)
        call(model)
    return run


RESPONSES_API_CALLS_TO_PERPLEXITY = [
    pytest.param(_responses_create(perplexity_cost.PERPLEXITY_BASE_URL), id="create"),
    pytest.param(_async_responses_create(perplexity_cost.PERPLEXITY_BASE_URL), id="async-create"),
    pytest.param(_responses_create(perplexity_cost.PERPLEXITY_BASE_URL, stream=True), id="create-stream"),
    pytest.param(_async_responses_create(perplexity_cost.PERPLEXITY_BASE_URL, stream=True), id="async-create-stream"),
    pytest.param(_chat_openai_on_the_responses_api(lambda model: model.invoke("hi")), id="chat-openai-invoke"),
    pytest.param(_chat_openai_on_the_responses_api(lambda model: list(model.stream("hi"))), id="chat-openai-stream"),
]


@pytest.mark.parametrize("call", RESPONSES_API_CALLS_TO_PERPLEXITY)
def test_a_responses_api_call_to_perplexity_is_recorded_as_perplexity_with_its_cost(call, mock_revenium_client):
    """BACK-4293: the Responses API reaches Perplexity through the OpenAI client too."""
    # The stub answers every call with the same response id, which an earlier test's record claimed.
    call_ownership.reset_claimed_response_ids()
    call()
    calls.wait_for_metering()
    records = perplexity_cost.recorded_costs(mock_revenium_client)
    assert len(records) == 1, records
    assert (records[0]["provider"], records[0]["model_source"]) == ("PERPLEXITY", "PERPLEXITY")
    assert records[0]["transaction_id"] == "resp_0001"
    assert _costs(records[0]) == REPORTED_COST


def test_a_responses_api_call_to_openai_is_still_recorded_as_openai_without_a_reported_cost(mock_revenium_client):
    call_ownership.reset_claimed_response_ids()
    _responses_create(OPENAI_BASE_URL)()
    calls.wait_for_metering()
    records = perplexity_cost.recorded_costs(mock_revenium_client)
    assert len(records) == 1, records
    assert (records[0]["provider"], records[0]["total_cost"]) == ("OPENAI", None)


def _selectively_metered_responses_model(base_url_source, monkeypatch):
    call_ownership.reset_claimed_response_ids()
    monkeypatch.setenv("REVENIUM_SELECTIVE_METERING", "true")
    # The env source sets OPENAI_BASE_URL itself; this restores it afterwards.
    monkeypatch.setenv("OPENAI_BASE_URL", perplexity_cost.PERPLEXITY_BASE_URL)
    return perplexity_cost.chat_openai_at_perplexity([UnifiedReveniumCallbackHandler()], base_url_source,
                                                     use_responses_api=True)


@pytest.mark.parametrize("base_url_source", ["env", "root_client"])
def test_a_chat_openai_responses_call_to_perplexity_outside_a_metered_function_sends_no_record(
        base_url_source, monkeypatch, mock_revenium_client):
    """BACK-4320: with selective metering on, a call outside a @revenium_meter function is not recorded."""
    model = _selectively_metered_responses_model(base_url_source, monkeypatch)
    assert model.invoke("hi").text == perplexity_cost.ANSWER
    calls.wait_for_metering()
    assert perplexity_cost.recorded_costs(mock_revenium_client) == []


@pytest.mark.parametrize("base_url_source", ["env", "root_client"])
def test_a_chat_openai_responses_call_to_perplexity_inside_a_metered_function_is_recorded_once_with_its_cost(
        base_url_source, monkeypatch, mock_revenium_client):
    """BACK-4293, BACK-4320: inside a @revenium_meter function the OpenAI client records the call
    as Perplexity with Perplexity's cost, which langchain-openai drops from a Responses answer."""
    model = _selectively_metered_responses_model(base_url_source, monkeypatch)
    revenium_meter()(model.invoke)("hi")
    calls.wait_for_metering()
    records = perplexity_cost.recorded_costs(mock_revenium_client)
    assert len(records) == 1, records
    assert (records[0]["provider"], records[0]["model_source"]) == ("PERPLEXITY", "PERPLEXITY")
    assert records[0]["transaction_id"] == "resp_0001"
    assert _costs(records[0]) == REPORTED_COST

