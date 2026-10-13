"""The gateway takes an OpenAI cache write out of the prompt and reports it as
cache creation, once, on the guardrail and on the deprecated callback
(BACK-3998).

An OpenAI upstream reports the write only in
``prompt_tokens_details.cache_write_tokens`` (``input_tokens_details`` on the
Responses spelling). LiteLLM also mirrors an Anthropic or Bedrock write into
that nested field, beside the top-level ``cache_creation_input_tokens``; that
mirror is owned by the Anthropic folding and must not be taken out again.
"""
import datetime
from unittest.mock import patch

import pytest

from revenium_middleware._core.cache_tokens import billable_input_tokens

pytest.importorskip("litellm")
pytest.importorskip("fastapi")

from litellm.types.utils import Usage  # noqa: E402

from revenium_middleware.litellm.proxy import middleware as deprecated_callback  # noqa: E402
from revenium_middleware.litellm.proxy.guardrail import ReveniumGuardrail  # noqa: E402

from .proxy_hook_harness import (  # noqa: E402
    GuardrailResponse,
    base_kwargs,
    drive_metering,
    guardrail_data,
    make_key_dict,
    make_success_response,
    run_hook,
    submitted_args,
)
from .test_proxy_cache_token_parity import (  # noqa: E402
    ANTHROPIC_MODEL,
    CACHED_USAGE,
    anthropic_request,
    converted_usage,
)

OPENAI_MODEL = "gpt-5.6-luna"
NOW = datetime.datetime(2026, 10, 8, 14, 45, 41, tzinfo=datetime.timezone.utc)
LATER = NOW + datetime.timedelta(milliseconds=250)


def chat_shaped(prompt, completion, cached, written):
    return Usage(
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=prompt + completion,
        prompt_tokens_details={"cached_tokens": cached, "cache_write_tokens": written},
    )


def responses_shaped(prompt, completion, cached, written):
    return {
        "input_tokens": prompt,
        "output_tokens": completion,
        "total_tokens": prompt + completion,
        "input_tokens_details": {"cached_tokens": cached, "cache_write_tokens": written},
    }


@pytest.fixture
def guardrail():
    from revenium_middleware.litellm.proxy import _metering_owner

    _metering_owner.reset_metering_owner()
    instance = ReveniumGuardrail(
        guardrail_name="revenium", event_hook=["pre_call", "post_call"], default_on=True
    )
    yield instance
    _metering_owner.reset_metering_owner()


@pytest.fixture
def submit():
    with patch("revenium_middleware.litellm.proxy.guardrail.submit_ai_event") as mock, \
            patch("revenium_middleware.litellm.proxy.guardrail.run_async_in_thread",
                  side_effect=drive_metering):
        yield mock


async def through_the_guardrail(guardrail, usage, data=None):
    await guardrail.async_post_call_success_hook(
        data=data or guardrail_data(stream=False, model=OPENAI_MODEL, custom_llm_provider="openai"),
        user_api_key_dict=make_key_dict(),
        response=GuardrailResponse(response_id="chatcmpl-openai-write", usage=usage),
    )


@pytest.mark.asyncio
async def test_guardrail_takes_openai_cache_writes_out_of_the_prompt(guardrail, submit):
    await through_the_guardrail(guardrail, chat_shaped(4583, 5, 0, 4580))

    args = submitted_args(submit)
    assert args["input_token_count"] == 3
    assert args["cache_creation_token_count"] == 4580
    assert args["cache_read_token_count"] == 0
    assert args["total_token_count"] == 4588


@pytest.mark.asyncio
async def test_guardrail_takes_cache_writes_out_of_a_responses_shaped_usage(guardrail, submit):
    await through_the_guardrail(guardrail, responses_shaped(4583, 5, 0, 4580))

    args = submitted_args(submit)
    assert args["input_token_count"] == 3
    assert args["cache_creation_token_count"] == 4580
    assert args["total_token_count"] == 4588


@pytest.mark.asyncio
async def test_guardrail_subtracts_only_the_write_when_a_call_reads_and_writes(guardrail, submit):
    await through_the_guardrail(guardrail, chat_shaped(6000, 10, 4000, 1500))

    args = submitted_args(submit)
    assert args["input_token_count"] == 4500
    assert args["cache_read_token_count"] == 4000
    assert args["cache_creation_token_count"] == 1500


@pytest.mark.asyncio
async def test_guardrail_subtracts_only_the_write_from_a_responses_shaped_usage(guardrail, submit):
    """The read is not extracted from the Responses spelling on this route
    (pre-existing, out of scope), so only input and creation are pinned here."""
    await through_the_guardrail(guardrail, responses_shaped(6000, 10, 4000, 1500))

    args = submitted_args(submit)
    assert args["input_token_count"] == 4500
    assert args["cache_creation_token_count"] == 1500


@pytest.mark.asyncio
async def test_guardrail_anthropic_row_is_priced_as_before(guardrail, submit):
    """Regression guard for the LiteLLM mirror: the Anthropic write is taken out
    of the prompt once, by the folding, never a second time."""
    usage = converted_usage(CACHED_USAGE)
    assert usage.prompt_tokens_details.cache_write_tokens == CACHED_USAGE["cache_creation_input_tokens"]

    await through_the_guardrail(guardrail, usage, data=anthropic_request())

    args = submitted_args(submit)
    assert args["input_token_count"] == billable_input_tokens(
        usage.prompt_tokens,
        CACHED_USAGE["cache_read_input_tokens"],
        CACHED_USAGE["cache_creation_input_tokens"],
        True,
    )
    assert args["cache_creation_token_count"] == CACHED_USAGE["cache_creation_input_tokens"]


@pytest.mark.asyncio
async def test_guardrail_responses_bridge_to_anthropic_bills_the_write_once(guardrail, submit):
    """LiteLLM's ``/v1/responses`` bridge to an Anthropic upstream reports the
    write only in ``input_tokens_details``: it now leaves input and is priced
    once, at the creation rate."""
    data = guardrail_data(stream=False, model=ANTHROPIC_MODEL, custom_llm_provider="anthropic",
                          call_type="responses")

    await through_the_guardrail(guardrail, responses_shaped(4583, 5, 0, 4580), data=data)

    args = submitted_args(submit)
    assert args["input_token_count"] == 3
    assert args["cache_creation_token_count"] == 4580


def _success(usage):
    run_hook(deprecated_callback.proxy_handler_instance.async_log_success_event(
        base_kwargs(model=OPENAI_MODEL, custom_llm_provider="openai"),
        make_success_response(usage), NOW, LATER,
    ))


def _failure(usage):
    error = RuntimeError("upstream failure")
    error.usage = usage
    run_hook(deprecated_callback.proxy_handler_instance.async_log_failure_event(
        base_kwargs(model=OPENAI_MODEL, custom_llm_provider="openai"), error, NOW, LATER
    ))


@pytest.mark.parametrize("hook", [_success, _failure], ids=["success", "failure"])
@pytest.mark.parametrize(
    "usage, expected",
    [
        (chat_shaped(4583, 5, 0, 4580), {"input_token_count": 3, "cache_read_token_count": 0,
                                         "cache_creation_token_count": 4580}),
        (chat_shaped(6000, 10, 4000, 1500), {"input_token_count": 4500, "cache_read_token_count": 4000,
                                             "cache_creation_token_count": 1500}),
    ],
    ids=["writes", "reads_and_writes"],
)
def test_deprecated_callback_takes_openai_cache_writes_out_of_the_prompt(hook, usage, expected):
    with patch.object(deprecated_callback, "run_async_in_thread", side_effect=drive_metering), \
            patch.object(deprecated_callback, "get_client", return_value=object()), \
            patch.object(deprecated_callback, "submit_ai_event") as mock_submit:
        hook(usage)

        args = submitted_args(mock_submit)
        assert {field: args[field] for field in expected} == expected
