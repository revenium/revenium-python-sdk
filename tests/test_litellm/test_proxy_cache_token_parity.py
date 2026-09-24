"""A cached call costs the same whichever leg reports it (BACK-3334).

Revenium prices ``input_token_count`` at the model's input rate and
``cache_read_token_count`` / ``cache_creation_token_count`` at their own rates,
then adds the three up. LiteLLM's Anthropic conversion reports ``prompt_tokens``
as input + cache read + cache creation (``AnthropicConfig.calculate_usage``,
identical on 1.93.0 and 1.102.0), so a gateway row built on ``prompt_tokens``
paid for every cached token twice: on dev, one Claude Code call recorded
$0.0105 on its own telemetry and $0.0470 through the proxy plugin.

What these tests pin is the parity itself: for one Anthropic usage block, the
proxy's payload must carry the same input and cache counts the direct Anthropic
integration (``revenium_middleware.anthropic.middleware``) meters for it, on
every shape the proxy sees that block in -- the provider's own JSON, LiteLLM's
converted usage, and the assembled totals of a stream.

The measured call is the fixture: input 10, output 235, cache read 31,604,
cache creation 4,910 (dev trace 04f209d6-2d8c-4a4d-aa09-76cac15b151e), so a
regression reproduces the customer's 4.5x directly.
"""
import datetime
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from revenium_middleware._core.cache_tokens import billable_input_tokens

pytest.importorskip("litellm")
pytest.importorskip("fastapi")

from litellm.llms.anthropic.chat.transformation import AnthropicConfig  # noqa: E402

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

ANTHROPIC_MODEL = "claude-haiku-4-5-20251001"
NOW = datetime.datetime(2026, 9, 22, 14, 45, 41, tzinfo=datetime.timezone.utc)
LATER = NOW + datetime.timedelta(milliseconds=250)

CACHED_USAGE = {
    "input_tokens": 10,
    "output_tokens": 235,
    "cache_read_input_tokens": 31604,
    "cache_creation_input_tokens": 4910,
}
UNCACHED_USAGE = {"input_tokens": 898, "output_tokens": 11}

# The row both legs must agree on. The total is every priced bucket summed,
# which is what Claude Code's own telemetry row carried for this call (36,759)
# and what LiteLLM reports as total_tokens for the same usage.
EXPECTED_INPUT_TOKENS = 10
EXPECTED_OUTPUT_TOKENS = 235
EXPECTED_CACHE_READ_TOKENS = 31604
EXPECTED_CACHE_CREATION_TOKENS = 4910
EXPECTED_TOTAL_TOKENS = 36759

PRICED_FIELDS = (
    "input_token_count",
    "cache_read_token_count",
    "cache_creation_token_count",
)


def anthropic_body(usage, response_id="msg_01Back3334CachedCall"):
    """A non-streamed ``/v1/messages`` body, as the proxy is handed it."""
    return {
        "id": response_id,
        "type": "message",
        "role": "assistant",
        "model": ANTHROPIC_MODEL,
        "content": [{"type": "text", "text": "Metered on the Anthropic route."}],
        "stop_reason": "end_turn",
        "usage": dict(usage),
    }


def anthropic_request(**kwargs):
    """The request dict a guardrail hook gets for a call Anthropic served.

    The upstream's identity rides on the logging object, which is where a live
    litellm 1.102.0 proxy puts it on this route -- nothing else the hook is
    handed on ``/v1/messages`` names a provider at all.
    """
    return guardrail_data(
        stream=False,
        model=ANTHROPIC_MODEL,
        custom_llm_provider="anthropic",
        call_type="anthropic_messages",
        **kwargs,
    )


def converted_usage(usage):
    """The usage LiteLLM hands us once it has converted an Anthropic body.

    Built by the vendor's own entry point rather than by hand, so the fixture
    tracks what LiteLLM actually folds into ``prompt_tokens`` instead of what
    we assume it folds.
    """
    return AnthropicConfig().calculate_usage(dict(usage), None)


def direct_anthropic_payload(usage):
    """What ``revenium_middleware.anthropic.middleware`` meters for one usage block.

    Driven through the real ``create_wrapper``, so this is the integration's
    own answer rather than a restatement of it. ``total_tokens`` is supplied
    because that path reads it off the usage object; the total is asserted
    against the priced buckets rather than against this value, which no
    provider sends.
    """
    from revenium_middleware.anthropic.middleware import create_wrapper

    wrapper = getattr(create_wrapper, "_self_wrapper", create_wrapper)
    response = MagicMock()
    response.id = "msg_direct_anthropic"
    response.model = ANTHROPIC_MODEL
    response.stop_reason = "end_turn"
    response.usage.input_tokens = usage["input_tokens"]
    response.usage.output_tokens = usage["output_tokens"]
    response.usage.cache_read_input_tokens = usage.get("cache_read_input_tokens", 0)
    response.usage.cache_creation_input_tokens = usage.get("cache_creation_input_tokens", 0)
    response.usage.total_tokens = sum(usage.values())

    with patch("revenium_middleware.anthropic.middleware.submit_ai_event") as submit, \
            patch("revenium_middleware.run_async_in_thread", side_effect=drive_metering):
        wrapper(
            MagicMock(return_value=response),
            None,
            (),
            {"messages": [{"role": "user", "content": "Hello"}], "model": ANTHROPIC_MODEL},
        )
        return submitted_args(submit)


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


def assert_priced_like_the_direct_integration(args, usage):
    direct = direct_anthropic_payload(usage)
    assert {field: args[field] for field in PRICED_FIELDS} == {
        field: direct[field] for field in PRICED_FIELDS
    }
    assert args["output_token_count"] == direct["output_token_count"]


class TestOneCallIsPricedTheSameOnBothLegs:
    """Each shape the proxy meets a cached Anthropic call in."""

    @pytest.mark.asyncio
    async def test_the_providers_own_json_is_priced_like_the_direct_integration(
        self, guardrail, submit
    ):
        await guardrail.async_post_call_success_hook(
            data=anthropic_request(),
            user_api_key_dict=make_key_dict(),
            response=anthropic_body(CACHED_USAGE),
        )

        args = submitted_args(submit)
        assert_priced_like_the_direct_integration(args, CACHED_USAGE)
        assert args["input_token_count"] == EXPECTED_INPUT_TOKENS
        assert args["output_token_count"] == EXPECTED_OUTPUT_TOKENS
        assert args["total_token_count"] == EXPECTED_TOTAL_TOKENS

    @pytest.mark.asyncio
    async def test_litellms_converted_usage_is_priced_like_the_direct_integration(
        self, guardrail, submit
    ):
        """The shape that carried the defect: ``prompt_tokens`` is 36,524 here."""
        usage = converted_usage(CACHED_USAGE)
        assert usage.prompt_tokens == (
            EXPECTED_INPUT_TOKENS + EXPECTED_CACHE_READ_TOKENS + EXPECTED_CACHE_CREATION_TOKENS
        )

        await guardrail.async_post_call_success_hook(
            data=anthropic_request(),
            user_api_key_dict=make_key_dict(),
            response=GuardrailResponse(response_id="chatcmpl-converted", usage=usage),
        )

        args = submitted_args(submit)
        assert_priced_like_the_direct_integration(args, CACHED_USAGE)
        assert args["total_token_count"] == EXPECTED_TOTAL_TOKENS

    @pytest.mark.asyncio
    async def test_an_assembled_stream_is_priced_like_the_direct_integration(
        self, guardrail, submit
    ):
        """The streamed ``/v1/messages`` route meters from the logging event."""
        response = GuardrailResponse(
            response_id="msg_streamed", usage=converted_usage(CACHED_USAGE)
        )
        kwargs = {
            "model": ANTHROPIC_MODEL,
            "call_type": "anthropic_messages",
            "custom_llm_provider": "anthropic",
            "stream": True,
            "litellm_call_id": "litellm-call-id-parity",
            "litellm_params": {
                "metadata": {
                    "headers": {},
                    "hidden_params": {"optional_params": {"stream": True}},
                }
            },
        }

        await guardrail.async_log_success_event(kwargs, response, NOW, LATER)

        args = submitted_args(submit)
        assert args["is_streamed"] is True
        assert_priced_like_the_direct_integration(args, CACHED_USAGE)
        assert args["total_token_count"] == EXPECTED_TOTAL_TOKENS

    @pytest.mark.asyncio
    async def test_a_call_without_cache_tokens_is_priced_exactly_as_before(
        self, guardrail, submit
    ):
        """The 898-token call: both legs already agreed, and still do."""
        await guardrail.async_post_call_success_hook(
            data=anthropic_request(),
            user_api_key_dict=make_key_dict(),
            response=GuardrailResponse(
                response_id="chatcmpl-uncached", usage=converted_usage(UNCACHED_USAGE)
            ),
        )

        args = submitted_args(submit)
        assert args["input_token_count"] == 898
        assert args["output_token_count"] == 11
        assert args["total_token_count"] == 909
        assert args["cache_read_token_count"] == 0
        assert args["cache_creation_token_count"] == 0


@patch.object(deprecated_callback, "run_async_in_thread", side_effect=drive_metering)
@patch.object(deprecated_callback, "get_client", return_value=object())
@patch.object(deprecated_callback, "submit_ai_event")
class TestTheDeprecatedCallbackPricesItTheSameWay:
    """The callback a proxy mid-migration still runs gets the same split.

    It only ever sees LiteLLM's converted usage (see
    ``test_proxy_anthropic_route_metering``), so the conversion fixture is the
    whole of its exposure to this defect.
    """

    def test_converted_usage_is_priced_like_the_direct_integration(
        self, mock_submit, _get_client, _run
    ):
        response = make_success_response(converted_usage(CACHED_USAGE))

        run_hook(deprecated_callback.proxy_handler_instance.async_log_success_event(
            base_kwargs(model=ANTHROPIC_MODEL, custom_llm_provider="anthropic"), response, NOW, LATER
        ))

        args = submitted_args(mock_submit)
        assert_priced_like_the_direct_integration(args, CACHED_USAGE)
        assert args["total_token_count"] == EXPECTED_TOTAL_TOKENS

    def test_a_failed_call_carrying_cached_usage_is_priced_the_same(
        self, mock_submit, _get_client, _run
    ):
        error = RuntimeError("upstream failure")
        error.usage = converted_usage(CACHED_USAGE)

        run_hook(deprecated_callback.proxy_handler_instance.async_log_failure_event(
            base_kwargs(model=ANTHROPIC_MODEL, custom_llm_provider="anthropic"), error, NOW, LATER
        ))

        args = submitted_args(mock_submit)
        assert args["input_token_count"] == EXPECTED_INPUT_TOKENS
        assert args["cache_read_token_count"] == EXPECTED_CACHE_READ_TOKENS
        assert args["cache_creation_token_count"] == EXPECTED_CACHE_CREATION_TOKENS


class TestOnlyAnAnthropicUpstreamHasItsCacheOverlapRemoved:
    """Who served the call decides the arithmetic -- never the counts.

    The platform removes the overlap itself for the OpenAI-shaped cache pool
    (``AICompletionMetricProcessor.normalizeCacheOverlap``: OPENAI, GROQ, XAI,
    AZURE, GEMINI, decided from the catalog model, not from this row's
    ``provider=LITELLM``), and it expects the stored input count to be gross.
    Subtracting here as well would net those tokens out twice, clamp to zero
    and price the input leg at nothing -- the same class of billing bug as the
    one being fixed, with the sign flipped. Anthropic is deliberately outside
    that set, so the gateway row is the only place its overlap can go.
    """

    OPENAI_CACHED_USAGE = SimpleNamespace(
        prompt_tokens=1000,
        completion_tokens=50,
        total_tokens=1050,
        prompt_tokens_details=SimpleNamespace(cached_tokens=800, cache_creation_tokens=0),
    )

    @pytest.mark.asyncio
    async def test_an_openai_shaped_row_keeps_its_gross_prompt_count(
        self, guardrail, submit
    ):
        await guardrail.async_post_call_success_hook(
            data=guardrail_data(stream=False, model="gpt-4o", custom_llm_provider="openai"),
            user_api_key_dict=make_key_dict(),
            response=GuardrailResponse(
                response_id="chatcmpl-openai", usage=self.OPENAI_CACHED_USAGE
            ),
        )

        args = submitted_args(submit)
        assert args["input_token_count"] == 1000
        assert args["cache_read_token_count"] == 800
        assert args["cache_creation_token_count"] == 0

    @pytest.mark.asyncio
    async def test_an_event_naming_no_upstream_keeps_its_gross_prompt_count(
        self, guardrail, submit
    ):
        """Unknown is treated as not-Anthropic: the recoverable failure."""
        await guardrail.async_post_call_success_hook(
            data=guardrail_data(stream=False, model="gpt-4o"),
            user_api_key_dict=make_key_dict(),
            response=GuardrailResponse(
                response_id="chatcmpl-unknown", usage=self.OPENAI_CACHED_USAGE
            ),
        )

        assert submitted_args(submit)["input_token_count"] == 1000

    @pytest.mark.asyncio
    async def test_a_caller_cannot_name_the_upstream_from_its_own_request_body(
        self, guardrail, submit
    ):
        """metadata.hidden_params is the caller's JSON on /v1/messages."""
        await guardrail.async_post_call_success_hook(
            data=guardrail_data(
                stream=False,
                model="gpt-4o",
                metadata={"hidden_params": {"custom_llm_provider": "anthropic",
                                             "call_type": "anthropic_messages"}},
            ),
            user_api_key_dict=make_key_dict(),
            response=GuardrailResponse(
                response_id="chatcmpl-spoof", usage=self.OPENAI_CACHED_USAGE
            ),
        )

        assert submitted_args(submit)["input_token_count"] == 1000

    @pytest.mark.asyncio
    async def test_an_anthropic_row_has_its_overlap_removed(self, guardrail, submit):
        await guardrail.async_post_call_success_hook(
            data=anthropic_request(),
            user_api_key_dict=make_key_dict(),
            response=GuardrailResponse(
                response_id="chatcmpl-anthropic", usage=converted_usage(CACHED_USAGE)
            ),
        )

        assert submitted_args(submit)["input_token_count"] == EXPECTED_INPUT_TOKENS

    def test_the_deprecated_callback_keeps_an_openai_shaped_count_gross(self):
        with patch.object(deprecated_callback, "run_async_in_thread", side_effect=drive_metering), \
                patch.object(deprecated_callback, "get_client", return_value=object()), \
                patch.object(deprecated_callback, "submit_ai_event") as mock_submit:
            run_hook(deprecated_callback.proxy_handler_instance.async_log_success_event(
                base_kwargs(model="gpt-4o", custom_llm_provider="openai"),
                make_success_response(self.OPENAI_CACHED_USAGE),
                NOW, LATER,
            ))

            args = submitted_args(mock_submit)
            assert args["input_token_count"] == 1000
            assert args["cache_read_token_count"] == 800

    def test_the_deprecated_callbacks_failure_path_keeps_it_gross_too(self):
        error = RuntimeError("upstream failure")
        error.usage = self.OPENAI_CACHED_USAGE
        with patch.object(deprecated_callback, "run_async_in_thread", side_effect=drive_metering), \
                patch.object(deprecated_callback, "get_client", return_value=object()), \
                patch.object(deprecated_callback, "submit_ai_event") as mock_submit:
            run_hook(deprecated_callback.proxy_handler_instance.async_log_failure_event(
                base_kwargs(model="gpt-4o", custom_llm_provider="openai"), error, NOW, LATER
            ))

            args = submitted_args(mock_submit)
            assert args["input_token_count"] == 1000
            assert args["cache_read_token_count"] == 800

    def test_the_helper_takes_a_folded_count_apart_only_when_told_to(self):
        assert billable_input_tokens(1000, 800, 0, True) == 200
        assert billable_input_tokens(1000, 800, 0, False) == 1000

    @pytest.mark.asyncio
    async def test_anthropics_own_count_is_taken_as_it_stands_beside_its_cache(
        self, guardrail, submit
    ):
        """The raw ``/v1/messages`` body: no prompt count to take anything out of."""
        await guardrail.async_post_call_success_hook(
            data=anthropic_request(),
            user_api_key_dict=make_key_dict(),
            response=anthropic_body({"input_tokens": 100, "output_tokens": 5,
                                     "cache_read_input_tokens": 20,
                                     "cache_creation_input_tokens": 0}),
        )

        args = submitted_args(submit)
        assert args["input_token_count"] == 100
        assert args["cache_read_token_count"] == 20

    @pytest.mark.asyncio
    async def test_contradictory_counts_are_reported_unchanged_and_said_out_loud(
        self, guardrail, submit, caplog
    ):
        usage = SimpleNamespace(
            prompt_tokens=10,
            completion_tokens=235,
            total_tokens=None,
            cache_read_input_tokens=31604,
            cache_creation_input_tokens=4910,
        )

        with caplog.at_level("WARNING", logger="revenium_middleware"):
            await guardrail.async_post_call_success_hook(
                data=anthropic_request(),
                user_api_key_dict=make_key_dict(),
                response=GuardrailResponse(response_id="chatcmpl-odd", usage=usage),
            )

        args = submitted_args(submit)
        assert args["input_token_count"] == EXPECTED_INPUT_TOKENS
        assert args["total_token_count"] == EXPECTED_TOTAL_TOKENS
        assert "Usage counts contradict each other" in caplog.text
