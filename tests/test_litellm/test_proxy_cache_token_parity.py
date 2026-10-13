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

from revenium_middleware._core import cache_tokens as cache_tokens_module
from revenium_middleware._core.cache_tokens import billable_input_tokens

pytest.importorskip("litellm")
pytest.importorskip("fastapi")

from litellm.llms.anthropic.chat.transformation import AnthropicConfig  # noqa: E402
from litellm.llms.bedrock.chat.converse_transformation import AmazonConverseConfig  # noqa: E402
from litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini import VertexGeminiConfig  # noqa: E402
from litellm.llms.vertex_ai.vertex_ai_partner_models.anthropic.transformation import (  # noqa: E402
    VertexAIAnthropicConfig,
)

from revenium_middleware.litellm.proxy import middleware as deprecated_callback  # noqa: E402
from revenium_middleware.litellm.proxy.guardrail import ReveniumGuardrail  # noqa: E402

from .proxy_hook_harness import (  # noqa: E402
    GuardrailResponse,
    after_pre_call,
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
ONE_HOUR_USAGE = {
    **CACHED_USAGE,
    "cache_creation": {"ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 4910},
}
# Uncached input larger than the cache, so an overlap removed twice (or from a
# count that never held it) shows up as a wrong number rather than as the
# negative result the helper refuses to report.
LARGE_INPUT_USAGE = {
    "input_tokens": 9000,
    "output_tokens": 6,
    "cache_read_input_tokens": 2000,
    "cache_creation_input_tokens": 3000,
    "cache_creation": {"ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 3000},
}
TTL_FIELDS = ("cache_creation5m_token_count", "cache_creation1h_token_count")
CONTRADICTION_WARNING = "Usage counts contradict each other"

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
    response.usage.cache_creation = usage.get("cache_creation")
    response.usage.total_tokens = sum(
        value for value in usage.values() if isinstance(value, int)
    )

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


def ttl_counts(args):
    return {field: args.get(field, "<absent>") for field in TTL_FIELDS}


def assert_split_like_the_direct_integration(args, usage):
    direct = direct_anthropic_payload(usage)
    assert ttl_counts(direct) == {
        "cache_creation5m_token_count": usage["cache_creation"]["ephemeral_5m_input_tokens"],
        "cache_creation1h_token_count": usage["cache_creation"]["ephemeral_1h_input_tokens"],
    }
    assert ttl_counts(args) == ttl_counts(direct)
    assert {field: args[field] for field in PRICED_FIELDS} == {
        field: direct[field] for field in PRICED_FIELDS
    }


def streamed_logging_kwargs():
    return {
        "model": ANTHROPIC_MODEL,
        "call_type": "anthropic_messages",
        "custom_llm_provider": "anthropic",
        "stream": True,
        "litellm_call_id": "litellm-call-id-one-hour",
        "litellm_params": {
            "metadata": {
                "headers": {},
                "hidden_params": {"optional_params": {"stream": True}},
            }
        },
    }


class TestAOneHourWriteIsSplitOnEveryLeg:
    """A one-hour write is priced at the one-hour rate whichever leg reports it (BACK-3992)."""

    @pytest.mark.asyncio
    async def test_the_providers_own_json_carries_the_split(self, guardrail, submit):
        await guardrail.async_post_call_success_hook(
            data=anthropic_request(),
            user_api_key_dict=make_key_dict(),
            response=anthropic_body(ONE_HOUR_USAGE),
        )

        args = submitted_args(submit)
        assert_split_like_the_direct_integration(args, ONE_HOUR_USAGE)
        assert args["input_token_count"] == EXPECTED_INPUT_TOKENS

    @pytest.mark.asyncio
    async def test_litellms_converted_usage_carries_the_split(self, guardrail, submit):
        await guardrail.async_post_call_success_hook(
            data=anthropic_request(),
            user_api_key_dict=make_key_dict(),
            response=GuardrailResponse(
                response_id="chatcmpl-one-hour", usage=converted_usage(ONE_HOUR_USAGE)
            ),
        )

        args = submitted_args(submit)
        assert_split_like_the_direct_integration(args, ONE_HOUR_USAGE)
        assert args["input_token_count"] == EXPECTED_INPUT_TOKENS

    @pytest.mark.asyncio
    async def test_an_assembled_stream_carries_the_split(self, guardrail, submit):
        response = GuardrailResponse(
            response_id="msg_streamed_one_hour", usage=converted_usage(ONE_HOUR_USAGE)
        )

        await guardrail.async_log_success_event(streamed_logging_kwargs(), response, NOW, LATER)

        args = submitted_args(submit)
        assert args["is_streamed"] is True
        assert_split_like_the_direct_integration(args, ONE_HOUR_USAGE)
        assert args["input_token_count"] == EXPECTED_INPUT_TOKENS

    @pytest.mark.asyncio
    async def test_a_failed_gateway_call_sends_no_split(self, guardrail, submit):
        """No usage on the failure hook, so no split is invented for it."""
        await guardrail.async_post_call_failure_hook(
            request_data=after_pre_call(anthropic_request()),
            original_exception=RuntimeError("upstream failure"),
            user_api_key_dict=make_key_dict(),
        )

        args = submitted_args(submit)
        assert args["stop_reason"] == "ERROR"
        for field in TTL_FIELDS:
            assert field not in args

    @pytest.mark.asyncio
    async def test_a_large_uncached_input_is_netted_once_on_the_gateway(
        self, guardrail, submit, caplog
    ):
        with caplog.at_level("WARNING", logger="revenium_middleware"):
            await guardrail.async_post_call_success_hook(
                data=anthropic_request(),
                user_api_key_dict=make_key_dict(),
                response=GuardrailResponse(
                    response_id="chatcmpl-large", usage=converted_usage(LARGE_INPUT_USAGE)
                ),
            )
            await guardrail.async_post_call_success_hook(
                data=anthropic_request(),
                user_api_key_dict=make_key_dict(),
                response=anthropic_body(LARGE_INPUT_USAGE, response_id="msg_large"),
            )

        assert submit.call_count == 2
        for call in submit.call_args_list:
            args = call[0][1]
            assert args["input_token_count"] == 9000
            assert args["cache_read_token_count"] == 2000
            assert args["cache_creation_token_count"] == 3000
            assert ttl_counts(args) == {
                "cache_creation5m_token_count": 0,
                "cache_creation1h_token_count": 3000,
            }
        assert CONTRADICTION_WARNING not in caplog.text

    def test_the_deprecated_callback_carries_the_split(self):
        with patch.object(deprecated_callback, "run_async_in_thread", side_effect=drive_metering), \
                patch.object(deprecated_callback, "get_client", return_value=object()), \
                patch.object(deprecated_callback, "submit_ai_event") as mock_submit:
            run_hook(deprecated_callback.proxy_handler_instance.async_log_success_event(
                base_kwargs(model=ANTHROPIC_MODEL, custom_llm_provider="anthropic"),
                make_success_response(converted_usage(ONE_HOUR_USAGE)),
                NOW, LATER,
            ))

            args = submitted_args(mock_submit)
        assert_split_like_the_direct_integration(args, ONE_HOUR_USAGE)
        assert args["input_token_count"] == EXPECTED_INPUT_TOKENS

    def test_the_deprecated_callbacks_failure_path_carries_the_split(self):
        error = RuntimeError("upstream failure")
        error.usage = converted_usage(ONE_HOUR_USAGE)
        with patch.object(deprecated_callback, "run_async_in_thread", side_effect=drive_metering), \
                patch.object(deprecated_callback, "get_client", return_value=object()), \
                patch.object(deprecated_callback, "submit_ai_event") as mock_submit:
            run_hook(deprecated_callback.proxy_handler_instance.async_log_failure_event(
                base_kwargs(model=ANTHROPIC_MODEL, custom_llm_provider="anthropic"), error, NOW, LATER
            ))

            args = submitted_args(mock_submit)
        assert_split_like_the_direct_integration(args, ONE_HOUR_USAGE)
        assert args["input_token_count"] == EXPECTED_INPUT_TOKENS

    def test_the_deprecated_callback_nets_a_large_uncached_input_once(self, caplog):
        with patch.object(deprecated_callback, "run_async_in_thread", side_effect=drive_metering), \
                patch.object(deprecated_callback, "get_client", return_value=object()), \
                patch.object(deprecated_callback, "submit_ai_event") as mock_submit, \
                caplog.at_level("WARNING", logger="revenium_middleware"):
            run_hook(deprecated_callback.proxy_handler_instance.async_log_success_event(
                base_kwargs(model=ANTHROPIC_MODEL, custom_llm_provider="anthropic"),
                make_success_response(converted_usage(LARGE_INPUT_USAGE)),
                NOW, LATER,
            ))

            args = submitted_args(mock_submit)
        assert args["input_token_count"] == 9000
        assert ttl_counts(args) == {
            "cache_creation5m_token_count": 0,
            "cache_creation1h_token_count": 3000,
        }
        assert CONTRADICTION_WARNING not in caplog.text


# --- Claude on LiteLLM's Bedrock and Vertex AI routes (BACK-4162) ---------

BEDROCK_CLAUDE = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
BEDROCK_NOVA = "us.amazon.nova-micro-v1:0"
VERTEX_CLAUDE = "claude-haiku-4-5@20251001"
VERTEX_GEMINI = "gemini-2.5-flash-lite"

# Uncached input 12, read 2,000, write 3,000: the same call on every route.
ROUTE_INPUT, ROUTE_READ, ROUTE_WRITE = 12, 2000, 3000


def bedrock_converse_usage(input_tokens=ROUTE_INPUT, cache_read=ROUTE_READ, cache_write=ROUTE_WRITE):
    """LiteLLM's own Converse conversion, as the gateway is handed it."""
    return AmazonConverseConfig().transform_usage({
        "inputTokens": input_tokens,
        "outputTokens": 5,
        "totalTokens": input_tokens + cache_read + cache_write + 5,
        "cacheReadInputTokens": cache_read,
        "cacheWriteInputTokens": cache_write,
    })


def vertex_claude_usage(input_tokens=ROUTE_INPUT, cache_read=ROUTE_READ, cache_write=ROUTE_WRITE):
    """LiteLLM's own Vertex AI Claude conversion."""
    return VertexAIAnthropicConfig().calculate_usage({
        "input_tokens": input_tokens,
        "output_tokens": 5,
        "cache_read_input_tokens": cache_read,
        "cache_creation_input_tokens": cache_write,
    }, None)


def vertex_gemini_usage():
    return VertexGeminiConfig._calculate_usage({"usageMetadata": {
        "promptTokenCount": 4911, "candidatesTokenCount": 1, "totalTokenCount": 4912,
        "cachedContentTokenCount": 4084,
    }})


CLAUDE_ROUTES = {
    "bedrock-claude": ("bedrock", BEDROCK_CLAUDE, bedrock_converse_usage),
    "vertex-claude": ("vertex_ai", VERTEX_CLAUDE, vertex_claude_usage),
}


def assert_netted(args, read=ROUTE_READ, write=ROUTE_WRITE):
    assert args["input_token_count"] == ROUTE_INPUT
    assert args["cache_read_token_count"] == read
    assert args["cache_creation_token_count"] == write


def priced(args):
    return {field: args[field] for field in PRICED_FIELDS}


def logging_kwargs(provider, model):
    """What a live proxy's logging event carries for a streamed ``/v1/messages`` call,
    the one chat route the guardrail meters from that event (Claude Code's)."""
    return {
        "model": model,
        "call_type": "anthropic_messages",
        "custom_llm_provider": provider,
        "stream": True,
        "litellm_call_id": "litellm-call-id-" + provider,
        "litellm_params": {
            "custom_llm_provider": provider,
            "metadata": {"headers": {}, "hidden_params": {"optional_params": {"stream": True}}},
        },
    }


def callback_success(kwargs, response):
    with patch.object(deprecated_callback, "run_async_in_thread", side_effect=drive_metering), \
            patch.object(deprecated_callback, "get_client", return_value=object()), \
            patch.object(deprecated_callback, "submit_ai_event") as mock_submit:
        run_hook(deprecated_callback.proxy_handler_instance.async_log_success_event(
            kwargs, response, NOW, LATER
        ))
        return submitted_args(mock_submit)


def callback_failure(kwargs, error):
    with patch.object(deprecated_callback, "run_async_in_thread", side_effect=drive_metering), \
            patch.object(deprecated_callback, "get_client", return_value=object()), \
            patch.object(deprecated_callback, "submit_ai_event") as mock_submit:
        run_hook(deprecated_callback.proxy_handler_instance.async_log_failure_event(
            kwargs, error, NOW, LATER
        ))
        return submitted_args(mock_submit)


def without_the_guardrail(submitted):
    """Rows the guardrail sent; afterwards the proxy runs the callback alone,
    since the callback steps aside while a guardrail owns the metering."""
    from revenium_middleware.litellm.proxy import _metering_owner

    _metering_owner.reset_metering_owner()
    return [call[0][1] for call in submitted.call_args_list]


def client_row(provider, model, usage, litellm_model_name=None):
    from revenium_middleware.litellm.client import middleware as client

    response = SimpleNamespace(id="chatcmpl-client", model=model, usage=usage, finish_reason="stop",
                               _hidden_params={"litellm_model_name": litellm_model_name})
    with patch.object(client, "run_async_in_thread", side_effect=drive_metering), \
            patch.object(client, "get_client", return_value=object()), \
            patch.object(client, "submit_ai_event") as mock_submit:
        client.handle_response(response, NOW, {}, False, upstream_provider=provider)
        return submitted_args(mock_submit)


class TestClaudeOnBedrockAndVertexIsNettedOnEveryLeg:
    """Every gateway leg nets a Claude call on Bedrock or Vertex AI like a direct one."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("route", sorted(CLAUDE_ROUTES))
    @pytest.mark.parametrize("where", ["model_call_details", "litellm_model_name"])
    async def test_the_guardrail_nets_bedrock_and_vertex_claude(self, guardrail, submit, route, where):
        """The non-streamed hook finds the deployment on either LiteLLM-filled source."""
        provider, model, usage = CLAUDE_ROUTES[route]
        in_details = where == "model_call_details"
        await guardrail.async_post_call_success_hook(
            data=guardrail_data(stream=False, model="caller-alias", custom_llm_provider=provider,
                                upstream_model=model if in_details else None),
            user_api_key_dict=make_key_dict(),
            response=GuardrailResponse(
                response_id="chatcmpl-" + route, usage=usage(),
                hidden_params={} if in_details else {"litellm_model_name": provider + "/" + model},
            ),
        )

        assert_netted(submitted_args(submit))

    @pytest.mark.asyncio
    @pytest.mark.parametrize("route", sorted(CLAUDE_ROUTES))
    async def test_the_guardrails_streamed_hook_nets_bedrock_and_vertex_claude(self, guardrail, submit, route):
        provider, model, usage = CLAUDE_ROUTES[route]
        response = GuardrailResponse(response_id="msg-" + route, usage=usage())

        await guardrail.async_log_success_event(logging_kwargs(provider, model), response, NOW, LATER)

        args = submitted_args(submit)
        assert args["is_streamed"] is True
        assert_netted(args)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("provider, model", [("bedrock", BEDROCK_CLAUDE), ("vertex_ai", VERTEX_CLAUDE)])
    async def test_a_raw_messages_body_is_not_netted_a_second_time(self, guardrail, submit, provider, model):
        """Non-streamed ``/v1/messages`` hands over Anthropic's own count, already net."""
        await guardrail.async_post_call_success_hook(
            data=guardrail_data(stream=False, model=model, custom_llm_provider=provider,
                                call_type="anthropic_messages", upstream_model=model),
            user_api_key_dict=make_key_dict(),
            response=anthropic_body({"input_tokens": ROUTE_INPUT, "output_tokens": 5,
                                     "cache_read_input_tokens": ROUTE_READ,
                                     "cache_creation_input_tokens": ROUTE_WRITE}),
        )

        assert_netted(submitted_args(submit))

    @pytest.mark.asyncio
    async def test_the_guardrail_keeps_claude_gross_when_no_source_names_a_model(self, guardrail, submit):
        await guardrail.async_post_call_success_hook(
            data=guardrail_data(stream=False, model=BEDROCK_CLAUDE, custom_llm_provider="bedrock"),
            user_api_key_dict=make_key_dict(),
            response=GuardrailResponse(response_id="chatcmpl-no-model", usage=bedrock_converse_usage()),
        )

        args = submitted_args(submit)
        assert args["input_token_count"] == ROUTE_INPUT + ROUTE_READ + ROUTE_WRITE
        assert args["cache_read_token_count"] == ROUTE_READ
        assert args["cache_creation_token_count"] == ROUTE_WRITE

    @pytest.mark.asyncio
    async def test_the_guardrails_failure_hook_nets_bedrock_claude(self, guardrail, submit):
        error = RuntimeError("upstream failure")
        error.usage = bedrock_converse_usage()
        data = guardrail_data(stream=False, model="caller-alias", custom_llm_provider="bedrock",
                              upstream_model=BEDROCK_CLAUDE)

        await guardrail.async_post_call_failure_hook(
            request_data=after_pre_call(data), original_exception=error, user_api_key_dict=make_key_dict(),
        )

        args = submitted_args(submit)
        assert args["stop_reason"] == "ERROR"
        assert_netted(args)

    @pytest.mark.parametrize("route", sorted(CLAUDE_ROUTES))
    def test_the_deprecated_callback_nets_bedrock_and_vertex_claude(self, route):
        provider, model, usage = CLAUDE_ROUTES[route]

        assert_netted(callback_success(base_kwargs(model=model, custom_llm_provider=provider),
                                       make_success_response(usage())))

    @pytest.mark.parametrize("route", sorted(CLAUDE_ROUTES))
    def test_the_deprecated_callbacks_failure_path_nets_bedrock_and_vertex_claude(self, route):
        provider, model, usage = CLAUDE_ROUTES[route]
        error = RuntimeError("upstream failure")
        error.usage = usage()

        assert_netted(callback_failure(base_kwargs(model=model, custom_llm_provider=provider), error))

    @pytest.mark.asyncio
    @pytest.mark.parametrize("route", sorted(CLAUDE_ROUTES))
    async def test_every_leg_stores_the_same_input_for_the_same_call(self, guardrail, submit, route):
        provider, model, usage = CLAUDE_ROUTES[route]
        await guardrail.async_post_call_success_hook(
            data=guardrail_data(stream=False, model="caller-alias", custom_llm_provider=provider,
                                upstream_model=model),
            user_api_key_dict=make_key_dict(),
            response=GuardrailResponse(response_id="chatcmpl-parity", usage=usage()),
        )
        await guardrail.async_log_success_event(logging_kwargs(provider, model), GuardrailResponse(
            response_id="msg-parity", usage=usage()), NOW, LATER)

        rows = without_the_guardrail(submit) + [
            callback_success(base_kwargs(model=model, custom_llm_provider=provider), make_success_response(usage())),
            client_row(provider, model, usage(), litellm_model_name=provider + "/" + model),
            client_row(provider, model, usage()),
        ]
        assert len(rows) == 5
        assert [priced(row) for row in rows] == [{
            "input_token_count": ROUTE_INPUT,
            "cache_read_token_count": ROUTE_READ,
            "cache_creation_token_count": ROUTE_WRITE,
        }] * 5

    @pytest.mark.asyncio
    async def test_vertex_gemini_stays_gross_on_every_leg(self, guardrail, submit):
        await guardrail.async_post_call_success_hook(
            data=guardrail_data(stream=False, model="caller-alias", custom_llm_provider="vertex_ai",
                                upstream_model=VERTEX_GEMINI),
            user_api_key_dict=make_key_dict(),
            response=GuardrailResponse(response_id="chatcmpl-gemini", usage=vertex_gemini_usage(),
                                       hidden_params={"litellm_model_name": "vertex_ai/" + VERTEX_GEMINI}),
        )
        await guardrail.async_log_success_event(logging_kwargs("vertex_ai", VERTEX_GEMINI), GuardrailResponse(
            response_id="msg-gemini", usage=vertex_gemini_usage()), NOW, LATER)

        rows = without_the_guardrail(submit) + [
            callback_success(base_kwargs(model=VERTEX_GEMINI, custom_llm_provider="vertex_ai"),
                             make_success_response(vertex_gemini_usage())),
            client_row("vertex_ai", VERTEX_GEMINI, vertex_gemini_usage()),
        ]
        assert [priced(row) for row in rows] == [{
            "input_token_count": 4911,
            "cache_read_token_count": 4084,
            "cache_creation_token_count": 0,
        }] * 4

    @pytest.mark.asyncio
    async def test_a_caller_alias_naming_claude_does_not_net_a_nova_deployment(self, guardrail, submit):
        """The alias the caller picked appears in the request, the metadata and a
        non-streamed ``response.model``; only the deployment decides."""
        data = guardrail_data(
            stream=False, model="claude-haiku-alias", custom_llm_provider="bedrock",
            upstream_model=BEDROCK_NOVA, metadata={"model_group": "claude-haiku-alias"},
        )
        data["litellm_params"] = {"custom_llm_provider": "bedrock",
                                  "metadata": {"model_group": "claude-haiku-alias"}}
        response = GuardrailResponse(response_id="chatcmpl-alias", usage=bedrock_converse_usage(),
                                     hidden_params={"litellm_model_name": "bedrock/" + BEDROCK_NOVA})
        response.model = "claude-haiku-alias"

        await guardrail.async_post_call_success_hook(
            data=data, user_api_key_dict=make_key_dict(), response=response,
        )

        args = submitted_args(submit)
        assert args["input_token_count"] == ROUTE_INPUT + ROUTE_READ + ROUTE_WRITE
        assert args["cache_read_token_count"] == ROUTE_READ
        assert args["cache_creation_token_count"] == ROUTE_WRITE

    def test_the_deprecated_callback_does_not_net_a_nova_deployment_behind_a_claude_alias(self):
        kwargs = base_kwargs(model=BEDROCK_NOVA, custom_llm_provider="bedrock")
        kwargs["litellm_params"]["metadata"]["model_group"] = "claude-haiku-alias"
        response = make_success_response(bedrock_converse_usage())
        response.model = "claude-haiku-alias"

        args = callback_success(kwargs, response)

        assert args["input_token_count"] == ROUTE_INPUT + ROUTE_READ + ROUTE_WRITE
        assert args["cache_read_token_count"] == ROUTE_READ
        assert args["cache_creation_token_count"] == ROUTE_WRITE

    @pytest.mark.asyncio
    @pytest.mark.parametrize("cache_tokens", [0, None])
    async def test_a_claude_call_without_cache_keeps_its_input(self, guardrail, submit, cache_tokens):
        usage = SimpleNamespace(prompt_tokens=900, completion_tokens=5, total_tokens=905,
                                cache_read_input_tokens=cache_tokens, cache_creation_input_tokens=cache_tokens)
        await guardrail.async_post_call_success_hook(
            data=guardrail_data(stream=False, custom_llm_provider="bedrock", upstream_model=BEDROCK_CLAUDE),
            user_api_key_dict=make_key_dict(),
            response=GuardrailResponse(response_id="chatcmpl-uncached-bedrock", usage=usage),
        )

        args = submitted_args(submit)
        assert args["input_token_count"] == 900
        assert args["cache_read_token_count"] == 0
        assert args["cache_creation_token_count"] == 0

    @pytest.mark.asyncio
    async def test_cache_larger_than_a_claude_prompt_is_reported_unchanged(self, guardrail, submit, caplog):
        usage = SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=None,
                                cache_read_input_tokens=ROUTE_READ, cache_creation_input_tokens=ROUTE_WRITE)

        with caplog.at_level("WARNING", logger="revenium_middleware"):
            await guardrail.async_post_call_success_hook(
                data=guardrail_data(stream=False, custom_llm_provider="vertex_ai", upstream_model=VERTEX_CLAUDE),
                user_api_key_dict=make_key_dict(),
                response=GuardrailResponse(response_id="chatcmpl-odd-vertex", usage=usage),
            )

        args = submitted_args(submit)
        assert args["input_token_count"] == 10
        assert args["cache_read_token_count"] == ROUTE_READ
        assert args["cache_creation_token_count"] == ROUTE_WRITE
        assert CONTRADICTION_WARNING in caplog.text

    @pytest.mark.asyncio
    async def test_a_litellm_that_folds_only_the_read_has_only_the_read_taken_out(
        self, guardrail, submit, monkeypatch
    ):
        """LiteLLM 1.66 to 1.77.x on Converse: 6,000 input, 2,000 read and 4,700 written arrive as 8,000."""
        monkeypatch.setattr(cache_tokens_module, "_installed_litellm_version", lambda: "1.70.0", raising=False)
        usage = SimpleNamespace(prompt_tokens=8000, completion_tokens=5, total_tokens=8005,
                                cache_read_input_tokens=2000, cache_creation_input_tokens=4700)
        await guardrail.async_post_call_success_hook(
            data=guardrail_data(stream=False, custom_llm_provider="bedrock", upstream_model=BEDROCK_CLAUDE),
            user_api_key_dict=make_key_dict(),
            response=GuardrailResponse(response_id="chatcmpl-read-only-fold", usage=usage),
        )
        rows = without_the_guardrail(submit) + [callback_success(
            base_kwargs(model=BEDROCK_CLAUDE, custom_llm_provider="bedrock"), make_success_response(usage)
        )]

        assert len(rows) == 2
        for args in rows:
            assert args["input_token_count"] == 6000
            assert args["cache_read_token_count"] == 2000
            assert args["cache_creation_token_count"] == 4700


def test_the_installed_litellm_folds_the_cache_the_way_the_netting_assumes():
    """Pins the vendor behaviour the netting relies on, so a LiteLLM change turns this red."""
    converse = bedrock_converse_usage()
    anthropic = AnthropicConfig().calculate_usage({
        "input_tokens": ROUTE_INPUT, "output_tokens": 5,
        "cache_read_input_tokens": ROUTE_READ, "cache_creation_input_tokens": ROUTE_WRITE,
    }, None)

    for usage, converse_route in ((converse, True), (anthropic, False), (vertex_claude_usage(), False)):
        folds_write = cache_tokens_module.litellm_folds_cache_write(converse=converse_route)
        folded_write = ROUTE_WRITE if folds_write else 0
        assert usage.prompt_tokens == ROUTE_INPUT + ROUTE_READ + folded_write
