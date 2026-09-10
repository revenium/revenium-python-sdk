"""Claude Code traffic through a LiteLLM proxy must still be metered.

Claude Code speaks the Anthropic message shape, so a customer who fronts it with
their own LiteLLM proxy sends that traffic over ``/v1/messages``, not
``/v1/chat/completions``. Our gateway callback
(``revenium_middleware.litellm.proxy.middleware.MiddlewareHandler``) reads
OpenAI-shaped usage -- ``response_obj["usage"]`` with
``prompt_tokens`` / ``completion_tokens`` / ``total_tokens``, and ``response.id``
for the transaction id. If that callback were handed a raw Anthropic body
instead, the outcome would be no record at all: a silent zero, which is worse
than a double count because there is nothing left to reconcile.

BACK-2405 determined, against litellm 1.100.0, that it is not handed a raw
Anthropic body. Two vendor facts make the Anthropic-shaped route safe, and this
module pins both so a vendor upgrade that breaks either one fails here rather
than in a customer's billing:

1. Dispatch is not route-gated. ``Logging._async_success_handler_body``
   (``litellm/litellm_core_utils/litellm_logging.py``) awaits
   ``callback.async_log_success_event(...)`` for every ``CustomLogger`` in the
   callback list with no endpoint or call-type exclusion. The only route-based
   skip lives in the *sync* path and excludes ``CallTypes.pass_through``,
   commented "pass-through endpoints call async_log_success_event" -- i.e. the
   async hook we implement is precisely the one those routes use.
2. The Anthropic shape is normalised *before* our callback runs.
   ``Logging._success_handler_helper_fn`` calls
   ``_handle_anthropic_messages_response_logging(result=...)`` when
   ``call_type == CallTypes.anthropic_messages.value``, converting the body into
   a ``litellm.ModelResponse`` via ``AnthropicConfig().transform_response`` /
   ``transform_parsed_response``. Both Anthropic-shaped routes carry that call
   type (``API_ROUTE_TO_CALL_TYPES`` in ``litellm/types/utils.py``), and the
   pass-through variant converts independently in
   ``AnthropicPassthroughLoggingHandler.anthropic_passthrough_handler``.

Upstream issue 27518 / PR 27609 name ``async_pre_call_hook``, which
``MiddlewareHandler`` does not implement, so they do not bear on this path.

Fact 1 is not taken on trust: ``TestLiteLLMActuallyDispatchesToUsOnTheAnthropicRoute``
drives ``Logging.async_success_handler`` -- the real vendor entry point, with the
real ``anthropic_messages`` call type and ``MiddlewareHandler`` registered as a
callback -- so a vendor change that stopped dispatching to us on that route
fails even though route mapping and response conversion still work. The classes
above it drive ``async_log_success_event`` directly and pin the payload mapping.

Every assertion here is written to fail on an *absent* submission, because the
subject of the ticket is an absence: ``submitted_args`` asserts exactly one
submission before it will hand back a payload to inspect, and the negative
controls prove the unconverted Anthropic shape does not produce a well-formed
one -- so the module cannot pass regardless of which shape the callback sees.
"""
import datetime
from unittest.mock import patch

import pytest

# The middleware module imports litellm at import time; skip when the optional
# dependency is absent (mirrors the other optional-provider suites).
pytest.importorskip("litellm")

import litellm  # noqa: E402
from litellm.litellm_core_utils.litellm_logging import Logging  # noqa: E402
from litellm.types.utils import API_ROUTE_TO_CALL_TYPES, CallTypes  # noqa: E402

from revenium_middleware.litellm.proxy import middleware as mw  # noqa: E402
from .proxy_hook_harness import (  # noqa: E402
    SubscriptableResponse,
    base_kwargs,
    run_hook,
    run_inline,
    submitted_args,
)

NOW = datetime.datetime.now(datetime.timezone.utc)

ANTHROPIC_MODEL = "claude-sonnet-4-5-20250929"

# A fixed, non-streaming /v1/messages response body -- the shape Claude Code
# gets back through a LiteLLM proxy. Cache counts are included because LiteLLM
# folds them into prompt_tokens during conversion (see EXPECTED_* below).
ANTHROPIC_MESSAGES_BODY = {
    "id": "msg_01BACK2405AnthropicShapedRoute",
    "type": "message",
    "role": "assistant",
    "model": ANTHROPIC_MODEL,
    "content": [{"type": "text", "text": "Metered over the Anthropic-shaped route."}],
    "stop_reason": "end_turn",
    "stop_sequence": None,
    "usage": {
        "input_tokens": 123,
        "output_tokens": 45,
        "cache_read_input_tokens": 20,
        "cache_creation_input_tokens": 7,
    },
}

# LiteLLM's AnthropicConfig.calculate_usage reports prompt_tokens as the sum of
# input_tokens and both cache buckets, and total_tokens as prompt + completion.
# These are the numbers the metering payload must carry, not the raw
# input_tokens/output_tokens from the Anthropic body.
EXPECTED_PROMPT_TOKENS = 150
EXPECTED_COMPLETION_TOKENS = 45
EXPECTED_TOTAL_TOKENS = 195
EXPECTED_CACHE_READ_TOKENS = 20
EXPECTED_CACHE_CREATION_TOKENS = 7


def converted_anthropic_response():
    """The ``ModelResponse`` LiteLLM hands our callback on the Anthropic route.

    Built by calling the vendor's own conversion entry point --
    ``Logging._handle_anthropic_messages_response_logging`` -- rather than a
    hand-written stand-in, so this fixture tracks whatever LiteLLM actually
    does on that path instead of what we assume it does. That method is what
    ``_success_handler_helper_fn`` invokes for
    ``CallTypes.anthropic_messages``, immediately before the callback loop runs.

    ``optional_params`` is assigned directly because LiteLLM populates it during
    the pre-call stage we are deliberately not running here; the conversion
    reads it for the (unset) ``speed`` parameter.
    """
    logging_obj = Logging(
        model=ANTHROPIC_MODEL,
        messages=[{"role": "user", "content": "meter me"}],
        stream=False,
        call_type=CallTypes.anthropic_messages.value,
        start_time=NOW,
        litellm_call_id="back-2405-call",
        function_id="back-2405-fn",
    )
    logging_obj.optional_params = {}
    return logging_obj._handle_anthropic_messages_response_logging(
        result=dict(ANTHROPIC_MESSAGES_BODY)
    )


def openai_shaped_response(response_id="chatcmpl-back-2405-openai-control"):
    """An OpenAI-shaped ``ModelResponse``, as the /v1/chat/completions route yields."""
    response = litellm.ModelResponse(id=response_id, model="gpt-4o-mini")
    setattr(
        response,
        "usage",
        litellm.Usage(
            prompt_tokens=EXPECTED_PROMPT_TOKENS,
            completion_tokens=EXPECTED_COMPLETION_TOKENS,
            total_tokens=EXPECTED_TOTAL_TOKENS,
        ),
    )
    return response


class TestAnthropicRouteConversionIsStillInPlace:
    """Vendor-side preconditions the metering path depends on.

    These do not exercise our code; they pin the two litellm facts that make the
    Anthropic-shaped route safe, so an upgrade that removes either is caught by
    a failing test instead of by missing revenue.
    """

    @pytest.mark.parametrize("route", ["/v1/messages", "/anthropic/v1/messages"])
    def test_both_anthropic_routes_map_to_the_converting_call_type(self, route):
        # The conversion in _success_handler_helper_fn is keyed on
        # CallTypes.anthropic_messages. A route that stopped resolving to it
        # would reach our callback unconverted.
        assert CallTypes.anthropic_messages in API_ROUTE_TO_CALL_TYPES[route]

    def test_openai_route_does_not_go_through_the_anthropic_conversion(self):
        assert CallTypes.anthropic_messages not in API_ROUTE_TO_CALL_TYPES["/v1/chat/completions"]

    def test_conversion_yields_an_openai_shaped_model_response(self):
        converted = converted_anthropic_response()

        assert isinstance(converted, litellm.ModelResponse)
        # The three fields the callback meters on, none of which exist on the
        # Anthropic body it was built from.
        assert converted["usage"].prompt_tokens == EXPECTED_PROMPT_TOKENS
        assert converted["usage"].completion_tokens == EXPECTED_COMPLETION_TOKENS
        assert converted["usage"].total_tokens == EXPECTED_TOTAL_TOKENS


@patch.object(mw, "run_async_in_thread", side_effect=run_inline)
@patch.object(mw, "get_client", return_value=object())
@patch.object(mw, "submit_ai_event")
class TestAnthropicShapedRouteIsMetered:
    def test_anthropic_shaped_response_produces_exactly_one_submission(
        self, mock_submit, _get_client, _run
    ):
        converted = converted_anthropic_response()

        run_hook(mw.proxy_handler_instance.async_log_success_event(
            base_kwargs(model=ANTHROPIC_MODEL), converted, NOW, NOW
        ))

        # submitted_args asserts call_count == 1, so both a missing submission
        # (the BACK-2405 risk) and a doubled one fail here.
        args = submitted_args(mock_submit)
        assert args["transaction_id"] == converted.id
        assert args["input_token_count"] == EXPECTED_PROMPT_TOKENS
        assert args["output_token_count"] == EXPECTED_COMPLETION_TOKENS
        assert args["total_token_count"] == EXPECTED_TOTAL_TOKENS
        assert args["cache_read_token_count"] == EXPECTED_CACHE_READ_TOKENS
        assert args["cache_creation_token_count"] == EXPECTED_CACHE_CREATION_TOKENS
        assert args["middleware_source"] == "PROXY"
        assert args["model"] == ANTHROPIC_MODEL
        assert args["provider"] == "LITELLM"
        assert args["model_source"] == "LITELLM"

    def test_transaction_id_is_the_litellm_id_not_the_anthropic_message_id(
        self, mock_submit, _get_client, _run
    ):
        """The metered id is synthesized by LiteLLM, not carried over from Anthropic.

        ``transform_parsed_response`` populates usage, model and choices on the
        ``litellm.ModelResponse()`` it is handed but never assigns ``.id``, so the
        response keeps LiteLLM's generated ``chatcmpl-<uuid>``. The submission is
        therefore well-formed and unique per call -- metering works -- but the
        transaction id cannot be joined to the upstream Anthropic ``msg_`` id.
        Anything that needs that correlation has to carry it separately.
        """
        converted = converted_anthropic_response()

        run_hook(mw.proxy_handler_instance.async_log_success_event(
            base_kwargs(model=ANTHROPIC_MODEL), converted, NOW, NOW
        ))

        args = submitted_args(mock_submit)
        assert args["transaction_id"]
        assert args["transaction_id"] != ANTHROPIC_MESSAGES_BODY["id"]
        assert args["transaction_id"].startswith("chatcmpl-")

    def test_openai_shaped_response_produces_exactly_one_submission(
        self, mock_submit, _get_client, _run
    ):
        """Positive control: the route we already had a live observation for."""
        response = openai_shaped_response()

        run_hook(mw.proxy_handler_instance.async_log_success_event(
            base_kwargs(), response, NOW, NOW
        ))

        args = submitted_args(mock_submit)
        assert args["transaction_id"] == "chatcmpl-back-2405-openai-control"
        assert args["input_token_count"] == EXPECTED_PROMPT_TOKENS
        assert args["output_token_count"] == EXPECTED_COMPLETION_TOKENS
        assert args["total_token_count"] == EXPECTED_TOTAL_TOKENS
        assert args["middleware_source"] == "PROXY"


@patch.object(mw, "run_async_in_thread", side_effect=run_inline)
@patch.object(mw, "get_client", return_value=object())
@patch.object(mw, "submit_ai_event")
class TestUnconvertedAnthropicPayloadIsNotMetered:
    """Negative controls: what a *missing* conversion would actually cost.

    These are the teeth. If LiteLLM ever stopped normalising the Anthropic shape
    before the callback loop, our handler would be handed one of these two
    payloads -- and neither yields a usable metering record. That is what makes
    the positive tests above meaningful rather than shape-agnostic.
    """

    def test_raw_anthropic_body_has_no_id_to_meter_on(self, mock_submit, _get_client, _run):
        # A plain /v1/messages body is a dict: response_obj["usage"] resolves,
        # but `response.id` does not exist on a dict, so the handler raises
        # before it can submit. LiteLLM's callback loop catches that, logs
        # "LiteLLM.LoggingError: [Non-Blocking] ..." and calls
        # _handle_callback_failure, which increments its callback-failure
        # counter -- so the divergence would be visible upstream even though
        # nothing reaches Revenium.
        with pytest.raises(AttributeError):
            run_hook(mw.proxy_handler_instance.async_log_success_event(
                base_kwargs(model=ANTHROPIC_MODEL), dict(ANTHROPIC_MESSAGES_BODY), NOW, NOW
            ))

        assert mock_submit.call_count == 0

    def test_anthropic_native_usage_zeroes_every_base_token_count(
        self, mock_submit, _get_client, _run
    ):
        # The subtler failure: a response that does carry an `.id` but reports
        # Anthropic-native usage (input_tokens/output_tokens, no
        # usage.prompt_tokens). A record is produced, so nothing errors -- but
        # every billable token count is zero, which is a silently unpriced call.
        # Only the cache fields survive, because extract_cache_tokens already
        # understands the Anthropic spelling.
        unconverted = SubscriptableResponse(
            ANTHROPIC_MESSAGES_BODY["id"], dict(ANTHROPIC_MESSAGES_BODY["usage"])
        )

        run_hook(mw.proxy_handler_instance.async_log_success_event(
            base_kwargs(model=ANTHROPIC_MODEL), unconverted, NOW, NOW
        ))

        args = submitted_args(mock_submit)
        assert args["input_token_count"] == 0
        assert args["output_token_count"] == 0
        assert args["total_token_count"] == 0
        assert args["cache_read_token_count"] == EXPECTED_CACHE_READ_TOKENS
        assert args["cache_creation_token_count"] == EXPECTED_CACHE_CREATION_TOKENS


@pytest.fixture
def registered_proxy_handler():
    """Register MiddlewareHandler with LiteLLM's async success callbacks, then remove it.

    The async dispatch loop reads ``litellm._async_success_callback``, which is
    what ``litellm.callbacks`` is propagated into during a real call's setup, so
    registration goes through the documented
    ``logging_callback_manager.add_litellm_async_success_callback``. Removal is
    unconditional: leaving our handler in a module-global callback list would
    silently meter every other litellm test in the session.
    """
    handler = mw.proxy_handler_instance
    litellm.logging_callback_manager.add_litellm_async_success_callback(handler)
    try:
        yield handler
    finally:
        litellm.logging_callback_manager.remove_callback_from_all_lists(handler)


def anthropic_route_logging_obj(headers=None):
    """A ``Logging`` object configured as the /v1/messages route configures one."""
    logging_obj = Logging(
        model=ANTHROPIC_MODEL,
        messages=[{"role": "user", "content": "meter me"}],
        stream=False,
        call_type=CallTypes.anthropic_messages.value,
        start_time=NOW,
        litellm_call_id="back-2405-dispatch-call",
        function_id="back-2405-dispatch-fn",
    )
    logging_obj.update_environment_variables(
        model=ANTHROPIC_MODEL,
        user="",
        optional_params={},
        litellm_params={"metadata": {"headers": headers or {}}},
    )
    return logging_obj


@patch.object(mw, "get_client", return_value=object())
@patch.object(mw, "submit_ai_event")
class TestLiteLLMActuallyDispatchesToUsOnTheAnthropicRoute:
    """The dispatch precondition, exercised rather than asserted.

    The tests above drive ``async_log_success_event`` directly, which proves the
    payload mapping but takes LiteLLM's dispatch on trust. These drive
    ``Logging.async_success_handler`` -- the real entry point, with the real
    ``anthropic_messages`` call type, a raw ``/v1/messages`` body and
    ``MiddlewareHandler`` registered as a callback -- so a vendor change that
    stopped dispatching to us on that route fails here even though route
    mapping and response conversion still worked.

    The dispatch loop swallows callback exceptions (it logs
    "LiteLLM.LoggingError: [Non-Blocking]" and increments a failure counter), so
    a handler that blew up would show up here as zero submissions, which these
    tests reject.
    """

    @staticmethod
    async def _dispatch(logging_obj, body, mock_submit):
        """Run the vendor dispatch, then drain the metering coroutines it queued.

        ``run_async_in_thread`` is replaced by a collector rather than by
        ``run_inline``: we are already inside a running event loop here, and
        ``run_inline`` calls ``asyncio.run()``. Awaiting the collected
        coroutines afterwards keeps the assertion deterministic -- no sleeps, no
        dependence on background-thread scheduling.
        """
        queued = []
        with patch.object(mw, "run_async_in_thread", side_effect=queued.append):
            await logging_obj.async_success_handler(
                result=dict(body), start_time=NOW, end_time=NOW, cache_hit=False
            )
        for coro in queued:
            await coro
        return submitted_args(mock_submit)

    @pytest.mark.asyncio
    async def test_raw_anthropic_body_through_the_dispatch_loop_is_metered_once(
        self, mock_submit, _get_client, registered_proxy_handler
    ):
        logging_obj = anthropic_route_logging_obj(
            headers={"x-revenium-trace-id": "back-2405-anthropic-route"}
        )

        args = await self._dispatch(logging_obj, ANTHROPIC_MESSAGES_BODY, mock_submit)

        assert args["input_token_count"] == EXPECTED_PROMPT_TOKENS
        assert args["output_token_count"] == EXPECTED_COMPLETION_TOKENS
        assert args["total_token_count"] == EXPECTED_TOTAL_TOKENS
        assert args["cache_read_token_count"] == EXPECTED_CACHE_READ_TOKENS
        assert args["cache_creation_token_count"] == EXPECTED_CACHE_CREATION_TOKENS
        assert args["middleware_source"] == "PROXY"
        assert args["trace_id"] == "back-2405-anthropic-route"
        # LiteLLM converted the body before handing it to us, so the metered id
        # is its own -- never the Anthropic msg_ id the client received.
        assert args["transaction_id"].startswith("chatcmpl-")
        assert args["transaction_id"] != ANTHROPIC_MESSAGES_BODY["id"]

    @pytest.mark.asyncio
    async def test_openai_route_through_the_dispatch_loop_is_metered_once(
        self, mock_submit, _get_client, registered_proxy_handler
    ):
        """Positive control on the same dispatch path, with the OpenAI call type."""
        logging_obj = Logging(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "meter me"}],
            stream=False,
            call_type=CallTypes.acompletion.value,
            start_time=NOW,
            litellm_call_id="back-2405-dispatch-openai",
            function_id="back-2405-dispatch-openai-fn",
        )
        logging_obj.update_environment_variables(
            model="gpt-4o-mini", user="", optional_params={},
            litellm_params={"metadata": {"headers": {"x-revenium-trace-id": "back-2405-openai-route"}}},
        )

        queued = []
        with patch.object(mw, "run_async_in_thread", side_effect=queued.append):
            await logging_obj.async_success_handler(
                result=openai_shaped_response(), start_time=NOW, end_time=NOW, cache_hit=False
            )
        for coro in queued:
            await coro

        args = submitted_args(mock_submit)
        assert args["transaction_id"] == "chatcmpl-back-2405-openai-control"
        assert args["input_token_count"] == EXPECTED_PROMPT_TOKENS
        assert args["output_token_count"] == EXPECTED_COMPLETION_TOKENS
        assert args["middleware_source"] == "PROXY"
        assert args["trace_id"] == "back-2405-openai-route"
