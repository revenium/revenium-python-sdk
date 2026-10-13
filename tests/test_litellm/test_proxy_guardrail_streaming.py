"""A streamed proxied call is metered once, with its assembled totals (FRONT-2409).

A stream is where proxy metering most easily disappears: the response leaves as
chunks, the usage totals only exist once the stream ends, and the hook that
normally meters -- ``async_post_call_success_hook`` -- takes a single assembled
response. LiteLLM closes that gap for ``/v1/chat/completions``, but **not** for
the Anthropic-shaped ``/v1/messages`` route, and which hook runs is decided by
``ProxyBaseLLMRequestProcessing._arm_deferred_stream_dispatch`` sniffing the
response object (verified against litellm 1.101.0):

1. A ``CustomStreamWrapper`` (``/v1/chat/completions``) arms a closure that runs
   ``_run_deferred_stream_guardrails``, which calls
   ``async_post_call_success_hook`` on every ``CustomGuardrail`` in
   ``litellm.callbacks`` -- with the **assembled** response and its full totals.
2. That call is skipped for any guardrail whose class defines
   ``async_post_call_streaming_iterator_hook`` in its own ``__dict__``: the
   vendor assumes such a guardrail already inspected the stream itself and
   refuses to scan twice. Defining an iterator hook here would therefore *cost*
   us the assembled usage rather than gain us anything, and
   ``test_defines_no_streaming_iterator_hook`` keeps a future contributor from
   adding one without reading this.
3. A native ``anthropic_messages`` stream iterator arms a different closure,
   ``_on_deferred_native_stream_complete``, which enqueues a ready-made logging
   coroutine and never reaches ``_run_deferred_stream_guardrails``. On that
   route the post-call success hook never fires at all, so the guardrail meters
   from ``async_log_success_event`` instead -- the ``CustomLogger`` surface the
   logging dispatch does reach. That is the route Claude Code uses, and it was
   silently unmetered until the guardrail implemented that method.

Because both hooks fire for one call on the routes where LiteLLM arms the
guardrail closure, and could both fire on the Anthropic route if a provider ever
returned a ``CustomStreamWrapper`` there, the guardrail also remembers the calls
it has already metered. ``TestOneRowPerCallAcrossBothHooks`` pins that.

``TestStreamedCallIsMeteredOnce`` and ``TestNativeAnthropicStreamArmsTheLoggingClosure``
drive the real vendor entry points rather than calling our hooks directly, so a
LiteLLM upgrade that changed either dispatch fails here -- which is the failure
mode that otherwise shows up as silently missing revenue.
"""

import copy
import datetime
import uuid

import pytest

pytest.importorskip("litellm")
pytest.importorskip("fastapi")

from unittest.mock import AsyncMock, MagicMock, patch  # noqa: E402

import litellm  # noqa: E402
from litellm.proxy.common_request_processing import (  # noqa: E402
    ProxyBaseLLMRequestProcessing,
)
from litellm.types.utils import CallTypes  # noqa: E402

from revenium_middleware.litellm.proxy import guardrail as guardrail_module  # noqa: E402
from revenium_middleware.litellm.proxy.guardrail import ReveniumGuardrail  # noqa: E402

from .proxy_hook_harness import (  # noqa: E402
    GuardrailResponse,
    drive_metering,
    guardrail_data,
    make_key_dict,
    submitted_args,
)

ANTHROPIC_MODEL = "claude-sonnet-4-5-20250929"

# The start and end times LiteLLM passes a logging event, a quarter of a second
# apart, which is the request duration the row should report when the hidden
# params carry none.
NOW = datetime.datetime(2026, 9, 15, 12, 0, 0, tzinfo=datetime.timezone.utc)
LATER = NOW + datetime.timedelta(milliseconds=250)

# A raw /v1/messages body, as the Anthropic route hands the post-call hook when
# the call is not streamed. Anthropic reports cache tokens *alongside*
# input_tokens rather than inside it, and Revenium prices those two buckets in
# their own fields, so the metered input count is input_tokens as it stands --
# the count LiteLLM's conversion folds all three into is the one that billed
# the cache twice (BACK-3334). The total still counts every bucket once.
ANTHROPIC_NON_STREAMED_BODY = {
    "id": "msg_01NonStreamedAnthropicRoute",
    "type": "message",
    "role": "assistant",
    "model": ANTHROPIC_MODEL,
    "content": [{"type": "text", "text": "Metered from a raw Anthropic body."}],
    "stop_reason": "end_turn",
    "usage": {
        "input_tokens": 10,
        "output_tokens": 2,
        "cache_read_input_tokens": 4,
        "cache_creation_input_tokens": 3,
    },
}

ANTHROPIC_EXPECTED_INPUT_TOKENS = 10
ANTHROPIC_EXPECTED_COMPLETION_TOKENS = 2
ANTHROPIC_EXPECTED_TOTAL_TOKENS = 19


class AssembledUsage:
    """The usage totals LiteLLM computes once a stream has finished."""

    prompt_tokens = 21
    completion_tokens = 34
    total_tokens = 55


def logging_kwargs(
    call_type,
    stream,
    headers=None,
    model=ANTHROPIC_MODEL,
    litellm_call_id="litellm-call-id-1",
):
    """The kwargs LiteLLM's logging dispatch hands a ``CustomLogger``.

    This is ``Logging.model_call_details``, which is a different shape from the
    dict a guardrail hook receives: metadata sits under ``litellm_params``, and
    the call type, the stream flag and LiteLLM's correlation id are top-level.
    """
    return {
        "model": model,
        "call_type": call_type,
        "stream": stream,
        "litellm_call_id": litellm_call_id,
        "litellm_params": {
            "metadata": {
                "headers": headers or {},
                "hidden_params": {
                    "optional_params": {"stream": stream},
                    "litellm_overhead_time_ms": 10,
                    "_response_ms": 250.0,
                },
            }
        },
    }


CHAT_COMPLETIONS_ROUTE = "/v1/chat/completions"


async def run_deferred_stream(captured_data, assembled_response):
    """Play the proxy finishing a ``/v1/chat/completions`` stream.

    This is the vendor's own closure body, so the guardrail is reached only if
    LiteLLM still dispatches the assembled response to it.
    """
    await ProxyBaseLLMRequestProcessing._run_deferred_stream_guardrails(
        captured_data=captured_data,
        captured_user_api_key_dict=make_key_dict(request_route=CHAT_COMPLETIONS_ROUTE),
        captured_logging_obj=success_dispatcher(),
        assembled_response=assembled_response,
        cache_hit=False,
    )


def success_dispatcher():
    """The logging object the vendor hands success logging to once the hooks ran.

    Its ``dispatch_success_handlers`` must return a coroutine, because the
    vendor schedules it with ``asyncio.create_task``.
    """
    logging_obj = MagicMock()
    logging_obj.dispatch_success_handlers = AsyncMock()
    return logging_obj


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


@pytest.fixture
def registered(guardrail):
    """Register the guardrail the way a configured proxy would."""
    original = list(litellm.callbacks)
    litellm.callbacks = original + [guardrail]
    yield guardrail
    litellm.callbacks = original


def test_defines_no_streaming_iterator_hook():
    """Defining one would make LiteLLM skip the hook that meters the stream.

    See this module's docstring: the vendor skips
    ``async_post_call_success_hook`` for any guardrail carrying an iterator
    hook, and that success hook is where the assembled usage totals arrive on
    the chat-completions route.
    """
    assert "async_post_call_streaming_iterator_hook" not in ReveniumGuardrail.__dict__


def test_defines_no_failure_logging_event():
    """Failures are metered once, from the post-call failure hook.

    ``async_log_failure_event`` fires once per failed *attempt*, including
    attempts a retry goes on to recover from, so metering there would invent
    rows for calls that succeeded. ``async_post_call_failure_hook`` fires once
    per failed request, which is the row we want.
    """
    assert "async_log_failure_event" not in ReveniumGuardrail.__dict__


def test_the_metered_call_type_is_still_the_vendor_spelling():
    """The route gate is a literal, so a vendor rename fails here, not in prod.

    ``async_log_success_event`` fires on every call and meters only the streamed
    Anthropic route. Importing ``CallTypes`` in the guardrail would make a
    vendor rename break proxy startup; a literal makes it silently stop
    metering. This test is the third option.
    """
    assert guardrail_module._ANTHROPIC_MESSAGES_CALL_TYPE == CallTypes.anthropic_messages.value


class TestStreamedCallIsMeteredOnce:
    """Driven through LiteLLM's own deferred-stream entry point."""

    @pytest.mark.asyncio
    async def test_assembled_stream_is_metered_with_its_totals(self, registered, submit):
        data = guardrail_data(stream=True, headers={"x-revenium-trace-id": "trace-s"})
        response = GuardrailResponse(
            response_id="chatcmpl-stream", usage=AssembledUsage()
        )
        await run_deferred_stream(
            captured_data=data,
            assembled_response=response,
        )
        args = submitted_args(submit)
        assert args["input_token_count"] == 21
        assert args["output_token_count"] == 34
        assert args["total_token_count"] == 55
        assert args["transaction_id"] == "chatcmpl-stream"
        assert args["trace_id"] == "trace-s"
        assert args["middleware_source"] == "GUARDRAIL"

    @pytest.mark.asyncio
    async def test_streamed_rows_are_flagged_as_streamed(self, registered, submit):
        """``is_streamed`` is what separates a stream from a plain completion."""
        data = guardrail_data(stream=True)
        await run_deferred_stream(
            captured_data=data,
            assembled_response=GuardrailResponse(usage=AssembledUsage()),
        )
        assert submitted_args(submit)["is_streamed"] is True

    @pytest.mark.asyncio
    async def test_a_non_streamed_call_is_not_flagged(self, guardrail, submit):
        await guardrail.async_post_call_success_hook(
            data=guardrail_data(stream=False),
            user_api_key_dict=make_key_dict(),
            response=GuardrailResponse(usage=AssembledUsage()),
        )
        assert submitted_args(submit)["is_streamed"] is False

    @pytest.mark.asyncio
    async def test_a_failing_stream_does_not_break_the_vendor_pipeline(
        self, registered, submit
    ):
        """The deferred path runs after the client already has the content.

        An exception escaping our hook here cannot un-send the response, but it
        does abort the vendor's loop over the remaining guardrails -- so it must
        not escape.
        """
        submit.side_effect = Exception("metering API down")
        await run_deferred_stream(
            captured_data=guardrail_data(stream=True),
            assembled_response=GuardrailResponse(usage=AssembledUsage()),
        )

        assert submit.call_count == 1


@pytest.mark.skipif(
    not hasattr(ProxyBaseLLMRequestProcessing, "_arm_deferred_stream_dispatch"),
    reason=(
        "ProxyBaseLLMRequestProcessing._arm_deferred_stream_dispatch arrived in "
        "litellm 1.101.0; on an older vendor the deferred native-stream path "
        "under test does not exist, so there is nothing to pin here"
    ),
)
class TestNativeAnthropicStreamArmsTheLoggingClosure:
    """The vendor fact that makes ``async_log_success_event`` necessary.

    This is the test that did not exist when the gap shipped. The streaming
    suite drove ``_run_deferred_stream_guardrails`` directly and so never
    exercised the arming step that decides whether that function runs at all.
    """

    @staticmethod
    def _arm(response, route_type):
        processing = ProxyBaseLLMRequestProcessing(data={})
        logging_obj = MagicMock()
        logging_obj._on_deferred_stream_complete = None
        processing._arm_deferred_stream_dispatch(
            response=response,
            route_type=route_type,
            user_api_key_dict=make_key_dict(),
            logging_obj=logging_obj,
        )
        return logging_obj._on_deferred_stream_complete

    @staticmethod
    async def _native_stream():
        yield b"event: message_stop\n\n"

    def test_a_native_anthropic_stream_never_reaches_the_guardrail_closure(self):
        armed = self._arm(self._native_stream(), "anthropic_messages")

        assert armed is not None
        assert armed.__name__ == "_on_deferred_native_stream_complete"

    def test_a_chat_completions_stream_does_reach_the_guardrail_closure(self):
        """The control: the route where the post-call success hook still meters."""
        from litellm.litellm_core_utils.streaming_handler import CustomStreamWrapper

        wrapper = CustomStreamWrapper(
            completion_stream=iter([]),
            model="gpt-4o-mini",
            logging_obj=MagicMock(),
        )
        armed = self._arm(wrapper, "acompletion")

        assert armed is not None
        assert armed.__name__ == "_on_deferred_stream_complete"


class TestStreamedAnthropicRouteIsMetered:
    """The route Claude Code uses, metered from the logging event."""

    @pytest.mark.asyncio
    async def test_streamed_anthropic_call_is_metered_with_its_assembled_totals(
        self, guardrail, submit
    ):
        kwargs = logging_kwargs(
            "anthropic_messages", stream=True, headers={"x-revenium-trace-id": "trace-a"}
        )
        response = GuardrailResponse(
            response_id="msg_01AnthropicStream", usage=AssembledUsage()
        )

        await guardrail.async_log_success_event(kwargs, response, NOW, LATER)

        args = submitted_args(submit)
        assert args["input_token_count"] == 21
        assert args["output_token_count"] == 34
        assert args["total_token_count"] == 55
        assert args["is_streamed"] is True
        assert args["stop_reason"] == "END"
        assert args["trace_id"] == "trace-a"
        assert args["middleware_source"] == "GUARDRAIL"
        assert args["model"] == ANTHROPIC_MODEL

    @pytest.mark.asyncio
    async def test_the_metered_id_is_neither_the_anthropic_message_id_nor_the_callers_call_id(
        self, monkeypatch, submit
    ):
        """Both rows on this route carry the request's own id.

        The non-streamed row on ``/v1/messages`` has no ``.id`` to read off its
        raw Anthropic body and falls through to the request's own id. Metering
        the streamed row on the assembled response's ``.id`` would give the two
        rows different identity shapes for the same route, and metering it on
        LiteLLM's call id would let a caller who repeats ``x-litellm-call-id``
        merge every call into one record.
        """
        monkeypatch.setenv(FLAG, "false")
        instance = ReveniumGuardrail(
            guardrail_name="revenium", event_hook=["pre_call", "post_call"], default_on=True
        )
        data = guardrail_data(
            stream=True, model=ANTHROPIC_MODEL, metadata_key="litellm_metadata"
        )
        stamped = await run_pre_call(instance, data, call_type="anthropic_messages")
        kwargs = logging_kwargs_from(data, litellm_call_id="litellm-anthropic-stream")
        response = GuardrailResponse(
            response_id="msg_01AnthropicStream", usage=AssembledUsage()
        )

        await instance.async_log_success_event(kwargs, response, NOW, LATER)

        transaction_id = submitted_args(submit)["transaction_id"]
        assert transaction_id == request_id_of(stamped)
        assert transaction_id != "msg_01AnthropicStream"
        assert transaction_id != "litellm-anthropic-stream"
        uuid.UUID(transaction_id)

    @pytest.mark.asyncio
    async def test_metering_failure_never_escapes_the_logging_event(
        self, guardrail, submit
    ):
        """LiteLLM logs and counts a raising callback; it must not raise."""
        submit.side_effect = Exception("metering API down")

        await guardrail.async_log_success_event(
            logging_kwargs("anthropic_messages", stream=True),
            GuardrailResponse(usage=AssembledUsage()),
            NOW,
            LATER,
        )


RESPONSES_MODEL = "gpt-4o-mini"


class TestStreamedResponsesCallIsMetered:
    """``/v1/responses`` with ``stream: true``, which Codex CLI speaks.

    On a model with a native Responses API the stream reaches no other hook of
    this guardrail and LiteLLM's client entry points are not called, so the
    logging event is the only meter there is.
    """

    @pytest.mark.asyncio
    async def test_a_streamed_responses_call_is_metered_from_the_logging_event(
        self, guardrail, submit
    ):
        kwargs = logging_kwargs("aresponses", stream=True, model=RESPONSES_MODEL)

        await guardrail.async_log_success_event(
            kwargs, GuardrailResponse(usage=AssembledUsage()), NOW, LATER
        )

        args = submitted_args(submit)
        assert args["input_token_count"] == AssembledUsage.prompt_tokens
        assert args["output_token_count"] == AssembledUsage.completion_tokens
        assert args["is_streamed"] is True
        assert args["middleware_source"] == "GUARDRAIL"
        assert args["model"] == RESPONSES_MODEL

    @pytest.mark.asyncio
    async def test_a_guardrail_that_does_not_own_metering_leaves_it_to_the_client_wrapper(
        self, submit
    ):
        """Without ownership the client wrapper meters a bridged stream, so a row here would be its second."""
        not_owning = ReveniumGuardrail(
            guardrail_name="revenium", event_hook=["pre_call", "post_call"], default_on=False
        )

        await not_owning.async_log_success_event(
            logging_kwargs("aresponses", stream=True),
            GuardrailResponse(usage=AssembledUsage()),
            NOW,
            LATER,
        )

        assert submit.call_count == 0

    @pytest.mark.asyncio
    async def test_a_key_opted_out_of_the_guardrail_is_left_to_the_client_wrapper(
        self, guardrail, submit
    ):
        kwargs = logging_kwargs("aresponses", stream=True)
        kwargs["litellm_params"]["metadata"]["user_api_key_metadata"] = {
            "opted_out_global_guardrails": ["revenium"]
        }

        await guardrail.async_log_success_event(
            kwargs, GuardrailResponse(usage=AssembledUsage()), NOW, LATER
        )

        assert submit.call_count == 0


class TestTheLoggingEventMetersNothingElse:
    """The route gate, which is load-bearing: this hook fires on every call."""

    @pytest.mark.asyncio
    async def test_a_non_streamed_responses_call_is_left_to_the_post_call_hook(
        self, guardrail, submit
    ):
        await guardrail.async_log_success_event(
            logging_kwargs("aresponses", stream=False),
            GuardrailResponse(usage=AssembledUsage()),
            NOW,
            LATER,
        )
        assert submit.call_count == 0

    @pytest.mark.asyncio
    async def test_a_non_streamed_anthropic_call_is_left_to_the_post_call_hook(
        self, guardrail, submit
    ):
        await guardrail.async_log_success_event(
            logging_kwargs("anthropic_messages", stream=False),
            GuardrailResponse(usage=AssembledUsage()),
            NOW,
            LATER,
        )
        assert submit.call_count == 0

    @pytest.mark.asyncio
    async def test_a_streamed_chat_completion_is_left_to_the_post_call_hook(
        self, guardrail, submit
    ):
        await guardrail.async_log_success_event(
            logging_kwargs("acompletion", stream=True),
            GuardrailResponse(usage=AssembledUsage()),
            NOW,
            LATER,
        )
        assert submit.call_count == 0


class TestOneRowPerCallAcrossBothHooks:
    """Whichever hook meters a call first, the other must not meter it again."""

    @pytest.fixture(autouse=True)
    def _shared_call_id_off(self, monkeypatch):
        """These rows carry the request's own id; the minted id has its own class below."""
        monkeypatch.setenv(FLAG, "false")

    @staticmethod
    async def _one_call(instance, litellm_call_id="litellm-shared-call-id", call_type="anthropic_messages"):
        """The two shapes one completed call reaches our two hooks in, after its pre-call."""
        data = guardrail_data(stream=True, model=ANTHROPIC_MODEL)
        data["litellm_call_id"] = litellm_call_id
        await instance.async_pre_call_hook(
            user_api_key_dict=make_key_dict(), cache=MagicMock(), data=data, call_type=call_type
        )
        kwargs = logging_kwargs_from(data, litellm_call_id=litellm_call_id, call_type=call_type)
        return kwargs, data

    @pytest.mark.asyncio
    async def test_the_post_call_hook_does_not_re_meter_a_logged_stream(
        self, guardrail, submit
    ):
        kwargs, data = await self._one_call(guardrail)
        response = GuardrailResponse(usage=AssembledUsage())

        await guardrail.async_log_success_event(kwargs, response, NOW, LATER)
        await guardrail.async_post_call_success_hook(
            data=data, user_api_key_dict=make_key_dict(), response=response
        )

        assert submitted_args(submit)["transaction_id"] != "litellm-shared-call-id"

    @pytest.mark.asyncio
    async def test_the_logging_event_does_not_re_meter_a_post_called_stream(
        self, guardrail, submit
    ):
        """The order a ``CustomStreamWrapper`` on the Anthropic route would take.

        ``_run_deferred_stream_guardrails`` runs the success hook first and then
        dispatches success logging, so this is the order that matters if a
        provider's ``anthropic_messages`` stream ever arrives as a wrapper.
        """
        kwargs, data = await self._one_call(guardrail)
        response = GuardrailResponse(usage=AssembledUsage())

        await guardrail.async_post_call_success_hook(
            data=data, user_api_key_dict=make_key_dict(), response=response
        )
        await guardrail.async_log_success_event(kwargs, response, NOW, LATER)

        assert submit.call_count == 1

    @pytest.mark.asyncio
    async def test_two_different_calls_still_produce_two_rows(self, guardrail, submit):
        """The guard remembers calls, not routes: distinct calls are distinct rows."""
        for call_id in ("litellm-call-a", "litellm-call-b"):
            kwargs, _ = await self._one_call(guardrail, litellm_call_id=call_id)
            await guardrail.async_log_success_event(
                kwargs, GuardrailResponse(usage=AssembledUsage()), NOW, LATER
            )

        assert submit.call_count == 2

    @pytest.mark.asyncio
    async def test_a_streamed_chat_completion_still_produces_exactly_one_row(
        self, registered, submit
    ):
        """The route the guard must not disturb: one row, from the post-call hook."""
        kwargs, data = await self._one_call(registered, call_type="acompletion")
        response = GuardrailResponse(response_id="chatcmpl-stream", usage=AssembledUsage())

        await run_deferred_stream(
            captured_data=data,
            assembled_response=response,
        )
        await registered.async_log_success_event(kwargs, response, NOW, LATER)

        assert submitted_args(submit)["transaction_id"] == "chatcmpl-stream"

    @pytest.mark.asyncio
    async def test_a_failed_call_is_still_metered_after_a_success_row(
        self, guardrail, submit
    ):
        """The failure path does not consult the guard, and must not.

        A retried request logs a failure for the attempt that failed and a
        success for the attempt that worked. Both rows are wanted, so the guard
        covers the success paths only.
        """
        kwargs, data = await self._one_call(guardrail)

        await guardrail.async_log_success_event(
            kwargs, GuardrailResponse(usage=AssembledUsage()), NOW, LATER
        )
        await guardrail.async_post_call_failure_hook(
            request_data=data,
            original_exception=Exception("upstream overloaded"),
            user_api_key_dict=make_key_dict(),
        )

        assert submit.call_count == 2
        assert submit.call_args_list[1][0][1]["stop_reason"] == "ERROR"


class TestNonStreamedAnthropicRowCarriesRealTokens:
    """The same route, not streamed: a raw Anthropic body must still be priced.

    The post-call hook is handed the provider's own JSON here, not a LiteLLM
    ``ModelResponse``, so its usage is spelled ``input_tokens`` /
    ``output_tokens``. Reading only the OpenAI spelling zeroed every billable
    count on this route while the row itself looked perfectly healthy.
    """

    @pytest.mark.asyncio
    async def test_anthropic_usage_spelling_is_read(self, guardrail, submit):
        await guardrail.async_post_call_success_hook(
            data=guardrail_data(
                stream=False, model=ANTHROPIC_MODEL,
                custom_llm_provider="anthropic", call_type="anthropic_messages",
            ),
            user_api_key_dict=make_key_dict(),
            response=dict(ANTHROPIC_NON_STREAMED_BODY),
        )

        args = submitted_args(submit)
        assert args["input_token_count"] == ANTHROPIC_EXPECTED_INPUT_TOKENS
        assert args["output_token_count"] == ANTHROPIC_EXPECTED_COMPLETION_TOKENS
        assert args["total_token_count"] == ANTHROPIC_EXPECTED_TOTAL_TOKENS
        assert args["cache_read_token_count"] == 4
        assert args["cache_creation_token_count"] == 3
        assert args["is_streamed"] is False

    @pytest.mark.asyncio
    async def test_the_openai_spelling_still_wins_when_both_are_present(
        self, guardrail, submit
    ):
        """LiteLLM's own conversion already folds the cache buckets into
        ``prompt_tokens``, so on this route -- where the upstream is Anthropic,
        the one provider the platform does not net out itself -- the metered
        input is that count with them taken back out, which is the same number
        the Anthropic spelling reports directly."""
        usage = dict(ANTHROPIC_NON_STREAMED_BODY["usage"])
        usage.update(prompt_tokens=17, completion_tokens=2, total_tokens=19)
        body = dict(ANTHROPIC_NON_STREAMED_BODY, usage=usage)

        await guardrail.async_post_call_success_hook(
            data=guardrail_data(
                stream=False, model=ANTHROPIC_MODEL,
                custom_llm_provider="anthropic", call_type="anthropic_messages",
            ),
            user_api_key_dict=make_key_dict(),
            response=body,
        )

        args = submitted_args(submit)
        assert args["input_token_count"] == ANTHROPIC_EXPECTED_INPUT_TOKENS
        assert args["output_token_count"] == ANTHROPIC_EXPECTED_COMPLETION_TOKENS
        assert args["total_token_count"] == ANTHROPIC_EXPECTED_TOTAL_TOKENS


# --- The streamed row meets the minted shared call id ----------------------
#
# BACK-3199 added the streamed Anthropic row on PR #113, below BACK-3190 (#114)
# and BACK-2399 (#115), so it could not call either of their helpers and shipped
# a self-contained header read and a flag-off-only identity. With the stack
# re-linked, both are available and the streamed row has to use them, or the one
# route Claude Code speaks is the one route the feature does not reach:
#
# * the identity has to be the id this proxy minted for the request and returned
#   to the client as ``request-id``, because that is the value Claude Code
#   copies onto its own usage record. A row on LiteLLM's call id instead cannot
#   collide with the telemetry row, which is the whole double-count the feature
#   exists to close;
# * the headers have to come from ``extract_request_headers``, because on
#   ``/v1/messages`` LiteLLM fills ``litellm_metadata`` and leaves ``metadata``
#   as the caller's own body, so reading ``metadata["headers"]`` drops every
#   documented ``x-revenium-*`` header on exactly this route;
# * the duplicate guard has to be keyed on the id that lands on the row, or the
#   two success paths key on different values for one call and both meter it.

FLAG = "REVENIUM_LITELLM_SHARED_CALL_ID"


def logging_kwargs_from(
    data,
    litellm_call_id="litellm-call-id-1",
    stream=True,
    call_type="anthropic_messages",
    model=ANTHROPIC_MODEL,
):
    """The logging kwargs for the request ``data`` describes.

    LiteLLM hands ``async_log_success_event`` the same metadata containers the
    guardrail hooks see, one level deeper under ``litellm_params``. Copying them
    across rather than rebuilding them keeps a minted id, and whichever key
    LiteLLM filled with the inbound headers, exactly where the request put them.
    """
    params = {}
    for key in ("litellm_metadata", "proxy_server_request", "metadata"):
        if key in data:
            params[key] = data[key]
    params.setdefault("metadata", {})
    return {
        "model": model,
        "call_type": call_type,
        "stream": stream,
        "litellm_call_id": litellm_call_id,
        "litellm_params": params,
    }


class TestTheStreamedRowTakesTheMintedId:
    """The reconciliation seam, where BACK-3199's row meets BACK-2399's mint."""

    @pytest.fixture(autouse=True)
    def _clean_process_state(self, monkeypatch):
        """Nothing here may inherit another test's process-wide state.

        The flag is on by default, so it starts opted out and each test that
        needs the shared id sets it.
        """
        from revenium_middleware.litellm.proxy import _metering_owner

        monkeypatch.setenv(FLAG, "false")
        _metering_owner.reset_metering_owner()
        guardrail_module._shared_call_id_missing_count = 0
        guardrail_module._shared_call_id_last_warned_at = None
        yield
        _metering_owner.reset_metering_owner()
        guardrail_module._shared_call_id_missing_count = 0
        guardrail_module._shared_call_id_last_warned_at = None

    @staticmethod
    def _new_guardrail():
        """A guardrail built the way a configured proxy builds one.

        ``mode`` has to carry ``pre_call``: LiteLLM gates the pre-call hook on
        it, and the shared call id is inactive without it.
        """
        return ReveniumGuardrail(
            guardrail_name="revenium",
            event_hook=["pre_call", "post_call"],
            default_on=True,
        )

    @staticmethod
    async def _mint(instance, data):
        """Run the pre-call hook, which is what stamps the minted id."""
        await instance.async_pre_call_hook(
            user_api_key_dict=make_key_dict(),
            cache=MagicMock(),
            data=data,
            call_type="anthropic_messages",
        )

    @staticmethod
    async def _returned_request_id(instance, data):
        """The ``request-id`` the client was handed for this request."""
        returned = await instance.async_post_call_response_headers_hook(
            data=data,
            user_api_key_dict=make_key_dict(),
            response=GuardrailResponse(usage=AssembledUsage()),
        )
        assert returned is not None, "no shared call id was minted for this request"
        return returned["request-id"]

    @pytest.mark.asyncio
    async def test_the_streamed_row_carries_the_id_the_client_was_handed(
        self, monkeypatch, submit
    ):
        """One call, one id: the metered row and the response header agree.

        Claude Code copies ``request-id`` onto its own usage record. A streamed
        row filed under LiteLLM's call id instead can never collide with it, so
        the customer is billed for both records -- which is the defect the flag
        exists to close, on the only route Claude Code uses.
        """
        monkeypatch.setenv(FLAG, "true")
        instance = self._new_guardrail()
        data = guardrail_data(
            stream=True, model=ANTHROPIC_MODEL, metadata_key="litellm_metadata"
        )
        await self._mint(instance, data)
        minted = await self._returned_request_id(instance, data)

        kwargs = logging_kwargs_from(data, litellm_call_id="litellm-not-the-minted-id")
        await instance.async_log_success_event(
            kwargs,
            GuardrailResponse(
                response_id="msg_01AnthropicStream", usage=AssembledUsage()
            ),
            NOW,
            LATER,
        )

        args = submitted_args(submit)
        assert args["transaction_id"] == minted
        assert args["transaction_id"] != "litellm-not-the-minted-id"
        assert args["transaction_id"] != "msg_01AnthropicStream"

    @pytest.mark.asyncio
    async def test_the_streamed_row_takes_a_per_request_id_with_the_flag_off(
        self, submit
    ):
        """With the flag off, the row carries the request's own id, never the caller's."""
        instance = self._new_guardrail()
        data = guardrail_data(
            stream=True, model=ANTHROPIC_MODEL, metadata_key="litellm_metadata"
        )
        stamped = await run_pre_call(instance, data, call_type="anthropic_messages")
        assert "revenium_call_id" not in data["litellm_metadata"]

        kwargs = logging_kwargs_from(data, litellm_call_id="litellm-anthropic-stream")
        await instance.async_log_success_event(
            kwargs,
            GuardrailResponse(
                response_id="msg_01AnthropicStream", usage=AssembledUsage()
            ),
            NOW,
            LATER,
        )

        transaction_id = submitted_args(submit)["transaction_id"]
        assert transaction_id == request_id_of(stamped)
        assert transaction_id != "litellm-anthropic-stream"
        assert transaction_id != "msg_01AnthropicStream"
        uuid.UUID(transaction_id)

    @pytest.mark.asyncio
    async def test_the_streamed_path_reads_the_header_key_this_route_fills(
        self, submit
    ):
        """``litellm_metadata`` is where the headers are on ``/v1/messages``.

        ``metadata`` on this route is the caller's own request body, so it is
        both empty of proxy-filled headers and forgeable. Preferring it would
        drop the customer's trace id and let a caller supply their own.
        """
        instance = self._new_guardrail()
        data = guardrail_data(
            headers={"x-revenium-trace-id": "trace-from-litellm-metadata"},
            stream=True,
            model=ANTHROPIC_MODEL,
            metadata_key="litellm_metadata",
            metadata={"headers": {"x-revenium-trace-id": "planted-by-the-caller"}},
        )

        await instance.async_log_success_event(
            logging_kwargs_from(data),
            GuardrailResponse(usage=AssembledUsage()),
            NOW,
            LATER,
        )

        assert submitted_args(submit)["trace_id"] == "trace-from-litellm-metadata"

    @pytest.mark.asyncio
    async def test_the_streamed_and_non_streamed_rows_read_the_same_header_key(
        self, submit
    ):
        """Parity: the two rows on this route must agree on where headers live."""
        headers = {"x-revenium-trace-id": "trace-on-the-messages-route"}
        streamed_guardrail = self._new_guardrail()
        await streamed_guardrail.async_log_success_event(
            logging_kwargs_from(
                guardrail_data(
                    headers=headers,
                    stream=True,
                    model=ANTHROPIC_MODEL,
                    metadata_key="litellm_metadata",
                ),
                litellm_call_id="streamed-call",
            ),
            GuardrailResponse(usage=AssembledUsage()),
            NOW,
            LATER,
        )
        streamed_trace = submitted_args(submit)["trace_id"]

        submit.reset_mock()
        non_streamed_guardrail = self._new_guardrail()
        non_streamed_data = guardrail_data(
            headers=headers, model=ANTHROPIC_MODEL, metadata_key="litellm_metadata"
        )
        non_streamed_data["litellm_call_id"] = "non-streamed-call"
        await non_streamed_guardrail.async_post_call_success_hook(
            data=non_streamed_data,
            user_api_key_dict=make_key_dict(),
            response=GuardrailResponse(usage=AssembledUsage()),
        )
        non_streamed_trace = submitted_args(submit)["trace_id"]

        assert streamed_trace == "trace-on-the-messages-route"
        assert streamed_trace == non_streamed_trace

    @pytest.mark.asyncio
    async def test_one_call_is_metered_once_across_both_hooks_with_the_flag_on(
        self, monkeypatch, submit
    ):
        """The guard has to key on the minted id, or it stops recognising a call.

        With the flag on, the post-call hook resolves the minted id and the
        logging event resolved LiteLLM's call id. Two different keys for one
        call means neither hook sees the other's claim and the call is metered
        twice -- the exact double count this feature removes, reintroduced
        inside our own process.
        """
        monkeypatch.setenv(FLAG, "true")
        instance = self._new_guardrail()
        data = guardrail_data(
            stream=True, model=ANTHROPIC_MODEL, metadata_key="litellm_metadata"
        )
        data["litellm_call_id"] = "litellm-one-call"
        await self._mint(instance, data)

        await instance.async_log_success_event(
            logging_kwargs_from(data, litellm_call_id="litellm-one-call"),
            GuardrailResponse(usage=AssembledUsage()),
            NOW,
            LATER,
        )
        await instance.async_post_call_success_hook(
            data=data,
            user_api_key_dict=make_key_dict(),
            response=GuardrailResponse(usage=AssembledUsage()),
        )

        assert submit.call_count == 1

    @pytest.mark.asyncio
    async def test_one_call_is_metered_once_in_the_other_hook_order(
        self, monkeypatch, submit
    ):
        """Whichever hook arrives first, the second must find the claim."""
        monkeypatch.setenv(FLAG, "true")
        instance = self._new_guardrail()
        data = guardrail_data(
            stream=True, model=ANTHROPIC_MODEL, metadata_key="litellm_metadata"
        )
        data["litellm_call_id"] = "litellm-one-call"
        await self._mint(instance, data)

        await instance.async_post_call_success_hook(
            data=data,
            user_api_key_dict=make_key_dict(),
            response=GuardrailResponse(usage=AssembledUsage()),
        )
        await instance.async_log_success_event(
            logging_kwargs_from(data, litellm_call_id="litellm-one-call"),
            GuardrailResponse(usage=AssembledUsage()),
            NOW,
            LATER,
        )

        assert submit.call_count == 1

    @pytest.mark.asyncio
    async def test_two_streamed_calls_still_produce_two_rows_with_the_flag_on(
        self, monkeypatch, submit
    ):
        """The guard may only ever suppress a repeat, never a second call."""
        monkeypatch.setenv(FLAG, "true")
        instance = self._new_guardrail()
        for index in range(2):
            data = guardrail_data(
                stream=True, model=ANTHROPIC_MODEL, metadata_key="litellm_metadata"
            )
            await self._mint(instance, data)
            await instance.async_log_success_event(
                logging_kwargs_from(data, litellm_call_id="litellm-call-%d" % index),
                GuardrailResponse(usage=AssembledUsage()),
                NOW,
                LATER,
            )

        assert submit.call_count == 2
        ids = [call[0][1]["transaction_id"] for call in submit.call_args_list]
        assert ids[0] != ids[1]


PIN_SETTING = "REVENIUM_LITELLM_PIN_ORGANIZATION_TO_KEY_TEAM"


class TestStreamedRecordPinnedToTheKeyTeam:
    """The streamed logging path reads the key's team only where LiteLLM wrote it (BACK-4363)."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("call_type, model", [
        ("anthropic_messages", ANTHROPIC_MODEL),
        ("aresponses", RESPONSES_MODEL),
    ], ids=["messages", "responses"])
    async def test_a_streamed_record_names_the_pinned_key_team(
        self, guardrail, submit, monkeypatch, call_type, model
    ):
        monkeypatch.setenv(PIN_SETTING, "true")
        data = guardrail_data(
            headers={"x-revenium-organization-name": "SomeoneElse"},
            stream=True,
            model=model,
            metadata_key="litellm_metadata",
        )
        data["litellm_metadata"]["user_api_key_team_alias"] = "AcmeTeam"
        kwargs = logging_kwargs_from(data, call_type=call_type, model=model)
        kwargs["litellm_params"]["metadata"].update(
            {"user_id": "claude-code-user", "user_api_key_team_alias": "Planted"}
        )

        await guardrail.async_log_success_event(
            kwargs, GuardrailResponse(usage=AssembledUsage()), NOW, LATER
        )

        assert submit.call_count == 1
        assert submitted_args(submit)["organization_name"] == "AcmeTeam"

    @pytest.mark.asyncio
    async def test_a_team_alias_in_the_top_level_logging_metadata_does_not_beat_litellm_params(
        self, guardrail, submit, monkeypatch
    ):
        """LiteLLM spreads request params into the logging kwargs, so a top-level
        ``metadata`` there can be the caller's body; ``litellm_params`` is LiteLLM's."""
        monkeypatch.setenv(PIN_SETTING, "true")
        data = guardrail_data(
            headers={"x-revenium-organization-name": "SomeoneElse"},
            stream=True,
            model=ANTHROPIC_MODEL,
            metadata_key="litellm_metadata",
        )
        data["litellm_metadata"]["user_api_key_team_alias"] = "AcmeTeam"
        kwargs = logging_kwargs_from(data)
        kwargs["metadata"] = {"user_id": "claude-code-user", "user_api_key_team_alias": "Planted"}

        await guardrail.async_log_success_event(
            kwargs, GuardrailResponse(usage=AssembledUsage()), NOW, LATER
        )

        assert submit.call_count == 1
        assert submitted_args(submit)["organization_name"] == "AcmeTeam"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("planted_auth", [
        "x",
        {},
        {"team_alias": "Planted"},
        ["AcmeTeam"],
    ], ids=["string", "empty-object", "object-with-team", "list"])
    async def test_a_user_api_key_auth_planted_in_the_body_does_not_hide_the_key_team(
        self, guardrail, submit, monkeypatch, planted_auth
    ):
        """On ``/v1/messages`` ``metadata`` is the caller's body, so a
        ``user_api_key_auth`` there is not LiteLLM's auth object."""
        monkeypatch.setenv(PIN_SETTING, "true")
        data = guardrail_data(
            headers={"x-revenium-organization-name": "SomeoneElse"},
            stream=True,
            model=ANTHROPIC_MODEL,
            metadata_key="litellm_metadata",
        )
        data["litellm_metadata"]["user_api_key_team_alias"] = "AcmeTeam"
        kwargs = logging_kwargs_from(data)
        kwargs["litellm_params"]["metadata"].update(
            {"user_id": "claude-code-user", "user_api_key_auth": planted_auth}
        )

        await guardrail.async_log_success_event(
            kwargs, GuardrailResponse(usage=AssembledUsage()), NOW, LATER
        )

        assert submit.call_count == 1
        assert submitted_args(submit)["organization_name"] == "AcmeTeam"


# --- A caller-chosen call id never suppresses a record ----------------------
#
# LiteLLM sets ``litellm_call_id`` from the inbound ``x-litellm-call-id`` header
# unvalidated, so it is a value any caller with a gateway key picks. The
# duplicate guard used to key on it: every later request carrying the same
# header, from that caller or any other, was answered and paid for upstream but
# sent no usage record and never counted toward a cap (BACK-4371).

CALLER_CALL_ID = "11111111-2222-4333-8444-555555555555"
METADATA_CONTAINERS = ("litellm_metadata", "metadata")


def _mapping_containers(data):
    return {
        key: data[key] for key in METADATA_CONTAINERS if isinstance(data.get(key), dict)
    }


async def run_pre_call(instance, data, call_type="acompletion", key_dict=None):
    """Run the real pre-call hook and return what it added to the request metadata.

    The return value is whatever the SDK stamped for this request, read off the
    containers rather than by name, so a test can replay it in another request
    exactly as a caller who obtained it would.
    """
    before = {key: set(container) for key, container in _mapping_containers(data).items()}
    await instance.async_pre_call_hook(
        user_api_key_dict=key_dict or make_key_dict(),
        cache=MagicMock(),
        data=data,
        call_type=call_type,
    )
    written = {}
    for key, container in _mapping_containers(data).items():
        for name, value in container.items():
            if name not in before.get(key, set()):
                written[name] = value
    return written


def plant_in_both_containers(data, planted):
    """Seed ``planted`` into both metadata containers of a request body."""
    data.setdefault("litellm_metadata", {}).update(planted)
    data["metadata"].update(planted)


def responses_logging_kwargs(data, litellm_call_id=CALLER_CALL_ID):
    """The logging kwargs of a streamed ``/v1/responses`` call.

    LiteLLM's Responses entry point hands the logging object the request's
    ``litellm_metadata`` as ``litellm_params["metadata"]`` (litellm
    ``responses/main.py``), not the two containers ``logging_kwargs_from``
    copies across.
    """
    return {
        "model": RESPONSES_MODEL,
        "call_type": "aresponses",
        "stream": True,
        "litellm_call_id": litellm_call_id,
        "litellm_params": {"metadata": data["litellm_metadata"]},
    }


class TestACallerChosenCallIdNeverSuppressesARecord:
    """Every answered call is recorded whatever ``x-litellm-call-id`` it carries."""

    @pytest.fixture(autouse=True)
    def _clean_metering_owner(self):
        from revenium_middleware.litellm.proxy import _metering_owner

        _metering_owner.reset_metering_owner()
        yield
        _metering_owner.reset_metering_owner()

    @staticmethod
    def _chat_request(headers=None):
        data = guardrail_data(headers=headers)
        data["litellm_call_id"] = CALLER_CALL_ID
        return data

    @staticmethod
    async def _post_call(instance, data, index):
        await instance.async_post_call_success_hook(
            data=data,
            user_api_key_dict=make_key_dict(),
            response=GuardrailResponse(
                response_id="chatcmpl-%d" % index, usage=AssembledUsage()
            ),
        )

    @pytest.mark.asyncio
    async def test_six_chat_calls_with_one_caller_call_id_send_six_records(
        self, guardrail, submit
    ):
        for index in range(6):
            data = self._chat_request()
            await run_pre_call(guardrail, data)
            await self._post_call(guardrail, data, index)

        assert submit.call_count == 6

    @pytest.mark.asyncio
    async def test_four_streamed_chat_calls_with_one_caller_call_id_send_four_records(
        self, registered, submit
    ):
        for index in range(4):
            data = guardrail_data(stream=True)
            data["litellm_call_id"] = CALLER_CALL_ID
            await run_pre_call(registered, data)
            response = GuardrailResponse(
                response_id="chatcmpl-stream-%d" % index, usage=AssembledUsage()
            )
            await run_deferred_stream(captured_data=data, assembled_response=response)
            await registered.async_log_success_event(
                logging_kwargs_from(
                    data,
                    litellm_call_id=CALLER_CALL_ID,
                    call_type="acompletion",
                    model="gpt-4o-mini",
                ),
                response,
                NOW,
                LATER,
            )

        assert submit.call_count == 4

    @pytest.mark.asyncio
    async def test_a_call_id_another_caller_sent_gets_its_own_record(
        self, guardrail, submit
    ):
        for index, subscriber in enumerate(("alice", "bob")):
            data = self._chat_request(headers={"x-revenium-subscriber-id": subscriber})
            await run_pre_call(guardrail, data)
            await self._post_call(guardrail, data, index)

        assert submit.call_count == 2
        assert submit.call_args_list[1][0][1]["subscriber"]["id"] == "bob"

    @pytest.mark.asyncio
    async def test_without_pre_call_in_mode_every_call_is_recorded(
        self, guardrail, submit
    ):
        """Nothing stamps a request in this mode, so nothing found on one is trusted.

        Each request carries a key the SDK really stamped for another request,
        in both containers: a guardrail that trusted it here would let the
        caller who planted it suppress every record after the first.
        """
        planted = await run_pre_call(guardrail, self._chat_request())
        post_call_only = ReveniumGuardrail(
            guardrail_name="revenium", event_hook=["post_call"], default_on=True
        )
        for index in range(3):
            data = self._chat_request()
            plant_in_both_containers(data, planted)
            await self._post_call(post_call_only, data, index)

        assert submit.call_count == 3

    @pytest.mark.asyncio
    async def test_a_meter_key_replayed_in_the_request_body_cannot_suppress_a_record(
        self, guardrail, submit
    ):
        first = self._chat_request()
        stamped = await run_pre_call(guardrail, first)
        await self._post_call(guardrail, first, 0)

        replay = self._chat_request()
        plant_in_both_containers(replay, stamped)
        await run_pre_call(guardrail, replay)
        await self._post_call(guardrail, replay, 1)

        assert submit.call_count == 2

    @pytest.mark.asyncio
    async def test_four_streamed_responses_calls_with_one_caller_call_id_send_four_records(
        self, guardrail, submit
    ):
        """``/v1/responses`` with ``stream: true`` is metered from the logging event."""
        for _ in range(4):
            data = guardrail_data(
                stream=True, model=RESPONSES_MODEL, metadata_key="litellm_metadata"
            )
            data["litellm_call_id"] = CALLER_CALL_ID
            await run_pre_call(guardrail, data, call_type="aresponses")
            await guardrail.async_log_success_event(
                responses_logging_kwargs(data),
                GuardrailResponse(usage=AssembledUsage()),
                NOW,
                LATER,
            )

        assert submit.call_count == 4

    @pytest.mark.asyncio
    async def test_streamed_messages_calls_with_one_caller_call_id_and_the_flag_off_send_one_record_each(
        self, monkeypatch, submit
    ):
        """With no minted shared id, the caller's id is all that told these calls apart."""
        monkeypatch.setenv(FLAG, "false")
        instance = ReveniumGuardrail(
            guardrail_name="revenium", event_hook=["pre_call", "post_call"], default_on=True
        )
        for _ in range(3):
            data = guardrail_data(
                stream=True, model=ANTHROPIC_MODEL, metadata_key="litellm_metadata"
            )
            data["litellm_call_id"] = CALLER_CALL_ID
            await run_pre_call(instance, data, call_type="anthropic_messages")
            await instance.async_log_success_event(
                logging_kwargs_from(data, litellm_call_id=CALLER_CALL_ID),
                GuardrailResponse(usage=AssembledUsage()),
                NOW,
                LATER,
            )

        assert submit.call_count == 3

    @pytest.mark.asyncio
    async def test_a_responses_pre_call_leaves_the_callers_metadata_unchanged(
        self, guardrail
    ):
        """On ``/v1/responses`` ``metadata`` is the caller's own body field.

        LiteLLM forwards it to the provider, which caps it at 16 pairs and
        returns it on the stored response, so nothing of ours may be written
        there.
        """
        data = guardrail_data(
            stream=True,
            model=RESPONSES_MODEL,
            metadata_key="litellm_metadata",
            metadata={"order": "42"},
        )
        callers_metadata = copy.deepcopy(data["metadata"])

        await run_pre_call(guardrail, data, call_type="aresponses")

        assert data["metadata"] == callers_metadata

    @pytest.mark.parametrize("logging_event_first", [True, False])
    @pytest.mark.asyncio
    async def test_one_call_through_both_paths_with_a_caller_call_id_is_recorded_once(
        self, monkeypatch, submit, logging_event_first
    ):
        """The guard still does its job: one call reaching both hooks is one record.

        The response carries its own id, which only the post-call hook reads,
        so a guard keyed on the stored transaction id would see two calls here.
        """
        monkeypatch.setenv(FLAG, "false")
        instance = ReveniumGuardrail(
            guardrail_name="revenium", event_hook=["pre_call", "post_call"], default_on=True
        )
        data = guardrail_data(
            stream=True, model=ANTHROPIC_MODEL, metadata_key="litellm_metadata"
        )
        data["litellm_call_id"] = CALLER_CALL_ID
        await run_pre_call(instance, data, call_type="anthropic_messages")
        response = GuardrailResponse(response_id="msg_distinct", usage=AssembledUsage())

        async def logging_event():
            await instance.async_log_success_event(
                logging_kwargs_from(data, litellm_call_id=CALLER_CALL_ID),
                response,
                NOW,
                LATER,
            )

        async def post_call():
            await instance.async_post_call_success_hook(
                data=data, user_api_key_dict=make_key_dict(), response=response
            )

        hooks = (logging_event, post_call) if logging_event_first else (post_call, logging_event)
        for hook in hooks:
            await hook()

        assert submit.call_count == 1


# --- A caller-chosen call id never merges records ---------------------------
#
# Sending a record is not enough: Revenium keeps one record per organization
# and transaction id. A record filed under the caller's ``x-litellm-call-id``
# is merged with every other call that repeated the header, so spend stops
# growing and a cap never fills (BACK-4376). A record whose response carries
# no provider id must take an id the caller cannot choose.

EMBEDDING_MODEL = "text-embedding-3-small"


def records_revenium_keeps(submit):
    """Revenium stores one record per (organization, transaction id)."""
    return {
        (call[0][1]["organization_name"], call[0][1]["transaction_id"])
        for call in submit.call_args_list
    }


def sent_transaction_ids(submit):
    return [call[0][1]["transaction_id"] for call in submit.call_args_list]


def request_id_of(stamped):
    """The id a request's record carries: its stamped meter key without this process's prefix."""
    return stamped[guardrail_module._METER_KEY_FIELD][len(guardrail_module._METER_KEY_PREFIX):]


def assert_no_id_is_the_callers_or_reveals_the_prefix(ids):
    process_prefix = guardrail_module._METER_KEY_PREFIX.rstrip(":")
    assert CALLER_CALL_ID not in ids
    for transaction_id in ids:
        assert process_prefix not in transaction_id


def embedding_response():
    """An ``EmbeddingResponse``, which has no ``.id`` for the record to take."""
    from litellm.types.utils import EmbeddingResponse, Usage as LiteLLMUsage

    return EmbeddingResponse(
        model=EMBEDDING_MODEL,
        data=[{"embedding": [0.1, 0.2], "index": 0, "object": "embedding"}],
        usage=LiteLLMUsage(prompt_tokens=7, completion_tokens=0, total_tokens=7),
    )


def flag_off_guardrail(monkeypatch):
    """A trusted guardrail with the shared call id off, read when it is built."""
    monkeypatch.setenv(FLAG, "false")
    return ReveniumGuardrail(
        guardrail_name="revenium", event_hook=["pre_call", "post_call"], default_on=True
    )


class TestACallerChosenCallIdNeverMergesRecords:
    """Every answered call is its own record whatever ``x-litellm-call-id`` it carries."""

    @pytest.fixture(autouse=True)
    def _clean_metering_owner(self):
        from revenium_middleware.litellm.proxy import _metering_owner

        _metering_owner.reset_metering_owner()
        yield
        _metering_owner.reset_metering_owner()

    @staticmethod
    async def _streamed_messages_request(instance):
        data = guardrail_data(
            stream=True, model=ANTHROPIC_MODEL, metadata_key="litellm_metadata"
        )
        data["litellm_call_id"] = CALLER_CALL_ID
        stamped = await run_pre_call(instance, data, call_type="anthropic_messages")
        return data, stamped

    @pytest.mark.asyncio
    async def test_four_streamed_responses_calls_with_one_caller_call_id_are_four_records(
        self, guardrail, submit
    ):
        expected = []
        for _ in range(4):
            data = guardrail_data(
                stream=True, model=RESPONSES_MODEL, metadata_key="litellm_metadata"
            )
            data["litellm_call_id"] = CALLER_CALL_ID
            stamped = await run_pre_call(guardrail, data, call_type="aresponses")
            expected.append(request_id_of(stamped))
            await guardrail.async_log_success_event(
                responses_logging_kwargs(data),
                GuardrailResponse(usage=AssembledUsage()),
                NOW,
                LATER,
            )

        assert len(records_revenium_keeps(submit)) == 4
        assert sent_transaction_ids(submit) == expected
        assert_no_id_is_the_callers_or_reveals_the_prefix(sent_transaction_ids(submit))

    @pytest.mark.parametrize("streamed", [True, False], ids=["streamed", "not_streamed"])
    @pytest.mark.asyncio
    async def test_three_messages_calls_with_one_caller_call_id_and_the_flag_off_are_three_records(
        self, monkeypatch, submit, streamed
    ):
        instance = flag_off_guardrail(monkeypatch)
        expected = []
        for _ in range(3):
            data = guardrail_data(
                stream=streamed, model=ANTHROPIC_MODEL, metadata_key="litellm_metadata"
            )
            data["litellm_call_id"] = CALLER_CALL_ID
            stamped = await run_pre_call(instance, data, call_type="anthropic_messages")
            expected.append(request_id_of(stamped))
            if streamed:
                await instance.async_log_success_event(
                    logging_kwargs_from(data, litellm_call_id=CALLER_CALL_ID),
                    GuardrailResponse(
                        response_id="msg_01AnthropicStream", usage=AssembledUsage()
                    ),
                    NOW,
                    LATER,
                )
            else:
                # The raw body this route hands the hook: a dict, so no ``.id``.
                await instance.async_post_call_success_hook(
                    data=data,
                    user_api_key_dict=make_key_dict(),
                    response=copy.deepcopy(ANTHROPIC_NON_STREAMED_BODY),
                )

        assert len(records_revenium_keeps(submit)) == 3
        assert sent_transaction_ids(submit) == expected
        assert_no_id_is_the_callers_or_reveals_the_prefix(sent_transaction_ids(submit))

    @pytest.mark.asyncio
    async def test_three_embeddings_calls_with_one_caller_call_id_are_three_records(
        self, guardrail, submit
    ):
        expected = []
        for _ in range(3):
            data = guardrail_data(model=EMBEDDING_MODEL)
            data["litellm_call_id"] = CALLER_CALL_ID
            stamped = await run_pre_call(guardrail, data, call_type="aembedding")
            expected.append(request_id_of(stamped))
            await guardrail.async_post_call_success_hook(
                data=data, user_api_key_dict=make_key_dict(), response=embedding_response()
            )

        assert len(records_revenium_keeps(submit)) == 3
        assert sent_transaction_ids(submit) == expected
        assert_no_id_is_the_callers_or_reveals_the_prefix(sent_transaction_ids(submit))

    @pytest.mark.asyncio
    async def test_without_pre_call_in_mode_idless_calls_with_one_caller_call_id_are_separate_records(
        self, submit
    ):
        """Nothing is stamped in this mode, so there is no per-request id to pin, only distinctness."""
        post_call_only = ReveniumGuardrail(
            guardrail_name="revenium", event_hook=["post_call"], default_on=True
        )
        for _ in range(3):
            data = guardrail_data(model=EMBEDDING_MODEL)
            data["litellm_call_id"] = CALLER_CALL_ID
            await post_call_only.async_post_call_success_hook(
                data=data, user_api_key_dict=make_key_dict(), response=embedding_response()
            )

        assert len(records_revenium_keeps(submit)) == 3
        assert_no_id_is_the_callers_or_reveals_the_prefix(sent_transaction_ids(submit))

    @pytest.mark.parametrize(
        "logging_event_first", [True, False], ids=["logging_first", "post_call_first"]
    )
    @pytest.mark.asyncio
    async def test_two_calls_with_one_caller_call_id_each_reported_through_both_paths_are_two_records(
        self, monkeypatch, submit, logging_event_first
    ):
        instance = flag_off_guardrail(monkeypatch)
        expected = []
        for _ in range(2):
            data, stamped = await self._streamed_messages_request(instance)
            expected.append(request_id_of(stamped))
            response = GuardrailResponse(response_id=None, usage=AssembledUsage())

            async def logging_event():
                await instance.async_log_success_event(
                    logging_kwargs_from(data, litellm_call_id=CALLER_CALL_ID),
                    response,
                    NOW,
                    LATER,
                )

            async def post_call():
                await instance.async_post_call_success_hook(
                    data=data, user_api_key_dict=make_key_dict(), response=response
                )

            hooks = (logging_event, post_call) if logging_event_first else (post_call, logging_event)
            for hook in hooks:
                await hook()

        assert submit.call_count == 2
        assert len(records_revenium_keeps(submit)) == 2
        assert sent_transaction_ids(submit) == expected
        assert_no_id_is_the_callers_or_reveals_the_prefix(sent_transaction_ids(submit))

    @pytest.mark.asyncio
    async def test_both_paths_file_one_call_under_one_transaction_id(
        self, monkeypatch, submit
    ):
        """With no provider id, Revenium's own duplicate check backs up the in-process claim.

        The claim memory here forgets every call at once, so both paths send.
        Both must name the same request id, or Revenium would keep two records
        for one call.
        """
        instance = flag_off_guardrail(monkeypatch)
        instance._metered_calls = guardrail_module._MeteredCalls(capacity=0)
        data, stamped = await self._streamed_messages_request(instance)
        response = GuardrailResponse(response_id=None, usage=AssembledUsage())

        await instance.async_log_success_event(
            logging_kwargs_from(data, litellm_call_id=CALLER_CALL_ID), response, NOW, LATER
        )
        await instance.async_post_call_success_hook(
            data=data, user_api_key_dict=make_key_dict(), response=response
        )

        assert submit.call_count == 2
        assert sent_transaction_ids(submit) == [request_id_of(stamped)] * 2
        assert len(records_revenium_keeps(submit)) == 1
        assert_no_id_is_the_callers_or_reveals_the_prefix(sent_transaction_ids(submit))
