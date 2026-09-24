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

import datetime

import pytest

pytest.importorskip("litellm")
pytest.importorskip("fastapi")

from unittest.mock import MagicMock, patch  # noqa: E402

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
        await ProxyBaseLLMRequestProcessing._run_deferred_stream_guardrails(
            captured_data=data,
            captured_user_api_key_dict=make_key_dict(),
            captured_logging_obj=MagicMock(),
            assembled_response=response,
            cache_hit=False,
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
        await ProxyBaseLLMRequestProcessing._run_deferred_stream_guardrails(
            captured_data=data,
            captured_user_api_key_dict=make_key_dict(),
            captured_logging_obj=MagicMock(),
            assembled_response=GuardrailResponse(usage=AssembledUsage()),
            cache_hit=False,
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
        await ProxyBaseLLMRequestProcessing._run_deferred_stream_guardrails(
            captured_data=guardrail_data(stream=True),
            captured_user_api_key_dict=make_key_dict(),
            captured_logging_obj=MagicMock(),
            assembled_response=GuardrailResponse(usage=AssembledUsage()),
            cache_hit=False,
        )


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
    async def test_the_metered_id_is_litellms_call_id_not_the_anthropic_message_id(
        self, guardrail, submit
    ):
        """Both rows on this route must carry the same kind of id.

        The non-streamed row on ``/v1/messages`` has no ``.id`` to read off its
        raw Anthropic body and already falls through to LiteLLM's call id.
        Metering the streamed row on the assembled response's ``.id`` would give
        the two rows different identity shapes for the same route.
        """
        kwargs = logging_kwargs(
            "anthropic_messages", stream=True, litellm_call_id="litellm-anthropic-stream"
        )
        response = GuardrailResponse(
            response_id="msg_01AnthropicStream", usage=AssembledUsage()
        )

        await guardrail.async_log_success_event(kwargs, response, NOW, LATER)

        args = submitted_args(submit)
        assert args["transaction_id"] == "litellm-anthropic-stream"
        assert args["transaction_id"] != "msg_01AnthropicStream"

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


class TestTheLoggingEventMetersNothingElse:
    """The route gate, which is load-bearing: this hook fires on every call."""

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

    @staticmethod
    def _one_call(litellm_call_id="litellm-shared-call-id", call_type="anthropic_messages"):
        """The two shapes one completed call reaches our two hooks in."""
        kwargs = logging_kwargs(call_type, stream=True, litellm_call_id=litellm_call_id)
        data = guardrail_data(stream=True, model=ANTHROPIC_MODEL)
        data["litellm_call_id"] = litellm_call_id
        return kwargs, data

    @pytest.mark.asyncio
    async def test_the_post_call_hook_does_not_re_meter_a_logged_stream(
        self, guardrail, submit
    ):
        kwargs, data = self._one_call()
        response = GuardrailResponse(usage=AssembledUsage())

        await guardrail.async_log_success_event(kwargs, response, NOW, LATER)
        await guardrail.async_post_call_success_hook(
            data=data, user_api_key_dict=make_key_dict(), response=response
        )

        assert submitted_args(submit)["transaction_id"] == "litellm-shared-call-id"

    @pytest.mark.asyncio
    async def test_the_logging_event_does_not_re_meter_a_post_called_stream(
        self, guardrail, submit
    ):
        """The order a ``CustomStreamWrapper`` on the Anthropic route would take.

        ``_run_deferred_stream_guardrails`` runs the success hook first and then
        dispatches success logging, so this is the order that matters if a
        provider's ``anthropic_messages`` stream ever arrives as a wrapper.
        """
        kwargs, data = self._one_call()
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
            kwargs, _ = self._one_call(litellm_call_id=call_id)
            await guardrail.async_log_success_event(
                kwargs, GuardrailResponse(usage=AssembledUsage()), NOW, LATER
            )

        assert submit.call_count == 2

    @pytest.mark.asyncio
    async def test_a_streamed_chat_completion_still_produces_exactly_one_row(
        self, registered, submit
    ):
        """The route the guard must not disturb: one row, from the post-call hook."""
        kwargs, data = self._one_call(call_type="acompletion")
        response = GuardrailResponse(response_id="chatcmpl-stream", usage=AssembledUsage())

        await ProxyBaseLLMRequestProcessing._run_deferred_stream_guardrails(
            captured_data=data,
            captured_user_api_key_dict=make_key_dict(),
            captured_logging_obj=MagicMock(),
            assembled_response=response,
            cache_hit=False,
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
        kwargs, data = self._one_call()

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
    async def test_the_streamed_row_falls_back_to_litellms_call_id_with_the_flag_off(
        self, submit
    ):
        """Flag off is the shipped default, and it must behave as it did."""
        instance = self._new_guardrail()
        data = guardrail_data(
            stream=True, model=ANTHROPIC_MODEL, metadata_key="litellm_metadata"
        )
        await self._mint(instance, data)
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

        assert submitted_args(submit)["transaction_id"] == "litellm-anthropic-stream"

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
