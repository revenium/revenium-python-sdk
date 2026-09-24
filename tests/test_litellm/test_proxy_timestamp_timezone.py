"""A proxy row's times are the UTC instant, whatever the proxy machine's clock says.

Every LiteLLM proxy row is published with a ``Z`` suffix, which means UTC. The
values behind it come from two different places, and only one of them was
already UTC:

* ``async_post_call_success_hook`` derives its start from the response's
  ``created`` epoch through ``fromtimestamp(..., tz=utc)`` and its end from
  ``datetime.now(timezone.utc)``, so both are aware.
* ``async_log_success_event`` -- the streamed Anthropic route, which is the one
  Claude Code speaks -- is handed LiteLLM's own ``start_time`` and ``end_time``,
  which are **naive** and carry the proxy machine's local wall clock. The
  deprecated ``MiddlewareHandler`` callback takes the same two arguments on both
  of its paths.

Formatting a naive local datetime with a literal ``Z`` publishes the wall-clock
reading as though it were UTC, so a proxy in US Mountain time files every row six
hours early. That was survivable while the Claude Code row carried the correct
time alongside it. Under the shared call id (BACK-2399) the gateway row is the
one the duplicate check keeps, so the wrong time becomes the only time, and every
"spend by hour" and "spend by day" chart for that customer shifts by the proxy's
offset.

These tests run under a fixed non-UTC zone so the bug cannot hide behind a UTC
build agent.
"""

import datetime
import os
import time
from types import SimpleNamespace
from unittest.mock import patch

import pytest

pytest.importorskip("litellm")
# The guardrail imports fastapi, which litellm ships only with its proxy extra.
# The vendor-shape sentinel job installs litellm without it, so this module
# must skip there the way every other guardrail test module does.
pytest.importorskip("fastapi")

from revenium_middleware.litellm.proxy import guardrail as guardrail_module  # noqa: E402
from revenium_middleware.litellm.proxy import middleware as mw  # noqa: E402
from revenium_middleware.litellm.proxy.guardrail import ReveniumGuardrail  # noqa: E402

from .proxy_hook_harness import (  # noqa: E402
    GuardrailResponse,
    base_kwargs,
    drive_metering,
    guardrail_data,
    make_key_dict,
    make_success_response,
    run_hook,
    run_inline,
    submitted_args,
)

ANTHROPIC_MODEL = "claude-sonnet-4-5-20250929"

# US Mountain time in September is UTC-6, which is the offset the local
# end-to-end capture actually ran under: a gateway row stamped
# "2026-09-15 21:52:55" sat beside its Claude Code partner at
# "2026-09-16 03:52:55" for the same call.
PROXY_ZONE = "America/Denver"

# What LiteLLM hands a logging event: naive, and reading the proxy machine's
# own wall clock.
NAIVE_LOCAL_START = datetime.datetime(2026, 9, 15, 21, 52, 55)
NAIVE_LOCAL_END = datetime.datetime(2026, 9, 15, 21, 52, 55, 250000)

# The same two instants, in UTC, which is what a ``Z`` suffix promises.
UTC_START_STR = "2026-09-16T03:52:55Z"
UTC_END_STR = "2026-09-16T03:52:55Z"

# An already-aware pair, for the control that proves the fix changes nothing on
# the path that was always correct.
AWARE_START = datetime.datetime(2026, 9, 16, 3, 52, 55, tzinfo=datetime.timezone.utc)
AWARE_END = AWARE_START + datetime.timedelta(milliseconds=250)


@pytest.fixture(autouse=True)
def proxy_runs_in_mountain_time():
    """Run every test in this module on a proxy machine that is not on UTC.

    ``datetime.astimezone`` resolves a naive value against the platform's local
    zone, which ``time.tzset()`` rereads from ``TZ``. Without this the suite
    would pass on a UTC build agent while the customer's proxy stayed wrong.
    """
    original = os.environ.get("TZ")
    os.environ["TZ"] = PROXY_ZONE
    time.tzset()
    assert NAIVE_LOCAL_START.astimezone(datetime.timezone.utc).hour == 3, (
        "the fixture did not take: this machine is not resolving naive "
        "datetimes against " + PROXY_ZONE
    )
    yield
    if original is None:
        os.environ.pop("TZ", None)
    else:
        os.environ["TZ"] = original
    time.tzset()


class AssembledUsage:
    """The usage totals LiteLLM computes once a stream has finished."""

    prompt_tokens = 21
    completion_tokens = 34
    total_tokens = 55


def logging_kwargs(call_type="anthropic_messages", stream=True, model=ANTHROPIC_MODEL):
    """``Logging.model_call_details`` as the logging dispatch hands it over."""
    return {
        "model": model,
        "call_type": call_type,
        "stream": stream,
        "litellm_call_id": "litellm-call-id-tz",
        "litellm_params": {
            "metadata": {
                "headers": {},
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
    with patch.object(guardrail_module, "submit_ai_event") as mock, \
            patch.object(guardrail_module, "run_async_in_thread",
                         side_effect=drive_metering):
        yield mock


class TestTheStreamedAnthropicRowCarriesTheUtcInstant:
    """The route Claude Code speaks, and the row the duplicate check keeps."""

    @pytest.mark.asyncio
    async def test_request_time_is_the_utc_instant_not_the_wall_clock(
        self, guardrail, submit
    ):
        await guardrail.async_log_success_event(
            logging_kwargs(),
            GuardrailResponse(response_id="msg_01Tz", usage=AssembledUsage()),
            NAIVE_LOCAL_START,
            NAIVE_LOCAL_END,
        )

        assert submitted_args(submit)["request_time"] == UTC_START_STR

    @pytest.mark.asyncio
    async def test_response_and_completion_start_are_the_utc_instant(
        self, guardrail, submit
    ):
        await guardrail.async_log_success_event(
            logging_kwargs(),
            GuardrailResponse(response_id="msg_01Tz", usage=AssembledUsage()),
            NAIVE_LOCAL_START,
            NAIVE_LOCAL_END,
        )

        args = submitted_args(submit)
        assert args["response_time"] == UTC_END_STR
        assert args["completion_start_time"] == UTC_END_STR

    @pytest.mark.asyncio
    async def test_an_already_aware_pair_is_published_unchanged(
        self, guardrail, submit
    ):
        """The control: the path that was always UTC must not move."""
        await guardrail.async_log_success_event(
            logging_kwargs(),
            GuardrailResponse(response_id="msg_01Tz", usage=AssembledUsage()),
            AWARE_START,
            AWARE_END,
        )

        args = submitted_args(submit)
        assert args["request_time"] == UTC_START_STR
        assert args["response_time"] == UTC_END_STR


class TestTheNonStreamedRowIsUnchanged:
    """``async_post_call_success_hook`` was already correct and stays correct."""

    @pytest.mark.asyncio
    async def test_the_created_epoch_still_reads_as_utc(self, guardrail, submit):
        created = int(AWARE_START.timestamp())
        await guardrail.async_post_call_success_hook(
            data=guardrail_data(stream=False, model=ANTHROPIC_MODEL),
            user_api_key_dict=make_key_dict(),
            response=GuardrailResponse(usage=AssembledUsage(), created=created),
        )

        assert submitted_args(submit)["request_time"] == UTC_START_STR


@patch.object(mw, "run_async_in_thread", side_effect=run_inline)
@patch.object(mw, "get_client", return_value=object())
@patch.object(mw, "submit_ai_event")
class TestTheDeprecatedCallbackCarriesTheUtcInstant:
    """``MiddlewareHandler`` takes the same naive pair on both of its paths."""

    @staticmethod
    def _usage():
        return SimpleNamespace(prompt_tokens=100, completion_tokens=10, total_tokens=110)

    def test_a_metered_success_reports_the_utc_instant(
        self, mock_submit, _get_client, _run
    ):
        from revenium_middleware.litellm.proxy import _metering_owner

        _metering_owner.reset_metering_owner()
        run_hook(mw.proxy_handler_instance.async_log_success_event(
            base_kwargs(),
            make_success_response(self._usage()),
            NAIVE_LOCAL_START,
            NAIVE_LOCAL_END,
        ))

        args = submitted_args(mock_submit)
        assert args["request_time"] == UTC_START_STR
        assert args["response_time"] == UTC_END_STR
        assert args["completion_start_time"] == UTC_END_STR

    def test_a_metered_failure_reports_the_utc_instant(
        self, mock_submit, _get_client, _run
    ):
        from revenium_middleware.litellm.proxy import _metering_owner

        _metering_owner.reset_metering_owner()
        run_hook(mw.proxy_handler_instance.async_log_failure_event(
            base_kwargs(),
            Exception("upstream refused"),
            NAIVE_LOCAL_START,
            NAIVE_LOCAL_END,
        ))

        args = submitted_args(mock_submit)
        assert args["request_time"] == UTC_START_STR
        assert args["response_time"] == UTC_END_STR
        assert args["completion_start_time"] == UTC_END_STR
