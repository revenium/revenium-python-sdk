"""The LiteLLM proxy's metering debug lines hide the subscriber (BACK-3986).

Both proxy integrations logged the whole usage record they were about to
send, with the subscriber's email and credential, whenever the
``revenium_middleware`` logger was at DEBUG. The record itself still carries
both; only the log line is sanitized.
"""

import datetime
import logging
import time
import warnings
from unittest.mock import patch

import pytest

pytest.importorskip("litellm")

from revenium_middleware.litellm.proxy import _metering_owner  # noqa: E402
from revenium_middleware.litellm.proxy import middleware as mw  # noqa: E402

from .proxy_hook_harness import (  # noqa: E402
    GuardrailResponse,
    SubscriptableResponse,
    after_pre_call,
    base_kwargs,
    drive_metering,
    guardrail_data,
    make_key_dict,
    run_hook,
    submitted_args,
)

NOW = datetime.datetime.now(datetime.timezone.utc)
LATER = NOW + datetime.timedelta(milliseconds=200)
EMAIL = "person@customer.example"
KEY_ALIAS = "sk-live-SUBSCRIBER-KEY-123"
EMAIL_HEADER = "x-revenium-subscriber-email"


class Usage(dict):
    def __init__(self):
        super().__init__(prompt_tokens=5, completion_tokens=10, total_tokens=15)
        self.prompt_tokens = 5
        self.completion_tokens = 10
        self.total_tokens = 15


@pytest.fixture(autouse=True)
def _clean_registry():
    _metering_owner.reset_metering_owner()
    mw._dedup_warning_emitted = False
    yield
    _metering_owner.reset_metering_owner()
    mw._dedup_warning_emitted = False


def callback_kwargs():
    kwargs = base_kwargs()
    kwargs["litellm_params"]["metadata"] = {
        "hidden_params": {"optional_params": {"stream": False}, "litellm_overhead_time_ms": 10},
        "user_api_key_alias": KEY_ALIAS,
        "headers": {EMAIL_HEADER: EMAIL},
    }
    return kwargs


def new_handler():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return mw.MiddlewareHandler()


def assert_subscriber_hidden(caplog, args):
    assert "args" in caplog.text
    assert EMAIL not in caplog.text
    assert KEY_ALIAS not in caplog.text
    assert args["subscriber"]["email"] == EMAIL, "the record sent to Revenium keeps the subscriber"


class TestProxyCallbackDebugLog:
    def test_success_callback_hides_subscriber(self, caplog):
        with patch.object(mw, "submit_ai_event") as submit, \
                patch.object(mw, "run_async_in_thread", side_effect=drive_metering), \
                caplog.at_level(logging.DEBUG, logger="revenium_middleware"):
            run_hook(new_handler().async_log_success_event(
                callback_kwargs(), SubscriptableResponse("chatcmpl-callback", Usage()), NOW, LATER,
            ))
        assert_subscriber_hidden(caplog, submitted_args(submit))

    def test_failure_callback_hides_subscriber(self, caplog):
        with patch.object(mw, "submit_ai_event") as submit, \
                patch.object(mw, "run_async_in_thread", side_effect=drive_metering), \
                caplog.at_level(logging.DEBUG, logger="revenium_middleware"):
            run_hook(new_handler().async_log_failure_event(
                callback_kwargs(), Exception("provider exploded"), NOW, LATER,
            ))
        assert_subscriber_hidden(caplog, submitted_args(submit))


class TestGuardrailDebugLog:
    @pytest.mark.asyncio
    async def test_guardrail_metering_line_hides_subscriber(self, caplog):
        pytest.importorskip("fastapi", reason="the guardrail imports the LiteLLM proxy, installed in CI's litellm job")
        from revenium_middleware.litellm.proxy import guardrail as gmod
        from revenium_middleware.litellm.proxy.guardrail import ReveniumGuardrail

        with patch.object(gmod, "submit_ai_event") as submit, \
                patch.object(gmod, "run_async_in_thread", side_effect=drive_metering), \
                caplog.at_level(logging.DEBUG, logger="revenium_middleware"):
            await ReveniumGuardrail(guardrail_name="revenium").async_post_call_success_hook(
                data=guardrail_data(headers={EMAIL_HEADER: EMAIL}, metadata={"user_api_key_alias": KEY_ALIAS}),
                user_api_key_dict=make_key_dict(user_email="owner@acme.com", key_alias=KEY_ALIAS),
                response=GuardrailResponse(usage=Usage()),
            )
        assert_subscriber_hidden(caplog, submitted_args(submit))

    @pytest.mark.asyncio
    async def test_guardrail_builds_no_debug_text_when_debug_is_off(self, caplog, monkeypatch):
        """BACK-4367: a debug line that is never printed must cost the gateway nothing."""
        pytest.importorskip("fastapi", reason="the guardrail imports the LiteLLM proxy, installed in CI's litellm job")
        from revenium_middleware.litellm.proxy import guardrail as gmod
        from revenium_middleware.litellm.proxy.guardrail import ReveniumGuardrail

        monkeypatch.delenv("REVENIUM_DEBUG", raising=False)
        monkeypatch.delenv("REVENIUM_LITELLM_SHARED_CALL_ID", raising=False)
        guardrail = ReveniumGuardrail(guardrail_name="revenium")

        with patch.object(gmod, "submit_ai_event") as submit, \
                patch.object(gmod, "run_async_in_thread", side_effect=drive_metering), \
                patch.object(gmod, "sanitize_for_logging") as spy, \
                caplog.at_level(logging.INFO, logger="revenium_middleware.extension"):
            await guardrail.async_post_call_success_hook(
                data=guardrail_data(),
                user_api_key_dict=make_key_dict(),
                response=GuardrailResponse(usage=Usage()),
            )

        submitted_args(submit)
        assert spy.call_count == 0

    @pytest.mark.asyncio
    @pytest.mark.parametrize("level", [logging.INFO, logging.DEBUG], ids=["info", "debug"])
    async def test_a_long_model_name_does_not_hold_the_event_loop(self, caplog, monkeypatch, level):
        """BACK-4367: recording a refused request with a 100,000 character model name stays fast."""
        pytest.importorskip("fastapi", reason="the guardrail imports the LiteLLM proxy, installed in CI's litellm job")
        from revenium_middleware.litellm.proxy import guardrail as gmod
        from revenium_middleware.litellm.proxy.guardrail import ReveniumGuardrail

        monkeypatch.delenv("REVENIUM_DEBUG", raising=False)
        monkeypatch.delenv("REVENIUM_LITELLM_SHARED_CALL_ID", raising=False)
        guardrail = ReveniumGuardrail(guardrail_name="revenium")
        long_model = "m" * 100_000

        with patch.object(gmod, "submit_ai_event") as submit, \
                patch.object(gmod, "run_async_in_thread", side_effect=drive_metering), \
                caplog.at_level(level, logger="revenium_middleware.extension"):
            started = time.perf_counter()
            await guardrail.async_post_call_failure_hook(
                request_data=after_pre_call(guardrail_data(model=long_model)),
                original_exception=ValueError("unknown model"),
                user_api_key_dict=make_key_dict(),
            )
            elapsed = time.perf_counter() - started

        assert submit.call_count == 1
        assert elapsed < 1.0
        assert "m" * 100 not in caplog.text
