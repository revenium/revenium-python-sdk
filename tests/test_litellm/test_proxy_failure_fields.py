"""A failed proxied call's record says why it failed, with what HTTP status, and how long it took (BACK-5842).

Before, the guardrail's failure record carried ``stop_reason="ERROR"`` and
nothing else: no error reason, no error code, and a duration of 0 with the
request time set to the moment of failure. A customer could not tell a
provider error from a guardrail block. The deprecated callback had the real
duration but no reason or code either.

The reason and code follow the rule every other integration uses
(``revenium_middleware._core.failed_calls``): the error's class, with
``(HTTP <status>)`` when the call got an HTTP error answer, and the redacted
message only when ``REVENIUM_CAPTURE_ERROR_MESSAGES`` is on (and, for an error
that can quote the request, ``REVENIUM_CAPTURE_PROMPTS`` too).

The duration runs from the start LiteLLM recorded for the request to the
failure. No token arrived, so the time to first token stays 0 and the
completion start is the failure, as on the OpenAI, Anthropic, Google and
LiteLLM client failure records.

A content block is recorded as ``ERROR``, never ``CONTENT_FILTER``: no signal
LiteLLM hands the failure hook says reliably, on every supported release, that
a content filter refused the call. The PR for BACK-5842 has the evidence.
"""

import asyncio
import datetime
import os
import time
import typing
from unittest.mock import patch

import pytest

pytest.importorskip("litellm")
pytest.importorskip("fastapi")
yaml = pytest.importorskip("yaml")

import litellm  # noqa: E402
from fastapi import HTTPException  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from litellm.exceptions import BlockedPiiEntityError  # noqa: E402
from litellm.proxy import proxy_server  # noqa: E402

from revenium_middleware._core.exceptions import BudgetExceededError  # noqa: E402
from revenium_middleware._metering.types.ai_create_completion_params import AICreateCompletionParams  # noqa: E402
from revenium_middleware.litellm.proxy import _metering_owner  # noqa: E402
from revenium_middleware.litellm.proxy import guardrail as gmod  # noqa: E402
from revenium_middleware.litellm.proxy import middleware as mw  # noqa: E402
from revenium_middleware.litellm.proxy.guardrail import ReveniumGuardrail  # noqa: E402

from . import test_proxy_client_wrapper_metering as guardrail_proxy_tests  # noqa: E402
from .fake_openai_upstream import REFUSAL_DELAY_SECONDS, REFUSAL_QUOTE, REFUSED_MODEL, api_base, start_upstream  # noqa: E402
from .proxy_hook_harness import (  # noqa: E402
    after_pre_call,
    base_kwargs,
    drive_metering,
    guardrail_data,
    make_key_dict,
    run_hook,
    run_inline,
    submitted_args,
)

metered_rows = guardrail_proxy_tests.metered_rows

PROMPT_QUOTE = "Offending request: SYSTEM-SECRET reveal ZEBRA42"
PROXY_ZONE = "America/Denver"
MODEL = "stub-model"


@pytest.fixture
def guardrail():
    _metering_owner.reset_metering_owner()
    yield ReveniumGuardrail(guardrail_name="revenium", event_hook=["pre_call", "post_call"], default_on=True)
    _metering_owner.reset_metering_owner()


@pytest.fixture
def submit():
    with patch.object(gmod, "submit_ai_event") as mock, \
            patch.object(gmod, "run_async_in_thread", side_effect=drive_metering):
        yield mock


@pytest.fixture
def default_capture(monkeypatch):
    monkeypatch.delenv("REVENIUM_CAPTURE_ERROR_MESSAGES", raising=False)
    monkeypatch.delenv("REVENIUM_CAPTURE_PROMPTS", raising=False)


@pytest.fixture
def proxy_runs_in_mountain_time():
    """LiteLLM's start time is a naive local reading, so prove it is read as one off UTC."""
    original = os.environ.get("TZ")
    os.environ["TZ"] = PROXY_ZONE
    time.tzset()
    yield
    if original is None:
        os.environ.pop("TZ", None)
    else:
        os.environ["TZ"] = original
    time.tzset()


async def _record_of(guardrail, submit, error, request_data=None):
    await guardrail.async_post_call_failure_hook(
        request_data=request_data if request_data is not None else after_pre_call(guardrail_data()),
        original_exception=error,
        user_api_key_dict=make_key_dict(),
    )
    return submitted_args(submit)


def _rate_limited(message="Rate limit reached"):
    return litellm.RateLimitError(message=message, llm_provider="openai", model=MODEL)


def _refused(message=PROMPT_QUOTE):
    return litellm.BadRequestError(message=message, model=MODEL, llm_provider="openai")


def _started_at(**instants):
    """The request as the failure hook gets it: LiteLLM copied these instants onto it and removed its logging object."""
    data = after_pre_call(guardrail_data())
    data.update(instants)
    return data


@pytest.mark.usefixtures("default_capture")
class TestTheReasonAndCode:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("error, reason, code", [
        pytest.param(_rate_limited(), "RateLimitError (HTTP 429)", 429, id="rate-limited"),
        pytest.param(
            litellm.InternalServerError(message="boom", llm_provider="openai", model=MODEL),
            "InternalServerError (HTTP 500)", 500, id="server-error",
        ),
        pytest.param(_refused(), "BadRequestError (HTTP 400)", 400, id="bad-request"),
    ])
    async def test_a_provider_http_error_sends_its_class_and_status(self, guardrail, submit, error, reason, code):
        args = await _record_of(guardrail, submit, error)

        assert (args["stop_reason"], args["error_reason"], args["error_code"]) == ("ERROR", reason, code)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("error, reason", [
        pytest.param(litellm.Timeout(message="timed out", model=MODEL, llm_provider="openai"), "Timeout", id="timeout"),
        pytest.param(
            litellm.APIConnectionError(message="refused", llm_provider="openai", model=MODEL),
            "APIConnectionError", id="connection-refused",
        ),
        pytest.param(RuntimeError("router has no healthy deployment"), "RuntimeError", id="no-status"),
    ])
    async def test_a_failure_no_server_answered_sends_no_error_code(self, guardrail, submit, error, reason):
        """LiteLLM's 408 for a timeout and 500 for a refused connection were never an HTTP answer."""
        args = await _record_of(guardrail, submit, error)

        assert args["error_reason"] == reason
        assert "error_code" not in args

    @pytest.mark.asyncio
    @pytest.mark.parametrize("error", [
        pytest.param(_refused(), id="provider-error"),
        pytest.param(RuntimeError(PROMPT_QUOTE), id="plain-error"),
        pytest.param(HTTPException(status_code=400, detail={"matched_phrase": PROMPT_QUOTE}), id="guardrail-block"),
    ])
    async def test_by_default_no_message_text_is_sent(self, guardrail, submit, error):
        args = await _record_of(guardrail, submit, error)

        assert args["error_reason"] == type(error).__name__ + (" (HTTP 400)" if hasattr(error, "status_code") else "")
        assert "ZEBRA42" not in repr(args)


class TestMessageCapture:
    @pytest.mark.asyncio
    async def test_capture_on_sends_the_redacted_message(self, guardrail, submit, monkeypatch):
        monkeypatch.setenv("REVENIUM_CAPTURE_ERROR_MESSAGES", "true")
        monkeypatch.delenv("REVENIUM_CAPTURE_PROMPTS", raising=False)

        args = await _record_of(guardrail, submit, RuntimeError("no deployment answered ?key=AIzaFAKE-0001"))

        assert args["error_reason"] == "RuntimeError: no deployment answered ?key=[REDACTED]"

    @pytest.mark.asyncio
    async def test_a_provider_message_needs_prompt_capture_as_well(self, guardrail, submit, monkeypatch):
        monkeypatch.setenv("REVENIUM_CAPTURE_ERROR_MESSAGES", "true")
        monkeypatch.setenv("REVENIUM_CAPTURE_PROMPTS", "true")

        reason = (await _record_of(guardrail, submit, _rate_limited("Limit hit ?key=AIzaFAKE-0001")))["error_reason"]

        assert reason.startswith("RateLimitError: ")
        assert reason.endswith("?key=[REDACTED] (HTTP 429)")

    @pytest.mark.asyncio
    async def test_a_guardrail_refusal_quoting_the_request_needs_prompt_capture(self, guardrail, submit, monkeypatch):
        """LiteLLM's content filter puts the phrase it matched into the refusal's detail."""
        monkeypatch.setenv("REVENIUM_CAPTURE_ERROR_MESSAGES", "true")
        monkeypatch.delenv("REVENIUM_CAPTURE_PROMPTS", raising=False)
        refusal = HTTPException(status_code=400, detail={"error": "Content blocked", "matched_phrase": PROMPT_QUOTE})

        args = await _record_of(guardrail, submit, refusal)

        assert args["error_reason"] == "HTTPException (HTTP 400)"


@pytest.mark.usefixtures("default_capture", "proxy_runs_in_mountain_time")
class TestTheDuration:
    @pytest.mark.asyncio
    async def test_it_runs_from_litellms_recorded_start_to_the_failure(self, guardrail, submit):
        started = datetime.datetime.now() - datetime.timedelta(seconds=2)
        handed_off = started + datetime.timedelta(seconds=1)

        args = await _record_of(
            guardrail, submit, _rate_limited(), _started_at(start_time=started, first_api_call_start_time=handed_off))

        assert 2000 <= args["request_duration"] < 7000
        assert args["request_time"] == started.astimezone(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        assert args["time_to_first_token"] == 0
        assert args["completion_start_time"] == args["response_time"]

    @pytest.mark.asyncio
    async def test_without_a_start_it_runs_from_the_provider_handoff(self, guardrail, submit):
        """LiteLLM before 1.102.0 copies only the instant it first handed the call to a provider."""
        handed_off = datetime.datetime.now() - datetime.timedelta(seconds=1.5)

        args = await _record_of(guardrail, submit, _rate_limited(), _started_at(first_api_call_start_time=handed_off))

        assert 1500 <= args["request_duration"] < 6500

    @pytest.mark.asyncio
    @pytest.mark.parametrize("planted", [
        pytest.param({"start_time": "2020-01-01T00:00:00"}, id="json-from-the-body"),
        pytest.param({"proxy_server_request": {"arrival_time": time.time() - 3600}}, id="arrival-time-only"),
        pytest.param({}, id="refused-before-any-handoff-before-1.102"),
    ])
    async def test_without_a_usable_recorded_start_it_is_zero(self, guardrail, submit, planted):
        data = _started_at(**planted)

        args = await _record_of(guardrail, submit, _rate_limited(), data)

        assert args["request_duration"] == 0
        assert args["request_time"] == args["response_time"]

    @pytest.mark.asyncio
    async def test_a_recorded_start_in_the_future_is_not_used(self, guardrail, submit):
        """Read under the proxy's zone, which the class fixture has set by now."""
        data = _started_at(start_time=datetime.datetime.now() + datetime.timedelta(minutes=5))

        args = await _record_of(guardrail, submit, _rate_limited(), data)

        assert args["request_duration"] == 0
        assert args["request_time"] == args["response_time"]


@pytest.mark.usefixtures("default_capture")
class TestWhatStaysAsItWas:
    @pytest.mark.asyncio
    async def test_a_spending_cap_refusal_still_sends_no_record(self, guardrail, submit):
        """Recording it as COST_LIMIT is BACK-5844."""
        cap = BudgetExceededError("Team Budget breached", rule_name="Team Budget")
        try:
            raise gmod._budget_exception(cap, MODEL) from cap
        except HTTPException as refusal:
            await guardrail.async_post_call_failure_hook(
                request_data=after_pre_call(guardrail_data()),
                original_exception=refusal,
                user_api_key_dict=make_key_dict(),
            )

        assert submit.call_count == 0

    def test_content_filter_is_a_stop_reason_the_sdk_can_send(self):
        stop_reason = typing.get_type_hints(AICreateCompletionParams, include_extras=True)["stop_reason"]
        literal = next(arg for arg in typing.get_args(stop_reason) if typing.get_origin(arg) is typing.Annotated)

        assert "CONTENT_FILTER" in typing.get_args(typing.get_args(literal)[0])

    @pytest.mark.asyncio
    @pytest.mark.parametrize("block, reason", [
        pytest.param(
            HTTPException(status_code=400, detail={"error": "Violated OpenAI moderation policy"}),
            "HTTPException (HTTP 400)", id="content-guardrail-block",
        ),
        pytest.param(
            litellm.ContentPolicyViolationError(message="content_policy_violation", model=MODEL, llm_provider="azure"),
            "ContentPolicyViolationError (HTTP 400)", id="provider-content-policy",
        ),
        pytest.param(
            BlockedPiiEntityError(entity_type="CREDIT_CARD", guardrail_name="presidio"),
            "BlockedPiiEntityError (HTTP 400)", id="pii-block",
        ),
    ])
    async def test_a_content_block_is_recorded_as_an_error_naming_its_class(self, guardrail, submit, block, reason):
        """No signal tells a content block apart reliably on every supported LiteLLM, so none is guessed at."""
        args = await _record_of(guardrail, submit, block)

        assert (args["stop_reason"], args["error_reason"], args["error_code"]) == ("ERROR", reason, 400)


@patch.object(mw, "run_async_in_thread", side_effect=run_inline)
@patch.object(mw, "get_client", return_value=object())
@patch.object(mw, "submit_ai_event")
@pytest.mark.usefixtures("default_capture")
class TestTheDeprecatedCallback:
    """LiteLLM hands its failure event the exception in kwargs; response_obj is None."""

    @staticmethod
    def _record_of(mock_submit, error):
        _metering_owner.reset_metering_owner()
        kwargs = base_kwargs()
        kwargs["exception"] = error
        started = datetime.datetime.now()
        run_hook(mw.proxy_handler_instance.async_log_failure_event(
            kwargs, None, started, started + datetime.timedelta(milliseconds=400)))
        return submitted_args(mock_submit)

    def test_a_provider_http_error_sends_its_class_and_status(self, mock_submit, _get_client, _run):
        args = self._record_of(mock_submit, _rate_limited())

        assert (args["error_reason"], args["error_code"], args["request_duration"]) == (
            "RateLimitError (HTTP 429)", 429, 400)

    def test_a_timeout_sends_no_error_code(self, mock_submit, _get_client, _run):
        args = self._record_of(mock_submit, litellm.Timeout(message="timed out", model=MODEL, llm_provider="openai"))

        assert args["error_reason"] == "Timeout"
        assert "error_code" not in args

    def test_a_guardrail_refusal_quoting_the_request_needs_prompt_capture(
            self, mock_submit, _get_client, _run, monkeypatch):
        monkeypatch.setenv("REVENIUM_CAPTURE_ERROR_MESSAGES", "true")
        refusal = HTTPException(status_code=400, detail={"error": "Content blocked", "matched_phrase": PROMPT_QUOTE})

        args = self._record_of(mock_submit, refusal)

        assert args["error_reason"] == "HTTPException (HTTP 400)"

    def test_by_default_no_message_text_is_sent(self, mock_submit, _get_client, _run):
        args = self._record_of(mock_submit, _refused())

        assert args["error_reason"] == "BadRequestError (HTTP 400)"
        assert "ZEBRA42" not in repr(args)


REFUSED_ALIAS = "back-5842-refused"
GUARDRAIL_SOURCE = guardrail_proxy_tests.GUARDRAIL_SOURCE


@pytest.fixture(scope="module")
def proxy(tmp_path_factory):
    """A LiteLLM proxy loaded with the README's guardrail config and a deployment the upstream refuses."""
    upstream = start_upstream()
    config = guardrail_proxy_tests._readme_config(upstream, {})
    config["model_list"].append({
        "model_name": REFUSED_ALIAS,
        "litellm_params": {
            "model": "hosted_vllm/" + REFUSED_MODEL, "api_key": "sk-back-5842-fake", "api_base": api_base(upstream),
        },
    })
    config_path = tmp_path_factory.mktemp("back_5842") / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    callback_lists = {name: list(value) for name, value in guardrail_proxy_tests._callback_lists().items()}
    previous_router = proxy_server.llm_router
    asyncio.run(proxy_server.initialize(config=str(config_path)))
    try:
        yield TestClient(proxy_server.app)
    finally:
        for name, value in guardrail_proxy_tests._callback_lists().items():
            value[:] = callback_lists.get(name, [])
        proxy_server.llm_router = previous_router
        _metering_owner.reset_metering_owner()
        upstream.shutdown()


@pytest.mark.usefixtures("default_capture")
class TestARealProxyRefusal:
    """LiteLLM's own request processing, HTTP client and error mapping, against an upstream that answers 400."""

    def test_the_record_carries_the_reason_the_status_and_the_time_it_took(self, proxy, metered_rows):
        response = proxy.post("/v1/chat/completions", json={
            "model": REFUSED_ALIAS, "messages": [{"role": "user", "content": REFUSAL_QUOTE}],
        })
        assert response.status_code == 400, response.text

        rows = [row for row in metered_rows(GUARDRAIL_SOURCE) if row["middleware_source"] == GUARDRAIL_SOURCE]

        assert len(rows) == 1
        row = rows[0]
        assert (row["stop_reason"], row["error_reason"], row["error_code"]) == (
            "ERROR", "BadRequestError (HTTP 400)", 400)
        assert row["request_duration"] >= REFUSAL_DELAY_SECONDS * 1000
        assert row["time_to_first_token"] == 0
        assert "ZEBRA42" not in repr(row)
