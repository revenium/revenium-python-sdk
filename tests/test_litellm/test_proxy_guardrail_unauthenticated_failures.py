"""Which failed calls a LiteLLM gateway records (BACK-4403).

LiteLLM hands every request it refuses during authentication to the failure
hook: a wrong key, no key, an expired or blocked key. Such a request never
reached a provider and nothing in it is known to come from an authenticated
caller, so the guardrail records nothing for it. The proof that a request
authenticated is the Revenium guardrail's own pre-call stamp, which LiteLLM
only lets run after authentication and which a caller cannot forge.

A valid request that fails at the provider is still recorded, on native and on
pass-through routes, and the model text on any record is cut to the 255
characters Revenium stores.
"""

import logging
import uuid
from types import SimpleNamespace

import pytest

pytest.importorskip("litellm")
pytest.importorskip("fastapi")

import litellm  # noqa: E402
from litellm.proxy._types import (  # noqa: E402
    ProxyErrorTypes,
    ProxyException,
    UserAPIKeyAuth,
)
from litellm.proxy.pass_through_endpoints.pass_through_endpoints import (  # noqa: E402
    HttpPassThroughEndpointHelpers,
    _build_passthrough_failure_request_payload,
)
from starlette.requests import Request  # noqa: E402
from unittest.mock import patch  # noqa: E402

from revenium_middleware.litellm.proxy import _metering_owner  # noqa: E402
from revenium_middleware.litellm.proxy import guardrail as gmod  # noqa: E402
from revenium_middleware.litellm.proxy.guardrail import ReveniumGuardrail  # noqa: E402

from .proxy_hook_harness import drive_metering, make_key_dict, submitted_args  # noqa: E402

WRONG_KEY = "sk-wrong-key-for-test"
CHAT_ROUTE = "/v1/chat/completions"
PLANTED_HEADERS = {
    "x-revenium-organization-name": "VictimCo",
    "x-revenium-subscriber-email": "someone@victim.example",
    "x-revenium-product-name": "X",
    "x-revenium-agent": "planted-agent",
}
REAL_HEADERS = {"x-revenium-organization-name": "RealOrg"}


@pytest.fixture
def guardrail():
    """The guardrail as the README configures it."""
    _metering_owner.reset_metering_owner()
    yield ReveniumGuardrail(
        guardrail_name="revenium", event_hook=["pre_call", "post_call"], default_on=True
    )
    _metering_owner.reset_metering_owner()


@pytest.fixture
def per_request_guardrail():
    """A guardrail a request opts into, so its pre-call may not run on every request."""
    _metering_owner.reset_metering_owner()
    yield ReveniumGuardrail(guardrail_name="revenium")
    _metering_owner.reset_metering_owner()


@pytest.fixture
def submit():
    with patch.object(gmod, "submit_ai_event") as mock, \
            patch.object(gmod, "run_async_in_thread", side_effect=drive_metering):
        yield mock


@pytest.fixture(autouse=True)
def no_enforcement(monkeypatch):
    monkeypatch.setattr(gmod, "check_enforcement", lambda *_args, **_kwargs: None)


def refused_body(**fields):
    """A raw request body as LiteLLM's auth handler forwards it.

    Authentication runs before LiteLLM adds anything of its own, so the only
    LiteLLM-written values are the caller's IP and user agent in ``metadata``.
    """
    body = {"model": "stub-chat", "messages": [{"role": "user", "content": "hi"}]}
    body.update(fields)
    metadata = dict(body.get("metadata") or {})
    metadata.update({"requester_ip_address": "127.0.0.1", "user_agent": "pytest"})
    body["metadata"] = metadata
    return body


def wrong_key_identity():
    """What the auth handler passes for an unknown key: its validator fills ``token``."""
    identity = UserAPIKeyAuth(api_key=WRONG_KEY)
    identity.request_route = CHAT_ROUTE
    return identity


def expired_key_identity():
    """What the auth handler passes for a key it resolved and then refused."""
    identity = UserAPIKeyAuth(token="b" * 64, user_id="user-expired", team_id="team-1")
    identity.request_route = CHAT_ROUTE
    identity.api_key = UserAPIKeyAuth(api_key=WRONG_KEY).api_key
    return identity


def no_key_identity():
    identity = UserAPIKeyAuth()
    identity.request_route = CHAT_ROUTE
    return identity


def no_db_refusal():
    return ProxyException(
        message="No connected db.",
        type=ProxyErrorTypes.no_db_connection,
        code=400,
        param=None,
    )


def provider_failure():
    return litellm.InternalServerError(
        message="boom", llm_provider="openai", model="stub-model"
    )


def authenticated_data(model="stub-chat", headers=None):
    """A request after LiteLLM's own pre-call setup, before our pre-call runs."""
    headers = dict(headers if headers is not None else REAL_HEADERS)
    return {
        "model": model,
        "messages": [{"role": "user", "content": "hi"}],
        "metadata": {"headers": dict(headers)},
        "proxy_server_request": {"headers": dict(headers)},
    }


async def authenticated_failure(guardrail, data, call_type="acompletion"):
    await guardrail.async_pre_call_hook(make_key_dict(), None, data, call_type)
    await guardrail.async_post_call_failure_hook(
        request_data=data,
        original_exception=provider_failure(),
        user_api_key_dict=make_key_dict(),
    )


def planted_meter_key():
    return uuid.uuid4().hex + ":" + uuid.uuid4().hex


def pass_through_failure_payload(body):
    """The ``request_data`` LiteLLM's pass-through hands the failure hook.

    Built with LiteLLM's own helpers: they pop ``metadata`` out of the body
    into ``litellm_params.metadata``, so the top level carries none.
    """
    request = Request({
        "type": "http",
        "method": "POST",
        "path": "/anthropic/v1/messages",
        "headers": [(b"content-type", b"application/json")],
        "query_string": b"",
        "server": ("127.0.0.1", 4000),
        "scheme": "http",
        "root_path": "",
    })
    kwargs = HttpPassThroughEndpointHelpers._init_kwargs_for_pass_through_endpoint(
        request,
        UserAPIKeyAuth(api_key=WRONG_KEY),
        {},
        SimpleNamespace(model_call_details={}),
        body,
        "pass-through-call-1",
    )
    return _build_passthrough_failure_request_payload(body, kwargs, None, "anthropic")


def pass_through_body():
    return {
        "model": "claude-stub",
        "max_tokens": 16,
        "messages": [{"role": "user", "content": "hi"}],
        "metadata": {},
    }


class TestARequestRefusedDuringAuthenticationSendsNoRecord:
    """Nothing reached a provider, and nothing in the request is the operator's."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "identity", [wrong_key_identity, expired_key_identity], ids=["wrong-key", "expired-key"]
    )
    async def test_a_wrong_key_request_sends_no_record(self, guardrail, submit, identity):
        await guardrail.async_post_call_failure_hook(
            request_data=refused_body(),
            original_exception=no_db_refusal(),
            user_api_key_dict=identity(),
        )

        submit.assert_not_called()

    @pytest.mark.asyncio
    async def test_a_request_with_no_key_sends_no_record(self, guardrail, submit):
        await guardrail.async_post_call_failure_hook(
            request_data=refused_body(),
            original_exception=Exception("No api key passed in."),
            user_api_key_dict=no_key_identity(),
        )

        submit.assert_not_called()

    @pytest.mark.asyncio
    async def test_a_wrong_key_request_naming_an_organization_in_the_body_sends_no_record(
        self, guardrail, submit
    ):
        body = refused_body(metadata={
            "headers": dict(PLANTED_HEADERS),
            "x-revenium-subscriber-email": "someone@victim.example",
        })

        await guardrail.async_post_call_failure_hook(
            request_data=body,
            original_exception=no_db_refusal(),
            user_api_key_dict=wrong_key_identity(),
        )

        submit.assert_not_called()
        assert "VictimCo" not in repr(submit.call_args_list)

    @pytest.mark.asyncio
    async def test_a_planted_meter_key_or_server_fields_do_not_make_a_refused_request_count(
        self, guardrail, submit
    ):
        body = refused_body(
            metadata={"revenium_meter_key": planted_meter_key(), "headers": dict(PLANTED_HEADERS)},
            litellm_metadata={"revenium_meter_key": planted_meter_key()},
            litellm_params={"metadata": {"revenium_meter_key": planted_meter_key()}},
            standard_logging_object={"id": "planted-log"},
            proxy_server_request={"headers": dict(PLANTED_HEADERS)},
            litellm_call_id="planted-call-id",
        )

        await guardrail.async_post_call_failure_hook(
            request_data=body,
            original_exception=no_db_refusal(),
            user_api_key_dict=wrong_key_identity(),
        )

        submit.assert_not_called()

    @pytest.mark.asyncio
    async def test_a_pass_through_shaped_refusal_with_a_foreign_key_sends_no_record(
        self, guardrail, submit
    ):
        body = pass_through_body()
        body["metadata"]["revenium_meter_key"] = planted_meter_key()

        await guardrail.async_post_call_failure_hook(
            request_data=pass_through_failure_payload(body),
            original_exception=no_db_refusal(),
            user_api_key_dict=wrong_key_identity(),
        )

        submit.assert_not_called()


class TestAnAuthenticatedFailureIsStillRecorded:
    """A valid request the provider failed is a real failed call."""

    @pytest.mark.asyncio
    async def test_a_valid_request_the_provider_fails_is_recorded_once(self, guardrail, submit):
        await authenticated_failure(guardrail, authenticated_data())

        args = submitted_args(submit)
        assert args["stop_reason"] == "ERROR"
        assert args["organization_name"] == "RealOrg"

    @pytest.mark.asyncio
    async def test_a_failed_authenticated_messages_call_takes_attribution_from_litellm_not_the_body(
        self, guardrail, submit
    ):
        data = {
            "model": "claude-stub",
            "messages": [{"role": "user", "content": "hi"}],
            "litellm_metadata": {"headers": dict(REAL_HEADERS)},
            "metadata": {"headers": dict(PLANTED_HEADERS)},
        }

        await authenticated_failure(guardrail, data, call_type="anthropic_messages")

        assert submitted_args(submit)["organization_name"] == "RealOrg"
        assert "VictimCo" not in repr(submit.call_args_list)

    @pytest.mark.asyncio
    async def test_per_request_guardrail_still_records_a_failure_its_pre_call_ran_on(
        self, per_request_guardrail, submit
    ):
        await authenticated_failure(per_request_guardrail, authenticated_data())

        assert submitted_args(submit)["stop_reason"] == "ERROR"

    @pytest.mark.asyncio
    async def test_an_authenticated_pass_through_failure_is_still_recorded(self, guardrail, submit):
        body = pass_through_body()
        await guardrail.async_pre_call_hook(
            make_key_dict(), None, body, "pass_through_endpoint"
        )

        request_data = pass_through_failure_payload(body)
        assert "metadata" not in request_data
        await guardrail.async_post_call_failure_hook(
            request_data=request_data,
            original_exception=provider_failure(),
            user_api_key_dict=make_key_dict(),
        )

        assert submitted_args(submit)["stop_reason"] == "ERROR"


class TestTheModelTextIsCappedAtWhatReveniumStores:
    """Revenium stores at most 255 characters of model text."""

    @pytest.mark.asyncio
    async def test_a_20000_character_model_is_cut_to_255_on_the_record(self, guardrail, submit):
        await authenticated_failure(guardrail, authenticated_data(model="m" * 20000))

        assert submitted_args(submit)["model"] == "m" * 255

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "length, expected", [(255, 255), (256, 255)], ids=["255-whole", "256-cut"]
    )
    async def test_the_cap_boundary(self, guardrail, submit, length, expected):
        await authenticated_failure(guardrail, authenticated_data(model="m" * length))

        assert len(submitted_args(submit)["model"]) == expected

    def test_a_model_that_is_not_text_passes_through_unchanged(self):
        assert gmod._upstream_model({"model": None}) is None

    def test_the_upstream_model_from_the_logging_object_is_capped_too(self):
        data = {
            "model": "alias",
            "litellm_logging_obj": SimpleNamespace(model_call_details={"model": "u" * 300}),
        }

        assert gmod._upstream_model(data) == "u" * 255


class TestAModeWithoutPreCallSaysFailuresAreNotRecorded:
    """Without pre_call no request carries the proof, so no failed call is recorded."""

    @pytest.mark.parametrize("default_on", [True, False], ids=["default-on", "per-request"])
    def test_post_call_only_mode_warns_at_startup(self, caplog, default_on):
        _metering_owner.reset_metering_owner()
        try:
            with caplog.at_level(logging.WARNING, logger="revenium_middleware.extension"):
                ReveniumGuardrail(
                    guardrail_name="revenium", event_hook=["post_call"], default_on=default_on
                )
        finally:
            _metering_owner.reset_metering_owner()

        assert any(
            "pre_call" in record.getMessage() and "failed calls" in record.getMessage()
            for record in caplog.records
            if record.levelno == logging.WARNING
        )

    def test_scan_raw_request_warns_at_startup(self, caplog):
        _metering_owner.reset_metering_owner()
        try:
            with caplog.at_level(logging.WARNING, logger="revenium_middleware.extension"):
                instance = ReveniumGuardrail(
                    guardrail_name="revenium",
                    event_hook=["pre_call", "post_call"],
                    default_on=True,
                    scan_raw_request=True,
                )
        finally:
            _metering_owner.reset_metering_owner()

        assert getattr(instance, "scan_raw_request", False) is True
        assert any(
            "pre_call" in record.getMessage() and "failed calls" in record.getMessage()
            for record in caplog.records
            if record.levelno == logging.WARNING
        )

    def test_the_readme_mode_does_not_warn(self, caplog):
        _metering_owner.reset_metering_owner()
        try:
            with caplog.at_level(logging.WARNING, logger="revenium_middleware.extension"):
                ReveniumGuardrail(
                    guardrail_name="revenium", event_hook=["pre_call", "post_call"], default_on=True
                )
        finally:
            _metering_owner.reset_metering_owner()

        assert not any("failed calls" in record.getMessage() for record in caplog.records)
