"""LiteLLM's own health checks are not metered; real calls still are, once.

``litellm.ahealth_check`` sends a real test call to every deployment (``GET
/health``, the Admin UI health page, ``POST /health/test_connection`` and the
background checks). Before this fix the client wrapper recorded each as a
``PYTHON`` row and the deprecated callback recorded it again as a ``PROXY`` row
under an organization named after LiteLLM's health-check service account.

The contract tests pin what LiteLLM writes on a probe and on a real call, so a
LiteLLM release that changes either turns the ``litellm-suite`` job red. The
rest drive a real in-process proxy under each integration and the client wrapper
directly. A test that expects no row sends a real call through the same path
afterwards and waits for its row first, so a row that is merely late cannot pass.
"""

import asyncio
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("litellm")
pytest.importorskip("fastapi")
pytest.importorskip("litellm.proxy.proxy_server")
yaml = pytest.importorskip("yaml")

import litellm  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from litellm.integrations.custom_logger import CustomLogger  # noqa: E402
from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER  # noqa: E402
from litellm.proxy import proxy_server  # noqa: E402

import revenium_middleware.openai  # noqa: E402,F401  patches the OpenAI SDK under LiteLLM's openai/ route
from revenium_middleware._core import metering_pool  # noqa: E402
from revenium_middleware.litellm.client import middleware as client_middleware  # noqa: E402
from revenium_middleware.litellm.proxy import _metering_owner  # noqa: E402
from revenium_middleware.litellm.proxy import middleware as proxy_middleware  # noqa: E402
from revenium_middleware.litellm.proxy._health_check import (  # noqa: E402
    LITELLM_HEALTH_CHECK_MARKER as MARKER,
    is_litellm_health_check,
)
from revenium_middleware.litellm.proxy.guardrail import ReveniumGuardrail  # noqa: E402

from . import test_proxy_client_wrapper_metering as guardrail_proxy_tests  # noqa: E402
from .fake_openai_upstream import api_base, start_upstream  # noqa: E402
from .test_proxy_callback_metering import _callback_config  # noqa: E402
from .test_proxy_client_wrapper_metering import (  # noqa: E402
    CHAT_ALIAS,
    CLIENT_WRAPPER_SOURCE,
    GUARDRAIL_SOURCE,
    MESSAGES,
    _callback_lists,
    _readme_config,
)

metered_rows = guardrail_proxy_tests.metered_rows

CALLBACK_SOURCE = "PROXY"
CALLBACK = "callback"
GUARDRAIL = "guardrail"
SOURCE_OF_A_PROXIED_CALL = {CALLBACK: CALLBACK_SOURCE, GUARDRAIL: GUARDRAIL_SOURCE}
PROBE_MODEL = "back-5781-probe-model"
CONTROL_MODEL = "gpt-4o-mini"
LOGGING_TIMEOUT_SECONDS = 10.0
FORGED_HEADERS = {"Authorization": "Bearer " + MARKER, "x-litellm-tags": MARKER}


class _CapturingLogger(CustomLogger):
    def __init__(self):
        super().__init__()
        self.containers = []

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        self.containers.append(kwargs.get("litellm_params") or {})

    async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
        self.containers.append(kwargs.get("litellm_params") or {})


def _flush_logging():
    asyncio.run(asyncio.wait_for(GLOBAL_LOGGING_WORKER.flush(), LOGGING_TIMEOUT_SECONDS))


@pytest.fixture(scope="module")
def upstream():
    server = start_upstream()
    yield server
    server.shutdown()


def _deployment_params(upstream, model, **extra):
    return {"model": model, "api_key": "sk-back-5781-fake", "api_base": api_base(upstream), **extra}


@pytest.fixture(scope="module", params=[CALLBACK, GUARDRAIL])
def integration(request, upstream, tmp_path_factory):
    """A LiteLLM proxy with no master key under each integration, as the README configures it."""
    if request.param == CALLBACK:
        config = _callback_config(upstream, {})
    else:
        config = _readme_config(upstream, {})
    config_path = tmp_path_factory.mktemp("back_5781") / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    callback_lists = {name: list(value) for name, value in _callback_lists().items()}
    previous_router = proxy_server.llm_router
    previous_master_key = proxy_server.master_key
    previous_handler = proxy_middleware._proxy_handler_instance
    proxy_middleware._proxy_handler_instance = None
    _metering_owner.reset_metering_owner()
    _flush_logging()
    asyncio.run(proxy_server.initialize(config=str(config_path)))
    proxy_server.master_key = None
    try:
        yield SimpleIntegration(request.param, TestClient(proxy_server.app))
    finally:
        for name, value in _callback_lists().items():
            value[:] = callback_lists.get(name, [])
        proxy_server.llm_router = previous_router
        proxy_server.master_key = previous_master_key
        proxy_middleware._proxy_handler_instance = previous_handler
        _metering_owner.reset_metering_owner()


class SimpleIntegration:
    def __init__(self, name, http):
        self.name = name
        self.http = http


@pytest.fixture
def capture(monkeypatch):
    logger = _CapturingLogger()
    monkeypatch.setattr(litellm, "callbacks", [*litellm.callbacks, logger])
    return logger


def _post(integration, path, body, headers=None):
    response = integration.http.post(path, json=body, headers=headers or {})
    assert response.status_code == 200, response.text
    return response


def _direct_completion():
    return litellm.completion(model=CONTROL_MODEL, messages=MESSAGES, mock_response="hello")


def _control_call(integration):
    """A real proxied call and, under the callback, a direct one: one row per writer."""
    _post(integration, "/v1/chat/completions", {"model": CHAT_ALIAS, "messages": MESSAGES})
    _direct_completion()


def _expected_control_sources(integration):
    return sorted([SOURCE_OF_A_PROXIED_CALL[integration.name], CLIENT_WRAPPER_SOURCE])


def _settled_sources(metered_rows, integration):
    rows = metered_rows(CLIENT_WRAPPER_SOURCE)
    for source in set(_expected_control_sources(integration)):
        rows = metered_rows(source, expected_count=_expected_control_sources(integration).count(source))
    return rows


def _assert_only_the_controls_were_metered(integration, metered_rows):
    _flush_logging()
    rows = _settled_sources(metered_rows, integration)
    assert sorted(row["middleware_source"] for row in rows) == _expected_control_sources(integration)
    assert MARKER not in {row.get("organization_name") for row in rows}


class TestLiteLLMsContract:
    def test_our_marker_is_litellms_health_service_account_name(self):
        from litellm.constants import LITTELM_INTERNAL_HEALTH_SERVICE_ACCOUNT_NAME
        from litellm.proxy._types import UserAPIKeyAuth

        service_account = UserAPIKeyAuth.get_litellm_internal_health_check_user_api_key_auth()

        assert MARKER == LITTELM_INTERNAL_HEALTH_SERVICE_ACCOUNT_NAME
        assert service_account.api_key == MARKER
        assert service_account.team_id == MARKER

    def test_litellms_probe_kwargs_are_recognised(self, upstream, monkeypatch):
        asked = []
        monkeypatch.setattr(
            client_middleware, "is_litellm_health_check", lambda container: asked.append(dict(container)) or False
        )

        asyncio.run(litellm.ahealth_check(_deployment_params(upstream, "hosted_vllm/" + PROBE_MODEL), mode="chat"))

        assert [is_litellm_health_check(container) for container in asked] == [True]

    def test_litellms_probe_logging_container_is_recognised(self, upstream, capture):
        _direct_completion()
        _flush_logging()
        capture.containers.clear()

        asyncio.run(litellm.ahealth_check(_deployment_params(upstream, "hosted_vllm/" + PROBE_MODEL), mode="chat"))
        _flush_logging()

        assert [is_litellm_health_check(container) for container in capture.containers] == [True]

    @pytest.mark.parametrize("stream", [False, True], ids=["messages", "messages-streamed"])
    def test_a_tagged_messages_call_logging_container_is_not_recognised(self, integration, capture, stream):
        _direct_completion()
        _flush_logging()
        capture.containers.clear()
        body = {
            "model": CHAT_ALIAS, "max_tokens": 16, "messages": MESSAGES, "stream": stream,
            "metadata": {"tags": [MARKER]},
        }

        _post(integration, "/v1/messages", body, {"x-litellm-tags": MARKER})
        _flush_logging()

        assert capture.containers
        assert [is_litellm_health_check(container) for container in capture.containers] == [False] * len(
            capture.containers
        )

    @pytest.mark.parametrize("stream", [False, True], ids=["messages", "messages-streamed"])
    def test_a_forged_service_key_on_an_open_proxy_is_still_metered_once(
        self, integration, metered_rows, stream
    ):
        body = {"model": CHAT_ALIAS, "max_tokens": 16, "messages": MESSAGES, "stream": stream}

        _post(integration, "/v1/messages", body, FORGED_HEADERS)

        rows = metered_rows(SOURCE_OF_A_PROXIED_CALL[integration.name])
        assert [row["middleware_source"] for row in rows] == [SOURCE_OF_A_PROXIED_CALL[integration.name]]


class TestTheClientWrapperDoesNotMeterAHealthCheck:
    def _health_check(self, params, mode):
        return asyncio.run(litellm.ahealth_check(params, mode=mode))

    def _assert_only_the_control_row(self, metered_rows):
        _direct_completion()
        rows = metered_rows(CLIENT_WRAPPER_SOURCE)
        assert [row["model"] for row in rows] == [CONTROL_MODEL]

    def test_a_chat_health_check_sends_no_row(self, upstream, metered_rows):
        self._health_check(_deployment_params(upstream, "hosted_vllm/" + PROBE_MODEL), "chat")

        self._assert_only_the_control_row(metered_rows)

    def test_an_embedding_health_check_sends_no_row(self, upstream, metered_rows):
        self._health_check(_deployment_params(upstream, "hosted_vllm/" + PROBE_MODEL), "embedding")

        self._assert_only_the_control_row(metered_rows)

    def test_a_responses_health_check_on_a_bridged_provider_sends_no_row(self, upstream, metered_rows):
        self._health_check(_deployment_params(upstream, "ovhcloud/" + PROBE_MODEL), "responses")

        self._assert_only_the_control_row(metered_rows)

    def test_a_failed_health_check_sends_no_row(self, metered_rows):
        params = {"model": "hosted_vllm/" + PROBE_MODEL, "api_key": "sk-back-5781-fake",
                  "api_base": "http://127.0.0.1:1", "num_retries": 0}

        result = self._health_check(params, "chat")

        assert "error" in result
        self._assert_only_the_control_row(metered_rows)

    def test_a_health_check_through_the_openai_sdk_sends_no_openai_row(self, upstream, mock_revenium_client, metered_rows):
        self._health_check(_deployment_params(upstream, "openai/" + PROBE_MODEL), "chat")

        self._assert_only_the_control_row(metered_rows)
        assert mock_revenium_client.ai.create_completion.call_count == 1

    def test_a_health_check_runs_no_spending_cap_check(self, upstream):
        with patch.object(client_middleware, "is_circuit_breaker_enabled", return_value=True), \
                patch.object(client_middleware, "is_bypass_enabled", return_value=False), \
                patch.object(client_middleware, "check_enforcement") as check:
            self._health_check(_deployment_params(upstream, "hosted_vllm/" + PROBE_MODEL), "chat")
            assert check.call_count == 0

            _direct_completion()
            assert check.call_count == 1

    def test_a_direct_call_beside_a_health_check_is_still_metered_once(self, upstream, metered_rows):
        async def both():
            probe = litellm.ahealth_check(_deployment_params(upstream, "hosted_vllm/" + PROBE_MODEL), mode="chat")
            call = litellm.acompletion(model=CONTROL_MODEL, messages=MESSAGES, mock_response="hello")
            return await asyncio.gather(probe, call)

        asyncio.run(both())

        rows = metered_rows(CLIENT_WRAPPER_SOURCE)
        assert [row["model"] for row in rows] == [CONTROL_MODEL]


@pytest.fixture
def probe_container(upstream, capture, mock_revenium_client):
    _direct_completion()
    _flush_logging()
    capture.containers.clear()
    metering_pool.wait_until_idle(5.0)
    mock_revenium_client.ai.create_completion.reset_mock()
    asyncio.run(litellm.ahealth_check(_deployment_params(upstream, "hosted_vllm/" + PROBE_MODEL), mode="chat"))
    _flush_logging()
    metering_pool.wait_until_idle(5.0)
    mock_revenium_client.ai.create_completion.reset_mock()
    assert len(capture.containers) == 1
    return capture.containers[0]


def _logging_kwargs(container, **extra):
    return {
        "model": PROBE_MODEL,
        "messages": MESSAGES,
        "litellm_params": container,
        "litellm_call_id": "call-back-5781",
        **extra,
    }


def _run(coro):
    return asyncio.run(coro)


@pytest.fixture
def callback_handler(monkeypatch):
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        handler = proxy_middleware.MiddlewareHandler()
    monkeypatch.setattr(_metering_owner, "guardrail_owns_metering", lambda: False)
    monkeypatch.setattr(proxy_middleware, "guardrail_owns_metering", lambda: False)
    return handler


class TestTheHandlersDoNotMeterAHealthCheck:
    def test_the_callback_sends_no_row_for_a_successful_health_check(
        self, callback_handler, probe_container, mock_revenium_client
    ):
        response = litellm.ModelResponse()

        _run(callback_handler.async_log_success_event(_logging_kwargs(probe_container), response, 0, 1))
        metering_pool.wait_until_idle(5.0)

        assert mock_revenium_client.ai.create_completion.call_count == 0

    def test_the_callback_sends_no_row_for_a_failed_health_check(
        self, callback_handler, probe_container, mock_revenium_client
    ):
        _run(callback_handler.async_log_failure_event(
            _logging_kwargs(probe_container, exception=RuntimeError("down")), None, 0, 1))
        metering_pool.wait_until_idle(5.0)

        assert mock_revenium_client.ai.create_completion.call_count == 0

    def test_the_guardrail_logging_event_sends_no_row_for_a_streamed_health_check(self, probe_container):
        _metering_owner.reset_metering_owner()
        guardrail = ReveniumGuardrail(
            guardrail_name="revenium", event_hook=["pre_call", "post_call"], default_on=True
        )
        kwargs = _logging_kwargs(probe_container, call_type="anthropic_messages", stream=True)
        try:
            with patch.object(ReveniumGuardrail, "_trusted_meter_key", MagicMock(return_value="k")) as meter_key:
                _run(guardrail.async_log_success_event(kwargs, None, 0, 1))
                assert meter_key.call_count == 0

                untagged = _logging_kwargs({"litellm_metadata": {}}, call_type="anthropic_messages", stream=True)
                _run(guardrail.async_log_success_event(untagged, None, 0, 1))
                assert meter_key.call_count == 1
        finally:
            _metering_owner.reset_metering_owner()


class TestAProxyHealthEndpointMetersNothing:
    def test_get_health_sends_no_row(self, integration, metered_rows):
        response = integration.http.get("/health")
        assert response.status_code == 200, response.text
        assert response.json()["healthy_count"] > 0

        _control_call(integration)

        _assert_only_the_controls_were_metered(integration, metered_rows)

    def test_test_connection_sends_no_row(self, integration, metered_rows, upstream, monkeypatch):
        body = {"litellm_params": _deployment_params(upstream, "hosted_vllm/" + PROBE_MODEL), "mode": "chat"}
        monkeypatch.setattr(proxy_server, "prisma_client", MagicMock())
        monkeypatch.setattr(proxy_server, "master_key", "sk-back-5781-master")

        response = _post(
            integration, "/health/test_connection", body, {"Authorization": "Bearer sk-back-5781-master"}
        )
        monkeypatch.undo()

        assert response.json()["status"] == "success"

        _control_call(integration)

        _assert_only_the_controls_were_metered(integration, metered_rows)


TAGGED_ROUTES = [
    pytest.param("/v1/chat/completions", {"model": CHAT_ALIAS, "messages": MESSAGES}, id="chat"),
    pytest.param("/v1/messages", {"model": CHAT_ALIAS, "max_tokens": 16, "messages": MESSAGES}, id="messages"),
    pytest.param(
        "/v1/messages",
        {"model": CHAT_ALIAS, "max_tokens": 16, "messages": MESSAGES, "stream": True},
        id="messages-streamed",
    ),
]
TAG_SOURCES = [
    pytest.param(lambda body: ({**body, "metadata": {"tags": [MARKER]}}, {}), id="body-metadata-tags"),
    pytest.param(lambda body: ({**body, "tags": [MARKER]}, {}), id="root-tags"),
    pytest.param(lambda body: (body, {"x-litellm-tags": MARKER}), id="header"),
]


class TestARealCallTaggedLikeAProbeIsStillMetered:
    @pytest.mark.parametrize("path, body", TAGGED_ROUTES)
    @pytest.mark.parametrize("tag", TAG_SOURCES)
    def test_a_real_call_tagged_like_a_probe_is_still_metered_once(self, integration, metered_rows, path, body, tag):
        tagged_body, headers = tag(body)

        _post(integration, path, tagged_body, headers)

        source = SOURCE_OF_A_PROXIED_CALL[integration.name]
        rows = metered_rows(source)
        assert [row["middleware_source"] for row in rows] == [source]
