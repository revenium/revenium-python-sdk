"""A proxy still on the deprecated callback records each proxied call once, with the caller's attribution (BACK-3990).

Resolving the callback string loads the client wrapper, which patches
``litellm.acompletion`` process-wide, and the proxy router calls that for every
request. Before BACK-3990 only the guardrail told the client wrapper to stand
aside, so a callback proxy sent two rows per call: the callback's (subscriber,
organization and product from the ``x-revenium-*`` headers) and the client
wrapper's (none of them). On ``/v1/messages`` streams the two did not even
share a transaction id.

The proxy here is a separate module from the guardrail's because both
configure the one process-wide LiteLLM proxy, and a module-scoped proxy fixture
stays configured until its module ends.
"""

import asyncio

import pytest

pytest.importorskip("litellm")
pytest.importorskip("fastapi")
yaml = pytest.importorskip("yaml")

import litellm  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER  # noqa: E402
from litellm.proxy import proxy_server  # noqa: E402

from revenium_middleware.litellm.proxy import _metering_owner  # noqa: E402
from revenium_middleware.litellm.proxy import middleware as proxy_middleware  # noqa: E402

from . import test_proxy_client_wrapper_metering as guardrail_proxy_tests  # noqa: E402
from .fake_openai_upstream import start_upstream  # noqa: E402
from .test_proxy_client_wrapper_metering import (  # noqa: E402
    CHAT_ALIAS,
    CHAT_TOKENS,
    CLIENT_WRAPPER_SOURCE,
    EMBEDDING_ALIAS,
    EMBEDDING_TOKENS,
    MESSAGES,
    STREAM_USAGE_SETTINGS,
    _callback_lists,
    _direct_completion,
    _readme_config,
)

metered_rows = guardrail_proxy_tests.metered_rows

CALLBACK_SOURCE = "PROXY"
CALLBACK_PATH = "revenium_middleware.litellm.proxy.middleware.proxy_handler_instance"
EARLIER_LOGGING_TIMEOUT_SECONDS = 10.0
SUBSCRIBER_EMAIL = "back-3990@example.com"
ORGANIZATION = "AcmeCorp"
PRODUCT = "Support"
ATTRIBUTION_HEADERS = {
    "x-revenium-subscriber-email": SUBSCRIBER_EMAIL,
    "x-revenium-organization-name": ORGANIZATION,
    "x-revenium-product-name": PRODUCT,
}


def _callback_config(upstream, extra_settings):
    config = _readme_config(upstream, extra_settings)
    del config["guardrails"]
    config["litellm_settings"] = {"callbacks": CALLBACK_PATH}
    return config


def _finish_logging_of_earlier_calls():
    """Run the success logging LiteLLM deferred for calls earlier tests made.

    LiteLLM queues each call's success logging and runs the queue on the next
    event loop, reading the callback lists only then. Left queued, the earlier
    calls would reach the callback this module lists and add rows to its tests.
    """
    asyncio.run(asyncio.wait_for(GLOBAL_LOGGING_WORKER.flush(), EARLIER_LOGGING_TIMEOUT_SECONDS))


@pytest.fixture(scope="module", params=sorted(STREAM_USAGE_SETTINGS))
def callback_proxy(request, tmp_path_factory):
    """A LiteLLM proxy app loaded with the deprecated callback instead of the guardrail."""
    upstream = start_upstream()
    config_path = tmp_path_factory.mktemp("back_3990") / "config.yaml"
    config_path.write_text(yaml.safe_dump(_callback_config(upstream, STREAM_USAGE_SETTINGS[request.param])))
    callback_lists = {name: list(value) for name, value in _callback_lists().items()}
    previous_router = proxy_server.llm_router
    previous_handler = proxy_middleware._proxy_handler_instance
    proxy_middleware._proxy_handler_instance = None
    _metering_owner.reset_metering_owner()
    _finish_logging_of_earlier_calls()
    asyncio.run(proxy_server.initialize(config=str(config_path)))
    try:
        yield TestClient(proxy_server.app)
    finally:
        for name, value in _callback_lists().items():
            value[:] = callback_lists.get(name, [])
        proxy_server.llm_router = previous_router
        proxy_middleware._proxy_handler_instance = previous_handler
        upstream.shutdown()


def _post(proxy, path, body):
    response = proxy.post(path, json=body, headers=ATTRIBUTION_HEADERS)
    assert response.status_code == 200, response.text
    return response


CALLBACK_METERED_ROUTES = [
    pytest.param("/v1/chat/completions", {"model": CHAT_ALIAS, "messages": MESSAGES}, id="chat"),
    pytest.param(
        "/v1/chat/completions", {"model": CHAT_ALIAS, "messages": MESSAGES, "stream": True}, id="chat-streamed"
    ),
    pytest.param("/v1/completions", {"model": CHAT_ALIAS, "prompt": "hi", "stream": True}, id="completions-streamed"),
    pytest.param("/v1/messages", {"model": CHAT_ALIAS, "max_tokens": 16, "messages": MESSAGES}, id="messages"),
    pytest.param(
        "/v1/messages",
        {"model": CHAT_ALIAS, "max_tokens": 16, "messages": MESSAGES, "stream": True},
        id="messages-streamed",
    ),
]


class TestTheDeprecatedCallbackIsTheOnlyMeter:
    @pytest.mark.parametrize("path, body", CALLBACK_METERED_ROUTES)
    def test_one_row_and_it_carries_the_callers_attribution(self, callback_proxy, metered_rows, path, body):
        _post(callback_proxy, path, body)

        rows = metered_rows(CALLBACK_SOURCE)

        assert [row["middleware_source"] for row in rows] == [CALLBACK_SOURCE]
        row = rows[0]
        assert row["subscriber"]["email"] == SUBSCRIBER_EMAIL
        assert row["organization_name"] == ORGANIZATION
        assert row["product_name"] == PRODUCT
        assert (row["input_token_count"], row["output_token_count"]) == CHAT_TOKENS

    def test_an_embedding_is_recorded_once(self, callback_proxy, metered_rows):
        """The callback has never metered embeddings, so the client wrapper keeps that row."""
        _post(callback_proxy, "/v1/embeddings", {"model": EMBEDDING_ALIAS, "input": ["hi"]})

        rows = metered_rows(CLIENT_WRAPPER_SOURCE)

        assert len(rows) == 1
        assert (rows[0]["input_token_count"], rows[0]["output_token_count"]) == EMBEDDING_TOKENS


class TestDirectCallsBesideTheCallbackProxy:
    """A background job in the callback proxy's process calls LiteLLM directly."""

    def test_the_client_wrapper_still_meters_it_once(self, callback_proxy, metered_rows):
        assert any(isinstance(callback, proxy_middleware.MiddlewareHandler) for callback in litellm.callbacks)

        _direct_completion()

        rows = metered_rows(CLIENT_WRAPPER_SOURCE)
        assert [row["middleware_source"] for row in rows].count(CLIENT_WRAPPER_SOURCE) == 1
