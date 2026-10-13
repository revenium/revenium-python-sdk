"""A proxied call is metered once, by the guardrail, while direct calls stay metered (BACK-3918).

Importing ``ReveniumGuardrail`` loads ``revenium_middleware.litellm``, which
loads the client wrapper, which patches ``litellm.acompletion`` and its siblings
for the whole process. The proxy router calls those for every request, so before
BACK-3918 each proxied call sent two rows: the guardrail's (subscriber set, the
proxy's model naming) and the client wrapper's (no subscriber, the upstream's
model name). The backend keeps whichever lands first, so the stored model and
cost flipped with arrival order; on ``/v1/messages`` the two rows did not even
share a transaction id.

These tests drive a real LiteLLM proxy app in-process, configured exactly as the
README configures the guardrail, against a local OpenAI-compatible upstream, and
count rows at the metering client, which is where each one becomes a POST.

The client wrapper still meters every call the guardrail does not:

* direct ``litellm.completion`` / ``acompletion`` calls in the same process,
  which carry no ``proxy_server_request``;
* proxied streams whose end reaches none of the guardrail's metering hooks
  (``/v1/completions``, on litellm 1.104.2).

Standing aside must not cost the guardrail its token counts. It meters a stream
from the response LiteLLM assembles, which carries the provider's counts only if
the stream ended with a usage chunk. LiteLLM 1.104.0 asks for one by default,
but a proxy with ``always_include_stream_usage: false`` does not, so every test
here runs against both configurations, and the fake upstream, like OpenAI, sends
usage only when the request asks for it.
"""

import asyncio
import json
import threading
import time
import warnings
from types import SimpleNamespace
from unittest.mock import patch

import pytest

pytest.importorskip("litellm")
pytest.importorskip("fastapi")
yaml = pytest.importorskip("yaml")

import litellm  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from litellm.proxy import proxy_server  # noqa: E402

import revenium_middleware.openai  # noqa: E402,F401  patches the OpenAI SDK under LiteLLM's openai/ route
from revenium_middleware._core import metering_pool  # noqa: E402
from revenium_middleware.litellm.client import middleware as client_middleware  # noqa: E402
from revenium_middleware.litellm.proxy import _metering_owner  # noqa: E402
from revenium_middleware.litellm.proxy.guardrail import ReveniumGuardrail  # noqa: E402
from revenium_middleware.litellm.proxy.middleware import MiddlewareHandler  # noqa: E402

from .fake_openai_upstream import EMBEDDING_USAGE, USAGE, api_base, start_upstream  # noqa: E402

# hosted_vllm rather than openai: it keeps /v1/messages on the chat-completions
# bridge, which calls litellm.acompletion, where an OpenAI model would go to the
# Responses API that the client wrapper does not patch.
CHAT_ALIAS = "back-3918-chat"
# openai: LiteLLM reaches it through the openai SDK, which the SDK's OpenAI
# middleware (imported above) patches, so this route proves that middleware
# stands aside for a call the guardrail meters.
OPENAI_SDK_CHAT_ALIAS = "back-4249-openai-chat"
# ovhcloud has no native Responses API, so LiteLLM bridges /v1/responses to
# litellm.acompletion for it, and reaches it over its own HTTP client, passing
# stream_options through, so the upstream's usage chunk is the one metered.
RESPONSES_BRIDGE_ALIAS = "back-3918-responses-bridge"
# openai has a native Responses API, which LiteLLM serves without calling
# litellm.acompletion, so only the guardrail sees these calls.
RESPONSES_NATIVE_ALIAS = "back-4001-responses-native"
EMBEDDING_ALIAS = "back-3918-embedding"

GUARDRAIL_SOURCE = "GUARDRAIL"
CLIENT_WRAPPER_SOURCE = "PYTHON"
SUBSCRIBER_ID = "sub-back-3918"
SUBSCRIBER_HEADERS = {
    "x-revenium-subscriber-id": SUBSCRIBER_ID,
    "x-revenium-subscriber-email": "back-3918@example.com",
}
MESSAGES = [{"role": "user", "content": "hi"}]

# Long enough for the deferred stream dispatch and the metering pool on a
# loaded CI runner; a passing run returns as soon as the expected row lands.
ROW_TIMEOUT_SECONDS = 15.0
# The guardrail's row for a stream is submitted after the response is handed
# back, so a duplicate can trail the row the test waits for.
TRAILING_ROW_GRACE_SECONDS = 0.5


def _deployment(alias, model, upstream):
    return {
        "model_name": alias,
        "litellm_params": {"model": model, "api_key": "sk-back-3918-fake", "api_base": api_base(upstream)},
    }


STREAM_USAGE_SETTINGS = {
    "readme": {},
    "without-stream-usage": {"general_settings": {"always_include_stream_usage": False}},
}


def _readme_config(upstream, extra_settings):
    return {
        **extra_settings,
        "model_list": [
            _deployment(CHAT_ALIAS, "hosted_vllm/back-3918-model", upstream),
            _deployment(OPENAI_SDK_CHAT_ALIAS, "openai/back-3918-model", upstream),
            _deployment(RESPONSES_BRIDGE_ALIAS, "ovhcloud/back-3918-model", upstream),
            _deployment(RESPONSES_NATIVE_ALIAS, "openai/back-3918-model", upstream),
            _deployment(EMBEDDING_ALIAS, "hosted_vllm/back-3918-embedding-model", upstream),
        ],
        "guardrails": [
            {
                "guardrail_name": "revenium",
                "litellm_params": {
                    "guardrail": "revenium_middleware.litellm.proxy.guardrail.ReveniumGuardrail",
                    "mode": ["pre_call", "post_call"],
                    "default_on": True,
                },
            }
        ],
    }


def _callback_lists():
    return {
        name: getattr(litellm, name)
        for name in dir(litellm)
        if name.endswith(("callback", "callbacks")) and isinstance(getattr(litellm, name), list)
    }


@pytest.fixture(scope="module", params=sorted(STREAM_USAGE_SETTINGS))
def proxy(request, tmp_path_factory):
    """A LiteLLM proxy app loaded from the README's guardrail config."""
    upstream = start_upstream()
    config_path = tmp_path_factory.mktemp("back_3918") / "config.yaml"
    config_path.write_text(yaml.safe_dump(_readme_config(upstream, STREAM_USAGE_SETTINGS[request.param])))
    callback_lists = {name: list(value) for name, value in _callback_lists().items()}
    previous_router = proxy_server.llm_router
    asyncio.run(proxy_server.initialize(config=str(config_path)))
    try:
        yield TestClient(proxy_server.app)
    finally:
        for name, value in _callback_lists().items():
            value[:] = callback_lists.get(name, [])
        proxy_server.llm_router = previous_router
        _metering_owner.reset_metering_owner()
        upstream.shutdown()


@pytest.fixture
def metered_rows(mock_revenium_client):
    """Every row handed to the metering client, once the expected ones have landed."""

    def rows():
        return [call.kwargs for call in mock_revenium_client.ai.create_completion.call_args_list]

    def settle(expected_source, expected_count=1):
        deadline = time.monotonic() + ROW_TIMEOUT_SECONDS
        while time.monotonic() < deadline:
            metering_pool.wait_until_idle(0.5)
            if sum(row["middleware_source"] == expected_source for row in rows()) >= expected_count:
                break
            time.sleep(0.05)
        time.sleep(TRAILING_ROW_GRACE_SECONDS)
        metering_pool.wait_until_idle(ROW_TIMEOUT_SECONDS)
        return rows()

    return settle


def _post(proxy, path, body):
    response = proxy.post(path, json=body, headers=SUBSCRIBER_HEADERS)
    assert response.status_code == 200, response.text
    return response


def _streamed_chunks(response):
    events = (line[len("data: "):] for line in response.text.splitlines() if line.startswith("data: "))
    return [json.loads(event) for event in events if event != "[DONE]"]


CHAT_TOKENS = (USAGE["prompt_tokens"], USAGE["completion_tokens"])
EMBEDDING_TOKENS = (EMBEDDING_USAGE["prompt_tokens"], 0)
CALLER_ASKS_FOR_USAGE = {"stream_options": {"include_usage": True}}

PROXIED_ROUTES = [
    pytest.param("/v1/chat/completions", {"model": CHAT_ALIAS, "messages": MESSAGES}, CHAT_TOKENS, id="chat"),
    pytest.param(
        "/v1/chat/completions",
        {"model": CHAT_ALIAS, "messages": MESSAGES, "stream": True},
        CHAT_TOKENS,
        id="chat-streamed",
    ),
    pytest.param(
        "/v1/chat/completions",
        {"model": CHAT_ALIAS, "messages": MESSAGES, "stream": True, **CALLER_ASKS_FOR_USAGE},
        CHAT_TOKENS,
        id="chat-streamed-caller-asks-for-usage",
    ),
    pytest.param(
        "/v1/messages", {"model": CHAT_ALIAS, "max_tokens": 16, "messages": MESSAGES}, CHAT_TOKENS, id="messages"
    ),
    pytest.param(
        "/v1/messages",
        {"model": CHAT_ALIAS, "max_tokens": 16, "messages": MESSAGES, "stream": True},
        CHAT_TOKENS,
        id="messages-streamed",
    ),
    pytest.param("/v1/embeddings", {"model": EMBEDDING_ALIAS, "input": ["hi"]}, EMBEDDING_TOKENS, id="embeddings"),
    pytest.param(
        "/v1/responses", {"model": RESPONSES_BRIDGE_ALIAS, "input": "hi", "stream": True}, CHAT_TOKENS,
        id="responses-streamed",
    ),
    pytest.param(
        "/v1/responses", {"model": RESPONSES_NATIVE_ALIAS, "input": "hi", "stream": True}, CHAT_TOKENS,
        id="responses-streamed-native",
    ),
]


class TestAProxiedCallThroughTheOpenAISDKIsMeteredOnce:
    @pytest.mark.parametrize("body", [
        pytest.param({"model": OPENAI_SDK_CHAT_ALIAS, "messages": MESSAGES}, id="chat"),
        pytest.param({"model": OPENAI_SDK_CHAT_ALIAS, "messages": MESSAGES, "stream": True}, id="chat-streamed"),
    ])
    def test_one_row_and_it_is_the_guardrails(self, proxy, metered_rows, body):
        """The OpenAI middleware leaves the call to the guardrail (BACK-4249)."""
        _post(proxy, "/v1/chat/completions", body)

        rows = metered_rows(GUARDRAIL_SOURCE)

        assert [row["middleware_source"] for row in rows] == [GUARDRAIL_SOURCE]
        assert rows[0]["subscriber"]["id"] == SUBSCRIBER_ID


class TestProxiedCallIsMeteredOnceByTheGuardrail:
    @pytest.mark.parametrize("path, body, tokens", PROXIED_ROUTES)
    def test_one_row_and_it_is_the_guardrails(self, proxy, metered_rows, path, body, tokens):
        _post(proxy, path, body)

        rows = metered_rows(GUARDRAIL_SOURCE)

        assert [row["middleware_source"] for row in rows] == [GUARDRAIL_SOURCE]
        assert rows[0]["subscriber"]["id"] == SUBSCRIBER_ID
        assert (rows[0]["input_token_count"], rows[0]["output_token_count"]) == tokens

    def test_a_stream_the_caller_sent_without_stream_options_reaches_them_without_a_usage_chunk(self, proxy):
        response = _post(proxy, "/v1/chat/completions", {"model": CHAT_ALIAS, "messages": MESSAGES, "stream": True})

        assert [chunk for chunk in _streamed_chunks(response) if chunk.get("usage")] == []

    def test_a_stream_the_caller_asked_usage_for_still_ends_with_it(self, proxy):
        body = {"model": CHAT_ALIAS, "messages": MESSAGES, "stream": True, **CALLER_ASKS_FOR_USAGE}

        response = _post(proxy, "/v1/chat/completions", body)

        usage_chunks = [chunk["usage"] for chunk in _streamed_chunks(response) if chunk.get("usage")]
        assert [(usage["prompt_tokens"], usage["completion_tokens"]) for usage in usage_chunks] == [CHAT_TOKENS]

    def test_a_burst_of_calls_sends_one_row_per_call(self, proxy, metered_rows):
        calls = 6
        for _ in range(calls):
            _post(proxy, "/v1/chat/completions", {"model": CHAT_ALIAS, "messages": MESSAGES})

        rows = metered_rows(GUARDRAIL_SOURCE, expected_count=calls)

        assert len(rows) == calls
        assert len({row["transaction_id"] for row in rows}) == calls
        assert {row["middleware_source"] for row in rows} == {GUARDRAIL_SOURCE}


UPSTREAM_MODEL_BY_ROUTE = {
    "chat": "back-3918-model",
    "chat-streamed": "back-3918-model",
    "chat-streamed-caller-asks-for-usage": "back-3918-model",
    "messages": "back-3918-model",
    "messages-streamed": "back-3918-model",
    "embeddings": "back-3918-embedding-model",
    "responses-streamed": "back-3918-model",
    "responses-streamed-native": "back-3918-model",
}


class TestTheRowNamesTheDeploymentModel:
    """The proxy's model alias is not in Revenium's model catalog, so a row named after it is priced at $0."""

    @pytest.mark.parametrize(
        "path, body, upstream_model",
        [pytest.param(path, body, UPSTREAM_MODEL_BY_ROUTE[route.id], id=route.id)
         for route in PROXIED_ROUTES for path, body, _tokens in [route.values]],
    )
    def test_the_row_names_the_upstream_model_not_the_proxy_alias(self, proxy, metered_rows, path, body, upstream_model):
        _post(proxy, path, body)

        rows = metered_rows(GUARDRAIL_SOURCE)

        assert [row["model"] for row in rows] == [upstream_model]


class TestCallsOnlyTheClientWrapperMetersStayMetered:
    """Proxied streams the guardrail never sees the end of.

    If LiteLLM starts routing one of these to a guardrail hook, these fail with
    two rows, and ``_STREAMED_CALL_TYPES_METERED_HERE`` should gain its call type.
    """

    @pytest.mark.parametrize(
        "path, body",
        [
            pytest.param(
                "/v1/completions", {"model": CHAT_ALIAS, "prompt": "hi", "stream": True}, id="completions-streamed"
            ),
        ],
    )
    def test_still_one_row_from_the_client_wrapper(self, proxy, metered_rows, path, body):
        _post(proxy, path, body)

        rows = metered_rows(CLIENT_WRAPPER_SOURCE)

        assert [row["middleware_source"] for row in rows] == [CLIENT_WRAPPER_SOURCE]


def _direct_completion():
    return litellm.completion(model="gpt-4o-mini", messages=MESSAGES, mock_response="hello")


def _direct_acompletion():
    return asyncio.run(litellm.acompletion(model="gpt-4o-mini", messages=MESSAGES, mock_response="hello"))


def _direct_streamed_acompletion():
    async def consume():
        stream = await litellm.acompletion(model="gpt-4o-mini", messages=MESSAGES, mock_response="hello", stream=True)
        return [chunk async for chunk in stream]

    return asyncio.run(consume())


class TestDirectCallsBesideTheProxyAreStillMetered:
    """A background job or callback in the proxy's process calls LiteLLM directly."""

    @pytest.mark.parametrize(
        "call",
        [
            pytest.param(_direct_completion, id="completion"),
            pytest.param(_direct_acompletion, id="acompletion"),
            pytest.param(_direct_streamed_acompletion, id="acompletion-streamed"),
        ],
    )
    def test_one_row_from_the_client_wrapper(self, proxy, metered_rows, call):
        assert _metering_owner.guardrail_owns_metering()

        call()

        rows = metered_rows(CLIENT_WRAPPER_SOURCE)
        assert [row["middleware_source"] for row in rows] == [CLIENT_WRAPPER_SOURCE]


def _proxied_kwargs(**overrides):
    return {
        "model": "gpt-4o-mini",
        "messages": MESSAGES,
        "proxy_server_request": {"url": "http://proxy/v1/chat/completions", "body": {}},
        "metadata": {},
        **overrides,
    }


def _run_in_own_thread(coro):
    thread = threading.Thread(target=asyncio.run, args=(coro,))
    thread.start()
    thread.join()
    return thread


@pytest.fixture
def client_wrapper_rows():
    """Every row the client wrapper submits, collected as soon as it is built."""
    recorded = []
    with patch.object(client_middleware, "run_async_in_thread", side_effect=_run_in_own_thread), \
            patch.object(client_middleware, "submit_ai_event", side_effect=lambda _op, args: recorded.append(args)):
        yield recorded


@pytest.fixture
def configured_callback(monkeypatch):
    """The deprecated callback, listed in ``litellm.callbacks`` as a proxy config puts it."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        handler = MiddlewareHandler()
    monkeypatch.setattr(litellm, "callbacks", [*litellm.callbacks, handler])
    return handler


class TestMetersRequest:
    """The per-call answer the client wrapper acts on, for cases a proxy run cannot stage."""

    @pytest.fixture
    def guardrail(self):
        _metering_owner.reset_metering_owner()
        guardrail = ReveniumGuardrail(
            guardrail_name="revenium", event_hook=["pre_call", "post_call"], default_on=True
        )
        yield guardrail
        _metering_owner.reset_metering_owner()

    def test_a_proxied_chat_call_is_the_guardrails(self, guardrail):
        assert _metering_owner.guardrail_meters_request(_proxied_kwargs()) is True

    def test_a_key_opted_out_of_the_guardrail_keeps_the_client_wrapper(self, guardrail):
        opted_out = {"user_api_key_metadata": {"opted_out_global_guardrails": ["revenium"]}}

        assert _metering_owner.guardrail_meters_request(_proxied_kwargs(metadata=opted_out)) is False

    def test_a_vendor_error_keeps_the_client_wrapper(self, guardrail, monkeypatch):
        def broken(*_args, **_kwargs):
            raise AttributeError("renamed upstream")

        monkeypatch.setattr(guardrail, "should_run_guardrail", broken)

        assert _metering_owner.guardrail_meters_request(_proxied_kwargs()) is False

    def test_no_owner_means_the_client_wrapper_meters(self):
        _metering_owner.reset_metering_owner()

        assert _metering_owner.guardrail_meters_request(_proxied_kwargs()) is False

    @pytest.mark.parametrize(
        "kwargs, rows_from_the_client_wrapper",
        [
            pytest.param(_proxied_kwargs(stream=True), 0, id="proxied"),
            pytest.param({**_proxied_kwargs(stream=True), "proxy_server_request": None}, 1, id="direct"),
        ],
    )
    def test_a_registered_callback_owns_proxied_calls_only(
        self, configured_callback, client_wrapper_rows, kwargs, rows_from_the_client_wrapper
    ):
        _metering_owner.reset_metering_owner()

        _call_sync(kwargs)

        assert len(client_wrapper_rows) == rows_from_the_client_wrapper

    def test_a_guardrail_outranks_the_callback(self, guardrail, configured_callback, client_wrapper_rows):
        """The guardrail never meters a streamed ``/v1/completions`` call, so the client wrapper must."""
        streamed_text_completion = _proxied_kwargs(
            stream=True, litellm_logging_obj=SimpleNamespace(call_type="atext_completion")
        )

        _call_sync(streamed_text_completion)

        assert len(client_wrapper_rows) == 1


def _chunk(usage=None, content=None, finish_reason=None):
    choices = [] if usage else [SimpleNamespace(delta=SimpleNamespace(content=content), finish_reason=finish_reason)]
    return SimpleNamespace(usage=usage, choices=choices)


PROVIDER_CHUNKS = [
    _chunk(content="hi"),
    _chunk(finish_reason="stop"),
    _chunk(usage=SimpleNamespace(prompt_tokens=USAGE["prompt_tokens"], completion_tokens=USAGE["completion_tokens"])),
]


class _RecordingEntryPoint:
    """Stands in for litellm.completion / acompletion and keeps the kwargs it was sent."""

    def __init__(self):
        self.kwargs = None

    def __call__(self, *_args, **kwargs):
        self.kwargs = kwargs
        return iter(PROVIDER_CHUNKS)

    async def acall(self, *_args, **kwargs):
        self.kwargs = kwargs

        async def stream():
            for chunk in PROVIDER_CHUNKS:
                yield chunk

        return stream()


def _guardrail_stream_kwargs(**overrides):
    return _proxied_kwargs(stream=True, litellm_logging_obj=SimpleNamespace(call_type="acompletion"), **overrides)


def _call_sync(kwargs):
    entry_point = _RecordingEntryPoint()
    chunks = list(client_middleware.completion_wrapper(entry_point, None, (), kwargs))
    return entry_point.kwargs, chunks


def _call_async(kwargs):
    entry_point = _RecordingEntryPoint()

    async def consume():
        stream = await client_middleware.acompletion_wrapper(entry_point.acall, None, (), kwargs)
        return [chunk async for chunk in stream]

    chunks = asyncio.run(consume())
    return entry_point.kwargs, chunks


class TestAStreamTheGuardrailMetersKeepsItsUsage:
    """Standing aside from metering still asks the provider for the usage chunk the guardrail meters from.

    A proxy with ``always_include_stream_usage: false`` sends a caller's stream
    without ``stream_options``; without this request the guardrail's assembled
    response has no provider counts. The caller still receives exactly what it
    asked for.
    """

    @pytest.fixture(autouse=True)
    def guardrail(self):
        _metering_owner.reset_metering_owner()
        yield ReveniumGuardrail(guardrail_name="revenium", event_hook=["pre_call", "post_call"], default_on=True)
        _metering_owner.reset_metering_owner()

    @pytest.mark.parametrize("call", [_call_sync, _call_async], ids=["completion", "acompletion"])
    def test_the_provider_is_asked_for_usage_and_the_caller_does_not_see_it(self, call, mock_revenium_client):
        sent, chunks = call(_guardrail_stream_kwargs())

        assert sent["stream_options"] == {"include_usage": True}
        assert [chunk.usage for chunk in chunks] == [None, None]
        assert mock_revenium_client.ai.create_completion.call_count == 0

    @pytest.mark.parametrize("call", [_call_sync, _call_async], ids=["completion", "acompletion"])
    def test_a_caller_who_asked_for_usage_still_gets_it(self, call):
        sent, chunks = call(_guardrail_stream_kwargs(stream_options={"include_usage": True}))

        assert sent["stream_options"] == {"include_usage": True}
        assert chunks[-1].usage.prompt_tokens == USAGE["prompt_tokens"]
