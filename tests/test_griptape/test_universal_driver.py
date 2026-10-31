"""Provider detection and delegation of the universal drivers."""
import asyncio
import importlib
import json
import threading
import uuid
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("griptape")
attrs = pytest.importorskip("attrs")
httpx = pytest.importorskip("httpx")

from griptape.common import PromptStack  # noqa: E402

from revenium_middleware import BudgetExceededError  # noqa: E402
from revenium_middleware.griptape.universal_driver import (  # noqa: E402
    ReveniumDriver,
    ReveniumEmbeddingDriver,
)


def _bare(cls):
    """Instance without running __init__, to unit-test detection helpers."""
    return cls.__new__(cls)


@pytest.mark.parametrize(
    "model,provider",
    [
        ("gpt-4", "openai"),
        ("gpt-4o-mini", "openai"),
        ("claude-3-5-sonnet-20241022", "anthropic"),
        ("claude-sonnet-4-5-20250929", "anthropic"),
        ("claude-opus-4-1", "anthropic"),
        ("llama3.2", "ollama"),
        ("mistral", "ollama"),
        ("gemini-1.5-flash", "litellm"),
        ("command-r", "litellm"),  # unknown → litellm fallback
        # Provider-qualified (LiteLLM convention) names route to litellm even
        # when they contain direct-provider substrings like "gpt-" or "claude".
        ("azure/gpt-4", "litellm"),
        ("bedrock/anthropic.claude-3-sonnet", "litellm"),
        ("openrouter/deepseek/deepseek-chat", "litellm"),
    ],
)
def test_detect_provider_from_model(model, provider):
    assert _bare(ReveniumDriver)._detect_provider_from_model(model) == provider


def test_requires_model_or_base_driver():
    with pytest.raises(ValueError):
        ReveniumDriver()


def test_detect_provider_from_driver_class_name():
    class FakeOpenAiChatPromptDriver:
        pass

    class FakeAnthropicPromptDriver:
        pass

    d = _bare(ReveniumDriver)
    assert d._detect_provider_from_driver(FakeOpenAiChatPromptDriver()) == "openai"
    assert d._detect_provider_from_driver(FakeAnthropicPromptDriver()) == "anthropic"


def test_wrap_existing_unknown_driver_raises():
    class SomeRandomPromptDriver:
        pass

    d = _bare(ReveniumDriver)
    d.usage_metadata = {}
    with pytest.raises(ValueError):
        d._wrap_existing_driver(SomeRandomPromptDriver())


def test_getattr_delegates_to_wrapped_driver():
    d = _bare(ReveniumDriver)
    d._wrapped_driver = type("W", (), {"model": "gpt-4", "ping": lambda self: "pong"})()
    assert d.model == "gpt-4"
    assert d.ping() == "pong"


def test_getattr_without_wrapped_driver_raises():
    d = _bare(ReveniumDriver)
    d._wrapped_driver = None
    with pytest.raises(AttributeError):
        _ = d.anything


@pytest.mark.parametrize(
    "model,provider",
    [
        ("text-embedding-3-large", "openai"),
        ("voyage-2", "voyageai"),
        ("embed-english-v3.0", "cohere"),
        ("nomic-embed-text", "ollama"),
        ("something-unknown", "openai"),  # unknown → openai fallback
    ],
)
def test_detect_embedding_provider_from_model(model, provider):
    assert (
        _bare(ReveniumEmbeddingDriver)._detect_provider_from_embedding_model(model)
        == provider
    )


def test_create_embedding_driver_unsupported_provider_raises():
    d = _bare(ReveniumEmbeddingDriver)
    d.usage_metadata = {}
    with pytest.raises(ValueError):
        d._create_embedding_driver_for_provider("cohere", "embed-english-v3.0")


def test_wrap_existing_unsupported_embedding_driver_raises():
    class CohereEmbeddingDriver:
        pass

    d = _bare(ReveniumEmbeddingDriver)
    d.usage_metadata = {}
    with pytest.raises(ValueError):
        d._wrap_existing_embedding_driver(CohereEmbeddingDriver())


def test_agent_accepts_revenium_driver():
    from griptape.structures import Agent
    from revenium_middleware.griptape import ReveniumDriver

    driver = ReveniumDriver(model="gpt-4o-mini", api_key="test-key")
    agent = Agent(prompt_driver=driver)
    # The agent must actually hold the Revenium wrapper, not a substitute.
    assert agent.prompt_driver is driver
    assert isinstance(agent.prompt_driver, ReveniumDriver)


def test_wrap_existing_openai_driver_preserves_base_url_and_organization():
    from griptape.drivers.prompt.openai_chat_prompt_driver import OpenAiChatPromptDriver

    base_driver = OpenAiChatPromptDriver(
        model="gpt-4o-mini",
        api_key="test-key",
        base_url="https://gateway.internal/v1",
        organization="org-123",
    )
    d = ReveniumDriver(base_driver=base_driver)
    assert d.wrapped_driver.base_url == "https://gateway.internal/v1"
    assert d.wrapped_driver.organization == "org-123"


def test_wrap_existing_anthropic_driver_preserves_top_p():
    from griptape.drivers.prompt.anthropic import AnthropicPromptDriver

    base_driver = AnthropicPromptDriver(
        model="claude-3-5-sonnet-20241022",
        api_key="test-key",
        top_p=0.9,
    )
    d = ReveniumDriver(base_driver=base_driver)
    assert d.wrapped_driver.top_p == 0.9
    assert d.wrapped_driver.api_key == "test-key"


def test_env_proxy_url_does_not_change_griptape_provider(monkeypatch):
    """BACK-4347: the environment setting alone no longer reroutes Griptape agents to the gateway."""
    import revenium_middleware.griptape.litellm_driver as litellm_driver

    monkeypatch.setenv("LITELLM_PROXY_URL", "https://proxy.internal/chat/completions")
    monkeypatch.setattr(litellm_driver, "LITELLM_AVAILABLE", True)

    d = ReveniumDriver(model="gpt-4", api_key="test-key")

    assert d.provider == "openai"
    assert type(d.wrapped_driver).__name__ == "ReveniumOpenAiDriver"


def test_wrap_existing_unknown_embedding_driver_raises():
    # Unrecognized class names detect as the "unknown" provider, so wrapping
    # must raise rather than silently treating the driver as OpenAI or handing
    # back the original unmetered driver.
    class SomeRandomEmbeddingDriver:
        model = "text-embedding-3-small"

    d = _bare(ReveniumEmbeddingDriver)
    d.usage_metadata = {}
    with pytest.raises(ValueError, match="Cannot wrap SomeRandomEmbeddingDriver"):
        d._wrap_existing_embedding_driver(SomeRandomEmbeddingDriver())


FAKE_OPENAI_KEY = "openai-test-key-not-real"
FAKE_AZURE_KEY = "azure-test-key-not-real"
AZURE_HOST = "myres.openai.azure.com"
AZURE_SETTINGS = dict(
    model="gpt-4o",
    azure_deployment="dep1",
    azure_endpoint=f"https://{AZURE_HOST}",
    api_key=FAKE_AZURE_KEY,
    api_version="2024-06-01",
)
SERIALIZATION_FIELDS = ("type", "module_name")


def _copied_fields(driver_class):
    return [f for f in attrs.fields(driver_class) if f.init and f.alias not in SERIALIZATION_FIELDS]


def _assert_every_setting_kept(base_driver, wrapped):
    for field in _copied_fields(type(base_driver)):
        base_value = getattr(base_driver, field.name)
        wrapped_value = getattr(wrapped, field.name)
        if field.alias == "client":
            assert wrapped_value is base_value, "the caller's client must be reused, not rebuilt"
        elif field.alias == "ignored_exception_types":
            assert wrapped_value == (*base_value, BudgetExceededError), \
                "a wrap keeps the caller's retry exclusions and adds the spending-cap error"
        else:
            assert wrapped_value == base_value, f"{field.alias} was not kept by the wrap"


def _user_stack():
    stack = PromptStack()
    stack.add_user_message("hi")
    return stack


class _Wire:
    """Answers every OpenAI-compatible request with a canned chat completion."""

    def __init__(self):
        self.requests = []

    def respond(self, request):
        self.requests.append(request)
        response = httpx.Response(200, json={
            "id": f"chatcmpl-{uuid.uuid4().hex}", "object": "chat.completion", "created": 1, "model": "gpt-4o",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4}})
        response.request = request
        return response


@pytest.fixture
def wire(monkeypatch):
    stub = _Wire()
    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", lambda self, request: stub.respond(request))
    return stub


def _run_now(coroutine):
    thread = threading.Thread(target=lambda: asyncio.run(coroutine))
    thread.start()
    thread.join(timeout=10)
    return thread


@pytest.fixture
def records(monkeypatch):
    import revenium_middleware.openai  # noqa: F401  installs the OpenAI patches

    monkeypatch.setenv("REVENIUM_METERING_API_KEY", "hak_test_griptape")
    captured = []

    def record(operation, payload, *args, **kwargs):
        captured.append(payload)
        return MagicMock(id="recorded", status_code=201)

    with patch("revenium_middleware.openai.middleware.submit_ai_event", side_effect=record), \
            patch("revenium_middleware.openai.middleware.run_async_in_thread", side_effect=_run_now):
        yield captured


def test_wrapped_azure_driver_calls_azure_resource_with_azure_key(wire, records):
    from griptape.drivers.prompt.azure_openai_chat_prompt_driver import AzureOpenAiChatPromptDriver

    driver = ReveniumDriver(base_driver=AzureOpenAiChatPromptDriver(**AZURE_SETTINGS))
    driver.run(_user_stack())

    assert wire.requests, "the wrapped Azure driver sent no request"
    for request in wire.requests:
        assert request.url.host == AZURE_HOST
        assert request.url.path == "/openai/deployments/dep1/chat/completions"
        assert request.url.params.get("api-version") == "2024-06-01"
        assert request.headers.get("api-key") == FAKE_AZURE_KEY
        assert "authorization" not in request.headers


def test_wrapped_openai_driver_keeps_every_setting_and_client(wire, records):
    import openai
    from griptape.drivers.prompt.openai_chat_prompt_driver import OpenAiChatPromptDriver

    settings = dict(
        model="gpt-4o-mini", api_key=FAKE_OPENAI_KEY, base_url="http://gateway.stub.invalid/v1",
        organization="org-123", temperature=0.7, max_tokens=55, min_retry_delay=0, max_retry_delay=1,
        max_attempts=5, extra_params={"metadata": {"k": "v"}}, user="u", response_format={"type": "json_object"},
        seed=7, tool_choice="required", reasoning_effort="low", use_native_tools=False,
        structured_output_strategy="tool", parallel_tool_calls=False, modalities=["text"],
        audio={"voice": "verse", "format": "wav"},
    )
    client = openai.OpenAI(api_key=FAKE_OPENAI_KEY, base_url="http://gateway.stub.invalid/v1")

    streaming = OpenAiChatPromptDriver(stream=True, client=client, **settings)
    _assert_every_setting_kept(streaming, ReveniumDriver(base_driver=streaming).wrapped_driver)

    plain = OpenAiChatPromptDriver(stream=False, client=client, **settings)
    ReveniumDriver(base_driver=plain).run(_user_stack())
    body = json.loads(wire.requests[-1].content)
    assert wire.requests[-1].url.host == "gateway.stub.invalid"
    assert body["seed"] == 7
    assert body["response_format"] == {"type": "json_object"}


def test_wrapped_azure_driver_keeps_azure_settings_and_client():
    import openai
    from griptape.drivers.prompt.azure_openai_chat_prompt_driver import AzureOpenAiChatPromptDriver

    client = openai.AzureOpenAI(api_key=FAKE_AZURE_KEY, azure_endpoint=f"https://{AZURE_HOST}",
                                api_version="2024-06-01")
    base_driver = AzureOpenAiChatPromptDriver(
        **AZURE_SETTINGS, azure_ad_token="ad-token-not-real", azure_ad_token_provider=lambda: "ad-token-not-real",
        stream=True, seed=7, client=client,
    )

    wrapped = ReveniumDriver(base_driver=base_driver).wrapped_driver

    assert isinstance(wrapped, AzureOpenAiChatPromptDriver)
    _assert_every_setting_kept(base_driver, wrapped)


def test_wrapping_openai_named_subclass_with_extra_field_succeeds():
    from griptape.drivers.prompt.openai_chat_prompt_driver import OpenAiChatPromptDriver

    @attrs.define
    class TenantOpenAiChatPromptDriver(OpenAiChatPromptDriver):
        tenant: str = attrs.field(default="acme", kw_only=True)

    base_driver = TenantOpenAiChatPromptDriver(model="gpt-4o-mini", api_key=FAKE_OPENAI_KEY, seed=7)

    wrapped = ReveniumDriver(base_driver=base_driver).wrapped_driver

    assert wrapped.seed == 7
    assert not hasattr(wrapped, "tenant")


def _wrapped_openai(usage_metadata):
    from griptape.drivers.prompt.openai_chat_prompt_driver import OpenAiChatPromptDriver

    return ReveniumDriver(base_driver=OpenAiChatPromptDriver(model="gpt-4o-mini", api_key=FAKE_OPENAI_KEY),
                          usage_metadata=usage_metadata)


def _wrapped_azure(usage_metadata):
    from griptape.drivers.prompt.azure_openai_chat_prompt_driver import AzureOpenAiChatPromptDriver

    return ReveniumDriver(base_driver=AzureOpenAiChatPromptDriver(**AZURE_SETTINGS), usage_metadata=usage_metadata)


def _from_model_name(usage_metadata):
    return ReveniumDriver(model="gpt-4o-mini", api_key=FAKE_OPENAI_KEY, usage_metadata=usage_metadata)


@pytest.mark.parametrize("build,provider", [
    (_wrapped_openai, "OPENAI"),
    (_wrapped_azure, "Azure"),
    (_from_model_name, "OPENAI"),
])
def test_wrapped_driver_call_is_metered_with_caller_attribution(wire, records, build, provider):
    driver = build({"trace_id": "t-1", "subscriber": {"email": "a@example.com"}})

    driver.run(_user_stack())

    assert len(records) == 1
    assert records[0]["trace_id"] == "t-1"
    assert records[0]["subscriber"] == {"email": "a@example.com"}
    assert records[0]["provider"] == provider


GATEWAY_HOST = "corp-gateway.stub.invalid"
PUBLIC_HOST = "public.stub.invalid"
FAKE_GATEWAY_KEY = "sk-ant-fake-gateway"
FAKE_ENV_ANTHROPIC_KEY = "sk-ant-fake-environment"
OLLAMA_HOST = "ollama.stub.invalid"


def _named_after_hook(retry_state):
    return None


def _assert_no_field_left_at_its_default(base_driver, default_driver):
    """Guards the settings tests: a field left at its default could not show that the wrap kept it."""
    for field in _copied_fields(type(base_driver)):
        if field.alias == "client":
            continue
        assert getattr(base_driver, field.name) != getattr(default_driver, field.name), (
            f"{field.alias} is still at its default in the test's base driver")


def _anthropic_base_driver(**overrides):
    import anthropic
    from griptape.drivers.prompt.anthropic import AnthropicPromptDriver
    from griptape.tokenizers import SimpleTokenizer

    settings = dict(
        model="claude-opus-4-7",
        client=anthropic.Anthropic(api_key=FAKE_GATEWAY_KEY, base_url=f"http://{GATEWAY_HOST}"),
        api_key=FAKE_GATEWAY_KEY, stream=True, max_attempts=1, min_retry_delay=0, max_retry_delay=1,
        use_native_tools=False, extra_params={"metadata": {"user_id": "u-42"}}, max_tokens=55,
        tool_choice={"type": "any"}, structured_output_strategy="rule", temperature=0.33, top_p=0.8, top_k=7,
        after_hook=_named_after_hook, ignored_exception_types=(ValueError,),
        tokenizer=SimpleTokenizer(characters_per_token=3, max_input_tokens=1234, max_output_tokens=55),
    )
    settings.update(overrides)
    return AnthropicPromptDriver(**settings)


def _ollama_base_driver(**overrides):
    from griptape.drivers.prompt.ollama import OllamaPromptDriver
    from griptape.tokenizers import SimpleTokenizer

    settings = dict(
        model="llama3.2", host=f"http://{OLLAMA_HOST}", api_key="ollama-fake-key",
        headers={"X-Corp-Auth": "abc"}, stream=True, extra_params={"keep_alive": "5m"}, max_attempts=1,
        min_retry_delay=0, max_retry_delay=1, use_native_tools=False, options={"num_ctx": 2048}, max_tokens=77,
        structured_output_strategy="tool", temperature=0.33, after_hook=_named_after_hook,
        ignored_exception_types=(ValueError,),
        tokenizer=SimpleTokenizer(characters_per_token=3, max_input_tokens=1234, max_output_tokens=77),
    )
    settings.update(overrides)
    return OllamaPromptDriver(**settings)


def test_wrapped_anthropic_driver_keeps_every_setting_and_client():
    from griptape.drivers.prompt.anthropic import AnthropicPromptDriver

    base_driver = _anthropic_base_driver()
    _assert_no_field_left_at_its_default(base_driver, AnthropicPromptDriver(model="claude-default-probe"))

    wrapped = ReveniumDriver(base_driver=base_driver).wrapped_driver

    _assert_every_setting_kept(base_driver, wrapped)


def _stub_anthropic_transport(monkeypatch):
    import anthropic

    client_class = anthropic.DefaultHttpxClient
    response_class = importlib.import_module(client_class.__mro__[1].__module__.split(".")[0]).Response
    captured = []

    def respond(self, request, **kwargs):
        captured.append(request)
        return response_class(200, request=request, json={
            "id": "msg_1", "type": "message", "role": "assistant", "model": "claude-opus-4-7",
            "content": [{"type": "text", "text": "hi"}], "stop_reason": "end_turn", "stop_sequence": None,
            "usage": {"input_tokens": 3, "output_tokens": 1}})

    monkeypatch.setenv("ANTHROPIC_BASE_URL", f"http://{PUBLIC_HOST}")
    monkeypatch.setenv("ANTHROPIC_API_KEY", FAKE_ENV_ANTHROPIC_KEY)
    monkeypatch.setattr(client_class, "send", respond)
    return captured


def _anthropic_auth_headers(api_key):
    import anthropic

    return anthropic.Anthropic(api_key=api_key).auth_headers


def test_wrapped_anthropic_driver_calls_through_callers_client(monkeypatch):
    captured = _stub_anthropic_transport(monkeypatch)
    base_driver = _anthropic_base_driver(stream=False)

    ReveniumDriver(base_driver=base_driver, usage_metadata={"trace_id": "t-1"}).run(_user_stack())

    gateway_auth = _anthropic_auth_headers(FAKE_GATEWAY_KEY)
    assert captured, "the wrapped Anthropic driver sent no request"
    for request in captured:
        assert request.url.host == GATEWAY_HOST
        assert {name: request.headers.get(name) for name in gateway_auth} == gateway_auth
        assert json.loads(request.content)["metadata"] == {"user_id": "u-42"}
    assert all(request.url.host != PUBLIC_HOST for request in captured)


def test_wrapped_ollama_driver_keeps_every_setting():
    from griptape.drivers.prompt.ollama import OllamaPromptDriver

    base_driver = _ollama_base_driver()
    _assert_no_field_left_at_its_default(base_driver, OllamaPromptDriver(model="default-probe"))

    wrapped = ReveniumDriver(base_driver=base_driver).wrapped_driver

    _assert_every_setting_kept(base_driver, wrapped)


def _ollama_chat_reply(request):
    response = httpx.Response(200, json={
        "model": "llama3.2", "created_at": "2024-01-01T00:00:00Z", "done": True, "done_reason": "stop",
        "message": {"role": "assistant", "content": "hi"}, "prompt_eval_count": 3, "eval_count": 1})
    response.request = request
    return response


def test_wrapped_ollama_driver_sends_callers_auth_header(monkeypatch):
    requests_seen = []

    def respond(self, request):
        requests_seen.append(request)
        return _ollama_chat_reply(request)

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", respond)
    base_driver = _ollama_base_driver(api_key=None, headers={"Authorization": "Bearer ollama-fake-token"},
                                      stream=False)

    ReveniumDriver(base_driver=base_driver).run(_user_stack())

    assert requests_seen, "the wrapped Ollama driver sent no request"
    for request in requests_seen:
        assert request.url.host == OLLAMA_HOST
        assert request.headers.get("authorization") == "Bearer ollama-fake-token"


def test_wrapping_driver_subclass_with_extra_field_warns_naming_it(caplog):
    from griptape.drivers.prompt.anthropic import AnthropicPromptDriver

    @attrs.define
    class TenantAnthropicPromptDriver(AnthropicPromptDriver):
        tenant: str = attrs.field(default="acme-secret-value", kw_only=True)

    with caplog.at_level("WARNING", logger="revenium_middleware.griptape.universal_driver"):
        ReveniumDriver(base_driver=TenantAnthropicPromptDriver(model="claude-opus-4-7", api_key="sk-ant-fake"))

    warnings = [r for r in caplog.records if r.levelname == "WARNING"]
    assert len(warnings) == 1
    assert "tenant" in warnings[0].getMessage()
    assert "acme-secret-value" not in caplog.text


def _meter_anthropic(monkeypatch, captured_records):
    _stub_anthropic_transport(monkeypatch)
    import revenium_middleware.anthropic  # noqa: F401  installs the Anthropic patches
    monkeypatch.setenv("REVENIUM_METERING_API_KEY", "hak_test_griptape")

    def record(operation, payload, *args, **kwargs):
        captured_records.append(payload)
        return MagicMock(id="recorded", status_code=201)

    return [
        patch("revenium_middleware.anthropic.middleware.submit_ai_event", side_effect=record),
        patch("revenium_middleware.anthropic.middleware._safe_run_async_in_thread",
              side_effect=lambda f, *a, **k: _run_now(f(*a, **k))),
    ]


def _meter_ollama(monkeypatch, captured_records):
    import revenium_middleware.ollama  # noqa: F401  installs the Ollama patches
    monkeypatch.setenv("REVENIUM_METERING_API_KEY", "hak_test_griptape")
    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", lambda self, request: _ollama_chat_reply(request))

    def record(operation, payload, *args, **kwargs):
        captured_records.append(payload)
        return MagicMock(id="recorded", status_code=201)

    return [
        patch("revenium_middleware.ollama.middleware.submit_ai_event", side_effect=record),
        patch("revenium_middleware.ollama.middleware.run_async_in_thread", side_effect=_run_now),
    ]


@pytest.mark.parametrize("meter,build_base", [
    (_meter_anthropic, lambda: _anthropic_base_driver(stream=False)),
    (_meter_ollama, lambda: _ollama_base_driver(api_key=None, stream=False)),
], ids=["anthropic", "ollama"])
def test_wrapped_anthropic_and_ollama_calls_are_metered_with_caller_attribution(monkeypatch, meter, build_base):
    """Guard: passes before and after the fix."""
    captured_records = []
    patches = meter(monkeypatch, captured_records)
    for p in patches:
        p.start()
    try:
        driver = ReveniumDriver(base_driver=build_base(),
                                usage_metadata={"trace_id": "t-1", "subscriber": {"email": "a@example.com"}})
        driver.run(_user_stack())
    finally:
        for p in patches:
            p.stop()

    assert len(captured_records) == 1
    assert captured_records[0]["trace_id"] == "t-1"
    assert captured_records[0]["subscriber"] == {"email": "a@example.com"}


class _GatewayResponse:
    def raise_for_status(self):
        return None

    def json(self):
        return {"choices": [{"message": {"content": "gateway reply"}}]}


def test_env_proxy_url_keeps_openai_route_and_one_record(wire, records, monkeypatch):
    import revenium_middleware.griptape.litellm_driver as litellm_driver

    gateway_posts = []

    def gateway_post(url, **kwargs):
        gateway_posts.append(url)
        return _GatewayResponse()

    monkeypatch.setenv("LITELLM_PROXY_URL", "http://gateway.stub.invalid")
    monkeypatch.setattr(litellm_driver, "LITELLM_AVAILABLE", True)
    monkeypatch.setattr(litellm_driver.requests, "post", gateway_post)

    driver = ReveniumDriver(model="gpt-4o-mini", api_key=FAKE_OPENAI_KEY, usage_metadata={"trace_id": "t-1"})
    driver.run(_user_stack())

    assert gateway_posts == []
    assert wire.requests, "the OpenAI route sent no request"
    assert all(request.url.host == "api.openai.com" for request in wire.requests)
    assert len(records) == 1
    assert records[0]["trace_id"] == "t-1"
    assert driver.provider == "openai"


def test_proxy_argument_with_base_driver_warns_and_keeps_provider(caplog):
    from griptape.drivers.prompt.openai_chat_prompt_driver import OpenAiChatPromptDriver

    base_driver = OpenAiChatPromptDriver(model="gpt-4o-mini", api_key=FAKE_OPENAI_KEY)
    with caplog.at_level("INFO", logger="revenium_middleware.griptape.universal_driver"):
        d = ReveniumDriver(base_driver=base_driver, proxy_url="http://gateway.stub.invalid")

    assert d.provider == "openai"
    assert "forcing LiteLLM" not in caplog.text
    warnings = [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]
    assert len(warnings) == 1
    assert "proxy" in warnings[0] and "ignored" in warnings[0]


def test_explicit_proxy_url_routes_griptape_through_gateway(monkeypatch, caplog):
    """Guard: passes before and after the fix."""
    import revenium_middleware.griptape.litellm_driver as litellm_driver

    monkeypatch.setattr(litellm_driver, "LITELLM_AVAILABLE", True)

    with caplog.at_level("INFO", logger="revenium_middleware.griptape.universal_driver"):
        d = ReveniumDriver(model="gpt-4o-mini", proxy_url="http://gateway.stub.invalid")

    assert d.provider == "litellm"
    assert d.wrapped_driver.proxy_url == "http://gateway.stub.invalid"
    assert len([r for r in caplog.records if "forcing LiteLLM" in r.getMessage()]) == 1


def test_explicit_proxy_api_key_with_env_url_routes_through_gateway(monkeypatch):
    """Guard: passes before and after the fix."""
    import revenium_middleware.griptape.litellm_driver as litellm_driver

    monkeypatch.setenv("LITELLM_PROXY_URL", "http://env-gateway.stub.invalid")
    monkeypatch.setattr(litellm_driver, "LITELLM_AVAILABLE", True)

    d = ReveniumDriver(model="gpt-4o-mini", proxy_api_key="sk-fake-gateway")

    assert d.provider == "litellm"
    assert d.wrapped_driver.proxy_url == "http://env-gateway.stub.invalid"
