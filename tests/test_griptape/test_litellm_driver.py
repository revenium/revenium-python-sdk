"""Tests for the Revenium LiteLLM Griptape driver."""
import pytest

pytest.importorskip("griptape")


def test_module_imports():
    """The driver module must import even when litellm is not installed."""
    import revenium_middleware.griptape.litellm_driver  # noqa: F401


def test_build_revenium_headers():
    """_build_revenium_headers returns x-revenium-* headers only (no Authorization)."""
    from revenium_middleware.griptape.litellm_driver import ReveniumLiteLLMDriver

    # Bypass __init__ so the test works without litellm installed
    driver = ReveniumLiteLLMDriver.__new__(ReveniumLiteLLMDriver)
    driver.usage_metadata = {
        "trace_id": "trace-123",
        "task_type": "unit-test",
        "subscriber": {"id": "sub-1", "email": "user@example.com"},
    }

    headers = driver._build_revenium_headers()

    assert headers["x-revenium-trace-id"] == "trace-123"
    assert headers["x-revenium-task-type"] == "unit-test"
    # Nested subscriber dict is flattened to the header the proxy
    # middleware actually reads; other subscriber fields are dropped.
    assert headers["x-revenium-subscriber-id"] == "sub-1"
    assert "x-revenium-subscriber" not in headers
    # Only x-revenium-* headers, never credentials
    assert "Authorization" not in headers
    assert all(key.startswith("x-revenium-") for key in headers)


def test_build_revenium_headers_empty_metadata():
    """No usage_metadata yields an empty header dict."""
    from revenium_middleware.griptape.litellm_driver import ReveniumLiteLLMDriver

    driver = ReveniumLiteLLMDriver.__new__(ReveniumLiteLLMDriver)
    driver.usage_metadata = {}

    assert driver._build_revenium_headers() == {}


def _make_driver(monkeypatch, **kwargs):
    """Construct a driver even without litellm installed (init guards on the flag)."""
    import revenium_middleware.griptape.litellm_driver as driver_module

    monkeypatch.setattr(driver_module, "LITELLM_AVAILABLE", True)
    return driver_module.ReveniumLiteLLMDriver(**kwargs)


def test_litellm_only_kwarg_routed_to_litellm_kwargs(monkeypatch):
    """A LiteLLM-only kwarg must not reach BasePromptDriver.__init__ (no TypeError)
    and must be stored for LiteLLM/proxy requests."""
    driver = _make_driver(
        monkeypatch,
        model="gemini/gemini-1.5-flash",
        api_version="2024-01-01",
    )

    assert driver.litellm_kwargs == {"api_version": "2024-01-01"}


def test_base_driver_kwarg_not_leaked_into_litellm_kwargs(monkeypatch):
    """A Griptape base-driver kwarg goes to the base driver only, not to LiteLLM params."""
    import inspect

    from griptape.drivers.prompt.base_prompt_driver import BasePromptDriver

    # Guard: `stream` must actually be a BasePromptDriver constructor parameter.
    assert "stream" in inspect.signature(BasePromptDriver.__init__).parameters

    driver = _make_driver(
        monkeypatch,
        model="gemini/gemini-1.5-flash",
        stream=True,
    )

    assert "stream" not in driver.litellm_kwargs
    assert driver.stream is True


@pytest.mark.parametrize(
    "proxy_url",
    [
        "ftp://example.com/chat/completions",
        "proxy.internal/chat",  # schemeless
        "ws://proxy.internal",
    ],
)
def test_invalid_proxy_url_scheme_raises(monkeypatch, proxy_url):
    """A proxy_url without an http(s) scheme is rejected at construction time."""
    with pytest.raises(ValueError):
        _make_driver(
            monkeypatch,
            model="gemini/gemini-1.5-flash",
            proxy_url=proxy_url,
        )


def test_https_proxy_url_accepted(monkeypatch):
    """An https proxy_url is accepted and the base URL strips /chat/completions."""
    driver = _make_driver(
        monkeypatch,
        model="gemini/gemini-1.5-flash",
        proxy_url="https://proxy.internal/chat/completions",
    )

    assert driver.proxy_base_url == "https://proxy.internal"


def test_mixed_case_https_proxy_url_accepted(monkeypatch):
    """urllib.parse.urlparse lowercases the scheme, so HTTPS:// is accepted."""
    driver = _make_driver(
        monkeypatch,
        model="gemini/gemini-1.5-flash",
        proxy_url="HTTPS://proxy.internal/chat/completions",
    )

    assert driver.proxy_base_url == "HTTPS://proxy.internal"


class _StubResponse:
    """Minimal stand-in for requests.Response as consumed by _make_proxy_call."""

    status_code = 200

    def raise_for_status(self):
        return None

    def json(self):
        return {"choices": [{"message": {"content": "stub reply"}}]}


def _patch_requests_post(monkeypatch):
    """Replace requests.post in the driver module's namespace, capturing kwargs."""
    import revenium_middleware.griptape.litellm_driver as driver_module

    captured = {}

    def fake_post(url, headers=None, data=None, timeout=None):
        captured["url"] = url
        captured["headers"] = headers
        captured["data"] = data
        captured["timeout"] = timeout
        return _StubResponse()

    monkeypatch.setattr(driver_module.requests, "post", fake_post)
    return captured


def test_proxy_call_omits_authorization_without_key(monkeypatch):
    """Without a proxy API key, no Authorization header (not 'Bearer None')."""
    monkeypatch.delenv("LITELLM_API_KEY", raising=False)
    captured = _patch_requests_post(monkeypatch)

    driver = _make_driver(
        monkeypatch,
        model="gemini/gemini-1.5-flash",
        proxy_url="https://proxy.internal/chat/completions",
    )
    message = driver._make_proxy_call([{"role": "user", "content": "hi"}])

    assert "Authorization" not in captured["headers"]
    assert message.to_text() == "stub reply"


def test_proxy_call_sends_bearer_with_key(monkeypatch):
    """With a proxy API key, the Authorization header carries the bearer token."""
    captured = _patch_requests_post(monkeypatch)

    driver = _make_driver(
        monkeypatch,
        model="gemini/gemini-1.5-flash",
        proxy_url="https://proxy.internal/chat/completions",
        proxy_api_key="sk-1",
    )
    driver._make_proxy_call([{"role": "user", "content": "hi"}])

    assert captured["headers"]["Authorization"] == "Bearer sk-1"


def test_proxy_call_omits_null_max_tokens(monkeypatch):
    """max_tokens is omitted from the body when left unset (None)."""
    import json

    captured = _patch_requests_post(monkeypatch)

    driver = _make_driver(
        monkeypatch,
        model="gemini/gemini-1.5-flash",
        proxy_url="https://proxy.internal/chat/completions",
    )
    assert driver.max_tokens is None
    driver._make_proxy_call([{"role": "user", "content": "hi"}])

    body = json.loads(captured["data"])
    assert "max_tokens" not in body


def test_proxy_call_includes_max_tokens_when_set(monkeypatch):
    """max_tokens is included in the body when explicitly set."""
    import json

    captured = _patch_requests_post(monkeypatch)

    driver = _make_driver(
        monkeypatch,
        model="gemini/gemini-1.5-flash",
        proxy_url="https://proxy.internal/chat/completions",
        max_tokens=128,
    )
    driver._make_proxy_call([{"role": "user", "content": "hi"}])

    body = json.loads(captured["data"])
    assert body["max_tokens"] == 128


def test_litellm_params_strip_auth_keys(monkeypatch):
    """Revenium auth keys never reach the usage_metadata sent to LiteLLM."""
    import revenium_middleware.griptape.litellm_driver as driver_module

    monkeypatch.delenv("LITELLM_PROXY_URL", raising=False)
    monkeypatch.setattr(driver_module, "CLIENT_MIDDLEWARE_AVAILABLE", True)

    driver = _make_driver(
        monkeypatch,
        model="gemini/gemini-1.5-flash",
        usage_metadata={"trace_id": "t", "revenium_api_key": "secret"},
    )

    params = driver._build_litellm_params(None)

    assert params["usage_metadata"] == {"trace_id": "t"}


def test_revenium_headers_strip_auth_keys(monkeypatch):
    """Revenium auth keys are never turned into x-revenium-* headers."""
    driver = _make_driver(
        monkeypatch,
        model="gemini/gemini-1.5-flash",
        usage_metadata={"trace_id": "t", "revenium_api_key": "secret"},
    )

    headers = driver._build_revenium_headers()

    assert headers == {"x-revenium-trace-id": "t"}
    assert "x-revenium-revenium-api-key" not in headers
    assert "secret" not in str(headers)


def test_proxy_call_redacts_authorization_in_logs(monkeypatch, caplog):
    """DEBUG logs never contain the proxy API key; Authorization is redacted."""
    import logging

    captured = _patch_requests_post(monkeypatch)

    driver = _make_driver(
        monkeypatch,
        model="gemini/gemini-1.5-flash",
        proxy_url="https://proxy.internal/chat/completions",
        proxy_api_key="sk-secret-123",
        usage_metadata={"trace_id": "t"},
    )

    with caplog.at_level(
        logging.DEBUG, logger="revenium_middleware.griptape.litellm_driver"
    ):
        driver._make_proxy_call([{"role": "user", "content": "hi"}])

    # The key is still sent on the wire...
    assert captured["headers"]["Authorization"] == "Bearer sk-secret-123"
    # ...but never appears in logs.
    assert "sk-secret-123" not in caplog.text
    assert "Bearer ***" in caplog.text


class _StubArtifact:
    """Minimal artifact stub with a .value attribute."""

    def __init__(self, value):
        self.value = value


class _StubContentItem:
    """Minimal TextMessageContent-style stub (.artifact.value shape)."""

    def __init__(self, value):
        self.artifact = _StubArtifact(value)


class _StubMessage:
    """Minimal Griptape message stub with a list-of-content-items shape."""

    def __init__(self, items):
        self.content = items


def test_extract_message_content_rejects_non_text(monkeypatch):
    """Non-text (e.g. bytes) content raises a clear ValueError, not a TypeError."""
    driver = _make_driver(monkeypatch, model="gemini/gemini-1.5-flash")

    message = _StubMessage([_StubContentItem(b"\x89PNG fake image bytes")])

    with pytest.raises(ValueError, match="text content only"):
        driver._extract_message_content(message)


def test_extract_message_content_joins_text(monkeypatch):
    """Multiple text content items are joined with a newline."""
    driver = _make_driver(monkeypatch, model="gemini/gemini-1.5-flash")

    message = _StubMessage([_StubContentItem("hello"), _StubContentItem("world")])

    assert driver._extract_message_content(message) == "hello\nworld"


def test_revenium_headers_flatten_subscriber_id(monkeypatch):
    """A nested subscriber object becomes the flat header the proxy middleware reads."""
    driver = _make_driver(
        monkeypatch,
        model="gemini/gemini-1.5-flash",
        usage_metadata={"subscriber": {"id": "user-123", "email": "u@example.com"}},
    )

    headers = driver._build_revenium_headers()

    assert headers == {"x-revenium-subscriber-id": "user-123"}
    assert "x-revenium-subscriber" not in headers
    assert "u@example.com" not in str(headers)
