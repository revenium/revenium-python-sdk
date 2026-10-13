"""Tests for the Revenium LiteLLM Griptape driver."""
import logging
import re

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
    # Nested subscriber dict is flattened to the headers the proxy
    # middleware actually reads.
    assert headers["x-revenium-subscriber-id"] == "sub-1"
    assert headers["x-revenium-subscriber-email"] == "user@example.com"
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
    """DEBUG logs never contain the proxy API key; Authorization and the subscriber email are redacted."""
    import logging

    captured = _patch_requests_post(monkeypatch)

    driver = _make_driver(
        monkeypatch,
        model="gemini/gemini-1.5-flash",
        proxy_url="https://proxy.internal/chat/completions",
        proxy_api_key="sk-secret-123",
        usage_metadata={"trace_id": "t", "subscriber": {"id": "jane@example.com"}},
    )

    with caplog.at_level(
        logging.DEBUG, logger="revenium_middleware.griptape.litellm_driver"
    ):
        driver._make_proxy_call([{"role": "user", "content": "hi"}])

    # The key is still sent on the wire...
    assert captured["headers"]["Authorization"] == "Bearer sk-secret-123"
    # ...but never appears in logs.
    assert "sk-secret-123" not in caplog.text
    assert "'Authorization'" in caplog.text
    assert "Injected usage metadata via headers" in caplog.text
    assert "Bearer" not in caplog.text
    assert captured["headers"]["x-revenium-subscriber-id"] == "jane@example.com"
    assert "jane@example.com" not in caplog.text


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


def test_revenium_headers_flatten_subscriber_id_and_email(monkeypatch):
    """A nested subscriber object becomes the flat headers the proxy middleware reads."""
    driver = _make_driver(
        monkeypatch,
        model="gemini/gemini-1.5-flash",
        usage_metadata={"subscriber": {"id": "user-123", "email": "u@example.com", "credential": "c"}},
    )

    headers = driver._build_revenium_headers()

    assert headers == {"x-revenium-subscriber-id": "user-123", "x-revenium-subscriber-email": "u@example.com"}
    assert "x-revenium-subscriber" not in headers


PROXY_ENV_VARS = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy")
DRIVER_LOGGER = "revenium_middleware.griptape.litellm_driver"


def _lowercased(headers):
    return {key.lower(): value for key, value in headers.items()}


def _revenium_headers_sent(captured):
    return {key: value for key, value in _lowercased(captured["headers"]).items() if key.startswith("x-revenium-")}


def _user_stack():
    from griptape.common import PromptStack

    stack = PromptStack()
    stack.add_user_message("hi")
    return stack


class _ProxyStub:
    """A real HTTP server on localhost that answers like the LiteLLM proxy and records request headers."""

    def __init__(self):
        import http.server
        import json
        import threading

        received = self.received = []

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_POST(self):
                self.rfile.read(int(self.headers.get("Content-Length", 0)))
                received.append({key.lower(): value for key, value in self.headers.items()})
                body = json.dumps({"choices": [{"message": {"content": "stub reply"}}]}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass

        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}/chat/completions"
        self._thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self._thread.start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()
        self._thread.join(timeout=5)


@pytest.fixture
def proxy_stub(monkeypatch):
    for name in PROXY_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("NO_PROXY", "127.0.0.1")
    monkeypatch.setenv("no_proxy", "127.0.0.1")
    monkeypatch.delenv("LITELLM_API_KEY", raising=False)
    stub = _ProxyStub()
    yield stub
    stub.close()


def test_proxy_headers_use_names_the_proxy_reads(monkeypatch):
    """camelCase keys and the nested subscriber email reach the header names the proxy reads."""
    captured = _patch_requests_post(monkeypatch)
    driver = _make_driver(
        monkeypatch,
        model="gemini/gemini-1.5-flash",
        proxy_url="https://proxy.internal/chat/completions",
        usage_metadata={
            "traceId": "t-1", "taskType": "triage", "agenticJobId": "job-1", "organizationName": "Acme",
            "subscriber": {"id": "u-1", "email": "dev@example.com"},
        },
    )

    driver._make_proxy_call([{"role": "user", "content": "hi"}])

    assert _revenium_headers_sent(captured) == {
        "x-revenium-trace-id": "t-1",
        "x-revenium-task-type": "triage",
        "x-revenium-agentic-job-id": "job-1",
        "x-revenium-organization-name": "Acme",
        "x-revenium-subscriber-id": "u-1",
        "x-revenium-subscriber-email": "dev@example.com",
    }


@pytest.mark.parametrize("metadata,expected", [
    pytest.param({"trace_id": "snake", "traceId": "camel"}, {"x-revenium-trace-id": "snake"}, id="snake-first"),
    pytest.param({"traceId": "camel", "trace_id": "snake"}, {"x-revenium-trace-id": "snake"}, id="camel-first"),
    pytest.param({"subscriber": {"email": "nested@example.com"}, "subscriber_email": "flat@example.com"},
                 {"x-revenium-subscriber-email": "nested@example.com"}, id="nested-subscriber-first"),
    pytest.param({"subscriberEmail": "flat@example.com", "subscriber": {"email": "nested@example.com"}},
                 {"x-revenium-subscriber-email": "nested@example.com"}, id="flat-subscriber-first"),
])
def test_proxy_headers_collision_rules(monkeypatch, metadata, expected):
    """Two spellings of one tag produce one header, chosen by rule rather than by dict order."""
    driver = _make_driver(monkeypatch, model="gemini/gemini-1.5-flash", usage_metadata=metadata)

    assert _lowercased(driver._build_revenium_headers()) == expected


UNSENDABLE_TAGS = [
    pytest.param({"organization_name": "株式会社"}, "x-revenium-organization-name", None, id="non-latin"),
    pytest.param({"trace_id": "t-1\n"}, "x-revenium-trace-id", "t-1", id="trailing-newline"),
    pytest.param({"task_type": "summarize\r\n"}, "x-revenium-task-type", "summarize", id="trailing-crlf"),
    pytest.param({"task_type": "a\r\nX-Injected: yes"}, "x-revenium-task-type", None, id="embedded-crlf"),
]


@pytest.mark.parametrize("metadata,header,expected", UNSENDABLE_TAGS)
def test_proxy_call_survives_unsendable_tag_values(monkeypatch, caplog, proxy_stub, metadata, header, expected):
    """A tag value no HTTP header can carry is dropped with a warning; the customer's call still succeeds."""
    import logging

    driver = _make_driver(monkeypatch, model="gpt-4o-mini", proxy_url=proxy_stub.url, usage_metadata=metadata)

    with caplog.at_level(logging.WARNING, logger=DRIVER_LOGGER):
        message = driver.try_run(_user_stack())

    assert message.to_text() == "stub reply"
    received = proxy_stub.received[-1]
    assert "x-injected" not in received
    assert received.get(header) == expected
    if expected is None:
        (key,) = metadata
        assert key in caplog.text
        assert metadata[key] not in caplog.text


@pytest.mark.parametrize("metadata,header,expected", UNSENDABLE_TAGS + [
    pytest.param({"organization_name": "Café"}, "x-revenium-organization-name", None, id="non-ascii"),
])
def test_streaming_proxy_headers_are_sendable_by_httpx(monkeypatch, metadata, header, expected):
    """The litellm.completion path hands its headers to httpx, which only sends ASCII."""
    httpx = pytest.importorskip("httpx")
    driver = _make_driver(
        monkeypatch, model="gpt-4o-mini", proxy_url="http://proxy.invalid/chat/completions",
        usage_metadata=metadata,
    )

    params = driver._build_litellm_params(None)
    request = httpx.Request("POST", "http://proxy.invalid/chat/completions", headers=params["extra_headers"])

    assert request.headers.get(header) == expected
    assert "x-injected" not in request.headers


UNSENDABLE_TAG_NAMES = [
    pytest.param("customer name", id="space"),
    pytest.param("cost/center", id="slash"),
    pytest.param("team:region", id="colon"),
    pytest.param("ab\tcd", id="tab"),
    pytest.param("ab\x01cd", id="control-byte"),
    pytest.param("\u00e9quipe", id="accented"),
    pytest.param("\u90e8\u7f72", id="japanese"),
    pytest.param("s\u212Aill_name", id="kelvin-sign"),
    pytest.param(1, id="integer"),
    pytest.param(("a", "b"), id="tuple"),
]

SENDABLE_HEADER_NAME = re.compile(r"x-revenium-[a-z0-9-]+")


def _all_sendable(header_names):
    return all(
        SENDABLE_HEADER_NAME.fullmatch(name.lower())
        for name in header_names
        if name.lower().startswith("x-revenium")
    )


@pytest.mark.parametrize("key", UNSENDABLE_TAG_NAMES)
def test_proxy_call_survives_unsendable_tag_names(monkeypatch, caplog, proxy_stub, key):
    """BACK-4260: a tag name no HTTP header can carry is skipped with a warning
    naming it; the customer's call still succeeds and the other tags still flow."""
    driver = _make_driver(
        monkeypatch, model="gpt-4o-mini", proxy_url=proxy_stub.url,
        usage_metadata={key: "Acme", "trace_id": "t-1"},
    )

    with caplog.at_level(logging.WARNING, logger=DRIVER_LOGGER):
        message = driver.try_run(_user_stack())

    assert message.to_text() == "stub reply"
    assert proxy_stub.received[-1]["x-revenium-trace-id"] == "t-1"
    assert _all_sendable(driver._build_revenium_headers(encoding="latin-1"))
    assert repr(key) in caplog.text
    assert "Acme" not in caplog.text


@pytest.mark.parametrize("key", UNSENDABLE_TAG_NAMES)
def test_streaming_proxy_headers_skip_unsendable_tag_names(monkeypatch, key):
    """The litellm.completion path builds its headers the same way, so httpx can send them."""
    httpx = pytest.importorskip("httpx")
    driver = _make_driver(
        monkeypatch, model="gpt-4o-mini", proxy_url="http://proxy.invalid/chat/completions",
        usage_metadata={key: "Acme", "trace_id": "t-1"},
    )

    params = driver._build_litellm_params(None)
    request = httpx.Request("POST", "http://proxy.invalid/chat/completions", headers=params["extra_headers"])

    assert request.headers.get("x-revenium-trace-id") == "t-1"
    assert _all_sendable(request.headers.keys())


def test_an_unsendable_tag_name_is_warned_about_once(monkeypatch, caplog, proxy_stub):
    """An agent loop makes many calls; the warning for one bad key must not repeat on each."""
    driver = _make_driver(
        monkeypatch, model="gpt-4o-mini", proxy_url=proxy_stub.url,
        usage_metadata={"customer name": "Acme", "trace_id": "t-1"},
    )

    with caplog.at_level(logging.WARNING, logger=DRIVER_LOGGER):
        driver.try_run(_user_stack())
        driver.try_run(_user_stack())

    assert len(proxy_stub.received) == 2
    assert caplog.text.count(repr("customer name")) == 1


def test_a_non_ascii_spelling_never_replaces_a_valid_tag(monkeypatch):
    """A key that only lowercases to a known tag name, such as one with the Kelvin sign,
    is skipped, so the caller's real tag still arrives."""
    driver = _make_driver(
        monkeypatch, model="gemini/gemini-1.5-flash",
        usage_metadata={"s\u212Aill_name": "forged", "skillName": "real"},
    )

    assert _lowercased(driver._build_revenium_headers())["x-revenium-skill-name"] == "real"


def test_proxy_headers_skip_none_values(monkeypatch):
    """A tag left as None sends no header rather than the text "None"."""
    driver = _make_driver(
        monkeypatch, model="gemini/gemini-1.5-flash", usage_metadata={"trace_id": None, "task_type": "triage"},
    )

    assert _lowercased(driver._build_revenium_headers()) == {"x-revenium-task-type": "triage"}


def test_proxy_debug_log_never_contains_tag_values(monkeypatch, caplog):
    """DEBUG logging names the headers sent, never their values (trace ids, emails)."""
    import logging

    captured = _patch_requests_post(monkeypatch)
    driver = _make_driver(
        monkeypatch,
        model="gemini/gemini-1.5-flash",
        proxy_url="https://proxy.internal/chat/completions",
        usage_metadata={"trace_id": "trace-value-123", "subscriber": {"id": "u-1", "email": "dev@example.com"}},
    )

    with caplog.at_level(logging.DEBUG, logger=DRIVER_LOGGER):
        driver._make_proxy_call([{"role": "user", "content": "hi"}])

    assert _revenium_headers_sent(captured)["x-revenium-trace-id"] == "trace-value-123"
    assert "x-revenium-trace-id" in caplog.text
    assert "trace-value-123" not in caplog.text
    assert "dev@example.com" not in caplog.text


def test_proxy_call_keeps_latin1_tag_values(monkeypatch, proxy_stub):
    """The direct proxy call sends latin-1 header text, so an accented tag still arrives intact."""
    driver = _make_driver(
        monkeypatch, model="gpt-4o-mini", proxy_url=proxy_stub.url, usage_metadata={"organization_name": "Café"},
    )

    driver.try_run(_user_stack())

    assert proxy_stub.received[-1]["x-revenium-organization-name"] == "Café"


@pytest.mark.parametrize("configured,posted", [
    ("https://proxy.internal", "https://proxy.internal/chat/completions"),
    ("https://proxy.internal/", "https://proxy.internal/chat/completions"),
    ("https://proxy.internal/v1", "https://proxy.internal/v1/chat/completions"),
    ("https://proxy.internal/v1?tenant=a", "https://proxy.internal/v1/chat/completions?tenant=a"),
    ("https://proxy.internal/chat/completions", "https://proxy.internal/chat/completions"),
    ("https://proxy.internal/chat/completions?tenant=a", "https://proxy.internal/chat/completions?tenant=a"),
    ("https://proxy.internal/chat/completions/", "https://proxy.internal/chat/completions"),
])
def test_proxy_call_posts_to_chat_completions_for_any_gateway_address(monkeypatch, configured, posted):
    captured = _patch_requests_post(monkeypatch)
    driver = _make_driver(monkeypatch, model="gpt-4o-mini", proxy_url=configured)

    driver._make_proxy_call([{"role": "user", "content": "hi"}])

    assert captured["url"] == posted


@pytest.mark.parametrize("configured,api_base", [
    ("https://proxy.internal", "https://proxy.internal"),
    ("https://proxy.internal/", "https://proxy.internal"),
    ("https://proxy.internal/v1/", "https://proxy.internal/v1"),
    ("https://proxy.internal/chat/completions", "https://proxy.internal"),
    ("https://proxy.internal/chat/completions/", "https://proxy.internal"),
    ("https://proxy.internal/chat/completions?tenant=a", "https://proxy.internal?tenant=a"),
])
def test_streaming_call_sends_litellm_the_gateway_base_for_any_gateway_address(monkeypatch, configured, api_base):
    import revenium_middleware.griptape.litellm_driver as driver_module

    sent = {}

    def fake_completion(**kwargs):
        sent.update(kwargs)
        return iter(())

    monkeypatch.setattr(driver_module.litellm, "completion", fake_completion)
    driver = _make_driver(monkeypatch, model="gpt-4o-mini", proxy_url=configured)

    list(driver.try_stream(_user_stack()))

    assert sent["api_base"] == api_base
