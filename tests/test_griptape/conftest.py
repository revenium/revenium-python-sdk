"""Fixtures for the Griptape spending-cap tests: a breached cap, a stub provider and attempt counters.

Griptape is imported inside the fixtures only, so collecting this conftest never
needs the optional extra.
"""
import pytest

BREACHED_BLOCK_RULE = {
    "ruleId": 1, "name": "griptape test cap", "threshold": 1.0, "currentValue": 2.5,
    "windowEnd": "2999-01-01T00:00:00Z", "action": "BLOCK", "breached": True, "shadowMode": False,
}

CHAT_COMPLETION = {
    "id": "chatcmpl-griptape-test", "object": "chat.completion", "created": 1, "model": "gpt-4o-mini",
    "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
}

EMBEDDING = {
    "object": "list", "model": "text-embedding-3-small",
    "data": [{"object": "embedding", "index": 0, "embedding": [0.1, 0.2]}],
    "usage": {"prompt_tokens": 1, "total_tokens": 1},
}


class StubProvider:
    """An OpenAI-compatible provider that counts every request it receives."""

    def __init__(self, status=200):
        self.status = status
        self.hits = 0

    def respond(self, request):
        import httpx

        self.hits += 1
        if self.status != 200:
            return httpx.Response(self.status, json={"error": {"message": "stub provider failure"}})
        return httpx.Response(200, json=EMBEDDING if request.url.path.endswith("/embeddings") else CHAT_COMPLETION)

    def client(self):
        import httpx
        import openai

        return openai.OpenAI(api_key="sk-test", max_retries=0,
                             http_client=httpx.Client(transport=httpx.MockTransport(self.respond)))


@pytest.fixture
def breached_cap(monkeypatch):
    """Circuit breaker on with one breached BLOCK rule seeded; no network, no poller."""
    from revenium_middleware._core import enforcement

    monkeypatch.setenv("REVENIUM_CIRCUIT_BREAKER_ENABLED", "true")
    monkeypatch.delenv("REVENIUM_BYPASS", raising=False)
    monkeypatch.delenv("REVENIUM_CB_FAIL_MODE", raising=False)
    monkeypatch.setattr(enforcement, "_load_cache_from_disk", lambda: None)
    monkeypatch.setattr(enforcement, "_ensure_poller_running", lambda: None)
    monkeypatch.setattr(enforcement, "_fetch_rules", lambda *args, **kwargs: None)
    monkeypatch.setattr(enforcement, "_cached_rules", [dict(BREACHED_BLOCK_RULE)])
    monkeypatch.setattr(enforcement, "_cached_org_unit_blocks", {})
    monkeypatch.setattr(enforcement, "_cached_org_unit_block_balances", {})
    monkeypatch.setattr(enforcement, "_cached_org_unit_warnings", {})
    monkeypatch.setattr(enforcement, "_cache_timestamp", enforcement.time.monotonic())
    monkeypatch.setattr(enforcement, "_cache_initialized", True)
    enforcement._begin_cache_generation()


@pytest.fixture
def stub_provider():
    return StubProvider()


@pytest.fixture
def failing_provider():
    return StubProvider(status=500)


@pytest.fixture
def attempts(monkeypatch):
    """Counts each attempt the OpenAI prompt and embedding drivers make."""
    from revenium_middleware.griptape import ReveniumOpenAiDriver, ReveniumOpenAiEmbeddingDriver

    counts = {"prompt": 0, "embedding": 0}
    base_params = ReveniumOpenAiDriver._base_params
    embedding_params = ReveniumOpenAiEmbeddingDriver._params

    def counted_base_params(self, prompt_stack):
        counts["prompt"] += 1
        return base_params(self, prompt_stack)

    def counted_embedding_params(self, chunk):
        counts["embedding"] += 1
        return embedding_params(self, chunk)

    monkeypatch.setattr(ReveniumOpenAiDriver, "_base_params", counted_base_params)
    monkeypatch.setattr(ReveniumOpenAiEmbeddingDriver, "_params", counted_embedding_params)
    return counts


@pytest.fixture
def retries():
    """Every retry Griptape schedules, recorded through the driver's after_hook."""
    return []


@pytest.fixture
def prompt_driver(retries):
    """Builds a Revenium OpenAI prompt driver for a provider, with no retry waits."""
    from revenium_middleware.griptape import ReveniumOpenAiDriver

    def build(provider):
        return ReveniumOpenAiDriver(model="gpt-4o-mini", client=provider.client(), min_retry_delay=0,
                                    max_retry_delay=0, after_hook=retries.append)

    return build


@pytest.fixture
def embedding_driver(retries):
    """Builds a Revenium OpenAI embedding driver for a provider, with no retry waits."""
    from revenium_middleware.griptape import ReveniumOpenAiEmbeddingDriver

    def build(provider):
        return ReveniumOpenAiEmbeddingDriver(model="text-embedding-3-small", client=provider.client(),
                                             min_retry_delay=0, max_retry_delay=0, after_hook=retries.append)

    return build
