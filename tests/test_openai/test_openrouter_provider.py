"""OpenAI-client calls sent to OpenRouter are recorded and capped as OpenRouter (BACK-5421)."""
import httpx
import pytest
from openai import AsyncAzureOpenAI, AsyncOpenAI, AzureOpenAI, OpenAI

import revenium_middleware.perplexity  # noqa: F401  installs the Perplexity integration
from revenium_middleware.openai.provider import detect_provider, get_provider_metadata
from entry_points import calls, perplexity_cost

from .test_openai_sdk_entry_points import (  # noqa: F401  payloads is a fixture
    CHAT_REQUEST,
    NON_STREAMED,
    STREAMED_CHAT,
    STREAMED_RESPONSES,
    FakeOpenAI,
    _run_coro_to_completion,
    payloads,
)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_LABELS = {"provider": "OPENROUTER", "model_source": "OPENROUTER"}
EMBEDDING_MODEL = "openai/text-embedding-3-small"


def labels_for(client):
    return get_provider_metadata(detect_provider(client, client.base_url))


@pytest.mark.parametrize("base_url", [
    pytest.param(OPENROUTER_BASE_URL, id="openrouter"),
    pytest.param("https://eu.openrouter.ai/api/v1", id="subdomain"),
    pytest.param("https://OpenRouter.AI/api/v1", id="mixed-case"),
])
def test_an_openai_client_on_openrouter_is_labelled_openrouter(base_url):
    assert labels_for(OpenAI(api_key="x", base_url=base_url)) == OPENROUTER_LABELS


@pytest.mark.parametrize("base_url", [
    pytest.param("https://openrouter.ai.example.com/v1", id="openrouter-as-a-subdomain-of-another-host"),
    pytest.param("https://notopenrouter.ai/v1", id="host-ending-in-the-name"),
    pytest.param("https://proxy.example.com/openrouter.ai/v1", id="name-in-the-path"),
])
def test_a_host_that_only_contains_the_openrouter_name_is_not_openrouter(base_url):
    assert labels_for(OpenAI(api_key="x", base_url=base_url))["provider"] == "OPENAI"


@pytest.mark.parametrize("client, labels", [
    pytest.param(OpenAI(api_key="x", base_url="https://api.openai.com/v1"),
                 {"provider": "OPENAI", "model_source": "OPENAI"}, id="openai"),
    pytest.param(AzureOpenAI(api_key="x", azure_endpoint="https://example.openai.azure.com", api_version="2024-10-21"),
                 {"provider": "Azure", "model_source": "OPENAI"}, id="azure"),
    pytest.param(OpenAI(api_key="x", base_url="http://localhost:11434/v1"),
                 {"provider": "OLLAMA", "model_source": "OLLAMA"}, id="ollama"),
    pytest.param(OpenAI(api_key="x", base_url="https://api.x.ai/v1"),
                 {"provider": "XAI", "model_source": "XAI"}, id="xai"),
])
def test_other_providers_keep_their_labels(client, labels):
    assert labels_for(client) == labels


class FakeOpenAICompatible(FakeOpenAI):
    """Serves chat, Responses and embeddings calls on ``base_url``."""

    def __init__(self, base_url, **options):
        super().__init__(**options)
        self._base_url = base_url

    def __call__(self, request):
        if request.url.path.endswith("/embeddings"):
            return httpx.Response(200, json={
                "object": "list", "model": EMBEDDING_MODEL,
                "data": [{"object": "embedding", "index": 0, "embedding": [0.1]}],
                "usage": {"prompt_tokens": 3, "total_tokens": 3},
            })
        return super().__call__(request)

    def sync_client(self):
        return OpenAI(api_key="stub", base_url=self._base_url,
                      http_client=httpx.Client(transport=httpx.MockTransport(self)))

    def async_client(self):
        return AsyncOpenAI(api_key="stub", base_url=self._base_url,
                           http_client=httpx.AsyncClient(transport=httpx.MockTransport(self._async_handler())))


def sync_embeddings_create(fake, _options):
    return fake.sync_client().embeddings.create(model=EMBEDDING_MODEL, input="hi")


def async_embeddings_create(fake, _options):
    async def call():
        return await fake.async_client().embeddings.create(model=EMBEDDING_MODEL, input="hi")
    _run_coro_to_completion(call())


EVERY_METERED_CALL = [*NON_STREAMED, *STREAMED_CHAT, *STREAMED_RESPONSES, sync_embeddings_create,
                      async_embeddings_create]


def the_one_record(recorded):
    assert len(recorded) == 1
    return recorded[0]


@pytest.mark.parametrize("call", EVERY_METERED_CALL, ids=lambda c: c.__name__)
def test_every_call_on_an_openrouter_client_is_recorded_as_openrouter(call, payloads):
    call(FakeOpenAICompatible(OPENROUTER_BASE_URL), None)

    record = the_one_record(payloads)
    assert (record["provider"], record["model_source"]) == ("OPENROUTER", "OPENROUTER")


@pytest.mark.parametrize("call", EVERY_METERED_CALL, ids=lambda c: c.__name__)
def test_every_call_on_a_plain_openai_client_is_still_recorded_as_openai(call, payloads):
    call(FakeOpenAICompatible("https://api.openai.com/v1"), None)

    record = the_one_record(payloads)
    assert (record["provider"], record["model_source"]) == ("OPENAI", "OPENAI")


TOOLS = [{"type": "function", "function": {"name": "lookup", "parameters": {"type": "object", "properties": {}}}}]


def sync_tool_chat(fake):
    return fake.sync_client().chat.completions.create(**CHAT_REQUEST, tools=TOOLS)


def sync_tool_chat_stream(fake):
    for _chunk in fake.sync_client().chat.completions.create(**CHAT_REQUEST, tools=TOOLS, stream=True):
        pass


@pytest.mark.parametrize("call", [sync_tool_chat, sync_tool_chat_stream], ids=lambda c: c.__name__)
def test_a_tool_call_on_an_openrouter_client_keeps_its_function_call_subtype(call, payloads):
    call(FakeOpenAICompatible(OPENROUTER_BASE_URL))

    assert the_one_record(payloads).get("operation_subtype") == "function_call"


def test_a_chat_call_on_an_openai_client_bound_to_perplexity_is_still_recorded_as_perplexity(mock_revenium_client):
    client, _ = perplexity_cost.openai_clients_at(perplexity_cost.PERPLEXITY_BASE_URL)

    client.chat.completions.create(model=perplexity_cost.PERPLEXITY_MODEL, messages=calls.USER_MESSAGES)

    calls.wait_for_metering()
    records = perplexity_cost.recorded_costs(mock_revenium_client)
    assert [record["provider"] for record in records] == ["PERPLEXITY"]


def test_an_async_azure_client_is_still_labelled_azure():
    client = AsyncAzureOpenAI(api_key="x", azure_endpoint="https://example.openai.azure.com", api_version="2024-10-21")

    assert labels_for(client) == {"provider": "Azure", "model_source": "OPENAI"}
