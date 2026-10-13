"""Wrapping a LangChain embeddings model with wrap() or attach_to() says what it does not do.

LangChain embeddings models run no callbacks, so the handler wrap() builds can
never see a call. wrap() returns the model unchanged and logs one warning per
model class per process. OpenAI and Azure OpenAI embeddings are still recorded
by the OpenAI middleware and take their tags from @revenium_metadata; embeddings
that reach a vendor client with its own Revenium middleware, such as Ollama, are
recorded by that middleware; embeddings from other vendors are not metered (BACK-4323).

Real langchain classes run over stubbed transports, so no call leaves the process.
"""
import asyncio
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock, patch

import pytest

httpx = pytest.importorskip("httpx")
langchain_openai = pytest.importorskip("langchain_openai")

from langchain_core.callbacks import CallbackManager  # noqa: E402
from langchain_core.embeddings import FakeEmbeddings  # noqa: E402
from langchain_core.language_models.fake_chat_models import FakeListChatModel  # noqa: E402
from langchain_core.prompts import ChatPromptTemplate  # noqa: E402
from langchain_openai import AzureOpenAIEmbeddings, ChatOpenAI, OpenAIEmbeddings  # noqa: E402

import revenium_middleware.openai  # noqa: E402,F401  installs the OpenAI patches
import revenium_middleware.openai.langchain as langchain_module  # noqa: E402
from revenium_middleware import revenium_metadata  # noqa: E402
from revenium_middleware._core.call_ownership import reset_claimed_response_ids  # noqa: E402
from revenium_middleware.openai.langchain import attach_to, wrap  # noqa: E402
from revenium_middleware.openai.langchain.unified_handler import UnifiedReveniumCallbackHandler  # noqa: E402

from .openai_transport import CHUNKED_STREAMS, answer_every_transport, http_library_of  # noqa: E402

LOGGER_NAME = "revenium_middleware.langchain"
PROMPT_TOKENS = 9
AZURE_KWARGS = dict(azure_endpoint="https://x.openai.azure.com", api_version="2024-02-01", api_key="d",
                    model="text-embedding-3-small", max_retries=0, check_embedding_ctx_length=False)


WORKER_TIMEOUT_SECONDS = 10


def _run_now(coroutine, workers):
    thread = threading.Thread(target=lambda: asyncio.run(coroutine))
    workers.append(thread)
    thread.start()
    thread.join(timeout=WORKER_TIMEOUT_SECONDS)
    return thread


def openai_embeddings():
    return OpenAIEmbeddings(model="text-embedding-3-small", api_key="sk-test", max_retries=0,
                            check_embedding_ctx_length=False)


def azure_embeddings():
    return AzureOpenAIEmbeddings(**AZURE_KWARGS)


def warnings_from(caplog):
    return [r for r in caplog.records if r.name == LOGGER_NAME and r.levelname == "WARNING"]


@pytest.fixture(autouse=True)
def _fresh_warning_state(monkeypatch):
    monkeypatch.setattr(langchain_module, "_WARNED_NO_CALLBACKS_CLASSES", set(), raising=False)
    monkeypatch.setenv("REVENIUM_METERING_API_KEY", "hak_test_langchain")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    reset_claimed_response_ids()
    yield
    reset_claimed_response_ids()


@pytest.fixture
def records():
    captured = []
    workers = []
    lock = threading.Lock()

    def record(operation, payload, *args, **kwargs):
        with lock:
            captured.append(payload)
        return MagicMock(id="recorded", status_code=201)

    with patch("revenium_middleware.openai.middleware.submit_ai_event", side_effect=record), \
            patch("revenium_middleware.openai.middleware.run_async_in_thread", side_effect=lambda coroutine: _run_now(coroutine, workers)):
        yield captured
        stuck = [worker for worker in workers if worker.is_alive()]
        assert not stuck, f"{len(stuck)} submission worker(s) still running after {WORKER_TIMEOUT_SECONDS}s"


@pytest.fixture
def openai_stub(monkeypatch):
    def respond(request):
        inputs = json.loads(request.content or b"{}").get("input", [])
        inputs = inputs if isinstance(inputs, list) else [inputs]
        body = {"object": "list", "model": "text-embedding-3-small",
                "data": [{"object": "embedding", "index": i, "embedding": [0.1, 0.2]} for i in range(len(inputs))],
                "usage": {"prompt_tokens": PROMPT_TOKENS, "total_tokens": PROMPT_TOKENS}}
        library = http_library_of(request)
        return library.Response(200, headers={"content-type": "application/json"},
                                stream=CHUNKED_STREAMS[library]([json.dumps(body).encode()]))

    answer_every_transport(monkeypatch, respond)


def test_wrapping_an_embeddings_model_warns_once_naming_its_class(caplog):
    """Fails on main: the swallowed assignment logs nothing."""
    model = openai_embeddings()
    with caplog.at_level("WARNING", logger=LOGGER_NAME):
        wrap(model)
        wrap(model)
        attach_to(model)
    emitted = warnings_from(caplog)
    assert len(emitted) == 1
    text = emitted[0].getMessage()
    assert "OpenAIEmbeddings" in text
    assert "@revenium_metadata" in text
    assert "not metered" in text
    assert "Ollama" in text


def test_attach_to_an_embeddings_model_warns_once(caplog):
    """Fails on main: no warning."""
    model = FakeEmbeddings(size=2)
    with caplog.at_level("WARNING", logger=LOGGER_NAME):
        attach_to(model)
        attach_to(model)
    emitted = warnings_from(caplog)
    assert len(emitted) == 1
    assert "FakeEmbeddings" in emitted[0].getMessage()


def test_each_embeddings_class_gets_its_own_warning(caplog):
    """Fails on main: no warning. A single process-wide flag would hide the second class."""
    with caplog.at_level("WARNING", logger=LOGGER_NAME):
        wrap(openai_embeddings())
        wrap(FakeEmbeddings(size=2))
    texts = [r.getMessage() for r in warnings_from(caplog)]
    assert len(texts) == 2
    assert "OpenAIEmbeddings" in texts[0]
    assert "FakeEmbeddings" in texts[1]


def test_concurrent_wraps_warn_once(caplog):
    """Fails on main: no warning."""
    workers = 8
    barrier = threading.Barrier(workers)

    def wrap_together(_):
        barrier.wait(timeout=10)
        wrap(FakeEmbeddings(size=2))

    with caplog.at_level("WARNING", logger=LOGGER_NAME):
        with ThreadPoolExecutor(workers) as pool:
            list(pool.map(wrap_together, range(workers)))
    assert len(warnings_from(caplog)) == 1


@pytest.mark.parametrize("make_chat", [
    lambda: ChatOpenAI(model="gpt-4o-mini", api_key="sk-test"),
    lambda: ChatOpenAI(model="gpt-4o-mini", api_key="sk-test", callbacks=CallbackManager([])),
], ids=["no-callbacks", "callback-manager"])
def test_a_chat_model_gets_its_handler_and_no_warning(caplog, make_chat):
    """Passes on main; guards the chat path against the new warning."""
    with caplog.at_level("WARNING", logger=LOGGER_NAME):
        chat = wrap(make_chat())
    assert warnings_from(caplog) == []
    assert any(isinstance(c, UnifiedReveniumCallbackHandler) for c in chat.callbacks)


@pytest.mark.parametrize("make_model", [openai_embeddings, azure_embeddings, lambda: FakeEmbeddings(size=2)],
                         ids=["openai", "azure", "fake"])
def test_wrap_returns_the_same_embeddings_model_and_never_raises(make_model):
    """Passes on main; pins that wrap() and attach_to() hand the model back unchanged."""
    model = make_model()
    assert wrap(model, usage_metadata={"organizationName": "AcmeCorp"}) is model
    assert attach_to(model) is model


@pytest.mark.parametrize("make_model, provider", [(openai_embeddings, "openai"), (azure_embeddings, "azure")],
                         ids=["openai", "azure"])
@pytest.mark.parametrize("call", ["embed_documents", "aembed_query"])
def test_an_openai_embeddings_call_inside_revenium_metadata_after_wrap_carries_its_tags(
        records, openai_stub, make_model, provider, call):
    """Passes on main; pins the instruction the warning gives: tag embeddings with @revenium_metadata."""
    model = wrap(make_model(), usage_metadata={"organizationName": "WrapOnly"})

    @revenium_metadata(organization_name="AcmeCorp", agent="support_agent", trace_id="session-123")
    def embed():
        if call == "embed_documents":
            return model.embed_documents(["a", "b"])
        return asyncio.run(model.aembed_query("x"))

    embed()
    assert len(records) == 1
    record = records[0]
    assert record["operation_type"] == "EMBED"
    assert record["input_token_count"] == PROMPT_TOKENS
    assert record["organization_name"] == "AcmeCorp"
    assert record["agent"] == "support_agent"
    assert record["trace_id"] == "session-123"
    assert str(record["provider"]).lower() == provider


def test_a_wrapped_mistral_embeddings_call_sends_no_record_and_warns(caplog, records):
    """Fails on main (no warning). Documents the known gap: invert it when vendor embeddings metering lands."""
    pytest.importorskip("langchain_mistralai")
    tokenizers = pytest.importorskip("tokenizers")
    from langchain_mistralai import MistralAIEmbeddings

    def respond(request):
        inputs = json.loads(request.content)["input"]
        data = [{"object": "embedding", "index": i, "embedding": [0.1, 0.2]} for i in range(len(inputs))]
        return httpx.Response(200, json={"id": "e", "object": "list", "model": "mistral-embed", "data": data,
                                         "usage": {"prompt_tokens": 4, "total_tokens": 4}})

    tokenizer = tokenizers.Tokenizer(tokenizers.models.WordLevel({"[UNK]": 0}, unk_token="[UNK]"))
    model = MistralAIEmbeddings(
        model="mistral-embed", api_key="stub", tokenizer=tokenizer,
        client=httpx.Client(base_url="http://mistral.test/v1", transport=httpx.MockTransport(respond)),
        async_client=httpx.AsyncClient(base_url="http://mistral.test/v1", transport=httpx.MockTransport(respond)))
    with caplog.at_level("WARNING", logger=LOGGER_NAME):
        wrapped = wrap(model)
        first = wrapped.embed_documents(["a", "b"])
        wrapped.embed_documents(["c"])
        wrapped.embed_query("d")
    assert len(first) == 2
    assert records == []
    emitted = warnings_from(caplog)
    assert len(emitted) == 1
    assert "MistralAIEmbeddings" in emitted[0].getMessage()


def test_wrapping_a_chain_warns_once_without_embeddings_advice(caplog):
    """Fails on main: no warning. A chain has no callbacks field either, and embeddings advice would mislead."""
    chain = ChatPromptTemplate.from_template("{x}") | FakeListChatModel(responses=["x"])
    with caplog.at_level("WARNING", logger=LOGGER_NAME):
        assert wrap(chain) is chain
        wrap(chain)
    emitted = warnings_from(caplog)
    assert len(emitted) == 1
    text = emitted[0].getMessage()
    assert type(chain).__name__ in text
    assert "embeddings" not in text.lower()
