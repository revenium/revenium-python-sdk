"""A breached spending cap refuses Vertex AI calls before the gapic client is reached (BACK-4190).

The models are real vertexai objects whose prediction clients are the
entry-point matrix stubs. Runs in the vertex environment (``.[google-vertex]``).
"""
import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

pytest.importorskip("vertexai")

from revenium_middleware._core.exceptions import BudgetExceededError  # noqa: E402
from revenium_middleware.google.vertex_ai import middleware as vertex_middleware  # noqa: E402
from entry_points import calls  # noqa: E402
from entry_points.caps import breached_rule, record_cap_checks, scoped, seed_rules  # noqa: E402
from entry_points.stub_requests import count_provider_requests  # noqa: E402

RULE_ID = 4190
IMAGEN_MODEL = "imagen-3.0-generate-002"


def cap(*filters):
    return breached_rule(RULE_ID, *filters)


@pytest.fixture
def seed(monkeypatch):
    monkeypatch.delenv("REVENIUM_SELECTIVE_METERING", raising=False)
    return lambda *rules: seed_rules(monkeypatch, *rules)


@pytest.fixture
def gapic_requests(monkeypatch):
    return count_provider_requests(monkeypatch)


def _run(call):
    result = call()
    if asyncio.iscoroutine(result):
        asyncio.run(result)


VERTEX_CALLS = [
    pytest.param(lambda: calls.vertex_model().generate_content("hi"), calls.GEMINI_MODEL, id="generate_content"),
    pytest.param(lambda: calls.vertex_model().generate_content_async("hi"), calls.GEMINI_MODEL,
                 id="generate_content_async"),
    pytest.param(lambda: calls.vertex_model().start_chat().send_message("hi"), calls.GEMINI_MODEL, id="send_message"),
    pytest.param(lambda: calls.vertex_model().start_chat().send_message_async("hi"), calls.GEMINI_MODEL,
                 id="send_message_async"),
    pytest.param(lambda: calls.vertex_embedding_model().get_embeddings(["hi"]), calls.VERTEX_EMBED_MODEL,
                 id="get_embeddings"),
    pytest.param(lambda: calls.vertex_embedding_model().get_embeddings_async(["hi"]), calls.VERTEX_EMBED_MODEL,
                 id="get_embeddings_async"),
]


@pytest.mark.parametrize("call, model", VERTEX_CALLS)
def test_vertex_calls_are_refused_before_the_gapic_client(call, model, seed, gapic_requests, mock_revenium_client):
    seed(cap(scoped("PROVIDER", "IS", "Google"), scoped("MODEL", "IS", model)))
    with pytest.raises(BudgetExceededError):
        _run(call)
    assert sum(gapic_requests.values()) == 0
    calls.wait_for_metering()
    assert calls.recorded_payloads(mock_revenium_client) == []

    seed(cap(scoped("PROVIDER", "IS_NOT", "Google")))
    _run(call)
    calls.wait_for_metering()
    assert len(calls.recorded_payloads(mock_revenium_client)) == 1


ASYNC_VERTEX_CALLS = [
    pytest.param(lambda: calls.vertex_model().generate_content_async("hi"), id="generate_content_async"),
    pytest.param(lambda: calls.vertex_model().start_chat().send_message_async("hi"), id="send_message_async"),
]


@pytest.mark.parametrize("call", ASYNC_VERTEX_CALLS)
@pytest.mark.parametrize("selective", [False, True], ids=["metered", "selectively_unmetered"])
def test_an_async_vertex_call_created_before_a_cap_breaches_is_refused_when_awaited(
        call, selective, seed, gapic_requests, monkeypatch):
    if selective:
        monkeypatch.setenv("REVENIUM_SELECTIVE_METERING", "true")
    seed(cap(scoped("PROVIDER", "IS", "NOBODY")))

    async def run():
        pending = call()
        seed(cap())
        await pending

    with pytest.raises(BudgetExceededError):
        asyncio.run(run())
    assert sum(gapic_requests.values()) == 0


IMAGEN_IMPLS = [
    pytest.param(vertex_middleware._imagen_generate_images_impl, id="generate_images"),
    pytest.param(vertex_middleware._imagen_edit_image_impl, id="edit_image"),
]


@pytest.mark.parametrize("impl", IMAGEN_IMPLS)
@pytest.mark.parametrize("selective", [False, True], ids=["metered", "selectively_unmetered"])
def test_imagen_calls_are_refused_before_the_sdk(impl, selective, seed, monkeypatch):
    seed(cap(scoped("PROVIDER", "IS", "Google"), scoped("MODEL", "IS", IMAGEN_MODEL)))
    if selective:
        monkeypatch.setenv("REVENIUM_SELECTIVE_METERING", "true")
    wrapped = MagicMock(return_value=SimpleNamespace(images=[object()]))
    model = SimpleNamespace(_model_id=IMAGEN_MODEL)

    with pytest.raises(BudgetExceededError):
        impl(wrapped, model, ("prompt",), {})
    wrapped.assert_not_called()

    seed(cap(scoped("PROVIDER", "IS_NOT", "Google")))
    impl(wrapped, model, ("prompt",), {})
    wrapped.assert_called_once()


def test_selectively_unmetered_vertex_calls_are_still_refused(seed, monkeypatch, gapic_requests,
                                                             mock_revenium_client):
    seed(cap(scoped("PROVIDER", "IS", "Google"), scoped("MODEL", "IS", calls.GEMINI_MODEL)))
    monkeypatch.setenv("REVENIUM_SELECTIVE_METERING", "true")

    with pytest.raises(BudgetExceededError):
        calls.vertex_model().generate_content("hi", usage_metadata={"agent": "unmetered-agent"})
    assert sum(gapic_requests.values()) == 0

    seed(cap(scoped("PROVIDER", "IS_NOT", "Google")))
    calls.vertex_model().generate_content("hi")
    calls.wait_for_metering()
    assert calls.recorded_payloads(mock_revenium_client) == []


def test_a_nested_call_inside_a_metered_call_is_checked_once(seed, monkeypatch):
    """A vertexai release that sends a chat message through the wrapped public
    generate_content must still check the one request once, with the chat's identity."""
    seed(cap(scoped("AGENT", "IS", "nobody")))
    checks = record_cap_checks(monkeypatch)
    model = calls.vertex_model()
    chat = model.start_chat()

    vertex_middleware.send_message_wrapper_impl(
        lambda *args, **kwargs: model.generate_content("hi"), chat, ("hi",), {"usage_metadata": {"agent": "chat"}})

    assert [(check.model, check.provider, check.usage_metadata.get("agent")) for check in checks] == [
        (calls.GEMINI_MODEL, "Google", "chat")]
