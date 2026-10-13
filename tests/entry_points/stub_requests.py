"""How many requests reached a provider stub of ``calls``, keyed by stub.

LiteLLM answers from ``mock_response`` inside litellm and has no stub to count.
Provider packages are imported lazily: the vertex job installs none but Vertex.
"""
from collections import Counter

import httpx

from entry_points import calls


def _count_calls(monkeypatch, counter, owner, name, key):
    original = getattr(owner, name)

    def counted(*args, **kwargs):
        counter[key] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(owner, name, counted)


def count_provider_requests(monkeypatch):
    counter = Counter()
    _count_calls(monkeypatch, counter, httpx.MockTransport, "handle_request", "httpx")
    _count_calls(monkeypatch, counter, httpx.MockTransport, "handle_async_request", "httpx")
    try:
        import httpx2
    except ImportError:
        httpx2 = None
    if httpx2 is not None:
        _count_calls(monkeypatch, counter, httpx2.MockTransport, "handle_request", "httpx2")
        _count_calls(monkeypatch, counter, httpx2.MockTransport, "handle_async_request", "httpx2")
    _count_calls(monkeypatch, counter, calls._GeminiHandler, "do_POST", "gemini")
    _count_calls(monkeypatch, counter, calls._GeminiHandler, "do_GET", "gemini")
    for stub in (calls._VertexPredictionClient, calls._VertexPredictionAsyncClient):
        _count_calls(monkeypatch, counter, stub, "generate_content", "vertex")
        _count_calls(monkeypatch, counter, stub, "stream_generate_content", "vertex")
    for stub in (calls._VertexEndpointClient, calls._VertexEndpointAsyncClient):
        _count_calls(monkeypatch, counter, stub, "predict", "vertex")
    return counter
