"""Abandoning a Google (genai / Vertex) stream early must still meter."""
import datetime
import gc
import importlib.util
from unittest.mock import MagicMock, patch

import pytest

# When the optional SDKs are absent, the subpackage __init__ falls back to
# middleware = None instead of raising -- guard with skipif, not importorskip.
from revenium_middleware.google.google_ai import middleware as genai_mw

NOW = datetime.datetime.now(datetime.timezone.utc)

# Both google metering paths dispatch via google.common.utils.
DISPATCH = "revenium_middleware.google.common.utils.run_async_in_thread"

genai_missing = genai_mw is None
vertexai_missing = importlib.util.find_spec("vertexai") is None


def drain_coroutine(coro):
    coro.close()
    return MagicMock()


def make_chunk():
    chunk = MagicMock()
    chunk.usage_metadata.prompt_token_count = 3
    chunk.usage_metadata.candidates_token_count = 5
    chunk.usage_metadata.total_token_count = 8
    chunk.usage_metadata.cached_content_token_count = 0
    chunk.text = "tok"
    return chunk


@pytest.mark.skipif(genai_missing, reason="google-genai SDK not installed")
@patch(DISPATCH, side_effect=drain_coroutine)
class TestGoogleAIEarlyTermination:
    def test_early_break_dispatches_metering_once(self, mock_run):
        wrapper = genai_mw.handle_streaming_response(iter([make_chunk() for _ in range(5)]), NOW, {})
        for i, _ in enumerate(wrapper):
            if i == 1:
                break
        del wrapper
        gc.collect()

        assert mock_run.call_count == 1

    def test_full_consumption_dispatches_exactly_once(self, mock_run):
        wrapper = genai_mw.handle_streaming_response(iter([make_chunk() for _ in range(3)]), NOW, {})
        for _ in wrapper:
            pass
        del wrapper
        gc.collect()

        assert mock_run.call_count == 1


@pytest.mark.skipif(vertexai_missing, reason="vertexai SDK not installed")
@patch(DISPATCH, side_effect=drain_coroutine)
class TestVertexAIEarlyTermination:
    def test_early_break_dispatches_metering_once(self, mock_run):
        from revenium_middleware.google.vertex_ai import middleware as vertex_mw
        wrapper = vertex_mw.handle_vertex_ai_streaming_response(
            iter([make_chunk() for _ in range(5)]), NOW, {}, "gemini-test")
        for i, _ in enumerate(wrapper):
            if i == 1:
                break
        del wrapper
        gc.collect()

        assert mock_run.call_count == 1

    def test_full_consumption_dispatches_exactly_once(self, mock_run):
        from revenium_middleware.google.vertex_ai import middleware as vertex_mw
        wrapper = vertex_mw.handle_vertex_ai_streaming_response(
            iter([make_chunk() for _ in range(3)]), NOW, {}, "gemini-test")
        for _ in wrapper:
            pass
        del wrapper
        gc.collect()

        assert mock_run.call_count == 1
