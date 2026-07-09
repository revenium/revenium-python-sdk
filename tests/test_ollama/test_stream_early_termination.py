"""Abandoning an Ollama stream early must still meter the partial usage."""
import datetime
import gc
from types import SimpleNamespace
from unittest.mock import patch

from revenium_middleware.ollama import middleware as mw

NOW = datetime.datetime.now(datetime.timezone.utc)


def make_chunks(n):
    return [SimpleNamespace(message=SimpleNamespace(content=f"tok{i}"), done=False) for i in range(n)]


@patch("revenium_middleware.ollama.middleware.handle_response")
class TestOllamaEarlyTermination:
    def test_early_break_meters_once(self, mock_handle):
        gen = mw.handle_streaming_response(iter(make_chunks(5)), NOW, {}, "txn-1", "chat", {})
        for i, _ in enumerate(gen):
            if i == 1:
                break
        del gen
        gc.collect()

        assert mock_handle.call_count == 1
        assert mock_handle.call_args[0][3] is True  # is_streaming

    def test_full_consumption_meters_exactly_once(self, mock_handle):
        gen = mw.handle_streaming_response(iter(make_chunks(3)), NOW, {}, "txn-2", "chat", {})
        for _ in gen:
            pass
        del gen
        gc.collect()

        assert mock_handle.call_count == 1

    def test_zero_chunks_consumed_does_not_meter(self, mock_handle):
        gen = mw.handle_streaming_response(iter(make_chunks(3)), NOW, {}, "txn-3", "chat", {})
        del gen
        gc.collect()

        # Never started: no chunks arrived, nothing to meter.
        assert mock_handle.call_count == 0
