"""Abandoning a LiteLLM client stream early must still meter."""
import datetime
import gc
from types import SimpleNamespace
from unittest.mock import patch

import pytest

# The middleware module wrapt-patches litellm at import time; skip when the
# optional dependency is absent (mirrors the other optional-provider suites).
pytest.importorskip("litellm")

from revenium_middleware.litellm.client import middleware as mw  # noqa: E402

NOW = datetime.datetime.now(datetime.timezone.utc)


def make_chunks(n):
    return [SimpleNamespace(usage=None, choices=[]) for _ in range(n)]


@patch("revenium_middleware.litellm.client.middleware.handle_response")
class TestLiteLLMClientEarlyTermination:
    def test_early_break_meters_once(self, mock_handle):
        gen = mw.handle_streaming_response(iter(make_chunks(5)), NOW, {})
        for i, _ in enumerate(gen):
            if i == 1:
                break
        del gen
        gc.collect()

        assert mock_handle.call_count == 1
        assert mock_handle.call_args[0][3] is True  # is_streaming

    def test_full_consumption_meters_exactly_once(self, mock_handle):
        gen = mw.handle_streaming_response(iter(make_chunks(3)), NOW, {})
        for _ in gen:
            pass
        del gen
        gc.collect()

        assert mock_handle.call_count == 1
