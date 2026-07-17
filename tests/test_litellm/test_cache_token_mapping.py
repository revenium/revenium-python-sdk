"""Provider cache token fields must land in the matching metering fields.

LiteLLM exposes OpenAI-style cache reads at
``usage.prompt_tokens_details.cached_tokens`` and Anthropic passthrough
counts as top-level ``cache_read_input_tokens`` /
``cache_creation_input_tokens``. Reading a nonexistent ``usage.cached_tokens``
drops the count entirely, and routing reads into cache_creation over-costs
cache-heavy workloads.
"""
import asyncio
import datetime
from types import SimpleNamespace
from unittest.mock import patch

import pytest

# The middleware module wrapt-patches litellm at import time; skip when the
# optional dependency is absent (mirrors the other optional-provider suites).
pytest.importorskip("litellm")

from revenium_middleware.litellm.client import middleware as mw  # noqa: E402

NOW = datetime.datetime.now(datetime.timezone.utc)


def run_inline(coro):
    """Execute the metering coroutine synchronously so asserts see the call."""
    asyncio.run(coro)
    return SimpleNamespace(name="inline-metering")


def make_response(usage):
    return SimpleNamespace(
        id="txn-cache-mapping",
        usage=usage,
        model="gpt-4o-mini",
        finish_reason="stop",
        system_fingerprint=None,
    )


def submitted_args(mock_submit):
    assert mock_submit.call_count == 1
    return mock_submit.call_args[0][1]


@patch.object(mw, "run_async_in_thread", side_effect=run_inline)
@patch.object(mw, "submit_ai_event")
class TestLiteLLMCacheTokenMapping:
    def test_openai_style_cached_tokens_are_cache_reads(self, mock_submit, _):
        usage = SimpleNamespace(
            prompt_tokens=100,
            completion_tokens=10,
            prompt_tokens_details=SimpleNamespace(cached_tokens=80),
        )

        mw.handle_response(make_response(usage), NOW, {}, False)

        args = submitted_args(mock_submit)
        assert args["cache_read_token_count"] == 80
        assert args["cache_creation_token_count"] == 0

    def test_anthropic_passthrough_fields_map_to_both_counts(self, mock_submit, _):
        usage = SimpleNamespace(
            prompt_tokens=100,
            completion_tokens=10,
            cache_read_input_tokens=30,
            cache_creation_input_tokens=50,
        )

        mw.handle_response(make_response(usage), NOW, {}, False)

        args = submitted_args(mock_submit)
        assert args["cache_read_token_count"] == 30
        assert args["cache_creation_token_count"] == 50

    def test_none_valued_details_are_treated_as_zero(self, mock_submit, _):
        usage = SimpleNamespace(
            prompt_tokens=100,
            completion_tokens=10,
            prompt_tokens_details=SimpleNamespace(cached_tokens=None),
        )

        mw.handle_response(make_response(usage), NOW, {}, False)

        args = submitted_args(mock_submit)
        assert args["cache_read_token_count"] == 0
        assert args["cache_creation_token_count"] == 0

    def test_absent_cache_fields_default_to_zero(self, mock_submit, _):
        usage = SimpleNamespace(prompt_tokens=100, completion_tokens=10)

        mw.handle_response(make_response(usage), NOW, {}, False)

        args = submitted_args(mock_submit)
        assert args["cache_read_token_count"] == 0
        assert args["cache_creation_token_count"] == 0
