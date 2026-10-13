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

    def test_openai_cache_writes_are_taken_out_of_the_prompt(self, mock_submit, _):
        """An OpenAI upstream reports the write inside the prompt (BACK-3998)."""
        from litellm.types.utils import Usage

        usage = Usage(
            prompt_tokens=4583,
            completion_tokens=5,
            total_tokens=4588,
            prompt_tokens_details={"cached_tokens": 0, "cache_write_tokens": 4580},
        )

        mw.handle_response(make_response(usage), NOW, {}, False)

        args = submitted_args(mock_submit)
        assert args["input_token_count"] == 3
        assert args["cache_creation_token_count"] == 4580
        assert args["cache_read_token_count"] == 0
        assert args["total_token_count"] == 4588

    def test_openai_reads_stay_in_the_prompt_when_writes_are_taken_out(self, mock_submit, _):
        """The platform removes reads itself; only the write leaves the prompt."""
        from litellm.types.utils import Usage

        usage = Usage(
            prompt_tokens=6000,
            completion_tokens=10,
            total_tokens=6010,
            prompt_tokens_details={"cached_tokens": 4000, "cache_write_tokens": 1500},
        )

        mw.handle_response(make_response(usage), NOW, {}, False)

        args = submitted_args(mock_submit)
        assert args["input_token_count"] == 4500
        assert args["cache_read_token_count"] == 4000
        assert args["cache_creation_token_count"] == 1500
        assert args["total_token_count"] == 6010

    def test_anthropic_cache_writes_keep_todays_counts(self, mock_submit, _):
        """Regression guard: LiteLLM mirrors an Anthropic write into
        ``prompt_tokens_details.cache_write_tokens``; that mirror is not an
        OpenAI write and must not leave the prompt."""
        from litellm.types.utils import Usage

        usage = Usage(
            prompt_tokens=100,
            completion_tokens=5,
            total_tokens=105,
            cache_creation_input_tokens=40,
            cache_read_input_tokens=30,
        )
        assert usage.prompt_tokens_details.cache_write_tokens == 40

        mw.handle_response(make_response(usage), NOW, {}, False)

        args = submitted_args(mock_submit)
        assert args["cache_creation_token_count"] == 40
        assert args["cache_read_token_count"] == 30
        assert args["input_token_count"] == 100
