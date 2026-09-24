"""Cache token fields must land in the matching metering fields for the LiteLLM proxy.

Mirrors tests/test_litellm/test_cache_token_mapping.py (client-mode LiteLLM), but
exercises revenium_middleware.litellm.proxy.middleware.MiddlewareHandler, which reads
usage via CustomLogger's success/failure event hooks instead of a completion return
value. Before this test existed, the proxy middleware hardcoded
cache_creation_token_count / cache_read_token_count to 0 in both paths, silently
mis-rating every cache-heavy call routed through the LiteLLM proxy.
"""
import datetime
from types import SimpleNamespace
from unittest.mock import patch

import pytest

# The middleware module imports litellm at import time; skip when the optional
# dependency is absent (mirrors the other optional-provider suites).
pytest.importorskip("litellm")

from revenium_middleware.litellm.proxy import middleware as mw  # noqa: E402
from .proxy_hook_harness import (  # noqa: E402
    base_kwargs,
    make_success_response,
    run_hook,
    run_inline,
    submitted_args,
)

NOW = datetime.datetime.now(datetime.timezone.utc)


@patch.object(mw, "run_async_in_thread", side_effect=run_inline)
@patch.object(mw, "get_client", return_value=object())
@patch.object(mw, "submit_ai_event")
class TestLiteLLMProxyCacheTokenMappingSuccess:
    def test_openai_style_cached_tokens_are_cache_reads(self, mock_submit, _get_client, _run):
        usage = SimpleNamespace(
            prompt_tokens=100,
            completion_tokens=10,
            total_tokens=110,
            prompt_tokens_details=SimpleNamespace(cached_tokens=80),
        )
        response = make_success_response(usage)

        run_hook(mw.proxy_handler_instance.async_log_success_event(
            base_kwargs(), response, NOW, NOW
        ))

        args = submitted_args(mock_submit)
        assert args["cache_read_token_count"] == 80
        assert args["cache_creation_token_count"] == 0

    def test_anthropic_passthrough_fields_map_to_both_counts(self, mock_submit, _get_client, _run):
        usage = SimpleNamespace(
            prompt_tokens=100,
            completion_tokens=10,
            total_tokens=180,
            cache_read_input_tokens=30,
            cache_creation_input_tokens=50,
        )
        response = make_success_response(usage)

        run_hook(mw.proxy_handler_instance.async_log_success_event(
            base_kwargs(), response, NOW, NOW
        ))

        args = submitted_args(mock_submit)
        assert args["cache_read_token_count"] == 30
        assert args["cache_creation_token_count"] == 50

    def test_none_valued_details_are_treated_as_zero(self, mock_submit, _get_client, _run):
        usage = SimpleNamespace(
            prompt_tokens=100,
            completion_tokens=10,
            total_tokens=110,
            prompt_tokens_details=SimpleNamespace(cached_tokens=None),
        )
        response = make_success_response(usage)

        run_hook(mw.proxy_handler_instance.async_log_success_event(
            base_kwargs(), response, NOW, NOW
        ))

        args = submitted_args(mock_submit)
        assert args["cache_read_token_count"] == 0
        assert args["cache_creation_token_count"] == 0

    def test_absent_cache_fields_default_to_zero(self, mock_submit, _get_client, _run):
        usage = SimpleNamespace(prompt_tokens=100, completion_tokens=10, total_tokens=110)
        response = make_success_response(usage)

        run_hook(mw.proxy_handler_instance.async_log_success_event(
            base_kwargs(), response, NOW, NOW
        ))

        args = submitted_args(mock_submit)
        assert args["cache_read_token_count"] == 0
        assert args["cache_creation_token_count"] == 0

    def test_success_path_dict_style_usage_keeps_base_and_cache_tokens_consistent(
        self, mock_submit, _get_client, _run
    ):
        """Regression for a correctness finding on the success path: cache fields were
        read tolerantly via extract_cache_tokens(usage), but prompt_tokens/completion_tokens/
        total_tokens were read directly as usage.<attr> -- which assumes `usage` is always
        attribute-style. A dict-shaped usage value (as LiteLLM can produce) would crash on
        that direct access entirely rather than silently misreport. All fields must now be
        read through the same tolerant accessor regardless of usage's shape.
        """
        usage = {
            "prompt_tokens": 100,
            "completion_tokens": 10,
            "total_tokens": 110,
            "cache_read_input_tokens": 30,
            "cache_creation_input_tokens": 50,
        }
        response = make_success_response(usage)

        run_hook(mw.proxy_handler_instance.async_log_success_event(
            base_kwargs(), response, NOW, NOW
        ))

        args = submitted_args(mock_submit)
        assert args["cache_read_token_count"] == 30
        assert args["cache_creation_token_count"] == 50
        # The event names no upstream, so the prompt count is reported exactly
        # as LiteLLM built it. Only an Anthropic row may have its cache overlap
        # removed here -- see test_proxy_cache_token_parity.py (BACK-3334).
        assert args["input_token_count"] == 100
        assert args["output_token_count"] == 10
        assert args["total_token_count"] == 110


@patch.object(mw, "run_async_in_thread", side_effect=run_inline)
@patch.object(mw, "get_client", return_value=object())
@patch.object(mw, "submit_ai_event")
class TestLiteLLMProxyCacheTokenMappingFailure:
    def test_failure_path_maps_dict_style_anthropic_cache_fields(self, mock_submit, _get_client, _run):
        error = RuntimeError("upstream failure")
        error.usage = {
            "prompt_tokens": 100,
            "completion_tokens": 0,
            "total_tokens": 100,
            "cache_read_input_tokens": 30,
            "cache_creation_input_tokens": 50,
        }

        run_hook(mw.proxy_handler_instance.async_log_failure_event(
            base_kwargs(), error, NOW, NOW
        ))

        args = submitted_args(mock_submit)
        assert args["cache_read_token_count"] == 30
        assert args["cache_creation_token_count"] == 50

    def test_failure_path_maps_dict_style_openai_cache_fields(self, mock_submit, _get_client, _run):
        error = RuntimeError("upstream failure")
        error.usage = {
            "prompt_tokens": 100,
            "completion_tokens": 0,
            "total_tokens": 100,
            "prompt_tokens_details": {"cached_tokens": 80},
        }

        run_hook(mw.proxy_handler_instance.async_log_failure_event(
            base_kwargs(), error, NOW, NOW
        ))

        args = submitted_args(mock_submit)
        assert args["cache_read_token_count"] == 80
        assert args["cache_creation_token_count"] == 0

    def test_failure_path_without_usage_defaults_to_zero(self, mock_submit, _get_client, _run):
        error = RuntimeError("upstream failure, no usage info available")

        run_hook(mw.proxy_handler_instance.async_log_failure_event(
            base_kwargs(), error, NOW, NOW
        ))

        args = submitted_args(mock_submit)
        assert args["cache_read_token_count"] == 0
        assert args["cache_creation_token_count"] == 0

    def test_failure_path_attribute_style_usage_keeps_base_and_cache_tokens_consistent(
        self, mock_submit, _get_client, _run
    ):
        """Regression for an inconsistent-payload bug introduced during review iteration:
        cache tokens were read from the raw (possibly attribute-style) usage object, but
        prompt/completion/total tokens were read from a usage value first normalized to a
        zero-only dict for anything that wasn't already a dict -- so an attribute-style
        usage object produced non-zero cache counts alongside zeroed base token counts for
        the same call. All fields must now come from the same usage object consistently.
        """
        error = RuntimeError("upstream failure")
        error.usage = SimpleNamespace(
            prompt_tokens=100,
            completion_tokens=10,
            total_tokens=110,
            cache_read_input_tokens=30,
            cache_creation_input_tokens=50,
        )

        run_hook(mw.proxy_handler_instance.async_log_failure_event(
            base_kwargs(), error, NOW, NOW
        ))

        args = submitted_args(mock_submit)
        assert args["cache_read_token_count"] == 30
        assert args["cache_creation_token_count"] == 50
        # The event names no upstream, so the prompt count is reported exactly
        # as LiteLLM built it. Only an Anthropic row may have its cache overlap
        # removed here -- see test_proxy_cache_token_parity.py (BACK-3334).
        assert args["input_token_count"] == 100
        assert args["output_token_count"] == 10
        assert args["total_token_count"] == 110
