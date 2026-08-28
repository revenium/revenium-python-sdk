"""Reasoning effort forwarding on the LiteLLM paths (BACK-2710).

Client mode reads ``effort`` from ``usage_metadata`` like every other
provider integration. Proxy mode has no usage_metadata channel -- its
per-call attribution arrives as ``x-revenium-*`` request headers -- so it
reads ``x-revenium-effort``.
"""
import asyncio
import datetime
from types import SimpleNamespace
from unittest.mock import patch

import pytest

# Both middleware modules import litellm at import time; skip when the
# optional dependency is absent (mirrors the other optional-provider suites).
pytest.importorskip("litellm")

from revenium_middleware.litellm.client import middleware as client_mw  # noqa: E402
from revenium_middleware.litellm.proxy import middleware as proxy_mw  # noqa: E402

NOW = datetime.datetime.now(datetime.timezone.utc)


def run_inline(coro):
    """Execute the metering coroutine synchronously so asserts see the call."""
    asyncio.run(coro)
    return SimpleNamespace(name="inline-metering")


def run_hook(coro):
    """Drive MiddlewareHandler's async_log_*_event coroutine to completion."""
    try:
        coro.send(None)
    except StopIteration:
        pass


def make_response():
    return SimpleNamespace(
        id="txn-effort",
        usage=SimpleNamespace(prompt_tokens=100, completion_tokens=10),
        model="gpt-4o-mini",
        finish_reason="stop",
        system_fingerprint=None,
    )


class SubscriptableResponse:
    """Stand-in for LiteLLM's ModelResponse: `response_obj["usage"]` plus `.id`."""

    def __init__(self, response_id, usage):
        self.id = response_id
        self._usage = usage

    def __getitem__(self, key):
        if key == "usage":
            return self._usage
        raise KeyError(key)


def submitted_args(mock_submit):
    assert mock_submit.call_count == 1
    return mock_submit.call_args[0][1]


@patch.object(client_mw, "run_async_in_thread", side_effect=run_inline)
@patch.object(client_mw, "submit_ai_event")
class TestLiteLLMClientEffort:
    def test_effort_is_forwarded_from_usage_metadata(self, mock_submit, _):
        client_mw.handle_response(make_response(), NOW, {"effort": "high"}, False)

        assert submitted_args(mock_submit)["effort"] == "high"

    def test_unrecognised_level_is_forwarded_unchanged(self, mock_submit, _):
        client_mw.handle_response(make_response(), NOW, {"effort": "hyper_9"}, False)

        assert submitted_args(mock_submit)["effort"] == "hyper_9"

    def test_unset_effort_is_omitted(self, mock_submit, _):
        client_mw.handle_response(make_response(), NOW, {}, False)

        assert "effort" not in submitted_args(mock_submit)


@patch.object(proxy_mw, "run_async_in_thread", side_effect=run_inline)
@patch.object(proxy_mw, "get_client", return_value=object())
@patch.object(proxy_mw, "submit_ai_event")
class TestLiteLLMProxyEffort:
    @staticmethod
    def _kwargs(headers):
        return {
            "model": "gpt-4o-mini",
            "litellm_params": {"metadata": {"headers": headers}},
        }

    @staticmethod
    def _response():
        usage = SimpleNamespace(prompt_tokens=100, completion_tokens=10, total_tokens=110)
        return SubscriptableResponse("txn-proxy-effort", usage)

    def test_success_path_reads_the_effort_header(self, mock_submit, _get_client, _run):
        run_hook(proxy_mw.proxy_handler_instance.async_log_success_event(
            self._kwargs({"x-revenium-effort": "xhigh"}), self._response(), NOW, NOW
        ))

        assert submitted_args(mock_submit)["effort"] == "xhigh"

    def test_success_path_omits_effort_without_the_header(self, mock_submit, _get_client, _run):
        run_hook(proxy_mw.proxy_handler_instance.async_log_success_event(
            self._kwargs({}), self._response(), NOW, NOW
        ))

        assert "effort" not in submitted_args(mock_submit)

    def test_failure_path_reads_the_effort_header(self, mock_submit, _get_client, _run):
        run_hook(proxy_mw.proxy_handler_instance.async_log_failure_event(
            self._kwargs({"x-revenium-effort": "low"}), self._response(), NOW, NOW
        ))

        assert submitted_args(mock_submit)["effort"] == "low"
