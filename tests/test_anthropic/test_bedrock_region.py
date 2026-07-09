"""Bedrock region resolution must follow the boto3/AWS-CLI convention.

The invoke path previously read only AWS_REGION (defaulting to us-east-1),
while the trace-metadata resolver honors AWS_REGION then AWS_DEFAULT_REGION --
so the call could run in a different region than the metering metadata
reported. The client instance's own configured region was also never used.
"""
import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from revenium_middleware.anthropic import middleware as anthropic_middleware
from revenium_middleware.anthropic.bedrock_adapter import (
    BedrockStreamIterator,
    bedrock_invoke,
)
from revenium_middleware.anthropic.provider import Provider


def make_invoke_response():
    body = {
        "content": [{"type": "text", "text": "Hello!"}],
        "usage": {"inputTokens": 10, "outputTokens": 5},
    }
    response = {"body": MagicMock()}
    response["body"].read.return_value = json.dumps(body).encode()
    return response


@pytest.fixture()
def clean_region_env(monkeypatch):
    monkeypatch.delenv("AWS_REGION", raising=False)
    monkeypatch.delenv("AWS_DEFAULT_REGION", raising=False)
    return monkeypatch


@patch("revenium_middleware.anthropic.bedrock_adapter.get_bedrock_client")
class TestBedrockInvokeRegionResolution:
    def test_honors_aws_default_region(self, mock_get_client, clean_region_env):
        clean_region_env.setenv("AWS_DEFAULT_REGION", "eu-west-1")
        mock_client = MagicMock()
        mock_client.invoke_model.return_value = make_invoke_response()
        mock_get_client.return_value = mock_client

        bedrock_invoke("claude-3-sonnet-20240229", {"messages": []})

        mock_get_client.assert_called_once_with("eu-west-1")

    def test_aws_region_wins_over_default(self, mock_get_client, clean_region_env):
        clean_region_env.setenv("AWS_REGION", "us-west-2")
        clean_region_env.setenv("AWS_DEFAULT_REGION", "eu-west-1")
        mock_client = MagicMock()
        mock_client.invoke_model.return_value = make_invoke_response()
        mock_get_client.return_value = mock_client

        bedrock_invoke("claude-3-sonnet-20240229", {"messages": []})

        mock_get_client.assert_called_once_with("us-west-2")

    def test_explicit_region_wins_over_env(self, mock_get_client, clean_region_env):
        clean_region_env.setenv("AWS_DEFAULT_REGION", "eu-west-1")
        mock_client = MagicMock()
        mock_client.invoke_model.return_value = make_invoke_response()
        mock_get_client.return_value = mock_client

        bedrock_invoke("claude-3-sonnet-20240229", {"messages": []}, region="ap-southeast-2")

        mock_get_client.assert_called_once_with("ap-southeast-2")

    def test_falls_back_to_us_east_1(self, mock_get_client, clean_region_env):
        mock_client = MagicMock()
        mock_client.invoke_model.return_value = make_invoke_response()
        mock_get_client.return_value = mock_client

        bedrock_invoke("claude-3-sonnet-20240229", {"messages": []})

        mock_get_client.assert_called_once_with("us-east-1")


class TestStreamIteratorRegionResolution:
    def test_honors_aws_default_region(self, clean_region_env):
        clean_region_env.setenv("AWS_DEFAULT_REGION", "eu-west-1")

        iterator = BedrockStreamIterator("claude-3-sonnet-20240229", {"messages": []})

        assert iterator.region == "eu-west-1"

    def test_matches_trace_metadata_resolution(self, clean_region_env):
        """The invoked region and the metering trace region must agree."""
        from revenium_middleware._core import trace_fields

        clean_region_env.setenv("AWS_DEFAULT_REGION", "eu-west-1")

        iterator = BedrockStreamIterator("claude-3-sonnet-20240229", {"messages": []})

        assert iterator.region == trace_fields.get_region()


def middleware_fn(obj):
    # wrapt <2.2 leaves the patched name as a FunctionWrapper (middleware fn at
    # ._self_wrapper); wrapt >=2.2 leaves the plain wrapper function.
    return getattr(obj, "_self_wrapper", obj)


class TestClientRegionThreading:
    @patch("revenium_middleware.anthropic.middleware._handle_bedrock_request")
    @patch("revenium_middleware.anthropic.middleware.detect_provider", return_value=Provider.BEDROCK)
    def test_create_wrapper_threads_client_region(self, mock_detect, mock_handle, clean_region_env):
        mock_handle.return_value = SimpleNamespace(id="bedrock-response")
        instance = SimpleNamespace(_client=SimpleNamespace(aws_region="ap-southeast-2"))
        create_wrapper = middleware_fn(anthropic_middleware.create_wrapper)

        kwargs = {"messages": [{"role": "user", "content": "hi"}], "model": "claude-3-sonnet-20240229"}
        result = create_wrapper(MagicMock(), instance, (), kwargs)

        assert result.id == "bedrock-response"
        assert mock_handle.call_args.kwargs.get("region") == "ap-southeast-2"

    @patch("revenium_middleware.anthropic.middleware._handle_bedrock_stream_request")
    @patch("revenium_middleware.anthropic.middleware.detect_provider", return_value=Provider.BEDROCK)
    def test_stream_wrapper_threads_client_region(self, mock_detect, mock_handle, clean_region_env):
        mock_handle.return_value = SimpleNamespace(id="bedrock-stream-response")
        instance = SimpleNamespace(_client=SimpleNamespace(aws_region="ap-southeast-2"))
        stream_wrapper = middleware_fn(anthropic_middleware.stream_wrapper)

        kwargs = {"messages": [{"role": "user", "content": "hi"}], "model": "claude-3-sonnet-20240229"}
        result = stream_wrapper(MagicMock(), instance, (), kwargs)

        assert result.id == "bedrock-stream-response"
        assert mock_handle.call_args.kwargs.get("region") == "ap-southeast-2"
