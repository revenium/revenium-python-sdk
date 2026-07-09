"""Real Bedrock provider errors must reach the caller, not trigger silent retries.

The middleware used to catch every Bedrock handler failure and re-invoke the
request through the SDK-native path. For errors raised AFTER the AWS call was
attempted (throttling, auth, quota), that masked the provider error from the
host app and risked double invocation. Fallback remains only for failures
where the request was never sent (payload validation, missing boto3).
"""
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from revenium_middleware.anthropic import middleware as anthropic_middleware
from revenium_middleware.anthropic.bedrock_adapter import (
    BedrockInvokeError,
    BedrockStreamError,
    BedrockValidationError,
)
from revenium_middleware.anthropic.provider import Provider


def middleware_fn(obj):
    # wrapt <2.2 leaves the patched name as a FunctionWrapper (middleware fn at
    # ._self_wrapper); wrapt >=2.2 leaves the plain wrapper function.
    return getattr(obj, "_self_wrapper", obj)


create_wrapper = middleware_fn(anthropic_middleware.create_wrapper)
stream_wrapper = middleware_fn(anthropic_middleware.stream_wrapper)


def bedrock_instance():
    return SimpleNamespace(_client=SimpleNamespace(aws_region="us-east-1"))

def call_kwargs():
    return {"messages": [{"role": "user", "content": "hi"}], "model": "claude-3-sonnet-20240229"}


@patch("revenium_middleware.anthropic.middleware.detect_provider", return_value=Provider.BEDROCK)
class TestSyncBedrockErrors:
    @patch("revenium_middleware.anthropic.middleware._handle_bedrock_request",
           side_effect=BedrockInvokeError("ThrottlingException: rate exceeded"))
    def test_invoke_error_propagates_without_silent_retry(self, mock_handle, mock_detect):
        mock_wrapped = MagicMock()

        with pytest.raises(BedrockInvokeError, match="ThrottlingException"):
            create_wrapper(mock_wrapped, bedrock_instance(), (), call_kwargs())

        # The provider error must not be masked by re-invoking another path.
        mock_wrapped.assert_not_called()

    @patch("revenium_middleware.anthropic.middleware._handle_bedrock_request",
           side_effect=RuntimeError("post-invoke conversion failed"))
    def test_unexpected_error_propagates_without_silent_retry(self, mock_handle, mock_detect):
        mock_wrapped = MagicMock()

        with pytest.raises(RuntimeError, match="post-invoke"):
            create_wrapper(mock_wrapped, bedrock_instance(), (), call_kwargs())

        mock_wrapped.assert_not_called()

    @patch("revenium_middleware.anthropic.middleware._handle_bedrock_request",
           side_effect=BedrockValidationError("bad payload"))
    def test_validation_error_still_falls_back(self, mock_handle, mock_detect):
        response = SimpleNamespace(id="msg_fallback", usage=None)
        mock_wrapped = MagicMock(return_value=response)

        result = create_wrapper(mock_wrapped, bedrock_instance(), (), call_kwargs())

        assert result is response
        mock_wrapped.assert_called_once()

    @patch("revenium_middleware.anthropic.middleware._handle_bedrock_request",
           side_effect=ImportError("No module named 'boto3'"))
    def test_import_error_still_falls_back(self, mock_handle, mock_detect):
        response = SimpleNamespace(id="msg_fallback", usage=None)
        mock_wrapped = MagicMock(return_value=response)

        result = create_wrapper(mock_wrapped, bedrock_instance(), (), call_kwargs())

        assert result is response
        mock_wrapped.assert_called_once()


@patch("revenium_middleware.anthropic.middleware.detect_provider", return_value=Provider.BEDROCK)
class TestStreamBedrockErrors:
    @patch("revenium_middleware.anthropic.middleware._handle_bedrock_stream_request",
           side_effect=BedrockStreamError("stream invoke failed"))
    def test_stream_error_propagates_without_silent_retry(self, mock_handle, mock_detect):
        mock_wrapped = MagicMock()

        with pytest.raises(BedrockStreamError, match="stream invoke failed"):
            stream_wrapper(mock_wrapped, bedrock_instance(), (), call_kwargs())

        mock_wrapped.assert_not_called()

    @patch("revenium_middleware.anthropic.middleware._handle_bedrock_stream_request",
           side_effect=BedrockValidationError("bad payload"))
    def test_validation_error_still_falls_back(self, mock_handle, mock_detect):
        mock_wrapped = MagicMock()

        stream_wrapper(mock_wrapped, bedrock_instance(), (), call_kwargs())

        mock_wrapped.assert_called_once()
