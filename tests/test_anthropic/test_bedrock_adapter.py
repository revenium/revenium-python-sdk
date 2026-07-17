"""
Tests for Bedrock adapter functionality.
"""

import json
import pytest
from unittest.mock import patch, MagicMock
from botocore.stub import Stubber

from revenium_middleware.anthropic.bedrock_adapter import (
    _model_id,
    bedrock_invoke,
    bedrock_invoke_stream,
    BedrockStreamIterator,
    BedrockStreamWrapper,
    create_bedrock_payload,
    create_anthropic_response,
    get_bedrock_client,
    _import_boto3
)


class TestModelMapping:
    """Test model ID mapping functionality."""

    def test_known_model_mapping(self):
        """Test mapping for known models."""
        assert _model_id("claude-opus-4-7") == "anthropic.claude-opus-4-7"
        assert _model_id("us.claude-opus-4-7") == "us.anthropic.claude-opus-4-7"
        assert _model_id("eu.claude-opus-4-7") == "eu.anthropic.claude-opus-4-7"
        assert _model_id("au.claude-opus-4-7") == "au.anthropic.claude-opus-4-7"
        assert _model_id("global.claude-opus-4-7") == "global.anthropic.claude-opus-4-7"
        assert _model_id("claude-3-opus-20240229") == "anthropic.claude-3-opus-20240229-v1:0"
        assert _model_id("claude-3-sonnet-20240229") == "anthropic.claude-3-sonnet-20240229-v1:0"
        assert _model_id("claude-3-haiku-20240307") == "us.anthropic.claude-3-5-haiku-20241022-v1:0"

    def test_unknown_model_mapping(self):
        """Test mapping for unknown models uses fallback format."""
        assert _model_id("claude-4-future") == "anthropic.claude-4-future"
        assert _model_id("custom-model") == "anthropic.custom-model"

    def test_qualified_regional_id_passes_through_unchanged(self):
        """A fully qualified inference-profile ID is preserved byte-for-byte."""
        model = "us.anthropic.claude-opus-4-5-20251101-v1:0"
        assert _model_id(model) == model

    def test_qualified_global_id_passes_through_unchanged(self):
        model = "global.anthropic.claude-opus-4-8"
        assert _model_id(model) == model

    def test_qualified_non_anthropic_id_passes_through_unchanged(self):
        """Already-qualified IDs from other families are not rewritten."""
        model = "us.amazon.nova-micro-v1:0"
        assert _model_id(model) == model

    def test_arn_passes_through_unchanged(self):
        arn = (
            "arn:aws:bedrock:us-east-1:123456789012:inference-profile/"
            "us.anthropic.claude-opus-4-5-20251101-v1:0"
        )
        assert _model_id(arn) == arn

    def test_arn_without_provider_segment_passes_through_unchanged(self):
        """An ARN whose path has no provider substring exercises the arn:
        branch in isolation."""
        arn = (
            "arn:aws:bedrock:us-east-1:123456789012:"
            "application-inference-profile/profile-id"
        )
        assert _model_id(arn) == arn

    def test_unmapped_regional_bare_name_takes_anthropic_fallback(self):
        """Unmapped IDs without a provider segment get the bare-name fallback.

        Callers must fully qualify unmapped regional or non-Anthropic IDs;
        the adapter does not infer a provider segment for them.
        """
        assert _model_id("eu.new-model") == "anthropic.eu.new-model"

    def test_no_result_repeats_a_provider_prefix(self):
        cases = [
            "claude-opus-4-7",
            "us.claude-opus-4-7",
            "eu.claude-opus-4-7",
            "au.claude-opus-4-7",
            "global.claude-opus-4-7",
            "claude-3-opus-20240229",
            "claude-3-haiku-20240307",
            "us.anthropic.claude-opus-4-5-20251101-v1:0",
            "global.anthropic.claude-opus-4-8",
            "us.amazon.nova-micro-v1:0",
            "arn:aws:bedrock:us-east-1:123456789012:inference-profile/"
            "us.anthropic.claude-opus-4-5-20251101-v1:0",
            "claude-4-future",
            "eu.new-model",
        ]
        for model in cases:
            result = _model_id(model)
            assert result.count("anthropic.") <= 1, (model, result)
            assert result.count("amazon.") <= 1, (model, result)


class TestBoto3Import:
    """Test boto3 import handling."""

    def test_import_boto3_success(self):
        """Test successful boto3 import."""
        boto3 = _import_boto3()
        assert boto3 is not None

    @patch('builtins.__import__', side_effect=ImportError())
    def test_import_boto3_failure(self, mock_import):
        """Test boto3 import failure raises helpful error."""
        with pytest.raises(ImportError) as exc_info:
            _import_boto3()
        
        assert "boto3 is required for Bedrock support" in str(exc_info.value)
        assert "pip install revenium-python-sdk[anthropic]" in str(exc_info.value)


class TestBedrockInvoke:
    """Test Bedrock invoke functionality."""

    @patch('revenium_middleware.anthropic.bedrock_adapter.get_bedrock_client')
    def test_bedrock_invoke_success(self, mock_get_client):
        """Test successful Bedrock invocation."""
        # Mock the boto3 client
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        
        # Mock response
        mock_response_body = {
            "content": [
                {"type": "text", "text": "Hello, world!"}
            ],
            "usage": {
                "inputTokens": 10,
                "outputTokens": 5
            }
        }
        
        mock_response = {
            "body": MagicMock()
        }
        mock_response["body"].read.return_value = json.dumps(mock_response_body).encode()
        mock_client.invoke_model.return_value = mock_response
        
        # Test the function
        payload = {"messages": [{"role": "user", "content": "Hello"}]}
        text, input_tokens, output_tokens, cache_creation_tokens, cache_read_tokens = bedrock_invoke("claude-3-sonnet-20240229", payload)

        # Verify results
        assert text == "Hello, world!"
        assert input_tokens == 10
        assert output_tokens == 5
        assert cache_creation_tokens == 0
        assert cache_read_tokens == 0
        
        # Verify client was called correctly
        mock_client.invoke_model.assert_called_once_with(
            modelId="anthropic.claude-3-sonnet-20240229-v1:0",
            body=json.dumps(payload),
            accept="application/json"
        )

    @patch('revenium_middleware.anthropic.bedrock_adapter.get_bedrock_client')
    def test_bedrock_invoke_multiple_content_blocks(self, mock_get_client):
        """Test Bedrock invocation with multiple content blocks."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        
        # Mock response with multiple text blocks
        mock_response_body = {
            "content": [
                {"type": "text", "text": "Hello, "},
                {"type": "text", "text": "world!"},
                {"type": "image", "data": "base64data"}  # Should be ignored
            ],
            "usage": {
                "inputTokens": 15,
                "outputTokens": 8
            }
        }
        
        mock_response = {"body": MagicMock()}
        mock_response["body"].read.return_value = json.dumps(mock_response_body).encode()
        mock_client.invoke_model.return_value = mock_response
        
        payload = {"messages": [{"role": "user", "content": "Hello"}]}
        text, input_tokens, output_tokens, cache_creation_tokens, cache_read_tokens = bedrock_invoke("claude-3-haiku-20240307", payload)

        assert text == "Hello, world!"  # Concatenated text blocks only
        assert input_tokens == 15
        assert output_tokens == 8
        assert cache_creation_tokens == 0
        assert cache_read_tokens == 0

    @patch('revenium_middleware.anthropic.bedrock_adapter.get_bedrock_client')
    def test_bedrock_invoke_error_handling(self, mock_get_client):
        """Test error handling in Bedrock invocation."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        
        # Mock an exception
        mock_client.invoke_model.side_effect = Exception("AWS Error")
        
        payload = {"messages": [{"role": "user", "content": "Hello"}]}
        
        with pytest.raises(Exception) as exc_info:
            bedrock_invoke("claude-3-opus-20240229", payload)
        
        assert "AWS Error" in str(exc_info.value)

    @patch('revenium_middleware.anthropic.bedrock_adapter.get_bedrock_client')
    def test_bedrock_invoke_cache_tokens_extracted(self, mock_get_client):
        """Test that cache token fields are extracted from Bedrock response usage block."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        mock_response_body = {
            "content": [{"type": "text", "text": "Cached response"}],
            "usage": {
                "inputTokens": 20,
                "outputTokens": 10,
                "cacheWriteInputTokensCount": 150,
                "cacheReadInputTokensCount": 75,
            }
        }
        mock_response = {"body": MagicMock()}
        mock_response["body"].read.return_value = json.dumps(mock_response_body).encode()
        mock_client.invoke_model.return_value = mock_response

        payload = {"messages": [{"role": "user", "content": "Hello"}]}
        text, input_tokens, output_tokens, cache_creation_tokens, cache_read_tokens = bedrock_invoke(
            "claude-3-sonnet-20240229", payload
        )

        assert text == "Cached response"
        assert input_tokens == 20
        assert output_tokens == 10
        assert cache_creation_tokens == 150
        assert cache_read_tokens == 75

    @patch('revenium_middleware.anthropic.bedrock_adapter.get_bedrock_client')
    def test_bedrock_invoke_no_cache_tokens_defaults_to_zero(self, mock_get_client):
        """Test that absent cache token fields in Bedrock response default to 0."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        mock_response_body = {
            "content": [{"type": "text", "text": "Non-caching response"}],
            "usage": {
                "inputTokens": 12,
                "outputTokens": 6,
            }
        }
        mock_response = {"body": MagicMock()}
        mock_response["body"].read.return_value = json.dumps(mock_response_body).encode()
        mock_client.invoke_model.return_value = mock_response

        payload = {"messages": [{"role": "user", "content": "Hello"}]}
        text, input_tokens, output_tokens, cache_creation_tokens, cache_read_tokens = bedrock_invoke(
            "claude-3-sonnet-20240229", payload
        )

        assert cache_creation_tokens == 0
        assert cache_read_tokens == 0

    @patch.dict('os.environ', {'AWS_REGION': 'us-west-2'})
    @patch('revenium_middleware.anthropic.bedrock_adapter.get_bedrock_client')
    def test_bedrock_invoke_uses_env_region(self, mock_get_client):
        """Test that bedrock_invoke uses AWS_REGION environment variable."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        
        mock_response_body = {
            "content": [{"type": "text", "text": "Hello"}],
            "usage": {"inputTokens": 5, "outputTokens": 3}
        }
        mock_response = {"body": MagicMock()}
        mock_response["body"].read.return_value = json.dumps(mock_response_body).encode()
        mock_client.invoke_model.return_value = mock_response
        
        payload = {"messages": [{"role": "user", "content": "Hello"}]}
        bedrock_invoke("claude-3-sonnet-20240229", payload)
        
        # Verify the client was created with the correct region
        mock_get_client.assert_called_with("us-west-2")


class TestPayloadCreation:
    """Test Bedrock payload creation."""

    def test_create_basic_payload(self):
        """Test creating basic payload."""
        messages = [{"role": "user", "content": "Hello"}]
        payload = create_bedrock_payload(messages, max_tokens=100)
        
        assert payload["messages"] == messages
        assert payload["max_tokens"] == 100
        assert payload["anthropic_version"] == "bedrock-2023-05-31"

    def test_create_payload_with_optional_params(self):
        """Test creating payload with optional parameters."""
        messages = [{"role": "user", "content": "Hello"}]
        payload = create_bedrock_payload(
            messages,
            max_tokens=200,
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            system="You are helpful",
            stop_sequences=["STOP"]
        )
        
        assert payload["max_tokens"] == 200
        assert payload["temperature"] == 0.7
        assert payload["top_p"] == 0.9
        assert payload["top_k"] == 50
        assert payload["system"] == "You are helpful"
        assert payload["stop_sequences"] == ["STOP"]


class TestResponseCreation:
    """Test Anthropic response creation."""

    def test_create_anthropic_response(self):
        """Test creating Anthropic-compatible response."""
        response = create_anthropic_response(
            text="Hello, world!",
            input_tokens=10,
            output_tokens=5,
            model="claude-3-sonnet-20240229",
            request_id="test-123"
        )
        
        assert response["id"] == "test-123"
        assert response["type"] == "message"
        assert response["role"] == "assistant"
        assert response["model"] == "claude-3-sonnet-20240229"
        assert response["content"][0]["text"] == "Hello, world!"
        assert response["usage"]["input_tokens"] == 10
        assert response["usage"]["output_tokens"] == 5
        assert response["usage"]["total_tokens"] == 15
        assert response["stop_reason"] == "end_turn"

    def test_create_anthropic_response_auto_id(self):
        """Test creating response with auto-generated ID."""
        response = create_anthropic_response(
            text="Test",
            input_tokens=5,
            output_tokens=3,
            model="claude-3-haiku-20240307"
        )
        
        assert response["id"].startswith("msg_bedrock_")
        assert len(response["id"]) > 12  # Has the prefix plus generated number


class TestBedrockStreamIterator:
    """Test BedrockStreamIterator functionality."""

    @staticmethod
    def _chunk_event(chunk_data):
        return {"chunk": {"bytes": json.dumps(chunk_data).encode("utf-8")}}

    @staticmethod
    def _metadata_event(usage=None, metrics=None):
        return {"metadata": {"usage": usage, "metrics": metrics}}

    def test_bedrock_stream_iterator_creation(self):
        """Test creating a BedrockStreamIterator."""
        model = "claude-3-haiku-20240307"
        payload = {"messages": [{"role": "user", "content": "Hello"}]}
        region = "us-east-1"

        iterator = BedrockStreamIterator(model, payload, region)

        assert iterator.model == model
        assert iterator.payload == payload
        assert iterator.region == region
        assert iterator.accumulated_text == ""
        assert iterator.input_tokens == 0
        assert iterator.output_tokens == 0
        assert not iterator._started

    @patch('revenium_middleware.anthropic.bedrock_adapter.get_bedrock_client')
    def test_bedrock_stream_iterator_processing(self, mock_get_client):
        """Test BedrockStreamIterator processing streaming response."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        mock_events = [
            self._chunk_event({"type": "content_block_delta", "delta": {"text": "Hello"}}),
            self._chunk_event({"type": "content_block_delta", "delta": {"text": " world"}}),
            self._chunk_event({"type": "message_stop", "usage": {"inputTokens": 10, "outputTokens": 5}})
        ]

        mock_response = {"body": iter(mock_events)}
        mock_client.invoke_model_with_response_stream.return_value = mock_response

        iterator = BedrockStreamIterator("claude-3-haiku-20240307", {}, "us-east-1")

        # Collect all chunks
        chunks = list(iterator)

        assert chunks == ["Hello", " world"]
        assert iterator.accumulated_text == "Hello world"
        assert iterator.input_tokens == 10
        assert iterator.output_tokens == 5

    def test_bedrock_stream_iterator_cache_tokens_from_message_stop(self):
        """Test that BedrockStreamIterator extracts cache tokens from message_stop usage block."""
        mock_events = [
            self._chunk_event({"type": "content_block_delta", "delta": {"text": "Cached"}}),
            self._chunk_event({
                "type": "message_stop",
                "usage": {
                    "inputTokens": 30,
                    "outputTokens": 12,
                    "cacheWriteInputTokensCount": 200,
                    "cacheReadInputTokensCount": 100,
                }
            }),
        ]

        iterator = BedrockStreamIterator("claude-3-haiku-20240307", {}, "us-east-1")
        list(iterator._process_stream(iter(mock_events)))

        assert iterator.cache_creation_tokens == 200
        assert iterator.cache_read_tokens == 100
        assert iterator.input_tokens == 30
        assert iterator.output_tokens == 12

    def test_bedrock_stream_iterator_cache_tokens_from_message_delta(self):
        """Test usage is extracted from Anthropic/Bedrock message_delta stream events."""
        mock_events = [
            self._chunk_event({"type": "content_block_delta", "delta": {"text": "Cached"}}),
            self._chunk_event({
                "type": "message_delta",
                "usage": {
                    "output_tokens": 12,
                    "cache_write_input_tokens_count": 200,
                    "cache_read_input_tokens_count": 100,
                }
            }),
            self._chunk_event({"type": "message_stop"}),
        ]

        iterator = BedrockStreamIterator("claude-3-haiku-20240307", {}, "us-east-1")
        list(iterator._process_stream(iter(mock_events)))

        assert iterator.cache_creation_tokens == 200
        assert iterator.cache_read_tokens == 100
        assert iterator.output_tokens == 12

    def test_bedrock_stream_iterator_usage_from_message_start_message(self):
        """Test usage is extracted from nested Anthropic message_start message usage."""
        mock_events = [
            self._chunk_event({
                "type": "message_start",
                "message": {
                    "usage": {
                        "input_tokens": 30,
                        "cache_creation_input_tokens": 200,
                        "cache_read_input_tokens": 100,
                    }
                }
            }),
            self._chunk_event({
                "type": "message_delta",
                "usage": {"output_tokens": 12}
            }),
        ]

        iterator = BedrockStreamIterator("claude-3-haiku-20240307", {}, "us-east-1")
        list(iterator._process_stream(iter(mock_events)))

        assert iterator.cache_creation_tokens == 200
        assert iterator.cache_read_tokens == 100
        assert iterator.input_tokens == 30
        assert iterator.output_tokens == 12

    def test_bedrock_stream_iterator_cache_tokens_from_metadata_event(self):
        """Test usage is extracted from Bedrock metadata stream events."""
        mock_events = [
            self._metadata_event({
                "inputTokens": 30,
                "outputTokens": 12,
                "cacheWriteInputTokens": 200,
                "cacheReadInputTokens": 100,
            }),
        ]

        iterator = BedrockStreamIterator("claude-3-haiku-20240307", {}, "us-east-1")
        list(iterator._process_stream(iter(mock_events)))

        assert iterator.cache_creation_tokens == 200
        assert iterator.cache_read_tokens == 100
        assert iterator.input_tokens == 30
        assert iterator.output_tokens == 12

    def test_bedrock_stream_iterator_ignores_metadata_event_with_null_usage(self):
        """Test null metadata usage blocks do not crash stream processing."""
        iterator = BedrockStreamIterator("claude-3-haiku-20240307", {}, "us-east-1")
        list(iterator._process_stream(iter([self._metadata_event(usage=None, metrics=None)])))

        assert iterator.cache_creation_tokens == 0
        assert iterator.cache_read_tokens == 0
        assert iterator.input_tokens == 0
        assert iterator.output_tokens == 0

    def test_bedrock_stream_iterator_ignores_malformed_usage_blocks(self):
        """Test malformed external usage payloads do not crash stream processing."""
        mock_events = [
            {"metadata": "not-a-dict"},
            self._chunk_event({
                "type": "message_delta",
                "usage": "not-a-dict",
                "amazon-bedrock-invocationMetrics": ["not", "a", "dict"],
            }),
            self._chunk_event({
                "type": "message_start",
                "message": {"usage": {"input_tokens": -1, "output_tokens": True}},
            }),
            self._chunk_event({
                "type": "message_delta",
                "usage": {"output_tokens": "12"},
            }),
        ]

        iterator = BedrockStreamIterator("claude-3-haiku-20240307", {}, "us-east-1")
        list(iterator._process_stream(iter(mock_events)))

        assert iterator.input_tokens == 0
        assert iterator.output_tokens == 12

    def test_bedrock_stream_iterator_preserves_explicit_zero_usage_values(self):
        """Test explicit zero token values overwrite previously parsed counters."""
        mock_events = [
            self._metadata_event({
                "inputTokens": 0,
                "outputTokens": 0,
                "cacheWriteInputTokens": 0,
                "cacheReadInputTokens": 0,
            }),
        ]

        iterator = BedrockStreamIterator("claude-3-haiku-20240307", {}, "us-east-1")
        iterator.input_tokens = 30
        iterator.output_tokens = 12
        iterator.cache_creation_tokens = 200
        iterator.cache_read_tokens = 100

        list(iterator._process_stream(iter(mock_events)))

        assert iterator.cache_creation_tokens == 0
        assert iterator.cache_read_tokens == 0
        assert iterator.input_tokens == 0
        assert iterator.output_tokens == 0


class TestBedrockInvokeStream:
    """Test bedrock_invoke_stream function."""

    def test_bedrock_invoke_stream_returns_iterator(self):
        """Test that bedrock_invoke_stream returns a BedrockStreamIterator."""
        model = "claude-3-haiku-20240307"
        payload = {"messages": [{"role": "user", "content": "Hello"}]}
        region = "us-east-1"

        result = bedrock_invoke_stream(model, payload, region)

        assert isinstance(result, BedrockStreamIterator)
        assert result.model == model
        assert result.payload == payload
        assert result.region == region


class TestBedrockStreamWrapper:
    """Test BedrockStreamWrapper functionality."""

    def test_bedrock_stream_wrapper_creation(self):
        """Test creating a BedrockStreamWrapper."""
        import datetime

        model = "claude-3-haiku-20240307"
        payload = {"messages": [{"role": "user", "content": "Hello"}]}
        region = "us-east-1"
        usage_metadata = {"trace_id": "test-123", "organization_id": "anthropic-python-bedrock-wrapper"}
        request_time_dt = datetime.datetime.now(datetime.timezone.utc)
        request_time = request_time_dt.strftime("%Y-%m-%dT%H:%M:%SZ")

        wrapper = BedrockStreamWrapper(
            model=model,
            payload=payload,
            region=region,
            usage_metadata=usage_metadata,
            request_time_dt=request_time_dt,
            request_time=request_time
        )

        assert wrapper.model == model
        assert wrapper.payload == payload
        assert wrapper.region == region
        assert wrapper.usage_metadata == usage_metadata
        assert wrapper.request_time_dt == request_time_dt
        assert wrapper.request_time == request_time
        assert wrapper.accumulated_text == ""

    @patch('revenium_middleware.anthropic.middleware._get_thread_safe_client')
    @patch('revenium_middleware.anthropic.bedrock_adapter.bedrock_invoke_stream')
    def test_bedrock_stream_wrapper_context_manager(self, mock_bedrock_invoke_stream, mock_get_client):
        """Test BedrockStreamWrapper as context manager."""
        import datetime

        # Mock the metering client to prevent real API calls
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        mock_iterator = MagicMock()
        mock_bedrock_invoke_stream.return_value = mock_iterator

        wrapper = BedrockStreamWrapper(
            model="claude-3-haiku-20240307",
            payload={},
            region="us-east-1",
            usage_metadata={"organization_id": "anthropic-python-context-manager"},
            request_time_dt=datetime.datetime.now(datetime.timezone.utc),
            request_time="2023-01-01T00:00:00Z"
        )

        # Test context manager
        with wrapper as w:
            assert w is wrapper
            assert wrapper.stream_iterator is mock_iterator

        # Verify bedrock_invoke_stream was called
        mock_bedrock_invoke_stream.assert_called_once_with(
            wrapper.model, wrapper.payload, wrapper.region
        )

    @patch('revenium_middleware.anthropic.middleware._get_thread_safe_client')
    @patch('revenium_middleware.anthropic.bedrock_adapter.bedrock_invoke_stream')
    def test_bedrock_stream_wrapper_text_stream(self, mock_bedrock_invoke_stream, mock_get_client):
        """Test BedrockStreamWrapper text_stream property."""
        import datetime

        # Mock the metering client to prevent real API calls
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        # Create a proper mock iterator
        class MockIterator:
            def __init__(self, items):
                self.items = items
                self.index = 0
                self.input_tokens = 10
                self.output_tokens = 5

            def __iter__(self):
                return self

            def __next__(self):
                if self.index >= len(self.items):
                    raise StopIteration
                item = self.items[self.index]
                self.index += 1
                return item

        mock_iterator = MockIterator(["Hello", " world", "!"])
        mock_bedrock_invoke_stream.return_value = mock_iterator

        wrapper = BedrockStreamWrapper(
            model="claude-3-haiku-20240307",
            payload={},
            region="us-east-1",
            usage_metadata={"organization_id": "anthropic-python-text-stream"},
            request_time_dt=datetime.datetime.now(datetime.timezone.utc),
            request_time="2023-01-01T00:00:00Z"
        )

        with wrapper:
            chunks = list(wrapper.text_stream)

        assert chunks == ["Hello", " world", "!"]
        assert wrapper.accumulated_text == "Hello world!"

    @patch('revenium_middleware.anthropic.middleware._get_thread_safe_client')
    @patch('revenium_middleware.anthropic.bedrock_adapter.bedrock_invoke_stream')
    def test_bedrock_stream_wrapper_get_final_message(self, mock_bedrock_invoke_stream, mock_get_client):
        """Test BedrockStreamWrapper get_final_message method."""
        import datetime

        # Mock the metering client to prevent real API calls
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        # Mock iterator with token counts
        mock_iterator = MagicMock()
        mock_iterator.input_tokens = 15
        mock_iterator.output_tokens = 8
        mock_iterator.cache_creation_tokens = 0
        mock_iterator.cache_read_tokens = 0
        mock_bedrock_invoke_stream.return_value = mock_iterator

        wrapper = BedrockStreamWrapper(
            model="claude-3-haiku-20240307",
            payload={},
            region="us-east-1",
            usage_metadata={"organization_id": "anthropic-python-final-message"},
            request_time_dt=datetime.datetime.now(datetime.timezone.utc),
            request_time="2023-01-01T00:00:00Z"
        )
        wrapper.accumulated_text = "Test response"

        with wrapper:
            final_message = wrapper.get_final_message()

        assert final_message is not None
        assert hasattr(final_message, 'usage')
        assert final_message.usage.input_tokens == 15
        assert final_message.usage.output_tokens == 8
        assert final_message.usage.total_tokens == 23
        assert final_message.usage.cache_creation_input_tokens == 0
        assert final_message.usage.cache_read_input_tokens == 0
        assert final_message.model == "claude-3-haiku-20240307"
        assert final_message.content[0]["text"] == "Test response"
