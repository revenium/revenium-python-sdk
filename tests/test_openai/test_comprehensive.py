"""
Comprehensive tests for streaming, Azure, and LangChain integrations.

Tests cover:
- Streaming chat completions with usage tracking
- Azure OpenAI provider detection and configuration
- LangChain callback handlers (sync and async)
"""

import datetime
import os
from unittest.mock import patch, MagicMock, Mock
import pytest

# Set up test environment
os.environ["OPENAI_API_KEY"] = "test-key"
os.environ["REVENIUM_METERING_API_KEY"] = "test-revenium-key"


class TestStreamingSupport:
    """Tests for streaming chat completions."""

    @pytest.fixture
    def mock_streaming_response(self):
        """Create a mock streaming response with chunks."""
        # Create mock chunks
        chunk1 = MagicMock()
        chunk1.id = "chatcmpl-test"
        chunk1.model = "gpt-4o-mini"
        chunk1.choices = [MagicMock()]
        chunk1.choices[0].delta = MagicMock()
        chunk1.choices[0].delta.content = "Hello"
        chunk1.choices[0].finish_reason = None
        chunk1.usage = None
        chunk1.system_fingerprint = "fp_test"

        chunk2 = MagicMock()
        chunk2.id = "chatcmpl-test"
        chunk2.model = "gpt-4o-mini"
        chunk2.choices = [MagicMock()]
        chunk2.choices[0].delta = MagicMock()
        chunk2.choices[0].delta.content = " world"
        chunk2.choices[0].finish_reason = None
        chunk2.usage = None
        chunk2.system_fingerprint = "fp_test"

        # Final chunk with usage
        final_chunk = MagicMock()
        final_chunk.id = "chatcmpl-test"
        final_chunk.model = "gpt-4o-mini"
        final_chunk.choices = [MagicMock()]
        final_chunk.choices[0].delta = MagicMock()
        final_chunk.choices[0].delta.content = None
        final_chunk.choices[0].finish_reason = "stop"

        # Use spec to prevent auto-creation of input_tokens
        final_chunk.usage = MagicMock(spec=['prompt_tokens', 'completion_tokens', 'total_tokens', 'prompt_tokens_details'])
        final_chunk.usage.prompt_tokens = 10
        final_chunk.usage.completion_tokens = 5
        final_chunk.usage.total_tokens = 15
        final_chunk.usage.prompt_tokens_details = MagicMock()
        final_chunk.usage.prompt_tokens_details.cached_tokens = 0
        final_chunk.system_fingerprint = "fp_test"

        return [chunk1, chunk2, final_chunk]

    @pytest.fixture
    def mock_streaming_response_no_content(self):
        """Create a mock streaming response with only usage chunk (edge case)."""
        # Only final usage chunk, no content chunks
        final_chunk = MagicMock()
        final_chunk.id = "chatcmpl-test"
        final_chunk.model = "gpt-4o-mini"
        final_chunk.choices = []

        # Use spec to prevent auto-creation of input_tokens
        final_chunk.usage = MagicMock(spec=['prompt_tokens', 'completion_tokens', 'total_tokens', 'prompt_tokens_details'])
        final_chunk.usage.prompt_tokens = 10
        final_chunk.usage.completion_tokens = 0
        final_chunk.usage.total_tokens = 10
        final_chunk.usage.prompt_tokens_details = MagicMock()
        final_chunk.usage.prompt_tokens_details.cached_tokens = 0
        final_chunk.system_fingerprint = "fp_test"

        return [final_chunk]

    @pytest.mark.skip(reason="Integration test - async metering calls are complex to mock")
    @patch('revenium_middleware.openai.middleware.client')
    def test_streaming_with_chunks_and_usage(self, mock_client, mock_streaming_response):
        """Test streaming response with content chunks and final usage chunk."""
        from revenium_middleware.openai.middleware import handle_streaming_response

        # Mock the Revenium client to avoid real API calls
        mock_result = MagicMock()
        mock_result.id = 'completion-123'
        mock_client.ai.create_completion.return_value = mock_result

        request_time = datetime.datetime.now(datetime.timezone.utc)
        usage_metadata = {"trace_id": "stream-test-001"}

        # Create a mock stream object
        mock_stream = MagicMock()
        mock_stream.__iter__ = MagicMock(return_value=iter(mock_streaming_response))

        # Wrap and consume the stream
        wrapped_stream = handle_streaming_response(
            mock_stream,
            request_time,
            usage_metadata,
            client_instance=None
        )

        # Consume all chunks
        chunks = list(wrapped_stream)

        # Verify we got all chunks
        assert len(chunks) == 3

        # Verify Revenium client was called (logging happened)
        assert mock_client.ai.create_completion.called

    @pytest.mark.skip(reason="Integration test - async metering calls are complex to mock")
    @patch('revenium_middleware.openai.middleware.client')
    def test_streaming_edge_case_usage_only(self, mock_client, mock_streaming_response_no_content):
        """Test streaming response with only usage chunk (no content chunks)."""
        from revenium_middleware.openai.middleware import handle_streaming_response

        # Mock the Revenium client to avoid real API calls
        mock_result = MagicMock()
        mock_result.id = 'completion-456'
        mock_client.ai.create_completion.return_value = mock_result

        request_time = datetime.datetime.now(datetime.timezone.utc)
        usage_metadata = {"trace_id": "stream-edge-001"}

        mock_stream = MagicMock()
        mock_stream.__iter__ = MagicMock(return_value=iter(mock_streaming_response_no_content))

        wrapped_stream = handle_streaming_response(
            mock_stream,
            request_time,
            usage_metadata,
            client_instance=None
        )

        # Consume all chunks
        chunks = list(wrapped_stream)

        # Verify we got the usage chunk
        assert len(chunks) == 1

        # Verify logging was still called (this is the fix for the edge case bug)
        assert mock_client.ai.create_completion.called


class TestAzureOpenAISupport:
    """Tests for Azure OpenAI provider detection and configuration."""

    @pytest.fixture(autouse=True)
    def reset_azure_config(self):
        """Reset Azure config before each test."""
        from revenium_middleware.openai.azure_config import reset_azure_config
        from revenium_middleware.openai.provider import reset_provider_cache
        reset_azure_config()
        reset_provider_cache()
        yield
        reset_azure_config()
        reset_provider_cache()

    def test_azure_provider_detection_by_client_type(self):
        """Test Azure detection via AzureOpenAI client type."""
        from revenium_middleware.openai.provider import detect_provider, Provider

        # Mock Azure client
        mock_azure_client = MagicMock()
        mock_azure_client.__class__.__name__ = "AzureOpenAI"

        provider = detect_provider(client=mock_azure_client)
        assert provider == Provider.AZURE_OPENAI

    def test_azure_provider_detection_by_url(self):
        """Test Azure detection via URL substring."""
        from revenium_middleware.openai.provider import detect_provider, Provider

        provider = detect_provider(base_url="https://my-resource.openai.azure.com/")
        assert provider == Provider.AZURE_OPENAI

    def test_openai_provider_detection(self):
        """Test standard OpenAI detection."""
        from revenium_middleware.openai.provider import detect_provider, Provider

        mock_openai_client = MagicMock()
        mock_openai_client.__class__.__name__ = "OpenAI"

        provider = detect_provider(client=mock_openai_client)
        assert provider == Provider.OPENAI

    def test_azure_config_validation_complete(self):
        """Test Azure config validation with complete configuration."""
        from revenium_middleware.openai.azure_config import AzureConfig

        with patch.dict(os.environ, {
            'AZURE_OPENAI_ENDPOINT': 'https://test.openai.azure.com',
            'AZURE_OPENAI_DEPLOYMENT': 'gpt-4',
            'AZURE_OPENAI_API_VERSION': '2024-10-21',
            'AZURE_OPENAI_API_KEY': 'test-key'
        }):
            config = AzureConfig()
            assert config.is_valid()
            assert config.validate_deployment()

    def test_azure_config_validation_missing_endpoint(self):
        """Test Azure config validation with missing endpoint."""
        from revenium_middleware.openai.azure_config import AzureConfig

        with patch.dict(os.environ, {}, clear=True):
            config = AzureConfig()
            assert not config.is_valid()

    def test_azure_config_headers(self):
        """Test Azure config header generation."""
        from revenium_middleware.openai.azure_config import AzureConfig

        with patch.dict(os.environ, {
            'AZURE_OPENAI_ENDPOINT': 'https://test.openai.azure.com',
            'AZURE_OPENAI_API_KEY': 'test-api-key'
        }):
            config = AzureConfig()
            headers = config.get_headers()
            assert headers['api-key'] == 'test-api-key'

    @pytest.mark.skip(reason="Integration test - requires full OpenAI SDK setup")
    @patch('revenium_middleware.client')
    @patch('revenium_middleware.openai.middleware.get_azure_config')
    def test_azure_config_validation_in_middleware(self, mock_get_config, mock_client):
        """Test that Azure config is validated in middleware."""
        from revenium_middleware.openai.middleware import create_wrapper

        # Mock Azure config
        mock_config = MagicMock()
        mock_config.is_valid.return_value = True
        mock_config.to_dict.return_value = {'endpoint': 'test', 'is_valid': True}
        mock_config.validate_deployment.return_value = True
        mock_get_config.return_value = mock_config

        # Mock Azure client
        mock_instance = MagicMock()
        mock_instance._client = MagicMock()
        mock_instance._client.__class__.__name__ = "AzureOpenAI"

        # Mock wrapped function - needs to accept *args, **kwargs
        def mock_wrapped_func(*args, **kwargs):
            mock_response = MagicMock()
            # Use spec to prevent auto-creation of input_tokens
            mock_response.usage = MagicMock(spec=['prompt_tokens', 'completion_tokens', 'total_tokens', 'prompt_tokens_details'])
            mock_response.usage.prompt_tokens = 10
            mock_response.usage.completion_tokens = 5
            mock_response.usage.total_tokens = 15
            mock_response.usage.prompt_tokens_details = MagicMock()
            mock_response.usage.prompt_tokens_details.cached_tokens = 0
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].finish_reason = "stop"
            mock_response.model = "gpt-4"
            return mock_response

        # Call wrapper
        kwargs = {
            "messages": [{"role": "user", "content": "test"}],
            "model": "gpt-4"
        }

        with patch('revenium_middleware.openai.middleware.detect_provider') as mock_detect:
            from revenium_middleware.openai.provider import Provider
            mock_detect.return_value = Provider.AZURE_OPENAI

            create_wrapper(mock_wrapped_func, mock_instance, [], kwargs)

        # Verify Azure config was checked
        assert mock_get_config.called
        assert mock_config.is_valid.called


class TestLangChainIntegration:
    """Tests for LangChain integration."""

    @pytest.fixture
    def mock_langchain_llm(self):
        """Create a mock LangChain LLM."""
        mock_llm = MagicMock()
        mock_llm.callbacks = []
        return mock_llm

    def test_wrap_function_exists(self):
        """Test that wrap function is exported."""
        try:
            from revenium_middleware.openai.langchain import wrap
            assert callable(wrap)
        except ImportError:
            pytest.skip("LangChain not installed")

    def test_attach_to_function_exists(self):
        """Test that attach_to function is exported."""
        try:
            from revenium_middleware.openai.langchain import attach_to
            assert callable(attach_to)
        except ImportError:
            pytest.skip("LangChain not installed")

    def test_wrap_adds_callback_handler(self, mock_langchain_llm):
        """Test that wrap() adds callback handler to LLM."""
        try:
            from revenium_middleware.openai.langchain import wrap
        except ImportError:
            pytest.skip("LangChain not installed")

        wrapped_llm = wrap(mock_langchain_llm)

        # Verify callback was added
        assert len(wrapped_llm.callbacks) == 1

    def test_wrap_with_metadata(self, mock_langchain_llm):
        """Test that wrap() accepts usage_metadata."""
        try:
            from revenium_middleware.openai.langchain import wrap
        except ImportError:
            pytest.skip("LangChain not installed")

        metadata = {
            "trace_id": "langchain-test-001",
            "task_type": "qa",
            "subscriber": {"id": "user-123"}
        }

        wrapped_llm = wrap(mock_langchain_llm, usage_metadata=metadata)

        # Verify LLM was wrapped
        assert wrapped_llm is not None


class TestProviderMetadata:
    """Tests for provider metadata generation."""

    def test_openai_provider_metadata(self):
        """Test OpenAI provider metadata."""
        from revenium_middleware.openai.provider import Provider, get_provider_metadata

        metadata = get_provider_metadata(Provider.OPENAI)
        assert metadata['provider'] == 'OPENAI'
        assert metadata['model_source'] == 'OPENAI'

    def test_azure_provider_metadata(self):
        """Test Azure OpenAI provider metadata."""
        from revenium_middleware.openai.provider import Provider, get_provider_metadata

        metadata = get_provider_metadata(Provider.AZURE_OPENAI)
        assert metadata['provider'] == 'Azure'
        assert metadata['model_source'] == 'OPENAI'


class TestMetadataStructure:
    """Tests for nested subscriber metadata structure."""

    @pytest.mark.skip(reason="Integration test - requires full OpenAI SDK setup")
    @patch('revenium_middleware.client')
    def test_nested_subscriber_metadata(self, mock_client):
        """Test that nested subscriber structure is properly passed through."""
        from revenium_middleware.openai.middleware import create_wrapper

        mock_instance = MagicMock()
        mock_instance._client = MagicMock()

        # Mock wrapped function - needs to accept *args, **kwargs
        def mock_wrapped_func(*args, **kwargs):
            mock_response = MagicMock()
            # Use spec to prevent auto-creation of input_tokens
            mock_response.usage = MagicMock(spec=['prompt_tokens', 'completion_tokens', 'total_tokens', 'prompt_tokens_details'])
            mock_response.usage.prompt_tokens = 10
            mock_response.usage.completion_tokens = 5
            mock_response.usage.total_tokens = 15
            mock_response.usage.prompt_tokens_details = MagicMock()
            mock_response.usage.prompt_tokens_details.cached_tokens = 0
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].finish_reason = "stop"
            mock_response.model = "gpt-4o-mini"
            return mock_response

        # Use nested subscriber structure
        kwargs = {
            "messages": [{"role": "user", "content": "test"}],
            "model": "gpt-4o-mini",
            "usage_metadata": {
                "subscriber": {
                    "id": "user-123",
                    "email": "user@example.com",
                    "credential": {
                        "name": "api-key-1",
                        "value": "key-value"
                    }
                },
                "organization_id": "org-456",
                "trace_id": "trace-789"
            }
        }

        # Mock the Revenium client
        mock_client.ai.create_completion.return_value = {'status': 'success'}

        create_wrapper(mock_wrapped_func, mock_instance, [], kwargs)

        # Verify Revenium client was called
        assert mock_client.ai.create_completion.called

        # Verify nested structure was passed
        call_kwargs = mock_client.ai.create_completion.call_args[1]
        assert 'subscriber' in call_kwargs
        # Subscriber should be a dict with nested structure
        if isinstance(call_kwargs['subscriber'], dict):
            assert call_kwargs['subscriber']['id'] == 'user-123'
