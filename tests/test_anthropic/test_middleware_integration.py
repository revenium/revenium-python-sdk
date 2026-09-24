"""
Tests for middleware integration with Bedrock support.
"""

import pytest
import json
from types import SimpleNamespace
from unittest.mock import patch, MagicMock
from freezegun import freeze_time
from revenium_middleware.anthropic.middleware import create_wrapper, stream_wrapper, _handle_bedrock_request, _handle_bedrock_stream_request
from revenium_middleware.anthropic.provider import Provider

anthropic = pytest.importorskip("anthropic")

FOUNDRY_RESOURCE = "example-resource"

requires_foundry_client = pytest.mark.skipif(
    not hasattr(anthropic, "AnthropicFoundry"),
    reason="Installed anthropic SDK predates the Foundry client classes"
)


class TestMiddlewareIntegration:
    """Test middleware integration with provider detection and routing."""

    def test_provider_detection_integration(self):
        """Test that provider detection works correctly in middleware context."""
        from revenium_middleware.anthropic.provider import detect_provider

        # Test Bedrock detection
        mock_client = MagicMock()
        mock_client.meta.service_model.service_name = "bedrock-runtime"

        result = detect_provider(client=mock_client)
        assert result == Provider.BEDROCK

        # Test Anthropic detection
        result = detect_provider(base_url="https://api.anthropic.com")
        assert result == Provider.ANTHROPIC

    @patch('revenium_middleware.anthropic.middleware.detect_provider')
    @patch('revenium_middleware.anthropic.middleware._handle_bedrock_request')
    def test_create_wrapper_bedrock_provider(self, mock_handle_bedrock, mock_detect_provider):
        """Test create_wrapper with Bedrock provider routes to Bedrock handler."""
        # Setup mocks
        mock_detect_provider.return_value = Provider.BEDROCK
        mock_bedrock_response = {
            "id": "bedrock-123",
            "model": "claude-3-sonnet-20240229",
            "content": [{"type": "text", "text": "Hello from Bedrock!"}],
            "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
        }
        mock_handle_bedrock.return_value = mock_bedrock_response
        
        mock_wrapped = MagicMock()

        # Test the wrapper
        kwargs = {
            "model": "claude-3-sonnet-20240229",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 100
        }
        
        result = create_wrapper(mock_wrapped, None, (), kwargs)
        
        # Verify Bedrock handler was called
        mock_handle_bedrock.assert_called_once()
        
        # Verify original function was NOT called
        mock_wrapped.assert_not_called()
        
        # Verify Bedrock response is returned
        assert result == mock_bedrock_response

    @pytest.mark.skip(reason="Test setup issue with Anthropic client validation - core functionality tested elsewhere")
    @patch('revenium_middleware.anthropic.middleware.detect_provider')
    @patch('revenium_middleware.anthropic.middleware._handle_bedrock_request')
    @patch('revenium_middleware.anthropic.middleware.run_async_in_thread')
    @patch('revenium_middleware.anthropic.middleware.client')
    def test_create_wrapper_bedrock_fallback_on_error(self, mock_client, mock_run_async,
                                                     mock_handle_bedrock, mock_detect_provider):
        """Test create_wrapper falls back to Anthropic when Bedrock fails."""
        # Setup mocks
        mock_detect_provider.return_value = Provider.BEDROCK
        mock_handle_bedrock.side_effect = Exception("Bedrock error")

        # Create a mock response
        mock_response = MagicMock()
        mock_response.id = "anthropic-fallback-123"
        mock_response.model = "claude-3-sonnet-20240229"
        mock_response.usage.input_tokens = 10
        mock_response.usage.output_tokens = 5
        mock_response.usage.cache_creation_input_tokens = 0
        mock_response.usage.cache_read_input_tokens = 0
        mock_response.stop_reason = "end_turn"

        # Create a simple function that acts as the wrapped function
        def simple_wrapped(**kwargs):
            return mock_response

        mock_wrapped = simple_wrapped

        mock_thread = MagicMock()
        mock_run_async.return_value = mock_thread
        mock_client.ai.create_completion = MagicMock()

        # Test the wrapper
        kwargs = {
            "model": "claude-3-sonnet-20240229",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 100
        }

        result = create_wrapper(mock_wrapped, None, (), kwargs)

        # Verify Bedrock handler was attempted
        mock_handle_bedrock.assert_called_once()

        # Note: Can't easily verify call to simple function, but we can verify the result

        # Verify fallback response is returned
        assert result == mock_response

    @pytest.mark.skip(reason="Test setup issue with Anthropic client validation - core functionality tested elsewhere")
    @patch.dict('os.environ', {'REVENIUM_BEDROCK_DISABLE': '1'})
    @patch('revenium_middleware.anthropic.middleware.detect_provider')
    @patch('revenium_middleware.anthropic.middleware.run_async_in_thread')
    @patch('revenium_middleware.anthropic.middleware.client')
    def test_create_wrapper_bedrock_disabled(self, mock_client, mock_run_async, mock_detect_provider):
        """Test create_wrapper respects REVENIUM_BEDROCK_DISABLE environment variable."""
        # Setup mocks - detect_provider should not be called when disabled

        # Create a mock response
        mock_response = MagicMock()
        mock_response.id = "test-123"
        mock_response.model = "claude-3-sonnet-20240229"
        mock_response.usage.input_tokens = 10
        mock_response.usage.output_tokens = 5
        mock_response.usage.cache_creation_input_tokens = 0
        mock_response.usage.cache_read_input_tokens = 0
        mock_response.stop_reason = "end_turn"

        # Create a simple function that acts as the wrapped function
        def simple_wrapped(**kwargs):
            return mock_response

        mock_wrapped = simple_wrapped

        mock_thread = MagicMock()
        mock_run_async.return_value = mock_thread
        mock_client.ai.create_completion = MagicMock()

        # Test the wrapper
        kwargs = {
            "model": "claude-3-sonnet-20240229",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 100
        }

        result = create_wrapper(mock_wrapped, None, (), kwargs)

        # Verify provider detection was not called
        mock_detect_provider.assert_not_called()

        # Note: Can't easily verify call to simple function, but we can verify the result

        # Verify response is returned
        assert result == mock_response

    @pytest.mark.skip(reason="Test setup issue with Anthropic client validation - core functionality tested elsewhere")
    @patch('revenium_middleware.anthropic.middleware.detect_provider')
    @patch('revenium_middleware.anthropic.middleware._handle_bedrock_stream_request')
    def test_stream_wrapper_bedrock_warning(self, mock_handle_bedrock_stream, mock_detect_provider, caplog):
        """Test stream_wrapper logs warning for Bedrock and falls back to Anthropic."""
        # Setup mocks
        mock_detect_provider.return_value = Provider.BEDROCK
        # Make the Bedrock handler raise a validation error to trigger fallback
        from revenium_middleware.anthropic.bedrock_adapter import BedrockValidationError
        mock_handle_bedrock_stream.side_effect = BedrockValidationError("messages cannot be empty")

        # Create a proper mock that can be called as a function
        mock_wrapped = MagicMock()
        mock_stream = MagicMock()
        mock_wrapped.return_value = mock_stream

        # Test the wrapper
        kwargs = {
            "model": "claude-3-sonnet-20240229",
            "messages": [],  # Empty messages to trigger validation error
            "stream": True
        }

        # Mock usage_context
        with patch('revenium_middleware.anthropic.middleware.usage_context') as mock_context:
            mock_context.get.return_value = {}

            result = stream_wrapper(mock_wrapped, None, (), kwargs)

        # Verify Bedrock was attempted
        mock_handle_bedrock_stream.assert_called_once()

        # Verify error was logged
        assert "Bedrock streaming request failed" in caplog.text
        assert "Falling back to direct Anthropic API" in caplog.text

        # Verify original function was called (fallback)
        mock_wrapped.assert_called_once_with(**kwargs)

        # Verify stream wrapper is returned
        assert result is not None


class TestBedrockRequestHandler:
    """Test the Bedrock request handler."""

    @patch('revenium_middleware.anthropic.middleware.bedrock_invoke')
    @patch('revenium_middleware.anthropic.middleware.create_bedrock_payload')
    @patch('revenium_middleware.anthropic.middleware.create_anthropic_response')
    @patch('revenium_middleware.anthropic.middleware._create_bedrock_metering_call')
    def test_handle_bedrock_request(self, mock_metering, mock_create_response,
                                   mock_create_payload, mock_bedrock_invoke):
        """Test _handle_bedrock_request function."""
        # Setup mocks
        mock_create_payload.return_value = {"messages": [{"role": "user", "content": "Hello"}]}
        mock_bedrock_invoke.return_value = ("Hello from Bedrock!", 10, 5, 0, 0)
        mock_response = {
            "id": "bedrock-123",
            "model": "claude-3-sonnet-20240229",
            "content": [{"type": "text", "text": "Hello from Bedrock!"}],
            "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
        }
        mock_create_response.return_value = mock_response

        # Test the function
        args = ()
        kwargs = {
            "model": "claude-3-sonnet-20240229",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 100
        }
        usage_metadata = {"trace_id": "test-trace", "organization_id": "anthropic-python-bedrock"}

        import datetime
        request_time_dt = datetime.datetime.now(datetime.timezone.utc)
        request_time = request_time_dt.strftime("%Y-%m-%dT%H:%M:%SZ")

        result = _handle_bedrock_request(args, kwargs, usage_metadata, request_time_dt, request_time)

        # Verify all functions were called correctly
        # The kwargs should exclude 'messages' when passed to create_bedrock_payload
        expected_kwargs = {k: v for k, v in kwargs.items() if k != "messages"}
        mock_create_payload.assert_called_once_with([{"role": "user", "content": "Hello"}], **expected_kwargs)
        mock_bedrock_invoke.assert_called_once()
        mock_create_response.assert_called_once_with(
            text="Hello from Bedrock!",
            input_tokens=10,
            output_tokens=5,
            model="claude-3-sonnet-20240229",
            cache_creation_tokens=0,
            cache_read_tokens=0
        )
        mock_metering.assert_called_once()

        # Verify response is returned
        assert result == mock_response

    @patch('revenium_middleware.anthropic.middleware.bedrock_invoke')
    @patch('revenium_middleware.anthropic.middleware.create_bedrock_payload')
    @patch('revenium_middleware.anthropic.middleware.create_anthropic_response')
    @patch('revenium_middleware.anthropic.middleware._safe_run_async_in_thread')
    @patch('revenium_middleware.anthropic.middleware._get_thread_safe_client')
    @patch('revenium_middleware.anthropic.middleware.submit_ai_event')
    def test_cache_tokens_flow_to_submit_ai_event(
        self, mock_submit, mock_get_client, mock_safe_run,
        mock_create_response, mock_create_payload, mock_bedrock_invoke
    ):
        """Regression test: cache token counts from bedrock_invoke flow to submit_ai_event payload.

        Reverting middleware.py:339-340 back to hardcoded 0 must cause this test to fail.
        """
        import asyncio
        import datetime

        # Return a 5-tuple with non-zero cache values
        mock_bedrock_invoke.return_value = ("Hello!", 10, 5, 100, 50)
        mock_create_payload.return_value = {"messages": [{"role": "user", "content": "Hello"}]}

        mock_response = MagicMock()
        mock_response.id = "bedrock-cache-test-123"
        mock_response.model = "claude-3-sonnet-20240229"
        mock_response.usage.input_tokens = 10
        mock_response.usage.output_tokens = 5
        mock_response.usage.total_tokens = 15
        mock_create_response.return_value = mock_response

        # Execute the metering coroutine synchronously so submit_ai_event is called before assertions
        def run_coro_sync(*args):
            asyncio.run(args[0]())
            return MagicMock()

        mock_safe_run.side_effect = run_coro_sync
        mock_get_client.return_value = MagicMock()

        args = ()
        kwargs = {
            "model": "claude-3-sonnet-20240229",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 100,
        }
        usage_metadata = {"trace_id": "cache-flow-test"}

        request_time_dt = datetime.datetime.now(datetime.timezone.utc)
        request_time = request_time_dt.strftime("%Y-%m-%dT%H:%M:%SZ")

        with patch('revenium_middleware.shutdown_event') as mock_shutdown:
            mock_shutdown.is_set.return_value = False
            _handle_bedrock_request(args, kwargs, usage_metadata, request_time_dt, request_time)

        # Assert submit_ai_event received the non-zero cache token counts
        mock_submit.assert_called_once()
        payload = mock_submit.call_args[0][1]
        assert payload["cache_creation_token_count"] == 100
        assert payload["cache_read_token_count"] == 50


class TestBedrockStreamingIntegration:
    """Test Bedrock streaming integration."""

    @patch('revenium_middleware.anthropic.middleware.detect_provider')
    @patch('revenium_middleware.anthropic.middleware._handle_bedrock_stream_request')
    @patch('revenium_middleware.anthropic.middleware.merge_metadata')
    def test_stream_wrapper_routes_to_bedrock_streaming(self, mock_merge_metadata, mock_handle_bedrock_stream, mock_detect_provider):
        """Test that stream_wrapper routes Bedrock requests to streaming handler."""
        # Setup mocks
        mock_detect_provider.return_value = Provider.BEDROCK
        mock_stream_wrapper = MagicMock()
        mock_handle_bedrock_stream.return_value = mock_stream_wrapper

        # Mock merge_metadata to return the expected metadata
        mock_merge_metadata.return_value = {"trace_id": "test-123", "organization_id": "anthropic-python-streaming"}

        mock_wrapped = MagicMock()

        # Test the wrapper
        kwargs = {
            "model": "claude-3-sonnet-20240229",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 100
        }

        result = stream_wrapper(mock_wrapped, None, (), kwargs)

        # Verify Bedrock streaming handler was called exactly once. The exact
        # positional shape of the call args depends on how wrapt packs
        # arguments and varies across wrapt releases (wrapt 2.1 packs the
        # wrapper's (wrapped, instance, args, kwargs) into the first
        # positional; wrapt 2.2 forwards them unmodified), so we assert on
        # the routing decision rather than on argument layout. The
        # _handle_bedrock_stream_request contract itself is exercised by
        # TestBedrockStreamingIntegration::test_handle_bedrock_stream_request.
        mock_handle_bedrock_stream.assert_called_once()

        # Verify result is the Bedrock stream wrapper
        assert result == mock_stream_wrapper

        # Verify original wrapped function was NOT called
        mock_wrapped.assert_not_called()

    @pytest.mark.skip(reason="Test setup issue with Anthropic client validation - core functionality tested elsewhere")
    @patch('revenium_middleware.anthropic.middleware.detect_provider')
    @patch('revenium_middleware.anthropic.middleware._handle_bedrock_stream_request')
    def test_stream_wrapper_fallback_on_bedrock_error(self, mock_handle_bedrock_stream, mock_detect_provider, caplog):
        """Test stream_wrapper falls back to Anthropic API when Bedrock streaming fails."""
        # Setup mocks
        mock_detect_provider.return_value = Provider.BEDROCK
        mock_handle_bedrock_stream.side_effect = Exception("Bedrock error")

        # Create a proper mock that can be called as a function
        mock_wrapped = MagicMock()
        mock_stream = MagicMock()
        mock_wrapped.return_value = mock_stream

        kwargs = {
            "model": "claude-3-sonnet-20240229",
            "messages": [{"role": "user", "content": "Hello"}]
        }

        # Mock usage_context
        with patch('revenium_middleware.anthropic.middleware.usage_context') as mock_context:
            mock_context.get.return_value = {}

            stream_wrapper(mock_wrapped, None, (), kwargs)

        # Verify Bedrock streaming was attempted
        mock_handle_bedrock_stream.assert_called_once()

        # Verify fallback to Anthropic API
        # Note: The wrapped function is called without instance parameter in fallback
        mock_wrapped.assert_called_once_with(**kwargs)

        # Verify error was logged
        assert "Bedrock streaming request failed" in caplog.text
        assert "Falling back to direct Anthropic API" in caplog.text

    @patch('revenium_middleware.anthropic.middleware.BedrockStreamWrapper')
    @patch('revenium_middleware.anthropic.middleware.create_bedrock_payload')
    def test_handle_bedrock_stream_request(self, mock_create_payload, mock_stream_wrapper_class):
        """Test _handle_bedrock_stream_request function."""
        import datetime

        # Setup mocks
        mock_payload = {"messages": [{"role": "user", "content": "Hello"}], "max_tokens": 100}
        mock_create_payload.return_value = mock_payload

        mock_stream_wrapper = MagicMock()
        mock_stream_wrapper_class.return_value = mock_stream_wrapper

        # Test data
        args = ()
        kwargs = {
            "model": "claude-3-haiku-20240307",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 100,
            "temperature": 0.7,
            "region": "us-west-2"
        }
        usage_metadata = {"trace_id": "test-456", "organization_id": "anthropic-python-bedrock-streaming"}
        request_time_dt = datetime.datetime.now(datetime.timezone.utc)
        request_time = request_time_dt.strftime("%Y-%m-%dT%H:%M:%SZ")

        # Call the function
        result = _handle_bedrock_stream_request(args, kwargs, usage_metadata, request_time_dt, request_time)

        # Verify payload creation
        mock_create_payload.assert_called_once()
        payload_call_args = mock_create_payload.call_args
        assert payload_call_args[0][0] == kwargs["messages"]  # messages
        # Check that kwargs were passed (excluding 'messages')
        payload_kwargs = payload_call_args[1]
        assert payload_kwargs["model"] == "claude-3-haiku-20240307"
        assert payload_kwargs["max_tokens"] == 100
        assert payload_kwargs["temperature"] == 0.7
        assert payload_kwargs["region"] == "us-west-2"

        # Verify BedrockStreamWrapper creation
        mock_stream_wrapper_class.assert_called_once_with(
            model="claude-3-haiku-20240307",
            payload=mock_payload,
            messages=kwargs["messages"],
            region="us-west-2",
            usage_metadata=usage_metadata,
            request_time_dt=request_time_dt,
            request_time=request_time
        )

        # Verify result
        assert result == mock_stream_wrapper

    @pytest.mark.skip(reason="Test setup issue with Anthropic client validation - core functionality tested elsewhere")
    @patch('revenium_middleware.anthropic.middleware.detect_provider')
    def test_stream_wrapper_preserves_anthropic_behavior(self, mock_detect_provider):
        """Test that stream_wrapper preserves existing Anthropic behavior."""
        # Setup mocks
        mock_detect_provider.return_value = Provider.ANTHROPIC

        # Create a proper mock that can be called as a function
        mock_wrapped = MagicMock()
        mock_stream = MagicMock()
        mock_wrapped.return_value = mock_stream

        kwargs = {
            "model": "claude-3-sonnet-20240229",
            "messages": [{"role": "user", "content": "Hello"}]
        }

        # Mock usage_context
        with patch('revenium_middleware.anthropic.middleware.usage_context') as mock_context:
            mock_context.get.return_value = {}

            result = stream_wrapper(mock_wrapped, None, (), kwargs)

        # Verify original wrapped function was called
        mock_wrapped.assert_called_once_with(**kwargs)

        # Verify result is wrapped in StreamWrapper
        assert result is not None
        # The result should be a StreamWrapper instance wrapping the mock_stream


class TestAsyncBedrockAttribution:
    """Bedrock-routed async calls must carry AWS provider metadata.

    The async wrapper serves Bedrock clients natively (no re-routing), so
    the only Bedrock-specific behaviour is attribution: detection must run
    and the detected provider must reach the metering payload.
    """

    @staticmethod
    def _wrapper():
        import revenium_middleware.anthropic.middleware as mw
        # wrapt < 2.2 leaves the patched name as a FunctionWrapper.
        return getattr(mw.async_create_wrapper, "_self_wrapper", mw.async_create_wrapper)

    @staticmethod
    def _run_metering_inline(fn):
        import asyncio
        import threading
        thread = threading.Thread(target=lambda: asyncio.run(fn()))
        thread.start()
        thread.join()
        return thread

    def _make_response(self):
        response = MagicMock()
        response.id = "msg_async_bedrock"
        response.model = "claude-3-5-sonnet-20241022"
        response.stop_reason = "end_turn"
        response.usage.input_tokens = 10
        response.usage.output_tokens = 5
        response.usage.cache_creation_input_tokens = 0
        response.usage.cache_read_input_tokens = 0
        return response

    def _run_create(self, mock_detect_value=None, kwargs=None, instance=None):
        """Drive the async create wrapper once and return its metering payload.

        mock_detect_value=None leaves the real detect_provider in place, so a
        test can exercise provider signals on ``instance._client`` end to end.
        """
        import asyncio
        from contextlib import ExitStack
        import revenium_middleware.anthropic.middleware as mw

        response = self._make_response()

        async def wrapped(*a, **k):
            return response

        payloads = []
        with ExitStack() as stack:
            if mock_detect_value is not None:
                stack.enter_context(patch.object(mw, 'detect_provider', return_value=mock_detect_value))
            stack.enter_context(patch.object(mw, 'submit_ai_event',
                                             side_effect=lambda op, args: payloads.append(args)))
            stack.enter_context(patch.object(mw, '_safe_run_async_in_thread',
                                             side_effect=self._run_metering_inline))
            wrapper = self._wrapper()
            instance = instance if instance is not None else MagicMock()
            result = asyncio.run(wrapper(
                wrapped, instance, (),
                {"model": "claude-3-5-sonnet-20241022",
                 "messages": [{"role": "user", "content": "hi"}],
                 "max_tokens": 16, **(kwargs or {})}))

        assert result is response
        assert len(payloads) == 1
        return payloads[0]

    def test_async_bedrock_create_emits_aws_provider(self):
        payload = self._run_create(Provider.BEDROCK)
        assert payload["provider"] == "AWS"
        assert payload["model_source"] == "ANTHROPIC"

    def test_async_direct_anthropic_create_unchanged(self):
        payload = self._run_create(Provider.ANTHROPIC)
        assert payload["provider"] == "ANTHROPIC"

    def test_async_bedrock_emits_exactly_once(self):
        payload = self._run_create(Provider.BEDROCK)
        assert payload["input_token_count"] == 10
        assert payload["output_token_count"] == 5

    def test_async_bedrock_disable_env_forces_anthropic(self, monkeypatch):
        """The switch is applied by the real detect_provider, not by the wrapper.

        A genuine Bedrock signal on the client is ignored while the switch is
        on, so the payload is direct Anthropic. (With detect_provider mocked to
        BEDROCK this would be untestable: the wrapper no longer second-guesses
        the detector, which is what lets Foundry survive the switch.)
        """
        monkeypatch.setenv("REVENIUM_BEDROCK_DISABLE", "1")
        instance = MagicMock()
        instance._client.meta.service_model.service_name = "bedrock-runtime"
        payload = self._run_create(None, instance=instance)
        assert payload["provider"] == "ANTHROPIC"

    @requires_foundry_client
    def test_async_foundry_create_keeps_foundry_label_with_bedrock_disabled(self, monkeypatch):
        monkeypatch.setenv("REVENIUM_BEDROCK_DISABLE", "1")
        instance = MagicMock()
        instance._client = anthropic.AnthropicFoundry(api_key="test-key", resource=FOUNDRY_RESOURCE)
        payload = self._run_create(None, instance=instance)
        assert payload["provider"] == "Foundry"
        assert payload["model_source"] == "ANTHROPIC"

    def test_async_bedrock_raw_stream_finalizes_with_detected_provider(self):
        import asyncio
        import revenium_middleware.anthropic.middleware as mw

        class FakeAsyncStream:
            def __init__(self):
                self._events = iter([MagicMock(type="message_stop")])

            def __aiter__(self):
                return self

            async def __anext__(self):
                try:
                    return next(self._events)
                except StopIteration:
                    raise StopAsyncIteration

        async def wrapped(*a, **k):
            return FakeAsyncStream()

        with patch.object(mw, 'detect_provider', return_value=Provider.BEDROCK), \
                patch.object(mw, '_meter_raw_stream') as mock_meter:
            wrapper = self._wrapper()

            async def consume():
                stream = await wrapper(
                    wrapped, MagicMock(), (),
                    {"model": "claude-3-5-sonnet-20241022",
                     "messages": [{"role": "user", "content": "hi"}],
                     "max_tokens": 16, "stream": True})
                async for _ in stream:
                    pass

            asyncio.run(consume())

        assert mock_meter.call_count == 1
        assert mock_meter.call_args[0][5] == Provider.BEDROCK


class TestFoundryAttribution:
    """Foundry-served calls must carry Foundry provider metadata.

    Foundry clients inherit the same patched methods as direct Anthropic
    clients, so the only Foundry-specific behaviour is attribution: detection
    must recognise the client and the resulting label must reach the metering
    payload without disturbing any other field.
    """

    @staticmethod
    def _run_metering_inline(fn):
        import asyncio
        import threading
        thread = threading.Thread(target=lambda: asyncio.run(fn()))
        thread.start()
        thread.join()
        return thread

    @staticmethod
    def _make_response():
        response = MagicMock()
        response.id = "msg_foundry"
        response.model = "claude-3-5-sonnet-20241022"
        response.stop_reason = "end_turn"
        response.usage.input_tokens = 11
        response.usage.output_tokens = 7
        response.usage.cache_creation_input_tokens = 0
        response.usage.cache_read_input_tokens = 0
        return response

    def _capture_payload(self, client_instance):
        """Run the sync create wrapper for one client and return its payload."""
        import revenium_middleware.anthropic.middleware as mw
        # wrapt < 2.2 leaves the patched name as a FunctionWrapper.
        wrapper = getattr(mw.create_wrapper, "_self_wrapper", mw.create_wrapper)

        response = self._make_response()
        wrapped = MagicMock(return_value=response)
        instance = MagicMock()
        instance._client = client_instance

        payloads = []
        with patch.object(mw, 'submit_ai_event',
                          side_effect=lambda op, args: payloads.append(args)), \
                patch.object(mw, '_safe_run_async_in_thread',
                             side_effect=self._run_metering_inline):
            result = wrapper(wrapped, instance, (),
                             {"model": "claude-3-5-sonnet-20241022",
                              "messages": [{"role": "user", "content": "hi"}],
                              "max_tokens": 16})

        assert result is response
        assert len(payloads) == 1
        return payloads[0]

    @requires_foundry_client
    def test_foundry_sync_create_emits_foundry_provider(self):
        client = anthropic.AnthropicFoundry(api_key="test-key", resource=FOUNDRY_RESOURCE)

        payload = self._capture_payload(client)

        assert payload["provider"] == "Foundry"
        assert payload["model_source"] == "ANTHROPIC"
        # Token counts come from the stubbed response, proving the wrapper did
        # real work rather than emitting an empty payload.
        assert payload["input_token_count"] == 11
        assert payload["output_token_count"] == 7

    @requires_foundry_client
    @freeze_time("2023-01-01T12:00:00Z")
    def test_foundry_payload_differs_from_anthropic_only_by_provider(self):
        foundry_client = anthropic.AnthropicFoundry(api_key="test-key", resource=FOUNDRY_RESOURCE)
        direct_client = anthropic.Anthropic(api_key="test-key")

        foundry_payload = self._capture_payload(foundry_client)
        direct_payload = self._capture_payload(direct_client)

        assert direct_payload["provider"] == "ANTHROPIC"
        differing = {key for key in set(foundry_payload) | set(direct_payload)
                     if foundry_payload.get(key) != direct_payload.get(key)}
        assert differing == {"provider"}


def _run_metering_inline(fn):
    """Run the metering coroutine to completion on a dedicated thread.

    Mirrors production, where _safe_run_async_in_thread dispatches to a thread,
    while keeping the assertion synchronous.
    """
    import asyncio
    import threading
    thread = threading.Thread(target=lambda: asyncio.run(fn()))
    thread.start()
    thread.join(timeout=10)
    return thread


class _FakeMessageStream:
    """Stands in for the client.messages.stream() context-manager helper."""

    def __init__(self, final_message):
        self._final_message = final_message

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False

    def get_final_message(self):
        return self._final_message

    @property
    def current_message_snapshot(self):
        """What the real MessageStream exposes as it accumulates events.

        Finalisation reads this rather than draining the stream, so the
        stand-in has to carry it too. A fully consumed stream's snapshot is
        its final message.
        """
        return self._final_message


class TestStreamProviderAttribution:
    """messages.stream() must attribute to the provider it detected.

    Finalisation used to label every SDK-native stream as direct Anthropic, so
    a Foundry client's streaming spend landed in direct-Anthropic totals even
    though the non-streaming path attributed it correctly.
    """

    @staticmethod
    def _final_message():
        # SimpleNamespace rather than MagicMock: the per-TTL cache extraction
        # inspects usage, and an auto-attribute double cannot express "the
        # response carried no TTL split".
        return SimpleNamespace(
            id="msg_stream_attribution",
            model="claude-3-5-sonnet-20241022",
            stop_reason="end_turn",
            usage=SimpleNamespace(
                input_tokens=13,
                output_tokens=5,
                cache_creation_input_tokens=0,
                cache_read_input_tokens=0,
            ),
        )

    def _capture_stream_payload(self, client_instance, monkeypatch, bedrock_disabled=False):
        """Drive the stream wrapper for one client and return its payload."""
        import revenium_middleware.anthropic.middleware as mw
        if bedrock_disabled:
            monkeypatch.setenv("REVENIUM_BEDROCK_DISABLE", "1")
        else:
            monkeypatch.delenv("REVENIUM_BEDROCK_DISABLE", raising=False)
        # wrapt < 2.2 leaves the patched name as a FunctionWrapper.
        wrapper = getattr(mw.stream_wrapper, "_self_wrapper", mw.stream_wrapper)

        instance = MagicMock()
        instance._client = client_instance
        wrapped = MagicMock(return_value=_FakeMessageStream(self._final_message()))

        payloads = []
        with patch.object(mw, 'submit_ai_event',
                          side_effect=lambda op, args: payloads.append(args)), \
                patch.object(mw, '_safe_run_async_in_thread',
                             side_effect=_run_metering_inline), \
                patch.object(mw, '_get_thread_safe_client', return_value=MagicMock()):
            with wrapper(wrapped, instance, (),
                         {"model": "claude-3-5-sonnet-20241022",
                          "messages": [{"role": "user", "content": "hi"}],
                          "max_tokens": 16}):
                pass

        assert len(payloads) == 1
        return payloads[0]

    @requires_foundry_client
    def test_foundry_stream_emits_foundry_provider(self, monkeypatch):
        client = anthropic.AnthropicFoundry(api_key="test-key", resource=FOUNDRY_RESOURCE)

        payload = self._capture_stream_payload(client, monkeypatch)

        assert payload["provider"] == "Foundry"
        assert payload["model_source"] == "ANTHROPIC"
        assert payload["is_streamed"] is True
        # Token counts come from the stubbed final message, proving the wrapper
        # metered a real finalisation rather than an empty payload.
        assert payload["input_token_count"] == 13
        assert payload["output_token_count"] == 5

    @requires_foundry_client
    def test_foundry_stream_keeps_foundry_label_with_bedrock_disabled(self, monkeypatch):
        """REVENIUM_BEDROCK_DISABLE used to skip detection on the stream path
        entirely, relabelling Foundry spend as direct Anthropic (Greptile P1 on
        BACK-2566). The switch now suppresses Bedrock signals only.
        """
        client = anthropic.AnthropicFoundry(api_key="test-key", resource=FOUNDRY_RESOURCE)

        payload = self._capture_stream_payload(client, monkeypatch, bedrock_disabled=True)

        assert payload["provider"] == "Foundry"
        assert payload["model_source"] == "ANTHROPIC"

    @requires_foundry_client
    def test_direct_anthropic_stream_still_emits_anthropic_provider(self, monkeypatch):
        client = anthropic.Anthropic(api_key="test-key")

        payload = self._capture_stream_payload(client, monkeypatch)

        assert payload["provider"] == "ANTHROPIC"
        assert payload["model_source"] == "ANTHROPIC"
        assert payload["is_streamed"] is True

    def test_bedrock_fallback_stream_still_emits_anthropic_provider(self, monkeypatch):
        """A Bedrock stream the fast-path could not serve stays on Anthropic.

        Bedrock streaming normally never reaches this finalisation -- it is
        routed to the Bedrock transport, which meters itself. The one exception
        is the fast-path-unavailable fallback, and there the SDK-native call is
        what serves the request, matching the non-streaming wrapper.
        """
        import revenium_middleware.anthropic.middleware as mw
        monkeypatch.delenv("REVENIUM_BEDROCK_DISABLE", raising=False)
        wrapper = getattr(mw.stream_wrapper, "_self_wrapper", mw.stream_wrapper)

        wrapped = MagicMock(return_value=_FakeMessageStream(self._final_message()))

        payloads = []
        with patch.object(mw, 'detect_provider', return_value=Provider.BEDROCK), \
                patch.object(mw, '_handle_bedrock_stream_request',
                             side_effect=mw.BedrockValidationError("no payload")), \
                patch.object(mw, 'submit_ai_event',
                             side_effect=lambda op, args: payloads.append(args)), \
                patch.object(mw, '_safe_run_async_in_thread',
                             side_effect=_run_metering_inline), \
                patch.object(mw, '_get_thread_safe_client', return_value=MagicMock()):
            with wrapper(wrapped, MagicMock(), (),
                         {"model": "claude-3-5-sonnet-20241022",
                          "messages": [{"role": "user", "content": "hi"}],
                          "max_tokens": 16}):
                pass

        assert len(payloads) == 1
        assert payloads[0]["provider"] == "ANTHROPIC"


class TestBedrockTransportLoad:
    """Overhead and surge behaviour of the transport-layer patch."""

    @staticmethod
    def _fresh_response(operation):
        from test_anthropic.test_bedrock_transport import (
            converse_response,
            converse_stream_response,
            invoke_model_response,
            invoke_stream_response,
        )
        factory = {
            "InvokeModel": invoke_model_response,
            "Converse": converse_response,
            "InvokeModelWithResponseStream": invoke_stream_response,
            "ConverseStream": converse_stream_response,
        }[operation]
        return factory()

    @staticmethod
    def _consume(operation, result):
        if operation == "InvokeModelWithResponseStream":
            list(result["body"])
        elif operation == "ConverseStream":
            list(result["stream"])

    def _run_calls(self, bt, operation, count, request_id=None):
        from test_anthropic.test_bedrock_transport import make_instance

        instance = make_instance()
        for index in range(count):
            response = self._fresh_response(operation)
            if request_id is not None:
                response["ResponseMetadata"]["RequestId"] = f"{request_id}-{index}"
            result = bt._make_api_call_wrapper(
                lambda *a, **k: response, instance,
                (operation, {"modelId":
                             "us.anthropic.claude-3-5-sonnet-20241022-v2:0"}), {})
            self._consume(operation, result)

    def test_bedrock_transport_overhead_benchmark(self, monkeypatch, capsys):
        import statistics
        import time
        import tracemalloc

        from revenium_middleware.anthropic import bedrock_transport as bt

        operations = ["InvokeModel", "Converse",
                      "InvokeModelWithResponseStream", "ConverseStream"]
        calls_per_op = 1000
        report = {}

        for enabled in (False, True):
            monkeypatch.setenv("REVENIUM_BEDROCK_TRANSPORT", "1" if enabled else "0")
            with patch.object(bt, "_emit_completion") as mock_emit:
                tracemalloc.start()
                for operation in operations:
                    latencies = []
                    started = time.perf_counter()
                    for _ in range(calls_per_op):
                        call_start = time.perf_counter()
                        response = self._fresh_response(operation)
                        result = bt._make_api_call_wrapper(
                            lambda *a, **k: response,
                            self._instance(), (operation, {
                                "modelId":
                                "us.anthropic.claude-3-5-sonnet-20241022-v2:0"
                            }), {})
                        self._consume(operation, result)
                        latencies.append(time.perf_counter() - call_start)
                    elapsed = time.perf_counter() - started
                    latencies.sort()
                    report[(operation, enabled)] = {
                        "p50_us": latencies[len(latencies) // 2] * 1e6,
                        "p95_us": latencies[int(len(latencies) * 0.95)] * 1e6,
                        "throughput_cps": calls_per_op / elapsed,
                    }
                _, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                report[("peak_memory_kb", enabled)] = peak / 1024

                expected = len(operations) * calls_per_op if enabled else 0
                assert mock_emit.call_count == expected

        for operation in operations:
            off = report[(operation, False)]
            on = report[(operation, True)]
            print(f"{operation}: off p50={off['p50_us']:.1f}us p95={off['p95_us']:.1f}us "
                  f"tput={off['throughput_cps']:.0f}/s | on p50={on['p50_us']:.1f}us "
                  f"p95={on['p95_us']:.1f}us tput={on['throughput_cps']:.0f}/s")
        print(f"peak memory: off={report[('peak_memory_kb', False)]:.0f}KB "
              f"on={report[('peak_memory_kb', True)]:.0f}KB")

        out = capsys.readouterr().out
        assert "p95" in out  # evidence emitted for the review attachment
        with capsys.disabled():
            print("\n" + out, end="")

    @staticmethod
    def _instance():
        from test_anthropic.test_bedrock_transport import make_instance
        return make_instance()

    def test_bedrock_transport_concurrent_burst(self, monkeypatch):
        import asyncio
        import itertools
        import threading
        from concurrent.futures import ThreadPoolExecutor

        from revenium_middleware.anthropic import bedrock_transport as bt

        monkeypatch.setenv("REVENIUM_BEDROCK_TRANSPORT", "1")

        emitted = []
        lock = threading.Lock()

        def record(**kwargs):
            with lock:
                emitted.append(kwargs["transaction_id"])

        operations = itertools.cycle([
            "InvokeModel", "Converse",
            "InvokeModelWithResponseStream", "ConverseStream",
        ])
        sync_jobs = [(next(operations), f"burst-{i}") for i in range(80)]
        async_jobs = [(next(operations), f"burst-async-{i}") for i in range(40)]

        def one_call(operation, request_id):
            response = self._fresh_response(operation)
            response["ResponseMetadata"]["RequestId"] = request_id
            result = bt._make_api_call_wrapper(
                lambda *a, **k: response, self._instance(),
                (operation, {"modelId":
                             "us.anthropic.claude-3-5-sonnet-20241022-v2:0"}), {})
            self._consume(operation, result)

        with patch.object(bt, "_emit_completion", side_effect=record):
            with ThreadPoolExecutor(max_workers=16) as executor:
                futures = [executor.submit(one_call, op, rid)
                           for op, rid in sync_jobs]
                for future in futures:
                    future.result()

            async def async_burst():
                await asyncio.gather(*[
                    asyncio.to_thread(one_call, op, rid)
                    for op, rid in async_jobs
                ])

            asyncio.run(async_burst())

        total = len(sync_jobs) + len(async_jobs)
        assert len(emitted) == total  # zero missing, zero duplicated
        assert len(set(emitted)) == total  # unique request-derived IDs
        assert threading.active_count() < 20  # workers returned to idle
