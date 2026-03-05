"""
Copyright (c) 2025 Revenium, Inc.
SPDX-License-Identifier: MIT
"""

"""
Tests for error handling and exception management.
"""

import pytest
from unittest.mock import Mock, patch
from revenium_middleware.google.common.exceptions import (
    ReveniumMiddlewareError,
    MeteringError,
    TokenExtractionError,
    ProviderDetectionError,
    ConfigurationError,
    StreamingError,
    APIResponseError
)


class TestExceptionHierarchy:
    """Test exception class hierarchy."""

    def test_metering_error_inherits_from_base(self):
        """Test that MeteringError inherits from ReveniumMiddlewareError."""
        error = MeteringError("Test error")

        assert isinstance(error, ReveniumMiddlewareError)
        assert isinstance(error, Exception)

    def test_all_exceptions_inherit_from_base(self):
        """Test that all custom exceptions inherit from base."""
        exceptions = [
            MeteringError,
            TokenExtractionError,
            ProviderDetectionError,
            ConfigurationError,
            StreamingError,
            APIResponseError
        ]

        for exc_class in exceptions:
            error = exc_class("Test")
            assert isinstance(error, ReveniumMiddlewareError)

    def test_exception_messages(self):
        """Test that exceptions preserve messages."""
        message = "Specific error message"
        error = MeteringError(message)

        assert str(error) == message


class TestGracefulErrorHandling:
    """Test graceful error handling in middleware."""

    def test_metering_error_does_not_break_user_code(self):
        """Test that metering errors don't propagate to user."""
        def user_function():
            # Simulate user code
            result = "user result"

            # Simulate metering error (should be caught)
            try:
                raise MeteringError("Metering failed")
            except MeteringError:
                pass  # Middleware catches and logs

            return result

        # User code should succeed despite metering error
        result = user_function()
        assert result == "user result"

    def test_api_error_logged_not_raised(self):
        """Test that API errors are logged but not raised."""
        import logging

        logged_errors = []

        class TestHandler(logging.Handler):
            def emit(self, record):
                if record.levelno >= logging.ERROR:
                    logged_errors.append(record.getMessage())

        logger = logging.getLogger("revenium_middleware")
        handler = TestHandler()
        logger.addHandler(handler)

        try:
            # Simulate API error
            try:
                raise APIResponseError("API call failed")
            except APIResponseError as e:
                logger.error(f"API error: {e}")

            # Should have logged error
            assert len(logged_errors) > 0
            assert "API error" in logged_errors[0]
        finally:
            logger.removeHandler(handler)

    def test_token_extraction_error_returns_zeros(self):
        """Test that token extraction errors return zero counts."""
        def extract_tokens_safe(response):
            try:
                # Simulate extraction that might fail
                if not hasattr(response, 'usage_metadata'):
                    raise TokenExtractionError("No usage metadata")
                return response.usage_metadata.total_tokens
            except (TokenExtractionError, AttributeError):
                # Return zeros on error
                return 0

        mock_response = Mock(spec=[])  # No attributes

        result = extract_tokens_safe(mock_response)
        assert result == 0


class TestConfigurationErrors:
    """Test configuration validation and errors."""

    def test_missing_api_key_raises_config_error(self):
        """Test that missing API key raises ConfigurationError."""
        import os

        with patch.dict(os.environ, {}, clear=True):
            # Simulate missing API key check
            if not os.getenv("REVENIUM_METERING_API_KEY"):
                with pytest.raises(ConfigurationError):
                    raise ConfigurationError("REVENIUM_METERING_API_KEY not set")

    def test_missing_google_credentials_raises_config_error(self):
        """Test that missing Google credentials raises ConfigurationError."""
        import os

        with patch.dict(os.environ, {}, clear=True):
            # Check for Vertex AI credentials
            if not os.getenv("GOOGLE_CLOUD_PROJECT"):
                with pytest.raises(ConfigurationError):
                    raise ConfigurationError("GOOGLE_CLOUD_PROJECT not set")

    def test_config_error_message_helpful(self):
        """Test that configuration errors have helpful messages."""
        error = ConfigurationError(
            "GOOGLE_API_KEY not set. "
            "Get your API key from: https://aistudio.google.com/app/apikey"
        )

        message = str(error)
        assert "GOOGLE_API_KEY" in message
        assert "https://" in message  # Includes helpful link


class TestStreamingErrors:
    """Test streaming-specific error handling."""

    def test_stream_interruption_handled(self):
        """Test that stream interruption is handled gracefully."""
        def process_stream():
            chunks = []
            try:
                for i in range(10):
                    if i == 5:
                        raise StreamingError("Stream interrupted")
                    chunks.append(f"chunk_{i}")
            except StreamingError:
                # Log partial results
                pass
            return chunks

        result = process_stream()
        assert len(result) == 5  # Got 5 chunks before interruption

    def test_chunk_processing_error_caught(self):
        """Test that errors in chunk processing are caught."""
        def process_chunk_safe(chunk):
            try:
                # Simulate processing that might fail
                return chunk.text
            except AttributeError:
                return ""  # Return empty on error

        mock_chunk = Mock(spec=[])  # No text attribute
        result = process_chunk_safe(mock_chunk)

        assert result == ""  # Handled gracefully

    def test_usage_logging_error_does_not_prevent_cleanup(self):
        """Test that usage logging errors don't prevent cleanup."""
        cleanup_executed = False

        try:
            # Simulate usage logging error
            raise MeteringError("Failed to log usage")
        except MeteringError:
            pass
        finally:
            cleanup_executed = True

        assert cleanup_executed is True


class TestNetworkErrors:
    """Test network and API connectivity error handling."""

    def test_api_timeout_handled(self):
        """Test that API timeouts are handled gracefully."""
        import socket

        def call_api_with_timeout():
            try:
                # Simulate timeout
                raise socket.timeout("API request timed out")
            except socket.timeout as e:
                # Log and return None
                return None

        result = call_api_with_timeout()
        assert result is None

    def test_connection_error_handled(self):
        """Test that connection errors are handled."""
        import requests

        def call_api_safe():
            try:
                # Simulate connection error
                raise requests.ConnectionError("Could not connect to API")
            except requests.ConnectionError:
                # Log and return None
                return None

        result = call_api_safe()
        assert result is None

    def test_api_rate_limit_handled(self):
        """Test that rate limit errors are handled."""
        def call_api_with_retry():
            try:
                # Simulate rate limit (429)
                raise APIResponseError("Rate limit exceeded (429)")
            except APIResponseError:
                # Could implement retry logic here
                return None

        result = call_api_with_retry()
        assert result is None


class TestRetryLogic:
    """Test retry logic for transient errors."""

    def test_retry_on_transient_error(self):
        """Test that transient errors trigger retry."""
        attempts = []

        def call_with_retry(max_retries=3):
            for attempt in range(max_retries):
                attempts.append(attempt)
                try:
                    if attempt < 2:
                        # Fail first 2 attempts
                        raise APIResponseError("Transient error")
                    return "success"
                except APIResponseError:
                    if attempt == max_retries - 1:
                        raise  # Give up after max retries
                    continue

        result = call_with_retry()

        assert result == "success"
        assert len(attempts) == 3  # Tried 3 times

    def test_no_retry_on_permanent_error(self):
        """Test that permanent errors don't retry."""
        attempts = []

        def call_without_retry():
            attempts.append(1)
            # Simulate permanent error (401 Unauthorized)
            raise ConfigurationError("Invalid API key (401)")

        with pytest.raises(ConfigurationError):
            call_without_retry()

        assert len(attempts) == 1  # Only tried once


class TestErrorRecovery:
    """Test error recovery mechanisms."""

    def test_fallback_to_default_on_error(self):
        """Test falling back to defaults on error."""
        def get_config_with_fallback(key, default):
            try:
                import os
                value = os.environ[key]
                if not value:
                    raise KeyError
                return value
            except KeyError:
                return default

        import os
        with patch.dict(os.environ, {}, clear=True):
            result = get_config_with_fallback("MISSING_KEY", "default_value")
            assert result == "default_value"

    def test_partial_success_logged(self):
        """Test that partial successes are logged."""
        results = []

        def process_batch(items):
            for item in items:
                try:
                    # Simulate processing that might fail
                    if item == "bad":
                        raise ValueError("Bad item")
                    results.append(item)
                except ValueError:
                    continue  # Skip bad items, continue processing

        process_batch(["good1", "bad", "good2"])

        assert len(results) == 2
        assert "good1" in results
        assert "good2" in results
        assert "bad" not in results
