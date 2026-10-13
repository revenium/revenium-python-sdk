"""
Copyright (c) 2025 Revenium, Inc.
SPDX-License-Identifier: MIT
"""

"""
Tests for error handling and exception management.
"""

import httpx
import pytest
from google import genai
from google.genai import errors as genai_errors
from google.genai import types as genai_types
from unittest.mock import patch

from revenium_middleware.google.google_ai import middleware as genai_mw
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


RATE_LIMITED = {"error": {"code": 429, "message": "Resource has been exhausted (e.g. check quota).",
                          "status": "RESOURCE_EXHAUSTED"}}
GEMINI_REPLY = {
    "candidates": [{"content": {"role": "model", "parts": [{"text": "hi"}]}, "finishReason": "STOP", "index": 0}],
    "usageMetadata": {"promptTokenCount": 1, "candidatesTokenCount": 1, "totalTokenCount": 2},
    "modelVersion": "gemini-2.5-flash-lite",
}


def _gemini_client(status, body, **client_kwargs):
    transport = httpx.MockTransport(lambda request: httpx.Response(status, json=body))
    return genai.Client(api_key="stub", http_options=genai_types.HttpOptions(
        httpx_client=httpx.Client(transport=transport), retry_options=genai_types.HttpRetryOptions(attempts=1)),
        **client_kwargs)


def _generate_images(models):
    return models.generate_images(model="imagen-3.0-generate-002", prompt="a dot")


SYNC_CALLS = {
    "generate_content": ({}, lambda models: models.generate_content(model="gemini-2.5-flash-lite", contents="hi")),
    "embed_content": ({}, lambda models: models.embed_content(model="text-embedding-004", contents="hi")),
    # google-genai serves generate_images only in Vertex mode; an API key selects express mode.
    "generate_images": ({"vertexai": True}, _generate_images),
    "generate_content_stream": ({}, lambda models: list(
        models.generate_content_stream(model="gemini-2.5-flash-lite", contents="hi"))),
}


class TestGracefulErrorHandling:
    """The SDK stays out of the caller's error path (BACK-4042)."""

    @pytest.mark.parametrize("client_kwargs, call", SYNC_CALLS.values(), ids=SYNC_CALLS.keys())
    def test_provider_error_reaches_the_caller_unchanged(self, client_kwargs, call):
        client = _gemini_client(429, RATE_LIMITED, **client_kwargs)

        with pytest.raises(genai_errors.ClientError) as raised:
            call(client.models)

        assert raised.value.code == 429

    def test_metering_error_does_not_break_user_code(self):
        client = _gemini_client(200, GEMINI_REPLY)

        with patch.object(genai_mw, "create_google_ai_metering_call", side_effect=RuntimeError("boom")):
            response = client.models.generate_content(model="gemini-2.5-flash-lite", contents="hi")

        assert response.text == "hi"


class TestConfigurationErrors:
    """Test configuration validation and errors."""

    def test_config_error_message_helpful(self):
        """Test that configuration errors have helpful messages."""
        error = ConfigurationError(
            "GOOGLE_API_KEY not set. "
            "Get your API key from: https://aistudio.google.com/app/apikey"
        )

        message = str(error)
        assert "GOOGLE_API_KEY" in message
        assert "https://" in message  # Includes helpful link
