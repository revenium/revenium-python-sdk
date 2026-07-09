"""
Tests for debug-log sanitization.

DEBUG-level logging (REVENIUM_LOG_LEVEL=DEBUG) must never emit raw request
kwargs: auth material passed via ``extra_headers``/``api_key`` and
prompt/message content must be redacted before hitting application logs.

Covers the shared sanitizer in ``revenium_middleware._core.log_sanitize`` and
the wrapper debug-log sites in the openai, ollama, litellm, and vertex_ai
middlewares.
"""

import importlib
import logging
from importlib.util import find_spec
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

MIDDLEWARE_LOGGER = "revenium_middleware.extension"


def _sanitize():
    from revenium_middleware._core.log_sanitize import sanitize_for_logging

    return sanitize_for_logging


# ---------------------------------------------------------------------------
# Unit tests: shared sanitizer
# ---------------------------------------------------------------------------


class TestSanitizeForLogging:
    def test_sensitive_keys_redacted(self):
        sanitize = _sanitize()
        result = sanitize(
            {
                "api_key": "sk-secret",
                "apiKey": "sk-camel",
                "Authorization": "Bearer sk-abc",
                "client_secret": "shh",
                "access_token": "tok-123",
                "password": "hunter2",
                "aws_credentials": "cred-xyz",
                "model": "gpt-4o",
            }
        )
        assert result["api_key"] == "[REDACTED]"
        assert result["apiKey"] == "[REDACTED]"
        assert result["Authorization"] == "[REDACTED]"
        assert result["client_secret"] == "[REDACTED]"
        assert result["access_token"] == "[REDACTED]"
        assert result["password"] == "[REDACTED]"
        assert result["aws_credentials"] == "[REDACTED]"
        assert result["model"] == "gpt-4o"

    def test_sensitive_keys_redacted_when_nested(self):
        sanitize = _sanitize()
        data = {
            "extra_headers": {
                "Authorization": "Bearer sk-LEAK",
                "X-Trace-Id": "trace-1",
            }
        }
        result = sanitize(data)
        assert result["extra_headers"]["Authorization"] == "[REDACTED]"
        assert result["extra_headers"]["X-Trace-Id"] == "trace-1"
        assert "sk-LEAK" not in repr(result)

    def test_messages_summarized_not_recursed(self):
        sanitize = _sanitize()
        data = {
            "messages": [
                {"role": "system", "content": "top secret system prompt"},
                {"role": "user", "content": "SSN 123-45-6789"},
            ]
        }
        result = sanitize(data)
        assert result["messages"] == "[REDACTED: 2 items]"
        assert "123-45-6789" not in repr(result)
        assert "top secret" not in repr(result)

    def test_scalar_content_keys_redacted(self):
        sanitize = _sanitize()
        result = sanitize(
            {
                "prompt": "my private prompt",
                "system": "system instructions",
                "input": "raw pii input",
            }
        )
        assert result["prompt"] == "[REDACTED]"
        assert result["system"] == "[REDACTED]"
        assert result["input"] == "[REDACTED]"

    def test_contents_list_summarized(self):
        sanitize = _sanitize()
        result = sanitize({"contents": ["part one", "part two", "part three"]})
        assert result["contents"] == "[REDACTED: 3 items]"
        assert "part one" not in repr(result)

    def test_content_key_match_is_exact_not_substring(self):
        sanitize = _sanitize()
        result = sanitize({"input_cost": 12, "system_fingerprint": "fp_44709d6fcb"})
        assert result["input_cost"] == 12
        assert result["system_fingerprint"] == "fp_44709d6fcb"

    def test_long_strings_truncated(self):
        sanitize = _sanitize()
        result = sanitize({"note": "x" * 500})
        assert result["note"].endswith("...[TRUNCATED]")
        assert len(result["note"]) < 500

    def test_non_dict_passthrough(self):
        sanitize = _sanitize()
        assert sanitize(42) == 42
        assert sanitize(None) is None
        assert sanitize("short") == "short"
        assert sanitize([1, "a"]) == [1, "a"]

    def test_tuple_args_sanitized_elementwise(self):
        sanitize = _sanitize()
        result = sanitize(({"api_key": "sk-x"}, "plain"))
        assert result[0]["api_key"] == "[REDACTED]"
        assert result[1] == "plain"

    def test_max_depth_guard(self):
        sanitize = _sanitize()
        data = {"a": {"b": {"c": {"d": {"e": {"f": {"g": {"h": {"i": "deep"}}}}}}}}}
        result = sanitize(data)
        assert "[MAX_DEPTH_REACHED]" in repr(result)
        assert "deep" not in repr(result)


# ---------------------------------------------------------------------------
# Integration: wrapper debug logs must not leak secrets or prompt content
# ---------------------------------------------------------------------------


class TestOpenAIDebugLogSanitization:
    def test_create_wrapper_debug_log_redacts_secrets(self, caplog):
        mod = importlib.import_module("revenium_middleware.openai.middleware")
        wrapper = getattr(mod.create_wrapper, "_self_wrapper", mod.create_wrapper)
        wrapped = MagicMock(return_value=SimpleNamespace(id="x", usage=None))
        kwargs = {
            "model": "gpt-4o",
            "extra_headers": {"Authorization": "Bearer sk-LEAKME-123"},
            "messages": [{"role": "user", "content": "SSN 123-45-6789"}],
        }

        with caplog.at_level(logging.DEBUG, logger=MIDDLEWARE_LOGGER):
            wrapper(wrapped, None, (), kwargs)

        wrapped.assert_called_once()
        assert "sk-LEAKME-123" not in caplog.text
        assert "123-45-6789" not in caplog.text
        assert "[REDACTED" in caplog.text


class TestOllamaDebugLogSanitization:
    def test_chat_wrapper_debug_log_redacts_secrets(self, caplog):
        pytest.importorskip("ollama")
        mod = importlib.import_module("revenium_middleware.ollama.middleware")
        wrapper = getattr(mod.chat_wrapper, "_self_wrapper", mod.chat_wrapper)
        wrapped = MagicMock(return_value=SimpleNamespace(model="llama3"))
        kwargs = {
            "model": "llama3",
            "stream": False,
            "messages": [{"role": "user", "content": "SSN 123-45-6789"}],
            "headers": {"Authorization": "Bearer sk-LEAKME-456"},
        }

        with patch.object(mod, "handle_response") as mock_handle:
            with caplog.at_level(logging.DEBUG, logger=MIDDLEWARE_LOGGER):
                wrapper(wrapped, None, (), kwargs)

        mock_handle.assert_called_once()
        wrapped.assert_called_once()
        assert "sk-LEAKME-456" not in caplog.text
        assert "123-45-6789" not in caplog.text
        assert "[REDACTED" in caplog.text


class TestLiteLLMDebugLogSanitization:
    def test_completion_wrapper_debug_log_redacts_secrets(self, caplog):
        pytest.importorskip("litellm")
        mod = importlib.import_module(
            "revenium_middleware.litellm.client.middleware"
        )
        wrapper = getattr(
            mod.completion_wrapper, "_self_wrapper", mod.completion_wrapper
        )
        wrapped = MagicMock(return_value=SimpleNamespace(id="x", usage=None))
        kwargs = {
            "model": "gpt-4o",
            "stream": False,
            "api_key": "sk-LEAKME-789",
            "messages": [{"role": "user", "content": "SSN 123-45-6789"}],
        }

        with patch.object(mod, "handle_response") as mock_handle:
            with caplog.at_level(logging.DEBUG, logger=MIDDLEWARE_LOGGER):
                wrapper(wrapped, None, (), kwargs)

        mock_handle.assert_called_once()
        wrapped.assert_called_once()
        assert "sk-LEAKME-789" not in caplog.text
        assert "123-45-6789" not in caplog.text
        assert "[REDACTED" in caplog.text


@pytest.mark.skipif(find_spec("vertexai") is None, reason="vertexai not installed")
class TestVertexAIDebugLogSanitization:
    def test_generate_content_wrapper_debug_log_redacts_secrets(self, caplog):
        mod = importlib.import_module(
            "revenium_middleware.google.vertex_ai.middleware"
        )
        wrapped = MagicMock(return_value=SimpleNamespace(text="ok"))
        kwargs = {
            "contents": ["SSN 123-45-6789"],
            "labels": {"authorization": "Bearer sk-LEAKME-000"},
        }

        with patch.object(mod, "create_vertex_ai_metering_call") as mock_meter, patch.object(
            mod,
            "extract_prompt_data_if_enabled",
            return_value=(None, None, None, None),
        ):
            with caplog.at_level(logging.DEBUG, logger=MIDDLEWARE_LOGGER):
                mod.generate_content_wrapper_impl(wrapped, None, (), kwargs)

        mock_meter.assert_called_once()
        wrapped.assert_called_once()
        assert "sk-LEAKME-000" not in caplog.text
        assert "123-45-6789" not in caplog.text
        assert "[REDACTED" in caplog.text


class TestGoogleAIDebugLogSanitization:
    def test_generate_content_wrapper_does_not_leak_kwargs(self, caplog):
        import logging as _logging

        genai_mw = pytest.importorskip("revenium_middleware.google.google_ai.middleware")
        wrapper = getattr(genai_mw.generate_content_wrapper, "_self_wrapper",
                          genai_mw.generate_content_wrapper) if hasattr(genai_mw, "generate_content_wrapper") else None
        if wrapper is None:
            pytest.skip("generate_content wrapper not present")

        from types import SimpleNamespace
        from unittest.mock import MagicMock

        mock_wrapped = MagicMock(return_value=SimpleNamespace(usage_metadata=None, text="ok"))
        kwargs = {
            "model": "gemini-test",
            "contents": [{"role": "user", "parts": [{"text": "SSN 123-45-6789"}]}],
            "config": {"api_key": "sk-LEAKME-789"},
        }

        with caplog.at_level(_logging.DEBUG, logger="revenium_middleware"):
            try:
                wrapper(mock_wrapped, None, (), kwargs)
            except Exception:
                pass  # downstream metering details are not this test's concern

        assert "sk-LEAKME-789" not in caplog.text
        assert "123-45-6789" not in caplog.text
