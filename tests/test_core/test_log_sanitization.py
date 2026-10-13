"""
Tests for debug-log sanitization.

DEBUG-level logging (REVENIUM_LOG_LEVEL=DEBUG) must never emit raw request
kwargs: auth material passed via ``extra_headers``/``api_key`` and
prompt/message content must be redacted before hitting application logs.

Covers the shared sanitizer in ``revenium_middleware._core.log_sanitize`` and
the wrapper debug-log sites in the openai, ollama, litellm, and vertex_ai
middlewares.
"""

import ast
import importlib
import logging
import subprocess
import sys
import time
from importlib.util import find_spec
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

import revenium_middleware
from secret_samples import SAMPLES

MIDDLEWARE_LOGGER = "revenium_middleware.extension"

CRED = "sk-live-SUBSCRIBER-KEY-123"
EMAIL = "person@customer.example"
SYS = "SYSTEM-SECRET-PROMPT be terse"
USER = "USER-PRIVATE-MESSAGE ssn 123-45-6789"
REPLY = "REPLY-SECRET-TEXT about the patient"
SENTINELS = (CRED, EMAIL, "SYSTEM-SECRET-PROMPT", "USER-PRIVATE-MESSAGE", "123-45-6789", "REPLY-SECRET-TEXT")
SUBSCRIBER = {"id": "u1", "email": EMAIL, "credential": {"name": "k", "value": CRED}}


def assert_no_sentinel(text):
    leaked = [sentinel for sentinel in SENTINELS if sentinel in text]
    assert not leaked, f"{leaked} found in: {text}"


REPO_ROOT = Path(__file__).resolve().parents[2]
CHILD_TIMEOUT_SECONDS = 20


def child_elapsed_seconds(code):
    """Run ``code`` in a fresh interpreter and return the elapsed seconds it prints last.

    ``re`` holds the GIL for a whole match, so a slow pattern can only be held
    to a time limit from another process.
    """
    try:
        child = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=CHILD_TIMEOUT_SECONDS,
            cwd=REPO_ROOT,
        )
    except subprocess.TimeoutExpired:
        pytest.fail(f"the cleaner did not finish in {CHILD_TIMEOUT_SECONDS} s")
    assert child.returncode == 0, child.stderr
    return float(child.stdout.strip().splitlines()[-1])


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

    def test_long_strings_redacted_not_truncated(self):
        sanitize = _sanitize()
        result = sanitize({"note": "x" * 500})
        assert result["note"].startswith("[REDACTED")
        assert "x" * 20 not in repr(result)

    def test_long_unbroken_string_is_sanitized_in_linear_time(self):
        """BACK-4365: a long picture or field must not make the secret check quadratic."""
        sanitize = _sanitize()
        picture = ("QUJDRGVmZ2gxMjM0+/" * 4000)[:60_000]
        started = time.perf_counter()
        result = sanitize({"images": [picture]})
        elapsed = time.perf_counter() - started

        assert elapsed < 1.0
        assert result == {"images": ["[REDACTED: 60000 chars]"]}

    def test_hundred_thousand_character_image_sanitized_within_a_second(self):
        code = (
            "import time\n"
            "from revenium_middleware._core.log_sanitize import sanitize_for_logging\n"
            "picture = ('QUJDRGVmZ2gxMjM0+/' * 6000)[:100_000]\n"
            "started = time.perf_counter()\n"
            "assert sanitize_for_logging({'images': [picture]}) == {'images': ['[REDACTED: 100000 chars]']}\n"
            "print(time.perf_counter() - started)\n"
        )
        assert child_elapsed_seconds(code) < 1.0

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

    def test_prompt_capture_fields_redacted_in_snake_and_camel_case(self):
        sanitize = _sanitize()
        result = sanitize(
            {
                "system_prompt": SYS,
                "input_messages": USER,
                "output_response": REPLY,
                "systemPrompt": SYS,
                "inputMessages": USER,
                "outputResponse": REPLY,
            }
        )
        assert_no_sentinel(repr(result))

    def test_unlisted_free_text_field_is_redacted(self):
        sanitize = _sanitize()
        result = sanitize(
            {
                "brand_new_field": "UNLISTED-SECRET some words",
                "instructions": "INSTR-SECRET be nice",
                "query": "QUERY-SECRET who",
                "blob": "Z" * 300,
            }
        )
        text = repr(result)
        assert "UNLISTED-SECRET" not in text
        assert "INSTR-SECRET" not in text
        assert "QUERY-SECRET" not in text
        assert "Z" * 20 not in text

    def test_positional_message_content_redacted(self):
        sanitize = _sanitize()
        result = sanitize(("gpt-4.1-nano", [{"role": "user", "content": USER}]))
        assert "123-45-6789" not in repr(result)
        assert "gpt-4.1-nano" in repr(result)

    def test_objects_are_never_logged_by_repr(self):
        sanitize = _sanitize()

        class Opaque:
            def __repr__(self):
                return "OBJ-SECRET"

            __str__ = __repr__

        assert "OBJ-SECRET" not in repr(sanitize({"thing": Opaque(), "items": [Opaque()]}))

        types = pytest.importorskip("google.genai.types")
        config = types.GenerateContentConfig(system_instruction=SYS)
        assert "SYSTEM-SECRET" not in repr(sanitize({"config": config}))

    def test_counts_kept_numeric_secrets_redacted(self):
        sanitize = _sanitize()
        result = sanitize(
            {
                "input_token_count": 12,
                "output_token_count": 3,
                "prompt_tokens": 7,
                "inputTokenCount": 9,
                "prompts_truncated": False,
                "access_token": "tok-abc",
                "password": 1234,
                "credential": 987654,
            }
        )
        assert result["input_token_count"] == 12
        assert result["output_token_count"] == 3
        assert result["prompt_tokens"] == 7
        assert result["inputTokenCount"] == 9
        assert result["prompts_truncated"] is False
        assert result["access_token"] == "[REDACTED]"
        assert result["password"] == "[REDACTED]"
        assert result["credential"] == "[REDACTED]"

    def test_subscriber_email_and_credential_redacted(self):
        sanitize = _sanitize()
        result = sanitize(
            {
                "subscriber": {"id": "u1", "email": EMAIL, "credential": {"name": "k", "value": CRED}},
                "other_subscriber": {"id": "jane@example.com"},
                "extra_headers": {"Ocp-Apim-Subscription-Key": "abc123", "X-Custom-Key": "k-1"},
            }
        )
        text = repr(result)
        assert result["subscriber"]["id"] == "u1"
        assert EMAIL not in text
        assert CRED not in text
        assert "jane@example.com" not in text
        assert "abc123" not in text
        assert "k-1" not in text

    def test_secret_shaped_values_redacted_under_any_key(self):
        sanitize = _sanitize()
        result = sanitize(
            {
                "note": "sk-ant-abc123",
                "url_hint": "rev_mk_abc",
                "maps": "AIzaFAKE-0001",
                "header_copy": "Bearer abc.def",
                "transaction_id": "task-123",
            }
        )
        text = repr(result)
        for secret in ("sk-ant-abc123", "rev_mk_abc", "AIzaFAKE-0001", "abc.def"):
            assert secret not in text
        assert result["transaction_id"] == "task-123"

    def test_attribution_fields_print(self):
        sanitize = _sanitize()
        result = sanitize(
            {
                "organization_name": "Acme Corp",
                "product_name": "Gold Plan",
                "model": "gpt-4.1-nano",
                "task_type": "summarize ticket",
            }
        )
        assert result["organization_name"] == "Acme Corp"
        assert result["product_name"] == "Gold Plan"
        assert result["model"] == "gpt-4.1-nano"
        assert result["task_type"] == "summarize ticket"

    def test_max_depth_guard(self):
        sanitize = _sanitize()
        data = {"a": {"b": {"c": {"d": {"e": {"f": {"g": {"h": {"i": "deep"}}}}}}}}}
        result = sanitize(data)
        assert "[MAX_DEPTH_REACHED]" in repr(result)
        assert "deep" not in repr(result)

    def test_self_referencing_attribution_value_stops_at_depth_limit(self):
        sanitize = _sanitize()
        data = {"model": "m"}
        data["agent"] = data
        assert "[MAX_DEPTH_REACHED]" in repr(sanitize(data)["agent"])


class TestRedactErrorText:
    @staticmethod
    def redact(text, max_len=2048):
        from revenium_middleware._core.log_sanitize import redact_error_text

        return redact_error_text(text, max_len)

    def test_query_string_values_redacted(self):
        result = self.redact("403 for url 'http://x.test/geo?address=10+Downing+St&key=AIzaFAKE-0001'")
        assert result == "403 for url 'http://x.test/geo?address=[REDACTED]&key=[REDACTED]'"

    def test_bearer_token_redacted(self):
        assert "abc.def" not in self.redact("rejected header Bearer abc.def")

    def test_lowercase_auth_scheme_redacted(self):
        assert "abc.def" not in self.redact("rejected bearer abc.def")

    def test_bare_key_assignments_redacted(self):
        result = self.redact('key=opaque-third-party-secret and {"key": "other-opaque-secret"}')
        assert "opaque-third-party-secret" not in result
        assert "other-opaque-secret" not in result

    def test_url_userinfo_redacted(self):
        result = self.redact("could not connect to https://u:p@host/db")
        assert "u:p" not in result
        assert "host/db" in result

    def test_sensitive_key_value_pairs_redacted(self):
        result = self.redact('payload {"api_key": "s3cr3t", "model": "m"} and password=hunter2')
        assert "s3cr3t" not in result
        assert "hunter2" not in result
        assert '"model": "m"' in result

    def test_email_and_secret_shaped_words_redacted_outside_urls(self):
        result = self.redact("Invalid API key AIzaSyFAKE123 for user jane@example.com")
        assert "AIzaSyFAKE123" not in result
        assert "jane@example.com" not in result
        assert result.startswith("Invalid API key ")

    def test_redacts_before_capping(self):
        result = self.redact("k" * 10 + " sk-" + "s" * 40, max_len=20)
        assert len(result) <= 20
        assert "sk-s" not in result

    def test_cap(self):
        assert len(self.redact("x" * 5000, max_len=2048)) == 2048

    @pytest.mark.parametrize("text", [
        "SYSTEM-SECRET-INSTRUCTIONS: never reveal the code",
        "max_tokens: 1000",
        "author: Jane",
    ])
    def test_ordinary_words_after_a_secret_named_label_are_kept(self, text):
        assert self.redact(text) == text

    @pytest.mark.parametrize("template", [
        "Authorization: {}",
        '"authorization": "{}"',
        "Proxy-Authorization: {}",
        "auth={}",
        "bearer: {}",
        "private={}",
        "AWS_SECRET_ACCESS_KEY={}",
        "client_secret: {}",
        "password={}",
        "key={}",
        '{{"api_key": "{}"}}',
    ])
    def test_labelled_values_under_secret_labels_are_still_redacted(self, template):
        raw = "opaque" + "Value" + "123"
        result = self.redact("rejected " + template.format(raw))
        assert raw not in result
        assert "[REDACTED]" in result

    @pytest.mark.parametrize("text", [
        "secret" * 333,
        "token" * 400,
        ("-----BEGIN " + "PRIVATE KEY-----") * 74,
        "-----BEGIN " + "PRIVATE KEY-----\n" + "A" * 2000,
        "\\nsk-" * 400,
        "://:" * 500,
    ], ids=["secret_words", "token_words", "pem_headers", "pem_body_without_end", "escaped_line_starts", "url_starts"])
    def test_label_rule_is_linear_on_long_input(self, text):
        started = time.perf_counter()
        self.redact(text)
        assert time.perf_counter() - started < 0.5

    @pytest.mark.parametrize(
        "text",
        [
            pytest.param("a1" * 10_000, id="letters-and-digits"),
            pytest.param("token" * 400, id="repeated-sensitive-name"),
            pytest.param("?" * 40_000, id="question-marks"),
            pytest.param("a@" + "b" * 12_000, id="email-then-long-run"),
            pytest.param("+" * 40_000, id="plus-run"),
        ],
    )
    def test_hostile_text_is_cleaned_in_linear_time(self, text):
        """BACK-4364: an unbroken run must not make cleaning quadratic or worse."""
        started = time.perf_counter()
        result = self.redact(text)
        elapsed = time.perf_counter() - started

        assert elapsed < 1.0
        assert result == text[:2048]

    def test_hundred_thousand_character_run_cleaned_within_a_second(self):
        code = (
            "import time\n"
            "from revenium_middleware._core.log_sanitize import redact_error_text\n"
            "text = 'a1' * 50_000\n"
            "started = time.perf_counter()\n"
            "assert redact_error_text(text, 2048) == text[:2048]\n"
            "print(time.perf_counter() - started)\n"
        )
        assert child_elapsed_seconds(code) < 1.0

    def test_long_quoted_secret_is_redacted_whole(self):
        """A length cut before cleaning would leave the start of a long quoted secret visible."""
        result = self.redact('Invalid value: {"api_key": "' + "Q " * 100_000 + '", "model": "m"}')
        assert "Q Q" not in result
        assert result.startswith('Invalid value: {"api_key": [REDACTED]')


class TestReportableExceptionText:
    @staticmethod
    def text(exc, max_len=2048):
        from revenium_middleware._core.error_text import reportable_exception_text

        return reportable_exception_text(exc, max_len)

    def test_class_name_only_by_default(self, monkeypatch):
        monkeypatch.delenv("REVENIUM_CAPTURE_ERROR_MESSAGES", raising=False)
        assert self.text(RuntimeError("key=AIzaFAKE-0001")) == "RuntimeError"

    def test_opt_in_prefixes_class_and_redacts(self, monkeypatch):
        monkeypatch.setenv("REVENIUM_CAPTURE_ERROR_MESSAGES", "true")
        assert self.text(RuntimeError("failed ?key=AIzaFAKE-0001")) == "RuntimeError: failed ?key=[REDACTED]"

    def test_opt_in_empty_message_sends_class(self, monkeypatch):
        monkeypatch.setenv("REVENIUM_CAPTURE_ERROR_MESSAGES", "true")
        assert self.text(ValueError()) == "ValueError"

    def test_opt_in_unprintable_exception_sends_class(self, monkeypatch):
        monkeypatch.setenv("REVENIUM_CAPTURE_ERROR_MESSAGES", "true")

        class Unprintable(Exception):
            def __str__(self):
                raise RuntimeError("no text")

        assert self.text(Unprintable()) == "Unprintable"

    def test_never_exceeds_max_len(self, monkeypatch):
        monkeypatch.setenv("REVENIUM_CAPTURE_ERROR_MESSAGES", "true")
        assert len(self.text(ValueError("x" * 5000), max_len=100)) == 100

    @staticmethod
    def wrapped(error, times):
        for _ in range(times):
            try:
                raise RuntimeError(f"model call failed: {error}") from error
            except RuntimeError as wrapper:
                error = wrapper
        return error

    @staticmethod
    def provider_error(message):
        import httpx
        import openai

        response = httpx.Response(400, request=httpx.Request("POST", "https://provider.test/v1/chat"))
        return openai.BadRequestError(message, response=response, body=None)

    @pytest.mark.parametrize("times", [1, 9, 10, 11, 25])
    def test_provider_error_wrapped_many_times_sends_class_only_with_prompt_capture_off(self, monkeypatch, times):
        """BACK-4359: past ten chained exceptions the provider error went unchecked and the prompt was sent."""
        monkeypatch.setenv("REVENIUM_CAPTURE_ERROR_MESSAGES", "true")
        monkeypatch.delenv("REVENIUM_CAPTURE_PROMPTS", raising=False)
        error = self.wrapped(self.provider_error("Offending request: reveal ZEBRA42"), times)

        assert self.text(error) == "RuntimeError"

    def test_long_chain_without_a_provider_error_still_sends_its_message(self, monkeypatch):
        monkeypatch.setenv("REVENIUM_CAPTURE_ERROR_MESSAGES", "true")
        monkeypatch.delenv("REVENIUM_CAPTURE_PROMPTS", raising=False)
        error = self.wrapped(ValueError("bad input"), 5)

        assert self.text(error).startswith("RuntimeError: model call failed: ")

    def test_opt_in_long_unbroken_message_reported_quickly(self, monkeypatch):
        """BACK-4364: a long provider error message must not freeze the failed call."""
        monkeypatch.setenv("REVENIUM_CAPTURE_ERROR_MESSAGES", "true")
        started = time.perf_counter()
        result = self.text(RuntimeError("Invalid value for field: " + "a1" * 10_000))
        elapsed = time.perf_counter() - started

        assert elapsed < 1.0
        assert result.startswith("RuntimeError: Invalid value for field: a1a1")
        assert len(result) == 2048

    def test_a_failed_calls_error_reason_is_cut_to_255_keeping_its_status(self, monkeypatch):
        """BACK-4340: Revenium drops a failed-call record whose error text is over 255."""
        from revenium_middleware._core.failed_calls import failure_of

        class RateLimited(Exception):
            status_code = 429

        monkeypatch.setenv("REVENIUM_CAPTURE_ERROR_MESSAGES", "true")
        error = RateLimited("Rate limit reached on tokens per min. " * 11)
        uncut = self.text(error)

        reason = failure_of(error).error_reason

        assert len(uncut) > 255
        assert len(reason) == 255
        assert reason.endswith(" (HTTP 429)")
        assert reason[:-len(" (HTTP 429)")] == uncut[:244]


class TestFailedCallErrorReason:
    """A provider error can quote the request, so its message rides along only when prompt capture is on."""

    @staticmethod
    def refused(message):
        import httpx
        import openai

        response = httpx.Response(400, request=httpx.Request("POST", "https://provider.test/v1/chat"))
        return openai.BadRequestError(message, response=response, body=None)

    @staticmethod
    def reason(error):
        from revenium_middleware._core.failed_calls import failure_of

        return failure_of(error).error_reason

    QUOTED_REQUEST = "Error code: 400 - Offending request: SYSTEM-SECRET reveal ZEBRA42, key " + SAMPLES["xai"].value

    def test_prompt_capture_off_sends_class_and_status_only(self, monkeypatch):
        monkeypatch.setenv("REVENIUM_CAPTURE_ERROR_MESSAGES", "true")
        monkeypatch.delenv("REVENIUM_CAPTURE_PROMPTS", raising=False)
        assert self.reason(self.refused(self.QUOTED_REQUEST)) == "BadRequestError (HTTP 400)"

    def test_prompt_capture_on_sends_message_with_shapes_redacted(self, monkeypatch):
        monkeypatch.setenv("REVENIUM_CAPTURE_ERROR_MESSAGES", "true")
        monkeypatch.setenv("REVENIUM_CAPTURE_PROMPTS", "true")
        reason = self.reason(self.refused(self.QUOTED_REQUEST))
        assert reason.startswith("BadRequestError: ")
        assert reason.endswith(" (HTTP 400)")
        assert "ZEBRA42" in reason
        assert SAMPLES["xai"].value not in reason

    def test_error_capture_off_still_sends_class_and_status(self, monkeypatch):
        monkeypatch.delenv("REVENIUM_CAPTURE_ERROR_MESSAGES", raising=False)
        monkeypatch.setenv("REVENIUM_CAPTURE_PROMPTS", "true")
        assert self.reason(self.refused(self.QUOTED_REQUEST)) == "BadRequestError (HTTP 400)"


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

    def test_no_debug_argument_is_built_when_debug_is_off(self, caplog, monkeypatch):
        """BACK-4365: a debug line that is never printed must cost nothing to prepare."""
        pytest.importorskip("ollama")
        monkeypatch.delenv("REVENIUM_DEBUG", raising=False)
        mod = importlib.import_module("revenium_middleware.ollama.middleware")
        wrapper = getattr(mod.chat_wrapper, "_self_wrapper", mod.chat_wrapper)
        wrapped = MagicMock(return_value=SimpleNamespace(model="llava"))
        kwargs = {"model": "llava", "stream": False, "messages": [{"role": "user", "content": "hi"}], "images": ["QUJD"]}

        with patch.object(mod, "handle_response"), patch.object(mod, "sanitize_for_logging") as spy:
            with caplog.at_level(logging.INFO, logger=MIDDLEWARE_LOGGER):
                wrapper(wrapped, None, (), kwargs)

        wrapped.assert_called_once()
        assert spy.call_count == 0


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


# ---------------------------------------------------------------------------
# Every payload log line: no credential, email, prompt or reply in any record
# ---------------------------------------------------------------------------


class TestMeteringClientDebugLog:
    def test_request_options_line_hides_credential_and_prompts(self, caplog):
        import httpx

        from revenium_middleware._metering import ReveniumMetering

        posted = []

        def answer(request):
            posted.append(request.content.decode())
            return httpx.Response(200, json={"id": "m1", "label": "l", "resourceType": "metering", "signature": "s"})

        metering = ReveniumMetering(
            api_key="hak_test",
            base_url="http://metering.test",
            http_client=httpx.Client(transport=httpx.MockTransport(answer)),
        )
        with caplog.at_level(logging.DEBUG, logger="revenium_middleware"):
            metering.ai.create_completion(
                completion_start_time="2026-10-08T00:00:00Z", cost_type="AI", input_token_count=11,
                is_streamed=False, model="m", output_token_count=3, provider="p", request_duration=1,
                request_time="2026-10-08T00:00:00Z", response_time="2026-10-08T00:00:00Z", stop_reason="END",
                total_token_count=14, transaction_id="t1", subscriber=SUBSCRIBER,
                system_prompt=SYS, input_messages=USER, output_response=REPLY,
            )

        assert "Request options" in caplog.text
        assert_no_sentinel(caplog.text)
        assert "11" in caplog.text
        assert CRED in posted[0], "only the log is sanitized; the record sent to Revenium keeps the credential"


class TestOllamaPayloadDebugLog:
    def test_reply_and_credential_absent_from_every_ollama_line(self, caplog):
        ollama = pytest.importorskip("ollama")
        import json

        import httpx

        mod = importlib.import_module("revenium_middleware.ollama.middleware")
        recorded = []

        def answer(request):
            return httpx.Response(200, json={
                "model": "llama3", "created_at": "2026-10-08T00:00:00Z", "done": True, "done_reason": "stop",
                "prompt_eval_count": 4, "eval_count": 2, "message": {"role": "assistant", "content": REPLY},
            })

        def record(operation, payload, *args, **kwargs):
            recorded.append(payload)
            return MagicMock(id="recorded")

        def run_now(coroutine):
            import asyncio
            asyncio.run(coroutine)

        client = ollama.Client(host="http://ollama.internal:8080", transport=httpx.MockTransport(answer))
        with patch.object(mod, "get_client", return_value=MagicMock()), \
                patch.object(mod, "submit_ai_event", side_effect=record), \
                patch.object(mod, "run_async_in_thread", side_effect=run_now), \
                caplog.at_level(logging.DEBUG, logger="revenium_middleware"):
            client.chat(model="llama3", messages=[{"role": "user", "content": USER}],
                        usage_metadata={"subscriber": SUBSCRIBER})

        assert len(recorded) == 1
        assert_no_sentinel(caplog.text)
        assert CRED in json.dumps(recorded[0]), "the record sent to Revenium keeps the credential"


class TestLiteLLMPayloadDebugLog:
    def test_metadata_args_and_reply_absent(self, caplog):
        pytest.importorskip("litellm")
        mod = importlib.import_module("revenium_middleware.litellm.client.middleware")
        wrapper = getattr(mod.completion_wrapper, "_self_wrapper", mod.completion_wrapper)
        response = SimpleNamespace(
            id="chatcmpl-1",
            usage=SimpleNamespace(prompt_tokens=4, completion_tokens=2, total_tokens=6),
            choices=[SimpleNamespace(message=SimpleNamespace(content=REPLY))],
            reply=REPLY,
        )
        wrapped = MagicMock(return_value=response)

        def run_now(coroutine):
            import asyncio
            asyncio.run(coroutine)

        submit = MagicMock(return_value=MagicMock(id="recorded"))
        with patch.object(mod, "get_client", return_value=MagicMock()), \
                patch.object(mod, "submit_ai_event", submit), \
                patch.object(mod, "run_async_in_thread", side_effect=run_now), \
                caplog.at_level(logging.DEBUG, logger="revenium_middleware"):
            wrapper(wrapped, None, (), {
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": USER}],
                "usage_metadata": {"subscriber": SUBSCRIBER},
            })

        submit.assert_called_once()
        assert_no_sentinel(caplog.text)


class TestOpenAIFailureLog:
    def test_refused_record_logs_counts_not_prompt_reply_or_email(self, mock_revenium_client, caplog):
        import asyncio

        import httpx

        from revenium_middleware._metering._exceptions import UnprocessableEntityError
        from revenium_middleware.openai.middleware import log_token_usage

        request = httpx.Request("POST", "http://metering.test/v2/ai/completions")
        mock_revenium_client.ai.create_completion.side_effect = UnprocessableEntityError(
            "Error code: 422 - invalid record", response=httpx.Response(422, request=request), body=None,
        )
        with caplog.at_level(logging.DEBUG, logger="revenium_middleware"):
            asyncio.run(log_token_usage(
                response_id="chatcmpl-refused", model="gpt-4o-mini", prompt_tokens=10, completion_tokens=5,
                total_tokens=15, cached_tokens=0, stop_reason="END", request_time="2026-10-08T00:00:00Z",
                response_time="2026-10-08T00:00:01Z", request_duration=1000,
                usage_metadata={"subscriber": SUBSCRIBER},
                system_prompt=SYS, input_messages=USER, output_response=REPLY,
            ))

        assert "REVENIUM FAILURE" in caplog.text
        assert "422" in caplog.text
        assert "'input_token_count': 10" in caplog.text
        assert_no_sentinel(caplog.text)


class TestMeteringSubmissionFailureLog:
    def test_permanent_failure_line_has_status_and_counts(self, mock_revenium_client, caplog):
        import httpx

        from revenium_middleware._core.metering_submission import submit_ai_event
        from revenium_middleware._metering._exceptions import UnprocessableEntityError

        request = httpx.Request("POST", "http://metering.test/v2/ai/completions")
        mock_revenium_client.ai.create_completion.side_effect = UnprocessableEntityError(
            "Error code: 422 - invalid record", response=httpx.Response(422, request=request), body=None,
        )
        with patch("revenium_middleware._core.metering_submission.current_client", return_value=mock_revenium_client), \
                caplog.at_level(logging.DEBUG, logger="revenium_middleware"):
            with pytest.raises(UnprocessableEntityError):
                submit_ai_event("completion", {
                    "model": "m", "input_token_count": 21, "output_token_count": 8, "total_token_count": 29,
                    "subscriber": SUBSCRIBER, "system_prompt": SYS, "input_messages": USER, "output_response": REPLY,
                })

        failure = [r.getMessage() for r in caplog.records if "failed permanently" in r.getMessage()]
        assert len(failure) == 1
        assert "422" in failure[0]
        assert "21" in failure[0] and "8" in failure[0] and "29" in failure[0]
        assert_no_sentinel(caplog.text)


class TestGoogleAISystemInstructionDebugLog:
    def test_system_instruction_object_absent(self, caplog):
        types = pytest.importorskip("google.genai.types")
        genai_mw = importlib.import_module("revenium_middleware.google.google_ai.middleware")
        wrapper = getattr(genai_mw.generate_content_wrapper, "_self_wrapper", genai_mw.generate_content_wrapper)
        mock_wrapped = MagicMock(return_value=SimpleNamespace(usage_metadata=None, text="ok"))
        kwargs = {
            "model": "gemini-2.0-flash",
            "contents": "hello",
            "config": types.GenerateContentConfig(system_instruction=SYS),
        }

        with caplog.at_level(logging.DEBUG, logger="revenium_middleware"):
            try:
                wrapper(mock_wrapped, None, (), kwargs)
            except Exception:
                pass  # downstream metering details are not this test's concern

        assert_no_sentinel(caplog.text)


class TestBedrockStreamDebugLog:
    def test_stream_chunk_reply_text_absent(self, caplog):
        import json

        from revenium_middleware.anthropic.bedrock_adapter import BedrockStreamIterator

        def event(payload):
            return {"chunk": {"bytes": json.dumps(payload).encode("utf-8")}}

        stream = [
            event({"type": "message_start", "message": {"usage": {"input_tokens": 3}}}),
            event({"type": "content_block_delta", "delta": {"type": "text_delta", "text": REPLY}}),
            event({"type": "message_stop"}),
        ]
        iterator = BedrockStreamIterator("claude-3-haiku-20240307", {}, "us-east-1")
        with caplog.at_level(logging.DEBUG, logger="revenium_middleware"):
            chunks = list(iterator._process_stream(stream))

        assert chunks == [REPLY]
        assert "Bedrock stream chunk" in caplog.text
        assert_no_sentinel(caplog.text)

    def test_no_chunk_debug_argument_is_built_when_debug_is_off(self, caplog, monkeypatch):
        """BACK-4365: each stream chunk must not pay for a debug line that is never printed."""
        import json

        from revenium_middleware.anthropic.bedrock_adapter import BedrockStreamIterator

        monkeypatch.delenv("REVENIUM_DEBUG", raising=False)

        def event(payload):
            return {"chunk": {"bytes": json.dumps(payload).encode("utf-8")}}

        stream = [
            event({"type": "message_start", "message": {"usage": {"input_tokens": 3}}}),
            event({"type": "content_block_delta", "delta": {"type": "text_delta", "text": REPLY}}),
            event({"type": "message_stop"}),
        ]
        iterator = BedrockStreamIterator("claude-3-haiku-20240307", {}, "us-east-1")
        with patch("revenium_middleware.anthropic.bedrock_adapter.sanitize_for_logging") as spy:
            with caplog.at_level(logging.INFO, logger=MIDDLEWARE_LOGGER):
                chunks = list(iterator._process_stream(stream))

        assert chunks == [REPLY]
        assert spy.call_count == 0


# ---------------------------------------------------------------------------
# Guard: every logger argument is sanitized or known to be safe
# ---------------------------------------------------------------------------

LOGGER_METHODS = frozenset({"debug", "info", "warning", "warn", "error", "exception", "critical", "log"})
ALWAYS_SAFE_CALLS = frozenset({"sanitize_for_logging", "redact_error_text", "type", "len", "bool", "dir"})
SAFE_NAME_SUFFIXES = ("_tokens", "_count", "_id", "_ms")
SAFE_NAMES = frozenset({"id", "status_code", "endpoint", "url", "method", "name", "__name__"})
EXCEPTION_NAMES = frozenset({"e", "ex", "exc", "err", "error"})
EXCEPTION_NAME_SUFFIXES = ("_exc", "_error")


def _last_segment(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _is_exception_name(node):
    name = _last_segment(node)
    return isinstance(node, ast.Name) and (name in EXCEPTION_NAMES or name.endswith(EXCEPTION_NAME_SUFFIXES))


def _is_safe_call(node):
    func = node.func
    if isinstance(func, ast.Name) and func.id in ALWAYS_SAFE_CALLS:
        return True
    if isinstance(func, ast.Name) and func.id == "list" and len(node.args) == 1:
        inner = node.args[0]
        return isinstance(inner, ast.Call) and isinstance(inner.func, ast.Attribute) and inner.func.attr == "keys"
    if isinstance(func, ast.Name) and func.id in ("str", "repr") and len(node.args) == 1:
        return _is_exception_name(node.args[0])
    return ast.unparse(node) == "traceback.format_exc()"


def _is_safe_argument(node):
    """Constants, sanitized values, counts and ids, exceptions, and module constants (ALL_CAPS)."""
    if isinstance(node, ast.Constant):
        return True
    if isinstance(node, ast.Call):
        return _is_safe_call(node)
    if _is_exception_name(node):
        return True
    name = _last_segment(node)
    if name is None:
        return False
    return name.endswith(SAFE_NAME_SUFFIXES) or name in SAFE_NAMES or (name.isupper() and not name.startswith("_"))


def _formatted_values(node):
    """The values a log call interpolates: f-string parts, ``.format()`` and ``%`` operands, plain arguments."""
    if isinstance(node, ast.JoinedStr):
        for part in node.values:
            if isinstance(part, ast.FormattedValue):
                yield part.value
    elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "format" \
            and isinstance(node.func.value, ast.Constant):
        yield from node.args
        yield from (keyword.value for keyword in node.keywords)
    elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mod) and isinstance(node.left, ast.Constant):
        yield from (node.right.elts if isinstance(node.right, ast.Tuple) else [node.right])
    elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        yield from _formatted_values(node.left)
        yield from _formatted_values(node.right)
    else:
        yield node


def _logger_calls(tree):
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
                and node.func.attr in LOGGER_METHODS and "log" in ast.unparse(node.func.value).lower():
            yield node


def _unsafe_log_arguments(package_root):
    for path in sorted(package_root.rglob("*.py")):
        source = path.read_text(encoding="utf-8")
        lines = source.splitlines()
        relative = path.relative_to(package_root).as_posix()
        for call in _logger_calls(ast.parse(source)):
            arguments = call.args[1:] if call.func.attr == "log" else call.args
            for argument in arguments:
                for value in _formatted_values(argument):
                    if not _is_safe_argument(value):
                        yield relative, lines[value.lineno - 1].strip(), ast.unparse(value)



ATTRIBUTION = "attribution value the customer set (task type, agent, trace type, rule name)"
AZURE_CONFIG = "Azure endpoint, deployment and API version; to_dict() leaves the key out"
CALLABLE = "repr of a function or signal handler"
EXCEPTION_TEXT = "SDK exception text"
HEADERS = "metering API response status phrase and headers"
ID_LIST = "field or attribute names, not their values"
LOCATION = "URL, file path, module path or patch target"
METERING_RESULT = "metering API reply: record id, label and signature"
MODEL_OR_PROVIDER = "model, provider, region or deployment name"
NUMBER = "number, duration or count"
OPERATION = "operation, event kind or call type label"
OUTCOME_BODY = "Revenium's own 409 reply, capped at 200 characters"
RULE = "usage field name, the SDK's fixed rule text and lengths, never the value"
SANITIZED = "built by sanitize_for_logging on the line above"
SETTING = "SDK setting, flag or version"
THREAD = "metering thread object"
TOKEN_USAGE = "token usage counts"
UNSENDABLE_VALUE = "SDK text naming a field path, a record or tool, and a value's type, never the value"

SAFE_LOG_LINES = {
    ('__init__.py', 'logger.warning(f"Invalid log level: {log_level}, defaulting to INFO")'): SETTING,
    ('_core/config.py', 'logger.warning("Invalid %s=%r; using default %s", name, raw, default)'): SETTING,
    ('_core/config.py', 'logger.warning("%s=%d is below the minimum; using %d", name, length, MIN_PROMPT_LENGTH)'): SETTING,
    ('_core/config.py', 'logger.warning("%s=%r is not a recognised value; using %s", name, raw, fallback)'): SETTING,
    ('_core/delivery_circuit.py', 'self._failures_to_open, self._probe_interval,'): NUMBER,
    ('_core/enforcement.py', 'resource, _FETCH_MAX_ATTEMPTS,'): OPERATION,
    ('_core/enforcement.py', 'rule_name, clause,'): ATTRIBUTION,
    ('_core/enforcement.py', '"against the rules snapshot that loaded", blocks_path,'): LOCATION,
    ('_core/enforcement.py', 'len(rules), len(departments.blocks), path,'): LOCATION,
    ('_core/enforcement.py', 'response.status_code, retry_after, budget_remaining,'): NUMBER,
    ('_core/enforcement.py', 'budget_total, cooldown,'): NUMBER,
    ('_core/enforcement.py', 'owner[1], backoff,'): SETTING,
    ('_core/enforcement.py', 'logger.info("Enforcement rules of the previous team were dropped; loading the rules of team %s", owner[1])'): SETTING,
    ('_core/enforcement.py', 'rule_id, rule.get("action"),'): OPERATION,
    ('_core/enforcement.py', 'rule_id, sanitize_for_logging(dimension), sanitize_for_logging(operator), reason,'): OPERATION,
    ('_core/enforcement.py', 'sanitize_for_logging(rule.get("name", "cost limit")), _rule_identifier(rule), rule.get("action"),'): OPERATION,
    ('_core/enforcement.py', 'logger.debug("Invalid %s=%r, using default", Config.ENV_REVENIUM_CB_POLL_INTERVAL_SECONDS, raw)'): SETTING,
    ('_core/enforcement.py', '"Failed to read department map from %s", blocks_path, exc_info=True'): LOCATION,
    ('_core/enforcement.py', 'logger.debug("Failed to read enforcement cache from %s", path, exc_info=True)'): LOCATION,
    ('_core/enforcement.py', 'logger.debug("Failed to write enforcement cache to %s", path, exc_info=True)'): LOCATION,
    ('_core/enforcement.py', 'logger.debug("Failed to fetch enforcement %s, falling open", resource, exc_info=True)'): OPERATION,
    ('_core/fields.py', 'logger.warning(msg, old_camel, old_snake, new_camel, new_camel, new_snake)'): ID_LIST,
    ('_core/json_safe.py', '_text_without_surrogates(subject),'): UNSENDABLE_VALUE,
    ('_core/json_safe.py', '_text_without_surrogates(path),'): UNSENDABLE_VALUE,
    ('_core/json_safe.py', 'reason,'): UNSENDABLE_VALUE,
    ('_core/load_diagnostics.py', 'provider_label,'): MODEL_OR_PROVIDER,
    ('_core/metering.py', 'undelivered, budget, Config.ENV_REVENIUM_SHUTDOWN_TIMEOUT_SECONDS,'): NUMBER,
    ('_core/metering_buffer.py', 'logger.debug("Buffered undelivered %s metering event (buffer depth: %d)", kind, depth)'): NUMBER,
    ('_core/metering_buffer.py', 'deadline_seconds)'): NUMBER,
    ('_core/metering_buffer.py', 'sent, expired, discarded, remaining,'): NUMBER,
    ('_core/metering_buffer.py', 'evicted, total,'): NUMBER,
    ('_core/metering_buffer.py', 'len(blocking), deadline_seconds)'): NUMBER,
    ('_core/metering_buffer.py', 'event.kind, error)'): OPERATION,
    ('_core/metering_buffer.py', 'self._max_size,'): NUMBER,
    ('_core/metering_buffer.py', 'asked,'): LOCATION,
    ('_core/metering_buffer.py', '_redirect_target(response),'): LOCATION,
    ('_core/metering_pool.py', 'overflowed,'): NUMBER,
    ('_core/metering_pool.py', 'self.queue_size,'): NUMBER,
    ('_core/metering_status.py', '"Metering error callback %r raised; ignoring", callback, exc_info=True'): CALLABLE,
    ('_core/metering_submission.py', 'args.get("input_token_count"),'): NUMBER,
    ('_core/metering_submission.py', 'args.get("output_token_count"),'): NUMBER,
    ('_core/metering_submission.py', 'args.get("total_token_count"),'): NUMBER,
    ('_core/metering_submission.py', 'operation,'): OPERATION,
    ('_core/outcomes.py', 'response.text,'): OUTCOME_BODY,
    ('_core/parse_failures.py', '"the response, so the call\'s usage was not recorded", provider, type(error).__name__'): MODEL_OR_PROVIDER,
    ('_core/patch_registry.py', 'logger.debug("Skipping already-patched function: %s", function_path)'): LOCATION,
    ('_core/prompt_extraction.py', 'f"{max_prompt_length} characters"'): NUMBER,
    ('_core/prompt_extraction.py', 'f"truncated: {prompts_truncated}"'): NUMBER,
    ('_core/shutdown_signals.py', 'logger.debug("SIGTERM handler installed; it chains to %r.", previous)'): CALLABLE,
    ('_core/shutdown_signals.py', '"where signal handlers cannot be installed; shutdown relies on atexit.", flag)'): SETTING,
    ('_core/shutdown_signals.py', '"Shutdown relies on atexit.", flag, e)'): SETTING,
    ('_core/trace_fields.py', 'f"{TRACE_TYPE_MAX_LENGTH} characters: \'{trace_type}\'. "'): ATTRIBUTION,
    ('_core/trace_fields.py', 'f"\'{trace_type}\'. Field will be omitted."'): ATTRIBUTION,
    ('_core/wire_guards.py', 'logger.warning(_DROPPED_MESSAGE, prefix, field, problem)'): RULE,
    ('_core/wire_guards.py', 'logger.warning(_CUT_MESSAGE, prefix, field, len(value), limit)'): RULE,
    ('_metering/_base_client.py', 'log.info("Retrying request to %s in %f seconds", options.url, timeout)'): NUMBER,
    ('_metering/_base_client.py', 'response.reason_phrase,'): HEADERS,
    ('_metering/_base_client.py', 'response.headers,'): HEADERS,
    ('_metering/_base_client.py', 'log.debug("%i retries left", remaining_retries)'): NUMBER,
    ('_metering/decorator.py', 'logger.debug("[metered] %s:%s %dms", tool_id, operation or "execute", duration_ms)'): OPERATION,
    ('anthropic/bedrock_adapter.py', 'logger.debug(f"Created new Bedrock client for region {region}")'): MODEL_OR_PROVIDER,
    ('anthropic/bedrock_adapter.py', 'logger.debug(f"Invoking Bedrock model {model_id} in region {region}")'): MODEL_OR_PROVIDER,
    ('anthropic/bedrock_adapter.py', 'logger.debug(f"Removed cached client for region {oldest_key} (cache full)")'): MODEL_OR_PROVIDER,
    ('anthropic/bedrock_adapter.py', 'logger.error(error_msg)'): EXCEPTION_TEXT,
    ('anthropic/bedrock_adapter.py', 'logger.debug(f"Starting Bedrock streaming for model {model_id} in region {self.region}")'): MODEL_OR_PROVIDER,
    ('anthropic/bedrock_adapter.py', 'logger.debug("Metering thread started for Bedrock stream: %s", thread)'): THREAD,
    ('anthropic/bedrock_adapter.py', 'logger.error(f"Bedrock streaming invoke failed for model {self.model} in region {self.region}: {e}")'): MODEL_OR_PROVIDER,
    ('anthropic/bedrock_adapter.py', 'logger.debug("Metering call result for Bedrock stream: %s", result)'): METERING_RESULT,
    ('anthropic/bedrock_adapter.py', 'logger.debug(f"Bedrock stream: {chunk_type}")'): OPERATION,
    ('anthropic/bedrock_transport.py', 'logger.debug("Bedrock transport metering result: %s", result)'): METERING_RESULT,
    ('anthropic/inference_profiles.py',
     'kind.subject, profile_arn, kind.verb, reason, kind.caveat, unresolved.consequence,'): MODEL_OR_PROVIDER,
    ('anthropic/inference_profiles.py',
     '_remedy(reason, transient, unresolved.remedy, kind.permission),'): MODEL_OR_PROVIDER,
    ('anthropic/inference_profiles.py', 'kind.subject, profile_arn, kind.verb, kind.caveat)'): MODEL_OR_PROVIDER,
    ('anthropic/inference_profiles.py', 'kind.subject, profile_arn, model_id)'): MODEL_OR_PROVIDER,
    ('anthropic/inference_profiles.py',
     'logger.debug("Bedrock %s %s %s %s, which is not metered here", kind.subject, profile_arn, kind.verb, model_id)'):
        MODEL_OR_PROVIDER,
    ('anthropic/middleware.py', 'logger.debug("Bedrock metering thread started: %s", thread)'): THREAD,
    ('anthropic/middleware.py', 'logger.debug("Metering thread started for raw stream: %s", thread)'): THREAD,
    ('anthropic/middleware.py', 'logger.debug("Metering thread started for stream: %s", thread)'): THREAD,
    ('anthropic/middleware.py', 'logger.debug("Metering thread started: %s", thread)'): THREAD,
    ('anthropic/middleware.py', 'logger.debug(f"Started thread-safe async operation: {thread}")'): THREAD,
    ('anthropic/middleware.py', 'logger.warning(f"usage_metadata for {operation_name} should be a dict, got {type(api_metadata)}. Using empty dict.")'): OPERATION,
    ('anthropic/middleware.py', 'logger.warning(f"Metadata for {operation_name} exceeds maximum depth {max_depth}. Truncating.")'): NUMBER,
    ('anthropic/middleware.py', 'logger.debug(f"Detected provider: {provider}")'): MODEL_OR_PROVIDER,
    ('anthropic/middleware.py', 'kwargs.get("model"), kwargs.get("max_tokens"))'): MODEL_OR_PROVIDER,
    ('anthropic/middleware.py', 'logger.debug("%s.%s is not in the installed anthropic SDK; not wrapped", module_name, attribute)'): LOCATION,
    ('anthropic/middleware.py', 'logger.debug("Bedrock metering call result: %s", result)'): METERING_RESULT,
    ('anthropic/middleware.py', 'logger.debug("Metering call result for raw stream: %s", result)'): METERING_RESULT,
    ('anthropic/middleware.py', 'logger.debug("Metering call result for stream: %s", result)'): METERING_RESULT,
    ('anthropic/middleware.py', 'logger.debug("Metering call result: %s", result)'): METERING_RESULT,
    ('anthropic/middleware.py', 'response_id, result'): METERING_RESULT,
    ('anthropic/middleware.py', 'logger.warning("[REVENIUM ERROR] Metering call did not return success for transaction %s: %s", response_id, result)'): METERING_RESULT,
    ('anthropic/middleware.py', 'logger.debug("Merged decorator metadata for %s: %s", operation_name, sanitize_for_logging(usage_metadata))'): OPERATION,
    ('anthropic/middleware.py', 'logger.debug("Usage metadata for %s: %s", operation_name, sanitize_for_logging(usage_metadata))'): OPERATION,
    ('anthropic/middleware.py', 'logger.debug("Anthropic client.messages.create wrapper called: %s: %s", wrapped, sanitize_for_logging(args))'): CALLABLE,
    ('anthropic/prompt_extractor.py', 'f"truncated: {prompts_truncated}"'): NUMBER,
    ('anthropic/provider.py', 'logger.debug(f"Provider detection completed: {cache.detected_provider}")'): MODEL_OR_PROVIDER,
    ('anthropic/provider.py', 'logger.debug(f"Foundry provider detected via base_url: {candidate}")'): LOCATION,
    ('anthropic/provider.py', 'logger.debug(f"Bedrock provider detected via base_url: {base_url}")'): LOCATION,
    ('anthropic/provider.py', 'logger.debug(f"Bedrock provider detected via client.base_url: {client.base_url}")'): LOCATION,
    ('fal/_metering.py', 'logger.debug("Metering thread started: %s", thread)'): THREAD,
    ('fal/_metering.py', 'application, media_type,'): MODEL_OR_PROVIDER,
    ('fal/_metering.py', 'logger.debug("Metering call result: %s", metering_result)'): METERING_RESULT,
    ('fal/middleware.py', 'request_id, call.application,'): MODEL_OR_PROVIDER,
    ('fal/trace_fields.py', 'model,'): MODEL_OR_PROVIDER,
    ('google/__init__.py', '", ".join(active_sdks) if active_sdks else "none",'): LOCATION,
    ('google/common/utils.py', 'operation_type.lower(),'): OPERATION,
    ('google/common/utils.py', 'f"Metering call for {operation_type.value}: {transaction_id}, tokens: {prompt_tokens}+{completion_tokens}={total_tokens}"'): OPERATION,
    ('google/common/utils.py', 'operation_subtype,'): OPERATION,
    ('google/common/utils.py', 'available_attrs[:10],'): ID_LIST,
    ('google/common/utils.py', 'logger.info(" REVENIUM SUCCESS: %s successful: %s", label, result.id)'): OPERATION,
    ('google/common/utils.py', 'logger.debug("Metering call result: %s", result)'): METERING_RESULT,
    ('google/common/utils.py', 'logger.debug("Image metering call result: %s", result)'): METERING_RESULT,
    ('google/common/utils.py', 'logger.debug("Video metering call result: %s", result)'): METERING_RESULT,
    ('google/common/utils.py', '"Could not extract model name from response, using fallback: %s", fallback'): MODEL_OR_PROVIDER,
    ('google/common/utils.py', 'f"Flat subscriber keys are deprecated: {flat_keys_used}. "'): ID_LIST,
    ('google/common/utils.py', 'logger.debug("Response is None, using fallback model name: %s", fallback)'): MODEL_OR_PROVIDER,
    ('google/common/utils.py', 'logger.error(" REVENIUM FAILURE: Error details: %s", error_details)'): ID_LIST,
    ('google/google_ai/middleware.py', 'logger.debug(f"Usage data extracted: {usage_data}")'): TOKEN_USAGE,
    ('google/google_ai/middleware.py', 'logger.log(level, "Google AI %s metering failed: %s", operation, error, exc_info=True)'): OPERATION,
    ('google/google_ai/middleware.py', 'operation, sanitize_for_logging(args), sanitize_for_logging(kwargs)'): OPERATION,
    ('google/google_ai/middleware.py', 'self.model,'): MODEL_OR_PROVIDER,
    ('google/google_ai/middleware.py', 'time_to_first_token,'): NUMBER,
    ('google/google_ai/middleware.py', 'self._max_chunks,'): NUMBER,
    ('google/google_ai/provider.py', 'f"Google AI SDK endpoint detection completed: {_detected_endpoint}"'): MODEL_OR_PROVIDER,
    ('google/prompt_extractor.py', 'f"truncated: {prompts_truncated}"'): NUMBER,
    ('google/vertex_ai/middleware.py', 'f"Vertex AI Imagen generate_images: model={model_name}, "'): MODEL_OR_PROVIDER,
    ('google/vertex_ai/middleware.py', 'f"requested={number_of_images}, actual={actual_image_count}"'): NUMBER,
    ('google/vertex_ai/middleware.py', 'logger.debug(f" Final stop_reason after normalization: {stop_reason}")'): MODEL_OR_PROVIDER,
    ('google/vertex_ai/middleware.py', 'logger.info(" Vertex AI generation wrappers applied to: %s", ", ".join(applied))'): LOCATION,
    ('google/vertex_ai/middleware.py', 'logger.info(f" Vertex AI Imagen wrappers applied to: {\', \'.join(wrapped_modules)}")'): LOCATION,
    ('google/vertex_ai/middleware.py', 'f"Extracted model name from Vertex AI _raw_response.model_version: {model_name}"'): MODEL_OR_PROVIDER,
    ('google/vertex_ai/middleware.py', 'self.model,'): MODEL_OR_PROVIDER,
    ('google/vertex_ai/middleware.py', 'time_to_first_token,'): NUMBER,
    ('google/vertex_ai/middleware.py', 'f" Raw vertex_finish_reason: {vertex_finish_reason} (type: {type(vertex_finish_reason)})"'): MODEL_OR_PROVIDER,
    ('google/vertex_ai/middleware.py', 'logger.debug("Module %s not available", module_path)'): LOCATION,
    ('google/vertex_ai/middleware.py', 'self._max_chunks,'): NUMBER,
    ('google/vertex_ai/middleware.py', 'logger.debug(f"  Module {module_path} not available for Imagen")'): LOCATION,
    ('google/vertex_ai/middleware.py', 'logger.debug(f"  Error applying Imagen wrapper to {module_path}: {e}")'): LOCATION,
    ('google/vertex_ai/middleware.py', 'logger.debug(f" Converted enum to string: {vertex_finish_reason}")'): MODEL_OR_PROVIDER,
    ('google/vertex_ai/middleware.py', 'logger.debug(f" Converted to string: {vertex_finish_reason}")'): MODEL_OR_PROVIDER,
    ('google/vertex_ai/middleware.py', 'logger.debug(f" Applied Imagen wrapper to {patch_key}")'): LOCATION,
    ('google/vertex_ai/provider.py', 'f"Vertex AI configuration validated: project={project_id}, location={location}"'): MODEL_OR_PROVIDER,
    ('griptape/litellm_driver.py', 'logger.debug(f"Making LiteLLM completion call with model: {self.model}")'): MODEL_OR_PROVIDER,
    ('griptape/litellm_driver.py', 'logger.debug(f"Adjusted proxy base URL for OpenAI client: {self.proxy_base_url}")'): LOCATION,
    ('griptape/litellm_driver.py', 'logger.info(f"Initialized LiteLLM driver for model: {model} via proxy: {self.proxy_url}")'): MODEL_OR_PROVIDER,
    ('griptape/litellm_driver.py', 'logger.info(f"Initialized LiteLLM driver for model: {model} (direct API)")'): MODEL_OR_PROVIDER,
    ('griptape/litellm_driver.py', 'logger.debug(f"Making proxy call with model: {self.model} via {self.proxy_url}")'): MODEL_OR_PROVIDER,
    ('griptape/litellm_driver.py', 'logger.debug(f"Making official Revenium proxy call to: {self.proxy_url}")'): LOCATION,
    ('griptape/litellm_driver.py', 'logger.debug(f"Making streaming LiteLLM completion call with model: {self.model} via proxy: {self.proxy_url}")'): MODEL_OR_PROVIDER,
    ('griptape/litellm_driver.py', 'logger.debug(f"Making streaming LiteLLM completion call with model: {self.model}")'): MODEL_OR_PROVIDER,
    ('griptape/litellm_driver.py', 'logger.debug(f"OpenAI client will use base URL: {self.proxy_base_url}")'): LOCATION,
    ('griptape/litellm_driver.py', 'f"Subscriber fields {dropped} have no supported header path through the "'): ID_LIST,
    ('griptape/litellm_driver.py', 'logger.warning(f"Usage metadata \'{key}\' contains a line break or control character; "'): ID_LIST,
    ('griptape/litellm_driver.py', 'logger.warning(f"Usage metadata \'{key}\' is not representable in an HTTP header; "'): ID_LIST,
    ('griptape/litellm_driver.py', 'logger.warning(f"Usage metadata key {key!r} cannot be an HTTP header name; "'): ID_LIST,
    ('griptape/openai_driver.py', 'logger.info(f"ReveniumOpenAiDriver initialized with model: {self.model}")'): MODEL_OR_PROVIDER,
    ('griptape/openai_embedding_driver.py', 'logger.info(f"ReveniumOpenAiEmbeddingDriver initialized with model: {self.model}")'): MODEL_OR_PROVIDER,
    ('griptape/universal_driver.py', 'logger.info(f"Initialized Revenium driver for provider: {self._provider}")'): MODEL_OR_PROVIDER,
    ('griptape/universal_driver.py', 'logger.info(f"Unknown model \'{model}\', defaulting to LiteLLM provider")'): MODEL_OR_PROVIDER,
    ('griptape/universal_driver.py', 'logger.info(f"Initialized Revenium embedding driver for provider: {self._provider}")'): MODEL_OR_PROVIDER,
    ('griptape/universal_driver.py', 'logger.info(f"Unknown embedding model \'{model}\', defaulting to OpenAI provider")'): MODEL_OR_PROVIDER,
    ('griptape/universal_driver.py', 'f"Bare open-model name \'{model}\' assumed to be a local Ollama model; "'): MODEL_OR_PROVIDER,
    ('griptape/universal_driver.py', 'f"use a provider-qualified id (e.g. \'groq/{model}\') or force_provider to override"'): MODEL_OR_PROVIDER,
    ('griptape/universal_driver.py', 'logger.error(f"Failed to import {provider} driver: {e}")'): MODEL_OR_PROVIDER,
    ('griptape/universal_driver.py', 'type(base_driver).__name__, ", ".join(dropped))'): ID_LIST,
    ('griptape/universal_driver.py', 'logger.error(f"Failed to import {provider} embedding driver: {e}")'): MODEL_OR_PROVIDER,
    ('griptape/universal_driver.py', 'logger.info(f"Unknown driver type \'{driver_class}\', defaulting to LiteLLM provider")'): MODEL_OR_PROVIDER,
    ('griptape/universal_driver.py', 'logger.info(f"Unknown embedding driver type \'{driver_class}\'")'): MODEL_OR_PROVIDER,
    ('litellm/client/hooks.py', 'logger.debug(f"Registered metadata hook: {hook.__name__} with priority {priority}")'): NUMBER,
    ('litellm/client/hooks.py', 'logger.debug(f"Cleared {count} metadata hooks")'): NUMBER,
    ('litellm/client/hooks.py', 'logger.debug(f"Executing hook: {hook.__name__} (priority: {priority})")'): NUMBER,
    ('litellm/client/integrations/crewai.py', 'logger.info(f"CrewAI {\'.\'.join(map(str, version))} detected - using callback approach (crew-level metadata only)")'): LOCATION,
    ('litellm/client/integrations/crewai.py', 'logger.info(f"CrewAI {\'.\'.join(map(str, version))} detected - using monkey-patching for task-level metadata")'): LOCATION,
    ('litellm/client/integrations/crewai.py', 'logger.debug(f"Tracking task with agent={agent_role}, task_type={task_type}")'): ATTRIBUTION,
    ('litellm/client/integrations/crewai.py', 'f"Executing task with agent={agent_role}, "'): ATTRIBUTION,
    ('litellm/client/integrations/crewai.py', 'f"task_type={task_type}"'): ATTRIBUTION,
    ('litellm/client/integrations/crewai.py', 'logger.warning("CrewAI %s calls are not metered: import %s before kickoff", type(llm).__name__, middleware_module)'): LOCATION,
    ('litellm/client/middleware.py', 'logger.debug("Metering thread started: %s", thread)'): THREAD,
    ('litellm/client/middleware.py', 'logger.debug("Metering call result: %s", result)'): METERING_RESULT,
    ('litellm/proxy/guardrail.py', 'count,'): NUMBER,
    ('litellm/proxy/guardrail.py', 'getattr(self, "guardrail_name", None),'): SETTING,
    ('litellm/proxy/guardrail.py', 'getattr(self, "event_hook", None),'): SETTING,
    ('litellm/proxy/guardrail.py', 'scan_raw_request,'): SETTING,
    ('litellm/proxy/guardrail.py', 'getattr(self, "default_on", False),'): SETTING,
    ('litellm/proxy/guardrail.py', 'self._meters_every_request,'): SETTING,
    ('litellm/proxy/guardrail.py', '"ReveniumGuardrail.async_pre_call_hook fired (call_type=%s)", call_type'): OPERATION,
    ('litellm/proxy/guardrail.py', 'call_type,'): OPERATION,
    ('litellm/proxy/guardrail.py', 'self._shared_id_active,'): SETTING,
    ('litellm/proxy/guardrail.py', '_headers_hook_available(),'): SETTING,
    ('litellm/proxy/guardrail.py', 'litellm_version,'): SETTING,
    ('litellm/proxy/guardrail.py', 'error.rule_name,'): ATTRIBUTION,
    ('litellm/proxy/guardrail.py', 'error.current_value,'): NUMBER,
    ('litellm/proxy/guardrail.py', 'error.threshold,'): NUMBER,
    ('litellm/proxy/guardrail.py', 'logger.debug("ReveniumGuardrail: metering result (%s): %s", label, result)'): METERING_RESULT,
    ('litellm/proxy/guardrail.py', '"ReveniumGuardrail: metering call failed (%s): %s", label, error'): OPERATION,
    ('litellm/proxy/guardrail.py', 'logger.debug("ReveniumGuardrail: metering (%s) with args: %s", label, sanitize_for_logging(completion_args))'): OPERATION,
    ('litellm/proxy/middleware.py', 'logger.debug("Proxy metering call result: %s", result)'): METERING_RESULT,
    ('litellm/proxy/middleware.py', 'logger.debug("Result from create_completion (failure): %s", result)'): METERING_RESULT,
    ('ollama/middleware.py', 'logger.debug("Metering thread started: %s", thread)'): THREAD,
    ('ollama/middleware.py', 'logger.debug("Metering call result: %s", result)'): METERING_RESULT,
    ('openai/azure_config.py', 'f"deployment: {self.deployment}, api_version: {self.api_version}")'): MODEL_OR_PROVIDER,
    ('openai/azure_model_resolver.py', 'logger.debug(f"Resolving Azure model name for deployment: {deployment_name}")'): MODEL_OR_PROVIDER,
    ('openai/azure_model_resolver.py', 'logger.debug(f"No heuristic pattern matched for deployment: {deployment_name}")'): MODEL_OR_PROVIDER,
    ('openai/azure_model_resolver.py', 'logger.warning(f"No heuristic match found for Azure deployment \'{deployment_name}\'. "'): MODEL_OR_PROVIDER,
    ('openai/azure_model_resolver.py', 'logger.warning(f"Using deployment name \'{deployment_name}\' as model name. "'): MODEL_OR_PROVIDER,
    ('openai/azure_model_resolver.py', 'logger.debug(f"Heuristic mapping successful: {deployment_name} -> {heuristic_name}")'): MODEL_OR_PROVIDER,
    ('openai/azure_model_resolver.py', 'logger.debug(f"Attempting Azure API resolution for deployment: {deployment_name}")'): MODEL_OR_PROVIDER,
    ('openai/azure_model_resolver.py', 'logger.warning(f"Azure model resolution failed for \'{deployment_name}\': {str(e)}. "'): MODEL_OR_PROVIDER,
    ('openai/azure_model_resolver.py', 'logger.debug(f"Attempting Azure API model resolution: {api_url}")'): LOCATION,
    ('openai/azure_model_resolver.py', 'logger.debug(f"Found cached mapping: {deployment_name} -> {cached_name}")'): MODEL_OR_PROVIDER,
    ('openai/azure_model_resolver.py', 'logger.debug(f"Azure API resolved: {deployment_name} -> {actual_model}")'): MODEL_OR_PROVIDER,
    ('openai/azure_model_resolver.py', 'logger.debug(f"Azure API resolution failed for {deployment_name}: HTTP {response.status_code}")'): MODEL_OR_PROVIDER,
    ('openai/azure_model_resolver.py', 'logger.debug(f"Background Azure API resolution failed for {deployment_name}: {str(e)}")'): MODEL_OR_PROVIDER,
    ('agentic_outcomes.py', '", ".join(sorted(new_keys)),'): ID_LIST,
    ('openai/langchain/_perplexity_cost_capture.py',
     'logger.debug("Perplexity\'s reported cost is not captured for %s: %s", key, error)'): LOCATION,
    ('openai/langchain/_text_completion_usage.py',
     'logger.debug("A streamed text completion\'s usage is not captured for %s: %s", key, error)'): LOCATION,
    ('openai/langchain/unified_handler.py', 'logger.debug(f"LLM started ({context}) - run_id: {run_id}")'): OPERATION,
    ('openai/langchain/unified_handler.py', 'error.name or error,'): LOCATION,
    ('openai/langchain/unified_handler.py', '_INSTALL_COMMAND,'): SETTING,
    ('openai/langchain/unified_handler.py', 'logger.debug(f"LLM ended ({context}) - run_id: {run_id}")'): OPERATION,
    ('openai/langchain/unified_handler.py', 'logger.debug(f"Extracted model name: {model_name}")'): MODEL_OR_PROVIDER,
    ('openai/langchain/unified_handler.py', 'logger.debug(f"Found usage_metadata: {usage_data}")'): TOKEN_USAGE,
    ('openai/langchain/unified_handler.py', 'logger.debug(f"Found usage in response_metadata: {usage_data}")'): TOKEN_USAGE,
    ('openai/langchain/unified_handler.py', 'logger.debug(f"Found direct usage attribute: {usage_data}")'): TOKEN_USAGE,
    ('openai/langchain/unified_handler.py', 'logger.debug(f"Found usage in generation message: {usage_data}")'): TOKEN_USAGE,
    ('openai/langchain/unified_handler.py', 'logger.debug(f"Found token_usage in llm_output: {usage_data}")'): TOKEN_USAGE,
    ('openai/langchain/unified_handler.py', 'logger.debug(f"Found usage in additional_kwargs: {usage_data}")'): TOKEN_USAGE,
    ('openai/langchain/unified_handler.py', 'logger.debug(f"Revenium metering call successful: {result}")'): METERING_RESULT,
    ('openai/langchain/unified_handler.py', 'logger.debug(f"Found usage in llm_output: {usage_data}")'): TOKEN_USAGE,
    ('openai/middleware.py', 'logger.debug(f"Model resolution: {raw_model_name} -> {resolved_model_name}")'): MODEL_OR_PROVIDER,
    ('openai/middleware.py', 'operation_type.lower(), input_tokens, output_tokens, total_tokens, transaction_id'): OPERATION,
    ('openai/middleware.py', 'logger.debug("Metering call to Revenium for %s operation %s", operation_type.lower(), response_id)'): OPERATION,
    ('openai/middleware.py', 'logger.debug(f"Metering call for {operation_type.value}: {response_id}, tokens: {prompt_tokens}+{completion_tokens}={total_tokens}")'): OPERATION,
    ('openai/middleware.py', 'logger.debug("%s metering thread started: %s", operation_type, thread)'): THREAD,
    ('openai/middleware.py', '"will not be metered", self._capacity'): NUMBER,
    ('openai/middleware.py', 'logger.debug(f"Azure OpenAI configuration validated: {azure_config.to_dict()}")'): AZURE_CONFIG,
    ('openai/middleware.py', 'logger.warning(f"Unmapped finish reason \'{openai_finish_reason}\' defaulting to \'END\'. "'): MODEL_OR_PROVIDER,
    ('openai/middleware.py', 'logger.debug(f"Azure model resolution: {raw_model_name} -> {resolved_model_name}")'): MODEL_OR_PROVIDER,
    ('openai/middleware.py', 'logger.debug(f"OLLAMA provider detected via system_fingerprint: {system_fingerprint}")'): MODEL_OR_PROVIDER,
    ('openai/middleware.py', 'logger.debug(f"Using provider: {provider}, model_source: {model_source}")'): MODEL_OR_PROVIDER,
    ('openai/middleware.py', 'logger.debug("Metering call result: %s", result)'): METERING_RESULT,
    ('openai/middleware.py', 'f"{azure_config.to_dict()}"'): AZURE_CONFIG,
    ('openai/middleware.py', 'logger.error(f"❌ REVENIUM FAILURE: Completion args were: {sanitized_args}")'): SANITIZED,
    ('openai/middleware.py', 'logger.debug(f"Captured system_fingerprint from stream chunk: {self.system_fingerprint}")'): MODEL_OR_PROVIDER,
    ('openai/middleware.py', 'logger.debug(f"System fingerprint already set: {self.system_fingerprint}")'): MODEL_OR_PROVIDER,
    ('openai/middleware.py', 'logger.debug(f"Found usage data in chunk: {chunk.usage}")'): TOKEN_USAGE,
    ('openai/middleware.py', 'logger.debug("Streaming metering thread started: %s", thread)'): THREAD,
    ('openai/middleware.py', 'logger.debug(f"Azure streaming model resolution: {raw_model_name} -> {resolved_model_name}")'): MODEL_OR_PROVIDER,
    ('openai/middleware.py', 'logger.debug(f"Time to first token: {time_to_first_token}ms")'): NUMBER,
    ('openai/middleware.py', 'logger.debug("Provider detected: %s, metadata: %s", provider, sanitize_for_logging(provider_metadata))'): MODEL_OR_PROVIDER,
    ('openai/provider.py', 'logger.debug(f"Azure provider detected via base_url substring: {base_url}")'): LOCATION,
    ('openai/provider.py', 'logger.debug(f"OLLAMA provider detected via base_url: {base_url}")'): LOCATION,
    ('openai/provider.py', 'logger.debug(f"Azure provider detected via client type: {client_class_name}")'): MODEL_OR_PROVIDER,
    ('openai/provider.py', 'logger.debug(f"Azure provider detected via client.base_url: {client.base_url}")'): LOCATION,
    ('openai/provider.py', 'logger.debug(f"OLLAMA provider detected via client.base_url: {client.base_url}")'): LOCATION,
    ('openai/provider.py', 'logger.debug(f"Azure provider detected via AZURE_OPENAI_ENDPOINT: {azure_endpoint}")'): LOCATION,
    ('openai/provider.py', 'logger.debug(f"Provider detection completed: {_detected_provider}")'): MODEL_OR_PROVIDER,
    ('perplexity/middleware.py', 'logger.debug(f"Metering thread started: {thread}")'): THREAD,
    ('perplexity/middleware.py', 'logger.debug(f"Metering call result: {result}")'): METERING_RESULT,
    ('perplexity/patching.py', 'logger.debug("%s is not installed, so %s is not wrapped: %s", module, function_path, e)'): LOCATION,
    ('perplexity/patching.py', 'function_path, e,'): LOCATION,
    ('perplexity/patching.py', 'logger.debug("%s is not wrapped: %s", function_path, e)'): LOCATION,
    ('perplexity/perplexity_sdk.py', 'logger.debug(f"Metering call result: {result}")'): METERING_RESULT,
    ('perplexity/provider.py', 'logger.debug(f"Provider metadata: {metadata}")'): MODEL_OR_PROVIDER,
    ('perplexity/provider.py', 'logger.debug(f"Perplexity provider detected via base_url: {base_url}")'): LOCATION,
    ('perplexity/provider.py', 'logger.debug(f"Perplexity provider detected via client.base_url: {client.base_url}")'): LOCATION,
}


class TestNoUnsanitizedPayloadLogLine:
    def test_every_payload_log_argument_goes_through_the_sanitizer(self):
        package_root = Path(revenium_middleware.__file__).parent
        unsafe = sorted(
            f"{path}: {expression}  <-  {line}"
            for path, line, expression in _unsafe_log_arguments(package_root)
            if (path, line) not in SAFE_LOG_LINES
        )
        assert not unsafe, (
            "Log arguments that are neither sanitized nor known safe. Wrap payloads in "
            "sanitize_for_logging(...); add a genuinely safe value (an id, a count, a name) to "
            "SAFE_LOG_LINES with its reason:\n" + "\n".join(unsafe)
        )

    def test_allowlist_has_no_stale_entries(self):
        package_root = Path(revenium_middleware.__file__).parent
        flagged = {(path, line) for path, line, _ in _unsafe_log_arguments(package_root)}
        stale = sorted(set(SAFE_LOG_LINES) - flagged)
        assert not stale, f"SAFE_LOG_LINES entries no longer match a flagged log line: {stale}"
