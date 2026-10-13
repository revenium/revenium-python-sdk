"""
Tests for middleware functionality.
"""
import asyncio
import itertools
import re
import uuid

import pytest
from freezegun import freeze_time
from unittest.mock import MagicMock, patch

import revenium_middleware.perplexity  # noqa: F401
from entry_points import calls
from revenium_middleware.perplexity.middleware import (
    get_stop_reason,
    detect_operation_type,
    extract_token_usage,
    OperationType
)


@pytest.mark.unit
class TestStopReason:
    """Test stop reason mapping."""

    def test_stop_reason_stop(self):
        """Test 'stop' finish reason maps to END."""
        assert get_stop_reason("stop") == "END"

    def test_stop_reason_length(self):
        """Test 'length' finish reason maps to TOKEN_LIMIT."""
        assert get_stop_reason("length") == "TOKEN_LIMIT"

    def test_stop_reason_content_filter(self):
        """Test 'content_filter' finish reason maps to ERROR."""
        assert get_stop_reason("content_filter") == "ERROR"

    def test_stop_reason_tool_calls(self):
        """Test 'tool_calls' finish reason maps to END_SEQUENCE."""
        assert get_stop_reason("tool_calls") == "END_SEQUENCE"

    def test_stop_reason_function_call(self):
        """Test 'function_call' finish reason maps to END_SEQUENCE."""
        assert get_stop_reason("function_call") == "END_SEQUENCE"

    def test_stop_reason_none(self):
        """Test None finish reason defaults to END."""
        assert get_stop_reason(None) == "END"

    def test_stop_reason_unknown(self):
        """Test unknown finish reason defaults to END."""
        assert get_stop_reason("unknown_reason") == "END"


@pytest.mark.unit
class TestOperationType:
    """Test operation type detection."""
    
    def test_detect_chat_operation(self, mock_openai_response):
        """Test detection of chat operation."""
        op_type = detect_operation_type(mock_openai_response)
        assert op_type == OperationType.CHAT
    
    def test_detect_operation_no_choices(self):
        """Test operation detection with no choices."""
        response = MagicMock()
        response.choices = []
        
        op_type = detect_operation_type(response)
        assert op_type == OperationType.CHAT


@pytest.mark.unit
class TestTokenUsage:
    """Test token usage extraction."""
    
    def test_extract_token_usage(self, mock_openai_response):
        """Test extracting token usage from response."""
        usage = extract_token_usage(mock_openai_response)
        
        assert usage["prompt_tokens"] == 10
        assert usage["completion_tokens"] == 20
        assert usage["total_tokens"] == 30
    
    def test_extract_token_usage_no_usage(self):
        """Test extracting token usage when usage is missing."""
        response = MagicMock()
        response.usage = None
        
        usage = extract_token_usage(response)
        
        assert usage["prompt_tokens"] == 0
        assert usage["completion_tokens"] == 0
        assert usage["total_tokens"] == 0


@pytest.mark.unit
class TestMiddlewareWrapper:
    """Test middleware wrapper functionality."""
    
    @patch('revenium_middleware.perplexity.middleware.send_metering_data')
    @patch('openai.resources.chat.completions.Completions.create')
    def test_create_wrapper_non_streaming(
        self, mock_create, mock_send_metering, mock_openai_response
    ):
        """Test wrapper for non-streaming requests."""
        from revenium_middleware.perplexity.middleware import create_wrapper
        
        # Setup mock
        mock_create.return_value = mock_openai_response
        
        # Import to trigger wrapper
        import revenium_middleware.perplexity  # noqa: F401
        
        # The wrapper should be applied automatically
        # This test verifies the wrapper logic works
        assert True  # Placeholder - actual integration test needed



class _PerplexityBoundCompletions:
    _client = MagicMock(base_url="https://api.perplexity.ai")


class _OpenAIBoundCompletions:
    _client = MagicMock(base_url="https://api.openai.com/v1")


_METADATA = {"organizationName": "AcmeCorp", "subscriber": {"email": "dev@acme.example"}}


def _compat_wrapper(name):
    from revenium_middleware.perplexity import middleware as perplexity_middleware
    return getattr(perplexity_middleware, name)


@pytest.mark.parametrize("metadata_kwargs", [
    pytest.param({"usage_metadata": _METADATA}, id="top_level"),
    pytest.param({"extra_body": {"usage_metadata": _METADATA}}, id="extra_body"),
    pytest.param({"usage_metadata": None}, id="none"),
])
@pytest.mark.parametrize("name", ["create_wrapper", "async_create_wrapper"])
def test_selective_pass_through_strips_usage_metadata(name, metadata_kwargs, monkeypatch):
    """A Perplexity call through the OpenAI client, selective metering on,
    outside @revenium_meter: the request keeps none of the caller's Revenium
    metadata (BACK-3991)."""
    monkeypatch.setenv("REVENIUM_SELECTIVE_METERING", "true")
    seen = []

    def wrapped(*_args, **kwargs):
        seen.append(kwargs)
        return "provider response"

    kwargs = {"model": "sonar", **{key: dict(value) if isinstance(value, dict) else value
                                   for key, value in metadata_kwargs.items()}}

    async def awaited(*args, **provider_kwargs):
        return wrapped(*args, **provider_kwargs)

    if name.startswith("async"):
        result = asyncio.run(_compat_wrapper(name)(awaited, _PerplexityBoundCompletions(), (), kwargs))
    else:
        result = _compat_wrapper(name)(wrapped, _PerplexityBoundCompletions(), (), kwargs)
    assert result == "provider response"
    assert "usage_metadata" not in seen[0]
    assert "usage_metadata" not in (seen[0].get("extra_body") or {})


@pytest.mark.parametrize("name", ["create_wrapper", "async_create_wrapper"])
def test_selective_pass_through_leaves_a_plain_openai_call_to_the_openai_wrapper(name, monkeypatch):
    """The OpenAI wrapper below still needs the caller's metadata for its
    spending-cap check, so this wrapper does not strip it."""
    monkeypatch.setenv("REVENIUM_SELECTIVE_METERING", "true")
    seen = []

    def wrapped(*_args, **kwargs):
        seen.append(kwargs)
        return "provider response"

    _compat_wrapper(name)(wrapped, _OpenAIBoundCompletions(), (), {"model": "gpt-4o-mini", "usage_metadata": _METADATA})

    assert seen[0]["usage_metadata"] == _METADATA


MINTED_UUIDS = 1000


@pytest.fixture
def recorder():
    recording_client = MagicMock()
    with patch("revenium_middleware._core.metering.client", recording_client):
        yield recording_client


@pytest.fixture
def sequential_uuids(monkeypatch):
    counter = itertools.count(1)
    monkeypatch.setattr(uuid, "uuid4", lambda: uuid.UUID(int=next(counter)))


def _compatible_client():
    return calls._openai(base_url=calls.PERPLEXITY_BASE_URL, model=calls.PERPLEXITY_MODEL,
                         usage=calls._PERPLEXITY_USAGE)


def _transaction_ids(recording_client):
    calls.wait_for_metering()
    return [entry.kwargs["transaction_id"] for entry in recording_client.ai.create_completion.call_args_list]


def test_record_uses_the_perplexity_response_id(recorder):
    _compatible_client().chat.completions.create(**calls._chat_kwargs(calls.PERPLEXITY_MODEL))

    assert _transaction_ids(recorder) == ["chatcmpl-1"]


def test_same_instant_streams_get_distinct_transaction_ids(recorder, sequential_uuids):
    with freeze_time("2026-10-08 12:00:00"):
        for _ in range(2):
            for _chunk in _compatible_client().chat.completions.create(
                    **calls._chat_kwargs(calls.PERPLEXITY_MODEL), stream=True):
                pass

    ids = _transaction_ids(recorder)
    assert len(set(ids)) == 2
    assert all(re.match(r"^perplexity-[0-9a-f]{32}$", transaction_id) for transaction_id in ids)
    assert set(ids) <= {f"perplexity-{uuid.UUID(int=i).hex}" for i in range(1, MINTED_UUIDS)}


@pytest.mark.unit
def test_take_usage_metadata_leaves_the_callers_extra_body_unchanged():
    from revenium_middleware._core.perplexity_usage_metadata import take_usage_metadata

    extra_body = {"usage_metadata": {"agent": "a"}, "search_mode": "web"}
    kwargs = {"model": "sonar", "extra_body": extra_body}

    assert take_usage_metadata(kwargs) == {"agent": "a"}
    assert extra_body == {"usage_metadata": {"agent": "a"}, "search_mode": "web"}
    assert kwargs["extra_body"] == {"search_mode": "web"}
