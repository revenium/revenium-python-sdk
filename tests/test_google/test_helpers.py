"""
Copyright (c) 2025 Revenium, Inc.
SPDX-License-Identifier: MIT
"""

"""
Test helper utilities.

Provides utility functions for:
- Creating mock objects
- Validating responses
- Comparing metadata
- Generating test data
"""

import time
from typing import Dict, Any, Optional, List
from unittest.mock import Mock


def create_mock_response(
    text: str = "Test response",
    finish_reason: str = "STOP",
    prompt_tokens: int = 10,
    completion_tokens: int = 15,
    total_tokens: int = 25,
    cached_tokens: int = 0
) -> Mock:
    """
    Create a mock Google API response.

    Args:
        text: Response text
        finish_reason: Finish reason (STOP, MAX_TOKENS, etc.)
        prompt_tokens: Number of prompt tokens
        completion_tokens: Number of completion tokens
        total_tokens: Total tokens
        cached_tokens: Cached tokens

    Returns:
        Mock response object
    """
    mock_response = Mock()
    mock_response.text = text

    # Mock candidates
    mock_candidate = Mock()
    mock_candidate.finish_reason = finish_reason
    mock_candidate.content = Mock()
    mock_candidate.content.parts = [Mock(text=text)]
    mock_response.candidates = [mock_candidate]

    # Mock usage metadata
    mock_usage = Mock()
    mock_usage.prompt_token_count = prompt_tokens
    mock_usage.candidates_token_count = completion_tokens
    mock_usage.total_token_count = total_tokens
    mock_usage.cached_content_token_count = cached_tokens
    mock_response.usage_metadata = mock_usage

    return mock_response


def create_streaming_chunks(
    num_chunks: int = 5,
    final_finish_reason: str = "STOP",
    total_prompt_tokens: int = 50,
    total_completion_tokens: int = 75
) -> List[Mock]:
    """
    Create a list of mock streaming chunks.

    Args:
        num_chunks: Number of chunks to create
        final_finish_reason: Finish reason for final chunk
        total_prompt_tokens: Total prompt tokens (in final chunk)
        total_completion_tokens: Total completion tokens (in final chunk)

    Returns:
        List of mock chunk objects
    """
    chunks = []

    for i in range(num_chunks):
        mock_chunk = Mock()
        mock_chunk.text = f"chunk_{i}"

        # Mock candidates
        mock_candidate = Mock()
        if i == num_chunks - 1:
            # Final chunk has finish reason and usage
            mock_candidate.finish_reason = final_finish_reason

            mock_usage = Mock()
            mock_usage.prompt_token_count = total_prompt_tokens
            mock_usage.candidates_token_count = total_completion_tokens
            mock_usage.total_token_count = total_prompt_tokens + total_completion_tokens
            mock_usage.cached_content_token_count = 0
            mock_chunk.usage_metadata = mock_usage
        else:
            # Intermediate chunks
            mock_candidate.finish_reason = None
            mock_chunk.usage_metadata = None

        mock_candidate.content = Mock()
        mock_candidate.content.parts = [Mock(text=f"chunk_{i}")]
        mock_chunk.candidates = [mock_candidate]

        chunks.append(mock_chunk)

    return chunks


def generate_test_id(prefix: str = "test") -> str:
    """
    Generate a unique test ID.

    Args:
        prefix: Prefix for the test ID

    Returns:
        Unique test ID string
    """
    timestamp = int(time.time() * 1000)  # milliseconds
    return f"{prefix}-{timestamp}"


def create_nested_metadata(
    trace_id: Optional[str] = None,
    subscriber_id: str = "test-user",
    subscriber_email: Optional[str] = "test@example.com",
    **kwargs
) -> Dict[str, Any]:
    """
    Create nested metadata format.

    Args:
        trace_id: Trace ID (auto-generated if None)
        subscriber_id: Subscriber ID
        subscriber_email: Subscriber email
        **kwargs: Additional metadata fields

    Returns:
        Dictionary with nested metadata
    """
    if trace_id is None:
        trace_id = generate_test_id("trace")

    metadata = {
        "trace_id": trace_id,
        "subscriber": {
            "id": subscriber_id,
        }
    }

    if subscriber_email:
        metadata["subscriber"]["email"] = subscriber_email

    # Add any additional fields
    metadata.update(kwargs)

    return metadata


def create_flat_metadata(
    trace_id: Optional[str] = None,
    subscriber_id: str = "test-user",
    subscriber_email: Optional[str] = "test@example.com",
    **kwargs
) -> Dict[str, Any]:
    """
    Create flat (deprecated) metadata format.

    Args:
        trace_id: Trace ID (auto-generated if None)
        subscriber_id: Subscriber ID
        subscriber_email: Subscriber email
        **kwargs: Additional metadata fields

    Returns:
        Dictionary with flat metadata
    """
    if trace_id is None:
        trace_id = generate_test_id("trace")

    metadata = {
        "trace_id": trace_id,
        "subscriber_id": subscriber_id,
    }

    if subscriber_email:
        metadata["subscriber_email"] = subscriber_email

    # Add any additional fields
    metadata.update(kwargs)

    return metadata


def assert_subscriber_transformed(result: Dict[str, Any], expected_id: str, expected_email: Optional[str] = None):
    """
    Assert that subscriber data was transformed correctly.

    Args:
        result: Transformed metadata dictionary
        expected_id: Expected subscriber ID
        expected_email: Expected subscriber email (optional)
    """
    assert "subscriber" in result, "Subscriber object missing"
    assert isinstance(result["subscriber"], dict), "Subscriber should be a dict"
    assert result["subscriber"]["id"] == expected_id, f"Expected subscriber.id={expected_id}"

    if expected_email:
        assert result["subscriber"]["email"] == expected_email, f"Expected subscriber.email={expected_email}"


def assert_valid_token_counts(usage_metadata: Any):
    """
    Assert that token counts are valid.

    Args:
        usage_metadata: Usage metadata object with token counts
    """
    prompt_tokens = getattr(usage_metadata, "prompt_token_count", 0)
    completion_tokens = getattr(usage_metadata, "candidates_token_count", 0)
    total_tokens = getattr(usage_metadata, "total_token_count", 0)

    assert prompt_tokens >= 0, "Prompt tokens should be non-negative"
    assert completion_tokens >= 0, "Completion tokens should be non-negative"
    assert total_tokens >= 0, "Total tokens should be non-negative"
    assert total_tokens >= prompt_tokens, "Total should be >= prompt tokens"
    assert total_tokens >= completion_tokens, "Total should be >= completion tokens"


def assert_valid_revenium_payload(payload: Dict[str, Any]):
    """
    Assert that Revenium API payload is valid.

    Args:
        payload: Payload dictionary to validate
    """
    # Required fields
    assert "trace_id" in payload, "trace_id is required"
    assert "provider" in payload, "provider is required"
    assert "model_name" in payload, "model_name is required"
    assert "input_tokens" in payload, "input_tokens is required"
    assert "output_tokens" in payload, "output_tokens is required"

    # Token counts should be non-negative
    assert payload["input_tokens"] >= 0, "input_tokens should be non-negative"
    assert payload["output_tokens"] >= 0, "output_tokens should be non-negative"

    # Subscriber should be nested
    if "subscriber" in payload:
        assert isinstance(payload["subscriber"], dict), "subscriber should be a dict"
        assert "id" in payload["subscriber"], "subscriber.id is required"


def mock_requests_post_success():
    """
    Create a mock for successful requests.post call.

    Returns:
        Mock response with status 200
    """
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.ok = True
    mock_response.json.return_value = {"status": "success"}
    return mock_response


def mock_requests_post_error(status_code: int = 500):
    """
    Create a mock for failed requests.post call.

    Args:
        status_code: HTTP status code

    Returns:
        Mock response with error status
    """
    mock_response = Mock()
    mock_response.status_code = status_code
    mock_response.ok = False
    mock_response.json.return_value = {"status": "error", "message": "API error"}
    return mock_response


def create_batch_metadata(
    batch_id: str,
    batch_size: int,
    item_index: int,
    base_subscriber_id: str = "batch-user"
) -> Dict[str, Any]:
    """
    Create metadata for a batch processing item.

    Args:
        batch_id: Batch identifier
        batch_size: Total items in batch
        item_index: Index of this item (0-based)
        base_subscriber_id: Base subscriber ID

    Returns:
        Dictionary with batch metadata
    """
    return {
        "trace_id": f"{batch_id}-item-{item_index}",
        "batch_id": batch_id,
        "batch_item_number": item_index,
        "batch_total": batch_size,
        "subscriber": {
            "id": base_subscriber_id,
        }
    }


def create_conversation_metadata(
    conversation_id: str,
    turn_number: int,
    session_id: Optional[str] = None,
    subscriber_id: str = "conv-user"
) -> Dict[str, Any]:
    """
    Create metadata for a conversation turn.

    Args:
        conversation_id: Conversation identifier
        turn_number: Turn number (1-based)
        session_id: Optional session ID
        subscriber_id: Subscriber ID

    Returns:
        Dictionary with conversation metadata
    """
    metadata = {
        "trace_id": f"{conversation_id}-turn-{turn_number}",
        "conversation_id": conversation_id,
        "turn_number": turn_number,
        "subscriber": {
            "id": subscriber_id,
        }
    }

    if session_id:
        metadata["session_id"] = session_id

    return metadata


def compare_metadata_dicts(actual: Dict[str, Any], expected: Dict[str, Any], ignore_keys: Optional[List[str]] = None):
    """
    Compare two metadata dictionaries, optionally ignoring certain keys.

    Args:
        actual: Actual metadata dictionary
        expected: Expected metadata dictionary
        ignore_keys: List of keys to ignore in comparison

    Raises:
        AssertionError: If dictionaries don't match
    """
    if ignore_keys is None:
        ignore_keys = []

    # Filter out ignored keys
    actual_filtered = {k: v for k, v in actual.items() if k not in ignore_keys}
    expected_filtered = {k: v for k, v in expected.items() if k not in ignore_keys}

    assert actual_filtered == expected_filtered, (
        f"Metadata mismatch:\n"
        f"Expected: {expected_filtered}\n"
        f"Actual: {actual_filtered}"
    )


def extract_finish_reason(response: Any) -> Optional[str]:
    """
    Extract finish reason from a response.

    Args:
        response: Response object (mock or real)

    Returns:
        Finish reason string or None
    """
    try:
        if hasattr(response, "candidates") and response.candidates:
            return response.candidates[0].finish_reason
    except (AttributeError, IndexError):
        pass

    return None


def extract_token_counts(response: Any) -> Dict[str, int]:
    """
    Extract token counts from a response.

    Args:
        response: Response object (mock or real)

    Returns:
        Dictionary with token counts
    """
    counts = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "cached_tokens": 0,
    }

    try:
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            usage = response.usage_metadata
            counts["prompt_tokens"] = getattr(usage, "prompt_token_count", 0)
            counts["completion_tokens"] = getattr(usage, "candidates_token_count", 0)
            counts["total_tokens"] = getattr(usage, "total_token_count", 0)
            counts["cached_tokens"] = getattr(usage, "cached_content_token_count", 0)
    except AttributeError:
        pass

    return counts
