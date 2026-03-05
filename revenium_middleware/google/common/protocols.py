"""
Protocol definitions for Google AI API responses.

This module defines Protocol classes that describe the expected structure
of API responses from Google AI and Vertex AI SDKs, improving type safety
and enabling better static analysis.
"""

from typing import Protocol, Optional, List, Any, Union
from typing_extensions import runtime_checkable


@runtime_checkable
class UsageMetadataProtocol(Protocol):
    """Protocol for usage metadata in API responses."""

    # Google AI SDK attributes
    prompt_token_count: Optional[int]
    candidates_token_count: Optional[int]
    total_token_count: Optional[int]
    cached_content_token_count: Optional[int]

    # Alternative attribute names (for compatibility)
    prompt_tokens: Optional[int]
    completion_tokens: Optional[int]
    total_tokens: Optional[int]
    cached_tokens: Optional[int]
    input_tokens: Optional[int]
    output_tokens: Optional[int]


@runtime_checkable
class CandidateProtocol(Protocol):
    """Protocol for candidate objects in API responses."""

    finish_reason: Optional[str]
    content: Optional[Any]
    text: Optional[str]


@runtime_checkable
class EmbeddingProtocol(Protocol):
    """Protocol for embedding objects."""

    values: List[float]
    statistics: Optional[Any]
    _prediction_response: Optional[Any]


@runtime_checkable
class EmbeddingStatisticsProtocol(Protocol):
    """Protocol for embedding statistics."""

    token_count: Optional[int]
    billableCharacterCount: Optional[int]


@runtime_checkable
class ChatResponseProtocol(Protocol):
    """Protocol for chat/generation API responses."""

    # Model information
    model: Optional[str]
    model_name: Optional[str]
    model_version: Optional[str]
    _model_name: Optional[str]

    # Usage information
    usage: Optional[UsageMetadataProtocol]
    usage_metadata: Optional[UsageMetadataProtocol]

    # Response content
    text: Optional[str]
    content: Optional[str]
    candidates: Optional[List[CandidateProtocol]]

    # Completion information
    finish_reason: Optional[str]
    stop_reason: Optional[str]


@runtime_checkable
class EmbeddingResponseProtocol(Protocol):
    """Protocol for embedding API responses."""

    # Model information
    model: Optional[str]
    model_name: Optional[str]

    # Embeddings data
    embeddings: Optional[List[EmbeddingProtocol]]
    values: Optional[List[float]]  # For single embedding responses

    # Usage information
    usage: Optional[UsageMetadataProtocol]
    usage_metadata: Optional[UsageMetadataProtocol]
    statistics: Optional[EmbeddingStatisticsProtocol]


@runtime_checkable
class StreamChunkProtocol(Protocol):
    """Protocol for streaming response chunks."""

    # Model information
    model: Optional[str]
    model_version: Optional[str]

    # Content
    text: Optional[str]
    content: Optional[str]
    candidates: Optional[List[CandidateProtocol]]

    # Usage information (typically in final chunk)
    usage_metadata: Optional[UsageMetadataProtocol]

    # Completion information
    finish_reason: Optional[str]


@runtime_checkable
class ClientProtocol(Protocol):
    """Protocol for Google AI client objects."""

    _vertexai: Optional[bool]
    _api_client: Optional[Any]


# Type guards for runtime type checking
def is_chat_response(response: Any) -> bool:
    """Check if response matches ChatResponseProtocol."""
    return isinstance(response, ChatResponseProtocol)


def is_embedding_response(response: Any) -> bool:
    """Check if response matches EmbeddingResponseProtocol."""
    return isinstance(response, EmbeddingResponseProtocol)


def is_stream_chunk(chunk: Any) -> bool:
    """Check if chunk matches StreamChunkProtocol."""
    return isinstance(chunk, StreamChunkProtocol)


def has_usage_metadata(response: Any) -> bool:
    """Check if response has usage metadata."""
    return (
        hasattr(response, "usage_metadata") and response.usage_metadata is not None
    ) or (hasattr(response, "usage") and response.usage is not None)


def has_token_counts(usage: Any) -> bool:
    """Check if usage object has token count information."""
    if not usage:
        return False

    # Check for any token count attributes
    token_attrs = [
        "total_token_count",
        "total_tokens",
        "prompt_token_count",
        "prompt_tokens",
        "input_tokens",
        "candidates_token_count",
        "completion_tokens",
        "output_tokens",
    ]

    return any(
        hasattr(usage, attr) and getattr(usage, attr, 0) > 0 for attr in token_attrs
    )


# Utility functions for safe attribute access
def safe_getattr(obj: Any, attr: str, default: Any = None) -> Any:
    """Safely get attribute with fallback to default."""
    try:
        return getattr(obj, attr, default)
    except (AttributeError, TypeError):
        return default


def get_token_count(usage: Any, attr_names: List[str]) -> int:
    """Get token count from usage object, trying multiple attribute names."""
    if not usage:
        return 0

    for attr_name in attr_names:
        value = safe_getattr(usage, attr_name, 0)
        if isinstance(value, int) and value > 0:
            return value

    return 0
