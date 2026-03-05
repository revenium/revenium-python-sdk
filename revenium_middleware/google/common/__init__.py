"""
Common utilities and shared types for Google AI middleware.

This module contains shared functionality used by both Google AI SDK
and Vertex AI SDK middleware implementations.
"""

from .types import (
    OperationType,
    Provider,
    ProviderMetadata,
    UsageData,
    TokenCounts,
    normalize_stop_reason,
    GOOGLE_AI_STOP_REASONS,
    VERTEX_AI_STOP_REASONS,
)

from .utils import (
    generate_transaction_id,
    format_timestamp,
    calculate_duration_ms,
    log_token_usage,
    log_image_usage,
    log_video_usage,
    create_metering_call,
    create_image_metering_call,
    create_video_metering_call,
    extract_model_name,
    extract_token_counts,
    create_usage_data,
    is_debug_logging_enabled,
)

from .exceptions import (
    ReveniumMiddlewareError,
    MeteringError,
    TokenExtractionError,
    ProviderDetectionError,
    ConfigurationError,
    StreamingError,
    APIResponseError,
    handle_metering_error,
    safe_extract,
)

from .protocols import (
    UsageMetadataProtocol,
    CandidateProtocol,
    EmbeddingProtocol,
    ChatResponseProtocol,
    EmbeddingResponseProtocol,
    StreamChunkProtocol,
    ClientProtocol,
    is_chat_response,
    is_embedding_response,
    is_stream_chunk,
    has_usage_metadata,
    has_token_counts,
    safe_getattr,
    get_token_count,
)

from .summary_printer import (
    CompletionMetrics,
    fetch_completion_metrics,
    format_and_print_json_summary,
    format_and_print_human_summary,
    print_usage_summary,
)

__all__ = [
    # Types
    "OperationType",
    "Provider",
    "ProviderMetadata",
    "UsageData",
    "TokenCounts",
    "normalize_stop_reason",
    "GOOGLE_AI_STOP_REASONS",
    "VERTEX_AI_STOP_REASONS",
    # Utils
    "generate_transaction_id",
    "format_timestamp",
    "calculate_duration_ms",
    "log_token_usage",
    "log_image_usage",
    "log_video_usage",
    "create_metering_call",
    "create_image_metering_call",
    "create_video_metering_call",
    "extract_model_name",
    "extract_token_counts",
    "create_usage_data",
    "is_debug_logging_enabled",
    # Exceptions
    "ReveniumMiddlewareError",
    "MeteringError",
    "TokenExtractionError",
    "ProviderDetectionError",
    "ConfigurationError",
    "StreamingError",
    "APIResponseError",
    "handle_metering_error",
    "safe_extract",
    # Protocols
    "UsageMetadataProtocol",
    "CandidateProtocol",
    "EmbeddingProtocol",
    "ChatResponseProtocol",
    "EmbeddingResponseProtocol",
    "StreamChunkProtocol",
    "ClientProtocol",
    "is_chat_response",
    "is_embedding_response",
    "is_stream_chunk",
    "has_usage_metadata",
    "has_token_counts",
    "safe_getattr",
    "get_token_count",
    # Summary Printer
    "CompletionMetrics",
    "fetch_completion_metrics",
    "format_and_print_json_summary",
    "format_and_print_human_summary",
    "print_usage_summary",
]
