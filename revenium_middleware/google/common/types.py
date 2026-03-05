"""
Shared type definitions for Google AI middleware.

This module contains common data structures and enums used across both
Google AI SDK and Vertex AI SDK middleware implementations.
"""

import datetime
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Any, Optional


class OperationType(str, Enum):
    """Operation types for AI API calls."""

    CHAT = "CHAT"
    EMBED = "EMBED"
    IMAGE = "IMAGE"
    VIDEO = "VIDEO"


class Provider(Enum):
    """Supported Google AI providers."""

    GOOGLE_AI_SDK = "google_ai_sdk"
    VERTEX_AI_SDK = "vertex_ai_sdk"


@dataclass
class ProviderMetadata:
    """Provider-specific metadata for usage records."""

    provider: str  # Always "Google" for unified reporting
    model_source: str  # Always "GOOGLE"
    sdk_type: Provider  # Which SDK is being used

    @classmethod
    def for_google_ai_sdk(cls) -> "ProviderMetadata":
        """Create metadata for Google AI SDK."""
        return cls(
            provider="Google", model_source="GOOGLE", sdk_type=Provider.GOOGLE_AI_SDK
        )

    @classmethod
    def for_vertex_ai_sdk(cls) -> "ProviderMetadata":
        """Create metadata for Vertex AI SDK."""
        return cls(
            provider="Google", model_source="GOOGLE", sdk_type=Provider.VERTEX_AI_SDK
        )


@dataclass
class UsageData:
    """Standardized usage data structure for both SDKs."""

    # Core token counts (required)
    input_token_count: int
    output_token_count: int
    total_token_count: int

    # Operation details (required)
    operation_type: str  # OperationType value
    stop_reason: str
    transaction_id: str
    model: str

    # Provider information (required)
    provider: str  # Always "Google"
    model_source: str  # Always "GOOGLE"
    sdk_type: Provider  # Which SDK was used

    # Timing information (required)
    request_time: str
    response_time: str
    completion_start_time: str
    request_duration: int  # milliseconds

    # Streaming and timing (optional with defaults)
    is_streamed: bool = False
    time_to_first_token: int = 0

    # Advanced token counts (optional with defaults)
    cache_creation_token_count: int = 0
    cache_read_token_count: int = 0
    reasoning_token_count: int = 0

    # Cost information (optional with defaults)
    cost_type: str = "AI"
    input_token_cost: Optional[float] = None
    output_token_cost: Optional[float] = None
    total_cost: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API calls."""
        return {
            "input_token_count": self.input_token_count,
            "output_token_count": self.output_token_count,
            "total_token_count": self.total_token_count,
            "operation_type": self.operation_type,
            "stop_reason": self.stop_reason,
            "transaction_id": self.transaction_id,
            "model": self.model,
            "provider": self.provider,
            "model_source": self.model_source,
            "is_streamed": self.is_streamed,
            "time_to_first_token": self.time_to_first_token,
            "cache_creation_token_count": self.cache_creation_token_count,
            "cache_read_token_count": self.cache_read_token_count,
            "reasoning_token_count": self.reasoning_token_count,
            "request_time": self.request_time,
            "response_time": self.response_time,
            "completion_start_time": self.completion_start_time,
            "request_duration": self.request_duration,
            "cost_type": self.cost_type,
            "input_token_cost": self.input_token_cost,
            "output_token_cost": self.output_token_cost,
            "total_cost": self.total_cost,
        }

    @classmethod
    def create(
        cls,
        operation_type: OperationType,
        input_tokens: int,
        output_tokens: int,
        total_tokens: int,
        model: str,
        provider_metadata: ProviderMetadata,
        stop_reason: str,
        request_time: datetime.datetime,
        response_time: datetime.datetime,
        transaction_id: Optional[str] = None,
        **kwargs,
    ) -> "UsageData":
        """Create UsageData with common defaults."""
        import uuid

        if transaction_id is None:
            transaction_id = str(uuid.uuid4())

        request_time_str = request_time.strftime("%Y-%m-%dT%H:%M:%SZ")
        response_time_str = response_time.strftime("%Y-%m-%dT%H:%M:%SZ")
        request_duration = int((response_time - request_time).total_seconds() * 1000)

        return cls(
            input_token_count=input_tokens,
            output_token_count=output_tokens,
            total_token_count=total_tokens,
            operation_type=operation_type.value,
            stop_reason=stop_reason,
            transaction_id=transaction_id,
            model=model,
            provider=provider_metadata.provider,
            model_source=provider_metadata.model_source,
            sdk_type=provider_metadata.sdk_type,
            request_time=request_time_str,
            response_time=response_time_str,
            completion_start_time=response_time_str,
            request_duration=request_duration,
            **kwargs,
        )


@dataclass
class TokenCounts:
    """Token count information extracted from API responses."""

    input_tokens: int
    output_tokens: int
    total_tokens: int
    cached_tokens: int = 0

    @property
    def has_counts(self) -> bool:
        """Check if any token counts are available."""
        return self.total_tokens > 0 or self.input_tokens > 0 or self.output_tokens > 0


# Stop reason mappings for different SDKs
# Valid Revenium stop reasons: "END", "END_SEQUENCE", "TIMEOUT", "TOKEN_LIMIT", "ERROR"
GOOGLE_AI_STOP_REASONS = {
    "STOP": "END",
    "MAX_TOKENS": "TOKEN_LIMIT",
    "SAFETY": "ERROR",
    "RECITATION": "ERROR",
    "OTHER": "END",  # Default to END for unknown reasons
    None: "END",
}

VERTEX_AI_STOP_REASONS = {
    "STOP": "END",
    "MAX_TOKENS": "TOKEN_LIMIT",
    "SAFETY": "ERROR",
    "RECITATION": "ERROR",
    "FINISH_REASON_UNSPECIFIED": "END",  # Default to END for unknown reasons
    None: "END",
}


def normalize_stop_reason(finish_reason: Optional[str], sdk_type: Provider) -> str:
    """Normalize stop reasons across different SDKs."""
    if sdk_type == Provider.GOOGLE_AI_SDK:
        return GOOGLE_AI_STOP_REASONS.get(finish_reason, "END")
    elif sdk_type == Provider.VERTEX_AI_SDK:
        return VERTEX_AI_STOP_REASONS.get(finish_reason, "END")
    else:
        return "END"
