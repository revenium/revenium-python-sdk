"""
Custom exception classes for Revenium Google AI middleware.

This module defines a hierarchy of exceptions for different error scenarios
that can occur during middleware operation.
"""

from typing import Optional, Any


class ReveniumMiddlewareError(Exception):
    """Base exception for all Revenium middleware errors."""

    def __init__(self, message: str, details: Optional[dict] = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def __str__(self) -> str:
        if self.details:
            return f"{self.message} (Details: {self.details})"
        return self.message


class MeteringError(ReveniumMiddlewareError):
    """Raised when metering operations fail."""

    def __init__(
        self,
        message: str,
        transaction_id: Optional[str] = None,
        api_response: Optional[Any] = None,
        **kwargs,
    ):
        super().__init__(message, kwargs)
        self.transaction_id = transaction_id
        self.api_response = api_response


class TokenExtractionError(ReveniumMiddlewareError):
    """Raised when token count extraction fails."""

    def __init__(
        self,
        message: str,
        response_type: Optional[str] = None,
        operation_type: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(message, kwargs)
        self.response_type = response_type
        self.operation_type = operation_type


class ProviderDetectionError(ReveniumMiddlewareError):
    """Raised when provider detection fails."""

    def __init__(self, message: str, available_sdks: Optional[list] = None, **kwargs):
        super().__init__(message, kwargs)
        self.available_sdks = available_sdks or []


class ConfigurationError(ReveniumMiddlewareError):
    """Raised when configuration is invalid or missing."""

    def __init__(self, message: str, missing_config: Optional[list] = None, **kwargs):
        super().__init__(message, kwargs)
        self.missing_config = missing_config or []


class StreamingError(ReveniumMiddlewareError):
    """Raised when streaming operations fail."""

    def __init__(
        self,
        message: str,
        chunk_count: Optional[int] = None,
        stream_state: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(message, kwargs)
        self.chunk_count = chunk_count
        self.stream_state = stream_state


class APIResponseError(ReveniumMiddlewareError):
    """Raised when API response is malformed or unexpected."""

    def __init__(
        self,
        message: str,
        response_data: Optional[Any] = None,
        expected_format: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(message, kwargs)
        self.response_data = response_data
        self.expected_format = expected_format


# Utility functions for error handling
def handle_metering_error(func):
    """Decorator to handle metering errors gracefully."""

    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except MeteringError:
            # Re-raise metering errors as-is
            raise
        except Exception as e:
            # Convert other exceptions to MeteringError
            raise MeteringError(
                f"Unexpected error in {func.__name__}: {str(e)}",
                details={"original_error": type(e).__name__},
            ) from e

    return wrapper


def safe_extract(func):
    """Decorator to handle extraction errors gracefully."""

    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except (TokenExtractionError, APIResponseError):
            # Re-raise extraction errors as-is
            raise
        except Exception as e:
            # Convert other exceptions to TokenExtractionError
            raise TokenExtractionError(
                f"Unexpected error in {func.__name__}: {str(e)}",
                details={"original_error": type(e).__name__},
            ) from e

    return wrapper
