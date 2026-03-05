"""
Custom exceptions for Revenium middleware.

This module defines a hierarchy of exceptions that provide better error handling
and more specific error information for different failure scenarios.
"""


class ReveniumMiddlewareError(Exception):
    """Base exception for all Revenium middleware errors."""

    def __init__(self, message: str, original_error: Exception = None):
        super().__init__(message)
        self.original_error = original_error
        self.message = message

    def __str__(self):
        if self.original_error:
            return f"{self.message} (caused by: {self.original_error})"
        return self.message


class ConfigurationError(ReveniumMiddlewareError):
    """Raised when there are configuration-related issues."""
    pass


class ValidationError(ReveniumMiddlewareError):
    """Raised when input validation fails."""
    pass


class ProviderDetectionError(ReveniumMiddlewareError):
    """Raised when provider detection fails."""
    pass


class MeteringError(ReveniumMiddlewareError):
    """Raised when metering operations fail."""
    pass


class NetworkError(ReveniumMiddlewareError):
    """Raised when network operations fail."""
    pass


class AuthenticationError(ReveniumMiddlewareError):
    """Raised when authentication fails."""
    pass


class StreamingError(ReveniumMiddlewareError):
    """Raised when streaming operations fail."""
    pass


class ModelResolutionError(ReveniumMiddlewareError):
    """Raised when Azure model name resolution fails."""
    pass


def handle_exception_safely(func):
    """
    Decorator to handle exceptions safely without breaking the main application flow.

    This decorator ensures that middleware errors never propagate to break
    the user's application, following the principle of graceful degradation.
    """
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except ReveniumMiddlewareError as e:
            # Log middleware-specific errors
            import logging
            logger = logging.getLogger("revenium_middleware.extension")
            logger.error(f"Revenium middleware error in {func.__name__}: {e}")
            return None
        except Exception as e:
            # Log unexpected errors
            import logging
            logger = logging.getLogger("revenium_middleware.extension")
            logger.error(f"Unexpected error in {func.__name__}: {e}", exc_info=True)
            return None

    return wrapper


def categorize_exception(exception: Exception) -> ReveniumMiddlewareError:
    """
    Categorize a generic exception into a more specific Revenium middleware exception.

    Args:
        exception: The original exception to categorize

    Returns:
        A more specific ReveniumMiddlewareError subclass
    """
    error_message = str(exception)
    error_type = type(exception).__name__

    # Network-related errors
    if any(keyword in error_message.lower() for keyword in ['connection', 'timeout', 'network', 'dns']):
        return NetworkError(f"Network error: {error_message}", exception)

    # Authentication errors
    if any(keyword in error_message.lower() for keyword in ['auth', 'unauthorized', 'forbidden', 'api key']):
        return AuthenticationError(f"Authentication error: {error_message}", exception)

    # Validation errors
    if any(keyword in error_type.lower() for keyword in ['value', 'type', 'attribute']):
        return ValidationError(f"Validation error: {error_message}", exception)

    # Default to generic middleware error
    return ReveniumMiddlewareError(f"Middleware error ({error_type}): {error_message}", exception)
