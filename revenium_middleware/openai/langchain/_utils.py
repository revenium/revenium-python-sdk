"""
Utility functions for LangChain integration.

This module provides helper functions for dependency checking, error handling,
and other common functionality used across the LangChain integration.
"""

import functools
import logging
from typing import Any, Callable, TypeVar, Optional, Tuple

logger = logging.getLogger("revenium_middleware.langchain")

F = TypeVar('F', bound=Callable[..., Any])

# Global cache for LangChain availability check
_langchain_check_cache: Optional[Tuple[bool, Optional[Exception]]] = None


class LangChainNotInstalledError(ImportError):
    """
    Custom exception raised when LangChain functionality is used without LangChain installed.

    This provides a clear, actionable error message for users.
    """

    def __init__(self, message: str = None, feature: str = None):
        if message is None:
            feature_msg = f" for {feature}" if feature else ""
            message = (
                f"LangChain is required{feature_msg} but is not installed.\n"
                "Install it with one of the following commands:\n"
                "  pip install revenium-python-sdk[openai,langchain]\n"
                "  pip install langchain>=0.1.16,<1.0"
            )
        super().__init__(message)


def _check_langchain_cached() -> Tuple[bool, Optional[Exception]]:
    """
    Check if LangChain is available with caching to avoid repeated imports.

    Returns:
        Tuple of (is_available, import_error_if_any)
    """
    global _langchain_check_cache

    if _langchain_check_cache is not None:
        return _langchain_check_cache

    try:
        import langchain  # noqa: F401
        _langchain_check_cache = (True, None)
        logger.debug("LangChain is available")
    except ImportError as e:
        _langchain_check_cache = (False, e)
        logger.debug(f"LangChain is not available: {e}")

    return _langchain_check_cache


def requires_langchain(feature: str = None) -> Callable[[F], F]:
    """
    Decorator that ensures LangChain is available before calling the decorated function.

    Args:
        feature: Optional feature name for better error messages

    Returns:
        Decorator function

    Raises:
        LangChainNotInstalledError: If LangChain is not installed
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            is_available, import_error = _check_langchain_cached()
            if not is_available:
                raise LangChainNotInstalledError(feature=feature) from import_error

            return func(*args, **kwargs)

        return wrapper
    return decorator


def check_langchain_version() -> str:
    """
    Check the installed LangChain version and return it.

    Returns:
        LangChain version string

    Raises:
        LangChainNotInstalledError: If LangChain is not installed
    """
    is_available, import_error = _check_langchain_cached()
    if not is_available:
        raise LangChainNotInstalledError(feature="version checking") from import_error

    import langchain  # noqa: F401
    return getattr(langchain, '__version__', 'unknown')


def is_langchain_available() -> bool:
    """
    Check if LangChain is available without raising an exception.

    Returns:
        True if LangChain is available, False otherwise
    """
    is_available, _ = _check_langchain_cached()
    return is_available


def require_langchain_or_raise(feature: str = None) -> None:
    """
    Ensure LangChain is available, raising a helpful error if not.

    Args:
        feature: Optional feature name for better error messages

    Raises:
        LangChainNotInstalledError: If LangChain is not installed
    """
    is_available, import_error = _check_langchain_cached()
    if not is_available:
        raise LangChainNotInstalledError(feature=feature) from import_error
