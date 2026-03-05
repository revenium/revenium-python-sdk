"""
Context-based metadata injection for Revenium LiteLLM middleware.

This module provides thread-safe context management for injecting metadata into
LiteLLM completion calls without requiring explicit kwargs. It uses Python's
contextvars module to maintain isolated context per execution thread/task.

Example:
    >>> from revenium_middleware.litellm.client import metadata_context
    >>> 
    >>> # Using context manager
    >>> with metadata_context.set(agent="Lead Analyst", task_type="research"):
    ...     response = litellm.completion(model="gpt-4", messages=[...])
    ...     # Metadata automatically injected
    >>> 
    >>> # Using direct API
    >>> metadata_context.update(trace_id="abc-123")
    >>> response = litellm.completion(model="gpt-4", messages=[...])
    >>> metadata_context.clear()
"""

import contextvars
from typing import Dict, Any, Optional


# Thread-safe context variable for storing metadata
_metadata_context: contextvars.ContextVar[Dict[str, Any]] = contextvars.ContextVar(
    'revenium_metadata', 
    default={}
)


class MetadataContext:
    """
    Thread-safe metadata context manager for Revenium middleware.
    
    This class provides methods to set, get, update, and clear metadata that
    will be automatically injected into LiteLLM completion calls. It uses
    Python's contextvars module to ensure thread-safety and proper isolation
    in async contexts.
    
    All metadata set through this API will be merged with explicit usage_metadata
    kwargs, with explicit kwargs taking precedence.
    """
    
    @staticmethod
    def set(**kwargs) -> '_MetadataContextManager':
        """
        Set metadata for the current context using a context manager.
        
        This method returns a context manager that sets the metadata on entry
        and restores the previous metadata on exit. This is the recommended
        way to set metadata for a specific scope.
        
        Args:
            **kwargs: Metadata fields to set (e.g., agent="Lead Analyst",
                     task_type="research", trace_id="abc-123")
        
        Returns:
            _MetadataContextManager: Context manager for scoped metadata
        
        Example:
            >>> with metadata_context.set(agent="Analyst", task_type="research"):
            ...     # All LiteLLM calls here get this metadata
            ...     response = litellm.completion(...)
        """
        return _MetadataContextManager(kwargs)
    
    @staticmethod
    def get() -> Dict[str, Any]:
        """
        Get the current context metadata.
        
        Returns a copy of the current metadata dictionary. Modifying the
        returned dictionary will not affect the context.
        
        Returns:
            Dict[str, Any]: Copy of current metadata
        
        Example:
            >>> metadata_context.update(agent="Analyst")
            >>> current = metadata_context.get()
            >>> print(current)
            {'agent': 'Analyst'}
        """
        return _metadata_context.get().copy()
    
    @staticmethod
    def update(**kwargs) -> None:
        """
        Update the current context metadata.
        
        This merges the provided metadata with existing metadata in the
        current context. Unlike set(), this does not use a context manager
        and the metadata persists until explicitly cleared.
        
        Args:
            **kwargs: Metadata fields to update
        
        Example:
            >>> metadata_context.update(agent="Analyst")
            >>> metadata_context.update(task_type="research")
            >>> # Both fields are now set
            >>> metadata_context.clear()  # Clean up when done
        """
        current = _metadata_context.get().copy()
        current.update(kwargs)
        _metadata_context.set(current)
    
    @staticmethod
    def clear(*keys: str) -> None:
        """
        Clear specific metadata fields or all metadata.
        
        If no keys are provided, clears all metadata. If keys are provided,
        only those specific fields are removed.
        
        Args:
            *keys: Optional field names to clear. If not provided, clears all.
        
        Example:
            >>> metadata_context.update(agent="A", task_type="T", trace_id="123")
            >>> metadata_context.clear("agent")  # Remove only agent
            >>> metadata_context.clear()  # Remove all remaining
        """
        if not keys:
            # Clear all metadata
            _metadata_context.set({})
        else:
            # Clear specific keys
            current = _metadata_context.get().copy()
            for key in keys:
                current.pop(key, None)
            _metadata_context.set(current)


class _MetadataContextManager:
    """
    Context manager for scoped metadata injection.
    
    This class implements the context manager protocol to provide scoped
    metadata that is automatically cleaned up on exit. It saves the previous
    metadata state on entry and restores it on exit.
    
    This class should not be instantiated directly. Use MetadataContext.set()
    instead.
    """
    
    def __init__(self, metadata: Dict[str, Any]):
        """
        Initialize the context manager.
        
        Args:
            metadata: Metadata dictionary to set in this context
        """
        self.metadata = metadata
        self.token: Optional[contextvars.Token] = None
    
    def __enter__(self) -> '_MetadataContextManager':
        """
        Enter the context and set metadata.
        
        Saves the current metadata state and sets the new metadata.
        
        Returns:
            Self for use in with statements
        """
        # Merge new metadata with existing context
        current = _metadata_context.get().copy()
        current.update(self.metadata)
        self.token = _metadata_context.set(current)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        """
        Exit the context and restore previous metadata.
        
        Restores the metadata state that existed before entering this context.
        
        Args:
            exc_type: Exception type if an exception occurred
            exc_val: Exception value if an exception occurred
            exc_tb: Exception traceback if an exception occurred
        
        Returns:
            False to propagate any exception that occurred
        """
        if self.token is not None:
            _metadata_context.reset(self.token)
        return False


# Global instance for convenient access
metadata_context = MetadataContext()


__all__ = ['MetadataContext', 'metadata_context']

