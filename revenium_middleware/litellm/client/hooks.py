"""
Callback/hook system for extending Revenium LiteLLM middleware.

This module provides a hook system that allows users to register callbacks
that are executed before metadata is sent to Revenium. Hooks can modify,
enrich, or validate metadata.

Example:
    >>> from revenium_middleware.litellm.client import register_metadata_hook
    >>> 
    >>> def add_environment(metadata):
    ...     metadata['environment'] = 'production'
    ...     return metadata
    >>> 
    >>> register_metadata_hook(add_environment)
    >>> 
    >>> # Now all LiteLLM calls will include environment field
    >>> response = litellm.completion(...)
"""

import logging
from typing import Callable, Dict, Any, List, Optional

logger = logging.getLogger("revenium_middleware.hooks")

# Global registry of metadata hooks
_metadata_hooks: List[Callable[[Dict[str, Any]], Dict[str, Any]]] = []


def register_metadata_hook(
    hook: Callable[[Dict[str, Any]], Dict[str, Any]],
    priority: int = 0
) -> None:
    """
    Register a metadata hook to be called before sending data to Revenium.
    
    Hooks are called in order of priority (higher priority first), then
    in registration order for hooks with the same priority.
    
    Each hook receives the metadata dictionary and should return a modified
    (or the same) dictionary. Hooks can:
    - Add new fields
    - Modify existing fields
    - Remove fields (by returning a dict without them)
    - Validate metadata (raise exceptions to prevent sending)
    
    Args:
        hook: Callable that takes metadata dict and returns metadata dict
        priority: Priority for hook execution (higher = earlier). Default: 0
    
    Example:
        >>> def add_version(metadata):
        ...     metadata['app_version'] = '1.2.3'
        ...     return metadata
        >>> 
        >>> register_metadata_hook(add_version, priority=10)
        >>> 
        >>> def validate_required_fields(metadata):
        ...     if 'organization_id' not in metadata:
        ...         raise ValueError("organization_id is required")
        ...     return metadata
        >>> 
        >>> register_metadata_hook(validate_required_fields, priority=100)
    """
    # Add hook with priority
    _metadata_hooks.append((priority, hook))
    
    # Sort by priority (descending) to maintain execution order
    _metadata_hooks.sort(key=lambda x: x[0], reverse=True)
    
    logger.debug(f"Registered metadata hook: {hook.__name__} with priority {priority}")


def unregister_metadata_hook(hook: Callable[[Dict[str, Any]], Dict[str, Any]]) -> bool:
    """
    Unregister a previously registered metadata hook.
    
    Args:
        hook: The hook function to unregister
    
    Returns:
        True if hook was found and removed, False otherwise
    
    Example:
        >>> def my_hook(metadata):
        ...     return metadata
        >>> 
        >>> register_metadata_hook(my_hook)
        >>> # ... later ...
        >>> unregister_metadata_hook(my_hook)
    """
    global _metadata_hooks
    
    initial_count = len(_metadata_hooks)
    _metadata_hooks = [(p, h) for p, h in _metadata_hooks if h != hook]
    
    removed = len(_metadata_hooks) < initial_count
    if removed:
        logger.debug(f"Unregistered metadata hook: {hook.__name__}")
    else:
        logger.warning(f"Hook not found for unregistration: {hook.__name__}")
    
    return removed


def clear_metadata_hooks() -> None:
    """
    Clear all registered metadata hooks.
    
    Useful for testing or resetting the hook system.
    
    Example:
        >>> clear_metadata_hooks()
    """
    global _metadata_hooks
    count = len(_metadata_hooks)
    _metadata_hooks = []
    logger.debug(f"Cleared {count} metadata hooks")


def execute_metadata_hooks(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute all registered hooks on the metadata.
    
    Hooks are executed in priority order (highest first). If a hook raises
    an exception, it is logged and the hook is skipped, but execution continues
    with remaining hooks.
    
    This function is called internally by the middleware and should not
    normally be called by user code.
    
    Args:
        metadata: The metadata dictionary to process
    
    Returns:
        The processed metadata dictionary
    
    Example:
        >>> metadata = {'agent': 'Test'}
        >>> result = execute_metadata_hooks(metadata)
    """
    # Always return a copy to avoid modifying the original
    result = metadata.copy()

    if not _metadata_hooks:
        return result
    
    for priority, hook in _metadata_hooks:
        try:
            logger.debug(f"Executing hook: {hook.__name__} (priority: {priority})")
            result = hook(result)
            
            if not isinstance(result, dict):
                logger.error(
                    f"Hook {hook.__name__} returned non-dict value: {type(result)}. "
                    f"Skipping this hook."
                )
                result = metadata.copy()  # Restore original
                
        except Exception as e:
            logger.error(
                f"Error executing hook {hook.__name__}: {e}. "
                f"Skipping this hook and continuing with others.",
                exc_info=True
            )
            # Continue with other hooks even if one fails
    
    return result


def get_registered_hooks() -> List[Callable[[Dict[str, Any]], Dict[str, Any]]]:
    """
    Get a list of all registered hooks in execution order.
    
    Returns:
        List of hook functions in priority order
    
    Example:
        >>> hooks = get_registered_hooks()
        >>> print(f"Registered {len(hooks)} hooks")
    """
    return [hook for _, hook in _metadata_hooks]


__all__ = [
    'register_metadata_hook',
    'unregister_metadata_hook',
    'clear_metadata_hooks',
    'execute_metadata_hooks',
    'get_registered_hooks'
]

