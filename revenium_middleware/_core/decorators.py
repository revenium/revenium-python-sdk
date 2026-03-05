"""
Decorators for selective metering in Revenium middleware.

This module provides decorators that allow users to selectively meter
specific functions/endpoints instead of automatically metering all API calls.
"""

import functools
import asyncio
from typing import Optional, Dict, Any, Callable, TypeVar

from .context import (
    set_decorated_context,
    clear_decorated_context,
    set_injected_metadata,
    clear_injected_metadata,
)

# Type variables for generic decorator support
F = TypeVar('F', bound=Callable[..., Any])


def revenium_meter(
    operation_type: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    name: Optional[str] = None
) -> Callable[[F], F]:
    """
    Decorator to mark a function for selective metering.
    
    When selective metering is enabled in the middleware configuration,
    only functions decorated with @revenium_meter will have their API
    calls metered. When selective metering is disabled (default), this decorator
    has no effect and all API calls are metered automatically.
    
    Args:
        operation_type: Optional override for operation type (e.g., 'CHAT', 'EMBED')
        metadata: Optional default metadata to include in metering calls
        name: Optional friendly name for the function (for logging/debugging)
    
    Returns:
        Decorated function that sets context when called
    
    Example:
        @revenium_meter(metadata={'task_type': 'chat_analysis'})
        def analyze_text(text: str) -> str:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": text}]
            )
            return response.choices[0].message.content
    """
    def decorator(func: F) -> F:
        # Determine if function is async
        is_async = asyncio.iscoroutinefunction(func)
        
        if is_async:
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                # Prepare metadata
                func_metadata = {
                    'function_name': name or func.__name__,
                    'is_decorated': True,
                }
                if operation_type:
                    func_metadata['operation_type'] = operation_type
                if metadata:
                    func_metadata.update(metadata)
                
                # Set context before calling function
                set_decorated_context(True, func_metadata)
                try:
                    result = await func(*args, **kwargs)
                    return result
                finally:
                    # Clear context after function completes
                    clear_decorated_context()
            
            return async_wrapper  # type: ignore
        else:
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                # Prepare metadata
                func_metadata = {
                    'function_name': name or func.__name__,
                    'is_decorated': True,
                }
                if operation_type:
                    func_metadata['operation_type'] = operation_type
                if metadata:
                    func_metadata.update(metadata)
                
                # Set context before calling function
                set_decorated_context(True, func_metadata)
                try:
                    result = func(*args, **kwargs)
                    return result
                finally:
                    # Clear context after function completes
                    clear_decorated_context()
            
            return sync_wrapper  # type: ignore
    
    return decorator


# Alias for convenience
track_usage = revenium_meter


def revenium_metadata(**metadata_kwargs) -> Callable[[F], F]:
    """
    Decorator to inject metadata into all API calls within a function's scope.

    This decorator automatically injects metadata into every API call made
    within the decorated function, eliminating the need to pass usage_metadata to
    each individual call. API-level metadata (passed directly to API calls) takes
    precedence over decorator-injected metadata.

    This decorator works independently of selective metering and can be used:
    - With automatic metering (default behavior)
    - With selective metering (when @revenium_meter is also used)
    - Alongside @revenium_meter on the same function

    Args:
        **metadata_kwargs: Arbitrary keyword arguments that will be injected as metadata.
            Common fields include:
            - trace_id: Unique identifier for a conversation or session
            - task_type: Classification of the AI operation
            - organization_id: Customer or department ID
            - subscription_id: Reference to a billing plan
            - product_id: Your product or feature making the AI call
            - agent: Identifier for the specific AI agent
            - response_quality_score: Quality metric (0-1)
            - subscriber: Nested object with user information

    Returns:
        Decorated function that injects metadata into API calls

    Example:
        @revenium_metadata(org_id="acme", task_type="analysis")
        def analyze_documents(docs):
            # All API calls here automatically get the metadata
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": "Analyze this"}]
            )
            return response

        # Can be combined with @revenium_meter for selective metering
        @revenium_meter()
        @revenium_metadata(org_id="acme", trace_id="session-123")
        def metered_analysis():
            response = client.chat.completions.create(...)
            return response

        # API-level metadata overrides decorator metadata
        @revenium_metadata(task_type="default")
        def mixed_metadata():
            # Uses decorator metadata: {"task_type": "default"}
            response1 = client.chat.completions.create(...)

            # API-level overrides: {"task_type": "special"}
            response2 = client.chat.completions.create(
                ...,
                usage_metadata={"task_type": "special"}
            )
            return response1, response2
    """
    def decorator(func: F) -> F:
        # Determine if function is async
        is_async = asyncio.iscoroutinefunction(func)

        if is_async:
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                # Set injected metadata context before calling function
                set_injected_metadata(metadata_kwargs)
                try:
                    result = await func(*args, **kwargs)
                    return result
                finally:
                    # Clear injected metadata context after function completes
                    clear_injected_metadata()

            return async_wrapper  # type: ignore
        else:
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                # Set injected metadata context before calling function
                set_injected_metadata(metadata_kwargs)
                try:
                    result = func(*args, **kwargs)
                    return result
                finally:
                    # Clear injected metadata context after function completes
                    clear_injected_metadata()

            return sync_wrapper  # type: ignore

    return decorator

