"""
Decorator-based metadata injection for Revenium LiteLLM middleware.

This module provides convenient decorators for automatically injecting metadata
into LiteLLM completion calls. Decorators work with both sync and async functions.

Available decorators:
    - track_agent: Set agent metadata
    - track_task: Set task_type metadata
    - track_job: Set agentic job metadata (id, name, type, version)
    - track_trace: Set trace_id metadata
    - track_organization: Set organization_name metadata
    - track_subscription: Set subscription_id metadata
    - track_product: Set product_name metadata
    - track_subscriber: Set subscriber metadata (id, email, credential)
    - track_quality: Set response_quality_score metadata

Example:
    >>> from revenium_middleware.litellm.client import track_agent, track_task, track_trace
    >>>
    >>> @track_agent("Lead Analyst")
    >>> @track_trace("workflow-123")
    >>> def analyze_market():
    ...     response = litellm.completion(...)
    ...     return response
    >>>
    >>> @track_task("research")
    >>> async def research_topic():
    ...     response = await litellm.acompletion(...)
    ...     return response
    >>>
    >>> # Dynamic attribute extraction
    >>> @track_agent(name_from_arg="agent_name")
    >>> def process_with_agent(agent_name, data):
    ...     response = litellm.completion(...)
    ...     return response
"""

import functools
import inspect
from typing import Any, Callable, Dict, Optional
from .context import metadata_context


def track_agent(
    agent: Optional[str] = None,
    *,
    name_from_arg: Optional[str] = None,
    name_from_attr: Optional[str] = None
) -> Callable:
    """
    Decorator to automatically inject agent metadata into LiteLLM calls.
    
    This decorator sets the 'agent' field in the metadata context for the
    duration of the decorated function. It supports both static agent names
    and dynamic extraction from function arguments or object attributes.
    
    Works with both sync and async functions.
    
    Args:
        agent: Static agent name to use. Mutually exclusive with name_from_arg
              and name_from_attr.
        name_from_arg: Name of function argument to use as agent name.
                       The argument value will be converted to string.
        name_from_attr: Name of object attribute to use as agent name.
                        Only works when decorating methods. Uses self.{attr}.
    
    Returns:
        Decorated function that sets agent metadata
    
    Raises:
        ValueError: If multiple or no agent sources are specified
    
    Example:
        >>> # Static agent name
        >>> @track_agent("Lead Analyst")
        >>> def analyze():
        ...     return litellm.completion(...)
        >>> 
        >>> # Dynamic from argument
        >>> @track_agent(name_from_arg="agent_name")
        >>> def process(agent_name, data):
        ...     return litellm.completion(...)
        >>> 
        >>> # Dynamic from object attribute
        >>> class Agent:
        ...     def __init__(self, name):
        ...         self.name = name
        ...     
        ...     @track_agent(name_from_attr="name")
        ...     def execute(self):
        ...         return litellm.completion(...)
    """
    # Validate arguments
    sources = sum([
        agent is not None,
        name_from_arg is not None,
        name_from_attr is not None
    ])
    
    if sources == 0:
        raise ValueError("Must specify agent, name_from_arg, or name_from_attr")
    if sources > 1:
        raise ValueError("Can only specify one of: agent, name_from_arg, name_from_attr")
    
    def decorator(func: Callable) -> Callable:
        # Check if function is async
        is_async = inspect.iscoroutinefunction(func)
        
        if is_async:
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                # Determine agent name
                agent_name = _get_value(agent, name_from_arg, name_from_attr, args, kwargs, func, "agent")

                # Set context and call function
                with metadata_context.set(agent=agent_name):
                    return await func(*args, **kwargs)

            return async_wrapper
        else:
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                # Determine agent name
                agent_name = _get_value(agent, name_from_arg, name_from_attr, args, kwargs, func, "agent")

                # Set context and call function
                with metadata_context.set(agent=agent_name):
                    return func(*args, **kwargs)

            return sync_wrapper
    
    return decorator


def track_task(
    task_type: Optional[str] = None,
    *,
    type_from_arg: Optional[str] = None,
    type_from_attr: Optional[str] = None
) -> Callable:
    """
    Decorator to automatically inject task_type metadata into LiteLLM calls.
    
    This decorator sets the 'task_type' field in the metadata context for the
    duration of the decorated function. It supports both static task types
    and dynamic extraction from function arguments or object attributes.
    
    Works with both sync and async functions.
    
    Args:
        task_type: Static task type to use. Mutually exclusive with type_from_arg
                   and type_from_attr.
        type_from_arg: Name of function argument to use as task type.
                       The argument value will be converted to string.
        type_from_attr: Name of object attribute to use as task type.
                        Only works when decorating methods. Uses self.{attr}.
    
    Returns:
        Decorated function that sets task_type metadata
    
    Raises:
        ValueError: If multiple or no task type sources are specified
    
    Example:
        >>> # Static task type
        >>> @track_task("research")
        >>> def research():
        ...     return litellm.completion(...)
        >>> 
        >>> # Dynamic from argument
        >>> @track_task(type_from_arg="operation")
        >>> def process(operation, data):
        ...     return litellm.completion(...)
        >>> 
        >>> # Dynamic from object attribute
        >>> class Task:
        ...     def __init__(self, task_type):
        ...         self.task_type = task_type
        ...     
        ...     @track_task(type_from_attr="task_type")
        ...     def execute(self):
        ...         return litellm.completion(...)
    """
    # Validate arguments
    sources = sum([
        task_type is not None,
        type_from_arg is not None,
        type_from_attr is not None
    ])
    
    if sources == 0:
        raise ValueError("Must specify task_type, type_from_arg, or type_from_attr")
    if sources > 1:
        raise ValueError("Can only specify one of: task_type, type_from_arg, type_from_attr")
    
    def decorator(func: Callable) -> Callable:
        # Check if function is async
        is_async = inspect.iscoroutinefunction(func)
        
        if is_async:
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                # Determine task type
                task_type_value = _get_value(task_type, type_from_arg, type_from_attr, args, kwargs, func, "task_type")

                # Set context and call function
                with metadata_context.set(task_type=task_type_value):
                    return await func(*args, **kwargs)

            return async_wrapper
        else:
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                # Determine task type
                task_type_value = _get_value(task_type, type_from_arg, type_from_attr, args, kwargs, func, "task_type")

                # Set context and call function
                with metadata_context.set(task_type=task_type_value):
                    return func(*args, **kwargs)

            return sync_wrapper
    
    return decorator


def track_job(
    job_id: Optional[str] = None,
    *,
    name: Optional[str] = None,
    type: Optional[str] = None,
    version: Optional[str] = None,
    job_id_from_arg: Optional[str] = None,
    type_from_arg: Optional[str] = None
) -> Callable:
    """
    Decorator to automatically inject agentic job metadata into LiteLLM calls.

    Sets the agentic job fields in the metadata context for the duration of the
    decorated function so every completion inside inherits them. Supports static
    values and dynamic extraction from function arguments.

    Works with both sync and async functions.

    Args:
        job_id: Static job instance ID. Mutually exclusive with job_id_from_arg.
        name: Static human-readable job name.
        type: Static job type category. Mutually exclusive with type_from_arg.
        version: Static job version.
        job_id_from_arg: Name of function argument to use as the job ID.
        type_from_arg: Name of function argument to use as the job type.

    Returns:
        Decorated function that sets agentic job metadata

    Raises:
        ValueError: If job_id and job_id_from_arg are both given or both missing,
                    or if type and type_from_arg are both given.

    Example:
        >>> @track_job(job_id="loan-app-12345", name="Process Loan", type="loan_processing", version="1.2.0")
        >>> def process_loan():
        ...     return litellm.completion(...)
        >>>
        >>> # Dynamic extraction from arguments
        >>> @track_job(job_id_from_arg="job_id", type_from_arg="job_type")
        >>> def process(job_id, job_type, data):
        ...     return litellm.completion(...)
    """
    if (job_id is None) == (job_id_from_arg is None):
        raise ValueError("Must specify exactly one of: job_id, job_id_from_arg")
    if type is not None and type_from_arg is not None:
        raise ValueError("Can only specify one of: type, type_from_arg")

    def decorator(func: Callable) -> Callable:
        is_async = inspect.iscoroutinefunction(func)

        def _raw_arg(arg_name: str, args: tuple, kwargs: dict, field_name: str) -> Any:
            """Resolve a dynamic argument's raw value, without stringification."""
            # Check the call-site kwargs first (mirrors _get_value) — signature
            # binding files keyword args under a **kwargs catch-all parameter's
            # name, which would hide them from the bound-arguments lookup below.
            if arg_name in kwargs:
                return kwargs[arg_name]
            try:
                bound = inspect.signature(func).bind(*args, **kwargs)
                bound.apply_defaults()
            except TypeError:
                bound = None
            if bound is not None and arg_name in bound.arguments:
                return bound.arguments[arg_name]
            raise ValueError(f"Argument '{arg_name}' not found in function call for {field_name}")

        def _job_metadata(args: tuple, kwargs: dict) -> Dict[str, Any]:
            resolved_job_id = job_id
            if job_id_from_arg is not None:
                raw_job_id = _raw_arg(job_id_from_arg, args, kwargs, "job_id")
                if raw_job_id is None:
                    raise ValueError(
                        f"job_id argument '{job_id_from_arg}' is None — a job id is required"
                    )
                resolved_job_id = str(raw_job_id)
            metadata: Dict[str, Any] = {"agentic_job_id": resolved_job_id}
            if name is not None:
                metadata["agentic_job_name"] = name
            resolved_type = type
            if type_from_arg is not None:
                raw_type = _raw_arg(type_from_arg, args, kwargs, "job_type")
                resolved_type = str(raw_type) if raw_type is not None else None
            if resolved_type is not None:
                metadata["agentic_job_type"] = resolved_type
            if version is not None:
                metadata["agentic_job_version"] = version
            return metadata

        if is_async:
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                with metadata_context.set(**_job_metadata(args, kwargs)):
                    return await func(*args, **kwargs)

            return async_wrapper
        else:
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                with metadata_context.set(**_job_metadata(args, kwargs)):
                    return func(*args, **kwargs)

            return sync_wrapper

    return decorator


def _get_value(
    static_value: Optional[str],
    arg_name: Optional[str],
    attr_name: Optional[str],
    args: tuple,
    kwargs: dict,
    func: Callable,
    field_name: str
) -> str:
    """
    Generic helper to extract a value from static value, argument, or attribute.

    Args:
        static_value: Static value
        arg_name: Argument name to extract from
        attr_name: Attribute name to extract from
        args: Positional arguments
        kwargs: Keyword arguments
        func: The decorated function
        field_name: Name of the field (for error messages)

    Returns:
        Value as string

    Raises:
        ValueError: If argument/attribute not found or invalid
    """
    if static_value is not None:
        return static_value

    if arg_name is not None:
        # Try to get from kwargs first
        if arg_name in kwargs:
            return str(kwargs[arg_name])

        # Try to get from args using function signature
        sig = inspect.signature(func)
        param_names = list(sig.parameters.keys())

        if arg_name in param_names:
            arg_index = param_names.index(arg_name)
            if arg_index < len(args):
                return str(args[arg_index])

        raise ValueError(f"Argument '{arg_name}' not found in function call for {field_name}")

    if attr_name is not None:
        # Get from self (first argument for methods)
        if len(args) == 0:
            raise ValueError(f"Cannot use attr on function without self argument for {field_name}")

        self_obj = args[0]
        if not hasattr(self_obj, attr_name):
            raise ValueError(f"Object does not have attribute '{attr_name}' for {field_name}")

        return str(getattr(self_obj, attr_name))

    raise ValueError(f"No {field_name} source specified")


def track_trace(
    trace_id: Optional[str] = None,
    *,
    id_from_arg: Optional[str] = None,
    id_from_attr: Optional[str] = None
) -> Callable:
    """
    Decorator to automatically inject trace_id metadata into LiteLLM calls.

    This decorator sets the 'trace_id' field in the metadata context for the
    duration of the decorated function. It supports both static trace IDs
    and dynamic extraction from function arguments or object attributes.

    Works with both sync and async functions.

    Args:
        trace_id: Static trace ID to use. Mutually exclusive with id_from_arg
                  and id_from_attr.
        id_from_arg: Name of function argument to use as trace ID.
                     The argument value will be converted to string.
        id_from_attr: Name of object attribute to use as trace ID.
                      Only works when decorating methods. Uses self.{attr}.

    Returns:
        Decorated function that sets trace_id metadata

    Raises:
        ValueError: If multiple or no trace ID sources are specified

    Example:
        >>> # Static trace ID
        >>> @track_trace("workflow-123")
        >>> def process():
        ...     return litellm.completion(...)
        >>>
        >>> # Dynamic from argument
        >>> @track_trace(id_from_arg="workflow_id")
        >>> def process(workflow_id, data):
        ...     return litellm.completion(...)
    """
    # Validate arguments
    sources = sum([
        trace_id is not None,
        id_from_arg is not None,
        id_from_attr is not None
    ])

    if sources == 0:
        raise ValueError("Must specify trace_id, id_from_arg, or id_from_attr")
    if sources > 1:
        raise ValueError("Can only specify one of: trace_id, id_from_arg, id_from_attr")

    def decorator(func: Callable) -> Callable:
        # Check if function is async
        is_async = inspect.iscoroutinefunction(func)

        if is_async:
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                # Determine trace ID
                trace_id_value = _get_value(trace_id, id_from_arg, id_from_attr, args, kwargs, func, "trace_id")

                # Set context and call function
                with metadata_context.set(trace_id=trace_id_value):
                    return await func(*args, **kwargs)

            return async_wrapper
        else:
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                # Determine trace ID
                trace_id_value = _get_value(trace_id, id_from_arg, id_from_attr, args, kwargs, func, "trace_id")

                # Set context and call function
                with metadata_context.set(trace_id=trace_id_value):
                    return func(*args, **kwargs)

            return sync_wrapper

    return decorator


def track_organization(
    organization_name: Optional[str] = None,
    *,
    name_from_arg: Optional[str] = None,
    name_from_attr: Optional[str] = None,
    # Deprecated parameter names for backward compatibility
    organization_id: Optional[str] = None,
    id_from_arg: Optional[str] = None,
    id_from_attr: Optional[str] = None
) -> Callable:
    """
    Decorator to automatically inject organization_name metadata into LiteLLM calls.

    This decorator sets the 'organization_name' field in the metadata context for the
    duration of the decorated function. It supports both static organization names
    and dynamic extraction from function arguments or object attributes.

    Works with both sync and async functions.

    Args:
        organization_name: Static organization name to use. Mutually exclusive with
                          name_from_arg and name_from_attr.
        name_from_arg: Name of function argument to use as organization name.
                       The argument value will be converted to string.
        name_from_attr: Name of object attribute to use as organization name.
                        Only works when decorating methods. Uses self.{attr}.
        organization_id: Deprecated. Use organization_name instead.
        id_from_arg: Deprecated. Use name_from_arg instead.
        id_from_attr: Deprecated. Use name_from_attr instead.

    Returns:
        Decorated function that sets organization_name metadata

    Raises:
        ValueError: If multiple or no organization name sources are specified

    Example:
        >>> # Static organization name
        >>> @track_organization("AcmeCorp")
        >>> def process():
        ...     return litellm.completion(...)
        >>>
        >>> # Dynamic from argument
        >>> @track_organization(name_from_arg="org_name")
        >>> def process(org_name, data):
        ...     return litellm.completion(...)
    """
    # Support deprecated parameter names (new names take precedence)
    effective_name = organization_name if organization_name is not None else organization_id
    effective_from_arg = name_from_arg if name_from_arg is not None else id_from_arg
    effective_from_attr = name_from_attr if name_from_attr is not None else id_from_attr

    # Validate arguments
    sources = sum([
        effective_name is not None,
        effective_from_arg is not None,
        effective_from_attr is not None
    ])

    if sources == 0:
        raise ValueError("Must specify organization_name, name_from_arg, or name_from_attr")
    if sources > 1:
        raise ValueError("Can only specify one of: organization_name, name_from_arg, name_from_attr")

    def decorator(func: Callable) -> Callable:
        # Check if function is async
        is_async = inspect.iscoroutinefunction(func)

        if is_async:
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                # Determine organization name
                org_name_value = _get_value(effective_name, effective_from_arg, effective_from_attr, args, kwargs, func, "organization_name")

                # Set context using organization_name (not organization_id)
                with metadata_context.set(organization_name=org_name_value):
                    return await func(*args, **kwargs)

            return async_wrapper
        else:
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                # Determine organization name
                org_name_value = _get_value(effective_name, effective_from_arg, effective_from_attr, args, kwargs, func, "organization_name")

                # Set context using organization_name (not organization_id)
                with metadata_context.set(organization_name=org_name_value):
                    return func(*args, **kwargs)

            return sync_wrapper

    return decorator


def track_subscription(
    subscription_id: Optional[str] = None,
    *,
    id_from_arg: Optional[str] = None,
    id_from_attr: Optional[str] = None
) -> Callable:
    """
    Decorator to automatically inject subscription_id metadata into LiteLLM calls.

    This decorator sets the 'subscription_id' field in the metadata context for the
    duration of the decorated function. It supports both static subscription IDs
    and dynamic extraction from function arguments or object attributes.

    Works with both sync and async functions.

    Args:
        subscription_id: Static subscription ID to use. Mutually exclusive with
                         id_from_arg and id_from_attr.
        id_from_arg: Name of function argument to use as subscription ID.
                     The argument value will be converted to string.
        id_from_attr: Name of object attribute to use as subscription ID.
                      Only works when decorating methods. Uses self.{attr}.

    Returns:
        Decorated function that sets subscription_id metadata

    Raises:
        ValueError: If multiple or no subscription ID sources are specified

    Example:
        >>> # Static subscription ID
        >>> @track_subscription("sub-123")
        >>> def process():
        ...     return litellm.completion(...)
        >>>
        >>> # Dynamic from argument
        >>> @track_subscription(id_from_arg="sub_id")
        >>> def process(sub_id, data):
        ...     return litellm.completion(...)
    """
    # Validate arguments
    sources = sum([
        subscription_id is not None,
        id_from_arg is not None,
        id_from_attr is not None
    ])

    if sources == 0:
        raise ValueError("Must specify subscription_id, id_from_arg, or id_from_attr")
    if sources > 1:
        raise ValueError("Can only specify one of: subscription_id, id_from_arg, id_from_attr")

    def decorator(func: Callable) -> Callable:
        # Check if function is async
        is_async = inspect.iscoroutinefunction(func)

        if is_async:
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                # Determine subscription ID
                sub_id_value = _get_value(subscription_id, id_from_arg, id_from_attr, args, kwargs, func, "subscription_id")

                # Set context and call function
                with metadata_context.set(subscription_id=sub_id_value):
                    return await func(*args, **kwargs)

            return async_wrapper
        else:
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                # Determine subscription ID
                sub_id_value = _get_value(subscription_id, id_from_arg, id_from_attr, args, kwargs, func, "subscription_id")

                # Set context and call function
                with metadata_context.set(subscription_id=sub_id_value):
                    return func(*args, **kwargs)

            return sync_wrapper

    return decorator


def track_product(
    product_name: Optional[str] = None,
    *,
    name_from_arg: Optional[str] = None,
    name_from_attr: Optional[str] = None,
    # Deprecated parameter names for backward compatibility
    product_id: Optional[str] = None,
    id_from_arg: Optional[str] = None,
    id_from_attr: Optional[str] = None
) -> Callable:
    """
    Decorator to automatically inject product_name metadata into LiteLLM calls.

    This decorator sets the 'product_name' field in the metadata context for the
    duration of the decorated function. It supports both static product names
    and dynamic extraction from function arguments or object attributes.

    Works with both sync and async functions.

    Args:
        product_name: Static product name to use. Mutually exclusive with
                      name_from_arg and name_from_attr.
        name_from_arg: Name of function argument to use as product name.
                       The argument value will be converted to string.
        name_from_attr: Name of object attribute to use as product name.
                        Only works when decorating methods. Uses self.{attr}.
        product_id: Deprecated. Use product_name instead.
        id_from_arg: Deprecated. Use name_from_arg instead.
        id_from_attr: Deprecated. Use name_from_attr instead.

    Returns:
        Decorated function that sets product_name metadata

    Raises:
        ValueError: If multiple or no product name sources are specified

    Example:
        >>> # Static product name
        >>> @track_product("ai-assistant")
        >>> def process():
        ...     return litellm.completion(...)
        >>>
        >>> # Dynamic from argument
        >>> @track_product(name_from_arg="prod_name")
        >>> def process(prod_name, data):
        ...     return litellm.completion(...)
    """
    # Support deprecated parameter names (new names take precedence)
    effective_name = product_name if product_name is not None else product_id
    effective_from_arg = name_from_arg if name_from_arg is not None else id_from_arg
    effective_from_attr = name_from_attr if name_from_attr is not None else id_from_attr

    # Validate arguments
    sources = sum([
        effective_name is not None,
        effective_from_arg is not None,
        effective_from_attr is not None
    ])

    if sources == 0:
        raise ValueError("Must specify product_name, name_from_arg, or name_from_attr")
    if sources > 1:
        raise ValueError("Can only specify one of: product_name, name_from_arg, name_from_attr")

    def decorator(func: Callable) -> Callable:
        # Check if function is async
        is_async = inspect.iscoroutinefunction(func)

        if is_async:
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                # Determine product name
                prod_name_value = _get_value(effective_name, effective_from_arg, effective_from_attr, args, kwargs, func, "product_name")

                # Set context using product_name (not product_id)
                with metadata_context.set(product_name=prod_name_value):
                    return await func(*args, **kwargs)

            return async_wrapper
        else:
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                # Determine product name
                prod_name_value = _get_value(effective_name, effective_from_arg, effective_from_attr, args, kwargs, func, "product_name")

                # Set context using product_name (not product_id)
                with metadata_context.set(product_name=prod_name_value):
                    return func(*args, **kwargs)

            return sync_wrapper

    return decorator


def _build_subscriber_dict(
    subscriber_id: Optional[str],
    subscriber_email: Optional[str],
    credential_name: Optional[str],
    id_from_arg: Optional[str],
    email_from_arg: Optional[str],
    credential_from_arg: Optional[str],
    args: tuple,
    kwargs: dict,
    func: Callable
) -> dict:
    """
    Helper function to build subscriber dictionary from various sources.

    Args:
        subscriber_id: Static subscriber ID
        subscriber_email: Static subscriber email
        credential_name: Static credential name
        id_from_arg: Argument name for subscriber ID
        email_from_arg: Argument name for subscriber email
        credential_from_arg: Argument name for credential
        args: Positional arguments
        kwargs: Keyword arguments
        func: The decorated function

    Returns:
        Dictionary with subscriber information
    """
    subscriber = {}

    # Get subscriber ID
    if subscriber_id is not None:
        subscriber['id'] = subscriber_id
    elif id_from_arg is not None:
        try:
            subscriber['id'] = _get_value(None, id_from_arg, None, args, kwargs, func, "subscriber_id")
        except ValueError:
            pass  # Optional field, skip if not found

    # Get subscriber email
    if subscriber_email is not None:
        subscriber['email'] = subscriber_email
    elif email_from_arg is not None:
        try:
            subscriber['email'] = _get_value(None, email_from_arg, None, args, kwargs, func, "subscriber_email")
        except ValueError:
            pass  # Optional field, skip if not found

    # Get credential name
    if credential_name is not None:
        subscriber['credential'] = {'name': credential_name}
    elif credential_from_arg is not None:
        try:
            cred_name = _get_value(None, credential_from_arg, None, args, kwargs, func, "credential_name")
            subscriber['credential'] = {'name': cred_name}
        except ValueError:
            pass  # Optional field, skip if not found

    return subscriber


def track_subscriber(
    subscriber_id: Optional[str] = None,
    subscriber_email: Optional[str] = None,
    credential_name: Optional[str] = None,
    *,
    id_from_arg: Optional[str] = None,
    email_from_arg: Optional[str] = None,
    credential_from_arg: Optional[str] = None
) -> Callable:
    """
    Decorator to automatically inject subscriber metadata into LiteLLM calls.

    This decorator sets the 'subscriber' field (with id, email, and credential)
    in the metadata context for the duration of the decorated function.

    Works with both sync and async functions.

    Args:
        subscriber_id: Static subscriber ID to use.
        subscriber_email: Static subscriber email to use.
        credential_name: Static credential name to use.
        id_from_arg: Name of function argument to use as subscriber ID.
        email_from_arg: Name of function argument to use as subscriber email.
        credential_from_arg: Name of function argument to use as credential name.

    Returns:
        Decorated function that sets subscriber metadata

    Example:
        >>> # Static subscriber info
        >>> @track_subscriber(subscriber_id="user-123", subscriber_email="user@example.com")
        >>> def process():
        ...     return litellm.completion(...)
        >>>
        >>> # Dynamic from arguments
        >>> @track_subscriber(id_from_arg="user_id", email_from_arg="user_email")
        >>> def process(user_id, user_email, data):
        ...     return litellm.completion(...)
    """
    def decorator(func: Callable) -> Callable:
        # Check if function is async
        is_async = inspect.iscoroutinefunction(func)

        if is_async:
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                # Build subscriber dict
                subscriber = _build_subscriber_dict(
                    subscriber_id, subscriber_email, credential_name,
                    id_from_arg, email_from_arg, credential_from_arg,
                    args, kwargs, func
                )

                # Set context and call function
                if subscriber:
                    with metadata_context.set(subscriber=subscriber):
                        return await func(*args, **kwargs)
                else:
                    return await func(*args, **kwargs)

            return async_wrapper
        else:
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                # Build subscriber dict
                subscriber = _build_subscriber_dict(
                    subscriber_id, subscriber_email, credential_name,
                    id_from_arg, email_from_arg, credential_from_arg,
                    args, kwargs, func
                )

                # Set context and call function
                if subscriber:
                    with metadata_context.set(subscriber=subscriber):
                        return func(*args, **kwargs)
                else:
                    return func(*args, **kwargs)

            return sync_wrapper

    return decorator


def track_quality(
    quality_score: Optional[float] = None,
    *,
    score_from_arg: Optional[str] = None,
    score_from_attr: Optional[str] = None
) -> Callable:
    """
    Decorator to automatically inject response_quality_score metadata into LiteLLM calls.

    This decorator sets the 'response_quality_score' field in the metadata context
    for the duration of the decorated function. It supports both static quality scores
    and dynamic extraction from function arguments or object attributes.

    Works with both sync and async functions.

    Args:
        quality_score: Static quality score to use (0.0-1.0). Mutually exclusive with
                       score_from_arg and score_from_attr.
        score_from_arg: Name of function argument to use as quality score.
                        The argument value will be converted to float.
        score_from_attr: Name of object attribute to use as quality score.
                         Only works when decorating methods. Uses self.{attr}.

    Returns:
        Decorated function that sets response_quality_score metadata

    Raises:
        ValueError: If multiple or no quality score sources are specified

    Example:
        >>> # Static quality score
        >>> @track_quality(0.95)
        >>> def process():
        ...     return litellm.completion(...)
        >>>
        >>> # Dynamic from argument
        >>> @track_quality(score_from_arg="min_quality")
        >>> def process(min_quality, data):
        ...     return litellm.completion(...)
    """
    # Validate arguments
    sources = sum([
        quality_score is not None,
        score_from_arg is not None,
        score_from_attr is not None
    ])

    if sources == 0:
        raise ValueError("Must specify quality_score, score_from_arg, or score_from_attr")
    if sources > 1:
        raise ValueError("Can only specify one of: quality_score, score_from_arg, score_from_attr")

    def decorator(func: Callable) -> Callable:
        # Check if function is async
        is_async = inspect.iscoroutinefunction(func)

        if is_async:
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                # Determine quality score
                score_value = _get_quality_score(quality_score, score_from_arg, score_from_attr, args, kwargs, func)

                # Set context and call function
                with metadata_context.set(response_quality_score=score_value):
                    return await func(*args, **kwargs)

            return async_wrapper
        else:
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                # Determine quality score
                score_value = _get_quality_score(quality_score, score_from_arg, score_from_attr, args, kwargs, func)

                # Set context and call function
                with metadata_context.set(response_quality_score=score_value):
                    return func(*args, **kwargs)

            return sync_wrapper

    return decorator


def _get_quality_score(
    static_score: Optional[float],
    arg_name: Optional[str],
    attr_name: Optional[str],
    args: tuple,
    kwargs: dict,
    func: Callable
) -> float:
    """
    Extract quality score from static value, argument, or attribute.

    Args:
        static_score: Static quality score
        arg_name: Argument name to extract from
        attr_name: Attribute name to extract from
        args: Positional arguments
        kwargs: Keyword arguments
        func: The decorated function

    Returns:
        Quality score as float

    Raises:
        ValueError: If argument/attribute not found or invalid
    """
    # Use the generic _get_value function and convert to float
    value_str = _get_value(
        str(static_score) if static_score is not None else None,
        arg_name,
        attr_name,
        args,
        kwargs,
        func,
        "quality_score"
    )
    return float(value_str)


__all__ = [
    'track_agent',
    'track_task',
    'track_job',
    'track_trace',
    'track_organization',
    'track_subscription',
    'track_product',
    'track_subscriber',
    'track_quality',
]

