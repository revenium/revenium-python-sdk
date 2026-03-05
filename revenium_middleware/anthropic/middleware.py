import logging
import datetime
import wrapt
import time
import contextvars
import os
import threading
import queue
from typing import Optional, Callable, Any, Dict, Tuple

# Import our provider detection and Bedrock adapter
from .provider import Provider, detect_provider, get_provider_metadata
from .bedrock_adapter import (
    bedrock_invoke, create_bedrock_payload, create_anthropic_response, BedrockStreamWrapper,
    BedrockError, BedrockValidationError, BedrockInvokeError, BedrockStreamError
)

# Import decorator support and metering client from core package
from revenium_middleware import client, run_async_in_thread, shutdown_event, merge_metadata
from revenium_middleware._core.subscriber import extract_subscriber_from_metadata

# Import trace visualization functions
from .trace_fields import (
    get_environment, get_region, get_credential_alias,
    get_trace_type, get_trace_name, get_parent_transaction_id,
    get_transaction_name, get_retry_number, detect_operation_type,
    detect_vision_content
)

# Import configuration and prompt capture utilities
from .config import Config
from .prompt_extractor import (
    extract_prompts_from_request,
    extract_response_content,
    extract_streaming_response_content
)

# Import summary printer
from .summary_printer import print_usage_summary

logger = logging.getLogger("revenium_middleware.extension")

# Define usage context for thread-safe metadata storage
usage_context = contextvars.ContextVar('usage_metadata', default={})


def _emit_usage_summary(
    model: str,
    provider: str,
    request_duration: float,
    prompt_tokens: int,
    completion_tokens: int,
    response_id: str,
    usage_metadata: Dict[str, Any],
) -> None:
    """
    Helper function to emit usage summary asynchronously.

    Runs summary printing in a background thread to avoid blocking API responses.
    This is a fire-and-forget operation that never fails the main request.

    Args:
        model: The model name
        provider: The provider name
        request_duration: Request duration in milliseconds
        prompt_tokens: Number of input tokens
        completion_tokens: Number of output tokens
        response_id: The transaction ID
        usage_metadata: Metadata dictionary containing trace_id and other info
    """
    revenium_api_key = os.getenv(Config.ENV_REVENIUM_API_KEY)
    if revenium_api_key:
        # Run summary printing in background thread to avoid blocking
        def _print_summary_async():
            print_usage_summary(
                model=model,
                provider=provider,
                request_duration=request_duration,
                input_token_count=prompt_tokens,
                output_token_count=completion_tokens,
                total_token_count=prompt_tokens + completion_tokens,
                transaction_id=response_id,
                trace_id=usage_metadata.get("trace_id"),
                revenium_api_key=revenium_api_key,
            )

        run_async_in_thread(_print_summary_async)

# Ensure debug logging is enabled when REVENIUM_DEBUG is set
if os.getenv("REVENIUM_DEBUG", "").lower() in ("true", "1", "yes"):
    logger.setLevel(logging.DEBUG)
    # Also ensure the handler is configured
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('DEBUG - %(name)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)

# Thread-safe metering infrastructure
_metering_lock = threading.RLock()
_metering_queue = queue.Queue(maxsize=1000)  # Prevent memory issues
_client_cache = {}
_client_cache_lock = threading.RLock()


def _get_thread_safe_client():
    """Get a thread-safe Revenium client instance."""
    thread_id = threading.get_ident()

    with _client_cache_lock:
        if thread_id in _client_cache:
            return _client_cache[thread_id]

        # Import here to avoid circular imports and ensure proper initialization
        try:
            # Import the client module to get a fresh instance per thread
            import revenium_middleware

            # Use the pre-instantiated client instance from revenium_middleware
            thread_client = revenium_middleware.client
            _client_cache[thread_id] = thread_client
            logger.debug(f"Created thread-safe client for thread {thread_id}")
            return thread_client
        except Exception as e:
            logger.warning(f"Failed to create thread-safe client: {e}")
            return None


def _safe_run_async_in_thread(coro_func: Callable, *args, **kwargs):
    """Thread-safe wrapper for async operations with proper error handling."""
    try:
        from revenium_middleware import run_async_in_thread, shutdown_event

        if shutdown_event.is_set():
            logger.warning("Skipping async operation during shutdown")
            return None

        # Use a lock to prevent concurrent async operations from interfering
        with _metering_lock:
            thread = run_async_in_thread(coro_func(*args, **kwargs))
            logger.debug(f"Started thread-safe async operation: {thread}")
            return thread
    except Exception as e:
        logger.warning(f"Error in thread-safe async operation: {e}")
        return None


def extract_prompt_data_if_enabled(
    request_body: Optional[Dict[str, Any]],
    response: Any = None,
    accumulated_content: str = None
) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[bool]]:
    """
    Extract prompt data if capture is enabled.

    Args:
        request_body: Request body containing messages and system prompt
        response: API response object (for non-streaming)
        accumulated_content: Accumulated streaming content (for streaming)

    Returns:
        Tuple of (system_prompt, input_messages, output_response, prompts_truncated)
    """
    if not Config.CAPTURE_PROMPTS or not request_body:
        return None, None, None, None

    # Extract prompts from request
    prompt_data = extract_prompts_from_request(request_body)
    system_prompt = prompt_data.get('systemPrompt')
    input_messages = prompt_data.get('inputMessages')
    prompts_truncated = prompt_data.get('promptsTruncated', False)

    # Extract response content
    if accumulated_content is not None:
        # Streaming response
        response_data = extract_streaming_response_content(
            accumulated_content, prompts_truncated
        )
    elif response is not None:
        # Non-streaming response
        response_data = extract_response_content(response, prompts_truncated)
    else:
        response_data = {'outputResponse': None, 'promptsTruncated': prompts_truncated}

    output_response = response_data.get('outputResponse')
    prompts_truncated = response_data.get('promptsTruncated', prompts_truncated)

    logger.debug(
        f"Prompt capture - system_prompt: {bool(system_prompt)}, "
        f"input_messages: {bool(input_messages)}, "
        f"output_response: {bool(output_response)}, "
        f"truncated: {prompts_truncated}"
    )

    return system_prompt, input_messages, output_response, prompts_truncated


def _handle_bedrock_request(args, kwargs, usage_metadata, request_time_dt, request_time):  # pylint: disable=unused-argument
    """
    Handle a Bedrock request by converting parameters and invoking the Bedrock adapter.

    Returns:
        Anthropic-compatible response object
    """
    logger.debug("Handling Bedrock request")

    # Extract parameters from kwargs
    model = kwargs.get("model", "claude-3-sonnet-20240229")
    messages = kwargs.get("messages", [])

    # Create Bedrock payload - exclude 'messages' from kwargs to avoid conflict
    bedrock_kwargs = {k: v for k, v in kwargs.items() if k != "messages"}
    payload = create_bedrock_payload(messages, **bedrock_kwargs)

    # Invoke Bedrock
    text, input_tokens, output_tokens = bedrock_invoke(model, payload)

    # Create Anthropic-compatible response
    response = create_anthropic_response(
        text=text,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        model=model
    )

    # Calculate timing
    response_time_dt = datetime.datetime.now(datetime.timezone.utc)
    response_time = response_time_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    request_duration = (response_time_dt - request_time_dt).total_seconds() * 1000

    # Create metering call for Bedrock (pass kwargs for vision detection)
    _create_bedrock_metering_call(
        response, usage_metadata, request_time, response_time, request_duration, kwargs
    )

    return response


def _handle_bedrock_stream_request(args, kwargs, usage_metadata, request_time_dt, request_time):  # pylint: disable=unused-argument
    """
    Handle a Bedrock streaming request by creating a BedrockStreamWrapper.

    Returns:
        BedrockStreamWrapper: Stream wrapper compatible with Anthropic's interface
    """
    logger.debug("Handling Bedrock streaming request")

    # Extract parameters from kwargs
    model = kwargs.get("model", "claude-3-sonnet-20240229")
    messages = kwargs.get("messages", [])

    # Create Bedrock payload - exclude 'messages' from kwargs to avoid conflict
    bedrock_kwargs = {k: v for k, v in kwargs.items() if k != "messages"}
    payload = create_bedrock_payload(messages, **bedrock_kwargs)

    # Create and return BedrockStreamWrapper
    return BedrockStreamWrapper(
        model=model,
        payload=payload,
        messages=messages,
        region=kwargs.get("region"),
        usage_metadata=usage_metadata,
        request_time_dt=request_time_dt,
        request_time=request_time
    )


def _extract_trace_fields(usage_metadata, request_body=None):
    """
    Extract trace visualization fields from usage_metadata and environment variables.

    Args:
        usage_metadata: Dictionary containing usage metadata
        request_body: Optional request body for operation type detection

    Returns:
        Dictionary with trace visualization fields
    """
    # Get trace fields (usage_metadata takes precedence over environment variables)
    environment = usage_metadata.get('environment') or get_environment()
    region = usage_metadata.get('region') or get_region()
    credential_alias = (
        usage_metadata.get('credentialAlias') or
        usage_metadata.get('credential_alias') or
        get_credential_alias()
    )
    trace_type = (
        usage_metadata.get('traceType') or
        usage_metadata.get('trace_type') or
        get_trace_type()
    )
    trace_name = (
        usage_metadata.get('traceName') or
        usage_metadata.get('trace_name') or
        get_trace_name()
    )
    parent_transaction_id = (
        usage_metadata.get('parentTransactionId') or
        usage_metadata.get('parent_transaction_id') or
        get_parent_transaction_id()
    )
    transaction_name = (
        usage_metadata.get('transactionName') or
        usage_metadata.get('transaction_name') or
        get_transaction_name(usage_metadata)
    )
    retry_number = usage_metadata.get(
        'retryNumber',
        usage_metadata.get('retry_number', get_retry_number())
    )

    # Detect operation type and subtype
    request_body = request_body or {}
    operation_info = detect_operation_type(
        'anthropic', '/messages', request_body
    )
    operation_type = operation_info.get('operationType')
    operation_subtype = operation_info.get('operationSubtype')

    # Detect vision content in messages
    messages = request_body.get('messages', [])
    has_vision_content = detect_vision_content(messages)

    return {
        'environment': environment,
        'region': region,
        'credential_alias': credential_alias,
        'trace_type': trace_type,
        'trace_name': trace_name,
        'parent_transaction_id': parent_transaction_id,
        'transaction_name': transaction_name,
        'retry_number': retry_number,
        'operation_type': operation_type,
        'operation_subtype': operation_subtype,
        'has_vision_content': has_vision_content,
    }


def _create_bedrock_metering_call(response, usage_metadata, request_time, response_time, request_duration, request_kwargs=None):
    """Create a metering call for Bedrock usage."""

    # Get provider metadata
    provider_metadata = get_provider_metadata(Provider.BEDROCK)

    async def metering_call():
        try:
            from revenium_middleware import shutdown_event

            if shutdown_event.is_set():
                logger.warning("Skipping metering call during shutdown")
                return

            logger.debug("Metering call to Revenium for Bedrock completion %s", response.id)

            # Get thread-safe client
            client = _get_thread_safe_client()
            if not client:
                logger.warning("No thread-safe client available for Bedrock metering")
                return

            # Build subscriber object like Anthropic calls
            subscriber = {}
            if usage_metadata.get("subscriber_id"):
                subscriber["id"] = usage_metadata.get("subscriber_id")
            if usage_metadata.get("subscriber_email"):
                subscriber["email"] = usage_metadata.get("subscriber_email")
            if usage_metadata.get("subscriber_credential_name"):
                subscriber["credential"] = {
                    "name": usage_metadata.get("subscriber_credential_name"),
                    "value": usage_metadata.get("subscriber_credential")
                }

            # Extract trace visualization fields (pass request kwargs for vision detection)
            trace_fields = _extract_trace_fields(usage_metadata, request_kwargs)

            # Build extra_body for additional fields not in SDK
            extra_body = {}
            if trace_fields.get('has_vision_content'):
                extra_body['hasVisionContent'] = True

            # Extract organization and product names with backward compatibility
            organization_name, product_name = _extract_organization_and_product_names(usage_metadata)

            result = client.ai.create_completion(
                cache_creation_token_count=0,  # Bedrock doesn't provide cache info yet
                cache_read_token_count=0,
                input_token_cost=None,  # Backend calculates pricing
                output_token_cost=None,
                total_cost=None,
                output_token_count=response.usage.output_tokens,
                cost_type="AI",
                model=response.model,
                input_token_count=response.usage.input_tokens,
                provider=provider_metadata["provider"],
                model_source=provider_metadata["model_source"],
                reasoning_token_count=0,
                request_time=request_time,
                response_time=response_time,
                completion_start_time=response_time,
                request_duration=int(request_duration),
                time_to_first_token=int(request_duration),  # For non-streaming
                stop_reason="END",  # Simplified for MVP
                total_token_count=response.usage.total_tokens,
                transaction_id=response.id,
                trace_id=usage_metadata.get("trace_id"),
                task_type=usage_metadata.get("task_type"),
                subscriber=subscriber if subscriber else None,
                organization_name=organization_name,
                subscription_id=usage_metadata.get("subscription_id"),
                product_name=product_name,
                agent=usage_metadata.get("agent"),
                response_quality_score=usage_metadata.get("response_quality_score"),
                is_streamed=False,
                operation_type=trace_fields.get('operation_type', 'CHAT'),
                # Trace visualization fields
                environment=trace_fields.get('environment'),
                region=trace_fields.get('region'),
                credential_alias=trace_fields.get('credential_alias'),
                trace_type=trace_fields.get('trace_type'),
                trace_name=trace_fields.get('trace_name'),
                parent_transaction_id=trace_fields.get('parent_transaction_id'),
                transaction_name=trace_fields.get('transaction_name'),
                retry_number=trace_fields.get('retry_number'),
                operation_subtype=trace_fields.get('operation_subtype'),
                # Additional fields via extra_body
                extra_body=extra_body if extra_body else None,
            )
            logger.debug("Bedrock metering call result: %s", result)
        except Exception as e:
            from revenium_middleware import shutdown_event
            if not shutdown_event.is_set():
                logger.warning(f"Error in Bedrock metering call: {str(e)}")
                import traceback
                logger.warning(f"Traceback: {traceback.format_exc()}")

    thread = _safe_run_async_in_thread(metering_call)
    logger.debug("Bedrock metering thread started: %s", thread)


def extract_usage_metadata_and_timing(kwargs: dict, operation_name: str = "operation"):
    """
    Extract usage metadata from kwargs.
    Provides robust error handling for malformed metadata structures.

    Args:
        kwargs: The kwargs dict to extract from (will be modified)
        operation_name: Name of operation for logging (e.g., "create", "stream")

    Returns:
        tuple: (usage_metadata, request_time, request_time_dt)
    """
    # Extract API-level metadata from kwargs
    api_metadata = kwargs.pop("usage_metadata", {})

    # Validate and sanitize API-level metadata
    if not isinstance(api_metadata, dict):
        logger.warning(f"usage_metadata for {operation_name} should be a dict, got {type(api_metadata)}. Using empty dict.")
        api_metadata = {}

    # Merge with decorator metadata (API-level takes precedence)
    usage_metadata = merge_metadata(api_metadata)
    logger.debug(f"Merged decorator metadata for {operation_name}: {usage_metadata}")

    # Sanitize metadata structure (defensive programming)
    usage_metadata = _sanitize_metadata(usage_metadata, operation_name)

    # Create request timestamp
    request_time_dt = datetime.datetime.now(datetime.timezone.utc)
    request_time = request_time_dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    # Debug logging
    logger.debug(f"Usage metadata for {operation_name}: %s", usage_metadata)

    return usage_metadata, request_time, request_time_dt


def _sanitize_metadata(metadata: dict, operation_name: str, max_depth: int = 5, current_depth: int = 0) -> dict:
    """
    Sanitize metadata structure to prevent issues with deeply nested objects
    or problematic data types that could break metering calls.

    Args:
        metadata: The metadata dict to sanitize
        operation_name: Operation name for logging
        max_depth: Maximum allowed nesting depth
        current_depth: Current recursion depth

    Returns:
        dict: Sanitized metadata
    """
    if current_depth > max_depth:
        logger.warning(f"Metadata for {operation_name} exceeds maximum depth {max_depth}. Truncating.")
        return {}

    if not isinstance(metadata, dict):
        return {}

    sanitized = {}
    for key, value in metadata.items():
        # Ensure key is a string
        if not isinstance(key, str):
            key = str(key)

        # Sanitize value based on type
        if isinstance(value, dict):
            sanitized[key] = _sanitize_metadata(value, operation_name, max_depth, current_depth + 1)
        elif isinstance(value, (list, tuple)):
            # Convert lists/tuples to strings to avoid complex nested structures
            sanitized[key] = str(value)
        elif isinstance(value, (str, int, float, bool)):
            sanitized[key] = value
        elif value is None:
            sanitized[key] = None
        else:
            # Convert other types to string
            sanitized[key] = str(value)

    return sanitized


def _extract_organization_and_product_names(usage_metadata: dict) -> tuple:
    """
    Extract organization name and product name from usage metadata.

    Supports both new field names (organizationName, productName) and
    deprecated field names (organizationId, productId) for backward compatibility.
    New field names take precedence over deprecated ones.

    Precedence order (highest to lowest):
    1. organizationName / productName (camelCase - API-level, documented)
    2. organization_name / product_name (snake_case - alternative)
    3. organizationId / productId (camelCase - deprecated)
    4. organization_id / product_id (snake_case - deprecated)

    Args:
        usage_metadata: Dictionary containing usage metadata

    Returns:
        tuple: (organization_name, product_name)
    """
    # Extract organization name (support both new and deprecated field names)
    # Using 'or' logic to handle empty strings and fallback properly
    # Check camelCase first (API-level, documented format), then snake_case, then deprecated
    organization_name = (
        usage_metadata.get("organizationName")
        or usage_metadata.get("organization_name")
        or usage_metadata.get("organizationId")
        or usage_metadata.get("organization_id")
    )

    # Emit deprecation warning if using old field names
    if not (usage_metadata.get("organizationName") or usage_metadata.get("organization_name")):
        if usage_metadata.get("organizationId") or usage_metadata.get("organization_id"):
            logger.warning(
                "Fields 'organizationId' and 'organization_id' are deprecated. "
                "Use 'organizationName' instead. "
                "The old fields will be removed in a future version."
            )

    # Extract product name (support both new and deprecated field names)
    # Using 'or' logic to handle empty strings and fallback properly
    # Check camelCase first (API-level, documented format), then snake_case, then deprecated
    product_name = (
        usage_metadata.get("productName")
        or usage_metadata.get("product_name")
        or usage_metadata.get("productId")
        or usage_metadata.get("product_id")
    )

    # Emit deprecation warning if using old field names
    if not (usage_metadata.get("productName") or usage_metadata.get("product_name")):
        if usage_metadata.get("productId") or usage_metadata.get("product_id"):
            logger.warning(
                "Fields 'productId' and 'product_id' are deprecated. "
                "Use 'productName' instead. "
                "The old fields will be removed in a future version."
            )

    return organization_name, product_name


@wrapt.patch_function_wrapper('anthropic.resources.messages.messages', 'Messages.create')
def create_wrapper(wrapped, instance, args, kwargs):
    """
    Wraps the anthropic.ChatCompletion.create method to log token usage.
    Now supports both direct Anthropic API and AWS Bedrock routing.
    """
    logger.debug("Anthropic client.messages.create wrapper called: %s: %s", wrapped, args)

    # Extract usage metadata and timing using shared handler
    usage_metadata, request_time, request_time_dt = extract_usage_metadata_and_timing(kwargs, "create")

    # Check if Bedrock is disabled via environment variable
    if os.getenv("REVENIUM_BEDROCK_DISABLE") == "1":
        logger.debug("Bedrock support disabled via REVENIUM_BEDROCK_DISABLE")
        provider = Provider.ANTHROPIC
    else:
        # Detect provider based on client and parameters
        client_instance = getattr(instance, '_client', None) if instance else None
        base_url = kwargs.get('base_url', None)
        provider = detect_provider(client=client_instance, base_url=base_url)

    logger.debug(f"Detected provider: {provider}")

    # Route to appropriate handler
    if provider == Provider.BEDROCK:
        try:
            logger.debug("Routing to Bedrock handler")
            return _handle_bedrock_request(args, kwargs, usage_metadata, request_time_dt, request_time)
        except (BedrockValidationError, BedrockInvokeError) as e:
            logger.error(f"Bedrock request failed: {e}. Falling back to direct Anthropic API.")
            # Fall back to direct Anthropic API on Bedrock-specific errors
            provider = Provider.ANTHROPIC
        except ImportError as e:
            logger.error(f"Bedrock dependencies not available: {e}. Falling back to direct Anthropic API.")
            # Fall back to direct Anthropic API if boto3 not installed
            provider = Provider.ANTHROPIC
        except Exception as e:
            logger.error(f"Unexpected error in Bedrock handler: {e}. Falling back to direct Anthropic API.")
            # Fall back to direct Anthropic API on unexpected errors
            provider = Provider.ANTHROPIC

    # Handle direct Anthropic API (original logic)
    logger.debug("REVENIUM MIDDLEWARE: Calling client.messages.create with args: %s, kwargs: %s", args, kwargs)

    # Capture request kwargs for vision detection before calling wrapped
    # (need to preserve messages for detecting vision content in metering call)
    request_kwargs = dict(kwargs)

    response = wrapped(*args, **kwargs)
    logger.debug("REVENIUM MIDDLEWARE: Received response from client.messages.create: %s", response.id)
    logger.debug(
        "Anthropic client.messages.create response: %s",
        response)
    response_time_dt = datetime.datetime.now(datetime.timezone.utc)
    response_time = response_time_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    request_duration = (response_time_dt - request_time_dt).total_seconds() * 1000
    response_id = response.id

    prompt_tokens = response.usage.input_tokens
    completion_tokens = response.usage.output_tokens
    cache_creation_input_tokens = response.usage.cache_creation_input_tokens
    cache_read_input_tokens = response.usage.cache_read_input_tokens

    logger.debug(
        "Anthropic client.ai.create_completion token usage - prompt: %d, completion: %d, "
        "cache_creation_input_tokens: %d,cache_read_input_tokens: %d",
        prompt_tokens, completion_tokens, cache_creation_input_tokens, cache_read_input_tokens
    )

    anthropic_finish_reason = None
    if response.stop_reason:
        anthropic_finish_reason = response.stop_reason

    finish_reason_map = {
        "end_turn": "END",
        "tool_use": "END_SEQUENCE",
        "max_tokens": "TOKEN_LIMIT",
        "content_filter": "ERROR"
    }
    stop_reason = finish_reason_map.get(anthropic_finish_reason, "end_turn")  # type: ignore

    # Extract prompt data if capture is enabled
    (system_prompt, input_messages, output_response, prompts_truncated) = (
        extract_prompt_data_if_enabled(kwargs, response=response)
    )

    # Get provider metadata for metering
    provider_metadata = get_provider_metadata(provider)

    async def metering_call():
        try:
            from revenium_middleware import shutdown_event

            if shutdown_event.is_set():
                logger.warning("Skipping metering call during shutdown")
                return
            logger.debug("Metering call to Revenium for completion %s with usage_metadata: %s", response_id,
                         usage_metadata)

            # Get thread-safe client
            client = _get_thread_safe_client()
            if not client:
                logger.warning("No thread-safe client available for metering")
                return

            # Create subscriber object from usage metadata
            subscriber = extract_subscriber_from_metadata(usage_metadata)

            # Extract trace visualization fields (pass request kwargs for vision detection)
            trace_fields = _extract_trace_fields(usage_metadata, request_kwargs)

            # Build extra_body for additional fields not in SDK
            extra_body = {}
            if trace_fields.get('has_vision_content'):
                extra_body['hasVisionContent'] = True

            # Extract organization and product names with backward compatibility
            organization_name, product_name = _extract_organization_and_product_names(usage_metadata)

            result = client.ai.create_completion(
                cache_creation_token_count=cache_creation_input_tokens,
                cache_read_token_count=cache_read_input_tokens,
                input_token_cost=None,
                output_token_cost=None,
                total_cost=None,
                output_token_count=completion_tokens,
                cost_type="AI",
                model=response.model,
                input_token_count=prompt_tokens,
                provider=provider_metadata["provider"],
                model_source=provider_metadata["model_source"],
                reasoning_token_count=0,
                request_time=request_time,
                response_time=response_time,
                completion_start_time=response_time,
                request_duration=int(request_duration),
                time_to_first_token=int(request_duration),  # For non-streaming, use the full request duration
                stop_reason=stop_reason,
                total_token_count=prompt_tokens + completion_tokens,
                transaction_id=response_id,
                trace_id=usage_metadata.get("trace_id"),
                task_type=usage_metadata.get("task_type"),
                subscriber=subscriber if subscriber else None,
                organization_name=organization_name,
                subscription_id=usage_metadata.get("subscription_id"),
                product_name=product_name,
                agent=usage_metadata.get("agent"),
                response_quality_score=usage_metadata.get("response_quality_score"),
                is_streamed=False,
                operation_type=trace_fields.get('operation_type', 'CHAT'),
                middleware_source="PYTHON",
                # Trace visualization fields
                environment=trace_fields.get('environment'),
                region=trace_fields.get('region'),
                credential_alias=trace_fields.get('credential_alias'),
                trace_type=trace_fields.get('trace_type'),
                trace_name=trace_fields.get('trace_name'),
                parent_transaction_id=trace_fields.get('parent_transaction_id'),
                transaction_name=trace_fields.get('transaction_name'),
                retry_number=trace_fields.get('retry_number'),
                operation_subtype=trace_fields.get('operation_subtype'),
                # Prompt capture fields
                system_prompt=system_prompt,
                input_messages=input_messages,
                output_response=output_response,
                prompts_truncated=prompts_truncated,
                # Additional fields via extra_body (for vision detection)
                extra_body=extra_body if extra_body else None,
            )
            logger.debug("Metering call result: %s", result)
            # Treat any successful resource response as success; only warn on explicit failure
            success = False
            try:
                if result is None:
                    success = False
                elif hasattr(result, 'status_code'):
                    status_code = int(getattr(result, 'status_code', 0) or 0)
                    success = 200 <= status_code < 300
                elif hasattr(result, 'resource_type') or hasattr(result, 'resourceType') or hasattr(result, 'id'):
                    # Revenium SDK returns a resource object on success (e.g., MeteringResponseResource)
                    success = True
                else:
                    # Unknown shape but non-empty result; assume success
                    success = True
            except Exception:
                success = False

            if success:
                logger.debug("[REVENIUM SUCCESS] Metering call successful for transaction %s", response_id)
            else:
                logger.warning("[REVENIUM ERROR] Metering call did not return success for transaction %s: %s", response_id, result)

            # Print usage summary if enabled (async, fire-and-forget)
            # Print regardless of metering success - user should see API usage even if metering fails
            _emit_usage_summary(
                model=response.model,
                provider=provider_metadata["provider"],
                request_duration=request_duration,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                response_id=response_id,
                usage_metadata=usage_metadata,
            )
        except Exception as e:
            from revenium_middleware import shutdown_event
            if not shutdown_event.is_set():
                logger.warning(f"Error in metering call: {str(e)}")
                # Log the full traceback for better debugging
                import traceback
                logger.warning(f"Traceback: {traceback.format_exc()}")

    thread = _safe_run_async_in_thread(metering_call)
    logger.debug("Metering thread started: %s", thread)
    return response


@wrapt.patch_function_wrapper('anthropic.resources.messages.messages', 'Messages.stream')
def stream_wrapper(wrapped, instance, args, kwargs):
    """
    Wraps the anthropic.resources.messages.Messages.stream method to log token usage.
    Extracts usage data from the final message of the stream.

    Note: Bedrock streaming is not yet supported in MVP. Falls back to direct Anthropic API.
    """
    logger.debug("REVENIUM MIDDLEWARE: Intercepted client.messages.stream call - wrapper active")

    # Extract usage metadata and timing using shared handler
    usage_metadata, request_time, request_time_dt = extract_usage_metadata_and_timing(kwargs, "stream")

    # Check if this would be a Bedrock request
    if os.getenv("REVENIUM_BEDROCK_DISABLE") != "1":
        client_instance = getattr(instance, '_client', None) if instance else None
        base_url = kwargs.get('base_url', None)
        provider = detect_provider(client=client_instance, base_url=base_url)

        if provider == Provider.BEDROCK:
            try:
                logger.debug("Routing streaming request to Bedrock handler")
                return _handle_bedrock_stream_request(args, kwargs, usage_metadata, request_time_dt, request_time)
            except (BedrockValidationError, BedrockStreamError) as e:
                logger.error(f"Bedrock streaming request failed: {e}. Falling back to direct Anthropic API.")
                # Fall back to direct Anthropic API on Bedrock-specific errors
            except ImportError as e:
                logger.error(f"Bedrock dependencies not available: {e}. Falling back to direct Anthropic API.")
                # Fall back to direct Anthropic API if boto3 not installed
            except Exception as e:
                logger.error(f"Unexpected error in Bedrock streaming handler: {e}. Falling back to direct Anthropic API.")
                # Fall back to direct Anthropic API on unexpected errors

    logger.debug("REVENIUM MIDDLEWARE: Calling client.messages.stream with args: %s, kwargs: %s", args, kwargs)

    # Capture request kwargs for vision detection before calling wrapped
    # (need to preserve messages for detecting vision content in metering call)
    request_kwargs = dict(kwargs)

    stream = wrapped(*args, **kwargs)
    logger.debug("REVENIUM MIDDLEWARE: Received stream from client.messages.stream")

    # Create a wrapper for the stream that will capture the final message
    class StreamWrapper:
        def __init__(self, stream):
            self.stream = stream
            self.response_time_dt = None
            self.response_id = None
            self.collected_content = []
            self.final_message = None
            self.first_token_time = None
            self.request_start_time = time.time() * 1000  # Convert to milliseconds

        def __enter__(self):
            self.stream_context = self.stream.__enter__()
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            result = self.stream.__exit__(exc_type, exc_val, exc_tb)

            # Get the final message with usage information
            try:
                self.final_message = self.stream_context.get_final_message()
                self.response_time_dt = datetime.datetime.now(datetime.timezone.utc)
                self.response_time = self.response_time_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
                request_duration = (self.response_time_dt - request_time_dt).total_seconds() * 1000

                self.response_id = self.final_message.id

                prompt_tokens = self.final_message.usage.input_tokens
                completion_tokens = self.final_message.usage.output_tokens
                cache_creation_input_tokens = self.final_message.usage.cache_creation_input_tokens
                cache_read_input_tokens = self.final_message.usage.cache_read_input_tokens

                logger.debug(
                    "Anthropic client.messages.stream token usage - prompt: %d, completion: %d, "
                    "cache_creation_input_tokens: %d, cache_read_input_tokens: %d",
                    prompt_tokens, completion_tokens, cache_creation_input_tokens, cache_read_input_tokens
                )

                anthropic_finish_reason = None
                if self.final_message.stop_reason:
                    anthropic_finish_reason = self.final_message.stop_reason

                finish_reason_map = {
                    "end_turn": "END",
                    "tool_use": "END_SEQUENCE",
                    "max_tokens": "TOKEN_LIMIT",
                    "content_filter": "ERROR"
                }
                stop_reason = finish_reason_map.get(anthropic_finish_reason, "end_turn")  # type: ignore

                # Extract prompt data if capture is enabled (for streaming)
                accumulated_content = ''.join(self.collected_content)
                (system_prompt, input_messages, output_response, prompts_truncated) = (
                    extract_prompt_data_if_enabled(kwargs, accumulated_content=accumulated_content)
                )

                # For streaming, we always use Anthropic provider metadata since Bedrock streaming falls back
                provider_metadata = get_provider_metadata(Provider.ANTHROPIC)

                async def metering_call():
                    try:
                        from revenium_middleware import shutdown_event

                        if shutdown_event.is_set():
                            logger.warning("Skipping metering call during shutdown")
                            return
                        logger.debug("Metering call to Revenium for stream completion %s", self.response_id)

                        # Get thread-safe client
                        client = _get_thread_safe_client()
                        if not client:
                            logger.warning("No thread-safe client available for stream metering")
                            return

                        # Create subscriber object from usage metadata
                        subscriber = extract_subscriber_from_metadata(usage_metadata)

                        # Extract trace visualization fields (pass request kwargs for vision detection)
                        trace_fields = _extract_trace_fields(usage_metadata, request_kwargs)

                        # Build extra_body for additional fields not in SDK
                        extra_body = {}
                        if trace_fields.get('has_vision_content'):
                            extra_body['hasVisionContent'] = True

                        # Extract organization and product names with backward compatibility
                        organization_name, product_name = _extract_organization_and_product_names(usage_metadata)

                        result = client.ai.create_completion(
                            cache_creation_token_count=cache_creation_input_tokens,
                            cache_read_token_count=cache_read_input_tokens,
                            input_token_cost=None,
                            output_token_cost=None,
                            total_cost=None,
                            output_token_count=completion_tokens,
                            cost_type="AI",
                            model=self.final_message.model,
                            input_token_count=prompt_tokens,
                            provider=provider_metadata["provider"],
                            model_source=provider_metadata["model_source"],
                            reasoning_token_count=0,
                            request_time=request_time,
                            response_time=self.response_time,
                            completion_start_time=self.response_time,
                            request_duration=int(request_duration),
                            time_to_first_token=int(
                                self.first_token_time - self.request_start_time) if self.first_token_time else 0,
                            stop_reason=stop_reason,
                            total_token_count=prompt_tokens + completion_tokens,
                            transaction_id=self.response_id,
                            trace_id=usage_metadata.get("trace_id"),
                            task_type=usage_metadata.get("task_type"),
                            subscriber=subscriber if subscriber else None,
                            organization_name=organization_name,
                            subscription_id=usage_metadata.get("subscription_id"),
                            product_name=product_name,
                            agent=usage_metadata.get("agent"),
                            is_streamed=True,
                            operation_type=trace_fields.get('operation_type', 'CHAT'),
                            response_quality_score=usage_metadata.get("response_quality_score"),
                            middleware_source="PYTHON",
                            # Trace visualization fields
                            environment=trace_fields.get('environment'),
                            region=trace_fields.get('region'),
                            credential_alias=trace_fields.get('credential_alias'),
                            trace_type=trace_fields.get('trace_type'),
                            trace_name=trace_fields.get('trace_name'),
                            parent_transaction_id=trace_fields.get('parent_transaction_id'),
                            transaction_name=trace_fields.get('transaction_name'),
                            retry_number=trace_fields.get('retry_number'),
                            operation_subtype=trace_fields.get('operation_subtype'),
                            # Prompt capture fields
                            system_prompt=system_prompt,
                            input_messages=input_messages,
                            output_response=output_response,
                            prompts_truncated=prompts_truncated,
                            # Additional fields via extra_body (for vision detection)
                            extra_body=extra_body if extra_body else None,
                        )
                        logger.debug("Metering call result for stream: %s", result)
                        # Treat any successful resource response as success; only warn on explicit failure
                        success = False
                        try:
                            if result is None:
                                success = False
                            elif hasattr(result, 'status_code'):
                                status_code = int(getattr(result, 'status_code', 0) or 0)
                                success = 200 <= status_code < 300
                            elif hasattr(result, 'resource_type') or hasattr(result, 'resourceType') or hasattr(result, 'id'):
                                success = True
                            else:
                                success = True
                        except Exception:
                            success = False

                        if success:
                            logger.debug("[REVENIUM SUCCESS] Streaming metering call successful for transaction %s", self.response_id)
                        else:
                            logger.warning(
                                "[REVENIUM ERROR] Streaming metering call did not return success for transaction %s: %s",
                                self.response_id, result
                            )

                        # Print usage summary if enabled (async, fire-and-forget)
                        # Print regardless of metering success - user should see API usage even if metering fails
                        _emit_usage_summary(
                            model=self.final_message.model,
                            provider=provider_metadata["provider"],
                            request_duration=request_duration,
                            prompt_tokens=prompt_tokens,
                            completion_tokens=completion_tokens,
                            response_id=self.response_id,
                            usage_metadata=usage_metadata,
                        )
                    except Exception as e:
                        from revenium_middleware import shutdown_event
                        if not shutdown_event.is_set():
                            logger.warning(f"Error in metering call for stream: {str(e)}")
                            # Log the full traceback for better debugging
                            import traceback
                            logger.warning(f"Traceback: {traceback.format_exc()}")

                thread = _safe_run_async_in_thread(metering_call)
                logger.debug("Metering thread started for stream: %s", thread)

            except Exception as e:
                logger.warning(f"Error processing final message from stream: {str(e)}")
                import traceback
                logger.warning(f"Traceback: {traceback.format_exc()}")

            return result

        @property
        def text_stream(self):
            # Create a wrapper for the text_stream that doesn't consume it
            original_text_stream = self.stream_context.text_stream
            wrapper_self = self

            class TextStreamWrapper:
                def __iter__(self):
                    return self

                def __next__(self):
                    try:
                        chunk = next(original_text_stream)
                        # Record the time of the first token
                        if wrapper_self.first_token_time is None and chunk:
                            wrapper_self.first_token_time = time.time() * 1000  # Convert to milliseconds
                        return chunk
                    except StopIteration:
                        raise

            return TextStreamWrapper()

        def get_final_message(self):
            if self.final_message:
                return self.final_message
            return self.stream_context.get_final_message()

        def __iter__(self):
            return iter(self.stream_context)

        def __getattr__(self, name):
            return getattr(self.stream_context, name)

    return StreamWrapper(stream)


# Log middleware initialization
logger.debug("REVENIUM MIDDLEWARE: Anthropic middleware loaded and wrappers registered")
