import logging
import datetime
import wrapt
import types

logger = logging.getLogger("revenium_middleware.extension")

from revenium_middleware import client, run_async_in_thread, shutdown_event, merge_metadata
from revenium_middleware._core.subscriber import extract_subscriber_from_metadata
from .trace_fields import (
    get_environment,
    get_region,
    get_credential_alias,
    get_trace_type,
    get_trace_name,
    get_parent_transaction_id,
    get_transaction_name,
    get_retry_number,
    detect_operation_type
)


def add_transaction_id_to_response(response, transaction_id):
    """
    Add the Revenium transaction ID to an Ollama response object.

    This function adds the transaction ID as an attribute to the response
    object. The attribute can be accessed via response._revenium_transaction_id

    Args:
        response: The Ollama response object (ChatResponse or GenerateResponse)
        transaction_id: The transaction ID string to add
    """
    try:
        # Add as attribute (works with response.attribute access)
        # Ollama responses are Pydantic models, so we use setattr
        setattr(response, '_revenium_transaction_id', transaction_id)
        logger.debug(
            "Added transaction ID %s to response",
            transaction_id
        )
    except (TypeError, AttributeError) as e:
        # If attribute setting doesn't work, log a warning
        logger.warning(
            "Could not add transaction ID as attribute: %s",
            str(e)
        )


def extract_field_with_fallback(
    usage_metadata, new_snake, new_camel, old_snake, old_camel, field_label
):
    """
    Extract field with 4-level fallback and deprecation warning.

    Args:
        usage_metadata: Dictionary containing usage metadata
        new_snake: New snake_case field name (e.g., "organization_name")
        new_camel: New camelCase field name (e.g., "organizationName")
        old_snake: Old snake_case field name (e.g., "organization_id")
        old_camel: Old camelCase field name (e.g., "organizationId")
        field_label: Human-readable label for the field (e.g., "organization")

    Returns:
        The field value with precedence: new_snake > new_camel > old_snake > old_camel
    """
    # Extract with precedence: new_snake > new_camel > old_snake > old_camel
    value = usage_metadata.get(new_snake)
    if value is None:
        value = usage_metadata.get(new_camel)
    if value is None:
        value = usage_metadata.get(old_snake)
    if value is None:
        value = usage_metadata.get(old_camel)

    # Log deprecation warning if old fields are used without new fields
    if usage_metadata.get(old_snake) or usage_metadata.get(old_camel):
        if not (usage_metadata.get(new_snake) or usage_metadata.get(new_camel)):
            logger.warning(
                f"Fields '{old_camel}' and '{old_snake}' are deprecated. "
                f"Use '{new_camel}' or '{new_snake}' instead. "
                "The old fields will be removed in a future version."
            )

    return value



# extract_subscriber_from_metadata is imported from _core.subscriber


@wrapt.patch_function_wrapper('ollama', 'chat')
def chat_wrapper(wrapped, _, args, kwargs):
    """
    Wraps the ollama.chat method to log token usage.
    Handles both streaming and non-streaming responses.
    """
    logger.debug("Ollama chat wrapper called")

    # Extract API-level metadata from kwargs
    api_metadata = kwargs.pop("usage_metadata", {}) if "usage_metadata" in kwargs else {}

    # Merge with decorator metadata (API-level takes precedence)
    usage_metadata = merge_metadata(api_metadata)

    is_streaming = kwargs.get("stream", False)

    # If streaming is enabled, add stream_options to include usage information
    # if is_streaming and "stream_options" not in kwargs:
    #     kwargs["stream_options"] = {"include_usage": True}
    # elif is_streaming and isinstance(kwargs.get("stream_options"), dict):
    #     kwargs["stream_options"]["include_usage"] = True

    request_time_dt = datetime.datetime.now(datetime.timezone.utc)
    # Generate transaction ID using the same timestamp for consistency
    transaction_id = f"ollama-{request_time_dt.timestamp()}"

    logger.debug(f"Calling chat function with args: {args}, kwargs: {kwargs}")

    response = wrapped(*args, **kwargs)

    # Check if response is a generator (streaming response)
    if is_streaming and isinstance(response, types.GeneratorType):
        return handle_streaming_response(
            response, request_time_dt, usage_metadata,
            transaction_id, 'chat', kwargs
        )
    else:
        # Handle non-streaming response
        logger.debug("Ollama chat response: %s", response)

        # Add transaction ID to response object
        add_transaction_id_to_response(response, transaction_id)

        handle_response(
            response, request_time_dt, usage_metadata,
            False, transaction_id, 'chat', kwargs
        )
        return response


@wrapt.patch_function_wrapper('ollama', 'generate')
def generate_wrapper(wrapped, _, args, kwargs):
    """
    Wraps the ollama.generate method to log token usage.
    Handles both streaming and non-streaming responses.
    """
    logger.debug("Ollama generate wrapper called")

    # Extract API-level metadata from kwargs
    api_metadata = kwargs.pop("usage_metadata", {}) if "usage_metadata" in kwargs else {}

    # Merge with decorator metadata (API-level takes precedence)
    usage_metadata = merge_metadata(api_metadata)

    is_streaming = kwargs.get("stream", False)

    # Note: ollama.generate() doesn't support stream_options parameter
    # Token usage is included by default in the final chunk

    request_time_dt = datetime.datetime.now(datetime.timezone.utc)
    # Generate transaction ID using the same timestamp for consistency
    transaction_id = f"ollama-{request_time_dt.timestamp()}"

    logger.debug(f"Calling generate function with args: {args}, kwargs: {kwargs}")

    response = wrapped(*args, **kwargs)

    # Check if response is a generator (streaming response)
    if is_streaming and isinstance(response, types.GeneratorType):
        return handle_streaming_response(
            response, request_time_dt, usage_metadata,
            transaction_id, 'generate', kwargs
        )
    else:
        # Handle non-streaming response
        logger.debug("Ollama generate response: %s", response)

        # Add transaction ID to response object
        add_transaction_id_to_response(response, transaction_id)

        handle_response(
            response, request_time_dt, usage_metadata,
            False, transaction_id, 'generate', kwargs
        )
        return response


@wrapt.patch_function_wrapper('ollama', 'embed')
def embed_wrapper(wrapped, _, args, kwargs):
    """
    Wraps the ollama.embed method to log token usage for embeddings.
    Handles both single and batch embedding requests.
    """
    logger.debug("Ollama embed wrapper called")
    usage_metadata = kwargs.pop("usage_metadata", {}) if "usage_metadata" in kwargs else {}

    request_time_dt = datetime.datetime.now(datetime.timezone.utc)
    # Generate transaction ID using the same timestamp for consistency
    transaction_id = f"ollama-{request_time_dt.timestamp()}"

    logger.debug(f"Calling embed function with args: {args}, kwargs: {kwargs}")

    response = wrapped(*args, **kwargs)

    # Handle embeddings response (embeddings are not streamed)
    logger.debug("Ollama embed response: %s", response)

    # Add transaction ID to response object
    add_transaction_id_to_response(response, transaction_id)

    # Handle embeddings response - embeddings only have input tokens, no output tokens
    handle_embeddings_response(
        response, request_time_dt, usage_metadata,
        transaction_id, kwargs
    )
    return response


def handle_streaming_response(
    generator,
    request_time_dt,
    usage_metadata,
    transaction_id,
    endpoint,
    request_kwargs
):
    """
    Handles streaming responses by collecting all chunks and processing the
    final state. Returns a new generator that yields the same chunks with
    transaction IDs added.

    Args:
        generator: The original response generator
        request_time_dt: The request timestamp
        usage_metadata: Metadata for metering
        transaction_id: The transaction ID to add to responses
        endpoint: The endpoint being called ('chat', 'generate', etc.)
        request_kwargs: The request kwargs for operation type detection
    """
    chunks = []
    final_response = None

    def wrapped_generator():
        nonlocal final_response

        # Collect all chunks and add transaction ID to each
        for chunk in generator:
            chunks.append(chunk)
            # Add transaction ID to each chunk
            add_transaction_id_to_response(chunk, transaction_id)
            yield chunk

        # After all chunks are processed, construct the final response
        if chunks:
            # The last chunk should contain the complete response data
            final_response = chunks[-1]
            handle_response(
                final_response,
                request_time_dt,
                usage_metadata,
                True,
                transaction_id,
                endpoint,
                request_kwargs
            )

    return wrapped_generator()


def handle_response(
    response,
    request_time_dt,
    usage_metadata,
    is_streaming,
    transaction_id,
    endpoint,
    request_kwargs
):
    """
    Process a complete response (either streaming or non-streaming) and
    send metering data.

    Args:
        response: The Ollama response object
        request_time_dt: The request timestamp
        usage_metadata: Metadata for metering
        is_streaming: Whether this is a streaming response
        transaction_id: The transaction ID for this request
        endpoint: The endpoint being called ('chat', 'generate', etc.)
        request_kwargs: The request kwargs for operation type detection
    """

    async def metering_call():
        response_time_dt = datetime.datetime.now(datetime.timezone.utc)
        response_time = response_time_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        request_duration = (
            (response_time_dt - request_time_dt).total_seconds() * 1000
        )
        request_time = request_time_dt.strftime("%Y-%m-%dT%H:%M:%SZ")

        # Use the provided transaction ID
        response_id = transaction_id

        # Extract token counts from Ollama response
        prompt_tokens = getattr(response, 'prompt_eval_count', 0)
        completion_tokens = getattr(response, 'eval_count', 0)
        total_tokens = prompt_tokens + completion_tokens
        cached_tokens = 0  # Ollama doesn't provide cached tokens info

        logger.debug(
            "Ollama chat token usage - prompt: %d, completion: %d, total: %d",
            prompt_tokens, completion_tokens, total_tokens
        )

        ollama_finish_reason = getattr(response, 'done_reason', None)

        finish_reason_map = {
            "stop": "END",
            "length": "TOKEN_LIMIT",
            "error": "ERROR",
            "cancelled": "CANCELLED",  # British spelling
            "canceled": "CANCELLED",   # American spelling (Go standard library uses this)
            "tool_calls": "END_SEQUENCE"
        }
        stop_reason = finish_reason_map.get(ollama_finish_reason, "END")  # type: ignore

        try:
            if shutdown_event.is_set():
                logger.warning("Skipping metering call during shutdown")
                return
            logger.debug("Metering call to Revenium for completion %s", response_id)

            # Create subscriber object from usage metadata
            subscriber = extract_subscriber_from_metadata(usage_metadata)

            # Detect operation type
            operation_type = detect_operation_type(endpoint, request_kwargs)

            # Capture trace visualization fields
            environment = get_environment()
            region = get_region()
            credential_alias = get_credential_alias()
            trace_type = get_trace_type()
            trace_name = get_trace_name()
            parent_transaction_id = get_parent_transaction_id()
            transaction_name = get_transaction_name(usage_metadata)
            retry_number = get_retry_number()

            # Extract organization and product names with deprecation warnings
            organization_name = extract_field_with_fallback(
                usage_metadata,
                "organization_name",
                "organizationName",
                "organization_id",
                "organizationId",
                "organization",
            )
            product_name = extract_field_with_fallback(
                usage_metadata,
                "product_name",
                "productName",
                "product_id",
                "productId",
                "product",
            )

            # Prepare arguments for create_completion
            completion_args = {
                "cache_creation_token_count": cached_tokens,
                "cache_read_token_count": 0,
                "input_token_cost": None,
                "output_token_cost": None,
                "total_cost": None,
                "output_token_count": completion_tokens,
                "cost_type": "AI",
                "model": getattr(response, 'model', 'ollama-model'),
                "input_token_count": prompt_tokens,
                "provider": "OLLAMA",
                "model_source": "OLLAMA",
                "reasoning_token_count": 0,
                "request_time": request_time,
                "response_time": response_time,
                "completion_start_time": response_time,
                "request_duration": int(request_duration),
                "stop_reason": stop_reason,
                "total_token_count": total_tokens,
                "transaction_id": response_id,
                "trace_id": usage_metadata.get("trace_id"),
                "task_type": usage_metadata.get("task_type"),
                "subscriber": subscriber if subscriber else None,
                "organization_name": organization_name,
                "subscription_id": usage_metadata.get("subscription_id"),
                "product_name": product_name,
                "agent": usage_metadata.get("agent"),
                "response_quality_score": usage_metadata.get("response_quality_score"),
                "is_streamed": is_streaming,
                "middleware_source": "PYTHON",
                # Trace visualization fields
                "operation_type": operation_type,
                "environment": environment,
                "region": region,
                "credential_alias": credential_alias,
                "trace_type": trace_type,
                "trace_name": trace_name,
                "parent_transaction_id": parent_transaction_id,
                "transaction_name": transaction_name,
                "retry_number": retry_number
            }

            # Log the arguments at debug level
            logger.debug("Arguments for create_completion: %s", completion_args)

            # The client.ai.create_completion method is not async, so don't use await
            result = client.ai.create_completion(**completion_args)
            logger.debug("Metering call result: %s", result)
        except Exception as e:
            if not shutdown_event.is_set():
                logger.warning(f"Error in metering call: {str(e)}")
                # Log the full traceback for better debugging
                import traceback
                logger.warning(f"Traceback: {traceback.format_exc()}")

    thread = run_async_in_thread(metering_call())
    logger.debug("Metering thread started: %s", thread)


def handle_embeddings_response(
    response,
    request_time_dt,
    usage_metadata,
    transaction_id,
    request_kwargs
):
    """
    Process an embeddings response and send metering data.

    Args:
        response: The Ollama embeddings response object
        request_time_dt: The request timestamp
        usage_metadata: Metadata for metering
        transaction_id: The transaction ID for this request
        request_kwargs: The request kwargs for operation type detection
    """

    async def metering_call():
        response_time_dt = datetime.datetime.now(datetime.timezone.utc)
        response_time = response_time_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        request_duration = (
            (response_time_dt - request_time_dt).total_seconds() * 1000
        )
        request_time = request_time_dt.strftime("%Y-%m-%dT%H:%M:%SZ")

        # Use the provided transaction ID
        response_id = transaction_id

        # Extract token counts from Ollama embeddings response
        # Embeddings only have input tokens (prompt_eval_count), no output tokens
        prompt_tokens = getattr(response, 'prompt_eval_count', 0)
        completion_tokens = 0  # Embeddings don't generate output tokens
        total_tokens = prompt_tokens
        cached_tokens = 0  # Ollama doesn't provide cached tokens info

        logger.debug(
            "Ollama embeddings token usage - prompt: %d, total: %d",
            prompt_tokens, total_tokens
        )

        # Embeddings always complete successfully (no finish reason)
        stop_reason = "END"

        try:
            if shutdown_event.is_set():
                logger.warning("Skipping metering call during shutdown")
                return
            logger.debug("Metering call to Revenium for embeddings %s", response_id)

            # Create subscriber object from usage metadata
            subscriber = extract_subscriber_from_metadata(usage_metadata)

            # Detect operation type - should be 'EMBED' for embeddings
            operation_type = detect_operation_type('embed', request_kwargs)

            # Capture trace visualization fields
            environment = get_environment()
            region = get_region()
            credential_alias = get_credential_alias()
            trace_type = get_trace_type()
            trace_name = get_trace_name()
            parent_transaction_id = get_parent_transaction_id()
            transaction_name = get_transaction_name(usage_metadata)
            retry_number = get_retry_number()

            # Extract organization and product names with deprecation warnings
            organization_name = extract_field_with_fallback(
                usage_metadata,
                "organization_name",
                "organizationName",
                "organization_id",
                "organizationId",
                "organization",
            )
            product_name = extract_field_with_fallback(
                usage_metadata,
                "product_name",
                "productName",
                "product_id",
                "productId",
                "product",
            )

            # Prepare arguments for create_completion
            completion_args = {
                "cache_creation_token_count": cached_tokens,
                "cache_read_token_count": 0,
                "input_token_cost": None,
                "output_token_cost": None,
                "total_cost": None,
                "output_token_count": completion_tokens,
                "cost_type": "AI",
                "model": getattr(response, 'model', 'ollama-model'),
                "input_token_count": prompt_tokens,
                "provider": "OLLAMA",
                "model_source": "OLLAMA",
                "reasoning_token_count": 0,
                "request_time": request_time,
                "response_time": response_time,
                "completion_start_time": response_time,
                "request_duration": int(request_duration),
                "stop_reason": stop_reason,
                "total_token_count": total_tokens,
                "transaction_id": response_id,
                "trace_id": usage_metadata.get("trace_id"),
                "task_type": usage_metadata.get("task_type"),
                "subscriber": subscriber if subscriber else None,
                "organization_name": organization_name,
                "subscription_id": usage_metadata.get("subscription_id"),
                "product_name": product_name,
                "agent": usage_metadata.get("agent"),
                "response_quality_score": usage_metadata.get("response_quality_score"),
                "is_streamed": False,  # Embeddings are never streamed
                "middleware_source": "PYTHON",
                # Trace visualization fields
                "operation_type": operation_type,
                "environment": environment,
                "region": region,
                "credential_alias": credential_alias,
                "trace_type": trace_type,
                "trace_name": trace_name,
                "parent_transaction_id": parent_transaction_id,
                "transaction_name": transaction_name,
                "retry_number": retry_number
            }

            # Log the arguments at debug level
            logger.debug("Arguments for create_completion: %s", completion_args)

            # The client.ai.create_completion method is not async, so don't use await
            result = client.ai.create_completion(**completion_args)
            logger.debug("Metering call result: %s", result)
        except Exception as e:
            if not shutdown_event.is_set():
                logger.warning(f"Error in metering call: {str(e)}")
                # Log the full traceback for better debugging
                import traceback
                logger.warning(f"Traceback: {traceback.format_exc()}")

    thread = run_async_in_thread(metering_call())
    logger.debug("Metering thread started: %s", thread)
