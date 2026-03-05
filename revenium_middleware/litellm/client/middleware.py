import wrapt
import logging
import datetime
from revenium_middleware import client, run_async_in_thread, shutdown_event
from revenium_middleware._core.subscriber import extract_subscriber_from_metadata
from .context import metadata_context
from .hooks import execute_metadata_hooks
from . import trace_fields
from .summary_printer import print_usage_summary
from .config import get_api_key

logger = logging.getLogger("revenium_middleware.extension")


@wrapt.patch_function_wrapper('litellm', 'completion')
def completion_wrapper(wrapped, _, args, kwargs):
    """
    Wraps the litellm.completion method to log token usage.
    Handles both streaming and non-streaming responses.

    Metadata is collected from two sources:
    1. Context metadata (set via metadata_context API)
    2. Explicit usage_metadata kwarg

    Explicit kwargs take precedence over context metadata.
    """
    logger.debug("LiteLLM completion wrapper called")

    # Get context metadata first
    context_metadata = metadata_context.get()

    # Get explicit metadata from kwargs (takes precedence)
    explicit_metadata = kwargs.pop("usage_metadata", {}) if "usage_metadata" in kwargs else {}

    # Merge: context metadata as base, explicit metadata overrides
    usage_metadata = {**context_metadata, **explicit_metadata}

    # Execute registered hooks to allow modification/enrichment
    try:
        usage_metadata = execute_metadata_hooks(usage_metadata)
    except Exception as e:
        logger.error(f"Error executing metadata hooks: {e}. Continuing with unmodified metadata.")
        # Continue with the metadata we have - don't let hook errors break the middleware

    logger.debug("Usage metadata (merged from context and kwargs, after hooks): {}".format(usage_metadata))

    request_time_dt = datetime.datetime.now(datetime.timezone.utc)
    logger.debug(f"Calling chat function with args: {args}, kwargs: {kwargs}")

    is_streaming = kwargs.get("stream", False)
    logger.debug(f"is_streaming: {is_streaming}")

    response = wrapped(*args, **kwargs)
    if is_streaming:
        return handle_streaming_response(response, request_time_dt, usage_metadata)
    else:
        return handle_response(response, request_time_dt, usage_metadata, False)


def handle_streaming_response(generator, request_time_dt, usage_metadata):
    """
    Handles streaming responses by collecting all chunks and processing the final state.
    Returns a new generator that yields the same chunks.
    """
    chunks = []
    final_response = None

    def wrapped_generator():
        nonlocal final_response

        # Collect all chunks
        for chunk in generator:
            chunks.append(chunk)
            yield chunk

        # After all chunks are processed, construct the final response
        if chunks:
            # The last chunk should contain the complete response data
            final_response = chunks[-1]
            handle_response(final_response, request_time_dt, usage_metadata, True)

    return wrapped_generator()


def handle_response(response, request_time_dt, usage_metadata, is_streaming):
    """
    Process a complete response (either streaming or non-streaming) and send metering data.
    Returns the original response.
    """

    async def metering_call():
        response_time_dt = datetime.datetime.now(datetime.timezone.utc)
        response_time = response_time_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        request_duration = (response_time_dt - request_time_dt).total_seconds() * 1000
        request_time = request_time_dt.strftime("%Y-%m-%dT%H:%M:%SZ")

        # Generate a unique ID if not present in response
        response_id = getattr(response, 'id', f"litellm_client-{datetime.datetime.now().timestamp()}")

        # Extract token counts from LiteLLM response
        prompt_tokens = getattr(response.usage, 'prompt_tokens', 0)
        completion_tokens = getattr(response.usage, 'completion_tokens', 0)
        cached_tokens = getattr(response.usage, 'cached_tokens', 0)
        total_tokens = prompt_tokens + completion_tokens + cached_tokens

        logger.debug(
            "LiteLLM completion token usage - prompt: %d, completion: %d, cached: %d, total: %d",
            prompt_tokens, completion_tokens, cached_tokens, total_tokens
        )

        finish_reason = getattr(response, 'finish_reason', None)

        finish_reason_map = {
            "stop": "END",
            "length": "TOKEN_LIMIT",
            "error": "ERROR",
            "cancelled": "CANCELLED",
            "tool_calls": "END_SEQUENCE",
            "function_calls": "END_SEQUENCE"
        }

        stop_reason = finish_reason_map.get(finish_reason, "END")  # type: ignore
        try:
            if shutdown_event.is_set():
                logger.warning("Skipping metering call during shutdown")
                return
            logger.debug("Metering call to Revenium for completion %s", response_id)

            # Create subscriber object from usage metadata
            subscriber = extract_subscriber_from_metadata(usage_metadata)

            # Prepare arguments for create_completion
            # Extract organization name (support both old and new field names)
            # Priority: organization_name > organizationName > organization_id > organizationId
            organization_name = usage_metadata.get("organization_name")
            if organization_name is None:
                organization_name = usage_metadata.get("organizationName")
            if organization_name is None:
                # Fallback to deprecated field names
                organization_name = usage_metadata.get("organization_id")
            if organization_name is None:
                organization_name = usage_metadata.get("organizationId")

            # Extract product name (support both old and new field names)
            # Priority: product_name > productName > product_id > productId
            product_name = usage_metadata.get("product_name")
            if product_name is None:
                product_name = usage_metadata.get("productName")
            if product_name is None:
                # Fallback to deprecated field names
                product_name = usage_metadata.get("product_id")
            if product_name is None:
                product_name = usage_metadata.get("productId")

            # Log deprecation warning if old fields are used
            # Note: We check for old field names only when new field names are NOT present
            # The decorators now set organization_name/product_name, so this warning
            # only fires for users who directly use the deprecated field names
            if usage_metadata.get("organization_id") or usage_metadata.get("organizationId"):
                if not (usage_metadata.get("organization_name") or usage_metadata.get("organizationName")):
                    logger.warning(
                        "Fields 'organization_id' and 'organizationId' are deprecated. "
                        "Use 'organization_name' or 'organizationName' instead. "
                        "The old fields will be removed in a future version."
                    )

            if usage_metadata.get("product_id") or usage_metadata.get("productId"):
                if not (usage_metadata.get("product_name") or usage_metadata.get("productName")):
                    logger.warning(
                        "Fields 'product_id' and 'productId' are deprecated. "
                        "Use 'product_name' or 'productName' instead. "
                        "The old fields will be removed in a future version."
                    )

            completion_args = {
                "cache_creation_token_count": cached_tokens,
                "cache_read_token_count": 0,
                "input_token_cost": None,
                "output_token_cost": None,
                "total_cost": None,
                "output_token_count": completion_tokens,
                "cost_type": "AI",
                "model": getattr(response, 'model', 'litellm-model'),
                "input_token_count": prompt_tokens,
                "provider": "LITELLM",
                "model_source": "LITELLM",
                "reasoning_token_count": 0,
                "request_time": request_time,
                "response_time": response_time,
                "completion_start_time": response_time,
                "request_duration": int(request_duration),
                "stop_reason": stop_reason,
                "total_token_count": total_tokens,
                "transaction_id": response_id,
                # Support both snake_case and camelCase for trace_id
                "trace_id": usage_metadata.get("trace_id") or usage_metadata.get("traceId"),
                "task_type": usage_metadata.get("task_type"),
                "subscriber": subscriber if subscriber else None,
                "organization_name": organization_name,
                "subscription_id": usage_metadata.get("subscription_id"),
                "product_name": product_name,
                "agent": usage_metadata.get("agent"),
                "response_quality_score": usage_metadata.get("response_quality_score"),
                "is_streamed": is_streaming,
                "operation_type": "CHAT",
                "system_fingerprint": getattr(response, 'system_fingerprint', None),
                "middleware_source": "PYTHON"
            }

            # Add trace visualization fields (v0.3.0+)
            # These fields support both environment variables and usage_metadata parameters
            # Priority: usage_metadata > environment variable

            # Environment field
            environment = (
                usage_metadata.get("environment") or
                trace_fields.get_environment()
            )
            if environment:
                completion_args["environment"] = environment

            # Region field
            region = (
                usage_metadata.get("region") or
                trace_fields.get_region()
            )
            if region:
                completion_args["region"] = region

            # Credential alias field
            credential_alias = (
                usage_metadata.get("credential_alias") or
                usage_metadata.get("credentialAlias") or
                trace_fields.get_credential_alias()
            )
            if credential_alias:
                completion_args["credential_alias"] = credential_alias

            # Trace type field (with validation)
            trace_type = (
                usage_metadata.get("trace_type") or
                usage_metadata.get("traceType") or
                trace_fields.get_trace_type()
            )
            if trace_type:
                if usage_metadata.get("trace_type") or usage_metadata.get("traceType"):
                    trace_type = trace_fields.validate_trace_type(trace_type)
                if trace_type:
                    completion_args["trace_type"] = trace_type

            # Trace name field (with validation)
            trace_name = (
                usage_metadata.get("trace_name") or
                usage_metadata.get("traceName") or
                trace_fields.get_trace_name()
            )
            if trace_name:
                if usage_metadata.get("trace_name") or usage_metadata.get("traceName"):
                    trace_name = trace_fields.validate_trace_name(trace_name)
                if trace_name:
                    completion_args["trace_name"] = trace_name

            # Parent transaction ID field
            parent_transaction_id = (
                usage_metadata.get("parent_transaction_id") or
                usage_metadata.get("parentTransactionId") or
                trace_fields.get_parent_transaction_id()
            )
            if parent_transaction_id:
                completion_args["parent_transaction_id"] = parent_transaction_id

            # Transaction name field (with fallback to task_type)
            transaction_name = trace_fields.get_transaction_name(usage_metadata)
            if transaction_name:
                completion_args["transaction_name"] = transaction_name

            # Retry number field
            retry_number = trace_fields.get_retry_number()
            if retry_number > 0:
                completion_args["retry_number"] = retry_number

            # Log the arguments at debug level
            logger.debug("Calling client.ai.create_completion with args: %s", completion_args)

            # The client.ai.create_completion method is not async, so don't use await
            result = client.ai.create_completion(**completion_args)
            logger.debug("Metering call result: %s", result)

            # Print usage summary if enabled (fire-and-forget)
            # Summary can still show token/duration metrics even without API key
            revenium_api_key = get_api_key()

            # Support both snake_case and camelCase for trace_id
            trace_id = usage_metadata.get("trace_id")
            if trace_id is None:
                trace_id = usage_metadata.get("traceId")

            print_usage_summary(
                model=completion_args.get("model", "unknown"),
                provider=completion_args.get("provider", "LITELLM"),
                request_duration=request_duration,
                input_token_count=prompt_tokens,
                output_token_count=completion_tokens,
                total_token_count=total_tokens,
                transaction_id=response_id,
                trace_id=trace_id,
                revenium_api_key=revenium_api_key,
            )
        except Exception as e:
            if not shutdown_event.is_set():
                logger.warning(f"Error in metering call: {str(e)}")
                # Log the full traceback for better debugging
                import traceback
                logger.warning(f"Traceback: {traceback.format_exc()}")

    logger.debug("Handling LiteLLM response: {}".format(response))
    thread = run_async_in_thread(metering_call())
    logger.debug("Metering thread started: %s", thread)
    
    # Return the original response
    return response
