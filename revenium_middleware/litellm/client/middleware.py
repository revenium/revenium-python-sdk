import wrapt
import logging
import datetime
from revenium_middleware import client, get_client, run_async_in_thread, shutdown_event
from revenium_middleware._core.subscriber import extract_subscriber_from_metadata
from revenium_middleware._core.fields import extract_org_and_product, extract_common_metadata, extract_agentic_job_fields, merge_extra_body
from revenium_middleware._core.config import is_selective_metering_enabled
from revenium_middleware._core.context import is_inside_decorated_function
from revenium_middleware._core import submit_ai_event
from revenium_middleware._core.log_sanitize import sanitize_for_logging
from revenium_middleware._core.patch_registry import register_patch
from .context import metadata_context
from .hooks import execute_metadata_hooks
from . import trace_fields

logger = logging.getLogger("revenium_middleware.extension")


if register_patch("litellm.completion"):
    @wrapt.patch_function_wrapper('litellm', 'completion')
    def completion_wrapper(wrapped, _, args, kwargs):
        if is_selective_metering_enabled() and not is_inside_decorated_function():
            return wrapped(*args, **kwargs)

        logger.debug("LiteLLM completion wrapper called")

        context_metadata = metadata_context.get()
        explicit_metadata = kwargs.pop("usage_metadata", {}) if "usage_metadata" in kwargs else {}
        usage_metadata = {**context_metadata, **explicit_metadata}

        try:
            usage_metadata = execute_metadata_hooks(usage_metadata)
        except Exception as e:
            logger.error(f"Error executing metadata hooks: {e}. Continuing with unmodified metadata.")

        logger.debug("Usage metadata (merged from context and kwargs, after hooks): {}".format(usage_metadata))

        request_time_dt = datetime.datetime.now(datetime.timezone.utc)
        logger.debug(
            "Calling chat function with args: %s, kwargs: %s",
            sanitize_for_logging(args),
            sanitize_for_logging(kwargs),
        )

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

        # The finally block also runs on GeneratorExit when the caller breaks
        # out of the loop or abandons the generator, so partial
        # usage from an interrupted stream is still metered.
        try:
            for chunk in generator:
                chunks.append(chunk)
                yield chunk
        finally:
            if chunks:
                try:
                    for chunk in reversed(chunks):
                        if hasattr(chunk, 'usage') and chunk.usage is not None:
                            final_response = chunk
                            break
                    if final_response is None:
                        final_response = chunks[-1]
                    handle_response(final_response, request_time_dt, usage_metadata, True)
                except Exception as e:
                    logger.warning("Error metering interrupted/completed stream: %s", e)

    return wrapped_generator()


def handle_response(response, request_time_dt, usage_metadata, is_streaming):
    """
    Process a complete response (either streaming or non-streaming) and send metering data.
    Returns the original response.
    """
    if get_client() is None:
        return response  # metering disabled (no API key configured)

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
        # OpenAI-style responses report cache reads under
        # prompt_tokens_details; LiteLLM's Anthropic passthrough exposes
        # them as top-level cache_read/cache_creation_input_tokens.
        prompt_details = getattr(response.usage, 'prompt_tokens_details', None)
        cache_read_tokens = getattr(prompt_details, 'cached_tokens', 0) or 0
        if not cache_read_tokens:
            cache_read_tokens = getattr(response.usage, 'cache_read_input_tokens', 0) or 0
        cache_creation_tokens = getattr(response.usage, 'cache_creation_input_tokens', 0) or 0
        total_tokens = prompt_tokens + completion_tokens

        logger.debug(
            "LiteLLM completion token usage - prompt: %d, completion: %d, "
            "cache read: %d, cache creation: %d, total: %d",
            prompt_tokens, completion_tokens, cache_read_tokens,
            cache_creation_tokens, total_tokens
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

            organization_name, product_name = extract_org_and_product(usage_metadata)
            meta = extract_common_metadata(usage_metadata)
            agentic_fields = extract_agentic_job_fields(usage_metadata)
            extra_body = merge_extra_body(None, agentic_fields)

            completion_args = {
                "cache_creation_token_count": cache_creation_tokens,
                "cache_read_token_count": cache_read_tokens,
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
                "trace_id": meta["trace_id"],
                "task_type": meta["task_type"],
                "subscriber": subscriber if subscriber else None,
                "organization_name": organization_name,
                "subscription_id": meta["subscription_id"],
                "product_name": product_name,
                "agent": meta["agent"],
                "response_quality_score": meta["response_quality_score"],
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

            # Ticket ID field (FRONT-1545)
            ticket_id = trace_fields.get_ticket_id(usage_metadata)
            if ticket_id:
                completion_args["ticket_id"] = ticket_id

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
            result = submit_ai_event("completion", {**completion_args, "extra_body": extra_body})
            logger.debug("Metering call result: %s", result)
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
