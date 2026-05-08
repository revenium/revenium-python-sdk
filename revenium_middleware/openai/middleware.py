import datetime
import logging
import os
import uuid
from typing import Dict, Any, Optional, Tuple
from enum import Enum

import wrapt
from revenium_middleware import client, run_async_in_thread, shutdown_event, merge_metadata
from revenium_middleware._core.enforcement import check_enforcement
from revenium_middleware._core.exceptions import ReveniumCostLimitExceeded  # noqa: F401 — re-exported via openai.exceptions
from revenium_middleware._core.subscriber import extract_subscriber_from_metadata
from revenium_middleware._core.fields import extract_org_and_product, extract_common_metadata, extract_agentic_job_fields, merge_extra_body
from revenium_middleware._core.config import is_selective_metering_enabled, is_capture_prompts_enabled
from revenium_middleware._core.context import is_inside_decorated_function
from revenium_middleware._core.patch_registry import register_patch

# Azure OpenAI support imports
from .provider import Provider, detect_provider, get_provider_metadata, is_azure_provider
from .azure_model_resolver import resolve_azure_model_name
from .azure_config import get_azure_config
from .config import Config, SecurityConfig
from .exceptions import (
    ReveniumMiddlewareError, ValidationError, MeteringError,
    NetworkError, AuthenticationError, categorize_exception, handle_exception_safely
)
from .summary_printer import print_usage_summary

# LangChain integration utilities
from .langchain._utils import is_langchain_available

# Prompt capture utilities
from .prompt_extractor import (
    extract_prompts_from_request,
    extract_response_content,
    extract_streaming_response_content
)

logger = logging.getLogger("revenium_middleware.extension")


# Use centralized security configuration
SENSITIVE_FIELDS = SecurityConfig.SENSITIVE_FIELDS


def extract_prompt_data_if_enabled(
    request_body: Optional[Dict[str, Any]],
    response: Any = None,
    accumulated_content: str = None
) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[bool]]:
    """
    Extract prompt data if capture is enabled.

    Args:
        request_body: Request body containing messages
        response: API response object (for non-streaming)
        accumulated_content: Accumulated streaming content (for streaming)

    Returns:
        Tuple of (system_prompt, input_messages, output_response, prompts_truncated)
    """
    if not is_capture_prompts_enabled() or not request_body:
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


def sanitize_for_logging(data: Any, max_depth: int = Config.MAX_SANITIZATION_DEPTH) -> Any:
    """
    Sanitize data for secure logging by redacting sensitive fields.

    Args:
        data: Data to sanitize (dict, list, or primitive)
        max_depth: Maximum recursion depth to prevent infinite loops

    Returns:
        Sanitized data safe for logging
    """
    if max_depth <= 0:
        return "[MAX_DEPTH_REACHED]"

    if isinstance(data, dict):
        sanitized = {}
        for key, value in data.items():
            key_lower = str(key).lower()
            if any(sensitive in key_lower for sensitive in SENSITIVE_FIELDS):
                sanitized[key] = "[REDACTED]"
            else:
                sanitized[key] = sanitize_for_logging(value, max_depth - 1)
        return sanitized
    elif isinstance(data, (list, tuple)):
        return [sanitize_for_logging(item, max_depth - 1) for item in data]
    elif isinstance(data, str) and len(data) > Config.MAX_LOG_STRING_LENGTH:
        # Truncate very long strings that might contain sensitive data
        return data[:Config.MAX_LOG_STRING_LENGTH] + "...[TRUNCATED]"
    else:
        return data


class OperationType(str, Enum):
    """Operation types for AI API calls."""
    # Spec-compliant values matching revenium_metering API
    CHAT = "CHAT"
    GENERATE = "GENERATE"
    EMBED = "EMBED"
    CLASSIFY = "CLASSIFY"
    SUMMARIZE = "SUMMARIZE"
    TRANSLATE = "TRANSLATE"
    TOOL_CALL = "TOOL_CALL"
    RERANK = "RERANK"
    SEARCH = "SEARCH"
    MODERATION = "MODERATION"
    VISION = "VISION"
    TRANSFORM = "TRANSFORM"
    GUARDRAIL = "GUARDRAIL"
    OTHER = "OTHER"


def map_operation_type_to_sdk(operation_type: OperationType) -> str:
    """
    Map middleware OperationType to SDK-expected operation_type values.

    SDK expects: "CHAT", "GENERATE", "EMBED", "CLASSIFY", "SUMMARIZE", "TRANSLATE",
                 "TOOL_CALL", "RERANK", "SEARCH", "MODERATION", "VISION", "TRANSFORM",
                 "GUARDRAIL", "OTHER"

    Since the enum values now match the SDK values, we can return the value directly.
    """
    return operation_type.value


# Utility functions for token usage tracking
def get_stop_reason(openai_finish_reason: Optional[str]) -> str:
    """
    Map OpenAI/Azure OpenAI finish reasons to Revenium stop reasons.

    Supports both standard OpenAI and Azure-specific finish reasons.
    All unmapped reasons default to "END" to ensure compatibility.
    """
    finish_reason_map = {
        # Standard OpenAI finish reasons
        "stop": "END",
        "function_call": "END_SEQUENCE",
        "timeout": "TIMEOUT",
        "length": "TOKEN_LIMIT",
        "content_filter": "ERROR",

        # Azure OpenAI specific finish reasons
        "tool_calls": "END_SEQUENCE",  # Modern function calling in Azure
    }

    mapped_reason = finish_reason_map.get(openai_finish_reason or "", "END")

    # Log unmapped finish reasons for monitoring
    if openai_finish_reason and openai_finish_reason not in finish_reason_map:
        logger.warning(f"Unmapped finish reason '{openai_finish_reason}' defaulting to 'END'. "
                      f"Consider adding mapping to ensure accurate stop reason tracking.")

    return mapped_reason


def _validate_extract_usage_inputs(response: Any, operation_type: OperationType,
                                  request_time: str, response_time: str,
                                  request_duration: float) -> None:
    """
    Validate inputs for extract_usage_data function.

    Args:
        response: OpenAI API response object
        operation_type: OperationType enum value
        request_time: ISO formatted request timestamp
        response_time: ISO formatted response timestamp
        request_duration: Request duration in milliseconds

    Raises:
        ValidationError: If any input is invalid
    """
    if response is None:
        raise ValidationError("Response object cannot be None")

    if not isinstance(operation_type, OperationType):
        raise ValidationError(f"operation_type must be OperationType, got {type(operation_type)}")

    if not isinstance(request_time, str) or not request_time.strip():
        raise ValidationError("request_time must be a non-empty string")

    if not isinstance(response_time, str) or not response_time.strip():
        raise ValidationError("response_time must be a non-empty string")

    if not isinstance(request_duration, (int, float)) or request_duration < 0:
        raise ValidationError(f"request_duration must be a non-negative number, got {request_duration}")

    # Validate response has required attributes
    if not hasattr(response, 'model'):
        raise ValidationError("Response object must have 'model' attribute")

    if not hasattr(response, 'usage'):
        raise ValidationError("Response object must have 'usage' attribute")

    if response.usage is None:
        raise ValidationError("Response object 'usage' attribute must not be None")


def extract_usage_data(response, operation_type: OperationType, request_time: str, response_time: str, request_duration: float,
                      client_instance: Optional[Any] = None) -> Tuple[Dict[str, Any], str]:
    """
    Extract usage data from OpenAI/Azure OpenAI API responses.
    Unified function that handles both chat and embeddings responses with provider detection.

    Args:
        response: OpenAI API response object (ChatCompletion or CreateEmbeddingResponse) or LegacyAPIResponse
        operation_type: OperationType.CHAT or OperationType.EMBED
        request_time: ISO formatted request timestamp
        response_time: ISO formatted response timestamp
        request_duration: Request duration in milliseconds
        client_instance: OpenAI client instance for provider detection

    Returns:
        Tuple of (usage_data dict, transaction_id string)

    Raises:
        ValidationError: If inputs are invalid
    """
    # Handle LegacyAPIResponse from with_raw_response (used by langchain-openai)
    if hasattr(response, 'parse') and callable(response.parse):
        response = response.parse()

    # Validate all inputs before processing
    _validate_extract_usage_inputs(response, operation_type, request_time, response_time, request_duration)
    # Generate transaction ID - embeddings don't have response.id, chats do
    transaction_id = getattr(response, 'id', str(uuid.uuid4()))

    # Detect provider for this request
    provider = detect_provider(client_instance, getattr(client_instance, 'base_url', None) if client_instance else None)
    provider_metadata = get_provider_metadata(provider)

    # Extract raw model name from response
    raw_model_name = response.model

    # Resolve model name for Azure deployments
    if is_azure_provider(provider) and raw_model_name:
        # For Azure, response.model contains deployment name, resolve to LiteLLM model name
        base_url = getattr(client_instance, 'base_url', None) if client_instance else None
        headers = {}  # Headers would need to be passed from wrapper context
        resolved_model_name = resolve_azure_model_name(raw_model_name, base_url, headers)
        logger.debug(f"Azure model resolution: {raw_model_name} -> {resolved_model_name}")
    else:
        resolved_model_name = raw_model_name

    # Extract tokens based on operation type
    if operation_type == OperationType.EMBED:
        input_tokens = response.usage.prompt_tokens
        output_tokens = 0  # Embeddings don't produce output tokens
        total_tokens = response.usage.total_tokens
        stop_reason = "END"  # Embeddings always complete successfully
    else:  # CHAT (includes Responses API which is mapped to CHAT for backend compatibility)
        # Handle both Chat Completions and Responses API formats
        # Responses API uses input_tokens/output_tokens, Chat uses prompt_tokens/completion_tokens
        if hasattr(response.usage, 'input_tokens'):
            # Responses API format
            input_tokens = response.usage.input_tokens
            output_tokens = response.usage.output_tokens
        else:
            # Chat Completions format
            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens

        total_tokens = response.usage.total_tokens

        # Get stop reason - Responses API may not have choices
        if hasattr(response, 'choices') and response.choices:
            openai_finish_reason = response.choices[0].finish_reason
            stop_reason = get_stop_reason(openai_finish_reason)
        else:
            # Responses API doesn't have choices, use default
            stop_reason = "END"

    # Extract cached tokens (only available for chat completions)
    cached_tokens = 0
    if operation_type == OperationType.CHAT and hasattr(response.usage, 'prompt_tokens_details'):
        cached_tokens = getattr(response.usage.prompt_tokens_details, 'cached_tokens', 0)

    # Build unified usage data structure
    usage_data = {
        "input_token_count": input_tokens,
        "output_token_count": output_tokens,
        "total_token_count": total_tokens,
        "operation_type": operation_type.value,  # Convert enum to string
        "stop_reason": stop_reason,
        "transaction_id": transaction_id,
        "model": resolved_model_name,  # Use resolved model name for accurate pricing
        "provider": provider_metadata["provider"],
        "model_source": provider_metadata["model_source"],
        "is_streamed": False,  # Will be overridden for streaming
        "time_to_first_token": 0,  # Will be set by caller if applicable
        "cache_creation_token_count": cached_tokens,
        "cache_read_token_count": 0,
        "reasoning_token_count": 0,
        "request_time": request_time,
        "response_time": response_time,
        "completion_start_time": response_time,
        "request_duration": int(request_duration),
        "cost_type": "AI",
        "input_token_cost": None,  # Let backend calculate
        "output_token_cost": None,  # Let backend calculate
        "total_cost": None,  # Let backend calculate
    }

    # Debug logging for provider detection and model resolution
    logger.debug(f"Provider detected: {provider}, metadata: {provider_metadata}")
    logger.debug(f"Model resolution: {raw_model_name} -> {resolved_model_name}")

    logger.debug(
        "Extracted %s usage data - input: %d, output: %d, total: %d, transaction_id: %s",
        operation_type.lower(), input_tokens, output_tokens, total_tokens, transaction_id
    )

    return usage_data, transaction_id


async def log_token_usage(
        response_id: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        total_tokens: int,
        cached_tokens: int,
        stop_reason: str,
        request_time: str,
        response_time: str,
        request_duration: int,
        usage_metadata: Dict[str, Any],
        provider: str = "OPENAI",
        model_source: str = "OPENAI",
        system_fingerprint: Optional[str] = None,
        is_streamed: bool = False,
        time_to_first_token: int = 0,
        operation_type: OperationType = OperationType.CHAT,
        # New trace visualization fields
        environment: Optional[str] = None,
        operation_subtype: Optional[str] = None,
        retry_number: int = 0,
        parent_transaction_id: Optional[str] = None,
        transaction_name: Optional[str] = None,
        region: Optional[str] = None,
        credential_alias: Optional[str] = None,
        trace_type: Optional[str] = None,
        trace_name: Optional[str] = None,
        # Prompt capture fields
        system_prompt: Optional[str] = None,
        input_messages: Optional[str] = None,
        output_response: Optional[str] = None,
        prompts_truncated: Optional[bool] = None,
) -> None:
    """Log token usage to Revenium."""
    if client is None:
        return  # metering disabled (no API key configured)
    if shutdown_event.is_set():
        logger.warning("Skipping metering call during shutdown")
        return

    logger.debug("Metering call to Revenium for %s operation %s", operation_type.lower(), response_id)

    # Determine provider - check for OLLAMA first via system fingerprint, then use passed parameters
    if system_fingerprint == "fp_ollama":
        provider = "OLLAMA"
        model_source = "OLLAMA"
        logger.debug(f"OLLAMA provider detected via system_fingerprint: {system_fingerprint}")
    else:
        # Use provider information passed as parameters (already correctly detected for Azure/OpenAI)
        logger.debug(f"Using provider: {provider}, model_source: {model_source}")

    # Create subscriber object from usage metadata
    subscriber = extract_subscriber_from_metadata(usage_metadata)

    # Prepare arguments for create_completion
    # Build completion args, only including non-None values for optional fields
    completion_args = {
        "cache_creation_token_count": cached_tokens,
        "cache_read_token_count": 0,
        "output_token_count": completion_tokens,
        "cost_type": "AI",
        "model": model,
        "input_token_count": prompt_tokens,
        "provider": provider,
        "model_source": model_source,
        "reasoning_token_count": 0,
        "request_time": request_time,
        "response_time": response_time,
        "completion_start_time": response_time,
        "request_duration": int(request_duration),
        "stop_reason": stop_reason,
        "total_token_count": total_tokens,
        "transaction_id": response_id,
        "is_streamed": is_streamed,
        "operation_type": map_operation_type_to_sdk(operation_type),  # Map to SDK-expected values
        "time_to_first_token": time_to_first_token,
        "middleware_source": "PYTHON",
    }

    organization_name, product_name = extract_org_and_product(usage_metadata)
    meta = extract_common_metadata(usage_metadata)

    if meta["trace_id"]:
        completion_args["trace_id"] = meta["trace_id"]
    if meta["task_type"]:
        completion_args["task_type"] = meta["task_type"]
    if subscriber:
        completion_args["subscriber"] = subscriber
    if organization_name:
        completion_args["organization_name"] = organization_name
    if meta["subscription_id"]:
        completion_args["subscription_id"] = meta["subscription_id"]
    if product_name:
        completion_args["product_name"] = product_name
    if meta["agent"]:
        completion_args["agent"] = meta["agent"]
    if meta["response_quality_score"]:
        completion_args["response_quality_score"] = meta["response_quality_score"]

    agentic_fields = extract_agentic_job_fields(usage_metadata)
    if agentic_fields:
        completion_args["extra_body"] = merge_extra_body(completion_args.get("extra_body"), agentic_fields)

    # Add trace visualization fields only if they have values
    if environment:
        completion_args["environment"] = environment
    if operation_subtype:
        completion_args["operation_subtype"] = operation_subtype
    if retry_number is not None:  # 0 is a valid value
        completion_args["retry_number"] = retry_number
    if parent_transaction_id:
        completion_args["parent_transaction_id"] = parent_transaction_id
    if transaction_name:
        completion_args["transaction_name"] = transaction_name
    if region:
        completion_args["region"] = region
    if credential_alias:
        completion_args["credential_alias"] = credential_alias
    if trace_type:
        completion_args["trace_type"] = trace_type
    if trace_name:
        completion_args["trace_name"] = trace_name

    # Add prompt capture fields only if they have values
    if system_prompt is not None:
        completion_args["system_prompt"] = system_prompt
    if input_messages is not None:
        completion_args["input_messages"] = input_messages
    if output_response is not None:
        completion_args["output_response"] = output_response
    if prompts_truncated is not None:
        completion_args["prompts_truncated"] = prompts_truncated

    # Log the arguments at debug level (redact sensitive prompt data)
    safe_args = {k: v for k, v in completion_args.items()
                 if k not in ('system_prompt', 'input_messages', 'output_response')}
    if 'system_prompt' in completion_args:
        safe_args['system_prompt'] = '[REDACTED]'
    if 'input_messages' in completion_args:
        safe_args['input_messages'] = '[REDACTED]'
    if 'output_response' in completion_args:
        safe_args['output_response'] = '[REDACTED]'
    logger.debug("Calling client.ai.create_completion with args: %s", safe_args)

    # Debug logging for metering call
    logger.debug(f"Metering call for {operation_type.value}: {response_id}, tokens: {prompt_tokens}+{completion_tokens}={total_tokens}")

    try:
        # The client.ai.create_completion method is not async, so don't use await
        result = client.ai.create_completion(**completion_args)
        logger.debug("Metering call result: %s", result)
        logger.debug(f"✅ REVENIUM SUCCESS: Metering call successful: {result.id}")

        # Print usage summary if enabled (async, fire-and-forget)
        # Try to get API key from client, fallback to environment variable
        revenium_api_key = None
        if hasattr(client, 'api_key'):
            revenium_api_key = client.api_key
        else:
            revenium_api_key = os.getenv('REVENIUM_METERING_API_KEY')

        if revenium_api_key:
            # Run summary printing in background thread to avoid blocking
            def _print_summary_async():
                print_usage_summary(
                    model=model,
                    provider=provider,
                    request_duration=request_duration,
                    input_token_count=prompt_tokens,
                    output_token_count=completion_tokens,
                    total_token_count=total_tokens,
                    transaction_id=response_id,
                    trace_id=meta["trace_id"],
                    revenium_api_key=revenium_api_key,
                )

            run_async_in_thread(_print_summary_async)
    except Exception as e:
        if not shutdown_event.is_set():
            # Categorize the exception for better error handling
            categorized_error = categorize_exception(e)
            logger.error(f"❌ REVENIUM FAILURE: {categorized_error}")

            # Use sanitized logging to prevent sensitive data exposure
            sanitized_args = sanitize_for_logging(completion_args)
            logger.error(f"❌ REVENIUM FAILURE: Completion args were: {sanitized_args}")

            # Log the full traceback for better debugging
            import traceback
            logger.error(f"❌ REVENIUM FAILURE: Traceback: {traceback.format_exc()}")
        else:
            logger.debug("Metering call failed during shutdown - this is expected")


def create_metering_call(
    response,
    operation_type: OperationType,
    request_time_dt,
    usage_metadata,
    client_instance: Optional[Any] = None,
    time_to_first_token: int = 0,
    is_streamed: bool = False,
    request_body: Optional[Dict[str, Any]] = None
):
    """
    Unified function to create and execute metering calls for any operation
    type. Reduces duplication between chat and embeddings wrappers.
    """
    # Import trace field functions
    from .trace_fields import (
        get_environment, get_region, get_credential_alias,
        get_trace_type, get_trace_name, get_parent_transaction_id,
        get_transaction_name, get_retry_number, detect_operation_type,
        validate_trace_type, validate_trace_name
    )

    # Record timing
    response_time_dt = datetime.datetime.now(datetime.timezone.utc)
    response_time = response_time_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    request_time = request_time_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    request_duration = (
        (response_time_dt - request_time_dt).total_seconds() * 1000
    )

    try:
        usage_data, transaction_id = extract_usage_data(
            response, operation_type, request_time, response_time,
            request_duration, client_instance
        )
    except (ValidationError, Exception) as e:
        logger.warning("Skipping metering for response with invalid usage data: %s", e)
        return

    # Override streaming and timing info
    usage_data["is_streamed"] = is_streamed
    usage_data["time_to_first_token"] = time_to_first_token

    # Get system fingerprint if available (chat only)
    system_fingerprint = getattr(response, 'system_fingerprint', None)

    # Extract trace fields (usage_metadata takes precedence over env vars)
    environment = usage_metadata.get('environment') or get_environment()
    region = usage_metadata.get('region') or get_region()
    credential_alias = (
        usage_metadata.get('credentialAlias') or
        usage_metadata.get('credential_alias') or
        get_credential_alias()
    )

    # Validate trace_type from usage_metadata to prevent bypass
    trace_type_raw = (
        usage_metadata.get('traceType') or
        usage_metadata.get('trace_type')
    )
    trace_type = validate_trace_type(trace_type_raw) if trace_type_raw else get_trace_type()

    # Validate trace_name from usage_metadata to prevent bypass
    trace_name_raw = (
        usage_metadata.get('traceName') or
        usage_metadata.get('trace_name')
    )
    trace_name = validate_trace_name(trace_name_raw) if trace_name_raw else get_trace_name()
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
    provider = usage_data.get("provider", "OPENAI")
    # Determine endpoint from operation_type
    if operation_type == OperationType.CHAT:
        endpoint = "/chat/completions"
    elif operation_type == OperationType.EMBED:
        endpoint = "/embeddings"
    else:
        endpoint = "/chat/completions"  # Default

    operation_info = detect_operation_type(
        provider, endpoint, request_body or {}
    )
    operation_subtype = operation_info.get('operationSubtype')

    # Extract prompt data if capture is enabled (only for CHAT operations)
    system_prompt = None
    input_messages = None
    output_response = None
    prompts_truncated = None

    if operation_type == OperationType.CHAT:
        (system_prompt, input_messages, output_response, prompts_truncated) = (
            extract_prompt_data_if_enabled(request_body, response=response)
        )

    # Create async metering call
    async def metering_call():
        await log_token_usage(
            response_id=transaction_id,
            model=usage_data["model"],
            prompt_tokens=usage_data["input_token_count"],
            completion_tokens=usage_data["output_token_count"],
            total_tokens=usage_data["total_token_count"],
            cached_tokens=usage_data["cache_creation_token_count"],
            stop_reason=usage_data["stop_reason"],
            request_time=usage_data["request_time"],
            response_time=usage_data["response_time"],
            request_duration=usage_data["request_duration"],
            usage_metadata=usage_metadata,
            provider=usage_data["provider"],
            model_source=usage_data["model_source"],
            system_fingerprint=system_fingerprint,
            is_streamed=is_streamed,
            time_to_first_token=time_to_first_token,
            operation_type=operation_type,
            # New trace visualization fields
            environment=environment,
            operation_subtype=operation_subtype,
            retry_number=retry_number,
            parent_transaction_id=parent_transaction_id,
            transaction_name=transaction_name,
            region=region,
            credential_alias=credential_alias,
            trace_type=trace_type,
            trace_name=trace_name,
            # Prompt capture fields
            system_prompt=system_prompt,
            input_messages=input_messages,
            output_response=output_response,
            prompts_truncated=prompts_truncated,
        )

    # Start metering thread
    thread = run_async_in_thread(metering_call())
    logger.debug("%s metering thread started: %s", operation_type, thread)
    return thread


def _extract_langchain_usage_metadata():
    """
    Extract usage_metadata from LangChain context variables.

    LangChain stores context information in thread-local variables that we can access
    to get the usage_metadata that was passed to LangChain methods.

    This is optional functionality - if LangChain is not installed, this function
    gracefully returns an empty dict without logging errors.
    """
    # Only attempt LangChain integration if LangChain is available
    if not is_langchain_available():
        return {}

    try:
        # Try to import LangChain context variables
        from langchain_core.globals import get_llm_cache  # noqa: F401
        from langchain_core.callbacks.manager import CallbackManagerForLLMRun  # noqa: F401
        import contextvars  # noqa: F401

        # Try to get the current context
        # LangChain uses context variables to store run information
        # We need to look for the current callback manager or run context

        # Check if we're in a LangChain context by looking for context variables
        # This is a best-effort approach since LangChain's internal context handling
        # can vary between versions

        # Look for common LangChain context patterns
        import inspect
        frame = inspect.currentframe()

        # Walk up the call stack to find LangChain frames
        while frame:
            frame_locals = frame.f_locals
            frame_globals = frame.f_globals

            # Look for LangChain-specific variables in the call stack
            # Check for 'config' parameter which often contains metadata
            if 'config' in frame_locals and isinstance(frame_locals['config'], dict):
                config = frame_locals['config']
                if 'metadata' in config and isinstance(config['metadata'], dict):
                    metadata = config['metadata']
                    if 'usage_metadata' in metadata:
                        logger.debug(f"Found usage_metadata in LangChain config: {metadata['usage_metadata']}")
                        return metadata['usage_metadata']

            # Check for 'metadata' parameter directly
            if 'metadata' in frame_locals and isinstance(frame_locals['metadata'], dict):
                metadata = frame_locals['metadata']
                if 'usage_metadata' in metadata:
                    logger.debug(f"Found usage_metadata in LangChain metadata: {metadata['usage_metadata']}")
                    return metadata['usage_metadata']

            # Check for callback manager with metadata
            if 'callback_manager' in frame_locals:
                cb_manager = frame_locals['callback_manager']
                if hasattr(cb_manager, 'metadata') and isinstance(cb_manager.metadata, dict):
                    if 'usage_metadata' in cb_manager.metadata:
                        logger.debug(f"Found usage_metadata in callback manager: {cb_manager.metadata['usage_metadata']}")
                        return cb_manager.metadata['usage_metadata']

            frame = frame.f_back

        logger.debug("No usage_metadata found in LangChain context")
        return {}

    except Exception as e:
        # Only log at debug level for unexpected errors during metadata extraction
        # This is optional functionality, so we don't want to spam logs
        logger.debug(f"Error extracting LangChain usage_metadata: {e}")
        return {}


def embeddings_create_wrapper(wrapped, instance, args, kwargs):
    logger.debug("OpenAI/Azure OpenAI embeddings.create wrapper called")

    if is_selective_metering_enabled() and not is_inside_decorated_function():
        return wrapped(*args, **kwargs)

    request_body = kwargs.copy()

    # Extract API-level metadata from kwargs
    api_metadata = kwargs.pop("usage_metadata", {}) if "usage_metadata" in kwargs else {}

    # Try to extract usage_metadata from LangChain context if not found in kwargs
    if not api_metadata:
        api_metadata = _extract_langchain_usage_metadata()

    # Merge with decorator metadata (API-level takes precedence)
    usage_metadata = merge_metadata(api_metadata)

    # Detect provider and validate Azure config if needed
    client_instance = getattr(instance, '_client', None)
    provider = detect_provider(client=client_instance)

    # Validate Azure configuration if Azure provider detected
    if is_azure_provider(provider):
        azure_config = get_azure_config()
        if not azure_config.is_valid():
            logger.warning(
                "Azure OpenAI detected but configuration is incomplete. "
                "Set AZURE_OPENAI_ENDPOINT for proper Azure support."
            )
        else:
            logger.debug(
                f"Azure OpenAI configuration validated: "
                f"{azure_config.to_dict()}"
            )
            azure_config.validate_deployment()

    # Record request time
    request_time_dt = datetime.datetime.now(datetime.timezone.utc)

    # Enforcement pre-call check — may raise ReveniumCostLimitExceeded
    check_enforcement(usage_metadata)

    logger.debug(
        f"Calling wrapped embeddings function with args: {args}, "
        f"kwargs: {kwargs}"
    )

    # Call the original OpenAI function
    response = wrapped(*args, **kwargs)

    logger.debug("Handling embeddings response: %s", response)

    # Create metering call using unified function
    create_metering_call(
        response,
        OperationType.EMBED,
        request_time_dt,
        usage_metadata,
        client_instance=getattr(instance, '_client', None),
        request_body=request_body
    )

    return response


def create_wrapper(wrapped, instance, args, kwargs):
    logger.debug("OpenAI/Azure OpenAI chat.completions.create wrapper called")

    if is_selective_metering_enabled() and not is_inside_decorated_function():
        return wrapped(*args, **kwargs)

    # The Perplexity middleware patches the same Completions.create slot. When
    # both middlewares are loaded the wrappers chain via wrapt — defer
    # Perplexity-bound calls to the Perplexity wrapper to avoid double-metering.
    _client = getattr(instance, '_client', None)
    _base_url = getattr(_client, 'base_url', None) if _client else None
    if _base_url and "perplexity" in str(_base_url).lower():
        return wrapped(*args, **kwargs)

    # Capture request body before modifications (for operation detection)
    request_body = kwargs.copy()

    # Extract API-level metadata from kwargs
    api_metadata = kwargs.pop("usage_metadata", {}) if "usage_metadata" in kwargs else {}

    # Try to extract usage_metadata from LangChain context if not found in kwargs
    if not api_metadata:
        api_metadata = _extract_langchain_usage_metadata()

    # Merge with decorator metadata (API-level takes precedence)
    usage_metadata = merge_metadata(api_metadata)

    # Check if this is a streaming request
    stream = kwargs.get('stream', False)

    # If streaming, add stream_options to include usage information
    if stream:
        kwargs = dict(kwargs)
        if 'stream_options' not in kwargs:
            kwargs['stream_options'] = {}
        else:
            kwargs['stream_options'] = dict(kwargs['stream_options'])
        kwargs['stream_options']['include_usage'] = True
        logger.debug(
            "Added include_usage to stream_options for accurate token "
            "counting in streaming response"
        )

    # Detect provider and validate Azure config if needed
    client_instance = getattr(instance, '_client', None)
    provider = detect_provider(client=client_instance)

    # Validate Azure configuration if Azure provider detected
    if is_azure_provider(provider):
        azure_config = get_azure_config()
        if not azure_config.is_valid():
            logger.warning(
                "Azure OpenAI detected but configuration is incomplete. "
                "Set AZURE_OPENAI_ENDPOINT for proper Azure support."
            )
        else:
            logger.debug(
                f"Azure OpenAI configuration validated: "
                f"{azure_config.to_dict()}"
            )
            azure_config.validate_deployment()

    request_time_dt = datetime.datetime.now(datetime.timezone.utc)

    # Enforcement pre-call check — may raise ReveniumCostLimitExceeded
    check_enforcement(usage_metadata)

    logger.debug(
        f"Calling wrapped function with args: {args}, kwargs: {kwargs}"
    )

    response = wrapped(*args, **kwargs)

    # Record time to first token (for non-streaming, same as full response)
    first_token_time_dt = datetime.datetime.now(datetime.timezone.utc)
    time_to_first_token = int(
        (first_token_time_dt - request_time_dt).total_seconds() * 1000
    )

    # Handle based on response type
    if stream:
        # For streaming responses (openai.Stream)
        logger.debug("Handling streaming response")
        return handle_streaming_response(
            response,
            request_time_dt,
            usage_metadata,
            client_instance=getattr(instance, '_client', None),
            request_body=request_body
        )
    else:
        # For non-streaming responses (ChatCompletion)
        logger.debug("Handling non-streaming response: %s", response)

        # Create metering call using unified function
        create_metering_call(
            response,
            OperationType.CHAT,
            request_time_dt,
            usage_metadata,
            client_instance=getattr(instance, '_client', None),
            time_to_first_token=time_to_first_token,
            request_body=request_body
        )

        return response


def handle_streaming_response(
    stream,
    request_time_dt,
    usage_metadata,
    client_instance: Optional[Any] = None,
    request_body: Optional[Dict[str, Any]] = None
):
    """
    Handle streaming responses from OpenAI/Azure OpenAI.
    Wraps the stream to collect metrics and log them after completion.
    Similar to the approach used in the Ollama middleware.
    """

    # Create a wrapper for the streaming response with proper resource
    # management
    class StreamWrapper:
        def __init__(self, stream):
            self.stream = stream
            self.chunks = []
            self.response_id = None
            self.model = None
            self.finish_reason = None
            self.system_fingerprint = None
            self.request_time_dt = request_time_dt
            self.usage_metadata = usage_metadata
            self.final_usage = None
            self.completion_text = ""
            self.first_token_time = None
            # Store for Azure provider detection
            self.client_instance = client_instance
            self.request_body = request_body
            self._closed = False
            self._usage_logged = False

        def __iter__(self):
            return self

        def __next__(self):
            if self._closed:
                raise StopIteration("Stream has been closed")

            try:
                chunk = next(self.stream)
                self._process_chunk(chunk)
                return chunk
            except StopIteration:
                self._finalize()
                raise
            except Exception as e:
                # Ensure cleanup on any error
                self._finalize()
                logger.error(f"Error in streaming response: {e}")
                raise

        def __enter__(self):
            """Context manager entry."""
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            """Context manager exit with cleanup."""
            self._finalize()

        def _finalize(self):
            """Finalize the stream and log usage if not already done."""
            if not self._usage_logged:
                self._log_usage()
                self._usage_logged = True
            self._close_stream()

        def _close_stream(self):
            """Close the underlying stream if possible."""
            if not self._closed:
                try:
                    if hasattr(self.stream, 'close'):
                        self.stream.close()
                except Exception as e:
                    logger.debug(f"Error closing stream: {e}")
                finally:
                    self._closed = True

        def _process_chunk(self, chunk):
            # Extract response ID and model from the chunk if available
            if self.response_id is None and hasattr(chunk, 'id'):
                self.response_id = chunk.id
            if self.model is None and hasattr(chunk, 'model'):
                self.model = chunk.model
            if self.system_fingerprint is None and hasattr(chunk, 'system_fingerprint'):
                self.system_fingerprint = chunk.system_fingerprint
                logger.debug(f"Captured system_fingerprint from stream chunk: {self.system_fingerprint}")
            else:
                logger.debug(f"System fingerprint already set: {self.system_fingerprint}")


            # Check for finish reason in the chunk
            if chunk.choices and chunk.choices[0].finish_reason:
                self.finish_reason = chunk.choices[0].finish_reason

            # Check if this chunk has usage data (can be in final chunk with or without choices)
            if hasattr(chunk, 'usage') and chunk.usage:
                logger.debug(f"Found usage data in chunk: {chunk.usage}")
                self.final_usage = chunk.usage
                # Don't return yet - we still need to process the chunk for finish_reason etc.

            # Collect content for token estimation if needed
            if chunk.choices and hasattr(chunk.choices[0], 'delta') and hasattr(chunk.choices[0].delta, 'content') and \
                    chunk.choices[0].delta.content:
                # Record time of first token if not already set
                if self.first_token_time is None:
                    self.first_token_time = datetime.datetime.now(datetime.timezone.utc)
                self.completion_text += chunk.choices[0].delta.content

            # Store the chunk for later analysis
            self.chunks.append(chunk)

        def _log_usage(self):
            # Only return if we have neither chunks nor final usage data
            if not self.chunks and not self.final_usage:
                return

            # Record response time and calculate duration
            response_time_dt = datetime.datetime.now(datetime.timezone.utc)
            response_time = response_time_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
            request_duration = (response_time_dt - self.request_time_dt).total_seconds() * 1000

            # Get token usage information
            prompt_tokens = 0
            completion_tokens = 0
            total_tokens = 0
            cached_tokens = 0

            # First check if we have the final usage data from the special chunk
            if self.final_usage:
                prompt_tokens = self.final_usage.prompt_tokens
                completion_tokens = self.final_usage.completion_tokens
                total_tokens = self.final_usage.total_tokens
                # Check if we have cached tokens info
                if hasattr(self.final_usage, 'prompt_tokens_details') and hasattr(
                        self.final_usage.prompt_tokens_details, 'cached_tokens'):
                    cached_tokens = self.final_usage.prompt_tokens_details.cached_tokens
                logger.debug(
                    f"Using token usage from final chunk: prompt={prompt_tokens}, completion={completion_tokens}, total={total_tokens}")
            else:
                # If we don't have usage data, estimate from content
                logger.warning("No usage data found in streaming response!")

            stop_reason = get_stop_reason(self.finish_reason)

            # Log the token usage
            if self.response_id:
                logger.debug(
                    "Streaming token usage - response_id: %s, prompt: %d, completion: %d, total: %d",
                    self.response_id, prompt_tokens, completion_tokens, total_tokens
                )

                # Detect provider and resolve model name for Azure
                provider = detect_provider(self.client_instance,
                                         getattr(self.client_instance, 'base_url', None) if self.client_instance else None)
                provider_metadata = get_provider_metadata(provider)

                # Resolve model name for Azure deployments
                raw_model_name = self.model or "unknown"
                if is_azure_provider(provider) and raw_model_name != "unknown":
                    base_url = getattr(self.client_instance, 'base_url', None) if self.client_instance else None
                    headers = {}  # Headers would need to be passed from wrapper context
                    resolved_model_name = resolve_azure_model_name(raw_model_name, base_url, headers)
                    logger.debug(f"Azure streaming model resolution: {raw_model_name} -> {resolved_model_name}")
                else:
                    resolved_model_name = raw_model_name

                # Calculate time to first token if available
                time_to_first_token = 0
                if self.first_token_time:
                    time_to_first_token = int(
                        (self.first_token_time - self.request_time_dt)
                        .total_seconds() * 1000
                    )
                    logger.debug(f"Time to first token: {time_to_first_token}ms")

                # Extract trace fields for streaming response
                from .trace_fields import (
                    get_environment, get_region, get_credential_alias,
                    get_trace_type, get_trace_name,
                    get_parent_transaction_id,
                    get_transaction_name, get_retry_number,
                    detect_operation_type,
                    validate_trace_type, validate_trace_name
                )

                # Get trace fields (usage_metadata takes precedence)
                environment = (
                    self.usage_metadata.get('environment') or
                    get_environment()
                )
                region = (
                    self.usage_metadata.get('region') or
                    get_region()
                )
                credential_alias = (
                    self.usage_metadata.get('credentialAlias') or
                    self.usage_metadata.get('credential_alias') or
                    get_credential_alias()
                )

                # Validate trace_type from usage_metadata to prevent bypass
                trace_type_raw = (
                    self.usage_metadata.get('traceType') or
                    self.usage_metadata.get('trace_type')
                )
                trace_type = validate_trace_type(trace_type_raw) if trace_type_raw else get_trace_type()

                # Validate trace_name from usage_metadata to prevent bypass
                trace_name_raw = (
                    self.usage_metadata.get('traceName') or
                    self.usage_metadata.get('trace_name')
                )
                trace_name = validate_trace_name(trace_name_raw) if trace_name_raw else get_trace_name()
                parent_transaction_id = (
                    self.usage_metadata.get('parentTransactionId') or
                    self.usage_metadata.get('parent_transaction_id') or
                    get_parent_transaction_id()
                )
                transaction_name = (
                    self.usage_metadata.get('transactionName') or
                    self.usage_metadata.get('transaction_name') or
                    get_transaction_name(self.usage_metadata)
                )
                retry_number = self.usage_metadata.get(
                    'retryNumber',
                    self.usage_metadata.get('retry_number', get_retry_number())
                )

                # Detect operation type and subtype
                operation_info = detect_operation_type(
                    provider, "/chat/completions", self.request_body or {}
                )
                operation_subtype = operation_info.get('operationSubtype')

                # Extract prompt data if capture is enabled
                (system_prompt, input_messages, output_response, prompts_truncated) = (
                    extract_prompt_data_if_enabled(
                        self.request_body,
                        accumulated_content=self.completion_text
                    )
                )

                async def metering_call():
                    await log_token_usage(
                        response_id=self.response_id,
                        model=resolved_model_name,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        total_tokens=total_tokens,
                        cached_tokens=cached_tokens,
                        stop_reason=stop_reason,
                        request_time=self.request_time_dt.strftime(
                            "%Y-%m-%dT%H:%M:%SZ"
                        ),
                        response_time=response_time,
                        request_duration=int(request_duration),
                        usage_metadata=self.usage_metadata,
                        provider=provider_metadata["provider"],
                        model_source=provider_metadata["model_source"],
                        system_fingerprint=self.system_fingerprint,
                        is_streamed=True,
                        time_to_first_token=time_to_first_token,
                        operation_type=OperationType.CHAT,
                        # New trace visualization fields
                        environment=environment,
                        operation_subtype=operation_subtype,
                        retry_number=retry_number,
                        parent_transaction_id=parent_transaction_id,
                        transaction_name=transaction_name,
                        region=region,
                        credential_alias=credential_alias,
                        trace_type=trace_type,
                        trace_name=trace_name,
                        # Prompt capture fields
                        system_prompt=system_prompt,
                        input_messages=input_messages,
                        output_response=output_response,
                        prompts_truncated=prompts_truncated,
                    )

                thread = run_async_in_thread(metering_call())
                logger.debug("Streaming metering thread started: %s", thread)

    # Return the wrapped stream
    return StreamWrapper(iter(stream))


def responses_create_wrapper(wrapped, instance, args, kwargs):
    logger.debug("OpenAI Responses API create wrapper called")

    if is_selective_metering_enabled() and not is_inside_decorated_function():
        return wrapped(*args, **kwargs)

    api_metadata = kwargs.pop("usage_metadata", {})

    if not api_metadata:
        api_metadata = _extract_langchain_usage_metadata()

    usage_metadata = merge_metadata(api_metadata)

    # Check if this is a streaming request
    stream = kwargs.get('stream', False)

    # Record request time
    request_time_dt = datetime.datetime.now(datetime.timezone.utc)

    # Enforcement pre-call check — may raise ReveniumCostLimitExceeded
    check_enforcement(usage_metadata)

    logger.debug(f"Calling wrapped responses function with args: {args}, kwargs: {kwargs}")

    # Call the original OpenAI function
    response = wrapped(*args, **kwargs)

    # Record time to first token (for non-streaming, this is the same as the full response time)
    first_token_time_dt = datetime.datetime.now(datetime.timezone.utc)
    time_to_first_token = int((first_token_time_dt - request_time_dt).total_seconds() * 1000)

    # Handle based on response type
    if stream:
        # For streaming responses
        logger.debug("Handling streaming Responses API response")
        return handle_streaming_responses(
            response,
            request_time_dt,
            usage_metadata,
            client_instance=getattr(instance, '_client', None)
        )
    else:
        # For non-streaming responses
        logger.debug("Handling non-streaming Responses API response: %s", response)

        # Create metering call using unified function - pass client instance for provider detection
        # Map Responses API to CHAT operation type for Revenium backend compatibility
        # The backend does not yet support a separate RESPONSES operation type
        create_metering_call(response, OperationType.CHAT, request_time_dt, usage_metadata,
                            client_instance=getattr(instance, '_client', None),
                            time_to_first_token=time_to_first_token)

        return response


def handle_streaming_responses(stream, request_time_dt, usage_metadata,
                               client_instance: Optional[Any] = None):
    """
    Handle streaming responses from OpenAI Responses API.
    Wraps the stream to collect metrics and log them after completion.
    """

    # Create a wrapper for the streaming response with proper resource management
    class StreamResponseWrapper:
        def __init__(self, stream):
            self.stream = stream
            self.chunks = []
            self.response_id = None
            self.model = None
            self.request_time_dt = request_time_dt
            self.usage_metadata = usage_metadata
            self.final_usage = None
            self.client_instance = client_instance  # Store for provider detection
            self._closed = False
            self._usage_logged = False
            self.last_chunk = None  # Store the last chunk to extract usage data

        def __iter__(self):
            return self

        def __next__(self):
            if self._closed:
                raise StopIteration("Stream has been closed")

            try:
                chunk = next(self.stream)
                self._process_chunk(chunk)
                self.last_chunk = chunk  # Store the last chunk
                return chunk
            except StopIteration:
                self._finalize()
                raise
            except Exception as e:
                # Ensure cleanup on any error
                self._finalize()
                logger.error(f"Error in streaming Responses API response: {e}")
                raise

        def __enter__(self):
            """Context manager entry."""
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            """Context manager exit with cleanup."""
            self._finalize()

        def _finalize(self):
            """Finalize the stream and log usage if not already done."""
            if not self._usage_logged:
                self._log_usage()
                self._usage_logged = True
            self._close_stream()

        def _close_stream(self):
            """Close the underlying stream if possible."""
            if not self._closed:
                try:
                    if hasattr(self.stream, 'close'):
                        self.stream.close()
                except Exception as e:
                    logger.debug(f"Error closing stream: {e}")
                finally:
                    self._closed = True

        def _process_chunk(self, chunk):
            # Extract response ID and model from the chunk if available
            if self.response_id is None and hasattr(chunk, 'id'):
                self.response_id = chunk.id
            if self.model is None and hasattr(chunk, 'model'):
                self.model = chunk.model

            # Check if this is the final chunk with usage data
            if hasattr(chunk, 'usage') and chunk.usage:
                logger.debug(f"Found usage data in Responses API stream: {chunk.usage}")
                self.final_usage = chunk.usage
                return

            # Store the chunk for later analysis
            self.chunks.append(chunk)

        def _log_usage(self):
            if not self.chunks and not self.final_usage and not self.last_chunk:
                return

            # Record response time and calculate duration
            response_time_dt = datetime.datetime.now(datetime.timezone.utc)
            response_time = response_time_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
            request_duration = (response_time_dt - self.request_time_dt).total_seconds() * 1000

            # Get token usage information
            input_tokens = 0
            output_tokens = 0
            total_tokens = 0

            # Get usage data from the final chunk or last chunk
            if self.final_usage:
                input_tokens = self.final_usage.input_tokens
                output_tokens = self.final_usage.output_tokens
                total_tokens = self.final_usage.total_tokens
                logger.debug(
                    f"Using token usage from Responses API stream final chunk: input={input_tokens}, "
                    f"output={output_tokens}, total={total_tokens}")
            elif self.last_chunk and hasattr(self.last_chunk, 'usage') and self.last_chunk.usage:
                # Try to extract usage from the last chunk
                input_tokens = self.last_chunk.usage.input_tokens
                output_tokens = self.last_chunk.usage.output_tokens
                total_tokens = self.last_chunk.usage.total_tokens
                logger.debug(
                    f"Using token usage from Responses API last chunk: input={input_tokens}, "
                    f"output={output_tokens}, total={total_tokens}")
            else:
                # If we don't have usage data, log warning
                logger.warning("No usage data found in streaming Responses API response!")

            # Log the token usage
            if self.response_id:
                logger.debug(
                    "Streaming Responses API token usage - response_id: %s, input: %d, output: %d, total: %d",
                    self.response_id, input_tokens, output_tokens, total_tokens
                )

                # Detect provider and resolve model name for Azure
                provider = detect_provider(self.client_instance,
                                         getattr(self.client_instance, 'base_url', None)
                                         if self.client_instance else None)
                provider_metadata = get_provider_metadata(provider)

                # Resolve model name for Azure deployments
                raw_model_name = self.model or "unknown"
                if is_azure_provider(provider) and raw_model_name != "unknown":
                    base_url = getattr(self.client_instance, 'base_url', None) if self.client_instance else None
                    headers = {}  # Headers would need to be passed from wrapper context
                    resolved_model_name = resolve_azure_model_name(raw_model_name, base_url, headers)
                    logger.debug(f"Azure Responses API streaming model resolution: {raw_model_name} -> "
                                f"{resolved_model_name}")
                else:
                    resolved_model_name = raw_model_name

                from .trace_fields import (
                    get_environment, get_region, get_credential_alias,
                    get_trace_type, get_trace_name,
                    get_parent_transaction_id,
                    get_transaction_name, get_retry_number,
                    detect_operation_type,
                    validate_trace_type, validate_trace_name
                )

                environment = (
                    self.usage_metadata.get('environment') or
                    get_environment()
                )
                region = (
                    self.usage_metadata.get('region') or
                    get_region()
                )
                credential_alias = (
                    self.usage_metadata.get('credentialAlias') or
                    self.usage_metadata.get('credential_alias') or
                    get_credential_alias()
                )
                trace_type_raw = (
                    self.usage_metadata.get('traceType') or
                    self.usage_metadata.get('trace_type')
                )
                trace_type = validate_trace_type(trace_type_raw) if trace_type_raw else get_trace_type()
                trace_name_raw = (
                    self.usage_metadata.get('traceName') or
                    self.usage_metadata.get('trace_name')
                )
                trace_name = validate_trace_name(trace_name_raw) if trace_name_raw else get_trace_name()
                parent_transaction_id = (
                    self.usage_metadata.get('parentTransactionId') or
                    self.usage_metadata.get('parent_transaction_id') or
                    get_parent_transaction_id()
                )
                transaction_name = (
                    self.usage_metadata.get('transactionName') or
                    self.usage_metadata.get('transaction_name') or
                    get_transaction_name(self.usage_metadata)
                )
                retry_number = self.usage_metadata.get(
                    'retryNumber',
                    self.usage_metadata.get('retry_number', get_retry_number())
                )
                operation_info = detect_operation_type(
                    provider, "/responses", {}
                )
                operation_subtype = operation_info.get('operationSubtype')

                async def metering_call():
                    await log_token_usage(
                        response_id=self.response_id,
                        model=resolved_model_name,
                        prompt_tokens=input_tokens,
                        completion_tokens=output_tokens,
                        total_tokens=total_tokens,
                        cached_tokens=0,
                        stop_reason="END",
                        request_time=self.request_time_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
                        response_time=response_time,
                        request_duration=int(request_duration),
                        usage_metadata=self.usage_metadata,
                        provider=provider_metadata["provider"],
                        model_source=provider_metadata["model_source"],
                        system_fingerprint=None,
                        is_streamed=True,
                        time_to_first_token=0,
                        operation_type=OperationType.CHAT,
                        environment=environment,
                        operation_subtype=operation_subtype,
                        retry_number=retry_number,
                        parent_transaction_id=parent_transaction_id,
                        transaction_name=transaction_name,
                        region=region,
                        credential_alias=credential_alias,
                        trace_type=trace_type,
                        trace_name=trace_name,
                    )

                thread = run_async_in_thread(metering_call())
                logger.debug("Streaming Responses API metering thread started: %s", thread)

    # Return the wrapped stream
    return StreamResponseWrapper(iter(stream))


def _wrap_async_stream(stream, request_time_dt, usage_metadata, client_instance=None, request_body=None):
    class AsyncStreamWrapper:
        def __init__(self, stream):
            self.stream = stream
            self.chunks = []
            self.response_id = None
            self.model = None
            self.finish_reason = None
            self.final_usage = None
            self.first_token_time = None
            self._finalized = False

        def __aiter__(self):
            return self

        async def __anext__(self):
            try:
                chunk = await self.stream.__anext__()
            except StopAsyncIteration:
                self._finalize()
                raise

            if self.response_id is None and hasattr(chunk, 'id'):
                self.response_id = chunk.id
            if self.model is None and hasattr(chunk, 'model'):
                self.model = chunk.model
            if chunk.choices and chunk.choices[0].finish_reason:
                self.finish_reason = chunk.choices[0].finish_reason
            if hasattr(chunk, 'usage') and chunk.usage:
                self.final_usage = chunk.usage
            if self.first_token_time is None and chunk.choices and hasattr(chunk.choices[0], 'delta') and getattr(chunk.choices[0].delta, 'content', None):
                self.first_token_time = datetime.datetime.now(datetime.timezone.utc)
            self.chunks.append(chunk)
            return chunk

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            self._finalize()

        def _finalize(self):
            if self._finalized:
                return
            self._finalized = True

            if not self.chunks and not self.final_usage:
                return

            prompt_tokens = 0
            completion_tokens = 0
            cached_tokens = 0

            if self.final_usage:
                prompt_tokens = getattr(self.final_usage, 'prompt_tokens', 0) or 0
                completion_tokens = getattr(self.final_usage, 'completion_tokens', 0) or 0
                if hasattr(self.final_usage, 'prompt_tokens_details') and self.final_usage.prompt_tokens_details:
                    cached_tokens = getattr(self.final_usage.prompt_tokens_details, 'cached_tokens', 0) or 0

            total_tokens = prompt_tokens + completion_tokens
            time_to_first_token = int((self.first_token_time - request_time_dt).total_seconds() * 1000) if self.first_token_time else 0

            response_time_dt = datetime.datetime.now(datetime.timezone.utc)
            request_time = request_time_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
            response_time = response_time_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
            request_duration = (response_time_dt - request_time_dt).total_seconds() * 1000

            try:
                create_metering_call(
                    type('Response', (), {
                        'id': self.response_id,
                        'model': self.model or 'unknown',
                        'usage': self.final_usage,
                        'system_fingerprint': None,
                    })(),
                    OperationType.CHAT,
                    request_time_dt,
                    usage_metadata,
                    client_instance=client_instance,
                    is_streamed=True,
                    time_to_first_token=time_to_first_token,
                    request_body=request_body
                )
            except Exception as e:
                logger.warning("Async stream metering error: %s", e)

    return AsyncStreamWrapper(stream)


def async_create_wrapper(wrapped, instance, args, kwargs):
    logger.debug("OpenAI/Azure OpenAI async chat.completions.create wrapper called")

    if is_selective_metering_enabled() and not is_inside_decorated_function():
        return wrapped(*args, **kwargs)

    # See sync create_wrapper above: defer Perplexity-bound calls to the
    # Perplexity wrapper to avoid double-metering when both middlewares chain.
    _client = getattr(instance, '_client', None)
    _base_url = getattr(_client, 'base_url', None) if _client else None
    if _base_url and "perplexity" in str(_base_url).lower():
        return wrapped(*args, **kwargs)

    request_body = kwargs.copy()

    api_metadata = kwargs.pop("usage_metadata", {}) if "usage_metadata" in kwargs else {}

    if not api_metadata:
        api_metadata = _extract_langchain_usage_metadata()

    usage_metadata = merge_metadata(api_metadata)

    stream = kwargs.get('stream', False)

    if stream:
        kwargs = dict(kwargs)
        if 'stream_options' not in kwargs:
            kwargs['stream_options'] = {}
        else:
            kwargs['stream_options'] = dict(kwargs['stream_options'])
        kwargs['stream_options']['include_usage'] = True

    client_instance = getattr(instance, '_client', None)
    provider = detect_provider(client=client_instance)

    if is_azure_provider(provider):
        azure_config = get_azure_config()
        if not azure_config.is_valid():
            logger.warning(
                "Azure OpenAI detected but configuration is incomplete. "
                "Set AZURE_OPENAI_ENDPOINT for proper Azure support."
            )
        else:
            logger.debug(f"Azure OpenAI configuration validated: {azure_config.to_dict()}")
            azure_config.validate_deployment()

    request_time_dt = datetime.datetime.now(datetime.timezone.utc)

    # Enforcement pre-call check — may raise ReveniumCostLimitExceeded
    check_enforcement(usage_metadata)

    async def _async_invoke():
        response = await wrapped(*args, **kwargs)

        first_token_time_dt = datetime.datetime.now(datetime.timezone.utc)
        time_to_first_token = int(
            (first_token_time_dt - request_time_dt).total_seconds() * 1000
        )

        if stream:
            return _wrap_async_stream(
                response,
                request_time_dt,
                usage_metadata,
                client_instance=getattr(instance, '_client', None),
                request_body=request_body
            )

        create_metering_call(
            response,
            OperationType.CHAT,
            request_time_dt,
            usage_metadata,
            client_instance=getattr(instance, '_client', None),
            time_to_first_token=time_to_first_token,
            request_body=request_body
        )

        return response

    return _async_invoke()


def async_embeddings_create_wrapper(wrapped, instance, args, kwargs):
    logger.debug("OpenAI/Azure OpenAI async embeddings.create wrapper called")

    if is_selective_metering_enabled() and not is_inside_decorated_function():
        return wrapped(*args, **kwargs)

    request_body = kwargs.copy()

    api_metadata = kwargs.pop("usage_metadata", {}) if "usage_metadata" in kwargs else {}

    if not api_metadata:
        api_metadata = _extract_langchain_usage_metadata()

    usage_metadata = merge_metadata(api_metadata)

    client_instance = getattr(instance, '_client', None)
    provider = detect_provider(client=client_instance)

    if is_azure_provider(provider):
        azure_config = get_azure_config()
        if not azure_config.is_valid():
            logger.warning(
                "Azure OpenAI detected but configuration is incomplete. "
                "Set AZURE_OPENAI_ENDPOINT for proper Azure support."
            )
        else:
            logger.debug(f"Azure OpenAI configuration validated: {azure_config.to_dict()}")
            azure_config.validate_deployment()

    request_time_dt = datetime.datetime.now(datetime.timezone.utc)

    # Enforcement pre-call check — may raise ReveniumCostLimitExceeded
    check_enforcement(usage_metadata)

    async def _async_invoke():
        response = await wrapped(*args, **kwargs)

        create_metering_call(
            response,
            OperationType.EMBED,
            request_time_dt,
            usage_metadata,
            client_instance=getattr(instance, '_client', None),
            request_body=request_body
        )

        return response

    return _async_invoke()


if register_patch('openai.resources.chat.completions.Completions.create'):
    wrapt.wrap_function_wrapper(
        'openai.resources.chat.completions', 'Completions.create', create_wrapper
    )

if register_patch('openai.resources.chat.completions.AsyncCompletions.create'):
    wrapt.wrap_function_wrapper(
        'openai.resources.chat.completions', 'AsyncCompletions.create', async_create_wrapper
    )

if register_patch('openai.resources.embeddings.Embeddings.create'):
    wrapt.wrap_function_wrapper(
        'openai.resources.embeddings', 'Embeddings.create', embeddings_create_wrapper
    )

if register_patch('openai.resources.embeddings.AsyncEmbeddings.create'):
    wrapt.wrap_function_wrapper(
        'openai.resources.embeddings', 'AsyncEmbeddings.create', async_embeddings_create_wrapper
    )

if register_patch('openai.resources.responses.Responses.create'):
    wrapt.wrap_function_wrapper(
        'openai.resources.responses', 'Responses.create', responses_create_wrapper
    )
