import contextlib
import datetime
import functools
import importlib
import inspect
import itertools
import logging
import os
import threading
import time
import uuid
import weakref
from collections import OrderedDict
from collections.abc import Mapping
from contextvars import ContextVar
from numbers import Number
from types import SimpleNamespace
from typing import Dict, Any, NamedTuple, Optional, Tuple
from enum import Enum

import anyio
import wrapt

from revenium_middleware import client, get_client, run_async_in_thread, shutdown_event, merge_metadata
from revenium_middleware._core.metering import current_client, recorded_under
from revenium_middleware._core.call_context import CallContext
from revenium_middleware._core.enforcement import check_enforcement, is_bypass_enabled, is_circuit_breaker_enabled
from revenium_middleware._core.exceptions import BudgetExceededError  # noqa: F401 — re-exported via openai.exceptions
from revenium_middleware._core.subscriber import extract_subscriber_from_metadata
from revenium_middleware._core.fields import extract_org_and_product, extract_common_metadata, extract_agentic_job_fields, extract_skill_fields, extract_coding_assistant_fields, extract_service_tier_fields, extract_effort_field, extract_prompt_context_fields, merge_extra_body
from revenium_middleware._core.config import is_capture_prompts_enabled, is_selective_metering_enabled
from revenium_middleware._core.selective_metering import reported_if_unmetered
from revenium_middleware._core.context import (
    _agentic_job_context, is_inside_decorated_function, next_idempotency_key, overlay_metadata,
)
from revenium_middleware._core.parse_failures import metering_failed_parse
from revenium_middleware._core.patch_registry import is_patched, wrap_once
from revenium_middleware._core.call_ownership import (
    OPENAI, PERPLEXITY, bind_model_call_for_cost, call_owned_by_outer_wrapper, claim_call_for_transport,
    mark_model_call_served_by, peek_callback_metadata, with_callback_metadata
)
from revenium_middleware._core import submit_ai_event
from revenium_middleware._core.usage_iterations import billed_iterations, submit_iteration_records
from revenium_middleware._core.failed_calls import (
    ERROR_STOP_REASON, CallFailure, exception_in_flight, failed_transaction_id, failure_of,
)
from revenium_middleware._core.langchain_settings import langchain_usage_metadata
from revenium_middleware._core.log_sanitize import sanitize_for_logging
from revenium_middleware._core.perplexity_cost import extract_reported_cost, read_reported_usage, reported_cost_of
from revenium_middleware._core.perplexity_usage_metadata import take_usage_metadata
from revenium_middleware._core.cache_tokens import (
    billable_input_tokens,
    extract_cache_creation_ttl_counts,
    openai_cache_write_tokens,
    prompt_folds_claude_cache,
    without_openai_cache_writes,
)

# Azure OpenAI support imports
from .provider import Provider, detect_provider, get_provider_metadata, is_azure_provider
from .azure_model_resolver import resolve_azure_model_name
from .raw_response import is_raw_response, tap_async_sse_body, tap_sse_body
from openai import BaseModel as OpenAIBaseModel

try:
    from openai._legacy_response import LegacyAPIResponse
except ImportError:
    LegacyAPIResponse = None
from .azure_config import get_azure_config
from . import xai_usage
from .exceptions import (
    ReveniumMiddlewareError, ValidationError, MeteringError,
    NetworkError, AuthenticationError, categorize_exception, handle_exception_safely
)

# LangChain integration utilities
from .langchain._utils import is_langchain_available

# Prompt capture utilities
from revenium_middleware._core.prompt_extraction import (
    ReplyCollector, extract_prompt_data_if_enabled, extract_prompts_from_request, responses_output_text,
)
from revenium_middleware._core.responses_status import responses_stop_reason as _responses_stop_reason

logger = logging.getLogger("revenium_middleware.extension")


def _get_value(source: Any, key: str, default: Any = None) -> Any:
    """Read a value from either a dict-like or object-like SDK payload."""
    if source is None:
        return default
    if isinstance(source, dict):
        return source.get(key, default)
    return getattr(source, key, default)


def _coerce_int(value: Any, default: int = 0) -> int:
    if value is None:
        return default
    if isinstance(value, bool):
        return default
    if isinstance(value, Number):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return default
    return default


def _extract_cache_token_counts(usage: Any) -> Tuple[int, int]:
    """
    Return (cache_creation_token_count, cache_read_token_count).

    OpenAI reports cached prompt/input tokens as cache reads and
    ``cache_write_tokens`` as cache creation. Anthropic-shaped LangChain
    metadata reports read and creation counts separately.
    """
    input_token_details = _get_value(usage, "input_token_details")
    cache_creation_token_count = _coerce_int(
        _get_value(input_token_details, "cache_creation", 0)
    )
    cache_read_token_count = _coerce_int(
        _get_value(input_token_details, "cache_read", 0)
    )
    if cache_creation_token_count or cache_read_token_count:
        return cache_creation_token_count or openai_cache_write_tokens(usage), cache_read_token_count

    # Anthropic native usage shape: a live ChatAnthropic LLMResult exposes the
    # raw `response_metadata.usage` block with these flat keys rather than the
    # normalized `input_token_details` shape.
    cache_creation_token_count = _coerce_int(
        _get_value(usage, "cache_creation_input_tokens", 0)
    )
    cache_read_token_count = _coerce_int(
        _get_value(usage, "cache_read_input_tokens", 0)
    )
    if cache_creation_token_count or cache_read_token_count:
        return cache_creation_token_count or openai_cache_write_tokens(usage), cache_read_token_count

    input_tokens_details = _get_value(usage, "input_tokens_details")
    cache_read_token_count = _coerce_int(
        _get_value(input_tokens_details, "cached_tokens", 0)
    )
    if not cache_read_token_count:
        prompt_details = _get_value(usage, "prompt_tokens_details")
        cache_read_token_count = _coerce_int(
            _get_value(prompt_details, "cached_tokens", 0)
        )
    return openai_cache_write_tokens(usage), cache_read_token_count


def _billable_prompt_tokens(gross_prompt_tokens, usage, model, cache_creation_token_count, cache_read_token_count):
    """The input count to record for an OpenAI-shaped ``gross_prompt_tokens``."""
    if prompt_folds_claude_cache(model, usage):
        return billable_input_tokens(gross_prompt_tokens, cache_read_token_count, cache_creation_token_count, True)
    return without_openai_cache_writes(gross_prompt_tokens, usage)


def _billed_output_tokens(provider, prompt_tokens, completion_tokens, total_tokens, reasoning_token_count):
    if provider != Provider.XAI:
        return completion_tokens
    return xai_usage.billed_output_tokens(prompt_tokens, completion_tokens, total_tokens, reasoning_token_count)


def _parsed(response):
    """The parsed body of a ``with_raw_response`` result, or the response itself."""
    return response.parse() if callable(getattr(response, 'parse', None)) else response


def _perplexity_reported_cost(response):
    return extract_reported_cost(_parsed(response))


def _xai_reported_cost(response):
    return xai_usage.reported_cost(getattr(_parsed(response), 'usage', None))


def _streamed_reported_cost(provider, usage):
    return xai_usage.reported_cost(usage) if provider == Provider.XAI else None


_PROVIDER_REPORTED_COSTS = {
    "PERPLEXITY": _perplexity_reported_cost,
    "XAI": _xai_reported_cost,
}


def _provider_reported_cost(provider_label, response):
    """The provider's own charge for the call, for the providers that report one."""
    read_cost = _PROVIDER_REPORTED_COSTS.get(provider_label)
    return read_cost(response) if read_cost else None


def _extract_reasoning_token_count(usage: Any) -> int:
    """
    Return the reasoning token count the provider reported, or 0.

    OpenAI reports reasoning tokens in the nested output-token detail block:
    `completion_tokens_details.reasoning_tokens` on Chat Completions and
    `output_tokens_details.reasoning_tokens` on the Responses API. Both blocks
    are absent for non-reasoning models and for surfaces that never report the
    figure, which keeps the count at 0 rather than inventing one.
    """
    completion_tokens_details = _get_value(usage, "completion_tokens_details")
    reasoning_token_count = _coerce_int(
        _get_value(completion_tokens_details, "reasoning_tokens", 0)
    )
    if reasoning_token_count:
        return reasoning_token_count

    output_tokens_details = _get_value(usage, "output_tokens_details")
    return _coerce_int(
        _get_value(output_tokens_details, "reasoning_tokens", 0)
    )


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

    # Extract provider-specific cache token details.
    cache_creation_token_count, cache_read_token_count = _extract_cache_token_counts(
        response.usage
    )
    reasoning_token_count = _extract_reasoning_token_count(response.usage)

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
            gross_input_tokens = response.usage.input_tokens
            output_tokens = response.usage.output_tokens
        else:
            # Chat Completions format
            gross_input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens
        input_tokens = _billable_prompt_tokens(
            gross_input_tokens, response.usage, raw_model_name, cache_creation_token_count, cache_read_token_count
        )

        total_tokens = response.usage.total_tokens
        output_tokens = _billed_output_tokens(
            provider, gross_input_tokens, output_tokens, total_tokens, reasoning_token_count
        )

        # Get stop reason - Responses API may not have choices
        if hasattr(response, 'choices') and response.choices:
            openai_finish_reason = response.choices[0].finish_reason
            stop_reason = get_stop_reason(openai_finish_reason)
        else:
            stop_reason = _responses_stop_reason(response)

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
        "cache_creation_token_count": cache_creation_token_count,
        "cache_read_token_count": cache_read_token_count,
        "cache_creation_ttl_counts": extract_cache_creation_ttl_counts(response.usage),
        "reasoning_token_count": reasoning_token_count,
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
    logger.debug("Provider detected: %s, metadata: %s", provider, sanitize_for_logging(provider_metadata))
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
        ticket_id: Optional[str] = None,
        agent_version: Optional[str] = None,
        # Prompt capture fields
        system_prompt: Optional[str] = None,
        input_messages: Optional[str] = None,
        output_response: Optional[str] = None,
        prompts_truncated: Optional[bool] = None,
        cache_creation_token_count: Optional[int] = None,
        cache_read_token_count: Optional[int] = None,
        reasoning_token_count: Optional[int] = None,
        reported_cost: Optional[Dict[str, float]] = None,
        cache_creation_ttl_counts: Optional[Dict[str, int]] = None,
        usage_iterations: Any = None,
        iteration_request: Optional[Dict[str, Any]] = None,
        error_reason: Optional[str] = None,
        idempotency_key: Optional[str] = None,
        error_code: Optional[int] = None,
) -> None:
    """Log token usage to Revenium.

    ``usage_iterations`` is a provider usage (or its iterations list) that may
    hold Claude advisor or compaction passes, each of which is sent as its own
    record after the main one; ``iteration_request`` is the request those
    passes take their model from. ``error_reason``, ``idempotency_key`` and
    ``error_code`` are set only on the record of a failed call.
    """
    if current_client() is None:
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
    if cache_creation_token_count is None:
        cache_creation_tokens = 0
    else:
        cache_creation_tokens = cache_creation_token_count
    if cache_read_token_count is None:
        cache_read_token_count = _coerce_int(cached_tokens)
    # A caller that reports nothing keeps the field at 0 rather than omitting
    # it, so the emitted payload shape is unchanged for non-reasoning traffic.
    reasoning_tokens = _coerce_int(reasoning_token_count)

    completion_args = {
        "cache_creation_token_count": cache_creation_tokens,
        "cache_read_token_count": cache_read_token_count,
        "output_token_count": completion_tokens,
        "cost_type": "AI",
        "model": model,
        "input_token_count": prompt_tokens,
        "provider": provider,
        "model_source": model_source,
        "reasoning_token_count": reasoning_tokens,
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
    if meta["response_quality_score"] is not None:
        completion_args["response_quality_score"] = meta["response_quality_score"]

    agentic_fields = extract_agentic_job_fields(usage_metadata)
    if agentic_fields:
        completion_args["extra_body"] = merge_extra_body(completion_args.get("extra_body"), agentic_fields)

    # Skill attribution fields are first-class typed params, not extra_body
    skill_fields = extract_skill_fields(usage_metadata)
    if skill_fields:
        completion_args.update(skill_fields)

    # Coding assistant attribution is likewise a first-class typed param
    coding_assistant_fields = extract_coding_assistant_fields(usage_metadata)
    if coding_assistant_fields:
        completion_args.update(coding_assistant_fields)

    # Service tier and pricing fields are likewise first-class typed params
    service_tier_fields = extract_service_tier_fields(usage_metadata, "completion")
    if service_tier_fields:
        completion_args.update(service_tier_fields)

    # Reasoning effort is a caller-supplied pass-through: forwarded verbatim,
    # omitted entirely when unset.
    completion_args.update(extract_effort_field(usage_metadata))
    completion_args.update(extract_prompt_context_fields(usage_metadata))
    completion_args.update(reported_cost or {})
    completion_args.update(cache_creation_ttl_counts or {})

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
    if ticket_id:
        completion_args["ticket_id"] = ticket_id
    if agent_version:
        completion_args["agent_version"] = agent_version

    # Add prompt capture fields only if they have values
    if system_prompt is not None:
        completion_args["system_prompt"] = system_prompt
    if input_messages is not None:
        completion_args["input_messages"] = input_messages
    if output_response is not None:
        completion_args["output_response"] = output_response
    if prompts_truncated is not None:
        completion_args["prompts_truncated"] = prompts_truncated
    if error_reason is not None:
        completion_args["error_reason"] = error_reason
    if error_code is not None:
        completion_args["error_code"] = error_code

    # Log the arguments at debug level (redact sensitive prompt data)
    safe_args = {k: v for k, v in completion_args.items()
                 if k not in ('system_prompt', 'input_messages', 'output_response')}
    if 'system_prompt' in completion_args:
        safe_args['system_prompt'] = '[REDACTED]'
    if 'input_messages' in completion_args:
        safe_args['input_messages'] = '[REDACTED]'
    if 'output_response' in completion_args:
        safe_args['output_response'] = '[REDACTED]'
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug("Calling client.ai.create_completion with args: %s", sanitize_for_logging(safe_args))

    # Debug logging for metering call
    logger.debug(f"Metering call for {operation_type.value}: {response_id}, tokens: {prompt_tokens}+{completion_tokens}={total_tokens}")

    # Drawn here only when a pass is billed, so the pass records can derive their
    # keys from it; every other call leaves the draw to submit_ai_event as before.
    main_key = idempotency_key or (next_idempotency_key() if billed_iterations(usage_iterations) else None)
    try:
        # The client.ai.create_completion method is not async, so don't use await
        if main_key is None:
            result = submit_ai_event("completion", completion_args)
        else:
            result = submit_ai_event("completion", completion_args, idempotency_key=main_key)
        logger.debug("Metering call result: %s", result)
        if result is not None:  # None: submit_ai_event buffered the record instead of sending it
            logger.debug("REVENIUM SUCCESS: Metering call successful: %s", result.id)
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
    submit_iteration_records(submit_ai_event, completion_args, usage_iterations, iteration_request, main_key)


def create_metering_call(
    response,
    operation_type: OperationType,
    request_time_dt,
    usage_metadata,
    client_instance: Optional[Any] = None,
    time_to_first_token: int = 0,
    is_streamed: bool = False,
    request_body: Optional[Dict[str, Any]] = None,
    provider_metadata: Optional[Dict[str, str]] = None,
    usage_iterations: Any = None,
    iteration_request: Optional[Dict[str, Any]] = None,
    stop_reason: Optional[str] = None,
    failure: Optional[CallFailure] = None,
    request_prompts: Optional[Dict[str, Any]] = None,
    accumulated_content: Optional[str] = None,
):
    """
    Unified function to create and execute metering calls for any operation
    type. Reduces duplication between chat and embeddings wrappers.

    ``provider_metadata`` (``provider`` and ``model_source``) replaces the
    labels detected from ``client_instance``, for callers that know the
    provider without holding its client. ``stop_reason`` likewise replaces the
    one read from the response, for callers that know it without an OpenAI
    response.

    ``usage_iterations`` and ``iteration_request`` are passed to
    ``log_token_usage`` unchanged. ``failure`` marks the record of a failed
    call: stop reason ERROR, its error reason and key, no captured prompt.

    ``request_prompts`` is the captured prompt of a request that is not a chat
    body, so ``request_body`` keeps deciding the operation subtype alone.
    ``accumulated_content`` is a streamed call's reply text.
    """
    # Import trace field functions
    from .trace_fields import (
        get_environment, get_region, get_credential_alias,
        get_trace_type, get_trace_name, get_parent_transaction_id,
        get_transaction_name, get_retry_number, detect_operation_type,
        validate_trace_type, validate_trace_name, get_ticket_id,
        get_agent_version
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

    if provider_metadata:
        usage_data.update(provider_metadata)
    if stop_reason:
        usage_data["stop_reason"] = stop_reason
    if failure is not None:
        usage_data["stop_reason"] = ERROR_STOP_REASON
    reported_cost = _provider_reported_cost(usage_data.get("provider"), response)

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
    ticket_id = get_ticket_id(usage_metadata)
    agent_version = get_agent_version(usage_metadata)
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
    retry_number = get_retry_number(usage_metadata)

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

    if operation_type == OperationType.CHAT and failure is None:
        (system_prompt, input_messages, output_response, prompts_truncated) = (
            extract_prompt_data_if_enabled(request_body, response=_parsed(response),
                                           accumulated_content=accumulated_content,
                                           request_prompts=request_prompts)
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
            cache_creation_token_count=usage_data["cache_creation_token_count"],
            cache_read_token_count=usage_data["cache_read_token_count"],
            cache_creation_ttl_counts=usage_data["cache_creation_ttl_counts"],
            reasoning_token_count=usage_data["reasoning_token_count"],
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
            ticket_id=ticket_id,
            agent_version=agent_version,
            # Prompt capture fields
            system_prompt=system_prompt,
            input_messages=input_messages,
            output_response=output_response,
            prompts_truncated=prompts_truncated,
            reported_cost=reported_cost,
            usage_iterations=usage_iterations,
            iteration_request=iteration_request,
            error_reason=failure.error_reason if failure else None,
            idempotency_key=failure.idempotency_key if failure else None,
            error_code=failure.error_code if failure else None,
        )

    # Start metering thread
    thread = run_async_in_thread(metering_call())
    logger.debug("%s metering thread started: %s", operation_type, thread)
    return thread


def _extract_langchain_usage_metadata():
    """The ``usage_metadata`` passed to LangChain for this call, or ``{}``
    when LangChain is not installed (an optional integration)."""
    if not is_langchain_available():
        return {}
    return langchain_usage_metadata()


def embeddings_create_wrapper(wrapped, instance, args, kwargs):
    logger.debug("OpenAI/Azure OpenAI embeddings.create wrapper called")

    if _left_unmetered():
        return _call_unmetered(wrapped, instance, args, kwargs)

    request_body = kwargs.copy()

    # Extract API-level metadata from kwargs
    api_metadata = _take_call_metadata(instance, kwargs) or {}

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

    # Enforcement pre-call check — may raise BudgetExceededError
    check_enforcement(usage_metadata, model=kwargs.get("model"), provider=get_provider_metadata(provider)["provider"])

    logger.debug(
        "Calling wrapped embeddings function with args: %s, kwargs: %s",
        sanitize_for_logging(args),
        sanitize_for_logging(kwargs),
    )

    # Call the original OpenAI function
    response = wrapped(*args, **kwargs)

    logger.debug("Handling embeddings response: %s", sanitize_for_logging(response))

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


def _response_id_of(response):
    parse = getattr(response, 'parse', None)
    if callable(parse):
        response = parse()
    return getattr(response, 'id', None)


def _meter_owned_response(response, *args, **kwargs):
    """``create_metering_call`` for a response this transport wrap meters,
    claiming the call once its record is sent so an attached LangChain
    callback does not send a second one."""
    if create_metering_call(response, *args, **kwargs) is not None:
        claim_call_for_transport(OPENAI, _response_id_of(response))


def _with_forced_usage_reporting(kwargs):
    """Return a copy of ``kwargs`` with ``stream_options.include_usage`` on,
    plus whether the caller had asked for usage themselves.

    The extra usage chunk the API then appends is hidden from a caller who did
    not ask for it (see ``_is_injected_usage_chunk``).
    """
    # OpenAI's stream() helpers always pass stream_options, as the library's
    # Omit() placeholder when the caller set none; anything that is not a
    # mapping means "not supplied".
    caller_options = kwargs.get('stream_options')
    if not isinstance(caller_options, Mapping):
        caller_options = {}
    forced = dict(kwargs)
    forced['stream_options'] = {**caller_options, 'include_usage': True}
    return forced, bool(caller_options.get('include_usage'))


def _http_response_of(stream):
    """The ``httpx.Response`` behind an OpenAI stream, which OpenAI's
    ``stream()`` helpers read as ``raw_stream.response`` from whatever
    ``create(stream=True)`` returns."""
    return getattr(stream, 'response', None)


# OpenAI sends usage in the single chunk after the finish chunk; the bounds
# protect against n > 1 or a compatible provider that finishes early or stalls.
_MAX_CHUNKS_READ_AFTER_FINISH = 8
_SECONDS_TO_WAIT_FOR_USAGE_AFTER_FINISH = 2.0


class _ResponseThatMetersOnClose(wrapt.ObjectProxy):
    """The ``httpx.Response`` of a metered chat stream. OpenAI's ``stream()``
    helpers close this response, never the stream, so closing it finishes
    the stream's metering first. The stream is held weakly: an abandoned
    stream is still recorded as soon as it is dropped."""

    def __init__(self, response, stream_wrapper):
        super().__init__(response)
        self._self_owner = weakref.ref(stream_wrapper)
        self._self_closed = False

    def _owner_once(self):
        if self._self_closed:
            return None
        self._self_closed = True
        return self._self_owner()

    def close(self):
        try:
            owner = self._owner_once()
            if owner is not None:
                owner._finish()
        except Exception as error:
            logger.debug("Could not finish metering a closed chat stream: %s", error)
        finally:
            self.__wrapped__.close()

    async def aclose(self):
        try:
            owner = self._owner_once()
            if owner is not None:
                await owner._afinish()
        except Exception as error:
            logger.debug("Could not finish metering a closed chat stream: %s", error)
        finally:
            await self.__wrapped__.aclose()


def _usage_still_to_come(stream_wrapper, still_open):
    return still_open and stream_wrapper.finish_reason is not None and stream_wrapper.final_usage is None


def _log_unread_usage(error):
    logger.debug("Could not read the usage chunk of a finished chat stream: %s", type(error).__name__)


def _read_usage_after_finish(stream_wrapper, still_open):
    """Read on to the usage chunk that follows the finish chunk, when it is
    still unread. Never raises."""
    if not _usage_still_to_come(stream_wrapper, still_open):
        return
    deadline = time.monotonic() + _SECONDS_TO_WAIT_FOR_USAGE_AFTER_FINISH
    try:
        for chunk in itertools.islice(stream_wrapper.stream, _MAX_CHUNKS_READ_AFTER_FINISH):
            stream_wrapper._observe(chunk)
            if stream_wrapper.final_usage is not None or time.monotonic() >= deadline:
                return
    except Exception as error:
        _log_unread_usage(error)


async def _aread_usage_after_finish(stream_wrapper, still_open):
    """The async form of ``_read_usage_after_finish``."""
    if not _usage_still_to_come(stream_wrapper, still_open):
        return
    try:
        with anyio.move_on_after(_SECONDS_TO_WAIT_FOR_USAGE_AFTER_FINISH):
            for _ in range(_MAX_CHUNKS_READ_AFTER_FINISH):
                stream_wrapper._observe(await stream_wrapper.stream.__anext__())
                if stream_wrapper.final_usage is not None:
                    return
    except StopAsyncIteration:
        return
    except Exception as error:
        _log_unread_usage(error)


def _response_that_meters_on_close(stream, stream_wrapper):
    response = _http_response_of(stream)
    return None if response is None else _ResponseThatMetersOnClose(response, stream_wrapper)


_parse_call_in_progress: ContextVar[bool] = ContextVar("revenium_openai_parse_call_in_progress", default=False)


@contextlib.contextmanager
def _metered_by_parse():
    """Mark the enclosed provider call as metered by a ``parse()`` wrap, so a
    ``create()`` that an OpenAI release routes ``parse()`` through stays
    unmetered instead of sending a second record."""
    token = _parse_call_in_progress.set(True)
    try:
        yield
    finally:
        _parse_call_in_progress.reset(token)


_stream_helper_usage_metadata: ContextVar[Optional[Mapping]] = ContextVar(
    "revenium_openai_stream_helper_usage_metadata", default=None)


@contextlib.contextmanager
def _tagged_by_stream_helper(usage_metadata):
    """Hand a ``stream()`` helper's ``usage_metadata`` to the ``create()`` the
    helper calls, without sending it to OpenAI."""
    token = _stream_helper_usage_metadata.set(usage_metadata)
    try:
        yield
    finally:
        _stream_helper_usage_metadata.reset(token)


def _take_call_metadata(instance, kwargs):
    """Remove the call's own Revenium metadata from ``kwargs`` and return it.

    A Perplexity-bound call also carries it inside ``extra_body``, which
    must not reach Perplexity either."""
    if _is_perplexity_bound(instance):
        return take_usage_metadata(kwargs)
    return kwargs.pop("usage_metadata", None)


def _pop_usage_metadata(kwargs, instance=None):
    """The call's own ``usage_metadata``, else the one its ``stream()`` helper was given.

    A Perplexity-bound call merges the two field by field instead: the
    helper's keyword wins over the call's own ``extra_body`` metadata, as the
    keyword wins everywhere else on Perplexity."""
    if _is_perplexity_bound(instance):
        return {**take_usage_metadata(kwargs), **(_stream_helper_usage_metadata.get() or {})}
    return kwargs.pop("usage_metadata", None) or _stream_helper_usage_metadata.get()


class _TaggedSyncStreamManager(wrapt.ObjectProxy):
    """A sync ``stream()`` helper's manager, which calls ``create()`` only when
    its ``with`` block is entered."""

    def __init__(self, manager, usage_metadata):
        super().__init__(manager)
        self._self_usage_metadata = usage_metadata

    def __enter__(self):
        with _tagged_by_stream_helper(self._self_usage_metadata):
            return self.__wrapped__.__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        return self.__wrapped__.__exit__(exc_type, exc_val, exc_tb)


def _call_stream_helper(wrapped, args, kwargs):
    usage_metadata = kwargs.pop("usage_metadata", None)
    if not usage_metadata:
        return wrapped(*args, **kwargs), None
    with _tagged_by_stream_helper(usage_metadata):
        return wrapped(*args, **kwargs), usage_metadata


def stream_helper_wrapper(wrapped, instance, args, kwargs):
    manager, usage_metadata = _call_stream_helper(wrapped, args, kwargs)
    return manager if usage_metadata is None else _TaggedSyncStreamManager(manager, usage_metadata)


def async_stream_helper_wrapper(wrapped, instance, args, kwargs):
    # The async helpers call create() inside stream() itself.
    manager, _ = _call_stream_helper(wrapped, args, kwargs)
    return manager


def _selectively_unmetered():
    return reported_if_unmetered(is_selective_metering_enabled() and not is_inside_decorated_function())


def _left_unmetered():
    """Whether this call gets only its spending-cap check here, no record:
    selective metering skips it, or an outer integration (LiteLLM) whose
    provider call runs through this client meters it. The check stays,
    because a LiteLLM call the proxy serves was not checked by LiteLLM's
    wrapper; one it did check is marked so and not checked twice."""
    return call_owned_by_outer_wrapper() or _selectively_unmetered()


def _take_api_metadata(kwargs, instance=None):
    """Pop the Revenium-only ``usage_metadata`` keyword, which the OpenAI SDK rejects."""
    api_metadata = _pop_usage_metadata(kwargs, instance)
    return dict(api_metadata) if isinstance(api_metadata, Mapping) else {}


def _check_unmetered_call(api_metadata, model=None, provider=None):
    """Run the spending-cap check for a call selective metering leaves
    unmetered. No record is sent for it, so the LangChain caller's
    attribution is read without being taken."""
    if not is_circuit_breaker_enabled() or is_bypass_enabled():
        return
    if not api_metadata:
        langchain_metadata = _extract_langchain_usage_metadata()
        api_metadata = langchain_metadata if isinstance(langchain_metadata, Mapping) else {}
    check_enforcement(merge_metadata(overlay_metadata(peek_callback_metadata(OPENAI), api_metadata)),
                      model=model, provider=provider)


def _check_unmetered_client_call(instance, kwargs, cap_provider_label):
    _check_unmetered_call(_take_api_metadata(kwargs, instance), kwargs.get("model"), cap_provider_label(instance))


def _call_unmetered(wrapped, instance, args, kwargs):
    _check_unmetered_client_call(instance, kwargs, _client_cap_provider_label)
    return wrapped(*args, **kwargs)


def _call_responses_unmetered(wrapped, instance, args, kwargs):
    _check_unmetered_client_call(instance, kwargs, _chat_cap_provider_label)
    return wrapped(*args, **kwargs)


def _tell_langchain_when_perplexity_serves(instance):
    """A LangChain callback records a call this transport leaves unmetered,
    and only the client knows it calls Perplexity."""
    if _is_perplexity_bound(instance):
        mark_model_call_served_by(OPENAI, PERPLEXITY)


def _is_perplexity_bound(instance):
    """Whether this chat call targets Perplexity through the OpenAI client.

    The Perplexity middleware patches the same Completions.create slot. When
    both middlewares are loaded the wrappers chain via wrapt, so Perplexity-bound
    calls are left to the Perplexity wrapper to avoid double-metering.
    """
    client_instance = getattr(instance, '_client', None)
    base_url = getattr(client_instance, 'base_url', None) if client_instance else None
    return bool(base_url) and "perplexity" in str(base_url).lower()


_PERPLEXITY_CHAT_WRAP = "perplexity:openai.resources.chat.completions.Completions.create"
_PERPLEXITY_ASYNC_CHAT_WRAP = "perplexity:openai.resources.chat.completions.AsyncCompletions.create"


def _call_perplexity_bound(wrapped, instance, args, kwargs, perplexity_wrap):
    """Leave the call to the Perplexity wrapper when it is installed. Without
    it nothing else checks this call, so a breached cap must refuse it here.
    A LangChain model making the call is told Perplexity serves it, and a
    stream it makes hands it what Perplexity reports on the way through."""
    mark_model_call_served_by(OPENAI, PERPLEXITY)
    if not is_patched(perplexity_wrap):
        _check_unmetered_call(take_usage_metadata(kwargs), kwargs.get("model"), _chat_cap_provider_label(instance))
    model_call = bind_model_call_for_cost(OPENAI) if kwargs.get("stream") else ()
    if not model_call:
        return wrapped(*args, **kwargs)
    observe = functools.partial(_keep_what_perplexity_reported, model_call)
    result = wrapped(*args, **kwargs)
    if inspect.isawaitable(result):
        return _read_perplexity_stream_when_awaited(result, observe)
    return _read_perplexity_stream(result, observe, tap_sse_body)


def _keep_what_perplexity_reported(model_call, item):
    """Leave the last charge and usage a Perplexity stream reported on the
    LangChain run's publications, for the callback's one record.
    langchain-openai drops the charge from a stream, and adds up the usage
    Perplexity repeats, so far, on every chunk."""
    try:
        cost = reported_cost_of(item)
        usage = getattr(item, "usage", None)
    except Exception as error:  # a read-only reader must never fail the caller's stream
        logger.debug("Could not read what Perplexity reported: %s", error)
        return
    for publication in model_call:
        if cost is not None:
            publication.reported_cost = cost
        if usage:
            publication.reported_usage = usage


def _read_perplexity_stream(result, observe, tap_body):
    if is_raw_response(result):
        tap_body(result, observe, _nothing_to_finish)
        return result
    return read_reported_usage(result, observe)


async def _read_perplexity_stream_when_awaited(pending, observe):
    return _read_perplexity_stream(await pending, observe, tap_async_sse_body)


def _nothing_to_finish():
    pass


def _validate_azure_configuration(client_instance):
    if not is_azure_provider(detect_provider(client=client_instance)):
        return
    azure_config = get_azure_config()
    if not azure_config.is_valid():
        logger.warning(
            "Azure OpenAI detected but configuration is incomplete. "
            "Set AZURE_OPENAI_ENDPOINT for proper Azure support."
        )
        return
    logger.debug(f"Azure OpenAI configuration validated: {azure_config.to_dict()}")
    azure_config.validate_deployment()


_PERPLEXITY_LABELS = {"provider": "PERPLEXITY", "model_source": "PERPLEXITY"}


def _cap_provider_label(client_instance, provider_metadata=None):
    """The provider label this call's metering record will carry, for a cap
    filtered by provider. Only worked out when the cap check will run."""
    if not is_circuit_breaker_enabled():
        return None
    if provider_metadata is None:
        base_url = getattr(client_instance, 'base_url', None) if client_instance else None
        provider_metadata = get_provider_metadata(detect_provider(client_instance, base_url))
    return provider_metadata["provider"]


def _client_cap_provider_label(instance):
    return _cap_provider_label(getattr(instance, '_client', None))


def _chat_provider_metadata(instance):
    return _PERPLEXITY_LABELS if _is_perplexity_bound(instance) else None


def _chat_cap_provider_label(instance):
    return _cap_provider_label(getattr(instance, '_client', None), _chat_provider_metadata(instance))


class _MeteredCall(NamedTuple):
    usage_metadata: Dict[str, Any]
    request_time_dt: datetime.datetime
    client_instance: Any
    request_body: Optional[Dict[str, Any]]
    provider_metadata: Optional[Dict[str, str]] = None
    request_prompts: Optional[Dict[str, Any]] = None


def _begin_chat_call(instance, kwargs) -> _MeteredCall:
    """Resolve a chat call's usage metadata, validate the Azure configuration
    and run the enforcement pre-check (which may raise ``BudgetExceededError``)
    before the provider is called. Pops ``usage_metadata`` from ``kwargs``."""
    request_body = kwargs.copy()
    api_metadata = _pop_usage_metadata(kwargs, instance)
    if not api_metadata:
        api_metadata = _extract_langchain_usage_metadata()
    usage_metadata = merge_metadata(with_callback_metadata(OPENAI, api_metadata))
    client_instance = getattr(instance, '_client', None)
    _validate_azure_configuration(client_instance)
    request_time_dt = datetime.datetime.now(datetime.timezone.utc)
    provider_metadata = _chat_provider_metadata(instance)
    check_enforcement(usage_metadata, model=kwargs.get("model"),
                      provider=_cap_provider_label(client_instance, provider_metadata))
    return _MeteredCall(usage_metadata, request_time_dt, client_instance, request_body, provider_metadata)


def _elapsed_ms_since(request_time_dt):
    return int((datetime.datetime.now(datetime.timezone.utc) - request_time_dt).total_seconds() * 1000)


def _meter_chat_completion(response, call: _MeteredCall):
    _meter_owned_response(
        response,
        OperationType.CHAT,
        call.request_time_dt,
        call.usage_metadata,
        client_instance=call.client_instance,
        time_to_first_token=_elapsed_ms_since(call.request_time_dt),
        request_body=call.request_body,
        provider_metadata=call.provider_metadata
    )


def _failed_response(model):
    return SimpleNamespace(id=failed_transaction_id(), model=model or "unknown", usage=_ZERO_RESPONSES_USAGE,
                           system_fingerprint=None, choices=[])


@contextlib.contextmanager
def _recording_failure(call: _MeteredCall, model, is_streamed, metering_client):
    """Send one ERROR record for a call the provider refused, under the
    metering client current when the call started, then let the caller's
    exception through unchanged."""
    outer = exception_in_flight()
    try:
        yield
    except Exception as error:
        failure = failure_of(error, outer)
        if failure is not None:
            try:
                with recorded_under(metering_client):
                    create_metering_call(
                        _failed_response(model), OperationType.CHAT, call.request_time_dt, call.usage_metadata,
                        client_instance=call.client_instance, is_streamed=bool(is_streamed),
                        request_body=call.request_body, provider_metadata=call.provider_metadata, failure=failure,
                    )
            except Exception as metering_error:
                logger.warning("Could not record a failed OpenAI call: %s", metering_error)
        raise


def create_wrapper(wrapped, instance, args, kwargs):
    logger.debug("OpenAI/Azure OpenAI chat.completions.create wrapper called")

    if _parse_call_in_progress.get():
        return wrapped(*args, **kwargs)
    if _is_perplexity_bound(instance):
        return _call_perplexity_bound(wrapped, instance, args, kwargs, _PERPLEXITY_CHAT_WRAP)
    if _left_unmetered():
        return _call_unmetered(wrapped, instance, args, kwargs)

    call = _begin_chat_call(instance, kwargs)

    stream = kwargs.get('stream', False)
    caller_requested_usage = False
    if stream:
        kwargs, caller_requested_usage = _with_forced_usage_reporting(kwargs)

    logger.debug(
        "Calling wrapped function with args: %s, kwargs: %s",
        sanitize_for_logging(args),
        sanitize_for_logging(kwargs),
    )

    metering_client = current_client()
    with _recording_failure(call, kwargs.get("model"), stream, metering_client):
        response = wrapped(*args, **kwargs)

    if stream:
        logger.debug("Handling streaming response")
        with recorded_under(metering_client):
            return handle_streaming_response(
                response,
                call.request_time_dt,
                call.usage_metadata,
                client_instance=call.client_instance,
                request_body=call.request_body,
                caller_requested_usage=caller_requested_usage
            )

    logger.debug("Handling non-streaming response: %s", sanitize_for_logging(response))
    _meter_chat_completion(response, call)
    return response


def _answer_text_of(content):
    """The answer text of a streamed delta's content: a string as is, or the
    text parts of a list of parts (Mistral's thinking parts are not answer text)."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    texts = (_get_value(part, "text") for part in content if _get_value(part, "type") == "text")
    return "".join(text for text in texts if isinstance(text, str))


def _observing_chunk(process_chunk, chunk):
    """Meter one chunk without ever breaking the caller's read: an error in
    the middleware's own reading is logged, never raised or recorded as a
    provider failure."""
    try:
        process_chunk(chunk)
    except Exception as error:
        logger.warning("Could not meter a chat stream chunk (%s); the stream continues", type(error).__name__)


def _is_injected_usage_chunk(chunk, caller_requested_usage):
    """The final empty-choices usage chunk only exists because the middleware
    forced stream_options.include_usage; hide it from the caller unless they
    asked for usage themselves."""
    if caller_requested_usage:
        return False
    return not getattr(chunk, 'choices', None) and bool(getattr(chunk, 'usage', None))


def handle_streaming_response(
    stream,
    request_time_dt,
    usage_metadata,
    client_instance: Optional[Any] = None,
    request_body: Optional[Dict[str, Any]] = None,
    caller_requested_usage: bool = False
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
            self.stream = iter(stream)
            self.response = _response_that_meters_on_close(stream, self)
            self._metering_client = current_client()
            self._call_context = CallContext()
            self._saw_chunk = False
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
            self.caller_requested_usage = caller_requested_usage
            self._closed = False
            self._usage_logged = False
            self.failure = None
            self._outer = exception_in_flight()

        def __iter__(self):
            return self

        def __next__(self):
            if self._closed:
                raise StopIteration("Stream has been closed")

            while True:
                try:
                    chunk = next(self.stream)
                except StopIteration:
                    self._finalize()
                    raise
                except Exception as e:
                    # Ensure cleanup on any error
                    self._record_failure(e)
                    self._finalize()
                    logger.error(f"Error in streaming response: {e}")
                    raise

                self._observe(chunk)
                if not self._injected_usage_chunk(chunk):
                    return chunk

        def _observe(self, chunk):
            _observing_chunk(self._process_chunk, chunk)
            self._saw_chunk = True

        def _injected_usage_chunk(self, chunk):
            return _is_injected_usage_chunk(chunk, self.caller_requested_usage)

        def _record_failure(self, error):
            self.failure = self._call_context.run(failure_of, error, self._outer)

        def __enter__(self):
            """Context manager entry."""
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            """Context manager exit with cleanup."""
            self._finish()

        def close(self):
            """Close the stream, metering whatever was received so far."""
            self._finish()

        def _finish(self):
            try:
                _read_usage_after_finish(self, still_open=not self._closed)
            finally:
                self._finalize()

        def __del__(self):
            # Last-resort finalize: a broken-out-of loop leaves the wrapper to
            # the GC with no StopIteration/__exit__ ever firing.
            try:
                self._finalize()
            except Exception:
                pass

        def _finalize(self):
            """Finalize the stream and log usage if not already done."""
            if not self._usage_logged:
                self._call_context.run(self._log_usage)
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
            # The first non-empty id and model: Azure opens a stream with a
            # content-filter chunk whose id and model are empty.
            self.response_id = self.response_id or getattr(chunk, 'id', None)
            self.model = self.model or getattr(chunk, 'model', None)
            if self.system_fingerprint is None and hasattr(chunk, 'system_fingerprint'):
                self.system_fingerprint = chunk.system_fingerprint
                logger.debug(f"Captured system_fingerprint from stream chunk: {self.system_fingerprint}")
            else:
                logger.debug(f"System fingerprint already set: {self.system_fingerprint}")


            choices = getattr(chunk, 'choices', None)
            if choices and getattr(choices[0], 'finish_reason', None):
                self.finish_reason = choices[0].finish_reason

            # Check if this chunk has usage data (can be in final chunk with or without choices)
            if hasattr(chunk, 'usage') and chunk.usage:
                logger.debug(f"Found usage data in chunk: {chunk.usage}")
                self.final_usage = chunk.usage
                # Don't return yet - we still need to process the chunk for finish_reason etc.

            # Collect content for token estimation if needed
            if choices and hasattr(choices[0], 'delta') and hasattr(choices[0].delta, 'content') and \
                    choices[0].delta.content:
                # Record time of first token if not already set
                if self.first_token_time is None:
                    self.first_token_time = datetime.datetime.now(datetime.timezone.utc)
                self.completion_text += _answer_text_of(choices[0].delta.content)

            self._saw_chunk = True

        def _log_usage(self):
            # An empty stream sends no record, unless it failed before its first chunk.
            if not self._saw_chunk and not self.final_usage and self.failure is None:
                return

            # Record response time and calculate duration
            response_time_dt = datetime.datetime.now(datetime.timezone.utc)
            response_time = response_time_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
            request_duration = (response_time_dt - self.request_time_dt).total_seconds() * 1000

            # Get token usage information
            prompt_tokens = 0
            completion_tokens = 0
            total_tokens = 0
            cache_creation_token_count = 0
            cache_read_token_count = 0
            reasoning_token_count = 0
            reported_cost = None
            cache_creation_ttl_counts = {}
            provider = detect_provider(self.client_instance,
                                       getattr(self.client_instance, 'base_url', None) if self.client_instance else None)

            # First check if we have the final usage data from the special chunk
            if self.final_usage:
                cache_creation_ttl_counts = extract_cache_creation_ttl_counts(self.final_usage)
                cache_creation_token_count, cache_read_token_count = _extract_cache_token_counts(
                    self.final_usage
                )
                prompt_tokens = _billable_prompt_tokens(
                    self.final_usage.prompt_tokens, self.final_usage, self.model,
                    cache_creation_token_count, cache_read_token_count,
                )
                total_tokens = self.final_usage.total_tokens
                reasoning_token_count = _extract_reasoning_token_count(
                    self.final_usage
                )
                completion_tokens = _billed_output_tokens(
                    provider, self.final_usage.prompt_tokens, self.final_usage.completion_tokens,
                    total_tokens, reasoning_token_count,
                )
                reported_cost = _streamed_reported_cost(provider, self.final_usage)
                logger.debug(
                    f"Using token usage from final chunk: prompt={prompt_tokens}, completion={completion_tokens}, total={total_tokens}")
            else:
                # If we don't have usage data, estimate from content
                logger.warning("No usage data found in streaming response!")

            failure = self.failure
            stop_reason = ERROR_STOP_REASON if failure else get_stop_reason(self.finish_reason)
            response_id = self.response_id or (failed_transaction_id() if failure else None)

            # Log the token usage
            if response_id:
                logger.debug(
                    "Streaming token usage - response_id: %s, prompt: %d, completion: %d, total: %d",
                    response_id, prompt_tokens, completion_tokens, total_tokens
                )

                provider_metadata = get_provider_metadata(provider)

                # Resolve model name for Azure deployments
                raw_model_name = self.model or (failure and (self.request_body or {}).get("model")) or "unknown"
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
                    validate_trace_type, validate_trace_name,
                    get_ticket_id, get_agent_version
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
                ticket_id = get_ticket_id(self.usage_metadata)
                agent_version = get_agent_version(self.usage_metadata)
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
                retry_number = get_retry_number(self.usage_metadata)

                # Detect operation type and subtype
                operation_info = detect_operation_type(
                    provider, "/chat/completions", self.request_body or {}
                )
                operation_subtype = operation_info.get('operationSubtype')

                # Extract prompt data if capture is enabled
                (system_prompt, input_messages, output_response, prompts_truncated) = (
                    extract_prompt_data_if_enabled(
                        None if failure else self.request_body,
                        accumulated_content=self.completion_text
                    )
                )

                async def metering_call():
                    await log_token_usage(
                        response_id=response_id,
                        model=resolved_model_name,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        total_tokens=total_tokens,
                        cached_tokens=0,
                        cache_creation_token_count=cache_creation_token_count,
                        cache_creation_ttl_counts=cache_creation_ttl_counts,
                        cache_read_token_count=cache_read_token_count,
                        reasoning_token_count=reasoning_token_count,
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
                        ticket_id=ticket_id,
                        agent_version=agent_version,
                        # Prompt capture fields
                        system_prompt=system_prompt,
                        input_messages=input_messages,
                        output_response=output_response,
                        prompts_truncated=prompts_truncated,
                        reported_cost=reported_cost,
                        error_reason=failure.error_reason if failure else None,
                        idempotency_key=failure.idempotency_key if failure else None,
                        error_code=failure.error_code if failure else None,
                    )

                with recorded_under(self._metering_client):
                    thread = run_async_in_thread(metering_call())
                logger.debug("Streaming metering thread started: %s", thread)

    claim_call_for_transport(OPENAI)
    if is_raw_response(stream):
        meter = StreamWrapper(())
        tap_sse_body(stream, meter._process_chunk, meter._finalize, drop_event=meter._injected_usage_chunk,
                     on_error=meter._record_failure)
        return stream
    return StreamWrapper(stream)


def _captured_request_prompts(kwargs):
    """The capture fields of a Responses request, or None with capture off.

    Only these bounded strings are kept for the record, never the request
    itself, which a waiting background job would otherwise hold in memory.
    It runs before the call is sent, so a request it cannot read is recorded
    without its prompt rather than failing the call.
    """
    if not is_capture_prompts_enabled():
        return None
    try:
        return extract_prompts_from_request(dict(kwargs))
    except Exception as e:
        logger.warning("Prompt capture could not read this Responses request (%s); recording it without its prompt",
                       type(e).__name__)
        return None


def _begin_responses_call(instance, kwargs) -> _MeteredCall:
    """Resolve the call's usage metadata and run the enforcement pre-check
    (which may raise ``BudgetExceededError``) before the provider is called."""
    api_metadata = _pop_usage_metadata(kwargs, instance)
    request_prompts = _captured_request_prompts(kwargs)
    if not api_metadata:
        api_metadata = _extract_langchain_usage_metadata()
    usage_metadata = merge_metadata(with_callback_metadata(OPENAI, api_metadata))
    request_time_dt = datetime.datetime.now(datetime.timezone.utc)
    client_instance = getattr(instance, '_client', None)
    provider_metadata = _chat_provider_metadata(instance)
    check_enforcement(usage_metadata, model=kwargs.get("model"),
                      provider=_cap_provider_label(client_instance, provider_metadata))
    return _MeteredCall(usage_metadata, request_time_dt, client_instance, None, provider_metadata=provider_metadata,
                        request_prompts=request_prompts)


_RUNNING_RESPONSE_STATUSES = frozenset({"queued", "in_progress"})
_FINISHED_RESPONSE_STATUSES = frozenset({"completed", "incomplete", "failed", "cancelled"})
MAX_PENDING_BACKGROUND_RESPONSES = 1024


class _PendingBackgroundResponse(NamedTuple):
    usage_metadata: Dict[str, Any]
    request_time_dt: datetime.datetime
    request_prompts: Optional[Dict[str, Any]]
    metering_client: Any


def _pending_background_response(usage_metadata, request_time_dt, request_prompts, metering_client):
    """The create-time context of a background response, its job fields resolved now.

    The job is metered later by whichever call sees it finish, so its job
    fields and Revenium key must come from create time, not the poller's.
    """
    usage_metadata = usage_metadata or {}
    resolved = {**usage_metadata, **extract_agentic_job_fields(usage_metadata)}
    return _PendingBackgroundResponse(resolved, request_time_dt, request_prompts, metering_client)


@contextlib.contextmanager
def _outside_the_pollers_job():
    """Hide the polling caller's agentic job while a background job is metered.

    The job's own fields were resolved at create time; the poller's job must
    not fill in the ones it lacked.
    """
    token = _agentic_job_context.set(None)
    try:
        yield
    finally:
        _agentic_job_context.reset(token)


class _PendingBackgroundResponses:
    """Create-time context of background responses this process has not yet seen finish, keyed by response id."""

    def __init__(self, capacity):
        self._capacity = capacity
        self._lock = threading.Lock()
        self._pending = OrderedDict()
        self._warned_overflow = False

    def add(self, response_id, pending):
        with self._lock:
            self._pending[response_id] = pending
            overflowed = len(self._pending) > self._capacity
            while len(self._pending) > self._capacity:
                self._pending.popitem(last=False)
            warn = overflowed and not self._warned_overflow
            self._warned_overflow = self._warned_overflow or overflowed
        if warn:
            logger.warning(
                "More than %d OpenAI background responses are waiting to be retrieved; the oldest "
                "will not be metered", self._capacity
            )

    def holds(self, response_id):
        with self._lock:
            return response_id in self._pending

    def take(self, response_id):
        with self._lock:
            return self._pending.pop(response_id, None)


_pending_background_responses = _PendingBackgroundResponses(MAX_PENDING_BACKGROUND_RESPONSES)


def _read_response(response):
    """The parsed ``Response``, or None when the caller still owns an unread body."""
    if isinstance(response, OpenAIBaseModel):
        return response
    if LegacyAPIResponse is not None and isinstance(response, LegacyAPIResponse):
        return response.parse()
    return None


def _meter_non_streamed_response(response, call: _MeteredCall):
    parsed = _read_response(response)
    if parsed is not None and getattr(parsed, "status", None) in _RUNNING_RESPONSE_STATUSES:
        _pending_background_responses.add(
            parsed.id,
            _pending_background_response(call.usage_metadata, call.request_time_dt, call.request_prompts,
                                         current_client()))
        return
    # The Revenium backend has no separate RESPONSES operation type yet.
    _meter_owned_response(response, OperationType.CHAT, call.request_time_dt, call.usage_metadata,
                          client_instance=call.client_instance,
                          time_to_first_token=_elapsed_ms_since(call.request_time_dt),
                          provider_metadata=call.provider_metadata, request_prompts=call.request_prompts)


def _meter_finished_background_response(response, response_id, instance):
    """Meter a pending background response the first time it is seen finished.

    An unread body is passed through with the job left pending, so the next
    read that can see the response meters it.
    """
    parsed = _read_response(response)
    if parsed is None or getattr(parsed, "status", None) not in _FINISHED_RESPONSE_STATUSES:
        return
    pending = _pending_background_responses.take(response_id)
    if pending is None or getattr(parsed, "usage", None) is None:
        return
    with _outside_the_pollers_job(), recorded_under(pending.metering_client):
        _meter_owned_response(parsed, OperationType.CHAT, pending.request_time_dt, pending.usage_metadata,
                              client_instance=getattr(instance, '_client', None),
                              time_to_first_token=_elapsed_ms_since(pending.request_time_dt),
                              request_prompts=pending.request_prompts)


def _response_id_argument(args, kwargs):
    return args[0] if args else kwargs.get("response_id")


def _resumed_background_stream(stream, response_id, instance, wrap_stream):
    return wrap_stream(stream, None, None, client_instance=getattr(instance, '_client', None),
                       background=True, resumed_id=response_id)


def responses_retrieve_wrapper(wrapped, instance, args, kwargs):
    response_id = _response_id_argument(args, kwargs)
    if not _pending_background_responses.holds(response_id):
        return wrapped(*args, **kwargs)
    response = wrapped(*args, **kwargs)
    if kwargs.get("stream"):
        return _resumed_background_stream(response, response_id, instance, handle_streaming_responses)
    _meter_finished_background_response(response, response_id, instance)
    return response


def async_responses_retrieve_wrapper(wrapped, instance, args, kwargs):
    response_id = _response_id_argument(args, kwargs)
    if not _pending_background_responses.holds(response_id):
        return wrapped(*args, **kwargs)

    async def _async_invoke():
        response = await wrapped(*args, **kwargs)
        if kwargs.get("stream"):
            return _resumed_background_stream(response, response_id, instance, _wrap_async_responses_stream)
        _meter_finished_background_response(response, response_id, instance)
        return response

    return _async_invoke()


def responses_cancel_wrapper(wrapped, instance, args, kwargs):
    response_id = _response_id_argument(args, kwargs)
    response = wrapped(*args, **kwargs)
    _meter_finished_background_response(response, response_id, instance)
    return response


def async_responses_cancel_wrapper(wrapped, instance, args, kwargs):
    response_id = _response_id_argument(args, kwargs)

    async def _async_invoke():
        response = await wrapped(*args, **kwargs)
        _meter_finished_background_response(response, response_id, instance)
        return response

    return _async_invoke()


def responses_create_wrapper(wrapped, instance, args, kwargs):
    logger.debug("OpenAI Responses API create wrapper called")

    if _parse_call_in_progress.get():
        return wrapped(*args, **kwargs)
    _tell_langchain_when_perplexity_serves(instance)
    if _left_unmetered():
        return _call_responses_unmetered(wrapped, instance, args, kwargs)

    call = _begin_responses_call(instance, kwargs)

    logger.debug(
        "Calling wrapped responses function with args: %s, kwargs: %s",
        sanitize_for_logging(args),
        sanitize_for_logging(kwargs),
    )
    metering_client = current_client()
    with _recording_failure(call, kwargs.get("model"), kwargs.get('stream', False), metering_client):
        response = wrapped(*args, **kwargs)

    if kwargs.get('stream', False):
        with recorded_under(metering_client):
            return handle_streaming_responses(response, call.request_time_dt, call.usage_metadata,
                                              client_instance=call.client_instance,
                                              requested_model=kwargs.get("model"),
                                              background=bool(kwargs.get('background')),
                                              provider_metadata=call.provider_metadata,
                                              request_prompts=call.request_prompts)
    _meter_non_streamed_response(response, call)
    return response


def async_responses_create_wrapper(wrapped, instance, args, kwargs):
    logger.debug("OpenAI async Responses API create wrapper called")

    if _parse_call_in_progress.get():
        return wrapped(*args, **kwargs)
    _tell_langchain_when_perplexity_serves(instance)
    if _left_unmetered():
        return _call_responses_unmetered(wrapped, instance, args, kwargs)

    call = _begin_responses_call(instance, kwargs)
    metering_client = current_client()

    async def _async_invoke():
        with _recording_failure(call, kwargs.get("model"), kwargs.get('stream', False), metering_client):
            response = await wrapped(*args, **kwargs)

        if kwargs.get('stream', False):
            with recorded_under(metering_client):
                return _wrap_async_responses_stream(response, call.request_time_dt, call.usage_metadata,
                                                    client_instance=call.client_instance,
                                                    requested_model=kwargs.get("model"),
                                                    background=bool(kwargs.get('background')),
                                                    provider_metadata=call.provider_metadata,
                                                    request_prompts=call.request_prompts)
        _meter_non_streamed_response(response, call)
        return response

    return _async_invoke()


async def _await_unmetered_parse(wrapped, args, kwargs):
    with _metered_by_parse():
        return await wrapped(*args, **kwargs)


def _never_defers(_instance):
    return False


def _parse_wrappers(begin_call, meter_response, cap_provider_label, defers=_never_defers):
    """The sync and async wraps for a structured-output ``parse()`` method.

    ``parse()`` posts its request itself rather than calling the wrapped
    ``create()``, and has no streamed form (structured streaming goes through
    ``stream()``), so each call is metered here from its parsed response,
    unless ``defers(instance)`` leaves it to another integration's wrap.
    """
    def parse_wrapper(wrapped, instance, args, kwargs):
        if _parse_call_in_progress.get() or defers(instance):
            return wrapped(*args, **kwargs)
        if _left_unmetered():
            _check_unmetered_client_call(instance, kwargs, cap_provider_label)
            with _metered_by_parse():
                return wrapped(*args, **kwargs)
        call = begin_call(instance, kwargs)
        with _metered_by_parse(), metering_failed_parse(call, meter_response):
            response = wrapped(*args, **kwargs)
        meter_response(response, call)
        return response

    def async_parse_wrapper(wrapped, instance, args, kwargs):
        if _parse_call_in_progress.get() or defers(instance):
            return wrapped(*args, **kwargs)
        if _left_unmetered():
            _check_unmetered_client_call(instance, kwargs, cap_provider_label)
            return _await_unmetered_parse(wrapped, args, kwargs)
        call = begin_call(instance, kwargs)

        async def _async_invoke():
            with _metered_by_parse(), metering_failed_parse(call, meter_response):
                response = await wrapped(*args, **kwargs)
            meter_response(response, call)
            return response

        return _async_invoke()

    return parse_wrapper, async_parse_wrapper


PERPLEXITY_CHAT_PARSE_PATCH = "perplexity:openai.resources.chat.completions.Completions.parse"


def _perplexity_meters_parse(instance):
    return _is_perplexity_bound(instance) and is_patched(PERPLEXITY_CHAT_PARSE_PATCH)


# A Perplexity-bound parse() is left to the Perplexity middleware when it is
# loaded, and metered here, labelled PERPLEXITY, otherwise. The predicate is
# looked up per call so tests can pin either configuration.
chat_parse_wrapper, async_chat_parse_wrapper = _parse_wrappers(
    _begin_chat_call, _meter_chat_completion, _chat_cap_provider_label,
    defers=lambda instance: _perplexity_meters_parse(instance),
)
responses_parse_wrapper, async_responses_parse_wrapper = _parse_wrappers(
    _begin_responses_call, _meter_non_streamed_response, _chat_cap_provider_label
)


_ZERO_RESPONSES_USAGE = SimpleNamespace(input_tokens=0, output_tokens=0, total_tokens=0)


class _ResponsesReply:
    """The reply text of a Responses event stream.

    The text of the final response wins over the deltas, so a stream resumed
    part way through still gives the whole reply, and never twice.
    """

    def __init__(self):
        self._deltas = ReplyCollector()
        self._final = ReplyCollector()

    def observe(self, event):
        delta = getattr(event, 'delta', None)
        if getattr(event, 'type', None) == 'response.output_text.delta' and isinstance(delta, str):
            self._deltas.add(delta)
        response = getattr(event, 'response', None)
        final = responses_output_text(response) if response is not None else ""
        if final:
            self._final = ReplyCollector()
            self._final.add(final)
            self._deltas = ReplyCollector()

    def text(self):
        return self._final.text() or self._deltas.text()


class _ResponsesStreamMeter:
    """Collects a Responses API event stream and meters it exactly once.

    Responses events carry no top-level id, model or usage: the lifecycle
    events nest the ``Response`` under ``event.response``. ``response.created``
    already has the id and model; the terminal ``response.completed``,
    ``response.incomplete`` or ``response.failed`` event adds the usage.
    """

    def __init__(self, request_time_dt, usage_metadata, client_instance, requested_model=None,
                 background=False, resumed_id=None, provider_metadata=None, request_prompts=None):
        self._request_time_dt = request_time_dt
        self._request_prompts = request_prompts
        self._reply = _ResponsesReply() if is_capture_prompts_enabled() else None
        self._usage_metadata = usage_metadata
        self._client_instance = client_instance
        self._provider_metadata = provider_metadata
        self._requested_model = requested_model
        self.failure = None
        self._outer = exception_in_flight()
        self._metering_client = current_client()
        self._call_context = CallContext()
        self._background = background
        self._resumed = resumed_id is not None
        self._registered = self._resumed
        self._finished_job = None
        self.response_id = resumed_id
        self.model = None
        self.usage = None
        self.status = None
        self.incomplete_details = None
        self._saw_event = False
        self._metered = False
        self._first_output_dt = None

    def observe(self, event):
        self._saw_event = True
        if self._first_output_dt is None and getattr(event, 'type', None) == 'response.output_text.delta':
            self._first_output_dt = datetime.datetime.now(datetime.timezone.utc)
        if self._reply is not None:
            self._reply.observe(event)
        response = getattr(event, 'response', None)
        if response is None:
            return
        self.response_id = getattr(response, 'id', None) or self.response_id
        self.model = getattr(response, 'model', None) or self.model
        self.usage = getattr(response, 'usage', None) or self.usage
        self.status = getattr(response, 'status', None) or self.status
        self.incomplete_details = getattr(response, 'incomplete_details', None) or self.incomplete_details
        if self._background:
            self._track_background_job()

    def _track_background_job(self):
        # Runs on the caller's thread as events arrive, so meter(), which can
        # run from a finalizer, never touches the pending registry.
        if self.response_id is None:
            return
        if not self._registered:
            self._registered = True
            _pending_background_responses.add(self.response_id, self._call_context.run(
                _pending_background_response, self._usage_metadata, self._request_time_dt, self._request_prompts,
                self._metering_client))
        if self._finished_job is None and self.status in _FINISHED_RESPONSE_STATUSES:
            self._finished_job = _pending_background_responses.take(self.response_id)

    def record_failure(self, error):
        self.failure = self._call_context.run(failure_of, error, self._outer)

    def meter(self):
        if self._metered or (not self._saw_event and self.failure is None):
            return
        self._metered = True

        if self._background:
            # A background job cut off mid-stream is metered later, by the
            # retrieve or resumed stream that sees it finish.
            if self._finished_job is None or self.usage is None:
                return
            self._request_time_dt = self._finished_job.request_time_dt
            self._usage_metadata = self._finished_job.usage_metadata
            self._metering_client = self._finished_job.metering_client
            self._request_prompts = self._finished_job.request_prompts

        if self.usage is None and self.failure is None:
            logger.warning("No usage data found in streaming Responses API response!")
        if self.response_id is None and self.failure is None:
            return

        self._call_context.run(self._send_under_its_own_job)

    def _send_under_its_own_job(self):
        with _outside_the_pollers_job() if self._background else contextlib.nullcontext():
            self._send()

    def _send(self):
        # An interrupted stream is metered with zero token counts rather than
        # dropped, as the chat stream wrappers do.
        with recorded_under(self._metering_client):
            create_metering_call(
                SimpleNamespace(
                    id=self.response_id or failed_transaction_id(),
                    model=self.model or (self.failure and self._requested_model) or "unknown",
                    usage=self.usage or _ZERO_RESPONSES_USAGE,
                    system_fingerprint=None,
                    status=self.status,
                    incomplete_details=self.incomplete_details,
                ),
                OperationType.CHAT,
                self._request_time_dt,
                self._usage_metadata,
                client_instance=self._client_instance,
                is_streamed=True,
                time_to_first_token=self._time_to_first_token(),
                provider_metadata=self._provider_metadata,
                failure=self.failure,
                request_prompts=self._request_prompts,
                accumulated_content=self._reply.text() if self._reply is not None else None,
            )
        self._reply = None

    def _time_to_first_token(self):
        # A resumed stream's request time is the original create's, swapped in by meter().
        if self._first_output_dt is None or self._resumed:
            return 0
        return int((self._first_output_dt - self._request_time_dt).total_seconds() * 1000)


class _MeteredResponsesStream:
    def __init__(self, stream, meter):
        self.stream = iter(stream)
        self.response = _http_response_of(stream)
        self._meter = meter
        self._closed = False

    def __iter__(self):
        return self

    def __next__(self):
        if self._closed:
            raise StopIteration("Stream has been closed")

        try:
            event = next(self.stream)
        except StopIteration:
            self._finalize()
            raise
        except Exception as e:
            self._meter.record_failure(e)
            self._finalize()
            logger.error(f"Error in streaming Responses API response: {e}")
            raise
        self._meter.observe(event)
        return event

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._finalize()

    def close(self):
        """Close the stream, metering whatever was received so far."""
        self._finalize()

    def __del__(self):
        # Last-resort finalize: a broken-out-of loop leaves the wrapper to
        # the GC with no StopIteration/__exit__ ever firing.
        try:
            self._finalize()
        except Exception:
            pass

    def _finalize(self):
        self._meter.meter()
        if self._closed:
            return
        self._closed = True
        try:
            if hasattr(self.stream, 'close'):
                self.stream.close()
        except Exception as e:
            logger.debug(f"Error closing stream: {e}")


class _AsyncMeteredResponsesStream:
    def __init__(self, stream, meter):
        self.stream = stream
        self.response = _http_response_of(stream)
        self._meter = meter
        self._closed = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._closed:
            raise StopAsyncIteration
        try:
            event = await self.stream.__anext__()
        except StopAsyncIteration:
            await self._afinalize()
            raise
        except Exception as e:
            self._meter.record_failure(e)
            await self._afinalize()
            logger.error(f"Error in async streaming Responses API response: {e}")
            raise
        self._meter.observe(event)
        return event

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self._afinalize()

    async def close(self):
        """Close the stream, metering whatever was received so far."""
        await self._afinalize()

    async def _afinalize(self):
        self._meter.meter()
        if self._closed:
            return
        self._closed = True
        closer = getattr(self.stream, 'aclose', None) or getattr(self.stream, 'close', None)
        if closer is None:
            return
        try:
            await closer()
        except Exception as e:
            logger.debug(f"Error closing async stream: {e}")

    def __del__(self):
        # Last-resort finalize: an abandoned async iterator (break + GC)
        # would otherwise never meter; meter() only schedules the
        # fire-and-forget metering thread, so it is safe here.
        try:
            self._meter.meter()
        except Exception:
            pass


def handle_streaming_responses(stream, request_time_dt, usage_metadata,
                               client_instance: Optional[Any] = None, requested_model: Optional[str] = None,
                               background=False, resumed_id=None, provider_metadata=None, request_prompts=None):
    """Wrap a sync Responses API event stream so it is metered once consumed or closed."""
    meter = _ResponsesStreamMeter(request_time_dt, usage_metadata, client_instance, requested_model,
                                  background, resumed_id, provider_metadata, request_prompts=request_prompts)
    claim_call_for_transport(OPENAI)
    if is_raw_response(stream):
        tap_sse_body(stream, meter.observe, meter.meter, on_error=meter.record_failure)
        return stream
    return _MeteredResponsesStream(stream, meter)


def _wrap_async_responses_stream(stream, request_time_dt, usage_metadata,
                                 client_instance: Optional[Any] = None, requested_model: Optional[str] = None,
                                 background=False, resumed_id=None, provider_metadata=None, request_prompts=None):
    """Wrap an async Responses API event stream so it is metered once consumed or closed."""
    meter = _ResponsesStreamMeter(request_time_dt, usage_metadata, client_instance, requested_model,
                                  background, resumed_id, provider_metadata, request_prompts=request_prompts)
    claim_call_for_transport(OPENAI)
    if is_raw_response(stream):
        tap_async_sse_body(stream, meter.observe, meter.meter, on_error=meter.record_failure)
        return stream
    return _AsyncMeteredResponsesStream(stream, meter)


def _wrap_async_stream(stream, request_time_dt, usage_metadata, client_instance=None, request_body=None,
                       caller_requested_usage=False):
    class AsyncStreamWrapper:
        def __init__(self, stream):
            self.stream = stream
            self.response = _response_that_meters_on_close(stream, self)
            self._metering_client = current_client()
            self._call_context = CallContext()
            self._saw_chunk = False
            self.response_id = None
            self.model = None
            self.finish_reason = None
            self.final_usage = None
            self.first_token_time = None
            self._reply = ReplyCollector() if is_capture_prompts_enabled() else None
            self.caller_requested_usage = caller_requested_usage
            self._finalized = False
            self.failure = None
            self._outer = exception_in_flight()

        def __aiter__(self):
            return self

        async def __anext__(self):
            while True:
                try:
                    chunk = await self.stream.__anext__()
                except StopAsyncIteration:
                    self._finalize()
                    raise
                except Exception as e:
                    self._record_failure(e)
                    self._finalize()
                    raise

                self._observe(chunk)
                if not self._injected_usage_chunk(chunk):
                    return chunk

        def _observe(self, chunk):
            _observing_chunk(self._process_chunk, chunk)
            self._saw_chunk = True

        def _process_chunk(self, chunk):
            # The first non-empty id and model: Azure opens a stream with a
            # content-filter chunk whose id and model are empty.
            self.response_id = self.response_id or getattr(chunk, 'id', None)
            self.model = self.model or getattr(chunk, 'model', None)
            choices = getattr(chunk, 'choices', None)
            if choices and getattr(choices[0], 'finish_reason', None):
                self.finish_reason = choices[0].finish_reason
            if hasattr(chunk, 'usage') and chunk.usage:
                self.final_usage = chunk.usage
            content = getattr(getattr(choices[0], 'delta', None), 'content', None) if choices else None
            if self.first_token_time is None and content:
                self.first_token_time = datetime.datetime.now(datetime.timezone.utc)
            if self._reply is not None and content:
                self._reply.add(_answer_text_of(content))
            self._saw_chunk = True

        def _injected_usage_chunk(self, chunk):
            return _is_injected_usage_chunk(chunk, self.caller_requested_usage)

        def _record_failure(self, error):
            self.failure = self._call_context.run(failure_of, error, self._outer)

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            await self._afinish()

        async def aclose(self):
            try:
                await self._afinish()
            finally:
                if hasattr(self.stream, 'aclose'):
                    await self.stream.aclose()

        def _finish(self):
            # A sync close of the async response cannot read on; meter what arrived.
            self._finalize()

        async def _afinish(self):
            try:
                await _aread_usage_after_finish(self, still_open=not self._finalized)
            finally:
                self._finalize()

        def __del__(self):
            # Last-resort finalize: an abandoned async iterator (break + GC)
            # would otherwise never meter. _finalize only schedules
            # the fire-and-forget metering thread, so it is safe here.
            try:
                self._finalize()
            except Exception:
                pass

        def _finalize(self):
            if self._finalized:
                return
            self._finalized = True
            self._call_context.run(self._meter)

        def _meter(self):
            if not self._saw_chunk and not self.final_usage and self.failure is None:
                return

            if self.final_usage is None:
                # Interrupted before the usage chunk arrived: meter the
                # transaction with zero token counts rather than dropping it.
                self.final_usage = type('Usage', (), {
                    'prompt_tokens': 0,
                    'completion_tokens': 0,
                    'total_tokens': 0,
                })()

            time_to_first_token = int((self.first_token_time - request_time_dt).total_seconds() * 1000) if self.first_token_time else 0

            try:
                with recorded_under(self._metering_client):
                    create_metering_call(
                        type('Response', (), {
                            'id': self.response_id or (failed_transaction_id() if self.failure else None),
                            'model': self.model or (self.failure and (request_body or {}).get('model')) or 'unknown',
                            'usage': self.final_usage,
                            'system_fingerprint': None,
                            'choices': [SimpleNamespace(finish_reason=self.finish_reason)],
                        })(),
                        OperationType.CHAT,
                        request_time_dt,
                        usage_metadata,
                        client_instance=client_instance,
                        is_streamed=True,
                        time_to_first_token=time_to_first_token,
                        request_body=request_body,
                        failure=self.failure,
                        accumulated_content=self._reply.text() if self._reply is not None else None,
                    )
                self._reply = None
            except Exception as e:
                logger.error("Async stream metering error: %s", e)

    claim_call_for_transport(OPENAI)
    if is_raw_response(stream):
        meter = AsyncStreamWrapper(None)
        tap_async_sse_body(stream, meter._process_chunk, meter._finalize, drop_event=meter._injected_usage_chunk,
                           on_error=meter._record_failure)
        return stream
    return AsyncStreamWrapper(stream)


def async_create_wrapper(wrapped, instance, args, kwargs):
    logger.debug("OpenAI/Azure OpenAI async chat.completions.create wrapper called")

    if _parse_call_in_progress.get():
        return wrapped(*args, **kwargs)
    if _is_perplexity_bound(instance):
        return _call_perplexity_bound(wrapped, instance, args, kwargs, _PERPLEXITY_ASYNC_CHAT_WRAP)
    if _left_unmetered():
        return _call_unmetered(wrapped, instance, args, kwargs)

    call = _begin_chat_call(instance, kwargs)

    stream = kwargs.get('stream', False)
    caller_requested_usage = False
    if stream:
        kwargs, caller_requested_usage = _with_forced_usage_reporting(kwargs)

    metering_client = current_client()

    async def _async_invoke():
        with _recording_failure(call, kwargs.get("model"), stream, metering_client):
            response = await wrapped(*args, **kwargs)

        if stream:
            with recorded_under(metering_client):
                return _wrap_async_stream(
                    response,
                    call.request_time_dt,
                    call.usage_metadata,
                    client_instance=call.client_instance,
                    request_body=call.request_body,
                    caller_requested_usage=caller_requested_usage
                )

        _meter_chat_completion(response, call)
        return response

    return _async_invoke()


def async_embeddings_create_wrapper(wrapped, instance, args, kwargs):
    logger.debug("OpenAI/Azure OpenAI async embeddings.create wrapper called")

    if _left_unmetered():
        return _call_unmetered(wrapped, instance, args, kwargs)

    request_body = kwargs.copy()

    api_metadata = _take_call_metadata(instance, kwargs) or {}

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

    # Enforcement pre-call check — may raise BudgetExceededError
    check_enforcement(usage_metadata, model=kwargs.get("model"), provider=get_provider_metadata(provider)["provider"])

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


def _openai_defines(module_path, attribute_path):
    try:
        target = importlib.import_module(module_path)
        for name in attribute_path.split('.'):
            target = getattr(target, name)
    except (ImportError, AttributeError):
        return False
    return True


wrap_once('openai.resources.chat.completions.Completions.create',
          'openai.resources.chat.completions', 'Completions.create', create_wrapper)

wrap_once('openai.resources.chat.completions.AsyncCompletions.create',
          'openai.resources.chat.completions', 'AsyncCompletions.create', async_create_wrapper)

wrap_once('openai.resources.embeddings.Embeddings.create',
          'openai.resources.embeddings', 'Embeddings.create', embeddings_create_wrapper)

wrap_once('openai.resources.embeddings.AsyncEmbeddings.create',
          'openai.resources.embeddings', 'AsyncEmbeddings.create', async_embeddings_create_wrapper)

wrap_once('openai.resources.responses.Responses.create',
          'openai.resources.responses', 'Responses.create', responses_create_wrapper)

wrap_once('openai.resources.responses.AsyncResponses.create',
          'openai.resources.responses', 'AsyncResponses.create', async_responses_create_wrapper)

# parse() reached chat completions and Responses in later openai releases than
# the declared ``openai>=1.0.0`` floor; wrapping a missing attribute would fail
# the whole middleware import.
if _openai_defines('openai.resources.chat.completions', 'Completions.parse'):
    wrap_once('openai.resources.chat.completions.Completions.parse',
              'openai.resources.chat.completions', 'Completions.parse', chat_parse_wrapper)

if _openai_defines('openai.resources.chat.completions', 'AsyncCompletions.parse'):
    wrap_once('openai.resources.chat.completions.AsyncCompletions.parse',
              'openai.resources.chat.completions', 'AsyncCompletions.parse', async_chat_parse_wrapper)

for _module_path, _class_name, _wrapper in (
    ('openai.resources.chat.completions', 'Completions', stream_helper_wrapper),
    ('openai.resources.chat.completions', 'AsyncCompletions', async_stream_helper_wrapper),
    ('openai.resources.responses', 'Responses', stream_helper_wrapper),
    ('openai.resources.responses', 'AsyncResponses', async_stream_helper_wrapper),
):
    if _openai_defines(_module_path, f'{_class_name}.stream'):
        wrap_once(f'{_module_path}.{_class_name}.stream', _module_path, f'{_class_name}.stream', _wrapper)

for _class_name, _method, _wrapper in (
    ('Responses', 'retrieve', responses_retrieve_wrapper),
    ('AsyncResponses', 'retrieve', async_responses_retrieve_wrapper),
    ('Responses', 'cancel', responses_cancel_wrapper),
    ('AsyncResponses', 'cancel', async_responses_cancel_wrapper),
):
    if _openai_defines('openai.resources.responses', f'{_class_name}.{_method}'):
        wrap_once(f'openai.resources.responses.{_class_name}.{_method}',
                  'openai.resources.responses', f'{_class_name}.{_method}', _wrapper)

if _openai_defines('openai.resources.responses', 'Responses.parse'):
    wrap_once('openai.resources.responses.Responses.parse',
              'openai.resources.responses', 'Responses.parse', responses_parse_wrapper)

if _openai_defines('openai.resources.responses', 'AsyncResponses.parse'):
    wrap_once('openai.resources.responses.AsyncResponses.parse',
              'openai.resources.responses', 'AsyncResponses.parse', async_responses_parse_wrapper)

from .audio import AUDIO_WRAPPERS  # noqa: E402  (audio.py builds on the helpers defined above)

for _module_path, _attribute_path, _audio_wrapper in AUDIO_WRAPPERS:
    if _openai_defines(_module_path, _attribute_path):
        wrap_once(f'{_module_path}.{_attribute_path}', _module_path, _attribute_path, _audio_wrapper)
