import contextlib
import datetime
import inspect
import logging
import types
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Optional

import ollama

from revenium_middleware import get_client, run_async_in_thread, shutdown_event, merge_metadata
from revenium_middleware._core import submit_ai_event
from revenium_middleware._core.call_ownership import OLLAMA, claim_call_for_transport, with_callback_metadata
from revenium_middleware._core.subscriber import extract_subscriber_from_metadata
from revenium_middleware._core.fields import (
    extract_agentic_job_fields,
    extract_common_metadata,
    extract_effort_field,
    extract_org_and_product,
    extract_prompt_context_fields,
    extract_skill_fields,
    merge_extra_body,
)
from revenium_middleware._core.config import is_selective_metering_enabled
from revenium_middleware._core.context import is_inside_decorated_function
from revenium_middleware._core.selective_metering import reported_if_unmetered
from revenium_middleware._core.enforcement import check_enforcement, check_unmetered_call
from revenium_middleware._core.failed_calls import ERROR_STOP_REASON, exception_in_flight, failure_of
from revenium_middleware._core.log_sanitize import sanitize_for_logging
from revenium_middleware._core.call_context import CallContext
from revenium_middleware._core.metering import current_client, recorded_under
from revenium_middleware._core.patch_registry import wrap_once
from .trace_fields import (
    get_environment,
    get_region,
    get_credential_alias,
    get_trace_type,
    get_trace_name,
    get_parent_transaction_id,
    get_transaction_name,
    get_retry_number,
    get_ticket_id,
    get_agent_version,
    detect_operation_type
)

logger = logging.getLogger("revenium_middleware.extension")

EMBEDDING_ENDPOINTS = ("embed", "embeddings")
METERED_ENDPOINTS = ("chat", "generate") + EMBEDDING_ENDPOINTS
DEFAULT_MODEL = "ollama-model"
OLLAMA_PROVIDER = "OLLAMA"
_REFUSED_RESPONSE = types.SimpleNamespace()

FINISH_REASONS = {
    "stop": "END",
    "length": "TOKEN_LIMIT",
    "error": "ERROR",
    "cancelled": "CANCELLED",  # British spelling
    "canceled": "CANCELLED",   # American spelling (Go standard library uses this)
    "tool_calls": "END_SEQUENCE"
}


@dataclass
class OllamaCall:
    """What one intercepted Ollama call needs to be metered once it returns."""
    endpoint: str
    usage_metadata: Dict[str, Any]
    request_kwargs: Dict[str, Any]
    request_time_dt: datetime.datetime
    transaction_id: str
    request_model: Optional[str]

    @property
    def streamed(self) -> bool:
        return bool(self.request_kwargs.get("stream", False))


def _metering_skipped() -> bool:
    return reported_if_unmetered(is_selective_metering_enabled() and not is_inside_decorated_function())


def _requested_model(args, kwargs) -> Optional[str]:
    model = kwargs.get("model", args[0] if args else None)
    return model if isinstance(model, str) and model else None


def start_call(endpoint, args, kwargs, api_metadata) -> OllamaCall:
    request_time_dt = datetime.datetime.now(datetime.timezone.utc)
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "Calling Ollama %s with args: %s, kwargs: %s",
            endpoint,
            sanitize_for_logging(args),
            sanitize_for_logging(kwargs),
        )
    return OllamaCall(
        endpoint=endpoint,
        usage_metadata=merge_metadata(with_callback_metadata(OLLAMA, api_metadata)),
        request_kwargs=kwargs,
        request_time_dt=request_time_dt,
        transaction_id=f"ollama-{uuid.uuid4().hex}",
        request_model=_requested_model(args, kwargs),
    )


def meter_response(call: OllamaCall, response) -> None:
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug("Ollama %s response: %s", call.endpoint, sanitize_for_logging(response))
    if call.endpoint in EMBEDDING_ENDPOINTS:
        handle_embeddings_response(
            response, call.request_time_dt, call.usage_metadata,
            call.transaction_id, call.request_kwargs,
            endpoint=call.endpoint, request_model=call.request_model
        )
    else:
        handle_response(
            response, call.request_time_dt, call.usage_metadata,
            False, call.transaction_id, call.endpoint, call.request_kwargs,
            request_model=call.request_model
        )


def _failure_of(error, outer):
    """A ResponseError raised mid-stream has status_code -1, which names no HTTP status."""
    return failure_of(error, outer, status=getattr(error, "status_code", None))


@contextlib.contextmanager
def _recording_failure(call: OllamaCall, metering_client):
    """Send one ERROR record for a chat or generate call the server refused,
    then let the caller's exception through unchanged. Embeddings send none."""
    outer = exception_in_flight()
    try:
        yield
    except Exception as error:
        failure = None if call.endpoint in EMBEDDING_ENDPOINTS else _failure_of(error, outer)
        if failure is not None:
            try:
                with recorded_under(metering_client):
                    handle_response(
                        _REFUSED_RESPONSE, call.request_time_dt, call.usage_metadata,
                        call.streamed, call.transaction_id, call.endpoint, call.request_kwargs,
                        request_model=call.request_model, failure=failure
                    )
            except Exception as metering_error:
                logger.warning("Could not record a failed Ollama call: %s", metering_error)
        raise


def _check_caps(call: OllamaCall) -> None:
    check_enforcement(call.usage_metadata, model=call.request_model, provider=OLLAMA_PROVIDER)


def _check_unmetered_caps(args, kwargs, api_metadata) -> None:
    check_unmetered_call(api_metadata, model=_requested_model(args, kwargs), provider=OLLAMA_PROVIDER,
                         callback_transport=OLLAMA)


def sync_client_wrapper(endpoint):
    def wrapper(wrapped, _instance, args, kwargs):
        api_metadata = kwargs.pop("usage_metadata", None) or {}
        if _metering_skipped():
            _check_unmetered_caps(args, kwargs, api_metadata)
            return wrapped(*args, **kwargs)

        call = start_call(endpoint, args, kwargs, api_metadata)
        _check_caps(call)
        metering_client = current_client()
        with _recording_failure(call, metering_client):
            response = wrapped(*args, **kwargs)

        if call.streamed and isinstance(response, types.GeneratorType):
            claim_call_for_transport(OLLAMA)
            with recorded_under(metering_client):
                return handle_streaming_response(
                    response, call.request_time_dt, call.usage_metadata,
                    call.transaction_id, endpoint, call.request_kwargs,
                    request_model=call.request_model
                )
        meter_response(call, response)
        return response

    wrapper.__name__ = wrapper.__qualname__ = f"{endpoint}_wrapper"
    return wrapper


def async_client_wrapper(endpoint):
    def wrapper(wrapped, _instance, args, kwargs):
        api_metadata = kwargs.pop("usage_metadata", None) or {}
        if _metering_skipped():
            async def unmetered_call():
                _check_unmetered_caps(args, kwargs, api_metadata)
                return await wrapped(*args, **kwargs)

            return unmetered_call()

        call = start_call(endpoint, args, kwargs, api_metadata)

        async def metered_call():
            _check_caps(call)
            metering_client = current_client()
            with _recording_failure(call, metering_client):
                response = await wrapped(*args, **kwargs)

            if call.streamed and isinstance(response, types.AsyncGeneratorType):
                claim_call_for_transport(OLLAMA)
                with recorded_under(metering_client):
                    return handle_async_streaming_response(
                        response, call.request_time_dt, call.usage_metadata,
                        call.transaction_id, endpoint, call.request_kwargs,
                        request_model=call.request_model
                    )
            meter_response(call, response)
            return response

        return metered_call()

    wrapper.__name__ = wrapper.__qualname__ = f"async_{endpoint}_wrapper"
    return wrapper


chat_wrapper = sync_client_wrapper("chat")
generate_wrapper = sync_client_wrapper("generate")
embed_wrapper = sync_client_wrapper("embed")
embeddings_wrapper = sync_client_wrapper("embeddings")

CLIENT_WRAPPERS = {
    "Client": {
        "chat": chat_wrapper,
        "generate": generate_wrapper,
        "embed": embed_wrapper,
        "embeddings": embeddings_wrapper,
    },
    "AsyncClient": {endpoint: async_client_wrapper(endpoint) for endpoint in METERED_ENDPOINTS},
}


def rebind_default_client_functions():
    """Point ``ollama.chat`` and friends at the wrapped class methods.

    ollama binds its module-level functions to a default ``Client`` when it is
    imported, which is before any wrap here exists, so those bound methods keep
    calling the unwrapped functions until they are bound again.
    """
    for endpoint in METERED_ENDPOINTS:
        bound = getattr(ollama, endpoint, None)
        if inspect.ismethod(bound) and isinstance(bound.__self__, ollama.Client):
            setattr(ollama, endpoint, getattr(bound.__self__, endpoint))


def _carries_counts(chunk) -> bool:
    # Ollama only populates prompt_eval_count/eval_count on the final
    # done=True chunk.
    return bool(getattr(chunk, 'done', False)) or getattr(chunk, 'eval_count', None) is not None


class StreamTally:
    """The one chunk of a stream that its metering record is built from.

    Keeps the latest chunk that carries token counts, or the latest chunk
    while none has. Accepted tradeoff: a stream interrupted before the final
    chunk meters with zero counts to keep the transaction visible.
    """

    def __init__(self):
        self.metered_chunk = None
        self.metering_client = current_client()
        self.call_context = CallContext()
        self.failure = None
        self._outer = exception_in_flight()

    def observe(self, chunk) -> None:
        if _carries_counts(chunk) or not _carries_counts(self.metered_chunk):
            self.metered_chunk = chunk

    def fail(self, error) -> None:
        self.failure = _failure_of(error, self._outer)


def _meter_stream_end(tally, request_time_dt, usage_metadata, transaction_id, endpoint, request_kwargs,
                      request_model):
    if tally.metered_chunk is None and tally.failure is None:
        return
    try:
        tally.call_context.run_recorded_under(
            tally.metering_client, handle_response,
            tally.metered_chunk or _REFUSED_RESPONSE,
            request_time_dt,
            usage_metadata,
            True,
            transaction_id,
            endpoint,
            request_kwargs,
            request_model=request_model,
            failure=tally.failure
        )
    except Exception as e:
        logger.warning("Error metering interrupted/completed stream: %s", e)


def _close_quietly(generator) -> None:
    try:
        generator.close()
    except Exception as e:
        logger.debug("Error closing the Ollama stream: %s", e)


async def _aclose_quietly(generator) -> None:
    try:
        await generator.aclose()
    except Exception as e:
        logger.debug("Error closing the Ollama stream: %s", e)


def handle_streaming_response(
    generator,
    request_time_dt,
    usage_metadata,
    transaction_id,
    endpoint,
    request_kwargs,
    request_model=None,
    tally=None
):
    """
    Relay a sync stream unchanged and meter it once when it ends.

    Args:
        generator: The original response generator
        request_time_dt: The request timestamp
        usage_metadata: Metadata for metering
        transaction_id: The transaction ID to add to responses
        endpoint: The endpoint being called ('chat', 'generate', etc.)
        request_kwargs: The request kwargs for operation type detection
        request_model: The model named in the request, used when the response omits it
        tally: Where the metered chunk is kept (a new StreamTally when omitted)
    """
    tally = tally or StreamTally()

    def wrapped_generator():
        # The finally block also runs on GeneratorExit when the caller breaks
        # out of the loop or abandons the generator, so partial usage from an
        # interrupted stream is still metered and Ollama's HTTP stream is
        # released instead of staying open until the inner generator is
        # collected.
        try:
            for chunk in generator:
                tally.observe(chunk)
                yield chunk
        except Exception as error:
            tally.fail(error)
            raise
        finally:
            _meter_stream_end(tally, request_time_dt, usage_metadata, transaction_id, endpoint,
                              request_kwargs, request_model)
            _close_quietly(generator)

    return wrapped_generator()


def handle_async_streaming_response(
    generator,
    request_time_dt,
    usage_metadata,
    transaction_id,
    endpoint,
    request_kwargs,
    request_model=None,
    tally=None
):
    """Relay an async stream unchanged and meter it once when it ends; see handle_streaming_response."""
    tally = tally or StreamTally()

    async def wrapped_generator():
        try:
            async for chunk in generator:
                tally.observe(chunk)
                yield chunk
        except Exception as error:
            tally.fail(error)
            raise
        finally:
            _meter_stream_end(tally, request_time_dt, usage_metadata, transaction_id, endpoint,
                              request_kwargs, request_model)
            await _aclose_quietly(generator)

    return wrapped_generator()


def handle_response(
    response,
    request_time_dt,
    usage_metadata,
    is_streaming,
    transaction_id,
    endpoint,
    request_kwargs,
    request_model=None,
    failure=None
):
    """
    Process a complete chat or generate response (streamed or not) and send
    metering data.

    Args:
        response: The Ollama response object
        request_time_dt: The request timestamp
        usage_metadata: Metadata for metering
        is_streaming: Whether this is a streaming response
        transaction_id: The transaction ID for this request
        endpoint: The endpoint being called ('chat', 'generate', etc.)
        request_kwargs: The request kwargs for operation type detection
        request_model: The model named in the request, used when the response omits it
        failure: The failure of a call the server refused or broke off
    """
    prompt_tokens = getattr(response, 'prompt_eval_count', None) or 0
    completion_tokens = getattr(response, 'eval_count', None) or 0
    logger.debug(
        "Ollama %s token usage - prompt: %d, completion: %d",
        endpoint, prompt_tokens, completion_tokens
    )
    _dispatch_metering(
        response, request_time_dt, usage_metadata, transaction_id, endpoint, request_kwargs, request_model,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        stop_reason=ERROR_STOP_REASON if failure else FINISH_REASONS.get(getattr(response, 'done_reason', None), "END"),
        is_streaming=is_streaming,
        failure=failure,
    )


def handle_embeddings_response(
    response,
    request_time_dt,
    usage_metadata,
    transaction_id,
    request_kwargs,
    endpoint='embed',
    request_model=None
):
    """
    Process an embeddings response and send metering data.

    Embeddings only have input tokens. The legacy ``embeddings`` endpoint
    reports none, so its record carries zero input tokens.

    Args:
        response: The Ollama embeddings response object
        request_time_dt: The request timestamp
        usage_metadata: Metadata for metering
        transaction_id: The transaction ID for this request
        request_kwargs: The request kwargs for operation type detection
        endpoint: 'embed' or the legacy 'embeddings'
        request_model: The model named in the request, used when the response omits it
    """
    prompt_tokens = getattr(response, 'prompt_eval_count', None) or 0
    logger.debug("Ollama %s token usage - prompt: %d", endpoint, prompt_tokens)
    _dispatch_metering(
        response, request_time_dt, usage_metadata, transaction_id, endpoint, request_kwargs, request_model,
        prompt_tokens=prompt_tokens,
        completion_tokens=0,
        stop_reason="END",
        is_streaming=False,
    )


def _dispatch_metering(response, request_time_dt, usage_metadata, transaction_id, endpoint, request_kwargs,
                       request_model, *, prompt_tokens, completion_tokens, stop_reason, is_streaming, failure=None):
    if current_client() is None:
        return  # metering disabled (no API key configured)

    model = getattr(response, 'model', None) or request_model or DEFAULT_MODEL
    response_time_dt = datetime.datetime.now(datetime.timezone.utc)
    response_time = response_time_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    request_duration = (response_time_dt - request_time_dt).total_seconds() * 1000
    request_time = request_time_dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    async def metering_call():
        try:
            if shutdown_event.is_set():
                logger.warning("Skipping metering call during shutdown")
                return
            logger.debug("Metering call to Revenium for Ollama %s %s", endpoint, transaction_id)

            subscriber = extract_subscriber_from_metadata(usage_metadata)
            organization_name, product_name = extract_org_and_product(usage_metadata)
            meta = extract_common_metadata(usage_metadata)
            agentic_fields = extract_agentic_job_fields(usage_metadata)
            extra_body = merge_extra_body(None, agentic_fields)

            completion_args = {
                "cache_creation_token_count": 0,  # Ollama doesn't provide cached tokens info
                "cache_read_token_count": 0,
                "input_token_cost": None,
                "output_token_cost": None,
                "total_cost": None,
                "output_token_count": completion_tokens,
                "cost_type": "AI",
                "model": model,
                "input_token_count": prompt_tokens,
                "provider": OLLAMA_PROVIDER,
                "model_source": "OLLAMA",
                "reasoning_token_count": 0,
                "request_time": request_time,
                "response_time": response_time,
                "completion_start_time": response_time,
                "request_duration": int(request_duration),
                "stop_reason": stop_reason,
                "total_token_count": prompt_tokens + completion_tokens,
                "transaction_id": transaction_id,
                "trace_id": meta["trace_id"],
                "task_type": meta["task_type"],
                "subscriber": subscriber if subscriber else None,
                "organization_name": organization_name,
                "subscription_id": meta["subscription_id"],
                "product_name": product_name,
                "agent": meta["agent"],
                "response_quality_score": meta["response_quality_score"],
                "is_streamed": is_streaming,
                "middleware_source": "PYTHON",
                "operation_type": detect_operation_type(endpoint, request_kwargs),
                "environment": get_environment(usage_metadata),
                "region": get_region(usage_metadata),
                "credential_alias": get_credential_alias(usage_metadata),
                "trace_type": get_trace_type(usage_metadata),
                "trace_name": get_trace_name(usage_metadata),
                "ticket_id": get_ticket_id(usage_metadata),
                "agent_version": get_agent_version(usage_metadata),
                "parent_transaction_id": get_parent_transaction_id(usage_metadata),
                "transaction_name": get_transaction_name(usage_metadata),
                "retry_number": get_retry_number(usage_metadata),
                "extra_body": extra_body
            }

            # Reasoning effort level (caller-supplied, forwarded verbatim)
            completion_args.update(extract_effort_field(usage_metadata))
            completion_args.update(extract_prompt_context_fields(usage_metadata))
            completion_args.update(extract_skill_fields(usage_metadata))
            if failure is not None:
                completion_args.update(failure.record_fields())

            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("Arguments for create_completion: %s", sanitize_for_logging(completion_args))

            result = submit_ai_event("completion", completion_args,
                                     idempotency_key=failure.idempotency_key if failure else None)
            logger.debug("Metering call result: %s", result)
        except Exception as e:
            if not shutdown_event.is_set():
                logger.warning(f"Error in metering call: {str(e)}")
                # Log the full traceback for better debugging
                import traceback
                logger.warning(f"Traceback: {traceback.format_exc()}")

    thread = run_async_in_thread(metering_call())
    logger.debug("Metering thread started: %s", thread)
    if thread is not None and not is_streaming and failure is None:
        claim_call_for_transport(OLLAMA)


for _class_name, _wrappers in CLIENT_WRAPPERS.items():
    for _endpoint, _wrapper in _wrappers.items():
        wrap_once(f"ollama.{_class_name}.{_endpoint}", ollama, f"{_class_name}.{_endpoint}", _wrapper)
rebind_default_client_functions()
