"""Feature-flagged metering at the botocore Bedrock Runtime boundary.

Raw ``boto3.client("bedrock-runtime")`` calls and the Converse API bypass the
SDK's Anthropic wrappers entirely. This module intercepts
``BaseClient._make_api_call`` and emits one completion per supported
invocation (InvokeModel, InvokeModelWithResponseStream, Converse,
ConverseStream) for the Anthropic Claude model family.

Activation is opt-in via ``REVENIUM_BEDROCK_TRANSPORT=1`` (canary). The same
variable is the runtime kill switch: it is re-read on every call, so setting
it to ``0`` stops emission immediately without unpatching. Calls made by
Revenium's own Bedrock adapter are marked with a context-local suppression
guard so the existing Anthropic-Bedrock wrapper remains the sole emitter for
that path.

The wrapper never alters caller-visible behaviour: request parameters,
return values, streaming semantics, exceptions, and botocore retries pass
through untouched; metering failures are logged and swallowed.
"""
from __future__ import annotations

import contextlib
import contextvars
import datetime
import json
import logging
import os
import uuid
from typing import Any, Dict, Iterator, Optional

from revenium_middleware._core.patch_registry import register_patch

logger = logging.getLogger("revenium_middleware")

_SUPPORTED_OPERATIONS = frozenset({
    "InvokeModel",
    "InvokeModelWithResponseStream",
    "Converse",
    "ConverseStream",
})

_STREAMING_OPERATIONS = frozenset({
    "InvokeModelWithResponseStream",
    "ConverseStream",
})

# Marks botocore calls issued by Revenium's own Bedrock adapter so the
# transport layer sees but does not meter them (the adapter's wrapper is the
# sole emitter for that path). Context-local, so it is safe across threads
# and asyncio tasks.
_internal_adapter_call = contextvars.ContextVar(
    "revenium_bedrock_internal_call", default=False
)

_STOP_REASON_MAP = {
    "end_turn": "END",
    "stop_sequence": "END_SEQUENCE",
    "tool_use": "END_SEQUENCE",
    "max_tokens": "TOKEN_LIMIT",
    "content_filtered": "ERROR",
    "guardrail_intervened": "ERROR",
}


def is_transport_enabled() -> bool:
    """Re-read the flag on every call: this is also the runtime kill switch."""
    return os.getenv("REVENIUM_BEDROCK_TRANSPORT", "").strip().lower() in ("1", "true")


@contextlib.contextmanager
def suppress_transport_metering():
    """Mark botocore calls in this context as internal adapter traffic."""
    token = _internal_adapter_call.set(True)
    try:
        yield
    finally:
        _internal_adapter_call.reset(token)


def _is_supported_model(model_id: str) -> bool:
    """Only the Anthropic Claude family is metered by this layer."""
    return "anthropic." in model_id


def _transaction_id(response: Any) -> str:
    """Prefer the AWS request ID so retries and native-log collection map to
    one record; fall back to a UUID when the metadata is absent."""
    try:
        request_id = response["ResponseMetadata"]["RequestId"]
        if request_id:
            return str(request_id)
    except (KeyError, TypeError):
        pass
    return str(uuid.uuid4())


def _ambient_usage_metadata() -> Dict[str, Any]:
    """Optional usage metadata from the anthropic middleware's context, when
    that module is importable (it requires the anthropic SDK; we do not)."""
    try:
        from revenium_middleware.anthropic import middleware as anthropic_middleware

        return dict(anthropic_middleware.usage_context.get({}) or {})
    except Exception:
        return {}


def _emit_completion(
    *,
    model_id: str,
    input_tokens: int,
    output_tokens: int,
    cache_read_tokens: int,
    cache_creation_tokens: int,
    stop_reason: Optional[str],
    is_streamed: bool,
    region: Optional[str],
    transaction_id: str,
    request_time_dt: datetime.datetime,
    completion_start_dt: Optional[datetime.datetime] = None,
) -> None:
    from revenium_middleware import run_async_in_thread, shutdown_event
    from revenium_middleware._core import submit_ai_event
    from revenium_middleware._core.fields import (
        extract_agentic_job_fields,
        extract_common_metadata,
        extract_org_and_product,
        merge_extra_body,
    )
    from revenium_middleware._core.subscriber import extract_subscriber_from_metadata

    if shutdown_event.is_set():
        logger.warning("Skipping Bedrock transport metering during shutdown")
        return

    from revenium_middleware._core import trace_fields as core_trace_fields

    usage_metadata = _ambient_usage_metadata()

    response_time_dt = datetime.datetime.now(datetime.timezone.utc)
    request_time = request_time_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    response_time = response_time_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    duration_ms = int((response_time_dt - request_time_dt).total_seconds() * 1000)

    # Streams report the first chunk's arrival; non-stream calls have no
    # earlier observable moment than the parsed response.
    first_token_dt = completion_start_dt or response_time_dt
    completion_start_time = first_token_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    time_to_first_token = int(
        (first_token_dt - request_time_dt).total_seconds() * 1000
    )

    subscriber = extract_subscriber_from_metadata(usage_metadata)
    organization_name, product_name = extract_org_and_product(usage_metadata)
    meta = extract_common_metadata(usage_metadata)
    extra_body = merge_extra_body({}, extract_agentic_job_fields(usage_metadata))

    completion_args = {
        "cache_creation_token_count": cache_creation_tokens,
        "cache_read_token_count": cache_read_tokens,
        "input_token_cost": None,
        "output_token_cost": None,
        "total_cost": None,
        "output_token_count": output_tokens,
        "cost_type": "AI",
        "model": model_id,
        "input_token_count": input_tokens,
        "provider": "AWS",
        "model_source": "ANTHROPIC",
        "reasoning_token_count": 0,
        "request_time": request_time,
        "response_time": response_time,
        "completion_start_time": completion_start_time,
        "request_duration": duration_ms,
        "time_to_first_token": time_to_first_token,
        "stop_reason": _STOP_REASON_MAP.get(stop_reason, "END"),
        "total_token_count": input_tokens + output_tokens,
        "transaction_id": transaction_id,
        "is_streamed": is_streamed,
        "operation_type": "CHAT",
        "middleware_source": "PYTHON",
        "region": (usage_metadata.get("region")
                   or core_trace_fields.get_region()
                   or region),
        "environment": (usage_metadata.get("environment")
                        or core_trace_fields.get_environment()),
        "credential_alias": (usage_metadata.get("credential_alias")
                             or usage_metadata.get("credentialAlias")
                             or core_trace_fields.get_credential_alias()),
        "trace_type": (usage_metadata.get("trace_type")
                       or usage_metadata.get("traceType")
                       or core_trace_fields.get_trace_type()),
        "trace_name": (usage_metadata.get("trace_name")
                       or usage_metadata.get("traceName")
                       or core_trace_fields.get_trace_name()),
        "parent_transaction_id": (usage_metadata.get("parent_transaction_id")
                                  or usage_metadata.get("parentTransactionId")
                                  or core_trace_fields.get_parent_transaction_id()),
        "transaction_name": core_trace_fields.get_transaction_name(usage_metadata),
        "retry_number": core_trace_fields.get_retry_number() or None,
        "operation_subtype": usage_metadata.get("operation_subtype"),
        "trace_id": meta["trace_id"],
        "task_type": meta["task_type"],
        "subscriber": subscriber if subscriber else None,
        "organization_name": organization_name,
        "subscription_id": meta["subscription_id"],
        "product_name": product_name,
        "agent": meta["agent"],
        "response_quality_score": meta["response_quality_score"],
        "extra_body": extra_body if extra_body else None,
    }

    async def metering_call():
        try:
            result = submit_ai_event("completion", completion_args)
            logger.debug("Bedrock transport metering result: %s", result)
        except Exception as exc:  # noqa: BLE001 - metering must never break callers
            if not shutdown_event.is_set():
                logger.warning("Bedrock transport metering failed: %s", exc)

    run_async_in_thread(metering_call())


def _rewrap_streaming_body(response: Dict[str, Any]) -> bytes:
    """Read the InvokeModel body and hand the caller an equivalent one."""
    import io

    from botocore.response import StreamingBody

    raw = response["body"].read()
    response["body"] = StreamingBody(io.BytesIO(raw), len(raw))
    return raw


class _MeteredEventStream:
    """Pass-through wrapper over a botocore EventStream that meters exactly
    once: on exhaustion, on a mid-stream provider error, on close(), or --
    as a last resort for abandoned streams -- on garbage collection."""

    def __init__(self, inner: Any, on_event, finalize) -> None:
        self._inner = inner
        self._on_event = on_event
        self._finalize = finalize
        self._done = False

    def __iter__(self) -> Iterator[Any]:
        # finally: a provider error mid-stream (throttle, connection reset)
        # still meters the partial usage observed so far -- AWS consumed
        # tokens even though the stream did not finish.
        try:
            for event in self._inner:
                try:
                    self._on_event(event)
                except Exception as exc:  # noqa: BLE001
                    logger.debug("Bedrock transport stream parse error: %s", exc)
                yield event
        finally:
            self._finish()

    def close(self) -> None:
        try:
            close = getattr(self._inner, "close", None)
            if close is not None:
                close()
        finally:
            self._finish()

    def __del__(self) -> None:
        # Abandoned without exhaustion or close(): meter what was observed.
        try:
            self._finish()
        except Exception:  # noqa: BLE001 - never raise from a finalizer
            pass

    def _finish(self) -> None:
        if self._done:
            return
        self._done = True
        try:
            self._finalize()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Bedrock transport stream metering failed: %s", exc)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


class _StreamUsage:
    __slots__ = ("input_tokens", "output_tokens", "cache_read", "cache_creation",
                 "stop_reason", "first_event_dt")

    def __init__(self) -> None:
        self.input_tokens = 0
        self.output_tokens = 0
        self.cache_read = 0
        self.cache_creation = 0
        self.stop_reason: Optional[str] = None
        self.first_event_dt: Optional[datetime.datetime] = None

    def mark_event(self) -> None:
        if self.first_event_dt is None:
            self.first_event_dt = datetime.datetime.now(datetime.timezone.utc)


def _parse_invoke_chunk(event: Dict[str, Any], usage: _StreamUsage) -> None:
    usage.mark_event()
    chunk = event.get("chunk")
    if not chunk:
        return
    payload = json.loads(chunk["bytes"].decode("utf-8"))
    kind = payload.get("type")
    if kind == "message_start":
        message_usage = payload.get("message", {}).get("usage", {})
        usage.input_tokens = message_usage.get("input_tokens", 0) or 0
        usage.cache_read = message_usage.get("cache_read_input_tokens", 0) or 0
        usage.cache_creation = message_usage.get("cache_creation_input_tokens", 0) or 0
    elif kind == "message_delta":
        delta_usage = payload.get("usage", {})
        if delta_usage.get("output_tokens") is not None:
            usage.output_tokens = delta_usage["output_tokens"]
        delta = payload.get("delta", {})
        if delta.get("stop_reason"):
            usage.stop_reason = delta["stop_reason"]
    elif kind == "message_stop":
        metrics = payload.get("amazon-bedrock-invocationMetrics", {})
        if metrics:
            usage.input_tokens = metrics.get("inputTokenCount", usage.input_tokens)
            usage.output_tokens = metrics.get("outputTokenCount", usage.output_tokens)


def _parse_converse_event(event: Dict[str, Any], usage: _StreamUsage) -> None:
    usage.mark_event()
    if "metadata" in event:
        converse_usage = event["metadata"].get("usage", {})
        usage.input_tokens = converse_usage.get("inputTokens", usage.input_tokens)
        usage.output_tokens = converse_usage.get("outputTokens", usage.output_tokens)
        usage.cache_read = converse_usage.get(
            "cacheReadInputTokens", usage.cache_read
        )
        usage.cache_creation = converse_usage.get(
            "cacheWriteInputTokens", usage.cache_creation
        )
    elif "messageStop" in event:
        usage.stop_reason = event["messageStop"].get("stopReason")


def _observe_response(
    operation_name: str,
    api_params: Dict[str, Any],
    response: Dict[str, Any],
    instance: Any,
    request_time_dt: datetime.datetime,
) -> Dict[str, Any]:
    model_id = api_params.get("modelId", "")
    region = getattr(getattr(instance, "meta", None), "region_name", None)
    transaction_id = _transaction_id(response)

    def emit(usage: _StreamUsage, is_streamed: bool) -> None:
        _emit_completion(
            model_id=model_id,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cache_read_tokens=usage.cache_read,
            cache_creation_tokens=usage.cache_creation,
            stop_reason=usage.stop_reason,
            is_streamed=is_streamed,
            region=region,
            transaction_id=transaction_id,
            request_time_dt=request_time_dt,
            completion_start_dt=usage.first_event_dt,
        )

    if operation_name == "InvokeModel":
        raw = _rewrap_streaming_body(response)
        body = json.loads(raw.decode("utf-8"))
        body_usage = body.get("usage", {})
        usage = _StreamUsage()
        usage.input_tokens = body_usage.get("input_tokens", 0) or 0
        usage.output_tokens = body_usage.get("output_tokens", 0) or 0
        usage.cache_read = body_usage.get("cache_read_input_tokens", 0) or 0
        usage.cache_creation = body_usage.get("cache_creation_input_tokens", 0) or 0
        usage.stop_reason = body.get("stop_reason")
        emit(usage, is_streamed=False)

    elif operation_name == "Converse":
        converse_usage = response.get("usage", {})
        usage = _StreamUsage()
        usage.input_tokens = converse_usage.get("inputTokens", 0) or 0
        usage.output_tokens = converse_usage.get("outputTokens", 0) or 0
        usage.cache_read = converse_usage.get("cacheReadInputTokens", 0) or 0
        usage.cache_creation = converse_usage.get("cacheWriteInputTokens", 0) or 0
        usage.stop_reason = response.get("stopReason")
        emit(usage, is_streamed=False)

    elif operation_name == "InvokeModelWithResponseStream":
        usage = _StreamUsage()
        response["body"] = _MeteredEventStream(
            response["body"],
            lambda event: _parse_invoke_chunk(event, usage),
            lambda: emit(usage, is_streamed=True),
        )

    elif operation_name == "ConverseStream":
        usage = _StreamUsage()
        response["stream"] = _MeteredEventStream(
            response["stream"],
            lambda event: _parse_converse_event(event, usage),
            lambda: emit(usage, is_streamed=True),
        )

    return response


def _should_intercept(instance, args) -> bool:
    """Gate evaluation only -- must never invoke the wrapped call, so a
    gate error can safely fall back to plain pass-through without risking
    a second invocation."""
    try:
        if not is_transport_enabled():
            return False
        operation_name = args[0]
        api_params = args[1] if len(args) > 1 else {}
        service_name = instance.meta.service_model.service_name
        return (
            service_name == "bedrock-runtime"
            and operation_name in _SUPPORTED_OPERATIONS
            and not _internal_adapter_call.get()
            and _is_supported_model(api_params.get("modelId", ""))
        )
    except Exception as exc:  # noqa: BLE001 - never break the caller's request
        logger.debug("Bedrock transport gate error: %s", exc)
        return False


def _make_api_call_wrapper(wrapped, instance, args, kwargs):
    if not _should_intercept(instance, args):
        return wrapped(*args, **kwargs)

    request_time_dt = datetime.datetime.now(datetime.timezone.utc)
    response = wrapped(*args, **kwargs)  # exceptions/retries pass through untouched

    try:
        return _observe_response(
            args[0], args[1] if len(args) > 1 else {}, response, instance,
            request_time_dt
        )
    except Exception as exc:  # noqa: BLE001 - metering must never break callers
        logger.warning("Bedrock transport observation failed: %s", exc)
        return response


def activate_bedrock_transport() -> bool:
    """Patch botocore's Bedrock Runtime boundary. Raises ImportError with an
    actionable message when botocore is unavailable."""
    try:
        import botocore.client  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "Bedrock transport metering requires botocore/boto3. "
            "Install with: pip install revenium-python-sdk[anthropic]"
        ) from exc

    import wrapt

    if register_patch("botocore.client.BaseClient._make_api_call"):
        wrapt.wrap_function_wrapper(
            "botocore.client", "BaseClient._make_api_call", _make_api_call_wrapper
        )
        logger.debug("Bedrock transport metering patch applied")
    return True


def activate_if_enabled() -> bool:
    """Import-time hook: activate only when the canary flag is set; never
    fail the import for non-Bedrock users."""
    if not is_transport_enabled():
        return False
    try:
        return activate_bedrock_transport()
    except ImportError as exc:
        logger.warning("%s", exc)
        return False
