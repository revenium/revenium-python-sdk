"""Feature-flagged metering at the botocore Bedrock Runtime boundary.

Raw ``boto3.client("bedrock-runtime")`` calls and the Converse API bypass the
SDK's Anthropic wrappers entirely. This module intercepts
``BaseClient._make_api_call`` and emits one completion per supported
invocation (InvokeModel, InvokeModelWithResponseStream, Converse,
ConverseStream) for the Anthropic Claude model family, including Claude
behind an application inference profile or a Provisioned Throughput, resolved
once per ARN with GetInferenceProfile or GetProvisionedModelThroughput in the
background, never on the caller's call.

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

from revenium_middleware._core.cache_tokens import extract_cache_creation_ttl_counts
from revenium_middleware._core.call_ownership import BEDROCK, claim_call_for_transport, with_callback_metadata
from revenium_middleware._core.context import merge_metadata
from revenium_middleware._core.enforcement import check_enforcement, check_unmetered_call
from revenium_middleware._core.call_context import CallContext
from revenium_middleware._core.metering import current_client, recorded_under
from revenium_middleware._core.patch_registry import wrap_once
from revenium_middleware._core.selective_metering import selectively_unmetered
from revenium_middleware.anthropic.bedrock_model_names import cap_model_names, recorded_bedrock_model_id
from revenium_middleware.anthropic.inference_profiles import UNMETERED, InferenceProfileResolver, is_routed_arn

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


def intercepts_model(model_id: str) -> bool:
    """Only the Anthropic Claude family is metered by this layer."""
    return "anthropic." in model_id


_profile_resolver = InferenceProfileResolver(is_claude=intercepts_model)


def _forget_profile_lookups_in_flight() -> None:
    _profile_resolver.forget_in_flight()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_forget_profile_lookups_in_flight)


def resolve_application_profile(runtime_client, profile_arn, *, unresolved=UNMETERED) -> Optional[str]:
    """The Claude model id ``profile_arn`` routes to, from the lookup the transport shares."""
    return _profile_resolver.claude_model_id(runtime_client, profile_arn, unresolved=unresolved)


def cached_application_profile(profile_arn) -> Optional[str]:
    """The Claude model id ``profile_arn`` routes to when it was already looked up, without a lookup."""
    return _profile_resolver.cached_claude_model_id(profile_arn)


def _request_id(response: Any) -> Optional[str]:
    try:
        request_id = response["ResponseMetadata"]["RequestId"]
    except (KeyError, TypeError):
        return None
    return str(request_id) if request_id else None


def _transaction_id(response: Any) -> str:
    """Prefer the AWS request ID so retries and native-log collection map to
    one record; fall back to a UUID when the metadata is absent."""
    return _request_id(response) or str(uuid.uuid4())


def _ambient_usage_metadata() -> Dict[str, Any]:
    """Optional usage metadata from the anthropic middleware's context, when
    that module is importable (it requires the anthropic SDK; we do not)."""
    try:
        from revenium_middleware.anthropic import middleware as anthropic_middleware

        return dict(anthropic_middleware.usage_context.get({}) or {})
    except Exception:
        return {}


def _call_usage_metadata() -> Optional[Dict[str, Any]]:
    """The caller's usage metadata laid over an attached LangChain
    callback's, read when the call starts: a stream may finish in another
    context. None when it cannot be read, so the record falls back to the
    ambient metadata."""
    try:
        return merge_metadata(with_callback_metadata(BEDROCK, _ambient_usage_metadata()))
    except Exception as exc:  # noqa: BLE001 - metering must never break callers
        logger.debug("Bedrock transport could not read the call's usage metadata: %s", exc)
        return None


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
    usage_metadata: Optional[Dict[str, Any]] = None,
    cache_creation_ttl_counts: Optional[Dict[str, int]] = None,
    prompt_fields: Optional[Dict[str, Any]] = None,
) -> None:
    from revenium_middleware import run_async_in_thread, shutdown_event
    from revenium_middleware._core import submit_ai_event
    from revenium_middleware._core.fields import (
        extract_agentic_job_fields,
        extract_common_metadata,
        extract_effort_field,
        extract_org_and_product,
        extract_prompt_context_fields,
        extract_skill_fields,
        merge_extra_body,
    )
    from revenium_middleware._core.subscriber import extract_subscriber_from_metadata

    if shutdown_event.is_set():
        logger.warning("Skipping Bedrock transport metering during shutdown")
        return

    from revenium_middleware._core import trace_fields as core_trace_fields

    if usage_metadata is None:
        usage_metadata = merge_metadata(_ambient_usage_metadata())

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
        **(cache_creation_ttl_counts or {}),
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
        "trace_type": core_trace_fields.get_trace_type(usage_metadata),
        "trace_name": core_trace_fields.get_trace_name(usage_metadata),
        "ticket_id": core_trace_fields.get_ticket_id(usage_metadata),
        "agent_version": core_trace_fields.get_agent_version(usage_metadata),
        # Reasoning effort is caller-supplied and forwarded verbatim. Spread
        # the sparse resolver result rather than reading a key off it:
        # create_completion only drops NotGiven during serialization, so an
        # explicit None would go on the wire as "effort": null instead of
        # being omitted.
        **extract_effort_field(usage_metadata),
        **extract_prompt_context_fields(usage_metadata),
        **extract_skill_fields(usage_metadata),
        "parent_transaction_id": (usage_metadata.get("parent_transaction_id")
                                  or usage_metadata.get("parentTransactionId")
                                  or core_trace_fields.get_parent_transaction_id()),
        "transaction_name": core_trace_fields.get_transaction_name(usage_metadata),
        "retry_number": core_trace_fields.get_retry_number(usage_metadata),
        "operation_subtype": usage_metadata.get("operation_subtype"),
        "trace_id": meta["trace_id"],
        "task_type": meta["task_type"],
        "subscriber": subscriber if subscriber else None,
        "organization_name": organization_name,
        "subscription_id": meta["subscription_id"],
        "product_name": product_name,
        "agent": meta["agent"],
        "response_quality_score": meta["response_quality_score"],
        **(prompt_fields or {}),
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
        self._metering_client = current_client()
        self._call_context = CallContext()
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
            self._call_context.run_recorded_under(self._metering_client, self._finalize)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Bedrock transport stream metering failed: %s", exc)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


class _StreamUsage:
    __slots__ = ("input_tokens", "output_tokens", "cache_read", "cache_creation",
                 "cache_creation_ttl_counts", "stop_reason", "first_event_dt",
                 "keep_text", "reply_text", "reply_length")

    def __init__(self, keep_text: bool = False) -> None:
        self.keep_text = keep_text
        self.reply_text = []
        self.reply_length = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.cache_read = 0
        self.cache_creation = 0
        self.cache_creation_ttl_counts: Dict[str, int] = {}
        self.stop_reason: Optional[str] = None
        self.first_event_dt: Optional[datetime.datetime] = None

    def mark_event(self) -> None:
        if self.first_event_dt is None:
            self.first_event_dt = datetime.datetime.now(datetime.timezone.utc)

    def add_reply_text(self, text: Any) -> None:
        """Keep streamed reply text for prompt capture, only while capture is on
        and only up to one character past the capture cap, which is enough for
        the cap to cut it and flag it."""
        from revenium_middleware._core.config import Config

        if not self.keep_text or not isinstance(text, str) or self.reply_length > Config.MAX_PROMPT_LENGTH:
            return
        kept = text[:Config.MAX_PROMPT_LENGTH + 1 - self.reply_length]
        self.reply_text.append(kept)
        self.reply_length += len(kept)


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
        usage.cache_creation_ttl_counts = extract_cache_creation_ttl_counts(message_usage)
    elif kind == "content_block_delta":
        usage.add_reply_text(payload.get("delta", {}).get("text"))
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
        usage.cache_creation_ttl_counts = extract_cache_creation_ttl_counts(converse_usage)
    elif "contentBlockDelta" in event:
        usage.add_reply_text(event["contentBlockDelta"].get("delta", {}).get("text"))
    elif "messageStop" in event:
        usage.stop_reason = event["messageStop"].get("stopReason")


_BINARY_PLACEHOLDER = "[binary omitted]"


def _without_binary(value: Any) -> Any:
    if isinstance(value, (bytes, bytearray)):
        return _BINARY_PLACEHOLDER
    if isinstance(value, dict):
        return {key: _without_binary(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_without_binary(item) for item in value]
    return value


def _anthropic_block(block: Dict[str, Any]) -> Dict[str, Any]:
    """One Converse content block in the Anthropic shape the capture cap reads."""
    if isinstance(block.get("text"), str):
        return {"type": "text", "text": block["text"]}
    kind = next(iter(block))
    return {"type": kind, kind: _without_binary(block[kind])}


def _converse_request(api_params: Dict[str, Any]) -> Dict[str, Any]:
    system_texts = [block["text"] for block in api_params.get("system") or ()
                    if isinstance(block, dict) and isinstance(block.get("text"), str)]
    messages = [
        {"role": message.get("role"),
         "content": [_anthropic_block(block) for block in message.get("content") or () if isinstance(block, dict) and block]}
        for message in api_params.get("messages") or () if isinstance(message, dict)
    ]
    request: Dict[str, Any] = {"messages": messages}
    if system_texts:
        request["system"] = "\n".join(system_texts)
    return request


def _invoke_request(api_params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """The caller's InvokeModel body when it is a Messages API request. A
    file-like body is never read: the call has already consumed it."""
    body = api_params.get("body")
    if not isinstance(body, (str, bytes, bytearray)):
        return None
    request = json.loads(body)
    if isinstance(request, dict) and isinstance(request.get("messages"), list):
        return request
    return None


def _anthropic_request(operation_name: str, api_params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if operation_name in ("Converse", "ConverseStream"):
        return _converse_request(api_params)
    return _invoke_request(api_params)


def _block_text(block: Any) -> str:
    if isinstance(block, dict) and isinstance(block.get("text"), str):
        return block["text"]
    return json.dumps(_without_binary(block), ensure_ascii=False, default=str)


def _invoke_reply_text(body: Dict[str, Any]) -> str:
    return "\n".join(_block_text(block) for block in body.get("content") or ())


def _converse_reply_text(response: Dict[str, Any]) -> str:
    content = response.get("output", {}).get("message", {}).get("content") or ()
    return "\n".join(_block_text(block) for block in content)


def _reply_text(operation_name: str, reply_source: Any) -> str:
    if isinstance(reply_source, str):
        return reply_source
    if operation_name == "Converse":
        return _converse_reply_text(reply_source)
    return _invoke_reply_text(reply_source)


def _capture_enabled() -> bool:
    from revenium_middleware._core.config import is_capture_prompts_enabled

    return is_capture_prompts_enabled()


def _captured_fields(operation_name: str, api_params: Dict[str, Any], reply_source: Any) -> Dict[str, Any]:
    """The record's prompt capture fields, empty when capture is off or the
    request or reply cannot be read. ``reply_source`` is the parsed
    InvokeModel body, the Converse response, or the joined stream text."""
    try:
        if not _capture_enabled():
            return {}
        from revenium_middleware.anthropic.prompt_extractor import captured_prompt_fields

        return captured_prompt_fields(
            _anthropic_request(operation_name, api_params),
            accumulated_content=_reply_text(operation_name, reply_source),
        )
    except Exception as exc:  # noqa: BLE001 - capture must never cost the usage record
        logger.debug("Bedrock transport skipped prompt capture: %s", type(exc).__name__)
        return {}


def _observe_response(
    operation_name: str,
    api_params: Dict[str, Any],
    response: Dict[str, Any],
    instance: Any,
    request_time_dt: datetime.datetime,
    usage_metadata: Optional[Dict[str, Any]] = None,
    model_id: Optional[str] = None,
) -> Dict[str, Any]:
    model_id = model_id or api_params.get("modelId", "")
    region = getattr(getattr(instance, "meta", None), "region_name", None)
    transaction_id = _transaction_id(response)

    def emit(usage: _StreamUsage, is_streamed: bool, reply_source: Any) -> None:
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
            usage_metadata=usage_metadata,
            cache_creation_ttl_counts=usage.cache_creation_ttl_counts,
            prompt_fields=_captured_fields(operation_name, api_params, reply_source),
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
        usage.cache_creation_ttl_counts = extract_cache_creation_ttl_counts(body_usage)
        usage.stop_reason = body.get("stop_reason")
        emit(usage, is_streamed=False, reply_source=body)

    elif operation_name == "Converse":
        converse_usage = response.get("usage", {})
        usage = _StreamUsage()
        usage.input_tokens = converse_usage.get("inputTokens", 0) or 0
        usage.output_tokens = converse_usage.get("outputTokens", 0) or 0
        usage.cache_read = converse_usage.get("cacheReadInputTokens", 0) or 0
        usage.cache_creation = converse_usage.get("cacheWriteInputTokens", 0) or 0
        usage.cache_creation_ttl_counts = extract_cache_creation_ttl_counts(converse_usage)
        usage.stop_reason = response.get("stopReason")
        emit(usage, is_streamed=False, reply_source=response)

    elif operation_name == "InvokeModelWithResponseStream":
        usage = _StreamUsage(keep_text=_capture_enabled())
        response["body"] = _MeteredEventStream(
            response["body"],
            lambda event: _parse_invoke_chunk(event, usage),
            lambda: emit(usage, is_streamed=True, reply_source="".join(usage.reply_text)),
        )

    elif operation_name == "ConverseStream":
        usage = _StreamUsage(keep_text=_capture_enabled())
        response["stream"] = _MeteredEventStream(
            response["stream"],
            lambda event: _parse_converse_event(event, usage),
            lambda: emit(usage, is_streamed=True, reply_source="".join(usage.reply_text)),
        )

    return response


def _metered_model_id(instance, args) -> Optional[str]:
    """The Bedrock model id this call is checked and recorded under, or None
    to pass it through. Gate evaluation only -- must never invoke the wrapped
    call, so a gate error can safely fall back to plain pass-through without
    risking a second invocation."""
    try:
        if not is_transport_enabled():
            return None
        operation_name = args[0]
        api_params = args[1] if len(args) > 1 else {}
        service_name = instance.meta.service_model.service_name
        if (
            service_name != "bedrock-runtime"
            or operation_name not in _SUPPORTED_OPERATIONS
            or _internal_adapter_call.get()
        ):
            return None
        model_id = api_params.get("modelId", "")
        if intercepts_model(model_id):
            return recorded_bedrock_model_id(model_id)
        if is_routed_arn(model_id):
            return _profile_resolver.claude_model_id(instance, model_id, wait=False)
        return None
    except Exception as exc:  # noqa: BLE001 - never break the caller's request
        logger.debug("Bedrock transport gate error: %s", exc)
        return None


def _make_api_call_wrapper(wrapped, instance, args, kwargs):
    model_id = _metered_model_id(instance, args)
    if model_id is None:
        return wrapped(*args, **kwargs)

    requested_model_id = args[1].get("modelId") if len(args) > 1 else None
    if selectively_unmetered():
        check_unmetered_call(_ambient_usage_metadata(), model=cap_model_names(model_id, requested_model_id),
                             provider="AWS", callback_transport=BEDROCK)
        return wrapped(*args, **kwargs)

    request_time_dt = datetime.datetime.now(datetime.timezone.utc)
    usage_metadata = _call_usage_metadata()
    check_enforcement(usage_metadata, model=cap_model_names(model_id, requested_model_id), provider="AWS")
    metering_client = current_client()
    response = wrapped(*args, **kwargs)  # exceptions/retries pass through untouched

    try:
        with recorded_under(metering_client):
            observed = _observe_response(
                args[0], args[1] if len(args) > 1 else {}, response, instance,
                request_time_dt, usage_metadata, model_id=model_id
            )
        _claim_for_langchain_callback(response)
        return observed
    except Exception as exc:  # noqa: BLE001 - metering must never break callers
        logger.warning("Bedrock transport observation failed: %s", exc)
        return response


def _claim_for_langchain_callback(response: Any) -> None:
    """Tell an attached LangChain callback this call is metered here, once
    its record is sent or its stream is wrapped (never when metering is off)."""
    from revenium_middleware import get_client

    if get_client() is not None:
        claim_call_for_transport(BEDROCK, _request_id(response))


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

    if wrap_once("botocore.client.BaseClient._make_api_call",
                 "botocore.client", "BaseClient._make_api_call", _make_api_call_wrapper):
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
