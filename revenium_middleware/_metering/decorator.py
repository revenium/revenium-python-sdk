"""
Revenium metering decorator for tool calls.

Automatically meters function execution with timing, success/failure, and attribution.

Data flow: Python SDK -> Metering API (/v2/tool/events) -> Kafka -> Clickhouse
"""

from __future__ import annotations

import os
import time
import uuid
import asyncio
import functools
from typing import Any, Dict, List, TypeVar, Callable, Optional, cast
from datetime import datetime, timezone

import httpx

from .context import ReveniumContext, get_context
from ._utils._logs import logger

__all__ = ["meter_tool", "report_tool_call", "configure"]

F = TypeVar("F", bound=Callable[..., Any])

# Global configuration (set via configure()); None falls back to the
# REVENIUM_METERING_* environment variables.
_metering_url: Optional[str] = None
_api_key: Optional[str] = None

_DEFAULT_BASE_URL = "https://api.revenium.ai"


def configure(
    metering_url: Optional[str] = None,
    api_key: Optional[str] = None,
) -> None:
    """
    Configure tool metering explicitly.

    Explicit values take precedence over the REVENIUM_METERING_BASE_URL /
    REVENIUM_METERING_API_KEY environment variables; pass None to clear an
    override and fall back to the environment. Calling configure() is
    optional -- with the standard environment variables set, tool events are
    delivered without it.

    Example:
        from revenium_middleware import configure
        configure(
            metering_url="https://api.revenium.ai",
            api_key="hak_your_api_key",
        )
    """
    global _metering_url, _api_key
    _metering_url = metering_url
    _api_key = api_key


def _resolve_endpoint() -> "tuple[Optional[str], Optional[str]]":
    """Resolve (events_url, api_key) for tool events.

    Precedence: explicit configure() values > REVENIUM_METERING_* env vars.
    Returns (None, None) when no API key is available anywhere -- tool
    metering is then skipped instead of posting demo credentials to
    localhost. The /meter prefix matches the main metering client's API
    layout and is not duplicated when the base URL already carries it.
    """
    # Snapshot both overrides in a single statement so a concurrent
    # configure() cannot yield a mismatched key/URL pair for one event.
    url_override, key_override = _metering_url, _api_key
    key = key_override or os.environ.get("REVENIUM_METERING_API_KEY")
    if not key:
        return None, None
    base = url_override or os.environ.get("REVENIUM_METERING_BASE_URL") or _DEFAULT_BASE_URL
    base = base.rstrip("/")
    if not base.endswith("/meter"):
        base = base + "/meter"
    return f"{base}/v2/tool/events", key


def _build_event_payload(
    tool_id: str,
    operation: Optional[str],
    duration_ms: int,
    success: bool,
    error_message: Optional[str],
    usage_metadata: Optional[Dict[str, Any]],
    context: ReveniumContext,
) -> Dict[str, Any]:
    """Build the event payload for the metering API."""
    transaction_id = context.transaction_id or str(uuid.uuid4())
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    event_payload: Dict[str, Any] = {
        "transactionId": transaction_id,
        "toolId": tool_id,
        "operation": operation or "execute",
        "durationMs": duration_ms,
        "success": success,
        "timestamp": timestamp,
    }

    if error_message:
        event_payload["errorMessage"] = error_message

    if usage_metadata:
        event_payload["usageMetadata"] = usage_metadata

    # Add context fields (agent, product, organizationName, etc.)
    if context.agent:
        event_payload["agent"] = context.agent
    # Use new field names, fallback to deprecated fields for backward compatibility
    if context.organization_name:
        event_payload["organizationName"] = context.organization_name
    elif context.organization_id:
        event_payload["organizationName"] = context.organization_id
    if context.product_name:
        event_payload["productName"] = context.product_name
    elif context.product:
        event_payload["productName"] = context.product
    if context.subscriber_credential:
        event_payload["subscriberCredential"] = context.subscriber_credential
    if context.workflow_id:
        event_payload["workflowId"] = context.workflow_id
    if context.trace_id:
        event_payload["traceId"] = context.trace_id

    return event_payload


def _dispatch_tool_event(**event_kwargs: Any) -> None:
    """
    Schedule the tool event on the SDK's fire-and-forget metering thread.

    Tool metering must never block the wrapped call -- the same guarantee
    AI-completion metering provides by dispatching via run_async_in_thread.
    The metering thread is joined during SDK shutdown, so events still flush
    on process exit.
    """
    try:
        # Imported lazily: this module is part of revenium_middleware's import
        # cycle, and by the time an event fires the package is fully loaded.
        from revenium_middleware import run_async_in_thread, shutdown_event

        if shutdown_event.is_set():
            logger.warning("Skipping tool metering during shutdown")
            return

        # Resolve the endpoint NOW, not when the background coroutine runs:
        # a configure() call racing the queued event must not change where or
        # under which key this event is delivered.
        url, key = _resolve_endpoint()
        if url is None:
            logger.warning(
                "Tool metering skipped: no API key configured "
                "(set REVENIUM_METERING_API_KEY or call configure())"
            )
            return
        coro = _send_tool_event_async(url, key, **event_kwargs)
    except Exception as e:
        # Non-blocking - just log and continue
        logger.warning("metering error: %s", e)
        return

    # From here on the coroutine exists but is not yet scheduled; every path
    # that fails to hand it off must close() it, otherwise garbage collection
    # emits a "coroutine was never awaited" RuntimeWarning.
    try:
        thread = run_async_in_thread(coro)
        if thread is None:
            # Not scheduled (e.g. shutdown won the race after our pre-check).
            coro.close()
    except Exception as e:
        coro.close()
        # Non-blocking - just log and continue
        logger.warning("metering error: %s", e)


async def _send_tool_event_async(
    url: str,
    key: str,
    tool_id: str,
    operation: Optional[str],
    duration_ms: int,
    success: bool,
    error_message: Optional[str],
    usage_metadata: Optional[Dict[str, Any]],
    context: ReveniumContext,
) -> None:
    """
    Async version of _send_tool_event for use in async contexts.

    Uses httpx.AsyncClient to avoid blocking the event loop. The endpoint and
    key are resolved by the dispatcher at enqueue time and passed in.
    """
    event_payload = _build_event_payload(
        tool_id, operation, duration_ms, success, error_message, usage_metadata, context
    )

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.post(
                url,
                headers={
                    "x-api-key": key,
                    "Content-Type": "application/json",
                    # Dedupes redelivery (e.g. buffer replay after a network
                    # drop the backend actually accepted) when supported;
                    # harmless if the backend ignores it.
                    "Idempotency-Key": event_payload["transactionId"],
                },
                json=event_payload,
            )
            response.raise_for_status()
            logger.debug("[metered] %s:%s %dms", tool_id, operation or "execute", duration_ms)
    except Exception as e:
        try:
            from revenium_middleware._core.metering_buffer import get_buffer, is_retryable_failure

            if is_retryable_failure(e):
                get_buffer().push(
                    "tool", {"url": url, "key": key, "event_payload": event_payload}
                )
                logger.warning("metering error (event buffered for replay): %s", e)
                return
        except Exception as buffer_exc:  # buffering must never mask the original error
            logger.warning("metering buffer error: %s", buffer_exc)
        # Non-blocking - just log and continue
        logger.warning("metering error: %s", e)


def report_tool_call(
    tool_id: str,
    operation: Optional[str] = None,
    duration_ms: int = 0,
    success: bool = True,
    error_message: Optional[str] = None,
    usage_metadata: Optional[Dict[str, Any]] = None,
    agent: Optional[str] = None,
    # Deprecated parameters - kept for backward compatibility
    organization_id: Optional[str] = None,
    product: Optional[str] = None,
    # New parameters
    organization_name: Optional[str] = None,
    product_name: Optional[str] = None,
    subscriber_credential: Optional[str] = None,
    workflow_id: Optional[str] = None,
    trace_id: Optional[str] = None,
    transaction_id: Optional[str] = None,
) -> None:
    """
    Manually report a tool call.

    Example:
        revenium.report_tool_call(
            tool_id="firecrawl",
            operation="scrape",
            duration_ms=1234,
            success=True,
            usage_metadata={"pages": 5, "data_mb": 2.3},
            agent="my-agent",
            organization_name="AcmeCorp",
            product_name="chatbot"
        )
    """
    ctx = get_context().merge(
        agent=agent,
        organization_id=organization_id,
        product=product,
        organization_name=organization_name,
        product_name=product_name,
        subscriber_credential=subscriber_credential,
        workflow_id=workflow_id,
        trace_id=trace_id,
        transaction_id=transaction_id,
    )
    _dispatch_tool_event(
        tool_id=tool_id,
        operation=operation,
        duration_ms=duration_ms,
        success=success,
        error_message=error_message,
        usage_metadata=usage_metadata,
        context=ctx,
    )


def _extract_output_fields(result: Any, output_fields: Optional[List[str]]) -> Optional[Dict[str, Any]]:
    """Extract specified fields from the result for usage metadata.

    Wrapped in try/except to avoid overriding the wrapped function's exception
    if a property accessor throws.
    """
    if not output_fields or result is None:
        return None

    try:
        usage_metadata: Dict[str, Any] = {}
        for field_name in output_fields:
            if isinstance(result, dict):
                if field_name in result:
                    usage_metadata[field_name] = result[field_name]
            elif hasattr(result, field_name):
                usage_metadata[field_name] = getattr(result, field_name)

        return usage_metadata if usage_metadata else None
    except Exception:
        # Don't let output field extraction errors override the wrapped function's result
        return None


def meter_tool(
    tool_id: str,
    operation: Optional[str] = None,
    output_fields: Optional[List[str]] = None,
    agent: Optional[str] = None,
    # Deprecated parameters - kept for backward compatibility
    organization_id: Optional[str] = None,
    product: Optional[str] = None,
    # New parameters
    organization_name: Optional[str] = None,
    product_name: Optional[str] = None,
    subscriber_credential: Optional[str] = None,
    workflow_id: Optional[str] = None,
    trace_id: Optional[str] = None,
) -> Callable[[F], F]:
    """
    Decorator to automatically meter tool function calls.

    Captures timing, success/failure, and reports to Revenium metering.

    Example:
        @revenium.meter_tool("firecrawl")
        def scrape(url):
            return firecrawl.scrape(url)

        @revenium.meter_tool("fal_flux", operation="generate_image", agent="image-bot")
        async def generate_image(prompt):
            return await fal.run("fal-ai/flux", {"prompt": prompt})

        @revenium.meter_tool("runway", output_fields=["duration_seconds", "resolution"])
        def generate_video(prompt):
            return runway.generate(prompt)
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            ctx = get_context().merge(
                agent=agent,
                organization_id=organization_id,
                product=product,
                organization_name=organization_name,
                product_name=product_name,
                subscriber_credential=subscriber_credential,
                workflow_id=workflow_id,
                trace_id=trace_id,
            )
            # Only generate transaction_id if not provided by context
            if not ctx.transaction_id:
                ctx = ctx.merge(transaction_id=str(uuid.uuid4()))

            start_time = time.perf_counter()
            call_success = True
            call_error_message: Optional[str] = None
            result: Any = None

            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                call_success = False
                call_error_message = str(e)
                raise
            finally:
                duration_ms = int((time.perf_counter() - start_time) * 1000)
                usage_metadata = _extract_output_fields(result, output_fields)

                _dispatch_tool_event(
                    tool_id=tool_id,
                    operation=operation,
                    duration_ms=duration_ms,
                    success=call_success,
                    error_message=call_error_message,
                    usage_metadata=usage_metadata,
                    context=ctx,
                )

        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            ctx = get_context().merge(
                agent=agent,
                organization_id=organization_id,
                product=product,
                organization_name=organization_name,
                product_name=product_name,
                subscriber_credential=subscriber_credential,
                workflow_id=workflow_id,
                trace_id=trace_id,
            )
            # Only generate transaction_id if not provided by context
            if not ctx.transaction_id:
                ctx = ctx.merge(transaction_id=str(uuid.uuid4()))

            start_time = time.perf_counter()
            call_success = True
            call_error_message: Optional[str] = None
            result: Any = None

            try:
                result = await func(*args, **kwargs)
                return result
            except Exception as e:
                call_success = False
                call_error_message = str(e)
                raise
            finally:
                duration_ms = int((time.perf_counter() - start_time) * 1000)
                usage_metadata = _extract_output_fields(result, output_fields)

                _dispatch_tool_event(
                    tool_id=tool_id,
                    operation=operation,
                    duration_ms=duration_ms,
                    success=call_success,
                    error_message=call_error_message,
                    usage_metadata=usage_metadata,
                    context=ctx,
                )

        # Auto-detect sync vs async
        if asyncio.iscoroutinefunction(func):
            return cast(F, async_wrapper)
        return cast(F, sync_wrapper)

    return decorator
