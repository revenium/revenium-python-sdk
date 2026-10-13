"""
Revenium metering decorator for tool calls.

Automatically meters function execution with timing, success/failure, and attribution.

Data flow: Python SDK -> Metering API (/v2/tool/events) -> Kafka -> Clickhouse
"""

from __future__ import annotations

import os
import time
import uuid
import inspect
import contextlib
import functools
import itertools
import threading
from typing import Any, Dict, Iterator, List, TypeVar, Callable, NamedTuple, Optional, cast
from datetime import datetime, timezone
from collections.abc import Mapping

from .context import ReveniumContext, get_context
from ._client import normalize_meter_base_url, warn_that_the_address_query_was_dropped
from ._utils._logs import logger

# Safe at module level despite this file's import cycle with the top-level
# package: metering_status is a leaf submodule with only stdlib dependencies.
from revenium_middleware._core.metering_status import (
    record_metering_error,
    record_metering_success,
)
from revenium_middleware._core.json_safe import json_safe_record

# Same reasoning: fields (and the context module it reads the ambient job from)
# depend only on stdlib plus other _core leaves, and revenium_middleware/__init__
# finishes importing _core before it reaches _metering. Reusing this resolver --
# rather than re-implementing the precedence here -- is what keeps tool events
# and AI completions attributing spend to the same agentic job.
from revenium_middleware._core.fields import extract_agentic_job_fields, is_valid_agentic_job_id
from revenium_middleware._core.error_text import reportable_exception_text
from revenium_middleware._core.decorators import _is_generator_function
from revenium_middleware._core.context import _agentic_job_context, get_agentic_job_fields, snapshot_metadata

__all__ = ["meter_tool", "report_tool_call", "configure"]

F = TypeVar("F", bound=Callable[..., Any])

# Fixed namespace for the per-event transaction id derived from a caller's
# transaction id. Changing it changes every derived id, so a buffered event
# replayed by an upgraded process would no longer dedupe against its first send.
_TOOL_EVENT_TRANSACTION_ID_NAMESPACE = uuid.UUID("2305e5c5-6113-5d7f-a5c8-3906af47f224")

# Global configuration (set via configure()); None falls back to the
# REVENIUM_METERING_* environment variables.
_metering_url: Optional[str] = None
_api_key: Optional[str] = None
_ai_metering_warned = False
_no_key_warned = False
_warn_once_lock = threading.Lock()
_address_query_warned = False


def configure(
    metering_url: Optional[str] = None,
    api_key: Optional[str] = None,
) -> None:
    """
    Configure tool metering explicitly.

    Explicit values take precedence over the key and address of the AI
    metering client (initialize_metering() or the REVENIUM_METERING_API_KEY /
    REVENIUM_METERING_BASE_URL environment variables); pass None to clear an
    override and fall back to them. Calling configure() is optional -- with
    the standard environment variables set, tool events are delivered
    without it. Inside a ``metering_key()`` block, ``metering_url`` alone does
    not apply: the block's address goes with the block's key.

    Example:
        from revenium_middleware import configure
        configure(
            metering_url="https://api.revenium.ai",
            api_key="hak_your_api_key",
        )
    """
    global _metering_url, _api_key
    _metering_url = metering_url
    _api_key = api_key.strip() if api_key else api_key
    if _api_key:
        _warn_once_that_ai_calls_are_not_metered()


def _warn_once_that_tool_events_are_not_metered() -> None:
    global _no_key_warned
    with _warn_once_lock:
        if _no_key_warned:
            return
        _no_key_warned = True
    logger.warning(
        "Tool metering skipped: no valid API key configured "
        "(set REVENIUM_METERING_API_KEY, call initialize_metering() or configure())"
    )


def _warn_once_that_ai_calls_are_not_metered() -> None:
    global _ai_metering_warned
    # Imported lazily: this module is part of revenium_middleware's import cycle.
    from revenium_middleware._core.metering import get_client

    if get_client() is not None:
        return
    with _warn_once_lock:
        if _ai_metering_warned:
            return
        _ai_metering_warned = True
    logger.warning(
        "configure() sets the Revenium key for tool events only; AI calls are not metered until "
        "REVENIUM_METERING_API_KEY is set or revenium_middleware.initialize_metering(api_key=...) is called."
    )


def _warn_once_that_the_address_query_was_dropped() -> None:
    global _address_query_warned
    if _address_query_warned:
        return
    _address_query_warned = True
    warn_that_the_address_query_was_dropped()


def _env_or_default_base_url() -> str:
    # Imported lazily: this module is part of revenium_middleware's import cycle.
    from revenium_middleware._core.metering import DEFAULT_METERING_BASE_URL, _valid_base_url

    return _valid_base_url(os.environ.get("REVENIUM_METERING_BASE_URL")) or DEFAULT_METERING_BASE_URL


def _process_wide_base_url() -> str:
    # Imported lazily: this module is part of revenium_middleware's import cycle.
    from revenium_middleware._core.metering import process_client

    process = process_client()
    return str(process.base_url) if process is not None else _env_or_default_base_url()


def _resolve_endpoint() -> "tuple[Optional[str], Optional[str]]":
    """Resolve (events_url, api_key) for tool events.

    Precedence: configure()'s key with configure()'s address, else the
    process-wide address > inside a metering_key() block, the block's key
    with the block's address > configure()'s address with the process
    client's key > the AI metering client (built by initialize_metering() or
    from the REVENIUM_METERING_* env vars) > a valid
    REVENIUM_METERING_BASE_URL or the default. Inside a metering_key() block
    the key and the address always come from one source. Returns (None, None)
    when no valid API key is available anywhere -- tool metering is then
    skipped instead of posting demo credentials to localhost. The /meter
    prefix matches the main metering client's API layout and is not
    duplicated when the base URL already carries it.
    """
    # Imported lazily: this module is part of revenium_middleware's import cycle.
    from revenium_middleware._core.metering import current_client, is_scoped_client

    # Snapshot both overrides in a single statement so a concurrent
    # configure() cannot yield a mismatched key/URL pair for one event.
    url_override, key_override = _metering_url, _api_key
    client = current_client()
    if is_scoped_client(client):
        if key_override:
            key, base = key_override, url_override or _process_wide_base_url()
        else:
            key, base = client.api_key, str(client.base_url)
    else:
        key = key_override or (client.api_key if client is not None else None)
        base = url_override or (str(client.base_url) if client is not None else _env_or_default_base_url())
    if not key:
        return None, None
    meter_base, dropped = normalize_meter_base_url(base)
    if dropped:
        _warn_once_that_the_address_query_was_dropped()
    return f"{meter_base}v2/tool/events", key


def _resolve_agentic_job_id(usage_metadata: Any) -> Optional[str]:
    """Resolve the agentic job id for a tool event, or None when it is unset.

    Precedence mirrors the AI-completion path exactly, because it is the same
    resolver: the call's ``usage_metadata`` (``agentic_job_id``, then the
    ``agenticJobId`` alias) > the ambient job contextvar set by
    ``set_agentic_job_fields`` / ``JobContext`` > the ``REVENIUM_AGENTIC_JOB_ID``
    environment variable. ``usage_metadata`` is typed ``Any`` because
    ``report_tool_call`` forwards whatever the caller passed; anything that is
    not a mapping degrades to ``{}`` so the ambient/env sources still resolve.

    Never raises. Tool events are dispatched fire-and-forget from inside the
    wrapped tool call, so values the wire cannot carry drop the field rather
    than surface an error to the caller. An id that fails the server's
    ``agenticJobId`` pattern is dropped too (see ``is_valid_agentic_job_id``):
    sending it would 400 the whole event, silently losing the event and its
    cost instead of just the attribution.
    """
    try:
        source = usage_metadata if isinstance(usage_metadata, Mapping) else {}
        job_id = extract_agentic_job_fields(source).get("agenticJobId")
        if not (isinstance(job_id, str) and job_id):
            return None
        if not is_valid_agentic_job_id(job_id):
            logger.debug(
                "Dropping agenticJobId that would fail server validation: %r", job_id
            )
            return None
    except Exception:
        return None
    return job_id


def _build_event_payload(
    tool_id: str,
    operation: Optional[str],
    duration_ms: int,
    success: bool,
    error_message: Optional[str],
    usage_metadata: Optional[Dict[str, Any]],
    context: ReveniumContext,
    occurred_at: datetime,
    event_sequence: str,
) -> Dict[str, Any]:
    """Build the event payload for the metering API."""
    timestamp = occurred_at.strftime("%Y-%m-%dT%H:%M:%SZ")
    caller_transaction_id = context.transaction_id
    if caller_transaction_id:
        transaction_id = _derive_tool_event_transaction_id(
            caller_transaction_id, tool_id, operation, occurred_at, event_sequence
        )
        usage_metadata = _with_caller_transaction_id(usage_metadata, caller_transaction_id, context)
    else:
        transaction_id = str(uuid.uuid4())

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

    # Joins this tool's spend to the agentic job that drove it. Omitted
    # entirely when nothing resolves: the field is optional and nullable on
    # POST /v2/tool/events, and an explicit null carries no information.
    agentic_job_id = _resolve_agentic_job_id(usage_metadata)
    if agentic_job_id:
        event_payload["agenticJobId"] = agentic_job_id

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
    elif caller_transaction_id:
        event_payload["traceId"] = caller_transaction_id

    return json_safe_record(event_payload, f"Tool event for tool {tool_id}")


def _derive_tool_event_transaction_id(
    caller_transaction_id: str, tool_id: str, operation: Optional[str], occurred_at: datetime, event_sequence: str
) -> str:
    """One tool call's transaction id, derived from the caller's grouping id and the call itself.

    The server treats ``transactionId`` as one tool call's id: it dedupes on it
    and parses it as a UUID. The name joins the caller's id, the tool, the
    operation, the stamped time and the event's per-process dispatch sequence,
    so two real calls get different ids even when the clock repeats a
    timestamp, while an SDK retry or a buffer replay of the same event keeps
    its id.
    """
    name = "\x1f".join(
        [caller_transaction_id, tool_id, operation or "execute", occurred_at.isoformat(), event_sequence]
    )
    return str(uuid.uuid5(_TOOL_EVENT_TRANSACTION_ID_NAMESPACE, name))


_dispatch_nonce = uuid.uuid4().hex
_dispatch_counter = itertools.count()
_dispatch_lock = threading.Lock()


def _next_event_sequence() -> str:
    with _dispatch_lock:
        return f"{_dispatch_nonce}:{next(_dispatch_counter)}"


def _restart_event_sequence() -> None:
    """Start a forked child on its own nonce, counter and lock.

    A child that kept the parent's pair would reproduce the parent's next ids,
    and a lock held by a parent thread at fork time would never be released.
    """
    global _dispatch_nonce, _dispatch_counter, _dispatch_lock
    _dispatch_nonce = uuid.uuid4().hex
    _dispatch_counter = itertools.count()
    _dispatch_lock = threading.Lock()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_restart_event_sequence)


def _with_caller_transaction_id(
    usage_metadata: Optional[Dict[str, Any]], caller_transaction_id: str, context: ReveniumContext
) -> Optional[Dict[str, Any]]:
    """Carry the caller's transaction id in usage metadata when ``traceId`` already holds a trace id.

    Returns a copy; the caller's dict is never changed, and a
    ``callerTransactionId`` the caller supplied is kept.
    """
    if not context.trace_id:
        return usage_metadata
    if isinstance(usage_metadata, Mapping):
        return {"callerTransactionId": caller_transaction_id, **usage_metadata}
    if usage_metadata:
        return usage_metadata
    return {"callerTransactionId": caller_transaction_id}


def _dispatch_tool_event(**event_kwargs: Any) -> None:
    """
    Schedule the tool event on the SDK's fire-and-forget metering thread.

    Tool metering must never block the wrapped call -- the same guarantee
    AI-completion metering provides by dispatching via run_async_in_thread.
    At exit the shutdown drain gives queued events its budget to be sent.
    """
    tool_id = event_kwargs.get("tool_id")
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
            _warn_once_that_tool_events_are_not_metered()
            return
        event_kwargs["usage_metadata"] = snapshot_metadata(event_kwargs.get("usage_metadata"))
        coro = _send_tool_event_async(
            url,
            key,
            occurred_at=datetime.now(timezone.utc),
            event_sequence=_next_event_sequence(),
            **event_kwargs,
        )
    except Exception as e:
        record_metering_error(e, operation="tool")
        # Non-blocking - just log and continue
        logger.error("metering error for tool %s: %s", tool_id, e)
        return

    # From here on the coroutine exists but is not yet scheduled; every path
    # that fails to hand it off must close() it, otherwise garbage collection
    # emits a "coroutine was never awaited" RuntimeWarning.
    try:
        thread = run_async_in_thread(coro, gated_by_circuit=False)
        if thread is None:
            # Not scheduled (e.g. shutdown won the race after our pre-check).
            coro.close()
    except Exception as e:
        coro.close()
        record_metering_error(e, operation="tool")
        # Non-blocking - just log and continue
        logger.error("metering error for tool %s: %s", tool_id, e)


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
    occurred_at: datetime,
    event_sequence: str,
) -> None:
    """
    Deliver one tool event on a metering worker.

    Posts through the shared tool-event client, so consecutive events reuse
    pooled connections. Each worker runs one task at a time on its own loop,
    so the blocking post holds up no other event. The endpoint, the key and
    the event's ``occurred_at`` and ``event_sequence`` are resolved by the dispatcher at enqueue
    time and passed in.
    """
    event_payload = _build_event_payload(
        tool_id, operation, duration_ms, success, error_message, usage_metadata, context, occurred_at, event_sequence
    )
    tool_event = {"url": url, "key": key, "event_payload": event_payload}
    if deliver_tool_event(tool_event):
        logger.debug("[metered] %s:%s %dms", tool_id, operation or "execute", duration_ms)


def deliver_tool_event(tool_event: Dict[str, Any]) -> bool:
    """
    Send one tool event now, once, with the metering timeouts; True when it was delivered.

    A retryable failure puts the event in the store-and-forward buffer, and an
    event built after it overflowed goes there unsent. Failures are logged,
    never raised.
    """
    from revenium_middleware._core.metering_buffer import (
        buffer_deferred_event,
        get_buffer,
        is_delivery_deferred_to_buffer,
    )

    if is_delivery_deferred_to_buffer():
        buffer_deferred_event("tool", tool_event)
        return False

    from revenium_middleware._core.metering import metering_client_timeout

    from revenium_middleware._core.tool_event_transport import post_tool_event

    tool_id = tool_event["event_payload"].get("toolId")
    try:
        post_tool_event(tool_event, metering_client_timeout())
        record_metering_success()
        return True
    except Exception as e:
        record_metering_error(e, operation="tool")
        try:
            from revenium_middleware._core.metering_buffer import is_retryable_failure

            if is_retryable_failure(e):
                get_buffer().push("tool", tool_event)
                logger.error("metering error for tool %s (event buffered for replay): %s", tool_id, e)
                return False
        except Exception as buffer_exc:  # buffering must never mask the original error
            logger.warning("metering buffer error: %s", buffer_exc)
        # Non-blocking - just log and continue
        logger.error("metering error for tool %s: %s", tool_id, e)
        return False


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


class ToolCallStart(NamedTuple):
    context: ReveniumContext
    start_time: float
    job_fields: Optional[Dict[str, Any]]
    metering_client: Any


ToolCallBegin = Callable[[], ToolCallStart]
ToolCallReport = Callable[[ToolCallStart, bool, Optional[str], Any], None]


@contextlib.contextmanager
def _first_step_job(started: ToolCallStart) -> Iterator[None]:
    """Make the agentic job active at a stream's first step the one its event is reported under."""
    token = _agentic_job_context.set(started.job_fields)
    try:
        yield
    finally:
        _agentic_job_context.reset(token)


def _metered_function(func: Callable[..., Any], begin: ToolCallBegin, report: ToolCallReport) -> Callable[..., Any]:
    @functools.wraps(func)
    def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
        started = begin()
        call_success = True
        call_error_message: Optional[str] = None
        result: Any = None

        try:
            result = func(*args, **kwargs)
            return result
        except Exception as e:
            call_success = False
            call_error_message = reportable_exception_text(e)
            raise
        finally:
            report(started, call_success, call_error_message, result)

    return sync_wrapper


def _metered_coroutine_function(func: Callable[..., Any], begin: ToolCallBegin, report: ToolCallReport) -> Callable[..., Any]:
    @functools.wraps(func)
    async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
        started = begin()
        call_success = True
        call_error_message: Optional[str] = None
        result: Any = None

        try:
            result = await func(*args, **kwargs)
            return result
        except Exception as e:
            call_success = False
            call_error_message = reportable_exception_text(e)
            raise
        finally:
            report(started, call_success, call_error_message, result)

    return async_wrapper


def _metered_generator_function(func: Callable[..., Any], begin: ToolCallBegin, report: ToolCallReport) -> Callable[..., Any]:
    """A generator function that meters ``func``'s whole stream once, from its first step to its end."""
    @functools.wraps(func)
    def metered_generator(*args: Any, **kwargs: Any) -> Any:
        started = begin()
        success, error_message, last_item = True, None, None
        try:
            inner = func(*args, **kwargs)
            step, value = inner.send, None
            while True:
                try:
                    last_item = step(value)
                except StopIteration as stop:
                    return stop.value
                try:
                    value = yield last_item
                    step = inner.send
                except GeneratorExit:
                    inner.close()
                    raise
                except BaseException as exc:
                    step, value = inner.throw, exc
        except Exception as exc:
            success, error_message = False, reportable_exception_text(exc)
            raise
        finally:
            with _first_step_job(started):
                report(started, success, error_message, last_item)

    return metered_generator


def _metered_async_generator_function(
    func: Callable[..., Any], begin: ToolCallBegin, report: ToolCallReport
) -> Callable[..., Any]:
    """An async generator function that meters ``func``'s whole stream once, from its first step to its end."""
    @functools.wraps(func)
    async def metered_async_generator(*args: Any, **kwargs: Any) -> Any:
        started = begin()
        success, error_message, last_item = True, None, None
        try:
            inner = func(*args, **kwargs)
            step, value = inner.asend, None
            while True:
                try:
                    last_item = await step(value)
                except StopAsyncIteration:
                    return
                try:
                    value = yield last_item
                    step = inner.asend
                except GeneratorExit:
                    await inner.aclose()
                    raise
                except BaseException as exc:
                    step, value = inner.athrow, exc
        except Exception as exc:
            success, error_message = False, reportable_exception_text(exc)
            raise
        finally:
            with _first_step_job(started):
                report(started, success, error_message, last_item)

    return metered_async_generator


def _metered_function_kind(
    func: Callable[..., Any],
) -> Callable[[Callable[..., Any], ToolCallBegin, ToolCallReport], Callable[..., Any]]:
    if inspect.isasyncgenfunction(func):
        return _metered_async_generator_function
    if _is_generator_function(func):
        return _metered_generator_function
    if inspect.iscoroutinefunction(func):
        return _metered_coroutine_function
    return _metered_function


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

    Generator and async generator tools (streaming) stay generators and are
    recorded once, when the stream ends: it is exhausted, it raises (recorded as
    failed, and the exception still reaches the caller) or the caller closes it
    (recorded as a success). The duration runs from the first item requested to
    the end of the stream, so it includes the caller's time between items. The
    context, the agentic job, the Revenium key (the ``metering_key()`` block then
    active) and the start time are read at the first step, so open
    ``revenium.context(...)`` around the iteration rather than around creating
    the generator. ``output_fields`` are read from the last item the tool
    yielded. A stream closed before its first item, or never iterated, sends
    nothing. An abandoned async generator is recorded only when the event loop
    finalizes it (``aclose`` from its asyncgen hooks or ``shutdown_asyncgens``).
    Task cancellation mid-stream is recorded as a success, as it is for a
    coroutine tool.
    """

    def decorator(func: F) -> F:
        def begin() -> ToolCallStart:
            # Imported lazily: this module is part of revenium_middleware's import cycle.
            from revenium_middleware._core.metering import current_client

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
            return ToolCallStart(ctx, time.perf_counter(), get_agentic_job_fields(), current_client())

        def report(started: ToolCallStart, success: bool, error_message: Optional[str], result: Any) -> None:
            # Imported lazily: this module is part of revenium_middleware's import cycle.
            from revenium_middleware._core.metering import recorded_under

            duration_ms = int((time.perf_counter() - started.start_time) * 1000)
            usage_metadata = _extract_output_fields(result, output_fields)

            with recorded_under(started.metering_client):
                _dispatch_tool_event(
                    tool_id=tool_id,
                    operation=operation,
                    duration_ms=duration_ms,
                    success=success,
                    error_message=error_message,
                    usage_metadata=usage_metadata,
                    context=started.context,
                )

        return cast(F, _metered_function_kind(func)(func, begin, report))

    return decorator
