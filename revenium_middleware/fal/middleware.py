"""
Wrapt-based wrappers for fal_client API methods.

This module registers wrapt patches on fal_client. It requires fal_client
to be installed — the __init__.py handles graceful fallback when it is not.
"""

import contextlib
import contextvars
import copy
import dataclasses
import logging
import threading
from typing import Any, AsyncIterator, Dict, Iterator, List

import fal_client

from revenium_middleware._core.context import idempotency_key, overlay_metadata
from revenium_middleware._core.fields import extract_common_metadata, extract_org_and_product
from revenium_middleware._core.patch_registry import register_patch, unregister_patch, wrap_once
from revenium_middleware._core.subscriber import extract_subscriber_from_metadata
from ._call import FalCall, request_arguments
from ._queue import application_from_queue_url, job_transaction_id, queued_jobs

logger = logging.getLogger("revenium_middleware.fal")


def run_wrapper(wrapped, instance, args, kwargs):
    call = FalCall.from_request(args, kwargs)
    result = wrapped(*args, **kwargs)
    call.meter(result)
    return result


async def run_async_wrapper(wrapped, instance, args, kwargs):
    call = FalCall.from_request(args, kwargs)
    result = await wrapped(*args, **kwargs)
    call.meter(result)
    return result


def stream_wrapper(wrapped, instance, args, kwargs):
    call = FalCall.from_request(args, kwargs)
    return _metered_events(wrapped(*args, **kwargs), call, contextvars.copy_context())


def stream_async_wrapper(wrapped, instance, args, kwargs):
    return _metered_async_events(wrapped, args, kwargs, contextvars.copy_context())


def _meter_stream_end(call: FalCall, last_event, finished: bool) -> None:
    """One record for a stream however it ended; a stream that failed before any event has nothing to bill."""
    if last_event is None and not finished:
        return
    try:
        call.meter(last_event or {}, is_streamed=True)
    except Exception as e:
        logger.warning("Error metering the fal stream: %s", e)


# The meter runs in the caller's context from the start of the call, because a
# stream closed by the garbage collector may be finalised inside another job.
def _metered_events(events: Iterator[Dict[str, Any]], call: FalCall,
                    ctx: contextvars.Context) -> Iterator[Dict[str, Any]]:
    last_event, finished = None, False
    try:
        for event in events:
            last_event = event
            yield event
        finished = True
    finally:
        ctx.run(_meter_stream_end, call, last_event, finished)


async def _metered_async_events(wrapped, args, kwargs, ctx: contextvars.Context) -> AsyncIterator[Dict[str, Any]]:
    call = ctx.run(FalCall.from_request, args, kwargs)
    last_event, finished = None, False
    try:
        async for event in wrapped(*args, **kwargs):
            last_event = event
            yield event
        finished = True
    finally:
        ctx.run(_meter_stream_end, call, last_event, finished)


class _SubscribedSubmissions:
    """The submit each subscribe makes, already checked against the caps by the subscribe.

    With client_timeout, SyncClient.subscribe runs submit on fal's executor
    thread, where the caller's context variables are not visible, so the mark
    travels with the ``arguments`` object fal passes on to submit unchanged.
    Each subscribe hands fal its own copy of ``arguments``, so a direct submit
    that reuses the caller's dict never matches the mark.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._marks: Dict[int, List[object]] = {}

    @contextlib.contextmanager
    def marked(self, arguments):
        if arguments is None:
            yield
            return
        key, mark = id(arguments), object()
        with self._lock:
            self._marks.setdefault(key, []).append(mark)
        try:
            yield
        finally:
            with self._lock:
                self._discard(key, mark)

    def take(self, arguments) -> bool:
        if arguments is None:
            return False
        with self._lock:
            marks = self._marks.get(id(arguments))
            if not marks:
                return False
            self._discard(id(arguments), marks[0])
            return True

    def _discard(self, key, mark):
        marks = self._marks.get(key, [])
        if mark in marks:
            marks.remove(mark)
        if not marks:
            self._marks.pop(key, None)


_subscribed_submissions = _SubscribedSubmissions()


def _submitted_call(args, kwargs) -> FalCall:
    return FalCall.from_request(args, kwargs, already_checked=_subscribed_submissions.take(request_arguments(args, kwargs)))


def _track_job(request_id: str, call: FalCall) -> None:
    queued_jobs.track(request_id, call.awaiting_result())


def submit_wrapper(wrapped, instance, args, kwargs):
    call = _submitted_call(args, kwargs)
    handle = wrapped(*args, **kwargs)
    _track_job(handle.request_id, call)
    return handle


async def submit_async_wrapper(wrapped, instance, args, kwargs):
    call = _submitted_call(args, kwargs)
    handle = await wrapped(*args, **kwargs)
    _track_job(handle.request_id, call)
    return handle


def _subscribed_call(args, kwargs) -> FalCall:
    call = FalCall.from_request(args, kwargs)
    # With client_timeout, SyncClient.subscribe runs submit on fal's executor
    # thread, where the caller's context variables (injected metadata,
    # @revenium_meter scope) are not visible. on_enqueue is how fal hands the
    # request id back, so the caller-side context is attached to the job there.
    kwargs["on_enqueue"] = _tracking_on_enqueue(call, kwargs.get("on_enqueue"))
    return call


def _with_private_arguments(args, kwargs):
    """Swap the caller's ``arguments`` for a shallow copy; returns the new ``args`` and the copy."""
    arguments = request_arguments(args, kwargs)
    if arguments is None:
        return args, None
    private = copy.copy(arguments)
    if len(args) > 1:
        return (args[0], private, *args[2:]), private
    kwargs["arguments"] = private
    return args, private


def subscribe_wrapper(wrapped, instance, args, kwargs):
    args, arguments = _with_private_arguments(args, kwargs)
    _subscribed_call(args, kwargs)
    with _subscribed_submissions.marked(arguments):
        return wrapped(*args, **kwargs)


async def subscribe_async_wrapper(wrapped, instance, args, kwargs):
    args, arguments = _with_private_arguments(args, kwargs)
    _subscribed_call(args, kwargs)
    with _subscribed_submissions.marked(arguments):
        return await wrapped(*args, **kwargs)


def _tracking_on_enqueue(call: FalCall, on_enqueue):
    def track_then_notify(request_id):
        _track_job(request_id, call)
        if on_enqueue is not None:
            return on_enqueue(request_id)
        return None
    return track_then_notify


def get_handle_wrapper(wrapped, instance, args, kwargs):
    application, request_id = _handle_target(*args, **kwargs)
    queued_jobs.remember_application(request_id, application)
    return wrapped(*args, **kwargs)


def _handle_target(application, request_id):
    return application, request_id


def client_result_wrapper(wrapped, instance, args, kwargs):
    """``result(application, request_id, usage_metadata=...)``: fal's own result is
    ``get_handle(...).get()``, so the tags are handed to the handle's wrapped ``get``."""
    if "usage_metadata" not in kwargs:
        return wrapped(*args, **kwargs)
    fetch_tags = kwargs.pop("usage_metadata")
    return instance.get_handle(*args, **kwargs).get(usage_metadata=fetch_tags)


async def client_result_async_wrapper(wrapped, instance, args, kwargs):
    if "usage_metadata" not in kwargs:
        return await wrapped(*args, **kwargs)
    fetch_tags = kwargs.pop("usage_metadata")
    handle = await instance.get_handle(*args, **kwargs)
    return await handle.get(usage_metadata=fetch_tags)


def result_wrapper(wrapped, instance, args, kwargs):
    fetch_tags = kwargs.pop("usage_metadata", None)
    untracked = _untracked_job_call(instance, fetch_tags)
    result = wrapped(*args, **kwargs)
    _meter_job_once(instance, result, untracked, fetch_tags)
    return result


async def result_async_wrapper(wrapped, instance, args, kwargs):
    fetch_tags = kwargs.pop("usage_metadata", None)
    untracked = _untracked_job_call(instance, fetch_tags)
    result = await wrapped(*args, **kwargs)
    _meter_job_once(instance, result, untracked, fetch_tags)
    return result


def _untracked_job_call(handle, fetch_tags) -> FalCall:
    application = queued_jobs.application_of(handle.request_id)
    return FalCall.start(application or application_from_queue_url(handle.response_url), api_metadata=fetch_tags)


def _carries_customer_tags(usage_metadata) -> bool:
    organization, product = extract_org_and_product(usage_metadata)
    common = extract_common_metadata(usage_metadata)
    return any((organization, product, extract_subscriber_from_metadata(usage_metadata),
                common["trace_id"], common["agent"]))


def _warn_when_untagged(request_id: str, call: FalCall) -> None:
    if not call.metered or call.metering_client is None or _carries_customer_tags(call.usage_metadata):
        return
    logger.warning(
        "fal job %s (%s) has no customer tags in this process: it was submitted in another process, "
        "before a restart, or longer ago than this process remembers jobs. If this is the job's first record, "
        "it has no organization, product, subscriber, trace or agent; a repeat of a job already recorded is "
        "refused and the first record is kept. Pass usage_metadata to result() or to the handle's get() to tag it.",
        request_id, call.application,
    )


def _set_tags(usage_metadata) -> Dict[str, Any]:
    return {key: value for key, value in (usage_metadata or {}).items() if value not in (None, "", {}, [])}


def _meter_job_once(handle, result, untracked: FalCall, fetch_tags) -> None:
    call = queued_jobs.claim(handle.request_id, untracked)
    if call is None:
        return
    if call is untracked:
        _warn_when_untagged(handle.request_id, call)
    elif fetch_tags:
        call = dataclasses.replace(call, usage_metadata=overlay_metadata(fetch_tags, _set_tags(call.usage_metadata)))
    job_id = job_transaction_id(handle.request_id)
    with idempotency_key(job_id):
        dataclasses.replace(call, transaction_id=job_id).meter(result)


CLASS_WRAPPERS = (
    ("SyncClient.run", run_wrapper),
    ("AsyncClient.run", run_async_wrapper),
    ("SyncClient.stream", stream_wrapper),
    ("AsyncClient.stream", stream_async_wrapper),
    ("SyncClient.submit", submit_wrapper),
    ("AsyncClient.submit", submit_async_wrapper),
    ("SyncClient.subscribe", subscribe_wrapper),
    ("AsyncClient.subscribe", subscribe_async_wrapper),
    ("SyncClient.get_handle", get_handle_wrapper),
    ("AsyncClient.get_handle", get_handle_wrapper),
    ("SyncClient.result", client_result_wrapper),
    ("AsyncClient.result", client_result_async_wrapper),
    ("SyncRequestHandle.get", result_wrapper),
    ("AsyncRequestHandle.get", result_async_wrapper),
)

DEFAULT_CLIENT_ALIASES = (
    ("run", "sync_client", "run"),
    ("subscribe", "sync_client", "subscribe"),
    ("submit", "sync_client", "submit"),
    ("stream", "sync_client", "stream"),
    ("result", "sync_client", "result"),
    ("run_async", "async_client", "run"),
    ("subscribe_async", "async_client", "subscribe"),
    ("submit_async", "async_client", "submit"),
    ("stream_async", "async_client", "stream"),
    ("result_async", "async_client", "result"),
)


def _rebind_default_client_alias(alias: str, client_name: str, method: str) -> None:
    """Re-read ``fal_client.<alias>`` from its default client, recorded only once the rebind succeeded."""
    if not register_patch(f"fal_client.{alias}"):
        return
    try:
        setattr(fal_client, alias, getattr(getattr(fal_client, client_name), method))
    except BaseException:
        unregister_patch(f"fal_client.{alias}")
        raise


for _target, _wrapper in CLASS_WRAPPERS:
    wrap_once(f"fal_client.client.{_target}", "fal_client.client", _target, _wrapper)

# fal_client binds its module-level functions to a default client when it is
# imported, before the class methods above are wrapped, so each alias is
# re-read from its default client to pick the wrapped method up.
for _alias, _client_name, _method in DEFAULT_CLIENT_ALIASES:
    _rebind_default_client_alias(_alias, _client_name, _method)

logger.debug("REVENIUM MIDDLEWARE: fal.ai middleware loaded and wrappers registered")
