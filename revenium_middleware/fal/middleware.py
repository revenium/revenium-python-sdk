"""
Wrapt-based wrappers for fal_client API methods.

This module registers wrapt patches on fal_client. It requires fal_client
to be installed — the __init__.py handles graceful fallback when it is not.
"""

import logging
from typing import Any, AsyncIterator, Dict, Iterator

import fal_client
import wrapt

from revenium_middleware._core.patch_registry import register_patch
from ._call import FalCall
from ._queue import application_from_queue_url, queued_jobs

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
    return _metered_events(wrapped(*args, **kwargs), call)


def stream_async_wrapper(wrapped, instance, args, kwargs):
    call = FalCall.from_request(args, kwargs)
    return _metered_async_events(wrapped(*args, **kwargs), call)


def _metered_events(events: Iterator[Dict[str, Any]], call: FalCall) -> Iterator[Dict[str, Any]]:
    last_event = None
    for event in events:
        last_event = event
        yield event
    call.meter(last_event or {}, is_streamed=True)


async def _metered_async_events(events: AsyncIterator[Dict[str, Any]], call: FalCall) -> AsyncIterator[Dict[str, Any]]:
    last_event = None
    async for event in events:
        last_event = event
        yield event
    call.meter(last_event or {}, is_streamed=True)


def submit_wrapper(wrapped, instance, args, kwargs):
    call = FalCall.from_request(args, kwargs)
    handle = wrapped(*args, **kwargs)
    queued_jobs.track(handle.request_id, call)
    return handle


async def submit_async_wrapper(wrapped, instance, args, kwargs):
    call = FalCall.from_request(args, kwargs)
    handle = await wrapped(*args, **kwargs)
    queued_jobs.track(handle.request_id, call)
    return handle


def subscribe_wrapper(wrapped, instance, args, kwargs):
    call = FalCall.from_request(args, kwargs)
    # With client_timeout, SyncClient.subscribe runs submit on fal's executor
    # thread, where the caller's context variables (injected metadata,
    # @revenium_meter scope) are not visible. on_enqueue is how fal hands the
    # request id back, so the caller-side context is attached to the job there.
    kwargs["on_enqueue"] = _tracking_on_enqueue(call, kwargs.get("on_enqueue"))
    return wrapped(*args, **kwargs)


subscribe_async_wrapper = subscribe_wrapper


def _tracking_on_enqueue(call: FalCall, on_enqueue):
    def track_then_notify(request_id):
        queued_jobs.track(request_id, call)
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


def result_wrapper(wrapped, instance, args, kwargs):
    untracked = _untracked_job_call(instance)
    result = wrapped(*args, **kwargs)
    _meter_job_once(instance, result, untracked)
    return result


async def result_async_wrapper(wrapped, instance, args, kwargs):
    untracked = _untracked_job_call(instance)
    result = await wrapped(*args, **kwargs)
    _meter_job_once(instance, result, untracked)
    return result


def _untracked_job_call(handle) -> FalCall:
    application = queued_jobs.application_of(handle.request_id)
    return FalCall.start(application or application_from_queue_url(handle.response_url))


def _meter_job_once(handle, result, untracked: FalCall) -> None:
    call = queued_jobs.claim(handle.request_id, untracked)
    if call is not None:
        call.meter(result)


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
    ("SyncRequestHandle.get", result_wrapper),
    ("AsyncRequestHandle.get", result_async_wrapper),
)

DEFAULT_CLIENT_ALIASES = (
    ("run", "sync_client", "run"),
    ("subscribe", "sync_client", "subscribe"),
    ("submit", "sync_client", "submit"),
    ("stream", "sync_client", "stream"),
    ("run_async", "async_client", "run"),
    ("subscribe_async", "async_client", "subscribe"),
    ("submit_async", "async_client", "submit"),
    ("stream_async", "async_client", "stream"),
)

for _target, _wrapper in CLASS_WRAPPERS:
    if register_patch(f"fal_client.client.{_target}"):
        wrapt.wrap_function_wrapper("fal_client.client", _target, _wrapper)

# fal_client binds its module-level functions to a default client when it is
# imported, before the class methods above are wrapped, so each alias is
# re-read from its default client to pick the wrapped method up.
for _alias, _client_name, _method in DEFAULT_CLIENT_ALIASES:
    if register_patch(f"fal_client.{_alias}"):
        setattr(fal_client, _alias, getattr(getattr(fal_client, _client_name), _method))

logger.debug("REVENIUM MIDDLEWARE: fal.ai middleware loaded and wrappers registered")
