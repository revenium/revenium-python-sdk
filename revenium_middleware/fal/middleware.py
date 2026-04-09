"""
Wrapt-based wrappers for fal_client API methods.

This module registers wrapt patches on fal_client. It requires fal_client
to be installed — the __init__.py handles graceful fallback when it is not.
"""

import logging
import datetime
import wrapt
from typing import Any, Dict, Iterator, AsyncIterator

from revenium_middleware import merge_metadata
from ._metering import generate_transaction_id, handle_metering

logger = logging.getLogger("revenium_middleware.fal")


# =============================================================================
# Sync Wrappers
# =============================================================================


@wrapt.patch_function_wrapper("fal_client", "run")
def run_wrapper(wrapped, instance, args, kwargs):
    """Wrap fal_client.run to meter API calls."""
    logger.debug("fal_client.run wrapper called")

    api_metadata = kwargs.pop("usage_metadata", {}) if "usage_metadata" in kwargs else {}
    usage_metadata = merge_metadata(api_metadata)

    request_time_dt = datetime.datetime.now(datetime.timezone.utc)
    transaction_id = generate_transaction_id()
    application = args[0] if args else kwargs.get("application", "unknown")

    result = wrapped(*args, **kwargs)

    handle_metering(
        application=application,
        arguments=kwargs.get("arguments", {}),
        result=result,
        request_time_dt=request_time_dt,
        usage_metadata=usage_metadata,
        transaction_id=transaction_id,
        is_streamed=False,
    )

    if isinstance(result, dict):
        result["_revenium_transaction_id"] = transaction_id

    return result


@wrapt.patch_function_wrapper("fal_client", "subscribe")
def subscribe_wrapper(wrapped, instance, args, kwargs):
    """Wrap fal_client.subscribe to meter API calls."""
    logger.debug("fal_client.subscribe wrapper called")

    api_metadata = kwargs.pop("usage_metadata", {}) if "usage_metadata" in kwargs else {}
    usage_metadata = merge_metadata(api_metadata)

    request_time_dt = datetime.datetime.now(datetime.timezone.utc)
    transaction_id = generate_transaction_id()
    application = args[0] if args else kwargs.get("application", "unknown")

    result = wrapped(*args, **kwargs)

    handle_metering(
        application=application,
        arguments=kwargs.get("arguments", {}),
        result=result,
        request_time_dt=request_time_dt,
        usage_metadata=usage_metadata,
        transaction_id=transaction_id,
        is_streamed=False,
    )

    if isinstance(result, dict):
        result["_revenium_transaction_id"] = transaction_id

    return result


@wrapt.patch_function_wrapper("fal_client", "stream")
def stream_wrapper(wrapped, instance, args, kwargs):
    """Wrap fal_client.stream to meter streaming API calls."""
    logger.debug("fal_client.stream wrapper called")

    api_metadata = kwargs.pop("usage_metadata", {}) if "usage_metadata" in kwargs else {}
    usage_metadata = merge_metadata(api_metadata)

    request_time_dt = datetime.datetime.now(datetime.timezone.utc)
    transaction_id = generate_transaction_id()
    application = args[0] if args else kwargs.get("application", "unknown")

    stream = wrapped(*args, **kwargs)

    def wrapped_generator() -> Iterator[Dict[str, Any]]:
        events = []
        final_result = None

        for event in stream:
            events.append(event)
            if isinstance(event, dict):
                event["_revenium_transaction_id"] = transaction_id
            yield event

        if events:
            final_result = events[-1]

        handle_metering(
            application=application,
            arguments=kwargs.get("arguments", {}),
            result=final_result or {},
            request_time_dt=request_time_dt,
            usage_metadata=usage_metadata,
            transaction_id=transaction_id,
            is_streamed=True,
        )

    return wrapped_generator()


# =============================================================================
# Async Wrappers
# =============================================================================


@wrapt.patch_function_wrapper("fal_client", "run_async")
async def run_async_wrapper(wrapped, instance, args, kwargs):
    """Wrap fal_client.run_async to meter API calls."""
    logger.debug("fal_client.run_async wrapper called")

    api_metadata = kwargs.pop("usage_metadata", {}) if "usage_metadata" in kwargs else {}
    usage_metadata = merge_metadata(api_metadata)

    request_time_dt = datetime.datetime.now(datetime.timezone.utc)
    transaction_id = generate_transaction_id()
    application = args[0] if args else kwargs.get("application", "unknown")

    result = await wrapped(*args, **kwargs)

    handle_metering(
        application=application,
        arguments=kwargs.get("arguments", {}),
        result=result,
        request_time_dt=request_time_dt,
        usage_metadata=usage_metadata,
        transaction_id=transaction_id,
        is_streamed=False,
    )

    if isinstance(result, dict):
        result["_revenium_transaction_id"] = transaction_id

    return result


@wrapt.patch_function_wrapper("fal_client", "subscribe_async")
async def subscribe_async_wrapper(wrapped, instance, args, kwargs):
    """Wrap fal_client.subscribe_async to meter API calls."""
    logger.debug("fal_client.subscribe_async wrapper called")

    api_metadata = kwargs.pop("usage_metadata", {}) if "usage_metadata" in kwargs else {}
    usage_metadata = merge_metadata(api_metadata)

    request_time_dt = datetime.datetime.now(datetime.timezone.utc)
    transaction_id = generate_transaction_id()
    application = args[0] if args else kwargs.get("application", "unknown")

    result = await wrapped(*args, **kwargs)

    handle_metering(
        application=application,
        arguments=kwargs.get("arguments", {}),
        result=result,
        request_time_dt=request_time_dt,
        usage_metadata=usage_metadata,
        transaction_id=transaction_id,
        is_streamed=False,
    )

    if isinstance(result, dict):
        result["_revenium_transaction_id"] = transaction_id

    return result


@wrapt.patch_function_wrapper("fal_client", "stream_async")
async def stream_async_wrapper(wrapped, instance, args, kwargs):
    """Wrap fal_client.stream_async to meter streaming API calls."""
    logger.debug("fal_client.stream_async wrapper called")

    api_metadata = kwargs.pop("usage_metadata", {}) if "usage_metadata" in kwargs else {}
    usage_metadata = merge_metadata(api_metadata)

    request_time_dt = datetime.datetime.now(datetime.timezone.utc)
    transaction_id = generate_transaction_id()
    application = args[0] if args else kwargs.get("application", "unknown")

    stream = wrapped(*args, **kwargs)

    async def wrapped_async_generator() -> AsyncIterator[Dict[str, Any]]:
        events = []
        final_result = None

        async for event in stream:
            events.append(event)
            if isinstance(event, dict):
                event["_revenium_transaction_id"] = transaction_id
            yield event

        if events:
            final_result = events[-1]

        handle_metering(
            application=application,
            arguments=kwargs.get("arguments", {}),
            result=final_result or {},
            request_time_dt=request_time_dt,
            usage_metadata=usage_metadata,
            transaction_id=transaction_id,
            is_streamed=True,
        )

    return wrapped_async_generator()


logger.debug("REVENIUM MIDDLEWARE: fal.ai middleware loaded and wrappers registered")
