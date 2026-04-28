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
from revenium_middleware._core.config import is_selective_metering_enabled
from revenium_middleware._core.context import is_inside_decorated_function
from revenium_middleware._core.patch_registry import register_patch
from ._metering import generate_transaction_id, handle_metering

logger = logging.getLogger("revenium_middleware.fal")


# =============================================================================
# Sync Wrappers
# =============================================================================


if register_patch("fal_client.run"):
    @wrapt.patch_function_wrapper("fal_client", "run")
    def run_wrapper(wrapped, instance, args, kwargs):
        if is_selective_metering_enabled() and not is_inside_decorated_function():
            return wrapped(*args, **kwargs)

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

        return result


if register_patch("fal_client.subscribe"):
    @wrapt.patch_function_wrapper("fal_client", "subscribe")
    def subscribe_wrapper(wrapped, instance, args, kwargs):
        if is_selective_metering_enabled() and not is_inside_decorated_function():
            return wrapped(*args, **kwargs)

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

        return result


if register_patch("fal_client.stream"):
    @wrapt.patch_function_wrapper("fal_client", "stream")
    def stream_wrapper(wrapped, instance, args, kwargs):
        if is_selective_metering_enabled() and not is_inside_decorated_function():
            return wrapped(*args, **kwargs)

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


if register_patch("fal_client.run_async"):
    @wrapt.patch_function_wrapper("fal_client", "run_async")
    async def run_async_wrapper(wrapped, instance, args, kwargs):
        if is_selective_metering_enabled() and not is_inside_decorated_function():
            return await wrapped(*args, **kwargs)

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

        return result


if register_patch("fal_client.subscribe_async"):
    @wrapt.patch_function_wrapper("fal_client", "subscribe_async")
    async def subscribe_async_wrapper(wrapped, instance, args, kwargs):
        if is_selective_metering_enabled() and not is_inside_decorated_function():
            return await wrapped(*args, **kwargs)

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

        return result


if register_patch("fal_client.stream_async"):
    @wrapt.patch_function_wrapper("fal_client", "stream_async")
    async def stream_async_wrapper(wrapped, instance, args, kwargs):
        if is_selective_metering_enabled() and not is_inside_decorated_function():
            return wrapped(*args, **kwargs)

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
