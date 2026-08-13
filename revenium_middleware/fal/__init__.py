"""
Revenium Middleware for fal.ai Python Client.

When you install and import this library, it will automatically hook
fal_client.run, fal_client.subscribe, and fal_client.stream using wrapt,
and meter usage to Revenium for image, video, and audio generation tracking.
"""

import logging

logger = logging.getLogger(__name__)

# Conditionally import middleware (requires fal-client SDK)
try:
    import fal_client as _fal_client  # noqa: F401
    from .middleware import (
        run_wrapper,
        subscribe_wrapper,
        stream_wrapper,
        run_async_wrapper,
        subscribe_async_wrapper,
        stream_async_wrapper,
    )
except ImportError as e:
    from revenium_middleware._core.load_diagnostics import log_middleware_load_failure
    log_middleware_load_failure("fal", e, required_packages=("fal_client",))
    run_wrapper = None  # type: ignore
    subscribe_wrapper = None  # type: ignore
    stream_wrapper = None  # type: ignore
    run_async_wrapper = None  # type: ignore
    subscribe_async_wrapper = None  # type: ignore
    stream_async_wrapper = None  # type: ignore

from .trace_fields import detect_media_type, normalize_model_name

__all__ = [
    "run_wrapper",
    "subscribe_wrapper",
    "stream_wrapper",
    "run_async_wrapper",
    "subscribe_async_wrapper",
    "stream_async_wrapper",
    "detect_media_type",
    "normalize_model_name",
]
