"""Apply a registered wrap so the patch registry reports only wraps that took effect."""
import logging

import wrapt

from revenium_middleware._core.patch_registry import register_patch, unregister_patch

logger = logging.getLogger("revenium_middleware.perplexity")


def wrap_registered(function_path: str, module: str, attribute: str, wrapper) -> bool:
    if not register_patch(function_path):
        return False
    try:
        wrapt.wrap_function_wrapper(module, attribute, wrapper)
    except ImportError as e:
        unregister_patch(function_path)
        logger.debug("%s is not installed, so %s is not wrapped: %s", module, function_path, e)
        return False
    except AttributeError as e:
        unregister_patch(function_path)
        logger.warning(
            "Revenium could not wrap %s, so calls through it will NOT be metered: %s",
            function_path, e,
        )
        return False
    return True
