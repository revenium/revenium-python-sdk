"""Apply a registered wrap so the patch registry reports only wraps that took effect."""
import logging

from revenium_middleware._core.patch_registry import wrap_once

logger = logging.getLogger("revenium_middleware.perplexity")


def wrap_registered(function_path: str, module: str, attribute: str, wrapper) -> bool:
    try:
        return wrap_once(function_path, module, attribute, wrapper)
    except ImportError as e:
        logger.debug("%s is not installed, so %s is not wrapped: %s", module, function_path, e)
        return False
    except AttributeError as e:
        logger.warning(
            "Revenium could not wrap %s, so calls through it will NOT be metered: %s",
            function_path, e,
        )
        return False


def wrap_strip_only(function_path: str, module: str, attribute: str, wrapper) -> bool:
    """Apply a wrap that only removes Revenium metadata from a call the SDK never meters,
    so a release without the class loses nothing worth a warning."""
    try:
        return wrap_once(function_path, module, attribute, wrapper)
    except (ImportError, AttributeError) as e:
        logger.debug("%s is not wrapped: %s", function_path, e)
        return False
