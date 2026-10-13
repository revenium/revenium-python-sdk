"""The ``usage_metadata`` a caller gives in LangChain's per-call settings.

LangChain hands ``config={"metadata": {"usage_metadata": ...}}`` down the
call stack of the model call it makes, so a provider wrap finds it in a
calling frame. Shared by every wrap that resolves the caller for a model call
LangChain can make, so they agree on who the caller is.
"""
import functools
import importlib.util
import inspect
import logging
from typing import Any, Dict

from revenium_middleware._core.log_sanitize import sanitize_for_logging

logger = logging.getLogger("revenium_middleware.extension")


def _usage_metadata_in(frame_locals) -> Any:
    config = frame_locals.get('config')
    if isinstance(config, dict) and isinstance(config.get('metadata'), dict) \
            and 'usage_metadata' in config['metadata']:
        return config['metadata']['usage_metadata']
    metadata = frame_locals.get('metadata')
    if isinstance(metadata, dict) and 'usage_metadata' in metadata:
        return metadata['usage_metadata']
    cb_manager_metadata = getattr(frame_locals.get('callback_manager'), 'metadata', None)
    if isinstance(cb_manager_metadata, dict) and 'usage_metadata' in cb_manager_metadata:
        return cb_manager_metadata['usage_metadata']
    return None


@functools.lru_cache(maxsize=1)
def _langchain_installed() -> bool:
    return importlib.util.find_spec("langchain_core") is not None


def langchain_usage_metadata() -> Dict[str, Any]:
    """The nearest calling frame's LangChain ``usage_metadata``, or ``{}``
    without walking the stack when LangChain is not installed."""
    if not _langchain_installed():
        return {}
    try:
        frame = inspect.currentframe()
        while frame:
            usage_metadata = _usage_metadata_in(frame.f_locals)
            if usage_metadata is not None:
                logger.debug("Found usage_metadata in LangChain context: %s", sanitize_for_logging(usage_metadata))
                return usage_metadata
            frame = frame.f_back
        logger.debug("No usage_metadata found in LangChain context")
        return {}
    except Exception as e:
        logger.debug(f"Error extracting LangChain usage_metadata: {e}")
        return {}
