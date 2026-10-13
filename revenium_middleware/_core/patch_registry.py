import threading
import logging

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_patched = set()


def register_patch(function_path: str) -> bool:
    with _lock:
        if function_path in _patched:
            logger.debug("Skipping already-patched function: %s", function_path)
            return False
        _patched.add(function_path)
        return True


def is_patched(function_path: str) -> bool:
    with _lock:
        return function_path in _patched


def unregister_patch(function_path: str) -> None:
    with _lock:
        _patched.discard(function_path)


def wrap_once(function_path: str, module, attribute: str, wrapper) -> bool:
    """Wrap ``module.attribute`` with ``wrapper`` and record it as ``function_path``, once per process.

    ``module`` is the module or its import path, as ``wrapt.wrap_function_wrapper`` takes it.

    Returns False when it is already recorded. A wrap that raises is
    unrecorded before the exception propagates, so the registry never reports
    a method that was not wrapped.
    """
    if not register_patch(function_path):
        return False
    try:
        # Imported here: modules that only read the registry must not need wrapt.
        import wrapt
        wrapt.wrap_function_wrapper(module, attribute, wrapper)
    except BaseException:
        unregister_patch(function_path)
        raise
    return True
