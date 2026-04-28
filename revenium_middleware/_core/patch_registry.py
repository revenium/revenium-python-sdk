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
