"""Whether a dotted target is wrapped by Revenium in the running program, on any wrapt version.

``isinstance(x, wrapt.ObjectProxy)`` is False for a live wrap on wrapt 2.x, so
the wrapper is found by walking the layers through ``__wrapped__`` and reading
each wrapper function's module. The CI "Extras resolve in a clean env" step
inlines the same logic, since it runs where the test tree is not importable.
"""
import importlib

OUR_PACKAGE = "revenium_middleware"


def live_attributes(path):
    """The attribute ``path`` names as the program reads it, and as its owner stores it."""
    parts = path.split(".")
    for split in range(len(parts) - 1, 0, -1):
        try:
            owner = importlib.import_module(".".join(parts[:split]))
        except ImportError:
            continue
        for name in parts[split:-1]:
            owner = getattr(owner, name)
        return getattr(owner, parts[-1]), vars(owner).get(parts[-1])
    raise LookupError(f"{path}: no importable module prefix")


# Deeper than any real stack of instrumentation libraries on one method.
MAX_LAYERS = 32


def wrapped_by_revenium(path):
    for layer in live_attributes(path):
        for _ in range(MAX_LAYERS):
            if layer is None:
                break
            wrapper = getattr(layer, "_self_wrapper", None)
            if (getattr(wrapper, "__module__", None) or "").startswith(OUR_PACKAGE):
                return True
            layer = getattr(layer, "__wrapped__", None)
    return False
