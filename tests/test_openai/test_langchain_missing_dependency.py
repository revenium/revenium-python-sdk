"""A LangChain install that cannot meter says so, naming the package and the
install command (BACK-4011).

Runs in a subprocess because wrapt is already imported in the test process.
"""
import gc
import subprocess
import sys
import textwrap
import weakref

INSTALL_COMMAND = 'pip install "revenium-python-sdk[langchain]"'


def _build_handlers_without(package):
    script = textwrap.dedent(f"""
        import logging
        import sys
        sys.modules[{package!r}] = None
        logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s %(message)s")
        from revenium_middleware.openai.langchain.unified_handler import UnifiedReveniumCallbackHandler
        UnifiedReveniumCallbackHandler()
        UnifiedReveniumCallbackHandler()
    """)
    finished = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=120)
    assert finished.returncode == 0, finished.stderr
    return (finished.stdout + finished.stderr).splitlines()


def test_a_missing_package_is_named_with_the_install_command():
    lines = _build_handlers_without("wrapt")
    named = [line for line in lines if "ERROR" in line and "wrapt" in line and INSTALL_COMMAND in line]
    assert len(named) == 1, "\n".join(lines)


class _CallerState:
    pass


def _first_langchain_check_from_a_caller():
    from revenium_middleware.openai.langchain import _utils

    state = _CallerState()
    _utils._check_langchain_cached()
    return weakref.ref(state)


def test_a_missing_langchain_does_not_keep_the_first_callers_frames(monkeypatch):
    """BACK-4336: the cached ImportError must not pin the first caller's frames
    (and the stream it was metering) for the life of the process."""
    from revenium_middleware.openai.langchain import _utils

    monkeypatch.setattr(_utils, "_langchain_check_cache", None)
    monkeypatch.setitem(sys.modules, "langchain", None)

    caller_state = _first_langchain_check_from_a_caller()
    gc.collect()

    assert _utils._langchain_check_cache[0] is False
    assert caller_state() is None
