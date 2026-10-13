"""The SDK prints its log lines itself only while the application has not configured logging.

Importing ``revenium_middleware`` with no root handler installed adds a
StreamHandler to the ``revenium_middleware`` logger, which still propagates,
so an application that configured logging afterwards printed every SDK line
twice: once through that handler and once through its own root handler.
"""
import logging
import os
import subprocess
import sys
import textwrap

import pytest

PROBE = textwrap.dedent("""
    import logging, sys

    if sys.argv[1] == "before-import":
        logging.basicConfig(stream=sys.stderr, format="APP %(message)s")
    import revenium_middleware  # noqa: F401
    if sys.argv[2] == "google":
        import revenium_middleware.google  # noqa: F401
    if sys.argv[1] == "after-import":
        logging.basicConfig(stream=sys.stderr, format="APP %(message)s")

    logging.getLogger("revenium_middleware").info("probe-info")
    logging.getLogger("revenium_middleware.extension").warning("probe-warning")
""")


def _stderr_lines(app_logging, integration):
    env = {k: v for k, v in os.environ.items() if not k.startswith("REVENIUM_")}
    completed = subprocess.run([sys.executable, "-c", PROBE, app_logging, integration], capture_output=True,
                               text=True, timeout=120, env=env)
    assert completed.returncode == 0, completed.stderr[-2000:]
    return [line for line in completed.stderr.splitlines() if "probe-" in line]


@pytest.mark.parametrize("integration", ["core", "google"])
@pytest.mark.parametrize("app_logging", ["before-import", "after-import"])
def test_an_application_that_configures_logging_sees_each_sdk_line_once(app_logging, integration):
    lines = _stderr_lines(app_logging, integration)

    assert lines == ["APP probe-info", "APP probe-warning"]


@pytest.mark.parametrize("integration", ["core", "google"])
def test_a_script_that_never_configures_logging_still_sees_the_sdk_lines(integration):
    lines = _stderr_lines("never", integration)

    assert len(lines) == 2
    assert "probe-info" in lines[0] and "probe-warning" in lines[1]


def _run_probe(source, *args, **revenium_env):
    env = {k: v for k, v in os.environ.items() if not k.startswith("REVENIUM_")}
    env.update(revenium_env)
    return subprocess.run([sys.executable, "-c", textwrap.dedent(source), *args], capture_output=True,
                          text=True, timeout=120, env=env)


GOOGLE_LEVEL_PROBE = """
    import logging, sys
    import revenium_middleware  # noqa: F401
    if sys.argv[1:] == ["set-in-code"]:
        logging.getLogger("revenium_middleware").setLevel(logging.ERROR)
    import revenium_middleware.google  # noqa: F401
    print(logging.getLevelName(logging.getLogger("revenium_middleware").level))
"""


@pytest.mark.parametrize("revenium_env, set_in_code, expected", [
    ({"REVENIUM_LOG_LEVEL": "WARN"}, False, "WARNING"),
    ({}, True, "ERROR"),
], ids=["environment-WARN", "level-set-in-code"])
def test_importing_google_keeps_the_package_log_level(revenium_env, set_in_code, expected):
    """Importing the Google integration leaves the package log level as the
    environment or the application set it (BACK-4452 N17)."""
    args = ["set-in-code"] if set_in_code else []
    completed = _run_probe(GOOGLE_LEVEL_PROBE, *args, **revenium_env)

    assert completed.returncode == 0, completed.stderr[-2000:]
    assert completed.stdout.strip() == expected


def test_an_unknown_log_level_value_does_not_break_import():
    """A REVENIUM_LOG_LEVEL that is not a level name falls back to INFO
    instead of stopping the import (BACK-4452 N17)."""
    completed = _run_probe("""
        import logging
        import revenium_middleware  # noqa: F401
        print(logging.getLevelName(logging.getLogger("revenium_middleware").level))
    """, REVENIUM_LOG_LEVEL="basic_format")

    assert completed.returncode == 0, completed.stderr[-2000:]
    assert completed.stdout.strip() == "INFO"


REVENIUM_DEBUG_PROBE = """
    import logging, sys
    import revenium_middleware.anthropic  # noqa: F401
    if sys.argv[1:] == ["app-sets-warning"]:
        logging.getLogger("revenium_middleware").setLevel(logging.WARNING)
    extension = logging.getLogger("revenium_middleware.extension")
    extension.debug("probe-debug")
    print(len(extension.handlers), extension.level)
"""


def test_revenium_debug_prints_each_debug_line_once():
    """REVENIUM_DEBUG=true prints each debug line once, through the package
    handler, without a handler or level of the integration's own (BACK-4452 N15)."""
    completed = _run_probe(REVENIUM_DEBUG_PROBE, REVENIUM_DEBUG="true")

    assert completed.returncode == 0, completed.stderr[-2000:]
    assert len([line for line in completed.stderr.splitlines() if "probe-debug" in line]) == 1
    assert completed.stdout.split() == ["0", str(logging.NOTSET)]


def test_revenium_debug_follows_a_level_set_in_code():
    """An application that quiets the package after import is obeyed with
    REVENIUM_DEBUG=true set (BACK-4452 N15)."""
    completed = _run_probe(REVENIUM_DEBUG_PROBE, "app-sets-warning", REVENIUM_DEBUG="true")

    assert completed.returncode == 0, completed.stderr[-2000:]
    assert [line for line in completed.stderr.splitlines() if "probe-debug" in line] == []
