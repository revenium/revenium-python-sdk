"""The SDK prints its log lines itself only while the application has not configured logging.

Importing ``revenium_middleware`` with no root handler installed adds a
StreamHandler to the ``revenium_middleware`` logger, which still propagates,
so an application that configured logging afterwards printed every SDK line
twice: once through that handler and once through its own root handler.
"""
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
