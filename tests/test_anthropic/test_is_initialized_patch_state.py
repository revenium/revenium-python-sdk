"""BACK-2673: is_initialized() must reflect whether the Anthropic patch applied.

Installing ``revenium-python-sdk anthropic`` without the ``[anthropic]`` extra
used to leave ``wrapt`` uninstalled. The middleware import then failed, one
ERROR line was logged, metering was silently inactive, and ``is_initialized()``
still returned True because it only checked that the metering API key was set.

Each scenario runs in a fresh interpreter so that blocking ``wrapt`` never
re-imports (and re-wraps) the anthropic SDK inside this test process.
"""

import json
import os
import re
import subprocess
import sys
from pathlib import Path

_PROBE = r"""
import json, sys
if {block_wrapt}:
    sys.modules["wrapt"] = None  # `import wrapt` now raises ImportError
import revenium_middleware.anthropic as ra
print(json.dumps({{
    "patch_loaded": ra.middleware is not None,
    "is_initialized": ra.is_initialized(),
    "initialize": ra.initialize(),
}}))
"""


def _run(block_wrapt: bool) -> dict:
    env = dict(os.environ)
    env["REVENIUM_METERING_API_KEY"] = "hak_test_key"
    proc = subprocess.run(
        [sys.executable, "-c", _PROBE.format(block_wrapt=block_wrapt)],
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout.strip().splitlines()[-1])


def test_is_initialized_false_when_patch_failed_to_load():
    result = _run(block_wrapt=True)
    assert result["patch_loaded"] is False
    assert result["is_initialized"] is False
    assert result["initialize"] is False


def test_is_initialized_true_when_patch_loaded():
    result = _run(block_wrapt=False)
    assert result["patch_loaded"] is True
    assert result["is_initialized"] is True
    assert result["initialize"] is True


def test_wrapt_is_a_base_dependency():
    """A plain `pip install revenium-python-sdk` must pull wrapt: every
    provider patch is applied through it."""
    # Parsed with a regex rather than tomllib (3.11+) so the check also runs
    # on the older Python versions the package supports.
    pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
    project = re.search(r"^\[project\]\n(.*?)(?=^\[)", pyproject.read_text(), re.M | re.S)
    assert project, "no [project] table in pyproject.toml"
    block = re.search(r"^dependencies\s*=\s*\[(.*?)^\]", project.group(1), re.M | re.S)
    assert block, "no dependencies list in [project]"
    deps = re.findall(r'"([^"]+)"', block.group(1))
    assert deps, "dependencies list parsed empty"
    assert any(d.replace(" ", "").lower().startswith("wrapt") for d in deps), deps
