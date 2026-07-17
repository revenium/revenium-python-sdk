"""README import examples must reference the bundled dotted namespace.

The retired standalone packages exposed underscore module names such as
``revenium_middleware_openai``. Those modules do not ship with this bundle,
so any README snippet documenting them raises ModuleNotFoundError for users
who followed the README's own install instructions.
"""
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
README_LINES = (REPO_ROOT / "README.md").read_text(encoding="utf-8").splitlines()


def test_readme_never_references_standalone_underscore_modules():
    hits = [
        f"line {number}: {line.strip()}"
        for number, line in enumerate(README_LINES, start=1)
        if re.search(r"\brevenium_middleware_[a-z]", line)
    ]
    assert hits == [], (
        "README references standalone-package module names that the bundle "
        "does not install:\n" + "\n".join(hits)
    )


def test_readme_never_documents_standalone_pip_installs():
    hits = [
        f"line {number}: {line.strip()}"
        for number, line in enumerate(README_LINES, start=1)
        if "pip install" in line and "revenium-middleware-" in line
    ]
    assert hits == [], (
        "README documents standalone pip packages instead of this bundle:\n"
        + "\n".join(hits)
    )


def test_readme_import_statements_resolve_inside_the_bundle():
    imported = {
        match.group(1)
        for line in README_LINES
        for match in [re.match(r"\s*(?:import|from)\s+(revenium_middleware(?:\.\w+)*)", line)]
        if match
    }
    missing = []
    for module in sorted(imported):
        path = REPO_ROOT.joinpath(*module.split("."))
        if not (path.is_dir() or path.with_suffix(".py").is_file()):
            missing.append(module)
    assert missing == [], (
        f"README imports modules that do not exist in the package: {missing}"
    )
