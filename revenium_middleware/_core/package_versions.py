"""Installed versions of the libraries whose usage arithmetic the SDK follows."""
import functools
import importlib.metadata
import re
from typing import Optional, Tuple

_LEADING_RELEASE = re.compile(r"(\d+)\.(\d+)\.(\d+)")


@functools.lru_cache(maxsize=None)
def installed_version(distribution: str) -> Optional[str]:
    """The installed version string of ``distribution``, or None when it is not installed."""
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def release_of(version: Optional[str]) -> Optional[Tuple[int, int, int]]:
    """The leading major, minor and patch numbers of a version string, or None when it has none."""
    match = _LEADING_RELEASE.match(version or "")
    return tuple(int(part) for part in match.groups()) if match else None
