"""Log-level policy for provider middleware load failures.

Every provider package guards its middleware import with ``except
ImportError`` so that optional providers can be absent. That guard used to
log at DEBUG unconditionally, which hid real breakage: a customer with the
provider SDK installed whose middleware failed to import (missing ``wrapt``,
a broken internal import, a version conflict) got zero metering with no
visible signal.

The policy here distinguishes the two cases:

- Provider SDK **installed** but middleware import failed → ERROR, including
  the underlying exception. Metering was expected to work and silently won't.
- Provider SDK **absent** → DEBUG. The provider simply isn't in use (e.g.
  ``revenium_middleware.griptape`` imports several provider middlewares
  eagerly and most installs only have one provider SDK).
"""
from __future__ import annotations

import importlib.util
import logging
from typing import Tuple

logger = logging.getLogger("revenium_middleware")


def _package_installed(package: str) -> bool:
    try:
        return importlib.util.find_spec(package) is not None
    except ModuleNotFoundError:
        # find_spec("google.genai") raises when the parent package is absent.
        return False
    except Exception:
        # Any other spec-lookup failure means the package exists but is
        # broken — fail toward visibility.
        return True


def log_middleware_load_failure(
    provider_label: str,
    exc: BaseException,
    required_packages: Tuple[str, ...],
) -> None:
    """Log a provider middleware import failure at the appropriate level.

    Args:
        provider_label: Human-readable provider name for the log message.
        exc: The ImportError (or subclass) that aborted the middleware load.
        required_packages: Import names of the provider's own SDK package(s).
            If any of them is installed, the failure is unexpected and logged
            at ERROR; if none are, the provider is not in use and the failure
            is logged at DEBUG.
    """
    if any(_package_installed(pkg) for pkg in required_packages):
        logger.error(
            "Revenium %s middleware could not load: %s. "
            "Metering will NOT be active for %s calls.",
            provider_label,
            exc,
            provider_label,
        )
    else:
        logger.debug(
            "%s SDK not installed (%s); Revenium %s middleware not loaded",
            provider_label,
            exc,
            provider_label,
        )
