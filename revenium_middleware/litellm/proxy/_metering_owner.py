"""Which LiteLLM proxy integration owns metering for this process.

A proxy that enables both the deprecated ``MiddlewareHandler`` callback and
``ReveniumGuardrail`` would meter every call twice: the callback's
``async_log_success_event`` and the guardrail's ``async_post_call_success_hook``
both fire for the same request and both submit a completion row.

The client wrapper (``revenium_middleware.litellm.client``) is the other
double: importing the guardrail loads it, it patches ``litellm.acompletion``
process-wide, and the proxy router calls that for every request. It asks
``guardrail_meters_request`` per call rather than ``guardrail_owns_metering``
per process, because the same process can also call LiteLLM outside any proxied
request, and some proxied routes the guardrail never meters (see
``ReveniumGuardrail.meters_request``).

The guardrail claims ownership at construction time, but only when its
configuration guarantees it runs on *every* proxied request -- ``default_on``
is true and its event hooks include ``post_call``. Under any other
configuration the guardrail is applied per request (or not at all for
post-call), so suppressing the callback could drop metering entirely, and the
callback keeps metering while its deprecation warning names the hazard.

Deliberately dependency-free and Python 3.8-compatible: ``middleware.py``
imports it on every supported interpreter, while ``guardrail.py`` is gated to
3.10+ (see its import guard).
"""

import logging
import threading

logger = logging.getLogger("revenium_middleware.extension")

_lock = threading.Lock()
_owner = None


def register_metering_guardrail(guardrail):
    """Record that ``guardrail`` meters every proxied request.

    Args:
        guardrail: The ``ReveniumGuardrail`` instance claiming ownership.
    """
    global _owner
    with _lock:
        _owner = guardrail


def reset_metering_owner():
    """Forget the registered owner. For tests and re-configuration only."""
    global _owner
    with _lock:
        _owner = None


def guardrail_owns_metering():
    """True when a guardrail is registered to meter every proxied request."""
    with _lock:
        return _owner is not None


def guardrail_meters_request(request_kwargs):
    """True when the registered guardrail meters the call ``request_kwargs`` makes.

    Args:
        request_kwargs: The keyword arguments a patched LiteLLM entry point
            (``litellm.acompletion`` and its siblings) was called with.

    Returns:
        False when no guardrail claimed ownership, so a process without one
        keeps metering every direct call through the client wrapper.
    """
    with _lock:
        owner = _owner
    return owner is not None and owner.meters_request(request_kwargs)
