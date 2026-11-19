"""One shared HTTP client for delivering tool events, live and on buffer replay.

A client per event paid for a new connection pool, TCP connection, TLS
handshake and SSL context every time, which costs whole CPU cores at a few
dozen tool calls a second. The shared client is a sync ``httpx.Client``:
pool workers each run their own event loop, and async connections are bound
to the loop that opened them.

The client is never closed at exit, because the exit drain still sends
queued tool events after ``atexit`` has run.

Events of every customer key share the client's connections, which carry no
state of their own: each request names its key in its own header. A cookie
jar would carry state, so the client keeps none, and a Set-Cookie a gateway
answers one key's event with never rides on another key's.
"""

import os
import threading
from http.cookiejar import CookieJar, DefaultCookiePolicy
from typing import Any, Dict, Optional, Union

import httpx

_client: Optional[httpx.Client] = None
_lock = threading.Lock()


def _jar_that_keeps_no_cookies() -> CookieJar:
    return CookieJar(policy=DefaultCookiePolicy(allowed_domains=[]))


def _shared_client() -> httpx.Client:
    global _client
    client = _client
    if client is not None:
        return client
    with _lock:
        if _client is None:
            # Imported here: the _metering package imports the tool decorator,
            # which sends through this module.
            from revenium_middleware._metering._constants import DEFAULT_CONNECTION_LIMITS

            _client = httpx.Client(limits=DEFAULT_CONNECTION_LIMITS, cookies=_jar_that_keeps_no_cookies())
        return _client


def post_tool_event(tool_event: Dict[str, Any], timeout: Union[float, httpx.Timeout]) -> None:
    """POST one tool event to the endpoint, and under the key, it was dispatched with."""
    event_payload = tool_event["event_payload"]
    response = _shared_client().post(
        tool_event["url"],
        headers={
            "x-api-key": tool_event["key"],
            "Content-Type": "application/json",
            # Dedupes redelivery (e.g. buffer replay after a network drop the
            # backend actually accepted) when supported; harmless if the
            # backend ignores it.
            "Idempotency-Key": tool_event.get("idempotency_key") or event_payload["transactionId"],
        },
        json=event_payload,
        timeout=timeout,
    )
    response.raise_for_status()


def _forget_parent_client() -> None:
    """Give a forked child its own client and a free lock.

    The parent's sockets must not be shared with the child, and a lock held
    by another parent thread at fork time would stay held in the child.
    """
    global _client, _lock
    _client = None
    _lock = threading.Lock()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_forget_parent_client)
