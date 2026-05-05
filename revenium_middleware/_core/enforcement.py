"""
Enforcement engine for the Revenium circuit breaker.

Polls cost-limit rules from the Revenium API in a daemon thread and caches
them in memory. ``check_enforcement(...)`` is a pre-call hook that raises
``ReveniumCostLimitExceeded`` when a tripped rule matches the current
request, blocking the outbound provider call before any spend occurs.

Opt-in via ``REVENIUM_CIRCUIT_BREAKER_ENABLED``. Disabled by default so the
SDK stays no-op for callers who haven't enrolled in cost controls.
"""

import json
import logging
import os
import threading
import time
from typing import List, Optional
from urllib.parse import urlparse

import httpx

from .config import Config
from .exceptions import ReveniumCostLimitExceeded

logger = logging.getLogger("revenium_middleware.extension")

_DEFAULT_POLL_INTERVAL = 60      # seconds between background refreshes
_CACHE_TTL = 120                  # seconds before a cached rule is considered stale
_RULES_CACHE_FILENAME = "revenium_enforcement_rules.json"

_cached_rules: List[dict] = []
_cache_lock = threading.Lock()
_cache_timestamp = 0.0
# True once any successful fetch (even an empty list / HTTP 204) or a disk
# snapshot load has populated the cache. Distinguishes "server says no rules
# apply" from "we have never heard back from the server" — fail-closed must
# only block in the latter case.
_cache_initialized = False

_poll_thread: Optional[threading.Thread] = None
_poll_lock = threading.Lock()
_stop_event = threading.Event()

# Serializes synchronous stale-cache refreshes to prevent thundering herd
_refresh_lock = threading.Lock()

# Single-shot warnings so misconfigured environments don't spam logs
_team_id_warned = False
_disk_load_attempted = False


def _env_truthy(name: str) -> bool:
    return os.environ.get(name, "").lower() in ("1", "true", "yes", "on")


def is_circuit_breaker_enabled() -> bool:
    """Return True when the operator has opted in to enforcement."""
    return _env_truthy(Config.ENV_CIRCUIT_BREAKER_ENABLED)


def is_bypass_enabled() -> bool:
    """``REVENIUM_BYPASS=true`` short-circuits enforcement at every callsite."""
    return _env_truthy(Config.ENV_REVENIUM_BYPASS)


def _poll_interval_seconds() -> int:
    raw = os.environ.get(Config.ENV_REVENIUM_CB_POLL_INTERVAL_SECONDS, "")
    if not raw:
        return _DEFAULT_POLL_INTERVAL
    try:
        value = int(raw)
        return value if value > 0 else _DEFAULT_POLL_INTERVAL
    except ValueError:
        logger.debug("Invalid %s=%r, using default", Config.ENV_REVENIUM_CB_POLL_INTERVAL_SECONDS, raw)
        return _DEFAULT_POLL_INTERVAL


def _fail_mode_is_closed() -> bool:
    """``REVENIUM_CB_FAIL_MODE=closed`` raises when no usable cache exists."""
    return os.environ.get(Config.ENV_REVENIUM_CB_FAIL_MODE, "open").lower() == "closed"


def _cache_file_path() -> Optional[str]:
    cache_dir = os.environ.get(Config.ENV_REVENIUM_CACHE_DIR, "")
    if not cache_dir:
        return None
    return os.path.join(cache_dir, _RULES_CACHE_FILENAME)


def _get_enforcement_base_url() -> str:
    """Base URL for enforcement API calls.

    Prefers ``REVENIUM_ENFORCEMENT_BASE_URL`` so a context-path
    (``http://localhost:8080/profitstream``) survives intact. Falls back to
    the origin of the metering URL when unset.
    """
    explicit = os.environ.get(Config.ENV_REVENIUM_ENFORCEMENT_BASE_URL, "")
    if explicit:
        return explicit.rstrip("/")
    metering_url = os.environ.get(Config.ENV_REVENIUM_BASE_URL, "https://api.revenium.ai/meter/")
    parsed = urlparse(metering_url)
    return f"{parsed.scheme}://{parsed.netloc}"


def _load_cache_from_disk() -> None:
    global _cached_rules, _cache_timestamp, _disk_load_attempted, _cache_initialized
    if _disk_load_attempted:
        return
    _disk_load_attempted = True
    path = _cache_file_path()
    if not path or not os.path.exists(path):
        return
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        if isinstance(data, list):
            with _cache_lock:
                _cached_rules = data
                # Treat as stale so the next call still triggers a refresh,
                # but the disk snapshot prevents fail-closed from raising on
                # the very first request after a process restart.
                _cache_timestamp = 0.0
                _cache_initialized = True
            logger.debug("Loaded %d enforcement rule(s) from %s", len(data), path)
    except Exception:
        logger.debug("Failed to read enforcement cache from %s", path, exc_info=True)


def _persist_cache_to_disk(rules: list) -> None:
    path = _cache_file_path()
    if not path:
        return
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(rules, handle)
    except Exception:
        logger.debug("Failed to write enforcement cache to %s", path, exc_info=True)


def _fetch_rules() -> Optional[list]:
    """Fetch current enforcement rules from the Revenium API.

    Returns a list (possibly empty) on success, or ``None`` on failure so
    the caller can preserve the previous cache.
    """
    global _team_id_warned

    api_key = os.environ.get(Config.ENV_REVENIUM_API_KEY, "")
    if not api_key:
        logger.debug("No API key configured, skipping enforcement rule fetch")
        return None

    team_id = os.environ.get(Config.ENV_REVENIUM_TEAM_ID, "")
    if not team_id:
        if not _team_id_warned:
            logger.warning(
                "REVENIUM_TEAM_ID is not set — enforcement rule polling disabled. "
                "Set this to your hashed team ID to enable cost-limit enforcement."
            )
            _team_id_warned = True
        return None

    base_url = _get_enforcement_base_url()
    try:
        response = httpx.get(
            f"{base_url}/v2/api/ai/enforcement-rules/{team_id}",
            headers={"x-api-key": api_key},
            timeout=10,
        )
        # 204 No Content == no rules configured for this team; cache empty list
        if response.status_code == 204:
            return []
        response.raise_for_status()
        data = response.json()
        # Server currently returns ``{"rules": [...], "compiledAt": ...}`` but
        # accept a bare list too so a future schema change does not silently
        # AttributeError its way into a stale cache.
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            return data.get("rules", [])
        logger.warning("Unexpected enforcement response shape: %r", type(data).__name__)
        return None
    except Exception:
        logger.debug("Failed to fetch enforcement rules, falling open", exc_info=True)
        return None


def _refresh_cache() -> None:
    """Refresh the in-memory rule cache.

    Only advances ``_cache_timestamp`` on a successful fetch — a transient
    network error must not poison the stale-cache trigger and silently
    suppress retries for the next ``_CACHE_TTL`` window. Disk persistence
    happens outside ``_cache_lock`` so a slow filesystem write can't block
    concurrent ``check_enforcement`` callers on the pre-call path.
    """
    global _cached_rules, _cache_timestamp, _cache_initialized
    rules = _fetch_rules()
    if rules is None:
        return
    with _cache_lock:
        _cached_rules = rules
        _cache_timestamp = time.monotonic()
        _cache_initialized = True
    _persist_cache_to_disk(rules)


def _poll_loop() -> None:
    interval = _poll_interval_seconds()
    while not _stop_event.is_set():
        _refresh_cache()
        _stop_event.wait(interval)


def _ensure_poller_running() -> None:
    global _poll_thread
    with _poll_lock:
        if _poll_thread is not None and _poll_thread.is_alive():
            return
        _stop_event.clear()
        _poll_thread = threading.Thread(
            target=_poll_loop,
            name="revenium-enforcement-poll",
            daemon=True,
        )
        _poll_thread.start()


def _get_rules() -> list:
    """Return cached rules, refreshing synchronously when stale."""
    now = time.monotonic()
    with _cache_lock:
        age = now - _cache_timestamp
        rules = list(_cached_rules)
    if age > _CACHE_TTL:
        if _refresh_lock.acquire(blocking=False):
            try:
                _refresh_cache()
                with _cache_lock:
                    rules = list(_cached_rules)
            finally:
                _refresh_lock.release()
    return rules


def _coerce_float(value) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def check_enforcement(usage_metadata: Optional[dict] = None) -> None:
    """Pre-call enforcement check.

    Invoke before the upstream provider call. No-op when the circuit breaker
    is disabled or no rules are tripped.

    Raises:
        ReveniumCostLimitExceeded: when a cost-limit rule blocks the call.
            All structured fields (``rule_name``, ``current_value``,
            ``threshold``, ``resets_at``, ``rule_id``) are populated when the
            server provides them.
    """
    if is_bypass_enabled():
        return
    if not is_circuit_breaker_enabled():
        return

    _load_cache_from_disk()
    _ensure_poller_running()
    rules = _get_rules()

    # Fail-closed mode: only block when the cache has *never* loaded. An
    # empty list from a successful fetch (HTTP 204 = "no rules apply") is a
    # valid initialized state and must pass through.
    if _fail_mode_is_closed() and not _cache_initialized:
        raise ReveniumCostLimitExceeded(
            "Request blocked: enforcement cache is uninitialized and "
            "REVENIUM_CB_FAIL_MODE=closed."
        )

    credential = (usage_metadata or {}).get("subscriber_credential", "")

    for rule in rules:
        if not isinstance(rule, dict):
            continue
        # A rule is tripped when the server marks it as ``breached`` (current
        # nucleus schema) or ``blocked`` (legacy). Skip ``shadowMode`` rules
        # even when breached — those are observe-and-log only.
        is_tripped = rule.get("breached", False) or rule.get("blocked", False)
        if not is_tripped:
            continue
        if rule.get("shadowMode", False):
            continue

        rule_credential = rule.get("credential", "")
        if rule_credential and rule_credential != credential:
            continue

        rule_name = rule.get("name", "cost limit")
        raise ReveniumCostLimitExceeded(
            message=f"Request blocked by Revenium enforcement rule: {rule_name}",
            rule_name=rule_name,
            current_value=_coerce_float(rule.get("currentValue")),
            threshold=_coerce_float(rule.get("threshold")),
            resets_at=rule.get("resetsAt"),
            rule_id=rule.get("ruleId") or rule.get("id"),
        )


def stop_polling() -> None:
    """Gracefully stop the background polling thread.

    Reads ``_poll_thread`` under ``_poll_lock`` to avoid a TOCTOU race with
    ``_ensure_poller_running`` spinning the thread up on a concurrent first
    request — without the lock, shutdown could observe ``None`` and skip the
    ``join`` even though a poller is alive.
    """
    _stop_event.set()
    with _poll_lock:
        thread = _poll_thread
    if thread is not None:
        thread.join(timeout=5)
