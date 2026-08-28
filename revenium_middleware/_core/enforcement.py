"""
Enforcement engine for the Revenium circuit breaker.

Polls cost-limit rules from the Revenium API in a daemon thread and caches
them in memory. ``check_enforcement(...)`` is a pre-call hook that raises
``BudgetExceededError`` when a tripped rule matches the current
request, blocking the outbound provider call before any spend occurs.

Department (org-unit) budgets are decided by the server: the rules response
carries a top-level ``orgUnitBudgetBlocks`` map of subscriber email ->
blocking rule id, pre-computed from org-unit membership. The SDK looks the
caller's email up in that map and never resolves org-unit identity itself.

Opt-in via ``REVENIUM_CIRCUIT_BREAKER_ENABLED``. Disabled by default so the
SDK stays no-op for callers who haven't enrolled in cost controls.
"""

import datetime
import hashlib
import json
import logging
import os
import threading
import time
from email.utils import parsedate_to_datetime
from typing import Dict, List, NamedTuple, Optional, Tuple
from urllib.parse import quote, urlparse

import httpx

from .config import Config
from .exceptions import BudgetExceededError
from .subscriber import extract_subscriber_from_metadata

logger = logging.getLogger("revenium_middleware.extension")

_DEFAULT_POLL_INTERVAL = 60      # seconds between background refreshes
_CACHE_TTL = 120                  # seconds before a cached rule is considered stale

# Retry posture for the rule fetch (fleet backoff consistency): transient
# failures ride out a spike instead of leaving spend controls on stale
# limits until the next poll interval.
_FETCH_MAX_ATTEMPTS = 5
_FETCH_BACKOFF_INITIAL = 0.5      # seconds; doubles per attempt
_FETCH_BACKOFF_CAP = 8.0          # seconds; also clamps Retry-After
_FETCH_RETRYABLE_STATUS = frozenset({408, 429})  # plus any 5xx
_RULES_CACHE_FILENAME = "revenium_enforcement_rules.json"
# Department-budget map snapshot. A SEPARATE file, and not a new shape for the
# rules snapshot, on purpose: an older SDK's loader accepts only a bare list,
# so changing the rules file's shape would make a rollback silently discard
# the whole cache (fail-closed would then block everything during an API
# outage). The old loader never looks at this filename, so it can carry the
# new data without being load-bearing for a downgrade.
_ORG_UNIT_BLOCKS_CACHE_FILENAME = "revenium_enforcement_org_unit_blocks.json"

# Top-level key of the server-computed department-budget map, on both the API
# response and the disk snapshot.
_ORG_UNIT_BLOCKS_KEY = "orgUnitBudgetBlocks"
# ``groupBy`` value of an org-unit rule. Such rules are skipped by the
# per-rule loop: their verdict comes only from _ORG_UNIT_BLOCKS_KEY, matching
# the server's own ``applicableRules`` filter.
_ORG_UNIT_GROUP_BY = "ORG_UNIT"
# Reported when the email map names a rule id that is absent from the cached
# rules (a stale or racing payload). The server deliberately fails toward
# enforcing here, so the block still happens — just without a rule name.
_ORG_UNIT_FALLBACK_RULE_NAME = "Department budget"

_cached_rules: List[dict] = []
# Server-computed subscriber-email -> blocking-rule-id map for department
# (org-unit) budgets. Cached under _cache_lock with the same
# initialized/stale semantics as _cached_rules, and persisted alongside them.
_cached_org_unit_blocks: Dict[str, int] = {}
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


def _org_unit_blocks_cache_file_path() -> Optional[str]:
    cache_dir = os.environ.get(Config.ENV_REVENIUM_CACHE_DIR, "")
    if not cache_dir:
        return None
    return os.path.join(cache_dir, _ORG_UNIT_BLOCKS_CACHE_FILENAME)


def _rules_fingerprint(rules: list) -> str:
    """Content fingerprint binding the department map to the rules it was cut from.

    The two snapshot files are written separately, so a crash between the two
    writes can leave a newer map beside older rules. The rules file must stay
    a bare list (every published SDK reads exactly that shape), so the pairing
    proof lives in the map's envelope instead: the map carries the fingerprint
    of the rules payload it was computed against, and the loader drops a map
    whose fingerprint does not match the rules it actually loaded.
    """
    canonical = json.dumps(rules, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _write_private_json(path: str, payload: object) -> None:
    """Write ``payload`` as JSON readable only by the owning user (0600).

    The department map is keyed by subscriber email — PII that must not be
    left world-readable via the process umask on a shared host. ``os.open``
    applies the mode on create; ``os.fchmod`` tightens a file that already
    exists with looser permissions.
    """
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        if hasattr(os, "fchmod"):
            os.fchmod(handle.fileno(), 0o600)
        json.dump(payload, handle)


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
    global _cached_rules, _cached_org_unit_blocks
    global _cache_timestamp, _disk_load_attempted, _cache_initialized
    # Check-and-set under _cache_lock so two cold-start callers can't both
    # observe _disk_load_attempted=False and race on the snapshot read.
    with _cache_lock:
        if _disk_load_attempted:
            return
        _disk_load_attempted = True
    path = _cache_file_path()
    if not path or not os.path.exists(path):
        return
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        # The rules snapshot is a bare list — the one shape every SDK version
        # reads, which is what keeps a rollback from discarding the cache. A
        # dict carrying "rules" (a shape this branch briefly wrote) is
        # tolerated so no dev install strands its snapshot.
        if isinstance(data, list):
            rules = data
        elif isinstance(data, dict) and isinstance(data.get("rules"), list):
            rules = data["rules"]
        else:
            return
        # The department map lives in its own file; absent or malformed means
        # no department blocks until the next successful fetch.
        blocks: dict = {}
        blocks_path = _org_unit_blocks_cache_file_path()
        if blocks_path and os.path.exists(blocks_path):
            try:
                with open(blocks_path, "r", encoding="utf-8") as handle:
                    loaded = json.load(handle)
                # The envelope binds the map to the rules snapshot it was cut
                # from. A fingerprint mismatch means the pair on disk is torn
                # (e.g. the process died between the two writes): interpreting
                # rule IDs against the wrong rules could block the wrong
                # caller or fail open, so the map is dropped instead —
                # department blocks resume on the next successful fetch.
                if (
                    isinstance(loaded, dict)
                    and loaded.get("rules_fingerprint") == _rules_fingerprint(rules)
                    and isinstance(loaded.get("blocks"), dict)
                ):
                    blocks = loaded["blocks"]
                else:
                    logger.debug(
                        "Dropping department map from %s: it was not written "
                        "against the rules snapshot that loaded", blocks_path,
                    )
            except Exception:
                logger.debug(
                    "Failed to read department map from %s", blocks_path, exc_info=True
                )
        with _cache_lock:
            _cached_rules = rules
            _cached_org_unit_blocks = blocks
            # Treat as stale so the next call still triggers a refresh,
            # but the disk snapshot prevents fail-closed from raising on
            # the very first request after a process restart.
            _cache_timestamp = 0.0
            _cache_initialized = True
        logger.debug(
            "Loaded %d enforcement rule(s) and %d department block(s) from %s",
            len(rules), len(blocks), path,
        )
    except Exception:
        logger.debug("Failed to read enforcement cache from %s", path, exc_info=True)


def _persist_cache_to_disk(rules: list, org_unit_blocks: Optional[dict] = None) -> None:
    """Write the rules snapshot (legacy bare-list shape) and the map beside it.

    The rules file keeps the exact shape every published SDK version reads —
    a bare JSON list — so a rollback after this version has written a
    snapshot still loads it instead of silently discarding the cache. The
    department map rides in its own file (see
    ``_ORG_UNIT_BLOCKS_CACHE_FILENAME``), written 0600 because it is keyed by
    subscriber email. The map's envelope carries the
    fingerprint of the rules it was computed against, so however a crash
    interleaves the two writes, the loader can only ever pair a map with the
    exact rules snapshot it belongs to — a mismatched map is dropped, which
    merely skips department blocks until the next refresh.

    ``org_unit_blocks`` is optional so callers that only have rules (and the
    existing tests) keep working.
    """
    path = _cache_file_path()
    if not path:
        return
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        blocks_path = _org_unit_blocks_cache_file_path()
        if blocks_path:
            _write_private_json(blocks_path, {
                "rules_fingerprint": _rules_fingerprint(rules),
                "blocks": org_unit_blocks or {},
            })
        _write_private_json(path, rules)
    except Exception:
        logger.debug("Failed to write enforcement cache to %s", path, exc_info=True)


def _sleep(seconds: float) -> bool:
    """Backoff wait, interruptible by shutdown so the poller exits promptly.

    Returns True when shutdown was signalled, so callers can abort their
    retry loop instead of burning the remaining attempts with zero delay.
    """
    return _stop_event.wait(seconds)


def _is_retryable_status(status_code: int) -> bool:
    return status_code in _FETCH_RETRYABLE_STATUS or status_code >= 500


def _retry_after_seconds(response: "httpx.Response") -> Optional[float]:
    """Parse Retry-After (delta-seconds or HTTP-date), clamped to the cap."""
    raw = response.headers.get("Retry-After")
    if not raw:
        return None
    try:
        delay = float(raw)
    except ValueError:
        try:
            when = parsedate_to_datetime(raw)
            if when.tzinfo is None:
                when = when.replace(tzinfo=datetime.timezone.utc)
            delay = (when - datetime.datetime.now(datetime.timezone.utc)).total_seconds()
        except Exception:
            return None
    return max(0.0, min(delay, _FETCH_BACKOFF_CAP))


class _FetchedRules(NamedTuple):
    """One enforcement payload: the rules plus the department-budget map."""

    rules: list
    org_unit_blocks: dict


def _fetch_rules() -> Optional[_FetchedRules]:
    """Fetch the current enforcement payload from the Revenium API.

    Returns ``(rules, org_unit_blocks)`` on success — either may be empty --
    or ``None`` on failure so the caller can preserve the previous cache.
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
    # Percent-encode the team_id path segment so a misconfigured value
    # containing '/', '..', or '?' cannot retarget the request to a
    # different endpoint on the same origin.
    safe_team_id = quote(team_id, safe="")
    url = f"{base_url}/v2/api/ai/enforcement-rules/{safe_team_id}"
    for attempt in range(_FETCH_MAX_ATTEMPTS):
        backoff = min(_FETCH_BACKOFF_INITIAL * (2 ** attempt), _FETCH_BACKOFF_CAP)
        try:
            response = httpx.get(url, headers={"x-api-key": api_key}, timeout=10)
        except (httpx.TimeoutException, httpx.TransportError):
            if attempt == _FETCH_MAX_ATTEMPTS - 1:
                break
            if _sleep(backoff):
                break
            continue
        except Exception:
            logger.debug("Failed to fetch enforcement rules, falling open", exc_info=True)
            return None

        if _is_retryable_status(response.status_code):
            if attempt == _FETCH_MAX_ATTEMPTS - 1:
                break
            delay = _retry_after_seconds(response)
            if _sleep(delay if delay is not None else backoff):
                break
            continue

        try:
            # 204 No Content == no rules configured for this team; cache empty
            if response.status_code == 204:
                return _FetchedRules([], {})
            response.raise_for_status()
            data = response.json()
            # Server currently returns
            # ``{"rules": [...], "compiledAt": ..., "orgUnitBudgetBlocks": {...}}``
            # but accept a bare list too so a future schema change does not
            # silently AttributeError its way into a stale cache. A legacy
            # bare-list body — and a dict body predating the map — yields an
            # empty department map, which blocks nobody.
            if isinstance(data, list):
                return _FetchedRules(data, {})
            if isinstance(data, dict):
                blocks = data.get(_ORG_UNIT_BLOCKS_KEY)
                return _FetchedRules(
                    data.get("rules", []),
                    blocks if isinstance(blocks, dict) else {},
                )
            logger.warning("Unexpected enforcement response shape: %r", type(data).__name__)
            return None
        except Exception:
            logger.debug("Failed to fetch enforcement rules, falling open", exc_info=True)
            return None

    # Deliberate fail-open: an enforcement-refresh outage must never become a
    # customer traffic outage. The caller preserves the previous cache and the
    # next poll cycle tries again.
    logger.warning(
        "Enforcement rule fetch exhausted %d attempts; failing open on the previous cache",
        _FETCH_MAX_ATTEMPTS,
    )
    return None


def _refresh_cache() -> None:
    """Refresh the in-memory rule cache.

    Only advances ``_cache_timestamp`` on a successful fetch — a transient
    network error must not poison the stale-cache trigger and silently
    suppress retries for the next ``_CACHE_TTL`` window. Disk persistence
    happens outside ``_cache_lock`` so a slow filesystem write can't block
    concurrent ``check_enforcement`` callers on the pre-call path.
    """
    global _cached_rules, _cached_org_unit_blocks, _cache_timestamp, _cache_initialized
    fetched = _fetch_rules()
    if fetched is None:
        return
    with _cache_lock:
        _cached_rules = fetched.rules
        _cached_org_unit_blocks = fetched.org_unit_blocks
        _cache_timestamp = time.monotonic()
        _cache_initialized = True
    _persist_cache_to_disk(fetched.rules, fetched.org_unit_blocks)


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


def _snapshot_org_unit_blocks() -> dict:
    """Copy the cached department map. Caller must hold ``_cache_lock``."""
    return dict(_cached_org_unit_blocks) if isinstance(_cached_org_unit_blocks, dict) else {}


def _get_rules() -> Tuple[list, dict, bool]:
    """Return cached rules, the department map, and the initialized flag.

    Reading all three fields under one ``_cache_lock`` acquisition prevents
    the fail-closed path from torn-reading ``_cache_initialized`` against a
    half-written ``_cached_rules`` from the background poller, and keeps the
    department map consistent with the rules it references.
    """
    now = time.monotonic()
    with _cache_lock:
        age = now - _cache_timestamp
        rules = list(_cached_rules)
        org_unit_blocks = _snapshot_org_unit_blocks()
        initialized = _cache_initialized
    if age > _CACHE_TTL:
        if _refresh_lock.acquire(blocking=False):
            try:
                _refresh_cache()
                with _cache_lock:
                    rules = list(_cached_rules)
                    org_unit_blocks = _snapshot_org_unit_blocks()
                    initialized = _cache_initialized
            finally:
                _refresh_lock.release()
    return rules, org_unit_blocks, initialized


def _coerce_float(value) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _matching_group_entry(breakdown: list, usage_metadata: Optional[dict]) -> Optional[dict]:
    """Find this caller's entry in a rule's ``groupBreakdown``.

    ``EnforcementGroupEntry.groupValue`` is documented as "subscriber id,
    else email, else the unattributed sentinel", so the id is tried before
    the email — matching on email first would mis-attribute a caller who
    supplies both. Both are tried as separate lookups because a caller may
    send only one of them while the entry carries the other.

    Returns ``None`` when no group key resolves from the metadata or when no
    entry matches. The caller treats that as fail-open (no block): the
    unattributed sentinel's value is not published yet, so matching against
    it is deferred rather than guessed — a wrong guess would block a caller
    against a bucket that is not theirs.

    The flat ``subscriber_id`` / ``subscriber_email`` keys are consulted as
    fallbacks after the shared extractor: its nested branch suppresses the
    flat one entirely, so a caller sending a partial nested ``subscriber``
    plus a flat id would otherwise never match an id-keyed entry and slip
    past its own breached balance.
    """
    metadata = usage_metadata or {}
    subscriber = extract_subscriber_from_metadata(metadata)
    candidates = (
        subscriber.get("id"),
        metadata.get("subscriber_id"),
        subscriber.get("email"),
        metadata.get("subscriber_email"),
    )
    seen = set()
    for key in candidates:
        # Only strings are meaningful group keys (groupValue is a string in
        # the spec); anything else — including unhashable garbage a caller
        # might put in subscriber_id — must fail open, never abort the
        # pre-provider hook with a TypeError.
        if not isinstance(key, str) or not key or key in seen:
            continue
        seen.add(key)
        for entry in breakdown:
            if isinstance(entry, dict) and entry.get("groupValue") == key:
                return entry
    return None


def _rule_blocks(rule: dict, usage_metadata: Optional[dict]) -> Tuple[bool, Optional[float]]:
    """Decide whether ``rule`` blocks this caller, and with which balance.

    Returns ``(blocks, current_value)`` where ``current_value`` is the
    balance to report on ``BudgetExceededError`` — the caller's own group
    balance for a grouped rule, the pooled balance otherwise.

    Chosen semantics for grouped rules: when ``groupBreakdown`` is a
    non-empty list the caller's matching entry is authoritative in both
    directions — rule-level ``breached`` alone never blocks a caller whose
    entry is not breached, and an entry marked ``breached`` blocks even when
    the rule-level flag is false. This reading is pending platform-team
    confirmation of whether rule-level ``breached`` on a grouped rule means
    "the pooled total breached" or "at least one group breached"; deferring
    to the per-group balance is correct under either answer, whereas trusting
    the aggregate flag would apply one verdict to every caller.

    A missing, null, or empty ``groupBreakdown`` is normal and never an
    error: the field is null for pooled rules and is populated on API reads
    only (it is absent from the Redis snapshot), so those rules keep the
    rule-level behaviour unchanged.
    """
    breakdown = rule.get("groupBreakdown")
    if isinstance(breakdown, list) and breakdown:
        entry = _matching_group_entry(breakdown, usage_metadata)
        if entry is None or not entry.get("breached", False):
            return False, None
        return True, _coerce_float(entry.get("currentValue"))

    # Pooled rule: tripped when the server marks it ``breached`` (current
    # nucleus schema) or ``blocked`` (legacy).
    tripped = rule.get("breached", False) or rule.get("blocked", False)
    return bool(tripped), _coerce_float(rule.get("currentValue"))


def _is_org_unit_rule(rule: dict) -> bool:
    """True for any org-unit-scoped or org-unit-grouped rule.

    Mirrors the server's own ``applicableRules`` exclusion exactly:
    ``orgUnitId == null && groupBy != "ORG_UNIT"`` — a rule is org-unit when
    EITHER field says so. Ancestor-cap department rules carry
    ``orgUnitId != null`` with a non-ORG_UNIT (typically null) ``groupBy``,
    and their rule-level ``breached`` means "this department is over budget",
    not "this caller is over budget" — evaluated in the per-rule loop they
    would block every unrelated employee company-wide. Their verdict, like
    the grouped shape's, lives solely in the ``orgUnitBudgetBlocks`` map,
    because only the server knows which org unit the caller belongs to.
    """
    if rule.get("orgUnitId") is not None:
        return True
    group_by = rule.get("groupBy")
    return isinstance(group_by, str) and group_by.strip().upper() == _ORG_UNIT_GROUP_BY


def _caller_emails(usage_metadata: Optional[dict]) -> List[str]:
    """The caller's subscriber email(s), nested-first then the flat fallback.

    Deliberately email-only: ``orgUnitBudgetBlocks`` is keyed by subscriber
    email because that is the identity the server can resolve to an org-unit
    membership. A caller who supplies no email simply has no key, and — as
    with ``_matching_group_entry`` — no sentinel is invented for them: a
    guessed key could block someone against a department that is not theirs.
    """
    metadata = usage_metadata or {}
    subscriber = extract_subscriber_from_metadata(metadata)
    # The nested subscriber email is authoritative; the flat subscriber_email
    # is consulted ONLY when no usable nested email exists. Mixed metadata can
    # name two different people, and looking both up would let a stale or
    # spoofed flat email block the caller against a department that is not
    # theirs. Only strings are meaningful map keys, and a non-string lookup
    # key could be unhashable — neither may abort the pre-provider hook.
    for candidate in (subscriber.get("email"), metadata.get("subscriber_email")):
        if isinstance(candidate, str) and candidate:
            return [candidate]
    return []


def _org_unit_block_rule_id(org_unit_blocks: dict, usage_metadata: Optional[dict]) -> Optional[int]:
    """Look the caller up in the department map; return the blocking rule id.

    ``None`` means no block: no email on the caller, no entry for it, or a
    value that is not a rule id (the map is documented as email -> int, so
    anything else is treated as absent rather than as a blocking verdict).
    """
    if not isinstance(org_unit_blocks, dict) or not org_unit_blocks:
        return None
    for email in _caller_emails(usage_metadata):
        rule_id = org_unit_blocks.get(email)
        # bool is an int subclass; a JSON ``true`` is not a rule id.
        if isinstance(rule_id, int) and not isinstance(rule_id, bool):
            return rule_id
    return None


def _rule_by_id(rules: list, rule_id: int) -> Optional[dict]:
    """Resolve a rule from the cached list by id.

    Ids are matched, not names: rule names are not unique, so resolving the
    map's value by name could report — or suppress — the wrong rule.
    """
    for rule in rules:
        if not isinstance(rule, dict):
            continue
        candidate = rule.get("ruleId")
        if candidate is None:
            candidate = rule.get("id")
        if isinstance(candidate, bool) or not isinstance(candidate, (int, str)):
            continue
        if str(candidate) == str(rule_id):
            return rule
    return None


def _raise_org_unit_block(rules: list, org_unit_blocks: dict,
                          usage_metadata: Optional[dict]) -> None:
    """Raise ``BudgetExceededError`` when the department map blocks the caller.

    Mirrors the server's own consumer of ``orgUnitBudgetBlocks``: the map is
    the verdict, and the rule is looked up only to describe it. A map hit
    naming a rule that is not in the cached list — a stale or racing payload
    — still blocks, under a generic name, because the server publishes the
    map only for departments it has already decided are over budget.
    """
    rule_id = _org_unit_block_rule_id(org_unit_blocks, usage_metadata)
    if rule_id is None:
        return

    rule = _rule_by_id(rules, rule_id)
    if rule is None:
        raise BudgetExceededError(
            message=(
                "Request blocked by Revenium enforcement rule: "
                f"{_ORG_UNIT_FALLBACK_RULE_NAME}"
            ),
            rule_name=_ORG_UNIT_FALLBACK_RULE_NAME,
            rule_id=rule_id,
        )

    # Observe-and-log only. Other rules still evaluate normally.
    if rule.get("shadowMode", False):
        logger.debug("Department budget rule %s is in shadow mode; not blocking", rule_id)
        return

    # A rule that only warns must not block. An absent ``action`` is treated
    # as BLOCK: the map itself is the server's blocking decision, so a
    # payload that omits the field falls toward enforcing, not toward spend.
    action = rule.get("action")
    if action is not None and (not isinstance(action, str) or action.strip().upper() != "BLOCK"):
        logger.warning(
            "Department budget threshold crossed for rule %s (action=%s); not blocking",
            rule_id, action,
        )
        return

    rule_name = rule.get("name") or _ORG_UNIT_FALLBACK_RULE_NAME
    raise BudgetExceededError(
        message=f"Request blocked by Revenium enforcement rule: {rule_name}",
        rule_name=rule_name,
        current_value=_coerce_float(rule.get("currentValue")),
        threshold=_coerce_float(rule.get("threshold")),
        resets_at=rule.get("resetsAt"),
        rule_id=rule.get("ruleId") or rule.get("id") or rule_id,
    )


def _check_org_unit_block(rules: list, org_unit_blocks: dict,
                          usage_metadata: Optional[dict]) -> None:
    """``_raise_org_unit_block`` with a fail-open guard.

    The pre-provider hook may raise ``BudgetExceededError`` and nothing else,
    so a malformed map degrades to "no block" instead of taking the caller's
    request down with it.
    """
    try:
        _raise_org_unit_block(rules, org_unit_blocks, usage_metadata)
    except BudgetExceededError:
        raise
    except Exception:
        logger.debug("Department budget check failed; falling open", exc_info=True)


def check_enforcement(usage_metadata: Optional[dict] = None) -> None:
    """Pre-call enforcement check.

    Invoke before the upstream provider call. No-op when the circuit breaker
    is disabled or no rules are tripped.

    Subscriber-grouped rules are evaluated against the caller's own balance
    from ``groupBreakdown`` rather than the rule-level aggregate; see
    ``_rule_blocks``.

    Department (org-unit) budgets are decided before and independently of the
    per-rule loop, from the server's ``orgUnitBudgetBlocks`` email map; see
    ``_raise_org_unit_block``.

    Raises:
        BudgetExceededError: when a cost-limit rule blocks the call.
            All structured fields (``rule_name``, ``current_value``,
            ``threshold``, ``resets_at``, ``rule_id``) are populated when the
            server provides them. For a grouped rule ``current_value`` is the
            caller's group balance while ``threshold`` stays rule-level.
    """
    if is_bypass_enabled():
        return
    if not is_circuit_breaker_enabled():
        return

    _load_cache_from_disk()
    _ensure_poller_running()
    rules, org_unit_blocks, initialized = _get_rules()

    # Fail-closed mode: only block when the cache has *never* loaded. An
    # empty list from a successful fetch (HTTP 204 = "no rules apply") is a
    # valid initialized state and must pass through. Use the snapshot taken
    # under _cache_lock so the decision can't see a torn write.
    if _fail_mode_is_closed() and not initialized:
        raise BudgetExceededError(
            "Request blocked: enforcement cache is uninitialized and "
            "REVENIUM_CB_FAIL_MODE=closed."
        )

    # Department budgets: the server's pre-computed email map is the whole
    # verdict, so it is consulted before and independently of the per-rule
    # loop — a department block can name a rule the cache has not seen yet.
    _check_org_unit_block(rules, org_unit_blocks, usage_metadata)

    credential = (usage_metadata or {}).get("subscriber_credential", "")

    for rule in rules:
        if not isinstance(rule, dict):
            continue
        # Org-unit-grouped rules are decided only by the map above; the SDK
        # cannot evaluate them locally because it never learns the caller's
        # org unit. Skipping them mirrors the server's applicableRules filter.
        if _is_org_unit_rule(rule):
            continue
        # ``shadowMode`` rules are observe-and-log only, so they never block —
        # whether the verdict would have come from the pooled flag or from a
        # per-group balance.
        if rule.get("shadowMode", False):
            continue

        # Legacy-payload compatibility shim — NOT the grouped-rule mechanism.
        # ``credential`` is not a member of CompiledEnforcementRule in any
        # published spec version, so this branch is inert for compiled rules;
        # it is kept because a mismatch currently *skips* the rule, and
        # dropping it would widen blocking for any legacy payload still
        # carrying the field. Per-caller attribution for compiled rules goes
        # through ``groupBy``/``groupBreakdown`` (see ``_rule_blocks``).
        rule_credential = rule.get("credential", "")
        if rule_credential and rule_credential != credential:
            continue

        blocks, current_value = _rule_blocks(rule, usage_metadata)
        if not blocks:
            continue

        rule_name = rule.get("name", "cost limit")
        raise BudgetExceededError(
            message=f"Request blocked by Revenium enforcement rule: {rule_name}",
            rule_name=rule_name,
            current_value=current_value,
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
