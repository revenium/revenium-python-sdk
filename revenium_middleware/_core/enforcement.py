"""
Enforcement engine for the Revenium circuit breaker.

Polls cost-limit rules from the Revenium API in a daemon thread and caches
them in memory. ``check_enforcement(...)`` is a pre-call hook that raises
``BudgetExceededError`` when a tripped rule matches the current
request, blocking the outbound provider call before any spend occurs.

Department (org-unit) budgets are decided by the server: the rules response
carries a top-level ``orgUnitBudgetBlocks`` map of subscriber email ->
blocking rule id, pre-computed from org-unit membership, an
``orgUnitBudgetBlockBalances`` map giving the balance each of those people was
actually judged against, and an ``orgUnitBudgetWarnings`` map of the people who
have crossed a warn tier without being blocked. All three are keyed by the
*normalized* address (trimmed, lower-cased), so the SDK re-keys them by that
form as they come in and normalizes the caller's address the same way before
looking it up; it never resolves org-unit identity itself.

Decision (BACK-3077): the warn tier surfaces as a log line and nothing else.
A warning carries no directive a pre-call breaker can enforce -- raising on it
would turn "you are close" into "you are blocked" -- and the module has no
hook or callback pattern to extend, so inventing a notification framework here
would be a bigger interface than the signal justifies. The line is emitted
once per (cache generation, caller, rule) rather than once per call, so it
stays visible instead of becoming the noise it is meant to stand out from, and
it names no address because these maps are PII (see ``_normalize_map_keys``).
An integrator who wants more can read the map, which is cached and persisted
like the other two. Revisit if a caller asks for a programmatic warn signal.

Two inspection calls sit beside the enforcement path without being part of
it. ``fetch_enforcement_rule(rule_id)`` reads one rule straight from the
server, and ``fetch_enforcement_rule_roster(rule_id)`` reads the people (or
departments) that rule is measuring, each with their spend, cap and band.
Neither is cached and neither is consulted by ``check_enforcement``: they
answer "why was this caller blocked, and who else does this rule cover?" for
a person debugging, and a cached answer to that question would be worse than
no answer. The polling refresh stays team-wide for the reason given on
``_fetch_rules``.

Opt-in via ``REVENIUM_CIRCUIT_BREAKER_ENABLED``. Disabled by default so the
SDK stays no-op for callers who haven't enrolled in cost controls.
"""

import datetime
import hashlib
import json
import logging
import math
import os
import threading
import time
from email.utils import parsedate_to_datetime
from typing import Dict, List, NamedTuple, Optional, Set, Tuple
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
_FETCH_BACKOFF_CAP = 8.0          # seconds; caps the backoff we choose ourselves
_FETCH_RETRYABLE_STATUS = frozenset({408, 429})  # plus any 5xx

# Give-up threshold for *server-supplied* ``Retry-After`` waits, deliberately
# NOT _FETCH_BACKOFF_CAP. It is a budget for one whole ``_fetch_rules`` call,
# not a per-response limit: a response is honoured only if its interval fits
# entirely in what is left, so a server repeating ``Retry-After: 20`` across
# five attempts can park a caller for 20 s in total rather than 100 s.
#
# Inside the budget the fetch waits the whole interval that was asked for;
# outside it the fetch neither waits nor retries, records a cooldown (see
# ``_begin_refresh_cooldown``) and fails open on the cached rules. A bound on
# a delay the server chose is a give-up threshold, never a ``min()`` on the
# wait -- truncating it would retry sooner than the 429 permitted, which is
# the one thing the status code asks a client not to do (RFC 9110 §10.2.3).
#
# 20 s rather than the 60 s the metering client honours (see
# ``_parse_retry_after_header`` in ``_metering/_base_client.py``) because this
# fetch is synchronous: ``_get_rules`` calls ``_refresh_cache`` on the caller's
# thread once the cache is past ``_CACHE_TTL``, so the wait parks a customer
# request that is already blocked on its own provider call. If this path ever
# becomes background-only the bound becomes 60 s. Cross-SDK numbers:
# ``docs/conventions/retry-after.md``.
_RETRY_AFTER_GIVE_UP_SECONDS = 20.0

# Longest a single Retry-After may suppress rule refreshes. This is NOT the
# ``min()`` on a wait that rule 1 forbids: nothing is parked and no retry is
# issued early inside the cooldown window -- the fetch has already given up,
# every caller is served from cache, and no request goes out until the window
# ends. All the ceiling bounds is how long one absurd or malformed header
# ("Retry-After: 999999") can keep the SDK from refreshing at all, so cached
# limits cannot go stale for hours on the strength of a single response. Any
# realistic value (60 s, 120 s, 300 s) is honoured exactly.
_REFRESH_COOLDOWN_CEILING_SECONDS = 300.0
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
# Top-level key of the per-person balance map published beside the block map:
# normalized email -> the balance that person's threshold was compared against,
# in dollars. Absent on a server predating it, which is why every read of it
# falls back to the rule's own ``currentValue``.
_ORG_UNIT_BLOCK_BALANCES_KEY = "orgUnitBudgetBlockBalances"
# Top-level key of the warn-tier map published beside them: normalized email ->
# the rule whose warn tier that person crossed, for a per-person cap scoped to
# one department. Disjoint from _ORG_UNIT_BLOCKS_KEY (a person already blocked
# is not warned) and never a block verdict -- see the module docstring's
# BACK-3077 decision. Absent on a server predating it, which warns nobody.
_ORG_UNIT_WARNINGS_KEY = "orgUnitBudgetWarnings"
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
# Server-computed subscriber-email -> own-balance map for the same department
# budgets, cached and persisted alongside the block map. Empty when the server
# does not publish it.
#
# Both cached maps hold keys already normalized by _normalize_email: every path
# that fills them (_fetched_from_payload, _coerce_balances,
# _load_department_budgets_from_disk) re-keys on the way in, so the
# pre-provider lookup is a plain dict hit.
_cached_org_unit_block_balances: Dict[str, float] = {}
# Server-computed subscriber-email -> warn-tier-rule-id map, cached and
# persisted alongside the other two and re-keyed the same way. Nobody is
# blocked by it; it only decides who hears _warn_org_unit_threshold's line,
# once per _cache_generation.
_cached_org_unit_warnings: Dict[str, int] = {}
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

# Which cache generation the department maps in memory belong to. Bumped, under
# _cache_lock, wherever those maps are replaced -- a fetch or a disk load -- and
# copied into the snapshot each pre-provider check reads
# (_snapshot_department_budgets), so a check always knows which verdict it is
# acting on. Starts at 0, which is the generation of the empty maps a process
# starts with: nobody can be warned under it.
_cache_generation = 0

# (generation, caller key, rule id) triples already warned about. The
# generation is part of the key, not merely the reason the set is cleared: a
# check that snapshotted generation N can be descheduled, have a concurrent
# poll install generation N+1 and clear the set, and only then take its claim.
# Keyed by generation, that late claim lands under N and cannot make N+1 look
# already-warned, so the caller still hears the refreshed verdict. Clearing on a
# bump is what bounds the set. Its own lock, not _cache_lock: both writers of
# the maps hold _cache_lock while replacing them, and threading.Lock is not
# reentrant. Mutated in place so a test can clear it without rebinding.
_warn_dedupe_lock = threading.Lock()
_warned_org_unit_callers: Set[Tuple[int, str, int]] = set()

# Monotonic deadline before which no refresh may issue a request, set when the
# fetch declines to wait out a Retry-After. 0.0 means no cooldown in force.
# Read and written under _cache_lock.
_refresh_cooldown_until = 0.0

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


class _DepartmentBudgets(NamedTuple):
    """The server's department-budget verdicts, and what each one was judged on.

    ``blocks`` is normalized subscriber email -> blocking rule id and is the
    whole verdict. ``balances`` is normalized subscriber email -> the balance
    that person's threshold was compared against, in dollars; it is empty on a
    server that does not publish it, and has no entry for a caller whose rule
    is an ancestor cap (there the rule's own ``currentValue`` already is the
    scope's spend). ``warnings`` is normalized subscriber email -> the rule
    whose warn tier that person crossed *without* being blocked; it decides
    only who hears a log line, never who is blocked.

    All three are keyed by ``_normalize_email`` because whoever built them
    re-keyed them, never by the raw key the payload or the snapshot carried.

    ``generation`` is which cache generation these maps are, so a check can
    claim its warning against the verdict it actually read rather than against
    whatever is current by the time it gets to the claim. It is filled in by
    ``_snapshot_department_budgets``; the disk loader leaves it at 0, the
    generation of the empty maps a process starts with, because its result is
    unpacked into the cache rather than evaluated.
    """

    blocks: dict
    balances: dict
    warnings: dict
    generation: int = 0


def _normalize_map_keys(raw) -> dict:
    """Re-key a server map by the normalized address, once, as it comes in.

    Normalizing only the caller's side would leave the map to be scanned
    whenever the exact key missed -- and a miss is the ordinary case, since
    most callers are not blocked -- putting an O(n) ``strip().lower()`` per key
    in front of every provider call. Re-keying here instead makes the
    pre-provider lookup a plain dict hit. The server already publishes
    normalized keys, so on a current payload this rewrites nothing; it exists
    so a pre-normalization server, or a hand-edited cache file, still matches.

    First writer wins when two keys normalize to one address: there is no
    ordering rule that says which of them the server meant, and dropping both
    would fail a block it had already decided open. Non-string keys are
    skipped -- no caller's key can ever be one.
    """
    if not isinstance(raw, dict):
        return {}
    normalized: dict = {}
    for key, value in raw.items():
        if not isinstance(key, str):
            continue
        canonical = _normalize_email(key)
        if canonical in normalized:
            # Logged without the address: these maps are PII.
            logger.debug(
                "Ignoring a department map key that normalizes onto an existing entry"
            )
            continue
        normalized[canonical] = value
    return normalized


def _coerce_balances(raw) -> Dict[str, float]:
    """Re-key a balance map by normalized address and coerce it to floats.

    The field is a Kotlin ``BigDecimal`` on the wire, which serializes as a
    JSON number or as a string, so both are coerced once here rather than at
    every read. An absent or non-dict map, a non-string key, a boolean and an
    unparseable value all mean "no balance published for that person", which
    falls back to the rule's own value -- never to an exception on the
    pre-provider path. Unusable values are dropped before the collision rule
    applies, so a duplicate key cannot park garbage in a slot whose other
    spelling carried a real number.
    """
    if not isinstance(raw, dict):
        return {}
    balances: Dict[str, float] = {}
    for key, value in raw.items():
        # bool is an int subclass; a JSON ``true`` is not a balance.
        if not isinstance(key, str) or isinstance(value, bool):
            continue
        balance = _coerce_float(value)
        # A JSON-ish "NaN" or "Infinity" survives float(); a non-finite
        # balance is not a dollar figure and must fall back to the rule's
        # own value rather than reach BudgetExceededError.current_value.
        if balance is None or not math.isfinite(balance):
            continue
        canonical = _normalize_email(key)
        if canonical in balances:
            logger.debug(
                "Ignoring a balance entry that normalizes onto an existing entry"
            )
            continue
        balances[canonical] = balance
    return balances


def _load_department_budgets_from_disk(rules: list) -> _DepartmentBudgets:
    """Read the department-budget snapshot written beside ``rules``.

    The maps live in their own file; absent or malformed means no department
    blocks until the next successful fetch. The envelope binds them to the
    rules snapshot they were cut from. A fingerprint mismatch means the pair on
    disk is torn (e.g. the process died between the two writes): interpreting
    rule IDs against the wrong rules could block the wrong caller or fail open,
    so the maps are dropped instead -- department blocks resume on the next
    successful fetch.
    """
    empty = _DepartmentBudgets({}, {}, {})
    blocks_path = _org_unit_blocks_cache_file_path()
    if not blocks_path or not os.path.exists(blocks_path):
        return empty
    try:
        with open(blocks_path, "r", encoding="utf-8") as handle:
            loaded = json.load(handle)
        if (
            isinstance(loaded, dict)
            and loaded.get("rules_fingerprint") == _rules_fingerprint(rules)
            and isinstance(loaded.get("blocks"), dict)
        ):
            # Re-keyed on the way in, like the wire payload: a snapshot
            # written by an older SDK (or edited by hand) can carry keys the
            # server had not normalized. ``balances`` and ``warnings`` are
            # absent from an envelope written before each map existed: the
            # first reports the rule's own value as it always did, the second
            # warns nobody until the next successful fetch.
            return _DepartmentBudgets(
                _normalize_map_keys(loaded["blocks"]),
                _coerce_balances(loaded.get("balances")),
                _normalize_map_keys(loaded.get("warnings")),
            )
        logger.debug(
            "Dropping department map from %s: it was not written "
            "against the rules snapshot that loaded", blocks_path,
        )
    except Exception:
        logger.debug(
            "Failed to read department map from %s", blocks_path, exc_info=True
        )
    return empty


def _load_cache_from_disk() -> None:
    global _cached_rules, _cached_org_unit_blocks, _cached_org_unit_block_balances
    global _cached_org_unit_warnings
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
        departments = _load_department_budgets_from_disk(rules)
        with _cache_lock:
            _cached_rules = rules
            _cached_org_unit_blocks = departments.blocks
            _cached_org_unit_block_balances = departments.balances
            _cached_org_unit_warnings = departments.warnings
            _begin_cache_generation()
            # Treat as stale so the next call still triggers a refresh,
            # but the disk snapshot prevents fail-closed from raising on
            # the very first request after a process restart.
            _cache_timestamp = 0.0
            _cache_initialized = True
        logger.debug(
            "Loaded %d enforcement rule(s) and %d department block(s) from %s",
            len(rules), len(departments.blocks), path,
        )
    except Exception:
        logger.debug("Failed to read enforcement cache from %s", path, exc_info=True)


def _persist_cache_to_disk(rules: list, org_unit_blocks: Optional[dict] = None,
                           org_unit_block_balances: Optional[dict] = None,
                           org_unit_warnings: Optional[dict] = None) -> None:
    """Write the rules snapshot (legacy bare-list shape) and the maps beside it.

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

    The map arguments are optional so callers that only have rules (and the
    existing tests) keep working; an omitted map is written empty, which is
    also what an envelope predating that map loads as.
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
                "balances": org_unit_block_balances or {},
                "warnings": org_unit_warnings or {},
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
    """Parse Retry-After in either RFC 9110 form: delta-seconds or HTTP-date.

    Returns the full interval the server asked for, unclamped -- deciding
    whether it is short enough to honour belongs to the caller, which weighs
    it against ``_RETRY_AFTER_GIVE_UP_SECONDS``. ``None`` means the header was
    absent or unparseable, so the caller falls back to its own backoff.
    """
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
    return max(0.0, delay)


def _begin_refresh_cooldown(seconds: float) -> float:
    """Stop issuing refresh requests for ``seconds``, as the server asked.

    Giving up on a ``Retry-After`` is only half of honouring it. Without a
    cooldown the next customer request finds the same stale cache, triggers the
    same synchronous refresh and issues another control-plane request -- one
    per customer request for as long as the throttling lasts, which is worse
    than the truncated retry this replaced. Recording the deadline honours the
    server's interval in full while making nobody wait for it: refreshes are
    skipped and callers are served from the cached rules until it passes.

    Clamped to ``_REFRESH_COOLDOWN_CEILING_SECONDS`` -- a sanity bound on how
    long one header may suppress refreshes, not a truncated wait; see that
    constant. Never shortens a cooldown already in force. Returns the seconds
    remaining until the deadline now in effect.
    """
    global _refresh_cooldown_until
    now = time.monotonic()
    deadline = now + min(max(0.0, seconds), _REFRESH_COOLDOWN_CEILING_SECONDS)
    with _cache_lock:
        _refresh_cooldown_until = max(_refresh_cooldown_until, deadline)
        return _refresh_cooldown_until - now


def _refresh_is_on_cooldown() -> bool:
    """True while a Retry-After interval the fetch declined to wait is running."""
    with _cache_lock:
        return time.monotonic() < _refresh_cooldown_until


class _RetryPlan(NamedTuple):
    """What to do about one retryable response, and what budget is left after.

    ``wait_seconds`` of ``None`` means stop: do not wait, do not retry, fail
    open on the cached rules.
    """

    wait_seconds: Optional[float]
    budget_remaining: float


def _plan_retry(response: "httpx.Response", backoff: float, budget_remaining: float) -> _RetryPlan:
    """Decide how to handle a retryable response, spending the Retry-After budget.

    Three outcomes, and the ``None`` is the one that matters:

    * no usable ``Retry-After`` -- our own ``backoff``, which we are free to
      cap and which does not spend the budget: the budget exists to bound how
      long the *server* can park a caller, not how long we park ourselves;
    * a ``Retry-After`` that fits in ``budget_remaining`` -- that interval in
      full, never truncated, and the budget shrinks by it;
    * a ``Retry-After`` that does not fit -- ``None``. Waiting it out would
      park a customer request past the bound; retrying sooner would ignore the
      429. So the fetch does neither: it records the cooldown so the interval
      is still respected, and the caller keeps the rules it already has.
    """
    retry_after = _retry_after_seconds(response)
    if retry_after is None:
        return _RetryPlan(backoff, budget_remaining)
    if retry_after > budget_remaining:
        cooldown = _begin_refresh_cooldown(retry_after)
        logger.warning(
            "Enforcement rule fetch got HTTP %d asking for %.0fs with %.0fs of the %.0fs "
            "Retry-After budget left; failing open on the previous cache and not "
            "refreshing again for %.0fs",
            response.status_code, retry_after, budget_remaining,
            _RETRY_AFTER_GIVE_UP_SECONDS, cooldown,
        )
        return _RetryPlan(None, budget_remaining)
    return _RetryPlan(retry_after, budget_remaining - retry_after)


class _FetchedRules(NamedTuple):
    """One enforcement payload: the rules plus the department-budget maps."""

    rules: list
    org_unit_blocks: dict
    org_unit_block_balances: dict
    org_unit_warnings: dict


def _fetched_from_payload(data) -> Optional[_FetchedRules]:
    """Split one enforcement response body into rules and department maps.

    Returns ``None`` for a body that is neither a list nor an object, so the
    caller preserves its previous cache instead of AttributeError-ing its way
    into an empty one. A legacy bare-list body -- and a dict body predating any
    of the maps -- yields empty maps: nobody is blocked by the map, a
    department block reports the rule's own value exactly as before, and nobody
    is warned.

    The warn map is re-keyed exactly like the block map, and its values are
    validated as rule ids at the lookup rather than here, so the two maps
    cannot drift apart in what they accept (``_org_unit_map_entry``).
    """
    if isinstance(data, list):
        return _FetchedRules(data, {}, {}, {})
    if isinstance(data, dict):
        return _FetchedRules(
            data.get("rules", []),
            _normalize_map_keys(data.get(_ORG_UNIT_BLOCKS_KEY)),
            _coerce_balances(data.get(_ORG_UNIT_BLOCK_BALANCES_KEY)),
            _normalize_map_keys(data.get(_ORG_UNIT_WARNINGS_KEY)),
        )
    logger.warning("Unexpected enforcement response shape: %r", type(data).__name__)
    return None


class _FetchTarget(NamedTuple):
    """The credentials and the team-scoped URL every enforcement read hangs off."""

    api_key: str
    team_url: str


def _fetch_target() -> Optional[_FetchTarget]:
    """Resolve the API key and the team-scoped enforcement URL.

    ``None`` when the SDK is not configured to read enforcement at all, which
    every caller treats as "no reading available" rather than as an error.
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
    return _FetchTarget(api_key, f"{base_url}/v2/api/ai/enforcement-rules/{safe_team_id}")


def _get_enforcement(url: str, api_key: str, resource: str,
                     params: Optional[dict] = None) -> Optional["httpx.Response"]:
    """GET one enforcement resource under the shared retry and Retry-After budget.

    Returns the first response the retry policy does not retry -- the caller
    reads its status and its body -- or ``None`` once the fetch has given up,
    so every caller fails open on what it already holds. ``resource`` names
    the read in the log lines.

    Every enforcement read goes through here rather than calling ``httpx.get``
    itself: the Retry-After budget and the refresh cooldown (FRONT-1682) exist
    because these reads run on a caller's own request thread, and a second
    unguarded GET against the same throttled origin reintroduces the request
    amplification that ticket fixed.

    ``params`` of ``None`` sends the request httpx builds from the URL alone,
    which is what the unfiltered team-wide read has always sent.
    """
    # Server-requested wait time this call may still spend, in total across
    # every attempt. See _RETRY_AFTER_GIVE_UP_SECONDS.
    retry_after_budget = _RETRY_AFTER_GIVE_UP_SECONDS
    for attempt in range(_FETCH_MAX_ATTEMPTS):
        backoff = min(_FETCH_BACKOFF_INITIAL * (2 ** attempt), _FETCH_BACKOFF_CAP)
        try:
            response = httpx.get(url, params=params,
                                 headers={"x-api-key": api_key}, timeout=10)
        except (httpx.TimeoutException, httpx.TransportError):
            if attempt == _FETCH_MAX_ATTEMPTS - 1:
                break
            if _sleep(backoff):
                break
            continue
        except Exception:
            logger.debug("Failed to fetch enforcement %s, falling open", resource, exc_info=True)
            return None

        if not _is_retryable_status(response.status_code):
            return response

        plan = _plan_retry(response, backoff, retry_after_budget)
        retry_after_budget = plan.budget_remaining
        if plan.wait_seconds is None:
            # Give up without waiting and without retrying: the cooldown
            # _plan_retry recorded keeps the server's interval, and the caller
            # falls open on the reading it already has.
            return None
        if attempt == _FETCH_MAX_ATTEMPTS - 1:
            break
        if _sleep(plan.wait_seconds):
            break

    # Deliberate fail-open: an enforcement-refresh outage must never become a
    # customer traffic outage. The caller preserves what it has and the next
    # poll cycle tries again.
    logger.warning(
        "Enforcement %s fetch exhausted %d attempts; failing open",
        resource, _FETCH_MAX_ATTEMPTS,
    )
    return None


def _fetch_rules(rule_id: Optional[str] = None) -> Optional[_FetchedRules]:
    """Fetch the current enforcement payload from the Revenium API.

    Returns the rules and the three department-budget maps on success — any of
    them may be empty -- or ``None`` on failure so the caller can preserve the
    previous cache.

    ``rule_id`` sends the server's ``ruleId`` filter and is opt-in: the poller
    and the stale-cache refresh never send it. The server computes the
    department-budget maps team-wide and attaches them to the whole-team read,
    so a refresh narrowed to one rule would stop receiving them and department
    budgets would silently stop blocking anybody (BACK-3066). Narrowing is for
    an integrator inspecting one rule, never for the cache the pre-call path
    reads.
    """
    target = _fetch_target()
    if target is None:
        return None

    params = {"ruleId": rule_id} if rule_id else None
    response = _get_enforcement(target.team_url, target.api_key, "rule", params)
    if response is None:
        return None

    try:
        # 204 No Content == no rules configured for this team; cache empty
        if response.status_code == 204:
            return _FetchedRules([], {}, {}, {})
        response.raise_for_status()
        # Server currently returns ``{"rules": [...], "compiledAt": ...,
        # "orgUnitBudgetBlocks": {...}, "orgUnitBudgetBlockBalances":
        # {...}, "orgUnitBudgetWarnings": {...}}``; a bare list is accepted
        # too. See ``_fetched_from_payload``.
        return _fetched_from_payload(response.json())
    except Exception:
        logger.debug("Failed to fetch enforcement rules, falling open", exc_info=True)
        return None


def _fetch_roster(rule_id: str, page: int = 0, size: int = 25,
                  search: Optional[str] = None,
                  band: Optional[str] = None) -> Optional[dict]:
    """Fetch one page of the roster the server holds for a single rule.

    Deliberately not cached, unlike ``_cached_rules``: the roster is a
    point-in-time reading an integrator asks for while debugging, nothing on
    the pre-call path reads it, and a cached copy would carry a second
    lifetime that could outlive the rule it describes.

    It spends the same Retry-After budget as the rule fetch and records the
    same cooldown when it gives up, but unlike ``_refresh_cache`` a cooldown
    already in force does not suppress it: this is one read a person asked
    for, not one per customer request, and answering it with ``None`` because
    the poller was throttled would hide the reading rather than protect the
    origin.

    Returns the server's roster object, or ``None`` when there is no reading
    yet (HTTP 204), the team has no such rule, or the fetch gave up.
    """
    target = _fetch_target()
    if target is None:
        return None

    params: Dict[str, object] = {"ruleId": rule_id, "page": page, "size": size}
    if search:
        params["search"] = search
    if band:
        params["band"] = band

    response = _get_enforcement(f"{target.team_url}/roster", target.api_key,
                                "roster", params)
    if response is None:
        return None

    try:
        # 204 No Content == the rule has no compiled reading yet, which is a
        # different answer from the rules read's 204 (see the controller): it
        # carries no envelope to say how old the reading is.
        if response.status_code == 204:
            return None
        response.raise_for_status()
        payload = response.json()
    except Exception:
        logger.debug("Failed to fetch enforcement rule roster, falling open", exc_info=True)
        return None

    if not isinstance(payload, dict):
        logger.warning("Unexpected enforcement roster response shape: %r",
                       type(payload).__name__)
        return None
    return payload


def fetch_enforcement_rule(rule_id: str) -> Optional[dict]:
    """Read one enforcement rule from the server, bypassing the rule cache.

    An inspection call for integrators. It is not on the pre-call path, it
    neither reads nor writes the cache ``check_enforcement`` evaluates, and it
    is the only place the SDK asks the server for a single rule.

    Args:
        rule_id: Revenium hashid of the rule to read.

    Returns:
        The rule as the server compiled it, or ``None`` when the team has no
        such compiled rule (a disabled rule is never compiled) or the read
        could not be completed.

    Raises:
        ValueError: If ``rule_id`` is empty.
    """
    if not rule_id:
        raise ValueError("rule_id is required to read a single enforcement rule")

    fetched = _fetch_rules(rule_id)
    if fetched is None or not fetched.rules:
        return None
    return fetched.rules[0]


def fetch_enforcement_rule_roster(rule_id: str, page: int = 0, size: int = 25,
                                  search: Optional[str] = None,
                                  band: Optional[str] = None) -> Optional[dict]:
    """Read who one enforcement rule currently covers.

    An inspection call for integrators answering "why was this caller blocked,
    and who else is this rule measuring?". Like ``fetch_enforcement_rule`` it
    is off the pre-call path and is never cached, so it always reports what
    the server holds right now.

    Args:
        rule_id: Revenium hashid of the rule whose roster to read.
        page: Zero-based page index.
        size: Rows per page, 1 to 200.
        search: Case-insensitive substring over a row's key, label and email.
        band: One of ``BLOCKED``, ``WARNED``, ``UNDER`` or ``ALL``.

    Returns:
        The server's roster object -- ``rows`` plus the rule's window,
        threshold and the whole roster's ``blockedCount`` / ``warnedCount`` /
        ``underCount`` -- or ``None`` when the rule has no reading yet, groups
        on nothing, or the read could not be completed.

    Raises:
        ValueError: If ``rule_id`` is empty.
    """
    if not rule_id:
        raise ValueError("rule_id is required to read an enforcement rule roster")

    return _fetch_roster(rule_id, page=page, size=size, search=search, band=band)


def _refresh_cache() -> None:
    """Refresh the in-memory rule cache.

    Only advances ``_cache_timestamp`` on a successful fetch — a transient
    network error must not poison the stale-cache trigger and silently
    suppress retries for the next ``_CACHE_TTL`` window. Disk persistence
    happens outside ``_cache_lock`` so a slow filesystem write can't block
    concurrent ``check_enforcement`` callers on the pre-call path.

    Issues nothing at all while a Retry-After cooldown is in force, so a
    throttled tenant sends one control-plane request rather than one per
    customer request. Skipping is free here: the cached rules stay in force
    and no caller waits for the interval to elapse.
    """
    global _cached_rules, _cached_org_unit_blocks, _cached_org_unit_block_balances
    global _cached_org_unit_warnings
    global _cache_timestamp, _cache_initialized
    if _refresh_is_on_cooldown():
        logger.debug("Enforcement refresh skipped: Retry-After cooldown still in force")
        return
    fetched = _fetch_rules()
    if fetched is None:
        return
    with _cache_lock:
        _cached_rules = fetched.rules
        _cached_org_unit_blocks = fetched.org_unit_blocks
        _cached_org_unit_block_balances = fetched.org_unit_block_balances
        _cached_org_unit_warnings = fetched.org_unit_warnings
        _begin_cache_generation()
        _cache_timestamp = time.monotonic()
        _cache_initialized = True
    _persist_cache_to_disk(fetched.rules, fetched.org_unit_blocks,
                           fetched.org_unit_block_balances,
                           fetched.org_unit_warnings)


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


def _snapshot_department_budgets() -> _DepartmentBudgets:
    """Copy the cached department maps. Caller must hold ``_cache_lock``."""
    blocks = _cached_org_unit_blocks if isinstance(_cached_org_unit_blocks, dict) else {}
    balances = (_cached_org_unit_block_balances
                if isinstance(_cached_org_unit_block_balances, dict) else {})
    warnings = (_cached_org_unit_warnings
                if isinstance(_cached_org_unit_warnings, dict) else {})
    return _DepartmentBudgets(dict(blocks), dict(balances), dict(warnings),
                              _cache_generation)


def _get_rules() -> Tuple[list, _DepartmentBudgets, bool]:
    """Return cached rules, the department maps, and the initialized flag.

    Reading every field under one ``_cache_lock`` acquisition prevents the
    fail-closed path from torn-reading ``_cache_initialized`` against a
    half-written ``_cached_rules`` from the background poller, and keeps the
    department maps consistent with each other and with the rules they
    reference.
    """
    now = time.monotonic()
    with _cache_lock:
        age = now - _cache_timestamp
        rules = list(_cached_rules)
        departments = _snapshot_department_budgets()
        initialized = _cache_initialized
    if age > _CACHE_TTL:
        if _refresh_lock.acquire(blocking=False):
            try:
                _refresh_cache()
                with _cache_lock:
                    rules = list(_cached_rules)
                    departments = _snapshot_department_budgets()
                    initialized = _cache_initialized
            finally:
                _refresh_lock.release()
    return rules, departments, initialized


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


def _normalize_email(email: str) -> str:
    """The one canonical form of an address, as the server writes its map keys.

    ``EmailNormalizer.normalize`` on the platform side is ``trim().lowercase()``
    and every department-budget map is keyed by that form, because it is the
    only identity org-unit attribution can follow. A client looking a caller up
    has to agree byte for byte: comparing the address exactly as the
    application supplied it missed the block for anyone whose call carried
    "Dev@Example.com" or " dev@example.com ", and the paid provider call went
    out anyway.

    ``lower()`` deliberately, not ``casefold()``: Kotlin's ``lowercase()`` is
    locale-independent simple case mapping, so casefold's extra foldings
    (ß -> ss) would build a key the server never wrote.
    """
    return email.strip().lower()


def _caller_emails(usage_metadata: Optional[dict]) -> List[str]:
    """The caller's normalized subscriber email(s), nested-first then flat.

    Deliberately email-only: ``orgUnitBudgetBlocks`` is keyed by subscriber
    email because that is the identity the server can resolve to an org-unit
    membership. A caller who supplies no email simply has no key, and — as
    with ``_matching_group_entry`` — no sentinel is invented for them: a
    guessed key could block someone against a department that is not theirs.

    Normalization happens here, once, so every department-budget lookup (the
    block map and the balance map alike) agrees on the caller's key -- and it
    meets keys that ``_normalize_map_keys`` has already put in the same form.
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
            # Normalization can only empty an address that was whitespace
            # alone, which is no identity; the nested key still suppresses the
            # flat one in that case, so precedence is unchanged either way.
            normalized = _normalize_email(candidate)
            return [normalized] if normalized else []
    return []


def _org_unit_map_entry(department_map: dict,
                        usage_metadata: Optional[dict]) -> Optional[Tuple[str, int]]:
    """Look the caller up in a department map; return ``(caller key, rule id)``.

    Serves both maps -- ``blocks``, where the hit is the blocking verdict, and
    ``warnings``, where it is the rule whose warn tier the caller crossed --
    so the two cannot drift apart in what they accept. ``None`` means no hit:
    no email on the caller, no entry for it, or a value that is not a rule id
    (both maps are documented as email -> int, so anything else is treated as
    absent rather than as a verdict).

    The caller key comes back with the id because the warn path needs it to
    warn each person once (``_warn_org_unit_threshold``) while the block path
    needs only the id.

    A plain lookup, deliberately: both sides of it are already in the one
    canonical form -- the map was re-keyed by ``_normalize_map_keys`` when it
    came in, and ``_caller_emails`` normalizes the caller -- so the ordinary
    case, a caller in neither map, costs one dict miss rather than a walk of
    the map on the pre-provider path.
    """
    if not isinstance(department_map, dict) or not department_map:
        return None
    for email in _caller_emails(usage_metadata):
        rule_id = department_map.get(email)
        # bool is an int subclass; a JSON ``true`` is not a rule id.
        if isinstance(rule_id, int) and not isinstance(rule_id, bool):
            return email, rule_id
    return None


def _org_unit_block_rule_id(org_unit_blocks: dict, usage_metadata: Optional[dict]) -> Optional[int]:
    """The rule id blocking this caller under a department budget, else ``None``."""
    entry = _org_unit_map_entry(org_unit_blocks, usage_metadata)
    return entry[1] if entry is not None else None


def _caller_block_balance(balances: dict, usage_metadata: Optional[dict]) -> Optional[float]:
    """The balance the server judged *this* caller against, when it published one.

    Mirrors the server's own violation consumer, which reads
    ``orgUnitBudgetBlockBalances[email] ?: rule.currentValue``. For a per-person
    cap scoped to one department the rule's own ``currentValue`` is the highest
    single balance in that department, so reporting it hands the blocked
    developer somebody else's spend.

    ``None`` means the payload has no usable entry for this caller — an
    ancestor cap, whose own ``currentValue`` already *is* the scope's spend; a
    server predating the map; or a value that will not coerce to a number — and
    the caller falls back to the rule's value exactly as before.
    """
    if not isinstance(balances, dict) or not balances:
        return None
    for email in _caller_emails(usage_metadata):
        value = balances.get(email)
        # Coerced again here, not only at the ingestion boundary: the cache can
        # also be seeded directly (by a test, or by a future caller), and a bad
        # value must fall back rather than reach BudgetExceededError as a
        # string.
        balance = None if isinstance(value, bool) else _coerce_float(value)
        # Same finiteness rule as _coerce_balances: a NaN or Infinity seeded
        # straight into the cache is not a dollar figure either.
        if balance is not None and math.isfinite(balance):
            return balance
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


def _raise_org_unit_block(rules: list, departments: _DepartmentBudgets,
                          usage_metadata: Optional[dict]) -> None:
    """Raise ``BudgetExceededError`` when the department map blocks the caller.

    Mirrors the server's own consumer of ``orgUnitBudgetBlocks``: the map is
    the verdict, and the rule is looked up only to describe it. A map hit
    naming a rule that is not in the cached list — a stale or racing payload
    — still blocks, under a generic name, because the server publishes the
    map only for departments it has already decided are over budget.

    ``current_value`` is the caller's own balance whenever the payload carries
    one (see ``_caller_block_balance``) and the rule's own value otherwise.
    """
    rule_id = _org_unit_block_rule_id(departments.blocks, usage_metadata)
    if rule_id is None:
        return

    own_balance = _caller_block_balance(departments.balances, usage_metadata)
    rule = _rule_by_id(rules, rule_id)
    if rule is None:
        raise BudgetExceededError(
            message=(
                "Request blocked by Revenium enforcement rule: "
                f"{_ORG_UNIT_FALLBACK_RULE_NAME}"
            ),
            rule_name=_ORG_UNIT_FALLBACK_RULE_NAME,
            current_value=own_balance,
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
        current_value=(own_balance if own_balance is not None
                       else _coerce_float(rule.get("currentValue"))),
        threshold=_coerce_float(rule.get("threshold")),
        resets_at=rule.get("resetsAt"),
        rule_id=rule.get("ruleId") or rule.get("id") or rule_id,
    )


def _begin_cache_generation() -> None:
    """Start a new cache generation and forget what the old one had warned.

    Called wherever the department maps are replaced -- a successful fetch, a
    disk snapshot load -- so a caller who is still over their warn tier is told
    again on the refreshed verdict, and is told once in between. The bump
    happens under the ``_cache_lock`` both callers already hold while they swap
    the maps, which is why this must not take that lock itself (it is not
    reentrant); the set has its own.

    Clearing the set only bounds it. What keeps a stale claim from silencing
    this generation is that the generation is part of every key.
    """
    global _cache_generation
    _cache_generation += 1
    with _warn_dedupe_lock:
        _warned_org_unit_callers.clear()


def _claim_org_unit_warning(generation: int, caller_key: str, rule_id: int) -> bool:
    """True the first time this caller is warned about this rule, else False.

    ``generation`` is the one the claiming check's own snapshot carried, never
    whatever is current now: a check descheduled across a refresh must not be
    able to mark the refreshed verdict as already stated, and must still be
    deduplicated against its own.

    The claim is taken before the line is emitted, so two threads racing the
    same caller through the pre-provider hook produce one warning, not two.
    """
    key = (generation, caller_key, rule_id)
    with _warn_dedupe_lock:
        if key in _warned_org_unit_callers:
            return False
        _warned_org_unit_callers.add(key)
        return True


def _warning_spend_clause(balance: Optional[float], threshold: Optional[float]) -> str:
    """The ``812.50 of 1000.00 spent`` part of the line, when the payload has it.

    Empty when the server published no balance for this caller (an ancestor
    cap, or a server predating the map): the rule's own ``currentValue`` is the
    department's highest single balance, not this person's, so quoting it would
    tell the warned developer somebody else's spend -- the exact defect
    BACK-3066 fixed on the block error.
    """
    if balance is None:
        return ""
    if threshold is None:
        return f" ({balance:.2f} spent)"
    return f" ({balance:.2f} of {threshold:.2f} spent)"


def _warn_org_unit_threshold(rules: list, departments: _DepartmentBudgets,
                             usage_metadata: Optional[dict]) -> None:
    """Log once that this caller has crossed a department budget's warn tier.

    The non-blocking surface chosen in BACK-3077 (see the module docstring):
    the warn map is not a verdict, so this never raises and never blocks. The
    rule is looked up only to name it, and a map hit naming a rule the cached
    rules do not carry -- a stale or racing payload -- is still worth stating
    under its id.

    The line names no address: these maps are PII, and the process emitting the
    line is the warned caller's own.
    """
    entry = _org_unit_map_entry(departments.warnings, usage_metadata)
    if entry is None:
        return
    caller_key, rule_id = entry
    if not _claim_org_unit_warning(departments.generation, caller_key, rule_id):
        return

    rule = _rule_by_id(rules, rule_id)
    rule_name = (rule.get("name") if isinstance(rule, dict) else None) or str(rule_id)
    threshold = _coerce_float(rule.get("threshold")) if isinstance(rule, dict) else None
    clause = _warning_spend_clause(
        _caller_block_balance(departments.balances, usage_metadata), threshold,
    )
    logger.warning(
        "Approaching the Revenium department budget enforced by rule %s%s; "
        "calls are blocked once the threshold is crossed",
        rule_name, clause,
    )


def _check_org_unit_block(rules: list, departments: _DepartmentBudgets,
                          usage_metadata: Optional[dict]) -> None:
    """``_raise_org_unit_block`` with a fail-open guard, then the warn tier.

    The pre-provider hook may raise ``BudgetExceededError`` and nothing else,
    so a malformed map degrades to "no block" instead of taking the caller's
    request down with it -- and the warn tier, which cannot block at all, runs
    under the same guard.

    The warning is emitted only for a caller the block decision let through:
    the server keeps the two maps disjoint, and a person who is already blocked
    has no use for the news that they were nearly blocked.
    """
    try:
        _raise_org_unit_block(rules, departments, usage_metadata)
        _warn_org_unit_threshold(rules, departments, usage_metadata)
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
    per-rule loop, from the server's ``orgUnitBudgetBlocks`` email map, looked
    up under the normalized form of the caller's address; see
    ``_raise_org_unit_block``. A caller the server has only *warned* about
    (``orgUnitBudgetWarnings``) is never blocked and never raised on: they get
    one log line per refreshed verdict, so the block is not the first thing
    they hear about their department budget. See ``_warn_org_unit_threshold``.

    Raises:
        BudgetExceededError: when a cost-limit rule blocks the call.
            All structured fields (``rule_name``, ``current_value``,
            ``threshold``, ``resets_at``, ``rule_id``) are populated when the
            server provides them. For a grouped rule ``current_value`` is the
            caller's group balance while ``threshold`` stays rule-level; for a
            department budget it is the caller's own balance from
            ``orgUnitBudgetBlockBalances`` when the server publishes one.
    """
    if is_bypass_enabled():
        return
    if not is_circuit_breaker_enabled():
        return

    _load_cache_from_disk()
    _ensure_poller_running()
    rules, departments, initialized = _get_rules()

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
    # A caller who is only near the cap is warned here, never blocked.
    _check_org_unit_block(rules, departments, usage_metadata)

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
