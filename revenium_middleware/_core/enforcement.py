"""
Enforcement engine for the Revenium circuit breaker.

Polls cost-limit rules from the Revenium API in a daemon thread and caches
them in memory. ``check_enforcement(...)`` is a pre-call hook that raises
``BudgetExceededError`` when a tripped rule matches the current
request, blocking the outbound provider call before any spend occurs.

The pre-call hook never talks to the network (BACK-3917). It judges the
request against whatever rules are cached and, when they are stale, asks the
poller thread to refresh them; it never refreshes them itself. The hook runs
on the caller's thread, and for a LiteLLM proxy or an ``AsyncOpenAI`` caller
that thread is the event loop serving every request, so an inline refresh --
five attempts and their backoff sleeps, about 58 s against a hung endpoint --
froze all traffic, not just the one request. A refresh that fails is not
retried for one poll interval (``_refresh_failure_backoff_seconds``), so an
outage costs the control plane what the healthy poller costs and request
traffic adds nothing. Before the first fetch has landed nothing is cached:
the default fail-open mode lets the call through and
``REVENIUM_CB_FAIL_MODE=closed`` refuses it, both without waiting.

Department budgets are decided by the server: the rules response carries a
top-level ``departmentBudgetBlocks`` map of subscriber email -> blocking rule
id, pre-computed from department membership, a
``departmentBudgetBlockBalances`` map giving the balance each of those people
was actually judged against, and a ``departmentBudgetWarnings`` map of the
people who have crossed a warn tier without being blocked (a platform predating
BACK-3448 publishes them as ``orgUnitBudget*``, see ``_wire_value``). All three
are keyed by the *normalized* address (trimmed, lower-cased), so the SDK
re-keys them by that form as they come in and normalizes the caller's address
the same way before looking it up; it never resolves department identity
itself.

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

Three inspection calls sit beside the enforcement path without being part of
it. ``fetch_enforcement_rule(rule_id)`` reads one rule straight from the
server, ``fetch_enforcement_rule_roster(rule_id)`` reads the people (or
departments) that rule is measuring, each with their spend, cap and band, and
``fetch_advisory_budgets()`` reads the team's notify-only budgets (the
advisory rows the team-wide read serves only when asked). None is cached and
none is consulted by ``check_enforcement``: they answer "why was this caller
blocked, who else does this rule cover, and which advisory budgets exist?" for
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
import re
import threading
import time
from email.utils import parsedate_to_datetime
from typing import Dict, List, NamedTuple, Optional, Set, Tuple
from urllib.parse import quote, urlparse

import httpx

from .call_ownership import peek_callback_metadata
from .closable_scope import ClosableScope
from .config import Config, env_choice, env_flag_enabled
from .context import merge_metadata, overlay_metadata
from .exceptions import BudgetExceededError
from .fields import extract_common_metadata, extract_organization_name, extract_product_name
from .log_sanitize import sanitize_for_logging
from .subscriber import extract_subscriber_from_metadata, subscriber_with_credential_alias
from .trace_fields import get_credential_alias
from .vendor_model_aliases import SERVED_MODEL_BY_ALIAS
from .wire_guards import stored_subscriber_value

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

# Give-up thresholds for *server-supplied* ``Retry-After`` waits, deliberately
# NOT _FETCH_BACKOFF_CAP. Each is a budget for one whole read, not a
# per-response limit: a response is honoured only if its interval fits
# entirely in what is left, so a server repeating ``Retry-After: 20`` across
# five attempts can park a reader for 20 s in total rather than 100 s.
#
# Inside the budget the fetch waits the whole interval that was asked for;
# outside it the fetch neither waits nor retries, records a cooldown (see
# ``_begin_refresh_cooldown``) and fails open on the cached rules. A bound on
# a delay the server chose is a give-up threshold, never a ``min()`` on the
# wait -- truncating it would retry sooner than the 429 permitted, which is
# the one thing the status code asks a client not to do (RFC 9110 §10.2.3).
#
# The number follows the path shape. The rules refresh runs only on the
# poller thread, where a wait parks nobody, so it gets the 60 s the metering
# client honours (``_parse_retry_after_header`` in
# ``_metering/_base_client.py``). The three inspection reads,
# ``fetch_enforcement_rule``, ``fetch_enforcement_rule_roster`` and
# ``fetch_advisory_budgets``, still run on their caller's thread and keep 20 s. Cross-SDK numbers:
# ``docs/conventions/retry-after.md``.
_REFRESH_RETRY_AFTER_GIVE_UP_SECONDS = 60.0
_INSPECTION_RETRY_AFTER_GIVE_UP_SECONDS = 20.0

_ADVISORY_INCLUDE = "advisory"
_ADVISORY_SOURCE = "ANOMALY"

# Longest a single Retry-After may suppress rule refreshes. It applies to the
# server-supplied interval only; the backoff after a failed refresh is the SDK's
# own poll interval and is never clamped by it. This is NOT the
# ``min()`` on a wait that rule 1 forbids: nothing is parked and no retry is
# issued early inside the cooldown window -- the fetch has already given up,
# every caller is served from cache, and no request goes out until the window
# ends. All the ceiling bounds is how long one absurd or malformed header
# ("Retry-After: 999999") can keep the SDK from refreshing at all, so cached
# limits cannot go stale for hours on the strength of a single response. Any
# realistic value (60 s, 120 s, 300 s) is honoured exactly.
_REFRESH_COOLDOWN_CEILING_SECONDS = 300.0
# Snapshot file names are ``<prefix>.<scope>.json``, where the scope names the
# team and the environment the rules were read for (``_snapshot_scope``). Two
# teams' services on one host often share one REVENIUM_CACHE_DIR, and an
# unscoped name let a restarted process enforce another team's rules and hold
# that team's people's email addresses. The legacy unscoped files are never
# read and never deleted: an older SDK still running beside this one may own
# them.
_RULES_CACHE_FILE_PREFIX = "revenium_enforcement_rules"
# Department-budget map snapshot. A SEPARATE file, and not a new shape for the
# rules snapshot, on purpose: an older SDK's loader accepts only a bare list,
# so changing the rules file's shape would make a rollback silently discard
# the whole cache (fail-closed would then block everything during an API
# outage). The old loader never looks at this filename, so it can carry the
# new data without being load-bearing for a downgrade.
_ORG_UNIT_BLOCKS_CACHE_FILE_PREFIX = "revenium_enforcement_org_unit_blocks"

# Top-level keys of the server-computed department-budget map on the API
# response, current spelling first (see ``_wire_value``).
_ORG_UNIT_BLOCKS_KEYS = ("departmentBudgetBlocks", "orgUnitBudgetBlocks")
# Top-level key of the per-person balance map published beside the block map:
# normalized email -> the balance that person's threshold was compared against,
# in dollars. Absent on a server predating it, which is why every read of it
# falls back to the rule's own ``currentValue``.
_ORG_UNIT_BLOCK_BALANCES_KEYS = ("departmentBudgetBlockBalances", "orgUnitBudgetBlockBalances")
# Top-level key of the warn-tier map published beside them: normalized email ->
# the rule whose warn tier that person crossed, for a per-person cap scoped to
# one department. Disjoint from _ORG_UNIT_BLOCKS_KEYS (a person already blocked
# is not warned) and never a block verdict -- see the module docstring's
# BACK-3077 decision. Absent on a server predating it, which warns nobody.
_ORG_UNIT_WARNINGS_KEYS = ("departmentBudgetWarnings", "orgUnitBudgetWarnings")
# Top-level key of the per-person candidate lists: normalized email -> every
# department cap that could apply to that person, best first, each as
# ``{ruleId, tier, departmentId, balance}``. Published only for people with at
# least one candidate whose rule has a call-scope filter row; for them it
# replaces the three single-slot maps above, because one slot cannot say
# whether a second cap, matching a call the first does not cover, is hidden.
# It has no org-unit spelling: it postdates the rename.
_ORG_UNIT_CANDIDATES_KEYS = ("departmentBudgetCandidates",)
_CANDIDATE_TIERS = frozenset({"BLOCK", "WARN"})
# Filter rows of a candidate's rule that the server already decided when it
# put the candidate on the person's list, so the SDK never judges them against
# the call: a SUBSCRIBER row names whoever the cap was written for, not
# necessarily the caller it was resolved to.
_CANDIDATE_DECIDED_DIMENSIONS = frozenset({"DEPARTMENT", "ORG_UNIT", "SUBSCRIBER"})
# ``groupBy`` values of a department rule. Such rules are skipped by the
# per-rule loop: their verdict comes only from _ORG_UNIT_BLOCKS_KEYS, matching
# the server's own ``applicableRules`` filter.
_ORG_UNIT_GROUP_BY_VALUES = frozenset({"DEPARTMENT", "ORG_UNIT"})
# Rule field naming the department an ancestor-cap rule is scoped to.
_ORG_UNIT_ID_KEYS = ("departmentId", "orgUnitId")
# Reported when the email map names a rule id that is absent from the cached
# rules (a stale or racing payload). The server deliberately fails toward
# enforcing here, so the block still happens — just without a rule name.
_ORG_UNIT_FALLBACK_RULE_NAME = "Department budget"
# A rule's ``windowEnd`` once any ``[Zone]`` suffix is cut and a trailing ``Z``
# is spelled ``+00:00``: the server writes the last nanosecond of the window,
# e.g. ``2026-10-31T23:59:59.999999999Z``. Python 3.10's ``fromisoformat``
# accepts neither ``Z`` nor nine fraction digits, so the fraction is cut to
# the six it does accept before parsing.
_WINDOW_END_PATTERN = re.compile(
    r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})(?:\.(\d+))?([+-]\d{2}:\d{2})?"
)

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
# Server-computed subscriber-email -> candidate list map (see
# _ORG_UNIT_CANDIDATES_KEYS), cached, persisted and re-keyed like the others,
# each list already cleaned by _usable_candidates.
_cached_org_unit_candidates: Dict[str, List[dict]] = {}
_cache_lock = threading.Lock()
_cache_timestamp = 0.0
# The (enforcement base URL, team id) the cached rules were loaded for, the
# scope the saved snapshot is kept under too. None until a call, a fetch or a
# snapshot load adopts the current one. Read and written under _cache_lock.
_cache_owner: Optional[Tuple[str, str]] = None
# True once any successful fetch (even an empty list / HTTP 204) or a disk
# snapshot load has populated the cache. Distinguishes "server says no rules
# apply" from "we have never heard back from the server" — fail-closed must
# only block in the latter case.
_cache_initialized = False

_poll_thread: Optional[threading.Thread] = None
_poll_lock = threading.Lock()
_stop_event = threading.Event()

# Held for the whole of a refresh, so at most one is in flight per process
# however many threads ask for one; a refresh that finds it held is dropped,
# not queued.
_refresh_lock = threading.Lock()

# Set to wake the poller early: by a pre-call check that found the cache stale,
# and by ``stop_polling`` so the poller does not sleep out its interval.
_poll_wakeup = threading.Event()


def _forget_parent_refresh() -> None:
    """Give a forked child a free ``_refresh_lock``.

    A refresh in flight in the parent at fork time leaves the child a copy of
    the lock that is held by a poller thread the child does not have, so every
    refresh in the child would be skipped and its rules would never load.
    """
    global _refresh_lock
    _refresh_lock = threading.Lock()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_forget_parent_refresh)

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

# Rule id -> the ``windowEnd`` of the breach last warned about for a breached
# rule whose action does not block. An entry for a rule the refreshed cache
# still serves survives _begin_cache_generation: a refresh every poll interval
# would otherwise repeat the line once a minute for the rest of the window. An
# entry for a rule no longer served is dropped there, so the map is bounded by
# the current rule set. A new window brings a new ``windowEnd`` and so a new
# line. Guarded by _warn_dedupe_lock and mutated in place so a test can clear
# it.
_warned_non_blocking_rules: Dict[str, object] = {}

# (rule id, dimension, operator, reason) for every rule filter the SDK has
# logged that it cannot evaluate, so the debug line appears once rather than
# on every call. Guarded by _warn_dedupe_lock and mutated in place so a test
# can clear it.
_noted_unevaluable_filters: Set[Tuple[str, str, str, str]] = set()

# id(groupBreakdown list) -> (that list, {groupValue: entry}), so a per-person
# lookup is a dict hit instead of a walk of every person on the team on every
# call. One slot per list, so several per-person caps never evict each other.
# The list is held beside its index so its id cannot be reused by another list
# while the slot lives. Built lazily, outside any lock: two threads building
# the same index is harmless, and a published index is never mutated. Cleared
# by _begin_cache_generation, which every refresh and disk load calls, and
# published only under _cache_lock while the building check's generation is
# still current, so a check that read the rules before a refresh cannot pin
# the replaced lists here. Kept out of the rule dicts on purpose: those are
# persisted and fingerprinted.
_group_indexes: Dict[int, Tuple[list, Dict[str, dict]]] = {}

# Monotonic deadline before which no refresh may issue a request, set when the
# fetch declines to wait out a Retry-After (``_begin_refresh_cooldown``) and
# when a refresh fails for any other reason (``_begin_failure_backoff``). 0.0
# means no cooldown in force. Read and written under _cache_lock.
_refresh_cooldown_until = 0.0
# The owner (see _cache_owner) whose fetch set _refresh_cooldown_until, so a
# switch to another team drops it. Read and written under _cache_lock.
_refresh_cooldown_owner: Optional[Tuple[str, str]] = None

# Single-shot warnings so misconfigured environments don't spam logs
_team_id_warned = False
_api_key_warned = False
_disk_load_attempted = False


def is_circuit_breaker_enabled() -> bool:
    """Return True when the operator has opted in to enforcement."""
    return env_flag_enabled(Config.ENV_CIRCUIT_BREAKER_ENABLED)


def is_bypass_enabled() -> bool:
    """``REVENIUM_BYPASS=true`` short-circuits enforcement at every callsite."""
    return env_flag_enabled(Config.ENV_REVENIUM_BYPASS)


_outer_wrapper_check = ClosableScope("revenium_call_checked_by_outer_wrapper")


def checked_by_outer_wrapper():
    """Mark the provider call made inside this scope as already checked.

    A wrapper that calls through another metered client (LiteLLM over the
    OpenAI SDK) checks the call with the identity it resolved, then makes the
    inner call without it. The inner wrapper would check again with only the
    decorator's identity and could refuse a call the outer check allowed.

    The mark is shared with every context copied inside the scope and closed
    when the scope ends, so a task the call starts (an async LiteLLM success
    callback) that makes its own call after the outer call returned is checked."""
    return _outer_wrapper_check.entered()


def _checked_by_outer_wrapper() -> bool:
    return _outer_wrapper_check.is_active()


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


def _refresh_failure_backoff_seconds() -> float:
    """How long a failed refresh suppresses the next one.

    One poll interval: during an outage the SDK then asks the control plane
    exactly as often as it does when healthy, request traffic adds no
    attempts, and a recovered endpoint is noticed within the same interval a
    healthy cache is refreshed in. A 403 from a wrong key or team id is a
    misconfiguration that only an operator can fix, and one attempt a minute
    picks up the fix without flooding the log.
    """
    return float(_poll_interval_seconds())


def _fail_mode_is_closed() -> bool:
    """``REVENIUM_CB_FAIL_MODE=closed`` raises when no usable cache exists."""
    return env_choice(Config.ENV_REVENIUM_CB_FAIL_MODE, ("open", "closed"), "open") == "closed"


def _current_owner() -> Tuple[str, str]:
    """The enforcement base URL and team id the cache is for: whose rules these are."""
    return _get_enforcement_base_url(), os.environ.get(Config.ENV_REVENIUM_TEAM_ID, "")


def _snapshot_scope(owner: Optional[Tuple[str, str]] = None) -> Optional[str]:
    """Short digest of the environment and team the cached rules belong to.

    ``None`` without ``REVENIUM_TEAM_ID``: no team means no fetch, so there is
    nothing of this process's own to save or load. Hashed so neither the team
    id nor any path character from it reaches a file name.
    """
    base_url, team_id = owner or _current_owner()
    if not team_id:
        return None
    return hashlib.sha256(f"{base_url}|{team_id}".encode("utf-8")).hexdigest()[:16]


def _scoped_cache_file_path(prefix: str, owner: Optional[Tuple[str, str]] = None) -> Optional[str]:
    cache_dir = os.environ.get(Config.ENV_REVENIUM_CACHE_DIR, "")
    scope = _snapshot_scope(owner)
    if not cache_dir or scope is None:
        return None
    return os.path.join(cache_dir, f"{prefix}.{scope}.json")


def _cache_file_path(owner: Optional[Tuple[str, str]] = None) -> Optional[str]:
    return _scoped_cache_file_path(_RULES_CACHE_FILE_PREFIX, owner)


def _org_unit_blocks_cache_file_path(owner: Optional[Tuple[str, str]] = None) -> Optional[str]:
    return _scoped_cache_file_path(_ORG_UNIT_BLOCKS_CACHE_FILE_PREFIX, owner)


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


def _process_metering_client():
    """The process-wide metering client, never a ``metering_key()`` block's: caps belong to the process."""
    # Imported lazily: the metering module does not import this one.
    from revenium_middleware._core.metering import process_client
    return process_client()


def _process_metering_url() -> Optional[str]:
    from revenium_middleware._core.metering import _valid_base_url
    client = _process_metering_client()
    return _valid_base_url(str(client.base_url)) if client is not None else None


def _get_enforcement_base_url() -> str:
    """Base URL for enforcement API calls.

    Prefers ``REVENIUM_ENFORCEMENT_BASE_URL`` so a context-path
    (``http://localhost:8080/profitstream``) survives intact. Falls back to
    the origin of the metering client's URL, set by ``initialize_metering()``
    or ``REVENIUM_METERING_BASE_URL``.
    """
    explicit = os.environ.get(Config.ENV_REVENIUM_ENFORCEMENT_BASE_URL, "")
    if explicit:
        return explicit.rstrip("/")
    metering_url = _process_metering_url() or os.environ.get(
        Config.ENV_REVENIUM_BASE_URL, "https://api.revenium.ai/meter/")
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
    only who hears a log line, never who is blocked. ``candidates`` is
    normalized subscriber email -> that person's department caps, best first;
    for a person it names, it is the verdict and the other three are not read
    (``_check_org_unit_block``).

    All four are keyed by ``_normalize_email`` because whoever built them
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
    candidates: dict
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


def _json_integer(value) -> Optional[int]:
    """``value`` when it is a JSON integer, else ``None``."""
    # bool is an int subclass; a JSON ``true`` is not an id.
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _usable_candidates(raw) -> List[dict]:
    """The well-formed items of one person's candidate list, in their order.

    An item is dropped, not guessed at, when it is not an object, its
    ``ruleId`` is not an integer or its ``tier`` is neither ``BLOCK`` nor
    ``WARN``; an unusable ``balance`` becomes ``None``, which reports the
    rule's own value, as an absent balance always has.
    """
    if not isinstance(raw, list):
        return []
    usable = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        rule_id, tier = _json_integer(item.get("ruleId")), item.get("tier")
        if rule_id is None:
            continue
        if not isinstance(tier, str) or tier.strip().upper() not in _CANDIDATE_TIERS:
            continue
        balance = None if isinstance(item.get("balance"), bool) else _coerce_float(item.get("balance"))
        usable.append({
            "ruleId": rule_id,
            "tier": tier.strip().upper(),
            "departmentId": _json_integer(item.get("departmentId")),
            "balance": balance if balance is not None and math.isfinite(balance) else None,
        })
    return usable


def _coerce_candidates(raw) -> Dict[str, List[dict]]:
    """Re-key the candidate map by normalized address and clean every list.

    A person whose list is empty once its malformed items are dropped is left
    out, so the lookup sends them down the single-slot path, exactly like a
    person the server published no list for.
    """
    candidates: Dict[str, List[dict]] = {}
    for email, items in _normalize_map_keys(raw).items():
        usable = _usable_candidates(items)
        if usable:
            candidates[email] = usable
    return candidates


def _load_department_budgets_from_disk(rules: list, owner: Optional[Tuple[str, str]] = None) -> _DepartmentBudgets:
    """Read the department-budget snapshot written beside ``rules``.

    The maps live in their own file; absent or malformed means no department
    blocks until the next successful fetch. The envelope binds them to the
    rules snapshot they were cut from. A fingerprint mismatch means the pair on
    disk is torn (e.g. the process died between the two writes): interpreting
    rule IDs against the wrong rules could block the wrong caller or fail open,
    so the maps are dropped instead -- department blocks resume on the next
    successful fetch.
    """
    empty = _DepartmentBudgets({}, {}, {}, {})
    blocks_path = _org_unit_blocks_cache_file_path(owner)
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
            # server had not normalized. ``balances``, ``warnings`` and
            # ``candidates`` are absent from an envelope written before each
            # map existed: the first reports the rule's own value as it always
            # did, the second warns nobody until the next successful fetch,
            # and the third leaves everyone on the single-slot verdict.
            return _DepartmentBudgets(
                _normalize_map_keys(loaded["blocks"]),
                _coerce_balances(loaded.get("balances")),
                _normalize_map_keys(loaded.get("warnings")),
                _coerce_candidates(loaded.get("candidates")),
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
    global _cached_org_unit_warnings, _cached_org_unit_candidates
    global _cache_timestamp, _disk_load_attempted, _cache_initialized, _cache_owner
    # Check-and-set under _cache_lock so two cold-start callers can't both
    # observe _disk_load_attempted=False and race on the snapshot read.
    with _cache_lock:
        if _disk_load_attempted:
            return
        _disk_load_attempted = True
    owner = _current_owner()
    path = _cache_file_path(owner)
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
        departments = _load_department_budgets_from_disk(rules, owner)
        with _cache_lock:
            if _cache_owner not in (None, owner):
                return
            _cache_owner = owner
            _cached_rules = rules
            _cached_org_unit_blocks = departments.blocks
            _cached_org_unit_block_balances = departments.balances
            _cached_org_unit_warnings = departments.warnings
            _cached_org_unit_candidates = departments.candidates
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
                           org_unit_warnings: Optional[dict] = None,
                           *, owner: Optional[Tuple[str, str]] = None,
                           org_unit_candidates: Optional[dict] = None) -> None:
    """Write the rules snapshot (legacy bare-list shape) and the maps beside it.

    The rules file keeps the exact shape every published SDK version reads —
    a bare JSON list — so a rollback after this version has written a
    snapshot still loads it instead of silently discarding the cache. The
    department map rides in its own file (see
    ``_ORG_UNIT_BLOCKS_CACHE_FILE_PREFIX``), written 0600 because it is keyed by
    subscriber email. The map's envelope carries the
    fingerprint of the rules it was computed against, so however a crash
    interleaves the two writes, the loader can only ever pair a map with the
    exact rules snapshot it belongs to — a mismatched map is dropped, which
    merely skips department blocks until the next refresh.

    The map arguments are optional so callers that only have rules (and the
    existing tests) keep working; an omitted map is written empty, which is
    also what an envelope predating that map loads as. ``owner`` is whose
    rules these are, the current owner when omitted.
    """
    path = _cache_file_path(owner)
    if not path:
        return
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        blocks_path = _org_unit_blocks_cache_file_path(owner)
        if blocks_path:
            _write_private_json(blocks_path, {
                "rules_fingerprint": _rules_fingerprint(rules),
                "blocks": org_unit_blocks or {},
                "balances": org_unit_block_balances or {},
                "warnings": org_unit_warnings or {},
                "candidates": org_unit_candidates or {},
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
    it against its Retry-After budget. ``None`` means the header was
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


def _begin_refresh_cooldown(seconds: float, owner: Tuple[str, str]) -> Optional[float]:
    """Stop issuing refresh requests for the ``Retry-After`` the fetch declined to wait.

    Giving up on the header is only half of honouring it. Without a cooldown
    the next customer request finds the same stale cache and asks for the same
    refresh: one control-plane request per customer request for as long as the
    throttling lasts. Recording the deadline makes nobody wait for it:
    refreshes are skipped and callers are served from the cached rules until
    it passes.

    Clamped to ``_REFRESH_COOLDOWN_CEILING_SECONDS`` -- a sanity bound on how
    long one header may suppress refreshes, not a truncated wait; see that
    constant. Returns the seconds remaining until the deadline now in effect,
    or ``None`` when ``owner`` no longer owns the cache.
    """
    return _extend_refresh_cooldown(min(max(0.0, seconds), _REFRESH_COOLDOWN_CEILING_SECONDS), owner)


def _begin_failure_backoff(owner: Tuple[str, str]) -> Optional[float]:
    """Stop issuing refresh requests for ``_refresh_failure_backoff_seconds``.

    Not clamped by ``_REFRESH_COOLDOWN_CEILING_SECONDS``: that bounds what one
    server header can impose, while this is the operator's own poll interval,
    so a one-hour interval means one attempt an hour during an outage too.
    Returns the seconds remaining until the deadline now in effect, or
    ``None`` when ``owner`` no longer owns the cache.
    """
    return _extend_refresh_cooldown(_refresh_failure_backoff_seconds(), owner)


def _extend_refresh_cooldown(seconds: float, owner: Tuple[str, str]) -> Optional[float]:
    """Move ``owner``'s cooldown deadline ``seconds`` out, never shortening one in force.

    A fetch that fails after the process switched to another team sets
    nothing and returns ``None``: its cooldown would hold off the new team's
    refresh for a whole poll interval.
    """
    global _refresh_cooldown_until, _refresh_cooldown_owner
    now = time.monotonic()
    deadline = now + max(0.0, seconds)
    with _cache_lock:
        if _cache_owner not in (None, owner):
            return None
        _refresh_cooldown_until = max(_refresh_cooldown_until, deadline)
        _refresh_cooldown_owner = owner
        return _refresh_cooldown_until - now


def _refresh_cooldown_remaining() -> float:
    """Seconds until a refresh may issue a request again; 0.0 when it may now."""
    with _cache_lock:
        return max(0.0, _refresh_cooldown_until - time.monotonic())


def _refresh_is_on_cooldown() -> bool:
    """True while a declined Retry-After or a failed refresh's backoff is running."""
    return _refresh_cooldown_remaining() > 0.0


class _RetryPlan(NamedTuple):
    """What to do about one retryable response, and what budget is left after.

    ``wait_seconds`` of ``None`` means stop: do not wait, do not retry, fail
    open on the cached rules.
    """

    wait_seconds: Optional[float]
    budget_remaining: float


def _plan_retry(response: "httpx.Response", backoff: float, budget_remaining: float,
                budget_total: float, owner: Tuple[str, str]) -> _RetryPlan:
    """Decide how to handle a retryable response, spending the Retry-After budget.

    Three outcomes, and the ``None`` is the one that matters:

    * no usable ``Retry-After`` -- our own ``backoff``, which we are free to
      cap and which does not spend the budget: the budget exists to bound how
      long the *server* can park a reader, not how long we park ourselves;
    * a ``Retry-After`` that fits in ``budget_remaining`` -- that interval in
      full, never truncated, and the budget shrinks by it;
    * a ``Retry-After`` that does not fit -- ``None``. Waiting it out would
      park the reader past the bound; retrying sooner would ignore the 429.
      So the fetch does neither: it records the cooldown so the interval is
      still respected, and the reader keeps what it already has.
    """
    retry_after = _retry_after_seconds(response)
    if retry_after is None:
        return _RetryPlan(backoff, budget_remaining)
    if retry_after > budget_remaining:
        cooldown = _begin_refresh_cooldown(retry_after, owner)
        if cooldown is None:
            return _RetryPlan(None, budget_remaining)
        logger.warning(
            "Enforcement rule fetch got HTTP %d asking for %.0fs with %.0fs of the %.0fs "
            "Retry-After budget left; failing open on the previous cache and not "
            "refreshing again for %.0fs",
            response.status_code, retry_after, budget_remaining,
            budget_total, cooldown,
        )
        return _RetryPlan(None, budget_remaining)
    return _RetryPlan(retry_after, budget_remaining - retry_after)


def _wire_value(payload: dict, keys: Tuple[str, ...]):
    """The value under the first of ``keys`` that ``payload`` carries, else ``None``.

    Each department field is read under its current name and then under its
    org-unit spelling, which the platform served until BACK-3448 renamed it with
    no alias; the fallback stays for one release so the SDK works on both sides.
    """
    for key in keys:
        value = payload.get(key)
        if value is not None:
            return value
    return None


class _FetchedRules(NamedTuple):
    """One enforcement payload: the rules plus the department-budget maps."""

    rules: list
    org_unit_blocks: dict
    org_unit_block_balances: dict
    org_unit_warnings: dict
    org_unit_candidates: dict


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
        return _FetchedRules(data, {}, {}, {}, {})
    if isinstance(data, dict):
        return _FetchedRules(
            data.get("rules", []),
            _normalize_map_keys(_wire_value(data, _ORG_UNIT_BLOCKS_KEYS)),
            _coerce_balances(_wire_value(data, _ORG_UNIT_BLOCK_BALANCES_KEYS)),
            _normalize_map_keys(_wire_value(data, _ORG_UNIT_WARNINGS_KEYS)),
            _coerce_candidates(_wire_value(data, _ORG_UNIT_CANDIDATES_KEYS)),
        )
    logger.warning("Unexpected enforcement response shape: %r", type(data).__name__)
    return None


class _FetchTarget(NamedTuple):
    """The credentials and the team-scoped URL every enforcement read hangs off."""

    api_key: str
    team_url: str
    owner: Tuple[str, str]


def _fetch_target() -> Optional[_FetchTarget]:
    """Resolve the API key and the team-scoped enforcement URL.

    ``None`` when the SDK is not configured to read enforcement at all, which
    every caller treats as "no reading available" rather than as an error.
    """
    global _team_id_warned, _api_key_warned

    client = _process_metering_client()
    api_key = client.api_key if client is not None else ""
    if not api_key:
        if not _api_key_warned:
            if _fail_mode_is_closed():
                logger.warning(
                    "Every AI call is refused (REVENIUM_CB_FAIL_MODE=closed): no Revenium API key is configured. "
                    "Set REVENIUM_METERING_API_KEY or call revenium_middleware.initialize_metering(api_key=...)."
                )
            else:
                logger.warning(
                    "Spending caps are not enforced: no Revenium API key is configured. "
                    "Set REVENIUM_METERING_API_KEY or call revenium_middleware.initialize_metering(api_key=...)."
                )
            _api_key_warned = True
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
    return _FetchTarget(api_key, f"{base_url}/v2/api/ai/enforcement-rules/{safe_team_id}",
                        (base_url, team_id))


def _get_enforcement(url: str, api_key: str, resource: str,
                     params: Optional[dict], retry_after_budget: float,
                     owner: Tuple[str, str]) -> Optional["httpx.Response"]:
    """GET one enforcement resource under the shared retry and Retry-After budget.

    Returns the first response the retry policy does not retry -- the caller
    reads its status and its body -- or ``None`` once the fetch has given up,
    so every caller fails open on what it already holds. ``resource`` names
    the read in the log lines.

    Every enforcement read goes through here rather than calling ``httpx.get``
    itself: a second unguarded GET against an origin that has already asked
    for a Retry-After interval reintroduces the request amplification
    FRONT-1682 fixed.

    ``params`` of ``None`` sends the request httpx builds from the URL alone,
    which is what the unfiltered team-wide read has always sent.
    ``retry_after_budget`` is the server-requested wait this read may spend in
    total across every attempt; it depends on whose thread the read runs on
    (see ``_REFRESH_RETRY_AFTER_GIVE_UP_SECONDS``). ``owner`` is whose
    cooldown a declined ``Retry-After`` starts.
    """
    budget_total = retry_after_budget
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

        plan = _plan_retry(response, backoff, retry_after_budget, budget_total, owner)
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


def _rules_query(rule_id: Optional[str], include_advisory: bool) -> Optional[dict]:
    """The query string of one rules read, ``None`` for the bare whole-team read."""
    if rule_id and include_advisory:
        raise ValueError("advisory rows are served only on the whole-team read; drop rule_id")
    if rule_id:
        return {"ruleId": rule_id}
    if include_advisory:
        return {"include": _ADVISORY_INCLUDE}
    return None


def _fetch_rules(rule_id: Optional[str] = None,
                 retry_after_budget: float = _REFRESH_RETRY_AFTER_GIVE_UP_SECONDS,
                 *, include_advisory: bool = False,
                 ) -> Optional[_FetchedRules]:
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

    ``include_advisory`` sends ``include=advisory``, which makes the server
    append the team's notify-only per-person budgets (rows whose ``source`` is
    ``ANOMALY`` and whose ``action`` is ``NOTIFY``) after the cost-control
    rows. It is opt-in for the same reason as ``rule_id`` and must never reach
    the cache: the server says an anomaly's numeric id can equal a cost
    control's, and the pre-call path keys its warnings and lookups on that id.
    The server drops advisory rows whenever ``ruleId`` is sent, so asking for
    both raises ``ValueError``.
    """
    params = _rules_query(rule_id, include_advisory)
    target = _fetch_target()
    if target is None:
        return None

    response = _get_enforcement(target.team_url, target.api_key, "rule", params,
                                retry_after_budget, target.owner)
    if response is None:
        return None

    try:
        # 204 No Content == no rules configured for this team; cache empty
        if response.status_code == 204:
            return _FetchedRules([], {}, {}, {}, {})
        response.raise_for_status()
        # Server currently returns ``{"rules": [...], "compiledAt": ...,
        # "departmentBudgetBlocks": {...}, "departmentBudgetBlockBalances":
        # {...}, "departmentBudgetWarnings": {...}}``; a bare list is accepted
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

    It runs on its caller's thread, so it spends the shorter inspection
    Retry-After budget, and records the same cooldown as the rule refresh when
    it gives up, but unlike ``_refresh_cache`` a cooldown
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
                                "roster", params, _INSPECTION_RETRY_AFTER_GIVE_UP_SECONDS,
                                target.owner)
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

    fetched = _fetch_rules(rule_id, _INSPECTION_RETRY_AFTER_GIVE_UP_SECONDS)
    if fetched is None or not fetched.rules:
        return None
    return fetched.rules[0]


def _is_advisory_row(rule) -> bool:
    return isinstance(rule, dict) and rule.get("source") == _ADVISORY_SOURCE


def fetch_advisory_budgets() -> Optional[list]:
    """Read the team's notify-only per-person budgets, bypassing the rule cache.

    An inspection call for integrators, like ``fetch_enforcement_rule``: it is
    not on the pre-call path and neither reads nor writes the cache
    ``check_enforcement`` evaluates, so an advisory budget never blocks a call.

    Returns:
        The rows whose ``source`` is ``ANOMALY``, each with ``action``
        ``NOTIFY``, a per-person ``groupBreakdown`` and, when the current
        window has a reading, ``readingAt``. ``[]`` when the team has none,
        and ``None`` when the read could not be completed.
    """
    fetched = _fetch_rules(include_advisory=True,
                           retry_after_budget=_INSPECTION_RETRY_AFTER_GIVE_UP_SECONDS)
    if fetched is None or not isinstance(fetched.rules, list):
        return None
    return [rule for rule in fetched.rules if _is_advisory_row(rule)]


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
    """Refresh the in-memory rule cache, unless a refresh is already running.

    Called by the poller thread, never by the pre-call path. Holding
    ``_refresh_lock`` throughout keeps one refresh in flight per process.
    """
    if not _refresh_lock.acquire(blocking=False):
        return
    try:
        _refresh_cache_holding_lock()
    finally:
        _refresh_lock.release()


def _refresh_cache_holding_lock() -> None:
    """Fetch the rules and install them, or back off when the fetch fails.

    Issues nothing while a cooldown is in force. A failed fetch -- timeout,
    refusal, 5xx, any 4xx -- keeps the cached rules and starts a backoff of
    ``_refresh_failure_backoff_seconds`` before the next attempt, and
    ``_cache_timestamp`` stays where it was, so the cache keeps reading as
    stale and is refreshed as soon as the backoff allows. The backoff is
    recorded before ``_refresh_lock`` is released, so no pre-call check can
    see a stale cache with neither a refresh nor a backoff in force and wake
    the poller straight back into the outage.

    Disk persistence happens outside ``_cache_lock`` so a slow filesystem write
    can't block concurrent ``check_enforcement`` callers on the pre-call path.
    """
    global _cached_rules, _cached_org_unit_blocks, _cached_org_unit_block_balances
    global _cached_org_unit_warnings, _cached_org_unit_candidates
    global _cache_timestamp, _cache_initialized, _cache_owner
    if _refresh_is_on_cooldown():
        logger.debug("Enforcement refresh skipped: a cooldown is still in force")
        return
    owner = _current_owner()
    fetched = _fetch_rules()
    if fetched is None:
        backoff = _begin_failure_backoff(owner)
        if backoff is None:
            logger.debug("Dropped a failed enforcement fetch for the team the process switched away from")
            return
        # A missing key or team was already reported, once, by _fetch_target.
        if _fetch_target() is not None:
            logger.warning(
                "Enforcement rules for team %s could not be refreshed; enforcing the rules already "
                "cached (none, if no fetch has succeeded yet) and trying again in %.0fs",
                owner[1], backoff,
            )
        return
    # The process moved to another team or address while this fetch was in
    # flight: its rules must not judge the new team's calls or land in the new
    # team's snapshot.
    if _current_owner() != owner:
        logger.debug("Dropped enforcement rules fetched for the team the process switched away from")
        return
    with _cache_lock:
        if _cache_owner not in (None, owner):
            logger.debug("Dropped enforcement rules fetched for the team the process switched away from")
            return
        _cache_owner = owner
        _cached_rules = fetched.rules
        _cached_org_unit_blocks = fetched.org_unit_blocks
        _cached_org_unit_block_balances = fetched.org_unit_block_balances
        _cached_org_unit_warnings = fetched.org_unit_warnings
        _cached_org_unit_candidates = fetched.org_unit_candidates
        _begin_cache_generation()
        _cache_timestamp = time.monotonic()
        _cache_initialized = True
    _persist_cache_to_disk(fetched.rules, fetched.org_unit_blocks,
                           fetched.org_unit_block_balances,
                           fetched.org_unit_warnings, owner=owner,
                           org_unit_candidates=fetched.org_unit_candidates)


def _poll_loop() -> None:
    interval = _poll_interval_seconds()
    while not _stop_event.is_set():
        _refresh_cache()
        # A wake-up asked for while that refresh ran was answered by it, so it
        # is dropped. Stop is checked after the clear because stop_polling sets
        # _stop_event before _poll_wakeup, so a stop can never be cleared away.
        _poll_wakeup.clear()
        if _stop_event.is_set():
            break
        _poll_wakeup.wait(_next_poll_wait_seconds(interval))


def _next_poll_wait_seconds(interval: float) -> float:
    """Sleep until the cooldown ends when one is running, else one interval.

    Waiting a whole interval after a refresh the cooldown skipped would
    double the gap between attempts whenever the wait ends a moment before the
    cooldown does.
    """
    remaining = _refresh_cooldown_remaining()
    return remaining if remaining > 0.0 else interval


def _forget_another_owners_rules() -> None:
    """Drop rules, department maps and backoff cached for another enforcement address or team.

    A process that switches teams must not judge the new team's calls by the
    old team's caps or email list. The next disk load then reads the new
    team's own snapshot, and the poller is woken to fetch its rules. A cache
    nobody owns yet (never loaded, or seeded by hand) is adopted as it is.
    """
    global _cached_rules, _cached_org_unit_blocks, _cached_org_unit_block_balances
    global _cached_org_unit_warnings, _cached_org_unit_candidates, _cache_timestamp
    global _cache_initialized, _disk_load_attempted, _refresh_cooldown_until, _cache_owner
    owner = _current_owner()
    if _cache_owner == owner:
        return
    with _cache_lock:
        previous, _cache_owner = _cache_owner, owner
        if _refresh_cooldown_owner not in (None, owner):
            _refresh_cooldown_until = 0.0
        if previous is None or previous == owner:
            return
        _cached_rules = []
        _cached_org_unit_blocks = {}
        _cached_org_unit_block_balances = {}
        _cached_org_unit_warnings = {}
        _cached_org_unit_candidates = {}
        _begin_cache_generation()
        _cache_timestamp = 0.0
        _cache_initialized = False
        _disk_load_attempted = False
        _refresh_cooldown_until = 0.0
    logger.info("Enforcement rules of the previous team were dropped; loading the rules of team %s", owner[1])
    _poll_wakeup.set()


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
    candidates = (_cached_org_unit_candidates
                  if isinstance(_cached_org_unit_candidates, dict) else {})
    return _DepartmentBudgets(dict(blocks), dict(balances), dict(warnings),
                              dict(candidates), _cache_generation)


def _get_rules() -> Tuple[list, _DepartmentBudgets, bool]:
    """Return cached rules, the department maps, and the initialized flag.

    Never fetches. A stale cache is served as it is and the poller is woken to
    refresh it, unless a refresh is already running or a cooldown is in force,
    either of which will bring the cache up to date without being asked.

    Reading every field under one ``_cache_lock`` acquisition prevents the
    fail-closed path from torn-reading ``_cache_initialized`` against a
    half-written ``_cached_rules`` from the background poller, and keeps the
    department maps consistent with each other and with the rules they
    reference.
    """
    now = time.monotonic()
    with _cache_lock:
        stale = not _cache_initialized or now - _cache_timestamp > _CACHE_TTL
        cooling_down = now < _refresh_cooldown_until
        rules = list(_cached_rules)
        departments = _snapshot_department_budgets()
        initialized = _cache_initialized
    if stale and not cooling_down and not _refresh_lock.locked():
        _poll_wakeup.set()
    return rules, departments, initialized


def _coerce_float(value) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _group_index(breakdown: list,
                 generation: Optional[int] = None) -> Dict[str, dict]:
    """This ``groupBreakdown`` keyed by ``groupValue``, built once per list.

    Only dict entries with a string ``groupValue`` are indexed, and the first
    entry for a value wins, exactly as the first match of a scan would.

    ``generation`` is the cache generation the breakdown was read under. A
    freshly built index is still returned when that generation has since been
    replaced, but it is not kept. ``None`` means the caller vouches that the
    breakdown is current.
    """
    cached = _group_indexes.get(id(breakdown))
    if cached is not None and cached[0] is breakdown:
        return cached[1]
    index: Dict[str, dict] = {}
    for entry in breakdown:
        if isinstance(entry, dict):
            group_value = entry.get("groupValue")
            if isinstance(group_value, str):
                index.setdefault(group_value, entry)
    with _cache_lock:
        if generation is None or generation == _cache_generation:
            _group_indexes[id(breakdown)] = (breakdown, index)
    return index


def _subscriber_id_key(value) -> Optional[str]:
    """The ``groupValue`` the server stores for this subscriber id, if any."""
    if isinstance(value, str):
        return value or None
    if type(value) is int and value != 0:
        return str(value)
    return None


def _group_lookup_keys(subscriber: dict, metadata: dict):
    """The caller's candidate ``groupValue`` keys, ids first, then emails.

    Ids are looked up as the text the server stores for them: a text id
    exactly as given (ids are opaque, so case matters), an integer id as its
    decimal text. Bools, floats, a zero id and every other type are not looked
    up and fail open; a zero id is not looked up, as the shared extractor
    drops it. Garbage a caller might put in subscriber_id must never abort the
    pre-provider hook with a TypeError. Emails are tried in the server's
    normalized form first (the key it writes for an email), then as given, so
    a server that stored a raw email still matches. The flat ``subscriber_id``
    is consulted only when the extracted subscriber carries no usable id, as
    metering sends the extracted id alone and a second person's balance must
    not refuse this caller.
    """
    for candidate in (subscriber.get("id"), metadata.get("subscriber_id")):
        key = _subscriber_id_key(stored_subscriber_value("id", candidate))
        if key is not None:
            yield key
            break
    for raw in (subscriber.get("email"), metadata.get("subscriber_email")):
        candidate = stored_subscriber_value("email", raw)
        if isinstance(candidate, str) and candidate:
            normalized = normalize_email(candidate)
            if normalized:
                yield normalized
            yield candidate


def _matching_group_entry(breakdown: list, usage_metadata: Optional[dict],
                          generation: Optional[int] = None) -> Optional[dict]:
    """Find this caller's entry in a rule's ``groupBreakdown``.

    ``EnforcementGroupEntry.groupValue`` is documented as "subscriber id,
    else email, else the unattributed sentinel", so the id is tried before
    the email — matching on email first would mis-attribute a caller who
    supplies both. Both are tried as separate lookups because a caller may
    send only one of them while the entry carries the other. The server
    writes an email key trimmed and lower-cased, so the caller's email is
    looked up in that form; see ``_group_lookup_keys``.

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
    index = _group_index(breakdown, generation)
    seen = set()
    for key in _group_lookup_keys(subscriber, metadata):
        if key in seen:
            continue
        seen.add(key)
        entry = index.get(key)
        if entry is not None:
            return entry
    return None


def _parse_window_end(raw) -> Optional[datetime.datetime]:
    """``windowEnd`` as an aware UTC-comparable datetime, else ``None``.

    A value with no offset is read as UTC, the zone the server writes in.
    """
    if not isinstance(raw, str):
        return None
    text = raw.strip().split("[", 1)[0]
    if text[-1:] in ("Z", "z"):
        text = text[:-1] + "+00:00"
    match = _WINDOW_END_PATTERN.fullmatch(text)
    if match is None:
        return None
    seconds, fraction, offset = match.groups()
    parsed = datetime.datetime.fromisoformat(
        f"{seconds}.{(fraction or '')[:6].ljust(6, '0')}{offset or ''}"
    )
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=datetime.timezone.utc)
    return parsed


def _window_has_ended(rule: dict) -> bool:
    """True once the rule's budget window is over, judged on this call.

    A breach is a verdict about one window. When the rules service cannot be
    reached the cached verdict is all the SDK has, and without this check a
    cap breached yesterday would go on blocking after its window reset, across
    restarts too. Judged per call against the wall clock, never at load time,
    so a long-lived process and a restarted one agree. An absent or
    unreadable ``windowEnd`` is "not ended": the rule keeps the verdict it
    always had, and nothing here may raise on the pre-provider path.
    """
    try:
        window_end = _parse_window_end(rule.get("windowEnd"))
        return window_end is not None and window_end <= datetime.datetime.now(datetime.timezone.utc)
    except Exception:
        return False


def _is_per_person_rule(rule: dict, breakdown) -> bool:
    """True when ``groupBreakdown`` holds the verdict for this rule.

    A non-empty list always does. An empty one does only on a rule that names
    a ``groupBy``: there the server sends ``[]`` with ``breached`` set when it
    could not read the per-person balances, and the rule-level flag then only
    says that somebody is over, not who.
    """
    if not isinstance(breakdown, list):
        return False
    group_by = rule.get("groupBy")
    return bool(breakdown) or (isinstance(group_by, str) and bool(group_by.strip()))


def _rule_blocks(
    rule: dict,
    usage_metadata: Optional[dict],
    generation: Optional[int] = None,
) -> Tuple[bool, Optional[float]]:
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

    A missing or null ``groupBreakdown`` is normal and never an error: the
    field is null for pooled rules and is populated on API reads only (it is
    absent from the Redis snapshot), so those rules keep the rule-level
    behaviour unchanged. An empty list keeps it too on a pooled rule, but on a
    rule with a ``groupBy`` it means the balances could not be read, and then
    nobody is blocked by it (see ``_is_per_person_rule``).

    A rule whose window has ended blocks nobody, whatever it last said.
    """
    if _window_has_ended(rule):
        return False, None
    breakdown = rule.get("groupBreakdown")
    if _is_per_person_rule(rule, breakdown):
        entry = _matching_group_entry(breakdown, usage_metadata, generation)
        if entry is None or not entry.get("breached", False):
            return False, None
        return True, _coerce_float(entry.get("currentValue"))

    # Pooled rule: tripped when the server marks it ``breached`` (current
    # nucleus schema) or ``blocked`` (legacy).
    tripped = rule.get("breached", False) or rule.get("blocked", False)
    return bool(tripped), _coerce_float(rule.get("currentValue"))


_FILTER_OPERATORS = frozenset({"IS", "IS_NOT", "CONTAINS", "STARTS_WITH", "ENDS_WITH", "IN"})

# The server's ProviderVocabulary.ALIAS_BUCKETS (hypercurrent commons): the
# provider values a call carrying the key label also counts toward. One way
# only, so a cap on FOUNDRY never covers direct Anthropic calls.
_PROVIDER_ALIAS_BUCKETS = {
    "FOUNDRY": frozenset({"FOUNDRY", "ANTHROPIC"}),
    "CLAUDECODE": frozenset({"CLAUDECODE", "ANTHROPIC"}),
    "CLAUDECOWORK": frozenset({"CLAUDECOWORK", "ANTHROPIC"}),
    "CODEXCLI": frozenset({"CODEXCLI", "OPENAI"}),
    "GEMINICLI": frozenset({"GEMINICLI", "GOOGLE"}),
}

# Dimensions whose value the SDK itself puts on the metering record, so a call
# that carries none is compared the way the server compares it: outside every
# filter except IS_NOT. Emptying this set would hold such a call to the rule.
_DIMENSIONS_COMPARED_WHEN_ABSENT = frozenset(
    {"AGENT", "TASK_TYPE", "PRODUCT", "ORGANIZATION", "SUBSCRIBER", "CREDENTIAL"})

_MODEL_SNAPSHOT = r"(?:\d{4}-\d{2}-\d{2}|\d{8}|\d{4})"
_MODEL_CARRIES_SNAPSHOT = re.compile(r"-" + _MODEL_SNAPSHOT + r"(?=-|$)")
_MODEL_SNAPSHOT_TAIL = re.compile(r"-" + _MODEL_SNAPSHOT + r"$")
_VERTEX_SNAPSHOT_SEPARATOR = re.compile(r"@(?=" + _MODEL_SNAPSHOT + r"$)")
_SPELLING_FOLDED_OPERATORS = frozenset({"IS", "IN"})
_MODEL_LATEST_SUFFIX = "-latest"
_GEMINI_LATEST_ALIAS = re.compile(r"gemini-([a-z][a-z-]*?)-latest")
_GEMINI_SERVED_VERSION = r"gemini-\d+(?:\.\d+)*-"
_GEMINI_SERVED_REVISION = r"(?:-preview(?:-\d{2}-\d{2}(?:\d{2})?)?|-\d{3})?"
_AZURE_MODEL_PREFIXES = ("azure/", "azure_ai/")

_SCALAR_COMPARISONS = {
    "IS": lambda actual, expected: actual == expected,
    "IS_NOT": lambda actual, expected: actual != expected,
    "CONTAINS": lambda actual, expected: expected in actual,
    "STARTS_WITH": lambda actual, expected: actual.startswith(expected),
    "ENDS_WITH": lambda actual, expected: actual.endswith(expected),
}
_EMAIL_EQUALITY_OPERATORS = frozenset({"IS", "IS_NOT", "IN"})


class _SubscriberEmail(str):
    """A subscriber value the server compares as an email, not as an id."""


def _subscriber_value(metadata: dict):
    subscriber = extract_subscriber_from_metadata(metadata)
    email = stored_subscriber_value("email", subscriber.get("email"))
    if isinstance(email, str) and email.strip():
        return _SubscriberEmail(email)
    if email is not None and not isinstance(email, str):
        return email
    return stored_subscriber_value("id", subscriber.get("id"))


def _credential_sent(metadata: dict) -> dict:
    subscriber = subscriber_with_credential_alias(
        extract_subscriber_from_metadata(metadata), get_credential_alias(metadata))
    return subscriber.get("credential") or {}


def _credential_value(metadata: dict):
    credential = _credential_sent(metadata)
    name = credential.get("name")
    return name if name is not None else credential.get("value")


_METADATA_DIMENSION_READERS = {
    "AGENT": lambda metadata: extract_common_metadata(metadata)["agent"],
    "TASK_TYPE": lambda metadata: extract_common_metadata(metadata)["task_type"],
    "PRODUCT": extract_product_name,
    "ORGANIZATION": extract_organization_name,
    "SUBSCRIBER": _subscriber_value,
    "CREDENTIAL": _credential_value,
}


class _CallScope:
    """The call's value under each filter dimension, each read on first use.

    Only a rule carrying filters asks, so a check against unfiltered rules
    runs none of the extractors (two of which warn on deprecated keys). The
    readers are the ones the metering record is built with, so the SDK
    compares what the server will.
    """

    def __init__(self, usage_metadata: Optional[dict], model, provider):
        self._metadata = usage_metadata or {}
        self._values = {"MODEL": model, "PROVIDER": provider}

    @staticmethod
    def knows(dimension: str) -> bool:
        return dimension in ("MODEL", "PROVIDER") or dimension in _METADATA_DIMENSION_READERS

    def value(self, dimension: str):
        if dimension not in self._values:
            self._values[dimension] = _METADATA_DIMENSION_READERS[dimension](self._metadata)
        return self._values[dimension]

    def names_credential_only_by_value(self) -> bool:
        credential = _credential_sent(self._metadata)
        return credential.get("name") is None and credential.get("value") is not None


def _filter_cannot_be_evaluated(rule: dict, dimension, operator, reason: str) -> bool:
    """Keep the rule applying, and say why once per rule and filter.

    Names the rule, the dimension and the operator, never a value: a
    SUBSCRIBER value is an email and a CREDENTIAL value can be a key.
    """
    rule_id = str(_rule_identifier(rule))
    key = (rule_id, str(dimension), str(operator), reason)
    with _warn_dedupe_lock:
        first = key not in _noted_unevaluable_filters
        _noted_unevaluable_filters.add(key)
    if first:
        logger.debug(
            "Revenium enforcement rule %s: cannot evaluate its %s %s filter (%s); the rule still applies",
            rule_id, sanitize_for_logging(dimension), sanitize_for_logging(operator), reason,
        )
    return True


def _expected_values(entry: dict, operator: str) -> List[str]:
    """The filter's values. IN reads ``values`` only: the server also writes
    them comma-joined into ``value``, which would never equal a call value."""
    if operator == "IN":
        values = entry.get("values")
        return [value for value in values if isinstance(value, str)] if isinstance(values, list) else []
    value = entry.get("value")
    return [value] if isinstance(value, str) else []


def _compare(operator: str, actual: str, expected: List[str]) -> bool:
    if operator == "IN":
        return actual in expected
    return _SCALAR_COMPARISONS[operator](actual, expected[0])


def _as_given(value: str) -> str:
    return value


def _comparison_form(dimension: str, operator: str, actual: str, ignore_case: bool = False):
    """How both sides are put before comparing: a subscriber email ignores
    case, and is trimmed as well for the equality operators; any other value
    is compared as given unless ``ignore_case`` asks otherwise."""
    if isinstance(actual, _SubscriberEmail):
        return normalize_email if operator in _EMAIL_EQUALITY_OPERATORS else str.lower
    return str.lower if ignore_case else _as_given


def _lowercased_model(model: str) -> str:
    """``model`` in lower case, keeping the other names it is known by."""
    aliases = getattr(model, "aliases", None)
    if aliases is None:
        return model.lower()
    return ModelKnownByAliases(model.lower(), *(alias.lower() for alias in aliases))


def _satisfies_via_alias(actual: str, expected: str) -> bool:
    return expected.upper() in _PROVIDER_ALIAS_BUCKETS.get(actual.upper(), ())


def _provider_filter_matches(operator: str, actual: str, expected: List[str]) -> bool:
    """PROVIDER ignores case, and an aliased label is inside the filters that
    name its alias target. Substring operators skip the alias: it says who the
    provider is, not which characters its label carries."""
    if operator == "IN":
        return any(value.lower() == actual.lower() or _satisfies_via_alias(actual, value) for value in expected)
    if operator in ("IS", "IS_NOT") and _satisfies_via_alias(actual, expected[0]):
        return operator == "IS"
    return _compare(operator, actual.lower(), [value.lower() for value in expected])


class ModelKnownByAliases(str):
    """A model a cap may name in more than one way: the string is the model
    the record carries, ``aliases`` the other names of the same model."""

    def __new__(cls, model: str, *aliases: str):
        instance = super().__new__(cls, model)
        instance.aliases = frozenset(aliases)
        return instance


def _dash_dated(model: str) -> str:
    """``model`` with Vertex AI's ``@`` before a trailing snapshot written as a
    dash, so ``claude-haiku-4-5@20251001`` reads ``claude-haiku-4-5-20251001``."""
    return _VERTEX_SNAPSHOT_SEPARATOR.sub("-", model)


def _metered_model_forms(model: str) -> Set[str]:
    """The names the server may meter for a requested model: as requested,
    under each of its aliases, without a leading ``provider/`` segment, with
    a Vertex ``@`` snapshot written as a dash, and without a ``-latest`` tail."""
    forms = {model, *getattr(model, "aliases", ())}
    if "/" in model:
        forms.add(model.split("/", 1)[1])
    forms.update(_dash_dated(form) for form in list(forms))
    forms.update(form[:-len(_MODEL_LATEST_SUFFIX)] for form in list(forms) if form.endswith(_MODEL_LATEST_SUFFIX))
    return forms


_XAI_PROVIDER_LABELS = frozenset({"XAI"})
_XAI_ROUTE_PREFIXES = ("xai/",)
_GOOGLE_PROVIDER_LABELS = frozenset({"GOOGLE", "GEMINICLI"})
_GOOGLE_ROUTE_PREFIXES = ("gemini/", "vertex_ai/", "vertex_ai_beta/")
_LITELLM_PROVIDER_LABEL = "LITELLM"


def _may_reach_vendor(provider, model: str, labels: frozenset, route_prefixes: tuple) -> bool:
    """Whether a call can go to a vendor whose names only mean its own models:
    its provider label, a LiteLLM route to it or a bare LiteLLM model, which
    LiteLLM may send to it, or a provider the SDK was not told, which a filter
    must not use to weaken a cap."""
    if not isinstance(provider, str):
        return True
    if provider.upper() == _LITELLM_PROVIDER_LABEL and "/" not in model:
        return True
    return provider.upper() in labels or model.lower().startswith(route_prefixes)


def _metered_model_candidates(model: str, expected: List[str], provider) -> Set[str]:
    """The metered forms of the request, each also with the snapshot tail of
    an expected value when the form carries no snapshot of its own: OpenAI
    meters ``gpt-4o-mini-2024-07-18`` for ``gpt-4o-mini``, so a filter naming
    a dated model is judged against the request as the server will see it.
    A vendor's own alias for a model also yields the expected values the
    vendor may serve it as, on a call that can reach that vendor."""
    forms = _metered_model_forms(model)
    tails = {tail.group(0) for tail in map(_MODEL_SNAPSHOT_TAIL.search, expected) if tail}
    undated = [form for form in forms if not _MODEL_CARRIES_SNAPSHOT.search(form)]
    widened = {form + tail for form in undated for tail in tails}
    candidates = forms | widened
    if _may_reach_vendor(provider, model, _GOOGLE_PROVIDER_LABELS, _GOOGLE_ROUTE_PREFIXES):
        candidates |= _served_by_a_gemini_latest_alias(forms, expected)
    if _may_reach_vendor(provider, model, _XAI_PROVIDER_LABELS, _XAI_ROUTE_PREFIXES):
        candidates |= _served_by_a_vendor_alias(forms)
    return candidates


def _served_by_a_gemini_latest_alias(forms: Set[str], expected: List[str]) -> Set[str]:
    """The expected values Google may meter a ``gemini-<tier>-latest`` form
    as: the alias moves to each new model of its tier, so any version of the
    tier, ``gemini-<version>-<tier>`` with an optional preview date or
    revision, and never another tier or a TTS, audio or image variant."""
    aliases = (_GEMINI_LATEST_ALIAS.fullmatch(form.rsplit("/", 1)[-1]) for form in forms)
    served = [re.compile(_GEMINI_SERVED_VERSION + re.escape(alias.group(1)) + _GEMINI_SERVED_REVISION)
              for alias in aliases if alias]
    return {value for value in expected if any(name.fullmatch(value) for name in served)}


def _served_by_a_vendor_alias(forms: Set[str]) -> Set[str]:
    """Every model a vendor lists a form under as one of its published names."""
    return {served for form in forms for served in SERVED_MODEL_BY_ALIAS.get(form.rsplit("/", 1)[-1], ())}


def _model_filter_matches(operator: str, expected: List[str], model: str, provider) -> bool:
    """MODEL is judged on the requested model, but the server compares the one
    it meters, which can carry a snapshot date the request did not. So the
    filter matches when it would match any plausible metered form of the
    request, and an Azure deployment name, which says nothing about the model,
    always matches. IS_NOT excludes only a request that already names its
    snapshot, since an undated one may be metered under a date the filter
    does not name, and judges it only by the model the record carries, never
    an alias, because the server compares that."""
    is_azure = (isinstance(provider, str) and provider.lower() == "azure") or model.lower().startswith(
        _AZURE_MODEL_PREFIXES)
    if is_azure:
        return True
    if operator == "IS_NOT":
        if not _MODEL_CARRIES_SNAPSHOT.search(model):
            return True
        return not any(_compare("IS", form, expected) for form in _metered_model_forms(str(model)))
    return any(_compare(operator, candidate, values)
               for values in _filter_spellings(operator, expected)
               for candidate in _metered_model_candidates(model, values, provider))


def _filter_spellings(operator: str, expected: List[str]) -> List[List[str]]:
    """The filter values as written and, for an equality filter that names a
    Vertex ``@`` snapshot, with that snapshot written as a dash too."""
    if operator not in _SPELLING_FOLDED_OPERATORS:
        return [expected]
    dash_dated = [_dash_dated(value) for value in expected]
    return [expected] if dash_dated == expected else [expected, dash_dated]


def _filter_matches(rule: dict, entry, scope: _CallScope, ignore_case: bool = False,
                    decided_dimensions: frozenset = frozenset()) -> bool:
    """Whether the call is inside one entry of a rule's ``filters``.

    Mirrors the server's matcher, except that a filter the SDK cannot judge
    (a malformed entry, an unknown dimension or operator, a model or provider
    it was not told, a value that is not a string) matches, so the cap keeps
    applying rather than being weakened. An entry on one of
    ``decided_dimensions`` matches without being judged, and ``ignore_case``
    compares every dimension without regard to letter case.
    """
    if not isinstance(entry, dict):
        return _filter_cannot_be_evaluated(rule, None, None, "malformed entry")
    dimension, operator = entry.get("dimension"), entry.get("operator")
    if not isinstance(dimension, str) or not isinstance(operator, str):
        return _filter_cannot_be_evaluated(rule, dimension, operator, "malformed entry")
    if dimension in decided_dimensions:
        return True
    if operator not in _FILTER_OPERATORS:
        return _filter_cannot_be_evaluated(rule, dimension, operator, "unknown operator")
    if not scope.knows(dimension):
        return _filter_cannot_be_evaluated(rule, dimension, operator, "unknown dimension")
    expected = _expected_values(entry, operator)
    if not expected:
        return True
    actual = scope.value(dimension)
    if actual is None and dimension in _DIMENSIONS_COMPARED_WHEN_ABSENT:
        return operator == "IS_NOT"
    if not isinstance(actual, str):
        return _filter_cannot_be_evaluated(rule, dimension, operator, "call value unknown")
    if dimension == "MODEL":
        if ignore_case:
            actual, expected = _lowercased_model(actual), [value.lower() for value in expected]
        return _model_filter_matches(operator, expected, actual, scope.value("PROVIDER"))
    if dimension == "PROVIDER":
        return _provider_filter_matches(operator, actual, expected)
    form = _comparison_form(dimension, operator, actual, ignore_case)
    return _compare(operator, form(actual), [form(value) for value in expected])


def _rule_applies_to_call(rule: dict, scope: _CallScope, ignore_case: bool = False,
                          decided_dimensions: frozenset = frozenset()) -> bool:
    """True when the call is inside every one of the rule's ``filters``.

    A rule with no filters applies to every call. Anything that goes wrong
    while judging them keeps the rule applying: this runs before the paid
    provider call, so it must neither raise nor weaken a cap. ``ignore_case``
    and ``decided_dimensions`` are passed to every ``_filter_matches``.
    """
    filters = rule.get("filters")
    if filters is None or filters == []:
        return True
    if not isinstance(filters, list):
        return _filter_cannot_be_evaluated(rule, None, None, "filters not a list")
    try:
        return all(_filter_matches(rule, entry, scope, ignore_case, decided_dimensions)
                   for entry in filters)
    except Exception:
        return _filter_cannot_be_evaluated(rule, None, None, "evaluation failed")


def _is_org_unit_rule(rule: dict) -> bool:
    """True for any department-scoped or department-grouped rule.

    Mirrors the server's own ``applicableRules`` exclusion exactly:
    ``departmentId == null && groupBy != "DEPARTMENT"`` — a rule is a
    department rule when EITHER field says so (``orgUnitId`` / ``ORG_UNIT`` on a
    platform predating BACK-3448). Ancestor-cap department rules carry
    ``departmentId != null`` with a non-DEPARTMENT (typically null) ``groupBy``,
    and their rule-level ``breached`` means "this department is over budget",
    not "this caller is over budget" — evaluated in the per-rule loop they
    would block every unrelated employee company-wide. Their verdict, like
    the grouped shape's, lives solely in the ``departmentBudgetBlocks`` map,
    because only the server knows which department the caller belongs to.
    """
    if _wire_value(rule, _ORG_UNIT_ID_KEYS) is not None:
        return True
    group_by = rule.get("groupBy")
    return isinstance(group_by, str) and group_by.strip().upper() in _ORG_UNIT_GROUP_BY_VALUES


def normalize_email(email: str) -> str:
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


_normalize_email = normalize_email


def _caller_emails(usage_metadata: Optional[dict]) -> List[str]:
    """The caller's normalized subscriber email(s), nested-first then flat.

    Deliberately email-only: ``departmentBudgetBlocks`` is keyed by subscriber
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
    for raw in (subscriber.get("email"), metadata.get("subscriber_email")):
        candidate = stored_subscriber_value("email", raw)
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
    ``departmentBudgetBlockBalances[email] ?: rule.currentValue``. For a per-person
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


def _action_blocks(rule: dict) -> bool:
    """True when the rule's ``action`` asks for the call to be refused.

    Only ``BLOCK`` blocks: the server's own live block is
    ``!shadowMode && action == "BLOCK"``, and ``WARN_ONLY`` (or the legacy
    ``THROTTLE``) is a notice the customer chose instead of a stop, and
    ``NOTIFY`` marks an advisory budget, which never blocks. An absent
    ``action`` is treated as BLOCK, so a payload that omits the field falls
    toward enforcing, not toward spend; the server always sends it.
    """
    action = rule.get("action")
    if action is None:
        return True
    return isinstance(action, str) and action.strip().upper() == "BLOCK"


def _rule_identifier(rule: dict):
    return rule.get("ruleId") or rule.get("id")


def _claim_non_blocking_warning(rule: dict) -> bool:
    """True the first time a non-blocking breach of this rule is seen in its window."""
    key = str(_rule_identifier(rule))
    window_end = rule.get("windowEnd")
    with _warn_dedupe_lock:
        if key in _warned_non_blocking_rules and _warned_non_blocking_rules[key] == window_end:
            return False
        _warned_non_blocking_rules[key] = window_end
        return True


def _warn_non_blocking_breach(rule: dict) -> None:
    """Log, once per breach, that a rule which does not block has been crossed.

    Names the rule, never a caller: for a per-person rule the matched entry is
    keyed by an email address.
    """
    if not _claim_non_blocking_warning(rule):
        return
    logger.warning(
        "Revenium enforcement rule %s (id %s) is over its threshold (action=%s); "
        "not blocking",
        sanitize_for_logging(rule.get("name", "cost limit")), _rule_identifier(rule), rule.get("action"),
    )


def _deny_message(rule) -> Optional[str]:
    """The message the customer wrote for ``rule``, shown as written apart from
    surrounding whitespace being trimmed (a blank message names no message):
    a placeholder in it is never filled, and a value that is not text names
    no message, so a block can never fail on it."""
    if not isinstance(rule, dict):
        return None
    deny = rule.get("denyMessage")
    if not isinstance(deny, str):
        return None
    return deny.strip() or None


def _block_message(rule_name, deny: Optional[str]) -> str:
    message = f"Request blocked by Revenium enforcement rule: {rule_name}"
    return message if deny is None else f"{message}. {deny}"


def _act_on_breached_rule(rule: dict, current_value: Optional[float]) -> None:
    """Raise ``BudgetExceededError`` for a breached rule whose action blocks.

    A breached rule that only warns is logged instead, once per breach, and
    the caller's loop goes on to the rules after it.
    """
    if not _action_blocks(rule):
        _warn_non_blocking_breach(rule)
        return
    rule_name = rule.get("name", "cost limit")
    deny = _deny_message(rule)
    raise BudgetExceededError(
        message=_block_message(rule_name, deny),
        rule_name=rule_name,
        current_value=current_value,
        threshold=_coerce_float(rule.get("threshold")),
        resets_at=rule.get("resetsAt"),
        rule_id=_rule_identifier(rule),
        deny_message=deny,
    )


def _raise_org_unit_block(rules: list, departments: _DepartmentBudgets,
                          usage_metadata: Optional[dict]) -> None:
    """Raise ``BudgetExceededError`` when the department map blocks the caller.

    Mirrors the server's own consumer of ``departmentBudgetBlocks``: the map is
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
    _raise_department_verdict(
        rules, rule_id, _caller_block_balance(departments.balances, usage_metadata))


def _raise_department_verdict(rules: list, rule_id: int, own_balance: Optional[float]) -> None:
    """Act on a department block the server decided for the caller under ``rule_id``.

    ``own_balance`` is the balance the server judged the caller against, or
    ``None`` to report the rule's own ``currentValue``.
    """
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

    if _window_has_ended(rule):
        logger.debug("Department budget rule %s is from a window that has ended; not blocking", rule_id)
        return

    # Observe-and-log only. Other rules still evaluate normally.
    if rule.get("shadowMode", False):
        logger.debug("Department budget rule %s is in shadow mode; not blocking", rule_id)
        return

    if not _action_blocks(rule):
        logger.warning(
            "Department budget threshold crossed for rule %s (action=%s); not blocking",
            rule_id, rule.get("action"),
        )
        return

    rule_name = rule.get("name") or _ORG_UNIT_FALLBACK_RULE_NAME
    deny = _deny_message(rule)
    raise BudgetExceededError(
        message=_block_message(rule_name, deny),
        rule_name=rule_name,
        current_value=(own_balance if own_balance is not None
                       else _coerce_float(rule.get("currentValue"))),
        threshold=_coerce_float(rule.get("threshold")),
        resets_at=rule.get("resetsAt"),
        rule_id=rule.get("ruleId") or rule.get("id") or rule_id,
        deny_message=deny,
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

    The per-person lookup indexes (``_group_indexes``) belong to the replaced
    rules, so they are dropped here too, as are the warn-only dedupe entries of
    rules the new cache no longer serves.
    """
    global _cache_generation
    _cache_generation += 1
    _group_indexes.clear()
    served = {str(_rule_identifier(rule))
              for rule in _cached_rules if isinstance(rule, dict)}
    with _warn_dedupe_lock:
        _warned_org_unit_callers.clear()
        for rule_id in set(_warned_non_blocking_rules) - served:
            del _warned_non_blocking_rules[rule_id]
        _noted_unevaluable_filters.difference_update(
            [key for key in _noted_unevaluable_filters if key[0] not in served])


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
    _warn_department_threshold(
        rules, departments.generation, caller_key, rule_id,
        _caller_block_balance(departments.balances, usage_metadata),
    )


def _warn_department_threshold(rules: list, generation: int, caller_key: str, rule_id: int,
                               balance: Optional[float]) -> None:
    """Log, once per cache generation, that ``caller_key`` crossed ``rule_id``'s warn tier."""
    if not _claim_org_unit_warning(generation, caller_key, rule_id):
        return

    rule = _rule_by_id(rules, rule_id)
    rule_name = (rule.get("name") if isinstance(rule, dict) else None) or str(rule_id)
    threshold = _coerce_float(rule.get("threshold")) if isinstance(rule, dict) else None
    clause = _warning_spend_clause(balance, threshold)
    logger.warning(
        "Approaching the Revenium department budget enforced by rule %s%s; "
        "calls are blocked once the threshold is crossed",
        rule_name, clause,
    )


def _caller_candidates(candidates: dict,
                       usage_metadata: Optional[dict]) -> Optional[Tuple[str, List[dict]]]:
    """The caller's key and candidate list, or ``None`` when there is none.

    The cached lists were cleaned by ``_coerce_candidates`` on every path that
    fills the cache, so they are returned as stored.
    """
    if not isinstance(candidates, dict) or not candidates:
        return None
    for email in _caller_emails(usage_metadata):
        usable = candidates.get(email)
        if isinstance(usable, list) and usable:
            return email, usable
    return None


def _candidate_unjudged_dimensions(scope: _CallScope) -> frozenset:
    """The rows of a candidate's rule that count as covering the call unjudged.

    Besides the rows the server decided, a CREDENTIAL row when the call names
    its credential only by raw value: the server counted the cap's total under
    the credential's saved name, which a raw value cannot be matched to, so
    the cap keeps applying, as it does on the server.
    """
    if scope.names_credential_only_by_value():
        return _CANDIDATE_DECIDED_DIMENSIONS | {"CREDENTIAL"}
    return _CANDIDATE_DECIDED_DIMENSIONS


def _first_candidate_for_call(rules: list, candidates: List[dict],
                              scope: _CallScope) -> Optional[dict]:
    """The first candidate whose cap covers this call, else ``None``.

    A candidate whose rule the cached rules do not carry covers it, as a
    single-slot block naming a missing rule always has. One whose window has
    ended is passed over for the next. Otherwise its rule's filter rows are
    judged against the call as the server counted the cap's total: without
    regard to case, and without the rows the server decided in building the
    list.
    """
    for candidate in candidates:
        rule = _rule_by_id(rules, candidate["ruleId"])
        if rule is None:
            return candidate
        if _window_has_ended(rule):
            continue
        if _rule_applies_to_call(rule, scope, ignore_case=True,
                                 decided_dimensions=_candidate_unjudged_dimensions(scope)):
            return candidate
    return None


def _act_on_department_candidates(rules: list, generation: int, caller_key: str,
                                  candidates: List[dict], scope: _CallScope) -> None:
    """Block or warn on the first candidate covering the call; nothing when none does."""
    chosen = _first_candidate_for_call(rules, candidates, scope)
    if chosen is None:
        return
    if chosen["tier"] == "BLOCK":
        _raise_department_verdict(rules, chosen["ruleId"], chosen["balance"])
    else:
        _warn_department_threshold(rules, generation, caller_key, chosen["ruleId"], chosen["balance"])


def _check_org_unit_block(rules: list, departments: _DepartmentBudgets,
                          usage_metadata: Optional[dict],
                          scope: Optional[_CallScope] = None) -> None:
    """The caller's department verdict, behind a fail-open guard.

    A caller the server published candidates for is judged on the first one
    covering this call (``_first_candidate_for_call``), and the single-slot maps
    are not read for them. Without a ``scope`` the call's model and provider
    are unknown, which keeps a cap scoped to either applying. Anyone else
    gets ``_raise_org_unit_block`` and then the warn tier, exactly as before
    the candidates existed.

    The pre-provider hook may raise ``BudgetExceededError`` and nothing else,
    so a malformed map degrades to "no block" instead of taking the caller's
    request down with it -- and the warn tier, which cannot block at all, runs
    under the same guard.

    The warning is emitted only for a caller the block decision let through:
    the server keeps the two maps disjoint, and a person who is already blocked
    has no use for the news that they were nearly blocked.
    """
    try:
        listed = _caller_candidates(departments.candidates, usage_metadata)
        if listed is not None:
            _act_on_department_candidates(rules, departments.generation, *listed,
                                          scope or _CallScope(usage_metadata, None, None))
            return
        _raise_org_unit_block(rules, departments, usage_metadata)
        _warn_org_unit_threshold(rules, departments, usage_metadata)
    except BudgetExceededError:
        raise
    except Exception:
        logger.debug("Department budget check failed; falling open", exc_info=True)


def check_enforcement(usage_metadata: Optional[dict] = None, *,
                      model: Optional[str] = None, provider: Optional[str] = None) -> None:
    """Pre-call enforcement check.

    Invoke before the upstream provider call. No-op when the circuit breaker
    is disabled or no rules are tripped.

    Never waits on the network, so it is safe to call on an event loop: the
    request is judged against the rules already cached, and refreshing them is
    the poller thread's job. Until the first fetch lands there are none, and
    the call goes through -- or, with ``REVENIUM_CB_FAIL_MODE=closed``, is
    refused -- at once. The only I/O is the first call's read of the
    ``REVENIUM_CACHE_DIR`` snapshot saved for this process's own team, a
    local file read once per process.

    Subscriber-grouped rules are evaluated against the caller's own balance
    from ``groupBreakdown`` rather than the rule-level aggregate; see
    ``_rule_blocks``. Only a breached rule whose ``action`` is ``BLOCK`` (or
    absent) refuses the call; a breached ``WARN_ONLY`` rule lets it through and
    logs one warning per rule per window (``_warn_non_blocking_breach``).

    Department budgets are decided before and independently of the
    per-rule loop, from the server's ``departmentBudgetBlocks`` email map, looked
    up under the normalized form of the caller's address; see
    ``_raise_org_unit_block``. A caller the server has only *warned* about
    (``departmentBudgetWarnings``) is never blocked and never raised on: they get
    one log line per refreshed verdict, so the block is not the first thing
    they hear about their department budget. See ``_warn_org_unit_threshold``.
    A caller the server also lists in ``departmentBudgetCandidates`` is judged
    instead on the first of those caps whose own filters cover this call, so a
    department cap scoped to one provider or model no longer blocks the rest;
    see ``_first_candidate_for_call``.

    A rule that carries ``filters`` (a cap scoped to an agent, model, provider,
    subscriber, credential, task type, product or organization) blocks or
    warns only a call that matches every entry, compared as the server
    compares it; see ``_filter_matches``. ``model`` and ``provider`` are the
    requested model and the provider label the call's metering record will
    carry. A call that carries no agent, task type, product, organization,
    subscriber or credential is outside a filter on it, except an ``IS_NOT``
    one, because the server never counts such a call toward that cap. A filter
    the SDK cannot judge (a model or provider not passed, an unknown dimension
    or operator, a malformed entry) keeps the rule applying, so a cap is never
    weakened. Department rules stay decided by the department map alone.

    Raises:
        BudgetExceededError: when a cost-limit rule blocks the call.
            All structured fields (``rule_name``, ``current_value``,
            ``threshold``, ``resets_at``, ``rule_id``) are populated when the
            server provides them. For a grouped rule ``current_value`` is the
            caller's group balance while ``threshold`` stays rule-level; for a
            department budget it is the caller's own balance from
            ``departmentBudgetBlockBalances`` when the server publishes one.
    """
    if is_bypass_enabled() or _checked_by_outer_wrapper():
        return
    if not is_circuit_breaker_enabled():
        return

    _forget_another_owners_rules()
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
    scope = _CallScope(usage_metadata, model, provider)
    _check_org_unit_block(rules, departments, usage_metadata, scope)

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
        if (rule_credential and rule_credential != credential) or not _rule_applies_to_call(rule, scope):
            continue

        blocks, current_value = _rule_blocks(
            rule, usage_metadata, departments.generation)
        if blocks:
            _act_on_breached_rule(rule, current_value)


def check_unmetered_call(api_metadata: Optional[dict] = None, *, model: Optional[str] = None,
                         provider: Optional[str] = None, callback_transport: Optional[str] = None) -> None:
    """Run the pre-call check for a call that gets no metering record of its own.

    Selective metering, or an outer wrapper that meters the call, leaves the
    provider call unmetered here, but a breached cap must still refuse it. The
    caller's attribution is the record's would-be identity: the decorator's
    metadata, the LangChain callback's publication for ``callback_transport``
    (read without consuming it, so the record that does meter the call still
    gets it), and the call's own ``usage_metadata`` on top.
    """
    if not is_circuit_breaker_enabled() or is_bypass_enabled():
        return
    callback_metadata = peek_callback_metadata(callback_transport) if callback_transport else {}
    check_enforcement(merge_metadata(overlay_metadata(callback_metadata, api_metadata)),
                      model=model, provider=provider)


def stop_polling() -> None:
    """Gracefully stop the background polling thread.

    Reads ``_poll_thread`` under ``_poll_lock`` to avoid a TOCTOU race with
    ``_ensure_poller_running`` spinning the thread up on a concurrent first
    request — without the lock, shutdown could observe ``None`` and skip the
    ``join`` even though a poller is alive.

    ``_poll_wakeup`` is set after ``_stop_event`` so a poller sleeping out its
    interval wakes now instead of up to one poll interval later.
    """
    _stop_event.set()
    _poll_wakeup.set()
    with _poll_lock:
        thread = _poll_thread
    if thread is not None:
        thread.join(timeout=5)
