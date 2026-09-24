"""Shared stubs for the enforcement rule-fetch tests.

``test_enforcement_fetch_retry.py`` and ``test_enforcement_retry_after_bound.py``
both drive ``_fetch_rules`` against a scripted sequence of responses with the
backoff wait recorded rather than taken. The helpers live here so the two files
cannot drift apart -- an earlier copy of ``SequencedGet`` had silently lost the
branch that raises connection errors.
"""
import httpx
import pytest

from revenium_middleware._core import enforcement

URL = "https://api.test/v2/api/ai/enforcement-rules/team-1"


def make_response(status_code, headers=None, json_body=None):
    request = httpx.Request("GET", URL)
    return httpx.Response(status_code, headers=headers or {},
                          json=json_body if json_body is not None else {"rules": []},
                          request=request)


class SequencedGet:
    """httpx.get stand-in returning (or raising) a scripted sequence.

    The last outcome repeats once the sequence runs out, so a test that wants
    "429 forever" can pass one response. ``calls`` is the control-plane request
    count the tests assert on, and ``requests`` is the ``(url, kwargs)`` of
    each one for the tests that assert on the request the fetch built.
    """

    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = 0
        self.requests = []

    def __call__(self, *args, **kwargs):
        url = args[0] if args else kwargs.get("url")
        self.requests.append((url, kwargs))
        outcome = self.outcomes[min(self.calls, len(self.outcomes) - 1)]
        self.calls += 1
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    @property
    def last_url(self):
        return self.requests[-1][0]

    @property
    def last_params(self):
        return self.requests[-1][1].get("params")


def stub_get(monkeypatch, outcomes):
    stub = SequencedGet(outcomes)
    monkeypatch.setattr(enforcement.httpx, "get", stub)
    return stub


def stale_cache_timestamp():
    """A ``_cache_timestamp`` that ``_get_rules`` is guaranteed to treat as stale.

    Not ``0.0``. Staleness is ``time.monotonic() - _cache_timestamp``, and
    ``time.monotonic()`` is seconds since boot, so ``0.0`` means "as old as
    this host's uptime". On a developer laptop that is days and the cache reads
    as stale; on a freshly booted CI runner it is a few seconds, which is
    younger than ``_CACHE_TTL``, so the cache reads as **fresh** and no refresh
    happens at all -- a test expecting one then sees zero requests.

    Reads the clock through ``enforcement.time`` so a test that installed a
    ``FakeClock`` gets a timestamp stale on that clock, not on the real one.
    """
    return enforcement.time.monotonic() - enforcement._CACHE_TTL * 2


class FakeClock:
    """Monotonic clock a test can advance by hand.

    ``time.monotonic`` is the only clock ``enforcement`` reads, so replacing
    the module's ``time`` lets a test step over a Retry-After cooldown deadline
    without sleeping through it.
    """

    def __init__(self, now=1_000_000.0):
        self.now = now

    def monotonic(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


@pytest.fixture(autouse=True)
def reset_refresh_cooldown(monkeypatch):
    """Clear any Retry-After cooldown around every test in this package.

    The cooldown is module state, so a test that gives up on a 429 would
    otherwise suppress the refresh in whichever test ran next.
    """
    monkeypatch.setattr(enforcement, "_refresh_cooldown_until", 0.0)


@pytest.fixture(autouse=True)
def reset_department_warning_dedupe():
    """Forget which callers have been warned, around every test in this package.

    The warn-once set is module state that ``monkeypatch`` cannot restore
    (it is mutated in place, not rebound), so a caller warned in one test
    would otherwise silence the same caller in whichever test ran next.
    """
    enforcement._begin_cache_generation()
    yield
    enforcement._begin_cache_generation()


@pytest.fixture()
def fetch_env(monkeypatch):
    """A fetchable enforcement config with every wait recorded, never taken."""
    monkeypatch.setenv("REVENIUM_METERING_API_KEY", "hak_enforcement_test")
    monkeypatch.setenv("REVENIUM_TEAM_ID", "team-1")
    sleeps = []
    monkeypatch.setattr(enforcement, "_sleep", lambda seconds: sleeps.append(seconds), raising=False)
    return monkeypatch, sleeps


@pytest.fixture()
def department_cache(monkeypatch):
    """Circuit breaker on, the whole department payload seeded, nothing live.

    Returns a callable taking ``(rules, blocks, balances, warnings)`` and
    installing them as a fresh, initialized cache, so ``check_enforcement``
    evaluates them with no network and no poller thread. A map is installed
    exactly as given -- a malformed value reaches the code under test rather
    than being coerced to an empty dict -- and only an omitted (``None``) map
    becomes one. Seeding the cache by hand is a new cache generation, so the
    warn-once set is cleared and the generation bumped with it, exactly as a
    fetch or a disk load does.

    The sibling ``load_rules`` in ``test_enforcement_group_breakdown.py``
    predates ``orgUnitBudgetBlockBalances`` and seeds only the block map; tests
    that need the balances or the warnings take this one.
    """
    monkeypatch.setenv("REVENIUM_CIRCUIT_BREAKER_ENABLED", "true")
    monkeypatch.delenv("REVENIUM_BYPASS", raising=False)
    monkeypatch.delenv("REVENIUM_CB_FAIL_MODE", raising=False)
    monkeypatch.setattr(enforcement, "_load_cache_from_disk", lambda: None)
    monkeypatch.setattr(enforcement, "_ensure_poller_running", lambda: None)

    def load(rules, blocks=None, balances=None, warnings=None):
        # ``is None`` means "not supplied", never ``or {}``: a test that seeds a
        # deliberately malformed map has to see it reach the cache, and every
        # interesting malformed value ([], "", 0) is falsy. Coercing those to an
        # ordinary empty dict defanged the fail-open tests that pass them.
        monkeypatch.setattr(enforcement, "_cached_rules", list(rules))
        monkeypatch.setattr(enforcement, "_cached_org_unit_blocks",
                            blocks if blocks is not None else {})
        monkeypatch.setattr(enforcement, "_cached_org_unit_block_balances",
                            balances if balances is not None else {})
        monkeypatch.setattr(enforcement, "_cached_org_unit_warnings",
                            warnings if warnings is not None else {})
        monkeypatch.setattr(enforcement, "_cache_timestamp", enforcement.time.monotonic())
        monkeypatch.setattr(enforcement, "_cache_initialized", True)
        enforcement._begin_cache_generation()

    return load


@pytest.fixture()
def department_snapshot_dir(monkeypatch, tmp_path):
    """A snapshot directory plus a cache that has never been loaded.

    Yields the directory so a test can read (or corrupt) the two snapshot
    files the enforcement cache writes into it.
    """
    monkeypatch.setenv("REVENIUM_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(enforcement, "_disk_load_attempted", False)
    monkeypatch.setattr(enforcement, "_cached_rules", [])
    monkeypatch.setattr(enforcement, "_cached_org_unit_blocks", {})
    monkeypatch.setattr(enforcement, "_cached_org_unit_block_balances", {})
    monkeypatch.setattr(enforcement, "_cached_org_unit_warnings", {})
    monkeypatch.setattr(enforcement, "_cache_timestamp", 0.0)
    monkeypatch.setattr(enforcement, "_cache_initialized", False)
    return tmp_path
