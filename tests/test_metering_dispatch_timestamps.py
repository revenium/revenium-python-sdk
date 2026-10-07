"""Metering coroutines must not read the clock: they run when a worker is free, not when the provider answered (BACK-3914).

``run_async_in_thread`` queues the coroutine it is handed (BACK-3904), and an
event that overflows the queue is built at the next buffer flush. A
``response_time`` or ``request_duration`` computed inside that coroutine
therefore measures the queue wait as well as the call. This module finds every
coroutine handed to the dispatcher and fails when it, or a function of its own
module that it calls, reads the wall or monotonic clock.
"""
import ast
import pathlib
import textwrap
from typing import Dict, Iterator, List, NamedTuple, Optional, Set

import pytest

import revenium_middleware

PACKAGE_ROOT = pathlib.Path(revenium_middleware.__file__).parent
DISPATCHER = "run_async_in_thread"
CLOCK_MODULES = {"datetime", "time", "date"}
CLOCK_READS = {
    "now", "utcnow", "today", "time", "time_ns", "monotonic", "monotonic_ns",
    "perf_counter", "perf_counter_ns",
}
EXPECTED_DISPATCHING_MODULES = {
    "_metering/decorator.py",
    "anthropic/bedrock_adapter.py",
    "anthropic/bedrock_transport.py",
    "anthropic/middleware.py",
    "fal/_metering.py",
    "google/common/utils.py",
    "litellm/client/middleware.py",
    "litellm/proxy/guardrail.py",
    "litellm/proxy/middleware.py",
    "ollama/middleware.py",
    "openai/middleware.py",
    "perplexity/middleware.py",
    "perplexity/perplexity_sdk.py",
}


class DispatchSite(NamedTuple):
    module: str
    line: int
    coroutine: str


class ClockRead(NamedTuple):
    site: DispatchSite
    call_path: str
    line: int


def _called_name(node: ast.expr) -> Optional[str]:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _is_clock_read(call: ast.Call) -> bool:
    func = call.func
    if not (isinstance(func, ast.Attribute) and func.attr in CLOCK_READS):
        return False
    return _called_name(func.value) in CLOCK_MODULES


class _Module:
    def __init__(self, relative: str, source: str):
        self.relative = relative
        self.tree = ast.parse(source)
        self.parents: Dict[ast.AST, ast.AST] = {}
        self.functions: Dict[str, List[ast.AST]] = {}
        for parent in ast.walk(self.tree):
            for child in ast.iter_child_nodes(parent):
                self.parents[child] = parent
            if isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self.functions.setdefault(parent.name, []).append(parent)

    def enclosing_function(self, node: ast.AST) -> Optional[ast.AST]:
        node = self.parents.get(node)
        while node is not None and not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            node = self.parents.get(node)
        return node

    def dispatch_calls(self, dispatchers: Set[str]) -> Iterator[ast.Call]:
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Call) and _called_name(node.func) in dispatchers and node.args:
                yield node

    def resolve_def(self, name: str, scope: Optional[ast.AST]) -> Optional[ast.AST]:
        candidates = self.functions.get(name, [])
        nested = [f for f in candidates if scope is not None and self.enclosing_function(f) is scope]
        if nested:
            return nested[-1]
        top_level = [f for f in candidates if self.enclosing_function(f) is None]
        return top_level[0] if top_level else None

    def assigned_call(self, name: str, scope: ast.AST) -> Optional[ast.Call]:
        for node in ast.walk(scope):
            if (isinstance(node, ast.Assign) and isinstance(node.value, ast.Call)
                    and any(isinstance(t, ast.Name) and t.id == name for t in node.targets)):
                return node.value
        return None


def _parameter_names(function: ast.AST) -> Set[str]:
    arguments = function.args
    every = arguments.posonlyargs + arguments.args + arguments.kwonlyargs
    return {a.arg for a in every}


def _resolve_coroutine(module: _Module, call: ast.Call):
    """The function whose body runs on the worker, or the enclosing function when it forwards a parameter."""
    scope = module.enclosing_function(call)
    argument = call.args[0]
    if isinstance(argument, ast.Call):
        return "def", module.resolve_def(_called_name(argument.func), scope)
    if isinstance(argument, ast.Name) and scope is not None:
        if argument.id in _parameter_names(scope):
            return "forwards", scope
        assigned = module.assigned_call(argument.id, scope)
        if assigned is not None:
            if _called_name(assigned.func) in _parameter_names(scope):
                return "forwards", scope
            return "def", module.resolve_def(_called_name(assigned.func), scope)
        return "def", module.resolve_def(argument.id, scope)
    return "def", None


def _clock_reads(module: _Module, coroutine: ast.AST, site: DispatchSite) -> List[ClockRead]:
    found = []
    seen = set()
    pending = [(coroutine, coroutine.name)]
    while pending:
        function, path = pending.pop()
        if function in seen:
            continue
        seen.add(function)
        for node in ast.walk(function):
            if not isinstance(node, ast.Call):
                continue
            if _is_clock_read(node):
                found.append(ClockRead(site, path, node.lineno))
                continue
            callee = _called_name(node.func)
            if callee == function.name:
                continue
            for helper in module.functions.get(callee, []):
                if module.enclosing_function(helper) in (None, function):
                    pending.append((helper, f"{path} -> {callee}"))
    return found


def scan(modules: Dict[str, str]):
    """Return every dispatch site and every clock read reachable from its coroutine."""
    parsed = [_Module(relative, source) for relative, source in modules.items()]
    dispatchers = {DISPATCHER}
    while True:
        forwarding = set()
        for module in parsed:
            for call in module.dispatch_calls(dispatchers):
                kind, function = _resolve_coroutine(module, call)
                if kind == "forwards":
                    forwarding.add(function.name)
        if forwarding <= dispatchers:
            break
        dispatchers |= forwarding

    sites, unresolved, reads = [], [], []
    for module in parsed:
        for call in module.dispatch_calls(dispatchers):
            kind, function = _resolve_coroutine(module, call)
            if kind == "forwards":
                continue
            name = function.name if function is not None else ast.unparse(call.args[0])
            site = DispatchSite(module.relative, call.lineno, name)
            sites.append(site)
            if function is None:
                unresolved.append(site)
            else:
                reads.extend(_clock_reads(module, function, site))
    return sites, unresolved, reads


def _package_sources() -> Dict[str, str]:
    sources = {}
    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        relative = path.relative_to(PACKAGE_ROOT).as_posix()
        if relative.startswith("_core/"):
            continue
        sources[relative] = path.read_text(encoding="utf-8")
    return sources


@pytest.fixture(scope="module")
def package_scan():
    return scan(_package_sources())


def test_every_integration_that_meters_is_enumerated(package_scan):
    sites, _, _ = package_scan
    assert EXPECTED_DISPATCHING_MODULES <= {site.module for site in sites}


def test_every_queued_coroutine_resolves_to_its_definition(package_scan):
    _, unresolved, _ = package_scan
    assert unresolved == []


def test_no_queued_coroutine_reads_the_clock(package_scan):
    _, _, reads = package_scan
    assert reads == [], "\n".join(
        f"{r.site.module}:{r.line} in {r.call_path} (queued at line {r.site.line})" for r in reads
    )


BUGGY_DISPATCH = textwrap.dedent("""
    import datetime

    def _duration(start):
        return (datetime.datetime.now(datetime.timezone.utc) - start).total_seconds()

    def meter(request_time_dt):
        async def metering_call():
            submit(request_duration=_duration(request_time_dt))
        run_async_in_thread(metering_call())
""")

FIXED_DISPATCH = textwrap.dedent("""
    import datetime

    def meter(request_time_dt):
        response_time_dt = datetime.datetime.now(datetime.timezone.utc)

        async def metering_call():
            submit(request_duration=(response_time_dt - request_time_dt).total_seconds())
        run_async_in_thread(metering_call())
""")

FORWARDED_DISPATCH = textwrap.dedent("""
    import time

    def _safe_dispatch(coro_func):
        coroutine = coro_func()
        run_async_in_thread(coroutine)

    def meter():
        async def metering_call():
            submit(at=time.time())
        _safe_dispatch(metering_call)
""")


class TestScanner:
    def test_flags_a_clock_read_in_a_helper_the_coroutine_calls(self):
        _, _, reads = scan({"buggy.py": BUGGY_DISPATCH})
        assert [r.call_path for r in reads] == ["metering_call -> _duration"]

    def test_accepts_timestamps_captured_before_queueing(self):
        sites, unresolved, reads = scan({"fixed.py": FIXED_DISPATCH})
        assert [s.coroutine for s in sites] == ["metering_call"]
        assert unresolved == [] and reads == []

    def test_follows_a_wrapper_that_forwards_the_coroutine(self):
        sites, _, reads = scan({"forwarded.py": FORWARDED_DISPATCH})
        assert [s.coroutine for s in sites] == ["metering_call"]
        assert [r.call_path for r in reads] == ["metering_call"]
