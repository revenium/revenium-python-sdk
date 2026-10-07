"""Execution matrix: every supported entry point must emit exactly one metering payload.

Rows come from manifest.yaml. Each executable row runs against a stubbed provider
transport, and the payload recorded at the metering client must satisfy the
row's per-operation oracle. Known gaps are strict xfails keyed by their ticket.
A row runs only in the test environment its ``env`` names (``default`` when
absent); see ``entry_points.instrumented``. The same rows also serve as a
timing oracle: with the metering worker busy, a payload's ``request_duration``
must still match the call it measured.
"""
import contextlib
import contextvars
import json
import os
import pathlib
import re
import subprocess
import sys
import threading
import time
from unittest.mock import DEFAULT, MagicMock, patch

import pytest
import yaml
from packaging.version import InvalidVersion, Version

from entry_points import calls
from entry_points.instrumented import (
    DEFAULT_ENVIRONMENT,
    ENVIRONMENTS,
    MATRIX_ENV_VARIABLE,
    import_all_middleware,
    selected_environment_name,
)

import_all_middleware()

HERE = pathlib.Path(__file__).parent
MANIFEST = yaml.safe_load((HERE / "manifest.yaml").read_text())
ROWS = MANIFEST["rows"]
PROVIDERS = MANIFEST["providers"]
EXECUTABLE_STATUSES = ("metered", "gap")
EXECUTABLE_ROWS = [row for row in ROWS if row["status"] in EXECUTABLE_STATUSES]
ENVIRONMENT = selected_environment_name()
TICKET_KEY = re.compile(r"^BACK-\d+$")
VERIFY_VERSION_VARIABLE = "REVENIUM_VERIFY_VERSION"
SUBPROCESS_TIMEOUT_SECONDS = 300
STUB_METERING_ENV = {
    "REVENIUM_METERING_API_KEY": "hak_stub_entry_point_matrix",
    "REVENIUM_METERING_BASE_URL": "http://127.0.0.1:9",
}


class OracleMismatch(AssertionError):
    """The call completed but its metering payloads do not match the row's oracle."""


def _call_id(row):
    return row.get("call", row["id"])


def _environment_of(row):
    return row.get("env", DEFAULT_ENVIRONMENT)


ROWS_IN_THIS_ENVIRONMENT = [row for row in EXECUTABLE_ROWS if _environment_of(row) == ENVIRONMENT]


APPROXIMATE_USAGE = "approximate"
TOKEN_FIELDS = {
    "input": "input_token_count",
    "output": "output_token_count",
    "cache_read": "cache_read_token_count",
    "reasoning": "reasoning_token_count",
}


def _expected(row):
    provider = PROVIDERS[row["provider"]]
    return row.get("model", provider["model"]), provider["expected_provider"]


def _expected_usage(row):
    return row.get("usage", PROVIDERS[row["provider"]].get("usage"))


def _require(condition, message, payload):
    if not condition:
        raise OracleMismatch(f"{message}: {payload}")


def _require_tokens(payload, usage, kinds):
    for kind in kinds:
        field = TOKEN_FIELDS[kind]
        if usage == APPROXIMATE_USAGE:
            if kind in ("input", "output"):
                _require((payload.get(field) or 0) > 0, f"{field} must be > 0", payload)
        elif kind in usage:
            _require(payload.get(field) == usage[kind], f"{field} must equal the stub's {usage[kind]}", payload)


def _check_completion(payload, usage):
    _require(payload["operation"] == "completion", "expected a completion payload", payload)
    _require_tokens(payload, usage, ("input", "output", "cache_read", "reasoning"))


def _check_embedding(payload, usage):
    _require(payload["operation"] == "completion", "expected a completion payload", payload)
    _require(payload.get("operation_type") == "EMBED", "expected operation_type EMBED", payload)
    _require_tokens(payload, usage, ("input",))


def _check_embedding_without_usage(payload, usage):
    _require(payload["operation"] == "completion", "expected a completion payload", payload)
    _require(payload.get("operation_type") == "EMBED", "expected operation_type EMBED", payload)
    _require(not payload.get("input_token_count"), "the stub response carries no usage", payload)


def _check_image(payload, usage):
    _require(payload["operation"] == "image", "expected an image payload", payload)
    _require((payload.get("actual_image_count") or 0) >= 1, "actual_image_count must be >= 1", payload)
    _require("input_token_count" not in payload, "image payloads carry no token fields", payload)


ORACLES = {
    "completion": _check_completion,
    "embedding": _check_embedding,
    "embedding_without_usage": _check_embedding_without_usage,
    "image": _check_image,
}


def check_oracle(row, payloads):
    if len(payloads) != 1:
        raise OracleMismatch(f"expected exactly one payload, got {len(payloads)}: {payloads}")
    payload = payloads[0]
    model, provider = _expected(row)
    _require(payload.get("model") == model, f"model must be {model!r}", payload)
    _require(payload.get("provider") == provider, f"provider must be {provider!r}", payload)
    ORACLES[row["operation"]](payload, _expected_usage(row))


def _run_calls_in_subprocess(call_ids, middleware=()):
    command = [sys.executable, str(HERE / "calls.py"), "--calls", ",".join(call_ids),
               "--middleware", ",".join(middleware)]
    env = {**os.environ, **STUB_METERING_ENV}
    completed = subprocess.run(command, capture_output=True, text=True, env=env,
                               timeout=SUBPROCESS_TIMEOUT_SECONDS)
    assert completed.returncode == 0, completed.stderr[-4000:]
    return json.loads(completed.stdout.strip().splitlines()[-1])


@pytest.fixture(scope="module")
def differential():
    """The same calls in a fresh interpreter with none of our middleware imported."""
    return _run_calls_in_subprocess(sorted({_call_id(row) for row in ROWS_IN_THIS_ENVIRONMENT}))


@pytest.fixture(scope="module")
def isolated_outcomes():
    groups = {}
    for row in ROWS_IN_THIS_ENVIRONMENT:
        if row.get("isolate"):
            groups.setdefault(tuple(row["isolate"]), []).append(row)
    outcomes = {}
    for middleware, rows in groups.items():
        result = _run_calls_in_subprocess(sorted({_call_id(r) for r in rows}), middleware)
        for row in rows:
            outcomes[row["id"]] = result["outcomes"][_call_id(row)]
    return outcomes


def _attribute_crash(row, error, differential):
    baseline = differential["outcomes"].get(_call_id(row), {}).get("error")
    if baseline is None:
        return f"{row['id']} raised only with our middleware imported, so the crash is ours: {error}"
    return f"{row['id']} raised with and without our middleware, so the stub is broken: {error}"


def _execute_in_process(row, recording_client):
    try:
        calls.run_call(_call_id(row))
    except Exception as exc:  # noqa: BLE001
        return f"{type(exc).__name__}: {exc}", []
    calls.wait_for_metering()
    return None, calls.recorded_payloads(recording_client)


def _metered_after_verified_release(row):
    verified = os.environ.get(VERIFY_VERSION_VARIABLE)
    return bool(verified and row.get("since")) and Version(row["since"]) > Version(verified)


def _param(row):
    marks = []
    environment = _environment_of(row)
    if environment != ENVIRONMENT:
        marks.append(pytest.mark.skip(
            reason=f"runs in the {environment} environment ({MATRIX_ENV_VARIABLE}={environment})"))
    elif row["status"] == "gap":
        marks.append(pytest.mark.xfail(strict=True, raises=OracleMismatch,
                                       reason=f"{row['ticket']}: {row['reason']}"))
    elif _metered_after_verified_release(row):
        marks.append(pytest.mark.xfail(strict=True, raises=OracleMismatch, reason=(
            f"metered since {row['since']}; {os.environ[VERIFY_VERSION_VARIABLE]} predates it, "
            "so it must still miss the oracle, and a crash stays red")))
    return pytest.param(row, id=row["id"], marks=marks)


@pytest.mark.parametrize("row", [_param(row) for row in EXECUTABLE_ROWS])
def test_entry_point_emits_one_payload(row, mock_revenium_client, request):
    if row.get("isolate"):
        outcome = request.getfixturevalue("isolated_outcomes")[row["id"]]
        error, payloads = outcome["error"], outcome["payloads"]
    else:
        error, payloads = _execute_in_process(row, mock_revenium_client)
    if error is not None:
        pytest.fail(_attribute_crash(row, error, request.getfixturevalue("differential")))
    check_oracle(row, payloads)


LATENCY_BUDGET_MS = 100
QUEUE_WAIT_SECONDS = 0.5
LATENCY_ROWS = [row for row in ROWS_IN_THIS_ENVIRONMENT
                if row["status"] == "metered" and not row.get("isolate") and not _metered_after_verified_release(row)]
_row_of_event = contextvars.ContextVar("row_of_event", default=None)


class RowTiming:
    def __init__(self):
        self.error = None
        self.elapsed_ms = None
        self.recorded_durations = []


def _row_stamping_pool():
    """A one-worker pool that stamps each queued event with the row that was running when it was queued.

    Stamped at enqueue rather than read from the caller's context because some
    entry points (litellm.batch_completion) meter from their own thread pool.
    """
    from revenium_middleware._core.metering_pool import MeteringWorkerPool

    class RowStampingPool(MeteringWorkerPool):
        row_id = None

        def new_task(self, coro, ctx, blocks_synchronously=False, gated_by_circuit=True):
            ctx.run(_row_of_event.set, self.row_id)
            return super().new_task(coro, ctx, blocks_synchronously, gated_by_circuit)

    return RowStampingPool(1, 10 * len(LATENCY_ROWS) + 10, lambda task: None)


@contextlib.contextmanager
def _busy_metering_worker(pool, recorder):
    """Route metering through ``pool`` with its only worker blocked until the block exits."""
    from revenium_middleware._core import metering, metering_pool

    released = threading.Event()

    async def occupy_the_worker():
        released.wait(60)

    with patch.object(metering_pool, "_pool", pool), \
            patch("revenium_middleware._core.metering.client", recorder), \
            patch("revenium_middleware.client", recorder):
        metering.shutdown_event.clear()
        pool.submit(pool.new_task(occupy_the_worker(), contextvars.copy_context()))
        try:
            yield
            time.sleep(QUEUE_WAIT_SECONDS)
        finally:
            released.set()
            pool.wait_until_idle(60)
            pool.stop(timeout=5)


def _recording_client(timings):
    def record(**kwargs):
        timings[_row_of_event.get()].recorded_durations.append(kwargs.get("request_duration"))
        return DEFAULT

    recorder = MagicMock()
    for operation in calls.OPERATIONS:
        getattr(recorder.ai, f"create_{operation}").side_effect = record
    return recorder


@pytest.fixture(scope="module")
def timings_behind_a_busy_worker():
    timings = {row["id"]: RowTiming() for row in LATENCY_ROWS}
    pool = _row_stamping_pool()
    with _busy_metering_worker(pool, _recording_client(timings)):
        for row in LATENCY_ROWS:
            timing = timings[row["id"]]
            pool.row_id = row["id"]
            started = time.monotonic()
            try:
                calls.run_call(_call_id(row))
            except Exception as exc:  # noqa: BLE001
                timing.error = f"{type(exc).__name__}: {exc}"
            finally:
                timing.elapsed_ms = (time.monotonic() - started) * 1000
    return timings


@pytest.mark.parametrize("row", [pytest.param(row, id=row["id"]) for row in LATENCY_ROWS])
def test_recorded_duration_excludes_the_metering_queue_wait(row, timings_behind_a_busy_worker):
    timing = timings_behind_a_busy_worker[row["id"]]
    assert timing.error is None
    assert timing.recorded_durations, "no payload was recorded for this call"
    for duration in timing.recorded_durations:
        assert abs(duration - timing.elapsed_ms) <= LATENCY_BUDGET_MS, (
            f"request_duration {duration} ms, call took {timing.elapsed_ms:.0f} ms")


def test_every_stub_runs_cleanly_without_our_middleware(differential):
    assert differential["revenium_modules"] == []
    failures = {call_id: o["error"] for call_id, o in differential["outcomes"].items() if o["error"]}
    assert failures == {}


class TestManifest:
    def test_row_ids_are_unique(self):
        ids = [row["id"] for row in ROWS]
        assert len(ids) == len(set(ids))

    def test_rows_are_well_formed(self):
        oracles = set(MANIFEST["oracles"])
        assert oracles == set(ORACLES)
        for row in ROWS:
            assert row["provider"] in PROVIDERS, row["id"]
            assert row["mode"] in ("sync", "async"), row["id"]
            assert isinstance(row["streamed"], bool), row["id"]
            assert row["operation"] in oracles, row["id"]
            assert row["status"] in EXECUTABLE_STATUSES + ("not_executed", "unsupported"), row["id"]
            assert _environment_of(row) in ENVIRONMENTS, row["id"]

    def test_gaps_and_unexecuted_rows_name_a_ticket_and_a_reason(self):
        for row in ROWS:
            if row["status"] in ("gap", "not_executed"):
                assert TICKET_KEY.match(row.get("ticket", "")), row["id"]
            if row["status"] != "metered":
                assert row.get("reason"), row["id"]

    def test_every_executable_row_has_a_call_and_every_call_a_row(self):
        used = {_call_id(row) for row in EXECUTABLE_ROWS}
        assert used - set(calls.CALLS) == set()
        assert set(calls.CALLS) - used == set()

    def test_token_rows_state_exact_stub_usage_or_say_why_not(self):
        for row in EXECUTABLE_ROWS:
            if row["operation"] not in ("completion", "embedding"):
                continue
            usage = _expected_usage(row)
            if usage == APPROXIMATE_USAGE:
                reason = row.get("usage_reason", PROVIDERS[row["provider"]].get("usage_reason"))
                assert reason, row["id"]
            else:
                assert isinstance(usage, dict) and {"input", "output"} <= set(usage), row["id"]
                assert set(usage) <= set(TOKEN_FIELDS), row["id"]

    def test_since_is_a_release_version_on_metered_rows_only(self):
        for row in ROWS:
            if "since" not in row:
                continue
            assert row["status"] == "metered", row["id"]
            try:
                Version(row["since"])
            except InvalidVersion:
                pytest.fail(f"{row['id']}: since={row['since']!r} is not a version")

    def test_every_provider_has_a_metered_control(self):
        executed = {row["provider"] for row in EXECUTABLE_ROWS}
        controlled = {row["provider"] for row in EXECUTABLE_ROWS if row["status"] == "metered"}
        assert executed == controlled


class TestOracle:
    ROW = {"id": "synthetic", "provider": "openai", "operation": "completion"}
    GOOD = {"operation": "completion", "model": "gpt-4o-mini", "provider": "OPENAI",
            "input_token_count": 11, "output_token_count": 7,
            "cache_read_token_count": 3, "reasoning_token_count": 2}

    def test_accepts_a_correct_payload(self):
        check_oracle(self.ROW, [self.GOOD])

    def test_rejects_a_zero_token_payload(self):
        with pytest.raises(OracleMismatch):
            check_oracle(self.ROW, [{**self.GOOD, "input_token_count": 0, "output_token_count": 0}])

    def test_rejects_swapped_input_and_output_tokens(self):
        with pytest.raises(OracleMismatch):
            check_oracle(self.ROW, [{**self.GOOD, "input_token_count": 7, "output_token_count": 11}])

    def test_rejects_a_dropped_cache_read_count(self):
        with pytest.raises(OracleMismatch):
            check_oracle(self.ROW, [{**self.GOOD, "cache_read_token_count": 0}])

    def test_rejects_a_dropped_reasoning_count(self):
        with pytest.raises(OracleMismatch):
            check_oracle(self.ROW, [{**self.GOOD, "reasoning_token_count": 0}])

    def test_approximate_usage_still_rejects_zero_tokens(self):
        row = {"id": "synthetic", "provider": "litellm", "operation": "completion"}
        good = {**self.GOOD, "provider": "LITELLM", "input_token_count": 10, "output_token_count": 20}
        check_oracle(row, [good])
        with pytest.raises(OracleMismatch):
            check_oracle(row, [{**good, "output_token_count": 0}])

    def test_rejects_a_double_count(self):
        with pytest.raises(OracleMismatch):
            check_oracle(self.ROW, [self.GOOD, self.GOOD])

    def test_rejects_no_payload(self):
        with pytest.raises(OracleMismatch):
            check_oracle(self.ROW, [])

    def test_rejects_the_wrong_model(self):
        with pytest.raises(OracleMismatch):
            check_oracle(self.ROW, [{**self.GOOD, "model": "other"}])


class TestReleaseGating:
    ROW = {"id": "synthetic", "provider": "openai", "operation": "completion", "status": "metered", "since": "0.9.0",
           "env": ENVIRONMENT}

    def _marks(self, monkeypatch, verified):
        if verified is None:
            monkeypatch.delenv(VERIFY_VERSION_VARIABLE, raising=False)
        else:
            monkeypatch.setenv(VERIFY_VERSION_VARIABLE, verified)
        return _param(self.ROW).marks

    def test_a_release_older_than_since_must_still_fail_the_row(self, monkeypatch):
        (mark,) = self._marks(monkeypatch, "0.8.0")
        assert mark.name == "xfail" and mark.kwargs["strict"] is True

    def test_only_an_oracle_miss_is_an_expected_failure_on_an_older_release(self, monkeypatch):
        (mark,) = self._marks(monkeypatch, "0.8.0")
        assert mark.kwargs["raises"] is OracleMismatch

    def test_the_release_that_ships_the_fix_requires_the_row(self, monkeypatch):
        assert not self._marks(monkeypatch, "0.9.0")

    def test_the_source_tree_requires_every_row(self, monkeypatch):
        assert not self._marks(monkeypatch, None)


@pytest.mark.skipif(ENVIRONMENT != DEFAULT_ENVIRONMENT, reason="anthropic is installed in the default environment only")
class TestMutatedWrapper:
    """A wrapper that still submits a payload, but with zero tokens, must turn its cell red."""

    ROW = next(row for row in EXECUTABLE_ROWS if row["id"] == "anthropic.messages.create.sync")

    def test_zero_token_submission_fails_the_cell(self, monkeypatch, mock_revenium_client):
        from revenium_middleware.anthropic import middleware as anthropic_middleware
        submit = anthropic_middleware.submit_ai_event

        def submit_zero_tokens(operation, args, *rest, **kwargs):
            return submit(operation, {**args, "input_token_count": 0, "output_token_count": 0}, *rest, **kwargs)

        monkeypatch.setattr(anthropic_middleware, "submit_ai_event", submit_zero_tokens)
        error, payloads = _execute_in_process(self.ROW, mock_revenium_client)

        assert error is None
        assert len(payloads) == 1
        with pytest.raises(OracleMismatch, match="input_token_count"):
            check_oracle(self.ROW, payloads)

    def test_the_unmutated_wrapper_passes_the_same_cell(self, mock_revenium_client):
        error, payloads = _execute_in_process(self.ROW, mock_revenium_client)

        assert error is None
        check_oracle(self.ROW, payloads)
