"""Unit tests for the examples/agentic_outcomes runtime (common.py)."""
from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

EXAMPLES_DIR = Path(__file__).resolve().parents[2] / "examples" / "agentic_outcomes"


@pytest.fixture
def common(monkeypatch):
    """Import examples/agentic_outcomes/common.py with a stub client wired in."""
    monkeypatch.syspath_prepend(str(EXAMPLES_DIR))
    if "common" in sys.modules:
        importlib.reload(sys.modules["common"])
    module = importlib.import_module("common")
    module.reload_from_env()
    yield module
    sys.modules.pop("common", None)


# ---------------------------------------------------------------------------
# Fix #1: outcomeValue must NOT be truncated to int — fractional cents matter.
# ---------------------------------------------------------------------------
def test_report_outcome_preserves_fractional_value(common, monkeypatch):
    captured = {}

    class FakeClient:
        def report_outcome(self, job_id, payload):
            captured["payload"] = payload

        def emit_completion(self, payload): ...
        def emit_tool_event(self, payload): ...
        def create_job(self, *a, **kw): ...

    monkeypatch.setattr(common, "_client", lambda: FakeClient())
    outcome = common.Outcome("CONVERTED", 42.99, "DEAL-1", "SUCCESS", "won")
    common.report_outcome(
        outcome=outcome, value=42.99, reported_by="ai-pipe",
        agentic_job_id="job-1", metadata={"k": "v"}, timestamp=None, dry_run=False,
    )
    assert captured["payload"]["outcomeValue"] == pytest.approx(42.99)
    assert isinstance(captured["payload"]["outcomeValue"], float)


def test_report_outcome_preserves_sub_dollar_value(common, monkeypatch):
    captured = {}

    class FakeClient:
        def report_outcome(self, job_id, payload):
            captured["payload"] = payload

    monkeypatch.setattr(common, "_client", lambda: FakeClient())
    outcome = common.Outcome("CONVERTED", 0.50, "MICRO-1", "SUCCESS", "won")
    common.report_outcome(
        outcome=outcome, value=0.50, reported_by="ai-pipe",
        agentic_job_id="job-2", metadata=None, timestamp=None, dry_run=False,
    )
    # int(0.50) == 0 would lose this — must be preserved.
    assert captured["payload"]["outcomeValue"] == pytest.approx(0.50)


# ---------------------------------------------------------------------------
# outcomeReason is the prescribed home for a failure explanation — it must reach
# the payload as its own field, and stay absent when there is nothing to explain.
# ---------------------------------------------------------------------------
def test_report_outcome_sends_outcome_reason_as_its_own_field(common, monkeypatch):
    captured = {}

    class FakeClient:
        def report_outcome(self, job_id, payload):
            captured["payload"] = payload

    monkeypatch.setattr(common, "_client", lambda: FakeClient())
    outcome = common.Outcome("UNSUCCESSFUL", 0.0, "FAIL-1", "FAILED", "task_failed")
    common.report_outcome(
        outcome=outcome, value=0.0, reported_by="ai-pipe",
        agentic_job_id="job-fail", metadata=None, timestamp=None, dry_run=False,
        outcome_reason="task_failed",
    )
    assert captured["payload"]["outcomeReason"] == "task_failed"
    assert "metadata" not in captured["payload"]


def test_report_outcome_omits_outcome_reason_when_absent(common, monkeypatch):
    captured = {}

    class FakeClient:
        def report_outcome(self, job_id, payload):
            captured["payload"] = payload

    monkeypatch.setattr(common, "_client", lambda: FakeClient())
    outcome = common.Outcome("CONVERTED", 1.0, "DEAL-1", "SUCCESS", "deal_closed")
    common.report_outcome(
        outcome=outcome, value=1.0, reported_by="ai-pipe",
        agentic_job_id="job-ok", metadata=None, timestamp=None, dry_run=False,
    )
    assert "outcomeReason" not in captured["payload"]


# ---------------------------------------------------------------------------
# Fix #2: spec.build_summary must be invoked by run_example.
# ---------------------------------------------------------------------------
def test_run_example_invokes_spec_build_summary(common, monkeypatch, capsys):
    summary_called = {}

    def custom_summary(count, totals, elapsed):
        summary_called["count"] = count
        summary_called["totals"] = totals
        return ["CUSTOM-MARKER-LINE-XYZ"]

    spec = common.ExampleSpec(
        key="probe", description="probe", subscriber={"email": "x@y.z"},
        organization_name="Org", squad_name="sq", squad_id="sq-id",
        product_name="Prod", agent_base="agent",
        agentic_job_type="probe_agent", trace_type="probe",
        transaction_prefix="probe", reported_by="probe-pipe",
        job_label="Probe job", step_spacing_seconds=0, span_seconds=60,
        llm_steps=(), tool_steps=(),
        pick_outcome=lambda rng, sc, idx: common.Outcome("CONVERTED", 1.0, "ID", "SUCCESS"),
        build_summary=custom_summary,
    )

    class FakeClient:
        def emit_completion(self, p): ...
        def emit_tool_event(self, p): ...
        def report_outcome(self, jid, p): ...
        def create_job(self, *a, **kw): ...

    monkeypatch.setattr(common, "_client", lambda: FakeClient())
    monkeypatch.setattr(sys, "argv", ["probe", "--count", "2", "--dry-run", "--seed", "s"])

    common.run_example(spec)

    out = capsys.readouterr().out
    assert "CUSTOM-MARKER-LINE-XYZ" in out, "spec.build_summary was not invoked by run_example"
    assert summary_called["count"] == 2


# ---------------------------------------------------------------------------
# Fix #3: _run_one_job no longer wraps report_outcome in a redundant try/except.
# (Behavior test: a failure inside report_outcome is captured once in _OUTCOME_FAILURES.)
# ---------------------------------------------------------------------------
def test_outcome_failure_recorded_once(common, monkeypatch):
    common._OUTCOME_FAILURES.clear()

    class FakeClient:
        def report_outcome(self, job_id, payload):
            raise RuntimeError("server exploded")

    monkeypatch.setattr(common, "_client", lambda: FakeClient())
    outcome = common.Outcome("CONVERTED", 1.0, "ID", "SUCCESS", "won")
    common.report_outcome(
        outcome=outcome, value=1.0, reported_by="ai-pipe",
        agentic_job_id="job-fail", metadata=None, timestamp=None, dry_run=False,
    )
    assert len(common._OUTCOME_FAILURES) == 1
    assert common._OUTCOME_FAILURES[0]["job_id"] == "job-fail"


# ---------------------------------------------------------------------------
# --count 0 must be rejected, not cause a ZeroDivisionError in build_summary.
# ---------------------------------------------------------------------------
def test_run_example_rejects_zero_count(common, monkeypatch):
    spec = common.ExampleSpec(
        key="probe", description="probe", subscriber={"email": "x@y.z"},
        organization_name="Org", squad_name="sq", squad_id="sq-id",
        product_name="Prod", agent_base="agent",
        agentic_job_type="probe_agent", trace_type="probe",
        transaction_prefix="probe", reported_by="probe-pipe",
        job_label="Probe job", step_spacing_seconds=0, span_seconds=60,
        llm_steps=(), tool_steps=(),
    )
    monkeypatch.setattr(sys, "argv", ["probe", "--count", "0", "--dry-run"])
    with pytest.raises(SystemExit):
        common.run_example(spec)


# ---------------------------------------------------------------------------
# Sanity check: example scripts import without error.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", ["sales", "coding", "support"])
def test_example_script_imports(monkeypatch, name):
    monkeypatch.syspath_prepend(str(EXAMPLES_DIR))
    sys.modules.pop(name, None)
    sys.modules.pop("common", None)
    module = importlib.import_module(name)
    assert hasattr(module, "SPEC")
    sys.modules.pop(name, None)
    sys.modules.pop("common", None)
