"""AI Coding Workflow outcome demo for Revenium."""
from __future__ import annotations

import random
from typing import Any

from common import ExampleSpec, LlmStep, Outcome, Scenario, ToolStep, run_example

# CUSTOMIZATION POINTS:
# - SCENARIOS controls workflow mix, per-scenario step sequences, and value ranges
# - LLM_STEPS and TOOL_STEPS define reusable step templates referenced by scenario
# - OUTCOME_WEIGHTS controls autonomous completion, escalation, and cancellation rates
SUBSCRIBER = {"email": "demo-runner@example.com"}
SCENARIOS = (
    Scenario(
        "pr-review", "PR-Review-Automation", 5.0, (2, 7), (5.0, 500.0),
        ("plan", "draft", "self-review", "submit"),
        ("repo.search", "ci.compile_test", "github.post_review"),
    ),
    Scenario(
        "test-gen", "Regression-Test-Generation", 4.0, (2, 7), (3.0, 220.0),
        ("plan", "draft", "self-review", "submit", "self-review"),
        ("repo.search", "ci.compile_test"),
    ),
    Scenario(
        "incident-rca", "Incident-RCA", 3.0, (2, 7), (5.0, 300.0),
        ("plan", "draft", "self-review"),
        ("repo.search", "ci.compile_test"),
    ),
    Scenario(
        "release-gate", "Release-Readiness-Check", 3.0, (2, 7), (3.0, 150.0),
        ("plan", "draft", "submit"),
        ("ci.compile_test", "github.post_review"),
    ),
    Scenario(
        "dep-risk", "Dependency-Risk-Analysis", 2.0, (2, 7), (5.0, 180.0),
        ("plan", "self-review", "submit"),
        ("repo.search", "github.post_review"),
    ),
)
OUTCOME_WEIGHTS = {"CONVERTED": 0.72, "ESCALATED": 0.10, "CUSTOM": 0.18}
LLM_STEPS = (
    LlmStep("plan", "claude-sonnet-4-5", "anthropic", 1500, 600, 1200, 0.0028,
            agent_role="orchestrator",
            system_prompt="Plan a safe coding-agent workflow from a developer request.",
            input_template="Create an execution plan for {scenario_type} in {job_name}.",
            output_template="Plan produced with affected files, tests, and risk notes."),
    LlmStep("draft", "gpt-5.5", "chatgpt", 4500, 2200, 3800, 0.0355,
            agent_role="implementation",
            system_prompt="Implement a focused code change that follows the plan.",
            input_template="Draft the code change for {scenario_key}; preserve existing behavior.",
            output_template="Patch drafted with implementation notes and expected test impact."),
    LlmStep("self-review", "claude-opus-4-7", "anthropic", 2200, 800, 900, 0.0008,
            agent_role="reviewer",
            system_prompt="Review the proposed patch for correctness and regression risk.",
            input_template="Review {job_name}; identify defects, missing tests, and simplifications.",
            output_template="Review complete with findings and approval status."),
    LlmStep("submit", "gemini-2.5-flash", "gemini", 1800, 500, 700, 0.0006,
            agent_role="release-validator",
            system_prompt="Validate final readiness for a coding-agent workflow.",
            input_template="Check release readiness for {job_name}; include CI and handoff status.",
            output_template="Release readiness check completed with final status."),
)
TOOL_STEPS = (
    ToolStep("repo.search", "fetch-diff", 0.10, 400, {"files": 3}, agent_role="repo-analyst"),
    ToolStep("ci.compile_test", "run-suite", 0.50, 180_000, {"minutes": 3}, agent_role="test-runner"),
    ToolStep("github.post_review", "submit-comment", 0.05, 200, {"comments": 1}, agent_role="reviewer"),
)
ESCALATION_TOOL = ToolStep("human_escalation", "engineering_review", 2.00, 1_800_000, {"review_minutes": 30}, agent_role="escalation-router")


def pick_outcome(rng: random.Random, scenario: Scenario | None, job_idx: int) -> Outcome:
    """Pick the business outcome and value for the selected coding workflow."""
    if scenario is None:
        raise ValueError("coding outcome selection requires a scenario")
    r = rng.random()
    n = job_idx + 11
    if r < OUTCOME_WEIGHTS["CONVERTED"]:
        vlo, vhi = scenario.value_range
        return Outcome(
            "CONVERTED", round(vlo + rng.random() * (vhi - vlo), 2),
            f"{scenario.key.upper()}-{n:03d}", "SUCCESS", "task_completed_autonomously",
        )
    if r < OUTCOME_WEIGHTS["CONVERTED"] + OUTCOME_WEIGHTS["ESCALATED"]:
        return Outcome("ESCALATED", 0.0, f"{scenario.key.upper()}-ESC-{n:03d}", "FAILED", "human_takeover")
    return Outcome("CUSTOM", 0.0, f"{scenario.key.upper()}-CXL-{n:03d}", "FAILED", "task_cancelled")


def build_metadata(scenario: Scenario | None, job_idx: int, n_llm: int, outcome: Outcome,
                   total_cost: float, job_id: str, job_name: str) -> dict[str, Any]:
    _ = job_idx
    if scenario is None:
        raise ValueError("coding metadata requires a scenario")
    return {
        "scenario_type": scenario.display_type,
        "scenario_key": scenario.key,
        "n_llm_steps": n_llm,
        "deflected_cost_usd": outcome.value if outcome.outcome_type == "CONVERTED" else 0.0,
        "agent_cost_usd": round(total_cost, 2),
        "agenticJobId": job_id,
        "agenticJobName": job_name,
        "outcome_reason": outcome.reason,
    }


def build_summary(count: int, totals: dict[str, float], elapsed: float) -> list[str]:
    total_ai = totals.get("ai", 0.0)
    total_tool = totals.get("tool", 0.0)
    converted = int(totals.get("CONVERTED", 0.0))
    escalated = int(totals.get("ESCALATED", 0.0))
    custom = int(totals.get("CUSTOM", 0.0))
    deflected = totals.get("deflected_cost_usd", 0.0)
    total_cost = total_ai + total_tool
    return [
        "",
        f"Jobs: {count}   CONVERTED: {converted}   ESCALATED: {escalated}   CUSTOM: {custom}",
        f"Deflection rate:    {(converted / count) * 100:>5.1f}%",
        f"Total agent cost:   ${total_cost:>10,.2f}  (AI ${total_ai:.2f} + tools ${total_tool:.2f})",
        f"Total deflected:    ${deflected:>10,.2f}  (engineering cost saved by autonomous completion)",
        f"Net value:          ${deflected - total_cost:>10,.2f}",
        f"Runtime: {elapsed:.2f}s",
    ]


SPEC = ExampleSpec(
    key="coding", description="AI Coding Workflow metering example.", subscriber=SUBSCRIBER,
    organization_name="Rev Demo", squad_name="engineering-platform", squad_id="squad-engineering-platform",
    product_name="AI Coding Workflow", agent_base="rev-coding",
    agentic_job_type="coding_agent", trace_type="coding_workflow",
    transaction_prefix="coding-agent", reported_by="ai-coding-agent-pipeline", job_label="Coding agent job",
    step_spacing_seconds=2, span_seconds=3600, llm_steps=LLM_STEPS, tool_steps=TOOL_STEPS,
    scenarios=SCENARIOS, escalation_tool=ESCALATION_TOOL,
    expected_escalation_rate=OUTCOME_WEIGHTS["ESCALATED"], pick_outcome=pick_outcome,
    build_metadata=build_metadata, build_summary=build_summary,
)


if __name__ == "__main__":
    run_example(SPEC)
