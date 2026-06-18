"""AI Sales Agent outcome demo for Revenium."""
from __future__ import annotations

import random
from typing import Any

from common import ExampleSpec, LlmStep, Outcome, Scenario, ToolStep, run_example

# CUSTOMIZATION POINTS:
# - change SUBSCRIBER, LLM_STEPS, TOOL_STEPS, and the conversion rule in pick_outcome
# - keep business-specific outcome metadata in build_metadata
SUBSCRIBER = {"email": "demo-runner@example.com"}
LLM_STEPS = (
    LlmStep("sales.prospecting", "claude-sonnet-4-5", "anthropic", 1200, 400, 1500, 0.0014,
            system_prompt="Identify high-fit prospects for a sales agent.",
            input_template="Review the next lead for {job_name} and summarize fit signals.",
            output_template="Lead fit scored and routed to qualification."),
    LlmStep("sales.qualification", "gpt-4o-mini", "chatgpt", 2200, 700, 2200, 0.0008,
            system_prompt="Qualify pipeline opportunities using CRM and product-fit context.",
            input_template="Qualify the prospect for {job_name}; include buying intent and risk.",
            output_template="Qualification complete with fit, urgency, and next best action."),
    LlmStep("sales.close", "claude-opus-4-7", "anthropic", 3100, 1100, 3600, 0.0355,
            system_prompt="Generate a concise close plan for a qualified sales opportunity.",
            input_template="Prepare final outreach for {job_name}; outcome target is {outcome_reason}.",
            output_template="Close plan completed with recommended offer and follow-up."),
)
TOOL_STEPS = (
    ToolStep("zoominfo.enrich", "enrich", 1.00, 900, {"records_returned": 1}),
    ToolStep("apollo.search", "search", 0.50, 600, {"records_returned": 12}),
    ToolStep("human_escalation", "sdr_handoff", 2.00, 1_800_000, {"sdr_minutes": 30}),
)


def pick_outcome(rng: random.Random, scenario: Scenario | None, job_idx: int) -> Outcome:
    """Sales funnel: 12 converted deals + 6 unsuccessful per 100 leads; rest lost."""
    _ = rng, scenario
    num = job_idx + 1
    bucket = job_idx % 100
    if bucket < 12:
        return Outcome("CONVERTED", 300.00, f"ACME-2026-Q2-{num:03d}", "SUCCESS", "deal_closed")
    if bucket < 18:
        return Outcome("UNSUCCESSFUL", 0.0, f"ACME-2026-Q2-FAIL-{num:03d}", "FAILED", "deal_unsuccessful")
    return Outcome(None, 0.0, f"ACME-2026-Q2-LOST-{num:03d}", "SUCCESS", "deal_lost")


def build_metadata(scenario: Scenario | None, job_idx: int, n_llm: int, outcome: Outcome,
                   total_cost: float, job_id: str, job_name: str) -> dict[str, Any]:
    _ = scenario, job_idx, n_llm
    return {
        "opportunity_stage": "closed_won" if outcome.outcome_type == "CONVERTED" else "closed_lost",
        "contract_term_months": 24,
        "ae_email": SUBSCRIBER["email"],
        "agent_cost_usd": round(total_cost, 2),
        "agenticJobId": job_id,
        "agenticJobName": job_name,
        "outcome_reason": outcome.reason,
    }


def build_summary(count: int, totals: dict[str, float], elapsed: float) -> list[str]:
    total_ai = totals.get("ai", 0.0)
    total_tool = totals.get("tool", 0.0)
    converted = int(totals.get("CONVERTED", 0.0))
    unsuccessful = int(totals.get("UNSUCCESSFUL", 0.0))
    return [
        "",
        f"Jobs: {count}   Converted: {converted}   Unsuccessful: {unsuccessful}",
        f"Emitted: {count * len(LLM_STEPS)} completions + {count * len(TOOL_STEPS)} tool events + {count} outcomes",
        f"Per-job AI cost:   ${total_ai / count:>10,.2f}",
        f"Per-job tool cost: ${total_tool / count:>10,.2f}",
        f"Total agent cost:  ${total_ai + total_tool:>10,.2f}",
        f"Runtime: {elapsed:.2f}s",
    ]


SPEC = ExampleSpec(
    key="sales", description="AI Sales Agent metering example.", subscriber=SUBSCRIBER,
    organization_name="Rev Demo", squad_name="sales-automation", squad_id="squad-sales-automation",
    product_name="AI Sales Agent", agent_base="rev-sales",
    agentic_job_type="sales_agent", trace_type="sales_workflow",
    transaction_prefix="sales-agent", reported_by="ai-sales-agent-pipeline", job_label="Sales agent job",
    step_spacing_seconds=2, span_seconds=3600, llm_steps=LLM_STEPS, tool_steps=TOOL_STEPS,
    pick_outcome=pick_outcome, build_metadata=build_metadata, build_summary=build_summary,
)


if __name__ == "__main__":
    run_example(SPEC)
