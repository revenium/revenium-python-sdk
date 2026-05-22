"""AI Customer Support Agent outcome demo for Revenium."""
from __future__ import annotations

import random
from typing import Any

from common import ExampleSpec, LlmStep, Outcome, Scenario, ToolStep, run_example

# CUSTOMIZATION POINTS:
# - SCENARIOS controls support workflow mix, step sequences, and deflection value ranges
# - LLM_STEPS and TOOL_STEPS define reusable step templates referenced by scenario
# - OUTCOME_WEIGHTS controls deflection, escalation, and conversion rates
SUBSCRIBER = {"email": "demo-runner@example.com"}
SCENARIOS = (
    Scenario(
        "ticket-triage", "Ticket-Triage", 3.0, (2, 7), (2.0, 50.0),
        ("classify", "kb-search", "verify"),
        ("kb.search", "crm.lookup", "zendesk.ticket"),
    ),
    Scenario(
        "ticket-resolution", "Ticket-Resolution", 2.5, (2, 7), (5.0, 80.0),
        ("classify", "kb-search", "draft-response", "verify"),
        ("kb.search", "crm.lookup", "zendesk.ticket"),
    ),
    Scenario(
        "escalation-handling", "Escalation-Handling", 1.5, (2, 7), (15.0, 200.0),
        ("classify", "draft-response", "verify"),
        ("crm.lookup", "zendesk.ticket"),
    ),
)
OUTCOME_WEIGHTS = {"DEFLECTED": 0.80, "ESCALATED": 0.08, "CONVERTED": 0.12}
UPSELL_VALUE_RANGE = (50.0, 300.0)
LLM_STEPS = (
    LlmStep("classify", "claude-sonnet-4-5", "anthropic", 1200, 300, 800, 0.0014,
            agent_role="orchestrator",
            system_prompt="Classify an inbound customer support ticket.",
            input_template="Classify {job_name}; identify intent, urgency, and routing.",
            output_template="Ticket classified with urgency and likely resolution path."),
    LlmStep("kb-search", "gpt-4o-mini", "chatgpt", 1500, 400, 600, 0.0006,
            agent_role="kb-retriever",
            system_prompt="Find relevant knowledge-base guidance for a support ticket.",
            input_template="Search support knowledge for {scenario_type} in {job_name}.",
            output_template="Relevant knowledge-base articles selected and summarized."),
    LlmStep("draft-response", "claude-opus-4-7", "anthropic", 3500, 1500, 2800, 0.0263,
            agent_role="response-drafter",
            system_prompt="Draft a customer-safe support response.",
            input_template="Draft the response for {job_name}; expected outcome is {outcome_reason}.",
            output_template="Customer response drafted with next steps and guardrails."),
    LlmStep("verify", "gemini-2.5-flash", "gemini", 1200, 300, 500, 0.0004,
            agent_role="quality-validator",
            system_prompt="Verify the support response before sending.",
            input_template="Verify final response for {job_name}; check tone and policy fit.",
            output_template="Response verified and ready for customer delivery."),
)
TOOL_STEPS = (
    ToolStep("kb.search", "search", 0.002, 1200, {"articles_returned": 5}, agent_role="kb-retriever"),
    ToolStep("crm.lookup", "get_customer", 0.003, 800, {"plan": "enterprise"}, agent_role="crm-analyst"),
    ToolStep("zendesk.ticket", "update_ticket", 0.004, 600, {"status": "in_progress"}, agent_role="ticket-updater"),
)
ESCALATION_TOOL = ToolStep("human_escalation", "ticket_review", 2.0, 600_000, {"reviewer": "l2-agent"}, agent_role="escalation-router")
OUTCOME_REASONS = {
    "DEFLECTED": ["ticket_resolved_without_human", "kb_match_first_response", "self_service_link_provided"],
    "ESCALATED": ["routed_to_human_agent", "complex_issue_requires_l2", "customer_requested_human"],
    "CONVERTED": ["upsell_during_support_session", "renewal_extended", "addon_purchased"],
}
def pick_outcome(rng: random.Random, scenario: Scenario | None, job_idx: int) -> Outcome:
    """Pick the support outcome and value for the selected workflow."""
    if scenario is None:
        raise ValueError("support outcome selection requires a scenario")
    r = rng.random()
    n = job_idx + 11
    raw_key = scenario.key.upper()
    key_clean = raw_key[len("TICKET-"):] if raw_key.startswith("TICKET-") else raw_key
    reason = ""
    if r < OUTCOME_WEIGHTS["DEFLECTED"]:
        vlo, vhi = scenario.value_range
        value = round(vlo + rng.random() * (vhi - vlo), 2)
        reason = rng.choice(OUTCOME_REASONS["DEFLECTED"])
        outcome = Outcome("DEFLECTED", value, f"TICKET-{key_clean}-{n:04d}", "SUCCESS", reason)
    elif r < OUTCOME_WEIGHTS["DEFLECTED"] + OUTCOME_WEIGHTS["ESCALATED"]:
        reason = rng.choice(OUTCOME_REASONS["ESCALATED"])
        outcome = Outcome("ESCALATED", 0.0, f"TICKET-{key_clean}-ESC-{n:04d}", "FAILED", reason)
    else:
        ulo, uhi = UPSELL_VALUE_RANGE
        value = round(ulo + rng.random() * (uhi - ulo), 2)
        reason = rng.choice(OUTCOME_REASONS["CONVERTED"])
        outcome = Outcome("CONVERTED", value, f"TICKET-{key_clean}-UPSELL-{n:04d}", "SUCCESS", reason)
    return outcome


def build_metadata(scenario: Scenario | None, job_idx: int, n_llm: int, outcome: Outcome,
                   total_cost: float, job_id: str, job_name: str) -> dict[str, Any]:
    _ = job_idx
    if scenario is None:
        raise ValueError("support metadata requires a scenario")
    return {
        "scenario_type": scenario.display_type,
        "scenario_key": scenario.key,
        "n_llm_steps": n_llm,
        "deflected_human_cost_usd": outcome.value if outcome.outcome_type == "DEFLECTED" else 0.0,
        "upsell_value_usd": outcome.value if outcome.outcome_type == "CONVERTED" else 0.0,
        "agent_cost_usd": round(total_cost, 2),
        "agenticJobId": job_id,
        "agenticJobName": job_name,
        "outcome_reason": outcome.reason,
    }


def build_summary(count: int, totals: dict[str, float], elapsed: float) -> list[str]:
    total_ai = totals.get("ai", 0.0)
    total_tool = totals.get("tool", 0.0)
    deflected_count = int(totals.get("DEFLECTED", 0.0))
    escalated = int(totals.get("ESCALATED", 0.0))
    converted = int(totals.get("CONVERTED", 0.0))
    deflected = totals.get("deflected_human_cost_usd", 0.0)
    upsell = totals.get("upsell_value_usd", 0.0)
    total_cost = total_ai + total_tool
    return [
        "",
        f"Tickets: {count}   DEFLECTED: {deflected_count}   ESCALATED: {escalated}   CONVERTED: {converted}",
        f"Deflection rate:    {(deflected_count / count) * 100:>5.1f}%",
        f"Total agent cost:   ${total_cost:>10,.2f}  (AI ${total_ai:.2f} + tools ${total_tool:.2f})",
        f"Total deflected:    ${deflected:>10,.2f}  (human-agent cost saved by autonomous deflection)",
        f"Total upsell:       ${upsell:>10,.2f}  (revenue from CONVERTED outcomes)",
        f"Net value:          ${deflected + upsell - total_cost:>10,.2f}",
        f"Runtime: {elapsed:.2f}s",
    ]


SPEC = ExampleSpec(
    key="support", description="AI Customer Support Agent metering example.", subscriber=SUBSCRIBER,
    organization_name="Rev Demo", squad_name="customer-success", squad_id="squad-customer-success",
    product_name="AI Customer Support Agent", agent_base="rev-support",
    agentic_job_type="support_agent", trace_type="support_workflow",
    transaction_prefix="support-agent", reported_by="ai-support-agent-pipeline", job_label="Support agent job",
    step_spacing_seconds=2, span_seconds=3600, llm_steps=LLM_STEPS, tool_steps=TOOL_STEPS,
    scenarios=SCENARIOS, escalation_tool=ESCALATION_TOOL,
    expected_escalation_rate=OUTCOME_WEIGHTS["ESCALATED"], pick_outcome=pick_outcome,
    build_metadata=build_metadata, build_summary=build_summary,
)


if __name__ == "__main__":
    run_example(SPEC)
