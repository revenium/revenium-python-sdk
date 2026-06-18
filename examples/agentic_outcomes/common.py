"""Common runtime for the Revenium AI outcome demo scripts.

Download this file beside one scenario file, then run the scenario directly:
python revenium_coding_demo.py --dry-run --count 5

Model: scenario data -> deterministic run plan -> dry-run or live emit.
Use --plan and --seed to preview exactly what a live run will send.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time as time_mod
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Optional


@dataclass(frozen=True)
class LlmStep:
    task_type: str
    model: str
    provider: str
    prompt_tokens: int
    response_tokens: int
    duration_ms: int
    cost_usd: float
    agent_role: str = ""
    system_prompt: str = "You are a specialist agent completing one step in an agentic workflow."
    input_template: str = "Run {task_type} for {job_name}."
    output_template: str = "Completed {task_type} for {job_name}."


@dataclass(frozen=True)
class ToolStep:
    tool_id: str
    operation: str
    cost_usd: float
    duration_ms: int
    metadata: dict[str, Any]
    agent_role: str = ""


@dataclass(frozen=True)
class Scenario:
    """Workflow recipe selected for a job.

    weight controls how often this scenario is picked. llm_sequence and
    tool_sequence name the steps that make this scenario's trace distinct.
    """
    key: str
    display_type: str
    weight: float
    trace_range: tuple[int, int]
    value_range: tuple[float, float]
    llm_sequence: tuple[str, ...] = ()
    tool_sequence: tuple[str, ...] = ()


@dataclass(frozen=True)
class Outcome:
    outcome_type: str | None
    value: float
    deal_id: str
    execution_status: str
    reason: str = ""


@dataclass
class JobResult:
    ai_cost: float
    tool_cost: float
    outcome_type: str | None
    metrics: dict[str, float]


@dataclass(frozen=True)
class JobPlan:
    """The exact per-job decisions that will be emitted.

    Building this first keeps --plan, --dry-run, and live runs aligned.
    """
    job_idx: int
    scenario: Scenario | None
    llm_steps: tuple[LlmStep, ...]
    tool_steps: tuple[ToolStep, ...]
    outcome: Outcome
    job_name: str


PickOutcome = Callable[[random.Random, Optional[Scenario], int], Outcome]
BuildMetadata = Callable[[Optional[Scenario], int, int, Outcome, float, str, str], dict[str, Any]]
BuildSummary = Callable[[int, dict[str, float], float], list[str]]


@dataclass(frozen=True)
class ExampleSpec:
    key: str
    description: str
    subscriber: dict[str, Any]
    organization_name: str
    squad_name: str
    squad_id: str
    product_name: str
    agent_base: str
    agentic_job_type: str
    trace_type: str
    transaction_prefix: str
    reported_by: str
    job_label: str
    step_spacing_seconds: int
    span_seconds: int
    llm_steps: tuple[LlmStep, ...]
    tool_steps: tuple[ToolStep, ...]
    scenarios: tuple[Scenario, ...] = ()
    escalation_tool: ToolStep | None = None
    expected_escalation_rate: float = 0.0
    pick_outcome: PickOutcome | None = None
    build_metadata: BuildMetadata | None = None
    build_summary: BuildSummary | None = None
    # Per-script failure-rate defaults (override CLI 0.0 default unless user passes flag).
    default_llm_failure_rate: float = 0.0
    default_tool_failure_rate: float = 0.0
    default_timeout_rate: float = 0.0


API_KEY = ""
BASE_URL = ""
METER_BASE_URL = ""
PROFITSTREAM_BASE_URL = ""
TEAM_ID = ""
METER_URL = ""
TOOL_URL = ""
OUTCOME_API_KEY = ""
_CLIENT = None
_CLIENT_SETTINGS: dict[str, str] = {}
_EMITTER_RNG = random.Random(1337)
# Collect outcome failures so the run completes (metering data lands) but failures are
# loud at the end, with non-zero exit. This is for the customer-facing example — silent
# swallowing of failures is the WRONG pattern; loud-but-non-fatal is right.
_OUTCOME_FAILURES: list[dict] = []


def reload_from_env() -> None:
    global API_KEY, BASE_URL, METER_BASE_URL, PROFITSTREAM_BASE_URL, TEAM_ID
    global METER_URL, TOOL_URL, OUTCOME_API_KEY, _CLIENT, _CLIENT_SETTINGS
    API_KEY = os.environ.get("REVENIUM_API_KEY", "")
    if API_KEY and not os.environ.get("REVENIUM_METERING_API_KEY"):
        os.environ["REVENIUM_METERING_API_KEY"] = API_KEY
    BASE_URL = os.environ.get("REVENIUM_API_BASE_URL", "https://api.revenium.io")
    METER_BASE_URL = os.environ.get("REVENIUM_METER_BASE_URL", BASE_URL)
    PROFITSTREAM_BASE_URL = os.environ.get("REVENIUM_PROFITSTREAM_BASE_URL", BASE_URL)
    TEAM_ID = os.environ.get("REVENIUM_TEAM_ID", "")
    METER_URL = f"{METER_BASE_URL}/meter/v2/ai/completions"
    TOOL_URL = f"{METER_BASE_URL}/meter/v2/tool/events"
    OUTCOME_API_KEY = os.environ.get("REVENIUM_OUTCOME_API_KEY", API_KEY)
    _CLIENT = None
    _CLIENT_SETTINGS = {
        "api_key": API_KEY,
        "meter_base_url": METER_BASE_URL,
        "profitstream_base_url": PROFITSTREAM_BASE_URL,
        "team_id": TEAM_ID,
        "outcome_api_key": OUTCOME_API_KEY,
    }


def _client():
    global _CLIENT
    if _CLIENT is None:
        from revenium_middleware.agentic_outcomes import AgenticOutcomeClient, AgenticOutcomeSettings

        if not _CLIENT_SETTINGS:
            reload_from_env()
        _CLIENT = AgenticOutcomeClient(AgenticOutcomeSettings(**_CLIENT_SETTINGS))
    assert _CLIENT is not None
    return _CLIENT


def _print_dry_run(url: str, payload: dict[str, Any]) -> None:
    # Thin wrapper over the SDK helper so the example file has a single import surface.
    from revenium_middleware.agentic_outcomes import _print_dry_run as _sdk_print
    _sdk_print("POST", url, payload)


def apply_cli_env_overrides(api_key: str | None, base_url: str | None) -> None:
    if api_key:
        os.environ["REVENIUM_API_KEY"] = api_key
    if base_url:
        os.environ["REVENIUM_API_BASE_URL"] = base_url
    reload_from_env()


def require_live_env() -> None:
    missing = [name for name, val in (("REVENIUM_API_KEY", API_KEY),) if not val]
    if missing:
        raise SystemExit(f"Missing env vars for live run: {', '.join(missing)}")


def parse_start_time(s: str) -> datetime:
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def _now_iso() -> str:
    return _iso(datetime.now(timezone.utc))


def _add_ms_iso(iso_str: str, ms: int) -> str:
    s = iso_str[:-1] + "+00:00" if iso_str.endswith("Z") else iso_str
    return _iso(datetime.fromisoformat(s).astimezone(timezone.utc) + timedelta(milliseconds=ms))


def _split_cost(total: float, input_share: float = 0.60) -> tuple[float, float]:
    in_cost = round(total * input_share, 6)
    return in_cost, round(total - in_cost, 6)


def send_completion(*, model: str, provider: str, prompt_tokens: int, response_tokens: int,
                    cost_usd: float, duration_ms: int, subscriber: dict[str, Any],
                    trace_id: str, task_type: str, agentic_job_id: str, agentic_job_name: str,
                    agent: str, agent_name: str, organization_name: str, product_name: str,
                    squad_name: str, squad_id: str, transaction_prefix: str, trace_type: str,
                    agentic_job_type: str, request_time: str | None,
                    parent_transaction_id: str | None, prompt_fields: dict[str, Any],
                    stop_reason: str, error_reason: str | None, dry_run: bool, rng: random.Random | None = None) -> dict[str, Any]:
    timestamp = request_time or _now_iso()
    ttft_ms = min(_EMITTER_RNG.randint(250, 800), max(duration_ms - 100, 50))
    in_cost, out_cost = _split_cost(cost_usd)
    tx_id = str(uuid.UUID(int=rng.getrandbits(128))) if rng else str(uuid.uuid4())
    payload = {
        "transactionId": tx_id, "traceId": trace_id, "traceType": trace_type,
        "traceName": agentic_job_name, "model": model, "provider": provider,
        "modelSource": "revenium-pricing-catalog", "costType": "AI", "operationType": "CHAT",
        "stopReason": stop_reason,
        "isStreamed": True, "inputTokenCount": prompt_tokens, "outputTokenCount": response_tokens,
        "totalTokenCount": prompt_tokens + response_tokens, "cacheReadTokenCount": 0,
        "cacheCreationTokenCount": 0, "reasoningTokenCount": 0,
        "inputTokenCost": in_cost, "outputTokenCost": out_cost,
        "totalCost": cost_usd, "currency": "USD", "requestDuration": duration_ms,
        "timeToFirstToken": ttft_ms, "requestTime": timestamp,
        "completionStartTime": _add_ms_iso(timestamp, ttft_ms),
        "responseTime": _add_ms_iso(timestamp, duration_ms), "subscriber": subscriber,
        "taskType": task_type, "transactionName": f"{transaction_prefix}-{task_type}",
        "agent": agent, "agentName": agent_name, "organizationName": organization_name,
        "productName": product_name, "squadName": squad_name, "squadId": squad_id,
        "agenticJobId": agentic_job_id, "agenticJobName": agentic_job_name,
        "agenticJobType": agentic_job_type, "agenticJobVersion": "1.0.0",
        "environment": "demo", "retryNumber": 0,
    }
    payload.update(prompt_fields)
    if parent_transaction_id:
        payload["parentTransactionId"] = parent_transaction_id
    if error_reason:
        payload["errorReason"] = error_reason
    if dry_run:
        _print_dry_run(METER_URL, payload)
        return payload
    _client().emit_completion(payload)
    return payload


def send_tool_event(*, step: ToolStep, cost_usd: float, agent: str, subscriber: dict[str, Any],
                    organization_name: str, product_name: str, trace_id: str,
                    trace_type: str, agentic_job_id: str, agentic_job_name: str, agentic_job_type: str,
                    parent_transaction_id: str | None, timestamp: str | None,
                    success: bool, error_message: str | None, dry_run: bool, rng: random.Random | None = None) -> dict[str, Any]:
    tx_id = str(uuid.UUID(int=rng.getrandbits(128))) if rng else str(uuid.uuid4())
    payload = {
        "transactionId": tx_id, "toolId": step.tool_id, "operation": step.operation,
        "durationMs": step.duration_ms, "success": success, "costUsd": cost_usd, "currency": "USD",
        "timestamp": timestamp or _now_iso(), "agent": agent, "traceName": agentic_job_name,
        "traceType": trace_type, "organizationName": organization_name,
        "productName": product_name, "subscriber": subscriber, "traceId": trace_id,
        "agenticJobId": agentic_job_id, "agenticJobName": agentic_job_name,
        "agenticJobType": agentic_job_type, "agenticJobVersion": "1.0.0",
        "environment": "demo",
    }
    if parent_transaction_id:
        payload["parentTransactionId"] = parent_transaction_id
    if step.metadata:
        payload["usageMetadata"] = step.metadata
    if error_message:
        payload["errorMessage"] = error_message
    if dry_run:
        _print_dry_run(TOOL_URL, payload)
        return payload
    _client().emit_tool_event(payload)
    return payload


def report_outcome(*, outcome: Outcome, value: float, reported_by: str,
                   agentic_job_id: str, metadata: dict[str, Any], timestamp: str | None,
                   dry_run: bool) -> None:
    global _OUTCOME_FAILURES
    payload = {
        "executionStatus": outcome.execution_status,
        "outcomeType": outcome.outcome_type or None,
        # Preserve fractional dollars. int() truncated $0.99 → 0 and stripped cents
        # from every reported value; outcomeValue MUST stay numeric, not coerced.
        "outcomeValue": float(value),
        "outcomeCurrency": "USD",
        "metadata": json.dumps(metadata) if metadata else None,
        "reportedBy": reported_by,
        "outcomeReportedAt": timestamp or _now_iso(),
    }
    payload = {k: v for k, v in payload.items() if v is not None}
    if dry_run:
        url = f"{PROFITSTREAM_BASE_URL}/profitstream/v2/api/jobs/{agentic_job_id}/outcome"
        _print_dry_run(url, payload)
        return
    try:
        _client().report_outcome(agentic_job_id, payload)
    except Exception as e:
        _OUTCOME_FAILURES.append({"job_id": agentic_job_id, "error": str(e)})


def _run_one_job(spec: ExampleSpec, item: JobPlan, job_base: datetime,
                 sleep_rng: random.Random, agent: str, org_name: str, squad_name: str,
                 subscriber: dict[str, Any], args: argparse.Namespace) -> JobResult:
    job_idx = item.job_idx
    scenario = item.scenario

    # Deterministic UUIDs for dry-run stability if seed is provided.
    seed_val = args.seed or job_base.isoformat()
    job_rng = random.Random(f"job:{seed_val}:{job_idx}")
    trace_id = str(uuid.UUID(int=job_rng.getrandbits(128)))
    job_id = str(uuid.UUID(int=job_rng.getrandbits(128)))
    if scenario:
        print(f"{spec.job_label} {job_idx + 1} ({scenario.key}) — traceId={trace_id} agenticJobId={job_id}")
    else:
        print(f"{spec.job_label} {job_idx + 1} — traceId={trace_id} agenticJobId={job_id}")
    job_name = item.job_name
    if args.tag:
        job_name = f"{job_name} ({args.tag})"
    llm_steps = item.llm_steps

    step_counter = 0

    def step_time() -> str:
        nonlocal step_counter
        ts = _iso(job_base + timedelta(seconds=step_counter * spec.step_spacing_seconds))
        step_counter += 1
        return ts

    parent_tx_id = None
    ai_cost = 0.0
    failure_rng = random.Random(f"fail:{args.seed or 'default'}:{job_idx}")
    for step in llm_steps:
        task_type = step.task_type if scenario is None else f"{scenario.key}.{step.task_type}"
        # Per-job variability: jitter tokens + duration so no two jobs are identical;
        # cost tracks the token jitter (realistic) on top of the global cost/value scales.
        # This is what makes P95/P99, outliers, anomalies, and the cost distribution real.
        pt = max(1, int(step.prompt_tokens * job_rng.uniform(0.55, 1.75)))
        rt = max(1, int(step.response_tokens * job_rng.uniform(0.55, 1.95)))
        dur = max(50, int(step.duration_ms * job_rng.uniform(0.6, 1.85)))
        tok_ratio = (pt + rt) / max(1, step.prompt_tokens + step.response_tokens)
        cost = step.cost_usd * tok_ratio * args.value_scale * args.cost_scale
        failed = _failed(failure_rng, args.llm_failure_rate)
        timed_out = (not failed) and _failed(failure_rng, args.timeout_rate)
        step_agent = _agent_name(agent, step.agent_role, args.agent_role_suffixes)
        payload = send_completion(
            model=step.model, provider=step.provider, prompt_tokens=pt,
            response_tokens=rt, cost_usd=cost, duration_ms=dur,
            subscriber=subscriber, trace_id=trace_id, task_type=task_type,
            agentic_job_id=job_id, agentic_job_name=job_name, agent=step_agent, agent_name=step_agent,
            organization_name=org_name, product_name=spec.product_name, squad_name=squad_name,
            squad_id=spec.squad_id, transaction_prefix=spec.transaction_prefix, trace_type=spec.trace_type,
            agentic_job_type=spec.agentic_job_type, parent_transaction_id=parent_tx_id,
            request_time=step_time(), prompt_fields=_prompt_fields(
                step, task_type=task_type, job_name=job_name, scenario=scenario, outcome=item.outcome,
            ),
            stop_reason="TIMEOUT" if timed_out else ("ERROR" if failed else "END"),
            error_reason="Request timed out" if timed_out else (f"{task_type} failed validation" if failed else None),
            dry_run=args.dry_run, rng=job_rng,
        )
        parent_tx_id = parent_tx_id or payload["transactionId"]
        ai_cost += cost
        _maybe_sleep(sleep_rng, 0.2, 0.8, args.sleep_scale)

    tool_cost = 0.0
    for step in item.tool_steps:
        cost = step.cost_usd * job_rng.uniform(0.6, 1.8) * args.value_scale * args.cost_scale
        failed = _failed(failure_rng, args.tool_failure_rate)
        step_agent = _agent_name(agent, step.agent_role, args.agent_role_suffixes)
        payload = send_tool_event(
            step=step, cost_usd=cost, agent=step_agent, subscriber=subscriber,
            organization_name=org_name, product_name=spec.product_name, trace_id=trace_id,
            trace_type=spec.trace_type,
            agentic_job_id=job_id, agentic_job_name=job_name,
            agentic_job_type=spec.agentic_job_type, parent_transaction_id=parent_tx_id,
            timestamp=step_time(), success=not failed,
            error_message=f"{step.tool_id} returned an error" if failed else None,
            dry_run=args.dry_run, rng=job_rng,
        )
        tool_cost += cost
        _maybe_sleep(sleep_rng, 0.2, 0.8, args.sleep_scale)

    outcome = item.outcome
    scaled_value = outcome.value * args.value_scale * args.outcome_scale
    metadata_outcome = Outcome(outcome.outcome_type, scaled_value, outcome.deal_id, outcome.execution_status, outcome.reason)
    if outcome.outcome_type == "ESCALATED" and spec.escalation_tool:
        step = spec.escalation_tool
        cost = step.cost_usd * job_rng.uniform(0.6, 1.8) * args.value_scale * args.cost_scale
        step_agent = _agent_name(agent, step.agent_role, args.agent_role_suffixes)
        payload = send_tool_event(
            step=step, cost_usd=cost, agent=step_agent, subscriber=subscriber,
            organization_name=org_name, product_name=spec.product_name, trace_id=trace_id,
            trace_type=spec.trace_type,
            agentic_job_id=job_id, agentic_job_name=job_name,
            agentic_job_type=spec.agentic_job_type, parent_transaction_id=parent_tx_id,
            timestamp=step_time(), success=True, error_message=None, dry_run=args.dry_run, rng=job_rng,
        )
        tool_cost += cost

    metadata = spec.build_metadata(scenario, job_idx, len(llm_steps), metadata_outcome, ai_cost + tool_cost, job_id, job_name) if spec.build_metadata else {}
    # report_outcome catches and records its own failures in _OUTCOME_FAILURES;
    # an outer try/except here would only ever fire on programming errors, which
    # should surface, not be silently swallowed.
    report_outcome(
        outcome=outcome, value=scaled_value, reported_by=spec.reported_by,
        agentic_job_id=job_id, metadata=metadata, timestamp=step_time(), dry_run=args.dry_run
    )
    return JobResult(ai_cost, tool_cost, outcome.outcome_type, metadata)


def _agent_name(agent: str, role: str, use_role_suffix: bool = False) -> str:
    if not use_role_suffix:
        return agent
    short = _short_role(role)
    role_suffix = f"-{short}" if short else ""
    return f"{agent}{role_suffix}"


def _short_role(role: str) -> str:
    """Compact role strings for UI readability."""
    raw = (role or "").strip().lower()
    if not raw:
        return ""
    mapping = {
        "orchestrator": "orch",
        "implementation": "impl",
        "reviewer": "rev",
        "release-validator": "relval",
        "repo-analyst": "repo",
        "test-runner": "ci",
        "kb-retriever": "kb",
        "response-drafter": "draft",
        "quality-validator": "qa",
        "crm-analyst": "crm",
        "ticket-updater": "ticket",
        "escalation-router": "escalate",
    }
    if raw in mapping:
        return mapping[raw]
    # Keep alnum + hyphen, then cap length.
    cleaned = "".join(ch for ch in raw if ch.isalnum() or ch == "-").strip("-")
    return cleaned[:12]


def _prompt_fields(step: LlmStep, *, task_type: str, job_name: str, scenario: Scenario | None,
                   outcome: Outcome) -> dict[str, Any]:
    values = {
        "task_type": task_type,
        "job_name": job_name,
        "scenario_key": scenario.key if scenario else "default",
        "scenario_type": scenario.display_type if scenario else "standard",
        "outcome_reason": outcome.reason or "pending",
    }
    return {
        "systemPrompt": step.system_prompt.format(**values),
        "inputMessages": json.dumps([
            {"role": "user", "content": step.input_template.format(**values)}
        ]),
        "outputResponse": step.output_template.format(**values),
        "promptsTruncated": False,
    }


def _failed(rng: random.Random, rate: float) -> bool:
    return rate > 0 and rng.random() < rate


def _maybe_sleep(rng: random.Random, lo: float, hi: float, scale: float) -> None:
    if scale > 0:
        time_mod.sleep(rng.uniform(lo, hi) * scale)


def run_example(spec: ExampleSpec) -> None:
    parser = argparse.ArgumentParser(description=spec.description)
    parser.add_argument("--count", type=int, default=5, help="Number of jobs to run")
    parser.add_argument("--seed", type=str, help="Seed for deterministic plan")
    parser.add_argument("--tag", type=str, help="Tag for job names")
    parser.add_argument("--dry-run", action="store_true", help="Print payloads only")
    parser.add_argument("--plan", action="store_true", help="Show decisions only")
    parser.add_argument("--api-key", type=str, help="Override REVENIUM_API_KEY")
    parser.add_argument("--base-url", type=str, help="Override REVENIUM_API_BASE_URL")
    parser.add_argument("--backdate-days", type=int, default=30, help="Days to backdate if no start time")
    parser.add_argument("--start-time", type=str, help="ISO8601 start time")
    parser.add_argument("--llm-failure-rate", type=float, default=None)
    parser.add_argument("--tool-failure-rate", type=float, default=None)
    parser.add_argument("--timeout-rate", type=float, default=None)
    parser.add_argument("--demo-variability", action="store_true", help="Opt-in to small failure rates")
    parser.add_argument("--value-scale", type=float, default=1.0, help="Scale business values")
    parser.add_argument("--cost-scale", type=float, default=1.0, help="Scale model/tool costs")
    parser.add_argument("--outcome-scale", type=float, default=1.0, help="Scale outcome values")
    parser.add_argument("--sleep-scale", type=float, default=0.0, help="Scale sleeps between steps")
    parser.add_argument("--agent-role-suffixes", action="store_true", help="Append role to agent name")
    args = parser.parse_args()

    if args.count < 1:
        parser.error("--count must be at least 1")

    apply_cli_env_overrides(args.api_key, args.base_url)
    if not args.dry_run and not args.plan:
        require_live_env()

    if args.demo_variability:
        if args.llm_failure_rate is None:
            args.llm_failure_rate = max(spec.default_llm_failure_rate, 0.03)
        if args.tool_failure_rate is None:
            args.tool_failure_rate = max(spec.default_tool_failure_rate, 0.02)
        if args.timeout_rate is None:
            args.timeout_rate = max(spec.default_timeout_rate, 0.01)

    if args.llm_failure_rate is None:
        args.llm_failure_rate = spec.default_llm_failure_rate
    if args.tool_failure_rate is None:
        args.tool_failure_rate = spec.default_tool_failure_rate
    if args.timeout_rate is None:
        args.timeout_rate = spec.default_timeout_rate

    for name in ("llm_failure_rate", "tool_failure_rate", "timeout_rate"):
        value = getattr(args, name)
        if not 0 <= value <= 1:
            raise SystemExit(f"--{name.replace('_', '-')} must be between 0 and 1")

    rng_seed = args.seed or "default-seed"
    plan_rng = random.Random(rng_seed)
    
    start_dt = parse_start_time(args.start_time) if args.start_time else \
               datetime.now(timezone.utc) - timedelta(days=args.backdate_days)

    print(f"Running {spec.key} demo: {args.count} jobs starting {start_dt.isoformat()}")
    if args.dry_run:
        print("[DRY-RUN] No data will be sent to Revenium.")
    elif args.plan:
        print("[PLAN] Decisions only; no HTTP ops.")

    plans = []
    for i in range(args.count):
        scenario = plan_rng.choices(spec.scenarios, weights=[s.weight for s in spec.scenarios])[0] if spec.scenarios else None
        llm_steps = [s for s in spec.llm_steps if s.task_type in scenario.llm_sequence] if scenario else list(spec.llm_steps)
        tool_steps = [s for s in spec.tool_steps if s.tool_id in scenario.tool_sequence] if scenario else list(spec.tool_steps)
        outcome = spec.pick_outcome(plan_rng, scenario, i) if spec.pick_outcome else Outcome("SUCCESS", 0.0, "", "SUCCESS")
        plans.append(JobPlan(i, scenario, tuple(llm_steps), tuple(tool_steps), outcome, f"{spec.job_label} {i+1:03}"))

    if args.plan:
        for p in plans:
            print(f"Job {p.job_idx+1}: scenario={p.scenario.key if p.scenario else 'None'} steps={len(p.llm_steps)} outcome={p.outcome.outcome_type}")
        return

    sleep_rng = random.Random(f"sleep:{rng_seed}")
    totals: dict[str, float] = {"ai": 0.0, "tool": 0.0}
    elapsed_start = time_mod.time()

    for item in plans:
        job_base = start_dt + timedelta(seconds=item.job_idx * spec.span_seconds / max(1, args.count))
        res = _run_one_job(spec, item, job_base, sleep_rng, spec.agent_base, spec.organization_name, spec.squad_name, spec.subscriber, args)
        totals["ai"] += res.ai_cost
        totals["tool"] += res.tool_cost
        if res.outcome_type:
            totals[res.outcome_type] = totals.get(res.outcome_type, 0.0) + 1.0
        for k, v in res.metrics.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                totals[k] = totals.get(k, 0.0) + float(v)

    elapsed = time_mod.time() - elapsed_start

    if spec.build_summary:
        for line in spec.build_summary(args.count, totals, elapsed):
            print(line)
    else:
        converted_count = int(totals.get("CONVERTED", 0.0)) + int(totals.get("DEFLECTED", 0.0))
        unsuccessful_count = int(totals.get("UNSUCCESSFUL", 0.0))
        print(f"Jobs: {args.count}   Converted: {converted_count}   Unsuccessful: {unsuccessful_count}")
        print(f"Emitted: {sum(len(p.llm_steps) for p in plans)} completions + {sum(len(p.tool_steps) for p in plans)} tool events + {len(plans)} outcomes")
        print(f"Per-job AI cost:   $ {totals['ai']/max(1, args.count):10.2f}")
        print(f"Per-job tool cost: $ {totals['tool']/max(1, args.count):10.2f}")
        print(f"Total agent cost:  $ {totals['ai'] + totals['tool']:10.2f}")

    if _OUTCOME_FAILURES:
        print(f"WARNING: {len(_OUTCOME_FAILURES)} outcome reporting failures:")
        for fail in _OUTCOME_FAILURES[:5]:
            print(f"  - Job {fail['job_id']}: {fail['error']}")
        raise SystemExit(1)
