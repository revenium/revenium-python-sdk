# Revenium Python SDK

[![PyPI version](https://img.shields.io/pypi/v/revenium-python-sdk.svg)](https://pypi.org/project/revenium-python-sdk/)
[![Python Versions](https://img.shields.io/pypi/pyversions/revenium-python-sdk.svg)](https://pypi.org/project/revenium-python-sdk/)
[![Documentation](https://img.shields.io/badge/docs-revenium.io-blue)](https://docs.revenium.io)
[![Website](https://img.shields.io/badge/website-revenium.ai-blue)](https://www.revenium.ai)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

The official Revenium Python SDK — unified AI metering middleware for deeply attributed AI usage metrics. Supports OpenAI, Anthropic, Google (Gemini/Vertex AI), fal.ai, Ollama, LiteLLM, and Perplexity.

## Features

- **Unified SDK**: Single package with middleware for all major AI providers — install only what you need
- **Zero Code Changes**: Drop-in integration — just import and all API calls are automatically metered
- **Streaming Support**: Full streaming support for all providers (both sync and async)
- **Decorator Support**: `@revenium_metadata` for automatic metadata injection and `@revenium_meter` for selective metering
- **Tool Metering**: `@meter_tool` to meter arbitrary tool/function calls alongside LLM API metering
- **Prompt Capture**: Optional capture of prompts and responses for analytics and debugging
- **Terminal Summary**: Real-time cost and usage summaries in your terminal (human-readable or JSON)
- **Distributed Tracing**: Built-in trace visualization fields for cross-service observability
- **Asynchronous Processing**: Background thread management for non-blocking metering operations
- **Graceful Shutdown**: Ensures all metering data is properly sent even during application shutdown
- **Thread-Safe**: Production-ready with `contextvars`-based context management for concurrent applications

## Supported Providers

| Provider | Extra | Install Command |
|----------|-------|----------------|
| OpenAI | `openai` | `pip install revenium-python-sdk[openai]` |
| Azure OpenAI | `openai` | `pip install revenium-python-sdk[openai]` |
| Anthropic | `anthropic` | `pip install revenium-python-sdk[anthropic]` |
| AWS Bedrock (Anthropic) | `anthropic` | `pip install revenium-python-sdk[anthropic]` |
| Google Gemini | `google-genai` | `pip install revenium-python-sdk[google-genai]` |
| Google Vertex AI | `google-vertex` | `pip install revenium-python-sdk[google-vertex]` |
| Ollama | `ollama` | `pip install revenium-python-sdk[ollama]` |
| LiteLLM (Client) | `litellm` | `pip install revenium-python-sdk[litellm]` |
| LiteLLM (Proxy) | `litellm-proxy` | `pip install revenium-python-sdk[litellm-proxy]` |
| Perplexity (via OpenAI) | `perplexity-openai` | `pip install revenium-python-sdk[perplexity-openai]` |
| Perplexity (Native SDK) | `perplexity-native` | `pip install revenium-python-sdk[perplexity-native]` |
| fal.ai | `fal` | `pip install revenium-python-sdk[fal]` |
| LangChain | `langchain` | `pip install revenium-python-sdk[langchain]` |
| Griptape | `griptape` | `pip install revenium-python-sdk[griptape]` |

## Feature Matrix

| Feature | OpenAI | Anthropic | Google | Ollama | LiteLLM | Perplexity | fal.ai |
|---------|--------|-----------|--------|--------|---------|------------|--------|
| Chat Completions | Yes | Yes | Yes | Yes | Yes | Yes | Yes |
| Streaming | Yes | Yes | Yes | Yes | Yes | Yes | Yes |
| Embeddings | Yes | - | Yes | Yes | Yes | - | - |
| Vision/Multimodal | Yes | Yes | Yes | - | Yes | - | Yes |
| Image Generation | - | - | Yes | - | - | - | Yes |
| Video Generation | - | - | Yes | - | - | - | Yes |
| Prompt Capture | Yes | Yes | Yes | - | Yes | - | - |
| Terminal Summary | Yes | Yes | Yes | Yes | Yes | - | - |
| Azure / Bedrock | Azure | Bedrock | Vertex AI | - | All | - | - |
| LangChain Integration | Yes | - | - | - | - | - | - |
| Griptape Integration | Yes | Yes | - | Yes | Yes | - | - |
| CrewAI Integration | - | - | - | - | Yes | - | - |
| Proxy Mode | - | - | - | - | Yes | - | - |

## Installation

```bash
# Core SDK only
pip install revenium-python-sdk

# With a specific provider
pip install revenium-python-sdk[openai]

# Multiple providers
pip install "revenium-python-sdk[openai,anthropic,ollama]"
```

## Quick Start

### 1. Configure Environment Variables

Create a `.env` file in your project directory:

```env
# Required
REVENIUM_METERING_API_KEY=hak_your_revenium_api_key_here
REVENIUM_METERING_BASE_URL=https://api.revenium.ai

# Provider API keys (set whichever you use)
OPENAI_API_KEY=sk-your_openai_key
ANTHROPIC_API_KEY=sk-ant-your_anthropic_key
GOOGLE_API_KEY=your_google_key
PERPLEXITY_API_KEY=pplx_your_key
FAL_KEY=your_fal_key
FIREWORKS_API_KEY=your_fireworks_key

# Optional
# REVENIUM_LOG_LEVEL=DEBUG
```

### 2. Import and Use

Just import the middleware for your provider. That's it - all API calls are automatically metered:

```python
from dotenv import load_dotenv
load_dotenv()

import openai
import revenium_middleware.openai  # Auto-initializes on import

client = openai.OpenAI()
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Hello!"}]
)
print(response.choices[0].message.content)
# Usage data automatically sent to Revenium
```

---

## Metering Error Visibility

Metering runs in background threads and never raises into your code path — a
metering failure will never break your AI calls. To make failures observable
anyway, the SDK provides two mechanisms:

**Subscribe to failures** with a callback:

```python
import revenium_middleware

@revenium_middleware.on_metering_error
def alert_on_metering_failure(event):
    # event.error      -- the exception (e.g. an HTTP 401/500 from Revenium)
    # event.operation  -- "completion", "image", "tool", ... (may be None)
    # event.timestamp  -- UTC datetime of the failure
    my_monitoring.notify(f"Revenium metering failed: {event.error}")
```

Callbacks run on the background metering thread; exceptions they raise are
suppressed and logged, so they can never disrupt your application.

Authentication headers (`x-api-key`, `authorization`) on any HTTP
request/response attached to the exception are redacted before the error is
exposed to callbacks or `last_error`.

**Poll the status counters:**

```python
status = revenium_middleware.get_metering_status()
print(status.success_count)   # events delivered successfully
print(status.error_count)     # delivery failures
print(status.last_error)      # most recent exception, or None
print(status.last_error_at)   # UTC datetime of the most recent failure
```

`reset_metering_status()` zeroes the counters and clears registered
callbacks; `remove_metering_error_callback(cb)` unsubscribes a single one.

Failures are also logged at **ERROR** level on the `revenium_middleware`
logger, including HTTP 4xx/5xx responses, a missing
`REVENIUM_METERING_API_KEY`, and a provider middleware that fails to import
when the provider's SDK is installed.

---

## Agentic Outcomes (Outcome-Based Metering)

Emit per-agent terminal outcomes (`CONVERTED`, `DEFLECTED`, `ESCALATED`) alongside completion and tool-event records, so dashboards show business value next to AI cost.

> **You need a write-scope key (`rev_sk_`) to use the agentic outcomes API.** Metering keys (`rev_mk_`) can only meter completions and tool events — they cannot report, amend, or read job outcomes, and the SDK rejects them client-side before any HTTP request is made. Key resolution: explicit `api_key=` > `REVENIUM_WRITE_API_KEY` > `REVENIUM_OUTCOME_API_KEY` (deprecated fallback) > `REVENIUM_METERING_API_KEY`.

### Job-Type Economics and Outcome Facts

Keep a metering key for AI telemetry and a separate write key for outcomes and
job-type configuration. A registered `valuePerUnit` rule takes precedence over
an outcome's `outcome_value`; the backend never sums the two value sources.

```python
from revenium_middleware import (
    Baseline, JobTypeEconomics, PeriodFactEntry, create_baseline,
    report_period_facts, upsert_job_type_economics,
)

upsert_job_type_economics("claim", JobTypeEconomics(
    unit_metric_key="completed_claims", unit_label="claim",
    metrics=[
        {
            "key": "completed_claims", "type": "COUNT",
            "direction": "HIGHER_IS_BETTER", "aggregation": "SUM",
            "resolution": "PER_JOB",
        },
        {
            "key": "claims_processed", "type": "COUNT",
            "direction": "HIGHER_IS_BETTER", "aggregation": "SUM",
            "resolution": "PERIOD",
        },
    ],
    dimensions=[{"key": "region", "allowedValues": ["us", "ca"]}],
    monetization={
        "metricKey": "completed_claims", "valuePerUnit": 4.25,
        "currency": "USD", "category": "COST_AVOIDED", "basis": "REALIZED",
    },
))
create_baseline("claim", Baseline(
    effective_from="2026-08-01T00:00:00Z", cost_per_unit=4.25, currency="USD",
))
report_period_facts("claim", [PeriodFactEntry(
    period_start="2026-08-01T00:00:00Z", period_end="2026-09-01T00:00:00Z",
    dimension_key="region", dimension_value="us",
    key="claims_processed", value=1280,
)])
```

`effective_from` is the only required field on a baseline; every other field
is optional, and a baseline without it is rejected. A job type must be
declared with `upsert_job_type_economics` before it accepts baselines or
facts, and `report_period_facts` accepts only metrics declared with
`"resolution": "PERIOD"`.

Job economics currency values must be USD. Baselines and period facts use
server-supplied attribution when their provenance,
reporter, and source fields are omitted. Set those fields only when you need an
explicit override. Economics metric directions are
`HIGHER_IS_BETTER` or `LOWER_IS_BETTER`; monetization categories are
`REVENUE`, `COST_AVOIDED`, `TIME_SAVED`, and `LEADING_VALUE`, with a
`REALIZED` or `EXPECTED` basis.

Use `CUSTOMER_DECLARED` or `MEASURED` for a baseline override. Use `MEASURED`,
`SELF_REPORTED`, or `DERIVED` for a period fact override.

Facts are append-only and keyed on the period, dimension and metric key
together. Re-appending that tuple supersedes the active fact, and the server
requires `reason=` on the entry when it does.

### JobContext

`JobContext` is the recommended high-level API: every AI call made inside the block is automatically metered against the job (all provider middlewares pick the job fields up from context), and the job's business outcome is reported when the work is done.

```python
from revenium_middleware import JobContext

with JobContext("loan-app-12345", type="loan_processing", version="2.1") as job:
    response = client.chat.completions.create(...)  # metered against the job automatically
    job.report_outcome(
        execution_status="SUCCESS",  # SUCCESS | FAILED | CANCELLED
        outcome_type="CONVERTED",
        outcome_value=500.0,
        outcome_currency="USD",
    )
```

- **Async:** `async with JobContext(...) as job:` works identically.
- **Auto-FAILED:** if an unhandled exception escapes the block before an outcome was reported, the context automatically reports `execution_status="FAILED"` (error message and class in metadata) and always re-raises the original exception.
- **Blocking:** outcome calls are synchronous HTTP requests with retries; tune how long they may block with the `retry_attempts`, `retry_initial_seconds`, and `retry_max_seconds` arguments, accepted by the `JobContext` constructor, `JobContext.attach()`, `get_outcome_history()`, and the CrewAI wrapper's `report_job_outcome`/`amend_job_outcome`.
- **Team resolution:** explicit `team_id=` > `REVENIUM_TEAM_ID` > automatic resolution from the API key; `OutcomeReportingError` is raised if none of these yields a team.
- **Nesting:** a nested `JobContext` is a different job (replace, not merge); exiting the inner context restores the outer job's fields.

To tag AI calls with job fields without a context manager, use per-call `usage_metadata={"agentic_job_id": ...}`, the `@track_job` decorator (LiteLLM), or the process-wide `REVENIUM_AGENTIC_JOB_*` environment variables — see [Optional Environment Variables](#optional-environment-variables).

### Amending an Outcome

Outcomes are amendable: when the business result changes after the fact, amend the recorded outcome instead of re-reporting it. `JobContext.attach()` returns a lightweight handle to an existing job — it is not entered as a context manager and does not touch AI-call scoping — so amendments work from a different process than the one that ran the job.

```python
from revenium_middleware import JobContext, get_outcome_history

# Two weeks after the agent converted the lead at $500,
# the customer expands to the annual plan.
job = JobContext.attach("sales-lead-8842")
job.amend_outcome(
    reason="Customer expanded to the annual plan after the initial conversion",
    outcome_value=750.0,
)
job.close()

history = get_outcome_history("sales-lead-8842")
# List[JobOutcomeAmendment], ordered by amendment_sequence (1 = the initial report)
```

`amend_outcome()` takes `reason` — the amendment's audit justification, still the first positional argument — plus the same optional fields as `report_outcome()` (`execution_status`, `outcome_type`, `outcome_value`, `outcome_currency`, `metadata`, `reported_by`, `outcome_reason`, `metrics`), and returns the updated job as a dict.

- **Detecting a lost update:** `report_outcome()` and `amend_outcome()` record the job's `entityVersion` from the response on the handle (readable as `job.entity_version`). The next `amend_outcome()` on that same handle sends it as `expectedEntityVersion`, so an amendment that would overwrite a change made by another writer in the meantime raises `OutcomeAmendConflictError` instead of silently winning. Pass `expected_entity_version=` to lock against a version you fetched yourself; use a fresh `JobContext.attach()` handle — which has recorded nothing — for the old last-write-wins behavior.

```python
from revenium_middleware import OutcomeAmendConflictError, get_outcome_history

try:
    job.amend_outcome(reason="Chargeback", outcome_value=0.0)
except OutcomeAmendConflictError as conflict:
    # The conflict reports the version the platform actually holds.
    print(conflict.current_entity_version)          # e.g. 9

    # Look at what the other writer changed, and only re-issue the amendment
    # if it still applies to what is recorded now.
    history = get_outcome_history("sales-lead-8842")
    if still_applies(history[-1]):
        job.amend_outcome(reason="Chargeback, re-checked", outcome_value=0.0,
                          expected_entity_version=conflict.current_entity_version)
```

The handle also records that version, so the retry above works with or without passing `expected_entity_version=` explicitly. `current_entity_version` is `None` when the conflict body carries no version; the version then has to come from a job read (`GET /v2/api/jobs/{agenticJobId}`), which this SDK does not wrap yet, and a retry without it is unlocked (last-write-wins). `get_outcome_history()` rows carry an `amendment_sequence`, not an entity version — history tells you *what* changed, never which version to retry with.

Every outcome call replaces the recorded version with the one its response reports, including clearing it when a response carries none, so a completed call never leaves a token behind that the platform has already moved past.

- **Omitting `reason`:** an API-key caller may leave `reason` out and the platform records an automated correction reason derived from the source. A session caller must supply one; a blank string is rejected client-side either way.
- **`reason` vs `outcome_reason`:** `reason` is the amendment's own audit justification (why the record changed); `outcome_reason` is the business explanation of why the job failed or was cancelled. Use `outcome_reason` for failure explanations rather than burying them in `metadata` — it is a first-class field on the outcome and is returned on every `get_outcome_history()` row.
- **Clearing `outcome_reason`:** omit the argument to leave the stored value untouched; pass an empty string (`outcome_reason=""`) to clear it.
- **`metrics`:** both `report_outcome()` and `amend_outcome()` accept a `metrics` argument for recording the measurable facts behind an outcome.

### Recording Metric Facts

Beyond the single `outcome_value`, a job can carry the measurable facts its job type declares — `quality_rate` and its siblings — either with the outcome or later, once they are measurable.

```python
from revenium_middleware import JobContext

with JobContext("claim-8842", type="claims_triage") as job:
    ...
    job.report_outcome(
        execution_status="SUCCESS",
        outcome_type="CONVERTED",
        metrics=[
            {"key": "quality_rate", "value": 0.93, "provenance": "MEASURED"},
            {"key": "cases_closed", "value": 12},
        ],
    )

# Two days later a human grades a sample of that same job's output.
handle = JobContext.attach("claim-8842")
handle.append_outcome_metrics([
    {"key": "quality_rate", "value": 0.87, "provenance": "ATTESTED",
     "reason": "graded sample of 200 cases"},
])
handle.close()
```

- **Declare the metric first:** a fact only lands if the job type's economics contract declares that key as a `PER_JOB` metric; an undeclared key is rejected with a 400. `quality_rate` is a rate and the platform range-checks it to 0..1.
- **Entry shape:** `key` and `value` are required; `provenance` (`MEASURED` | `SELF_REPORTED` | `DERIVED` | `ATTESTED`), `recordedBy`, `source`, `reason` and `recordedAt` are optional. Entries are sent exactly as you write them, so the fields you omit take the platform's defaults (`SELF_REPORTED`, the calling principal, `api`) instead of being guessed by the SDK. A missing `key` or `value` — or no entries at all on `append_outcome_metrics()` — raises `ValueError` before any HTTP request, on `JobContext` and `AgenticOutcomeClient` alike.
- **Append-only:** facts accumulate; the SDK never dedupes or replaces one, because the platform owns fact identity. `metrics=` on `amend_outcome()` appends as part of the amendment.
- **Not part of outcome history:** `get_outcome_history()` returns the outcome revisions only — appended facts do not appear in those rows.
- **Retries:** an append is retried only on `429`, which proves the platform rejected the request before recording anything. A `502`/`503`/`504` is raised instead of retried: the facts may already be recorded, and a second append is a second fact, so the decision to resend is yours (check the recorded facts first).
- **Locking is unaffected:** appending facts does not change the job's `entityVersion`, so the handle keeps the version it recorded and a following `amend_outcome()` still locks against it. (`report_outcome()` and `amend_outcome()` clear the recorded version when their response carries none, because those calls advance it; an append does not.)
- **Why it matters:** AI Alerts evaluate `QUALITY_RATE` from these facts, so a job whose integration emits none is invisible to those rules.

### Outcome Exceptions

All outcome exceptions are importable from `revenium_middleware` and share the `OutcomeReportingError` base, so `except OutcomeReportingError:` catches the whole family:

| Exception | Raised when | What to do |
|-----------|-------------|------------|
| `OutcomeReportingError` | Base class — configuration failures (no API key available, unresolvable `team_id`) | Fix the key / team configuration |
| `OutcomeAlreadyReportedError` | Re-reporting a job that already has an outcome (backend 409) | Amend with `amend_outcome()` instead; the exception carries `reported_at` and `amendment_count` |
| `OutcomeNotReportedError` | Amending a job that has no outcome yet (backend 422) | Call `report_outcome()` first |
| `OutcomeAmendConflictError` | A concurrent amendment changed the outcome (backend 409, optimistic lock) | Re-check the outcome against `get_outcome_history()`, then retry with `expected_entity_version=conflict.current_entity_version` — the SDK does not auto-retry |

### Low-Level Client

For manual control over every metric (one `emit_completion` per LLM call, one `emit_tool_event` per tool/step), use `AgenticOutcomeClient` directly:

```python
from revenium_middleware.agentic_outcomes import AgenticOutcomeClient, AgenticOutcomeSettings

settings = AgenticOutcomeSettings(api_key="rev_sk_...")
client = AgenticOutcomeClient(settings)

client.emit_completion(...)                # one per LLM call
client.emit_tool_event(...)                # one per tool / step
client.report_outcome(job_id, {...})       # close the job with a terminal outcome
client.append_outcome_metrics(job_id, [...])  # append declared per-job facts later
client.close()
```

The job is created implicitly by the first metric ingested for `agenticJobId`. Call `client.create_job(job_id)` explicitly if you need to record an agent run before emitting any metrics; it returns the created job resource merged over the fields you supplied, including the `entityVersion` an outcome amendment sends back as `expectedEntityVersion`.

See [`examples/agentic_outcomes/`](examples/agentic_outcomes/) for runnable demos (sales / coding / support) with configurable failure rates and outcome distributions.

**API reference:** [docs.revenium.io](https://docs.revenium.io) · per-endpoint reference at [revenium.readme.io/reference/meter_ai_completion](https://revenium.readme.io/reference/meter_ai_completion).

---

## Idempotency

Every metering POST from the provider middleware automatically includes an `Idempotency-Key` header. If the Revenium API receives the same key with the same body within 24 hours, it returns the cached response instead of double-billing — making metering submissions safe to retry.

### Default

A fresh UUID v4 is generated automatically for every metering call. No action required.

### Override

Use the `idempotency_key` context manager to tie metering to a business-level identifier so the same logical operation never double-meters across retries:

```python
from revenium_middleware import idempotency_key

with idempotency_key(f"order-{order_id}"):
    response = openai.chat.completions.create(...)
```

The context manager is backed by `contextvars`, so it scopes correctly across threads and asyncio tasks.

### Backend behavior

| Scenario | Backend response |
| -- | -- |
| First call with key K and body B | Executes, caches for 24h |
| Retry with same K and same B | Returns cached response (no double-bill) |
| Same K with different B | `409 idempotency_key_mismatch` |
| Concurrent in-flight with same K | `409 idempotency_key_in_progress` + `Retry-After: 1` |
| Malformed key | `400 invalid_idempotency_key` |

See [docs.revenium.io/integrations/idempotency](https://docs.revenium.io/integrations/idempotency) for full backend semantics.

### Key format

`Idempotency-Key` must be 1–255 printable ASCII characters. UUID v4 is the recommended format and what the SDK generates by default.

---

## Webhook Signature Verification

Revenium signs every outbound webhook with HMAC-SHA256 when a signing secret is configured. The SDK ships a verification helper so your handler can validate signatures without writing crypto.

Two headers arrive on every signed delivery:

| Header | Value |
| -- | -- |
| `X-Revenium-Signature-256` | `sha256=<hex>`. During a 24h rotation overlap: `sha256=A, sha256=B`. |
| `X-Revenium-Webhook-Timestamp` | Unix seconds at signing time. |

### FastAPI example

```python
import os

from fastapi import FastAPI, Header, HTTPException, Request

from revenium_middleware.webhooks import verify_signature

app = FastAPI()
SIGNING_SECRETS = [os.environ["REVENIUM_WEBHOOK_SECRET"]]


@app.post("/webhooks/revenium")
async def receive(
    request: Request,
    x_revenium_signature_256: str = Header(...),
    x_revenium_webhook_timestamp: str = Header(...),
):
    body = await request.body()
    if not verify_signature(
        payload=body,
        signature_header=x_revenium_signature_256,
        timestamp_header=x_revenium_webhook_timestamp,
        secrets=SIGNING_SECRETS,
    ):
        raise HTTPException(status_code=401, detail="Invalid signature")

    # ... process the event
    return {"ok": True}
```

### Secret rotation

When you rotate a signing secret in the Revenium dashboard with the default 24-hour overlap, both the old and new secrets are active simultaneously and every webhook is signed with both. Supply both values in `SIGNING_SECRETS` during the overlap window; remove the old one once it expires.

### Webhooks without a signing secret

Webhook deliveries without a configured signing secret arrive without HMAC headers. If your endpoint receives both signed and unsigned traffic, branch on header presence: treat missing headers as legacy unsigned mode and missing-signature-on-signed-only endpoints as an authentication failure.

---

## Provider Usage Guides

### OpenAI

Supports chat completions, streaming, embeddings, function calling, and vision/multimodal.

```python
from dotenv import load_dotenv
load_dotenv()

import openai
import revenium_middleware.openai  # Auto-initializes

client = openai.OpenAI()

# Basic chat completion
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Hello!"}],
    usage_metadata={
        "organizationName": "AcmeCorp",
        "productName": "customer-chatbot",
        "trace_id": "session-123",
        "task_type": "chat"
    }
)

# Streaming
stream = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Tell me a story"}],
    stream=True
)
for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)

# Embeddings
embedding = client.embeddings.create(
    model="text-embedding-3-small",
    input="The quick brown fox"
)
```

#### Azure OpenAI

The middleware automatically detects Azure OpenAI when using `AzureOpenAI()` and resolves deployment names to standard model names for accurate pricing.

```python
from openai import AzureOpenAI
import revenium_middleware.openai

client = AzureOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version="2024-02-01"
)

response = client.chat.completions.create(
    model="my-gpt4-deployment",  # Azure deployment name
    messages=[{"role": "user", "content": "Hello!"}]
)
# Model name automatically resolved for pricing
```

**Azure environment variables:**
- `AZURE_OPENAI_ENDPOINT` - Your Azure OpenAI endpoint
- `AZURE_OPENAI_API_KEY` - Your Azure OpenAI API key
- `AZURE_OPENAI_DEPLOYMENT` - Default deployment name

**Examples:** `examples/openai/` - `openai_basic.py`, `openai_streaming.py`, `azure_basic.py`, `azure_streaming.py`

---

### Anthropic

Supports messages, streaming, vision/multimodal, and AWS Bedrock integration.

```python
from dotenv import load_dotenv
load_dotenv()

import anthropic
import revenium_middleware.anthropic  # Auto-initializes

client = anthropic.Anthropic()

# Basic message
message = client.messages.create(
    model="claude-opus-4-7",
    max_tokens=100,
    messages=[{"role": "user", "content": "Hello!"}],
    usage_metadata={
        "organizationName": "AcmeCorp",
        "productName": "support-bot",
        "trace_id": "session-456"
    }
)

# Streaming
with client.messages.stream(
    model="claude-opus-4-7",
    max_tokens=200,
    messages=[{"role": "user", "content": "Tell me a story"}],
    usage_metadata={"task_type": "creative"}
) as stream:
    for text in stream.text_stream:
        print(text, end="", flush=True)
```

Async streaming is metered the same way, with the same metadata:

```python
import asyncio
import anthropic
import revenium_middleware.anthropic

client = anthropic.AsyncAnthropic()

async def main():
    async with client.messages.stream(
        model="claude-opus-4-7",
        max_tokens=200,
        messages=[{"role": "user", "content": "Tell me a story"}],
        usage_metadata={"task_type": "creative"}
    ) as stream:
        async for text in stream.text_stream:
            print(text, end="", flush=True)
        # await stream.get_final_message() works too; either way the
        # completion is metered once when the block exits.

asyncio.run(main())
```

**Note:** The middleware wraps the `messages.create` and `messages.stream` endpoints, sync and async alike (including `create(stream=True)`). Other Anthropic SDK features work normally but aren't metered.

#### AWS Bedrock

The middleware provides complete AWS Bedrock integration with automatic detection.

```python
import anthropic
import revenium_middleware.anthropic

# Bedrock is automatically detected when AWS credentials are available
# and base_url contains 'amazonaws.com'
client = anthropic.AnthropicBedrock(
    aws_region="us-east-1"
)

message = client.messages.create(
    model="claude-opus-4-7",
    max_tokens=100,
    messages=[{"role": "user", "content": "Hello from Bedrock!"}]
)
```

**Provider detection** automatically routes between Bedrock and direct Anthropic API based on:
- AWS credentials availability (`aws configure`, IAM roles, environment variables)
- Base URL detection (when `base_url` contains `amazonaws.com`)
- Defaults to direct Anthropic API - Bedrock only used when explicitly configured

**Bedrock environment variables:**

| Variable | Description | Default |
|----------|-------------|---------|
| `AWS_REGION` | AWS region for Bedrock | `us-east-1` |
| `REVENIUM_BEDROCK_DISABLE` | Set to `1` to disable Bedrock support (Bedrock detection only - Foundry detection is unaffected) | Not set |

**AWS authentication** uses the standard credential chain: environment variables, `~/.aws/credentials`, IAM roles, AWS SSO. Required permissions: `bedrock:InvokeModel` and `bedrock:InvokeModelWithResponseStream`.

**Supported Bedrock models:**

| Anthropic Model | Bedrock Model ID |
|----------------|------------------|
| `claude-opus-4-7` | `anthropic.claude-opus-4-7` |
| `us.claude-opus-4-7` | `us.anthropic.claude-opus-4-7` |
| `eu.claude-opus-4-7` | `eu.anthropic.claude-opus-4-7` |
| `au.claude-opus-4-7` | `au.anthropic.claude-opus-4-7` |
| `global.claude-opus-4-7` | `global.anthropic.claude-opus-4-7` |
| `claude-3-opus-20240229` | `anthropic.claude-3-opus-20240229-v1:0` |
| `claude-3-sonnet-20240229` | `anthropic.claude-3-sonnet-20240229-v1:0` |
| `claude-3-haiku-20240307` | `us.anthropic.claude-3-5-haiku-20241022-v1:0` |
| `claude-3-5-sonnet-20240620` | `anthropic.claude-3-5-sonnet-20240620-v1:0` |
| `claude-3-5-sonnet-20241022` | `anthropic.claude-3-5-sonnet-20241022-v2:0` |
| `claude-3-5-haiku-20241022` | `anthropic.claude-3-5-haiku-20241022-v1:0` |

For other models, the middleware uses the format `anthropic.{model_name}`.

#### Microsoft Foundry

Claude served through Microsoft Foundry is metered by the same patched endpoints as the
direct Anthropic API - the Anthropic SDK's Foundry clients need no extra setup.

```python
import anthropic
import revenium_middleware.anthropic

# Foundry is detected from the client class, so a custom base_url is fine too
client = anthropic.AnthropicFoundry(
    resource="your-resource",  # or ANTHROPIC_FOUNDRY_RESOURCE
)

message = client.messages.create(
    model="claude-opus-4-7",
    max_tokens=100,
    messages=[{"role": "user", "content": "Hello from Foundry!"}]
)
```

Foundry usage is reported with provider `Foundry` and model source `ANTHROPIC`, so the spend
is separated from direct-Anthropic totals while still priced against the Anthropic rate card
that Foundry bills at. `AsyncAnthropicFoundry` is metered the same way, as is
`client.messages.stream()`.

**Examples:** `examples/anthropic/` - `anthropic-basic.py`, `anthropic-streaming.py`, `anthropic-bedrock.py`, `anthropic-advanced.py`

---

### Google AI (Gemini / Vertex AI)

Supports chat completions, streaming, embeddings, image generation (Imagen), video generation, and vision/multimodal. Choose between Google AI SDK (simple API key setup) or Vertex AI SDK (production-grade with full token counting).

```bash
# Google AI SDK only (Gemini Developer API)
pip install "revenium-python-sdk[google-genai]"

# Vertex AI SDK only (recommended for production)
pip install "revenium-python-sdk[google-vertex]"
```

#### Google AI SDK

```python
from dotenv import load_dotenv
load_dotenv()

import revenium_middleware.google
from google import genai

client = genai.Client()
response = client.models.generate_content(
    model="gemini-2.0-flash-001",
    contents="Hello! Introduce yourself in one sentence.",
    usage_metadata={
        "organizationName": "AcmeCorp",
        "task_type": "chat"
    }
)
print(response.text)
```

#### Vertex AI SDK

```python
from dotenv import load_dotenv
load_dotenv()

import revenium_middleware.google
import vertexai
from vertexai.generative_models import GenerativeModel

vertexai.init(project="your-gcp-project", location="us-central1")
model = GenerativeModel("gemini-2.0-flash-001")
response = model.generate_content("Hello!")
print(response.text)
```

**Which SDK should I choose?**

| Use Case | Recommended SDK | Why |
|----------|----------------|-----|
| Quick prototyping | Google AI SDK | Simple API key setup |
| Production applications | Vertex AI SDK | Full token counting, enterprise features |
| Embeddings-heavy workloads | Vertex AI SDK | Complete token tracking for embeddings |
| Enterprise/GCP environments | Vertex AI SDK | Advanced Google Cloud integration |

**Note:** Google AI SDK embeddings don't return token counts due to API limitations, but requests are still tracked.

**Google AI environment variables:**
- `GOOGLE_API_KEY` - For Google AI SDK
- `GOOGLE_CLOUD_PROJECT` - For Vertex AI SDK
- `GOOGLE_CLOUD_LOCATION` - Vertex AI region (default: `us-central1`)

**For Vertex AI**, authenticate with: `gcloud auth application-default login`

**Examples:** `examples/google/` - `getting_started_google_ai.py`, `getting_started_vertex_ai.py`, `simple_streaming_test.py`, `simple_embeddings_test.py`

---

### Ollama

Supports chat completions, text generation, embeddings, and streaming. Works with any Ollama model.

```python
from dotenv import load_dotenv
load_dotenv()

import ollama
import revenium_middleware.ollama  # Auto-initializes

# Chat completion
response = ollama.chat(
    model='qwen2.5:0.5b',
    messages=[{'role': 'user', 'content': 'Why is the sky blue?'}],
    usage_metadata={
        "organizationName": "AcmeCorp",
        "task_type": "chat"
    }
)
print(response['message']['content'])

# Streaming
for chunk in ollama.chat(
    model='qwen2.5:0.5b',
    messages=[{'role': 'user', 'content': 'Tell me a story'}],
    stream=True
):
    print(chunk['message']['content'], end='', flush=True)

# Text generation
response = ollama.generate(model='qwen2.5:0.5b', prompt='Once upon a time')

# Embeddings (single and batch)
response = ollama.embed(model='nomic-embed-text', input='Hello world')
response = ollama.embed(model='nomic-embed-text', input=['Text 1', 'Text 2', 'Text 3'])
```

**Supported endpoints:** `ollama.chat()`, `ollama.generate()`, `ollama.embed()`

**OpenAI compatibility mode:** You can also use Ollama with the OpenAI SDK:

```python
import openai
import revenium_middleware.openai

openai.api_key = 'ollama'
openai.base_url = 'http://localhost:11434/v1/'

response = openai.chat.completions.create(
    model="gemma2:2b",
    messages=[{"role": "user", "content": "Hello!"}],
    usage_metadata={"organizationName": "AcmeCorp"}
)
```

**Prerequisites:** Ensure Ollama is running (`ollama serve`) before making API calls.

**Examples:** `examples/ollama/` - `getting_started.py`, `example_streaming.py`, `example_metadata.py`, `embeddings_example.py`

---

### LiteLLM

Supports all LLM providers available through LiteLLM with two integration patterns: client-side middleware and server-side proxy callbacks.

#### Client Mode

```python
from dotenv import load_dotenv
load_dotenv()

import revenium_middleware.litellm.client.middleware  # Auto-initializes
import litellm
import os

litellm.api_base = os.getenv("LITELLM_PROXY_URL")
litellm.api_key = os.getenv("LITELLM_API_KEY")

response = litellm.completion(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Hello!"}],
    usage_metadata={
        "organizationName": "AcmeCorp",
        "task_type": "chat"
    }
)
```

#### Proxy Mode

`ReveniumGuardrail` is the LiteLLM proxy integration. It is a LiteLLM
`CustomGuardrail` that **enforces the caller's budget before** the proxied call and
**meters usage after** it — successes, failures and streamed responses alike.

```bash
pip install "revenium-python-sdk[litellm-proxy]"   # requires Python 3.10+
```

```yaml
guardrails:
  - guardrail_name: "revenium"
    litellm_params:
      guardrail: revenium_middleware.litellm.proxy.guardrail.ReveniumGuardrail
      mode:
        - "pre_call"    # budget enforcement
        - "post_call"   # usage metering
      default_on: true
```

`guardrails` is a **top-level** key, not a member of `litellm_settings`. Nested
under `litellm_settings` it reaches LiteLLM's legacy v1 guardrail loader, which
expects a different shape and exits the proxy at startup with
`GuardrailItem() argument after ** must be a mapping, not str`.

`mode` must be a **list** to enable both hooks; a single string restricts the
guardrail to that one event type. `pre_call` alone enforces without metering;
`post_call` alone meters without enforcing.

**Budget enforcement** reuses the SDK's own circuit breaker, so a proxy enforces
exactly what every other Revenium integration enforces — including department
(org-unit) budgets. It is opt-in via `REVENIUM_CIRCUIT_BREAKER_ENABLED=true`; see
[Cost Controls](#cost-controls). A blocked call never reaches the provider and the
caller receives HTTP 429:

```json
{"error": {"message": "Request blocked by Revenium enforcement rule: Team Budget",
           "type": "budget_exceeded", "guardrail": "revenium", "model": "gpt-4o",
           "budgets": [{"name": "Team Budget", "ruleId": 7, "threshold": 10.0,
                        "currentValue": 11.5, "resetsAt": "2026-10-01T00:00:00Z"}]}}
```

Enforcement **fails open**: if the enforcement path is unreachable or misbehaves,
the call proceeds. Metering is likewise non-disruptive — nothing in the post-call
path can turn a successful LLM call into an error for the client.

**Attribution** travels as `x-revenium-*` request headers (subscriber, organization,
product, trace, task type, agent, subscription, quality score, and `x-revenium-effort`
for reasoning effort), with the calling virtual key's metadata as the fallback —
`revenium_user_id`, `revenium_organization_name`, `revenium_key_name`, and
`revenium_agentic_job_*`. Agentic job tags (`x-revenium-agentic-job-id`, `-name`,
`-type`, `-version`) ride along for cost/ROI correlation; the job id is required for
the others to be recorded.

##### Counting a Claude Code call once (shared call id)

If you run Claude Code through your proxy **and** point Claude Code's own usage
reporting at Revenium, every call is recorded twice: Claude Code files it under an
identifier of its own making, the proxy files it under another, and the two can
never match. `ReveniumGuardrail` makes them match, and does so by default.

The guardrail mints one identifier per proxied `/v1/messages` request, returns it to
the client as the `request-id` and `x-revenium-transaction-id` response headers, and
reports the same value as the call's transaction id. Claude Code copies `request-id`
onto its own record, Revenium's duplicate check sees two records with one identifier,
and one call becomes one record.

**It is on by default.** To opt out and keep LiteLLM's own response id as the
transaction id, with no `request-id` header added:

```bash
export REVENIUM_LITELLM_SHARED_CALL_ID=false
```

New calls go straight back to two records when you do; records already merged stay
merged. Four things have to be true for it to work:

* **LiteLLM 1.93.0 or newer.** That is the floor of the `litellm-proxy` extra, which
  is what installs the guardrail. On an older LiteLLM the guardrail logs one warning
  at startup and behaves exactly as it does when opted out: no header is added,
  the provider's response id is reported, and you get two records rather than none.
* **`mode` includes `pre_call`.** The identifier is minted in the pre-call hook, and
  LiteLLM runs that hook only when the configured mode asks for it. A
  `mode: ["post_call"]` proxy mints nothing, and the guardrail says so at startup
  with one warning naming this variable. Add `pre_call` to the mode, or set the
  variable to `false` if the proxy only meters.
* **The Anthropic messages route.** Other routes are untouched, so the provider's own
  `request-id` on the pass-through route is never overwritten.
* **Both sides report to the same Revenium team.** The duplicate check is
  team-scoped.

**Point Claude Code at the proxy root, not at `/anthropic`.** Set `ANTHROPIC_BASE_URL`
to the proxy's own base URL so Claude Code calls `<proxy>/v1/messages`:

```bash
export ANTHROPIC_BASE_URL=https://proxy.example.com
```

LiteLLM also offers an Anthropic pass-through at `<proxy>/anthropic/v1/messages`, and
its own API reference recommends that route over `/v1/messages`. The shared identifier
is minted only on `/v1/messages`. A proxy serving the pass-through route hands the
provider's response straight back, so nothing is minted there, no `request-id` of ours
is returned, and the counter described below stays silent on that route by design.
The flag will appear to be on and the calls will keep being counted twice, with
nothing in the proxy log to say why. If your Claude Code base URL ends in
`/anthropic`, drop that suffix.

Only the guardrail mints. A proxy still on the deprecated callback alone gets no
identifier and keeps reporting two records, which is one more reason to migrate.

One caution for a proxy running **both** the guardrail and the deprecated callback
without `default_on: true`. That configuration meters every call twice already, and
this flag hides the symptom rather than fixing it: both rows now carry the same
identifier and Revenium's duplicate check keeps one. The configuration is still
wrong. Delete the `litellm_settings.callbacks` entry.

Also do not register `ReveniumGuardrail` in `litellm_settings.callbacks` as well as
in the `guardrails` block. LiteLLM keys registered callbacks on the class name plus
its simple attributes, and the two instances differ, so both are registered and every
logging hook runs twice.

If the flag is on and an Anthropic messages call is metered with no identifier on it,
the guardrail counts it and logs a warning at most once a minute with the running
total, so a mint that quietly stopped shows up in the proxy log rather than as a
return of double counting. Your other routes never carry an identifier and are never
counted or warned about, so a proxy that also serves chat completions or embeddings
stays quiet.

##### Migrating from the callback

`revenium_middleware.litellm.proxy.middleware.MiddlewareHandler` — the
`litellm_settings.callbacks` entry `proxy_handler_instance` — is **deprecated**. It
meters but never enforces a budget. It keeps working in this release and emits a
`DeprecationWarning` (and a log line) when the proxy builds it.

To migrate, delete the callbacks entry and add the `guardrails` block above:

```diff
 litellm_settings:
-  callbacks: ["revenium_middleware.litellm.proxy.middleware.proxy_handler_instance"]
+
+guardrails:
+  - guardrail_name: "revenium"
+    litellm_params:
+      guardrail: revenium_middleware.litellm.proxy.guardrail.ReveniumGuardrail
+      mode: ["pre_call", "post_call"]
+      default_on: true
```

Nothing else changes: the same headers, the same metered fields. Metered rows
record `middleware_source: "GUARDRAIL"` instead of `"PROXY"`.

Leaving both enabled would meter every call twice. As a safety net for a proxy
mid-migration, when the guardrail is configured to run on every request
(`default_on: true` with `post_call` among its modes) it claims metering ownership
and the deprecated callback stops submitting rows, logging once to say so. That net
does **not** apply to a guardrail without `default_on`, without `post_call`, or
configured with a per-tag `Mode` (which selects hooks per request): such a guardrail
may not run on a given request, and suppressing the callback could drop metering
entirely. A per-tag configuration logs, at info level, that it is not claiming
ownership. Delete the callbacks entry rather than relying on the net.

##### Client API and guardrails

`CustomGuardrail`'s lifecycle hooks cannot be hosted by LiteLLM's client API.
Verified against **litellm 1.100.1**: `async_pre_call_hook`,
`async_post_call_success_hook` and `async_post_call_failure_hook` are dispatched
only from `litellm/proxy/utils.py` (`ProxyLogging`) and
`litellm/proxy/common_request_processing.py`. The client path
(`litellm_core_utils/litellm_logging.py`) dispatches only the `CustomLogger`
logging events, so a `CustomGuardrail` added to `litellm.callbacks` without the
proxy running would be a logger with no pre-call hook and no ability to block a
call — the enforcement half would silently not exist.

So the client integration keeps its own path, unchanged: use
`revenium_middleware.litellm.client` as documented above. Note that the LiteLLM
client wrapper meters but does not currently run the pre-call circuit breaker —
enforcement in client mode is available today through the OpenAI middleware, and
through this guardrail in proxy mode. Both the guardrail and the client wrapper
already share their metering plumbing (`revenium_middleware._core`: field
extraction, cache-token extraction and `submit_ai_event`), so the guardrail adds no
second copy of it.

#### LiteLLM Decorators

LiteLLM provides additional tracking decorators beyond the standard `@revenium_metadata` and `@revenium_meter`:

| Decorator | Purpose |
|-----------|---------|
| `@track_agent()` | Identify the AI agent |
| `@track_task()` | Classify the type of work |
| `@track_trace()` | Set trace ID for distributed tracing |
| `@track_organization()` | Track multi-tenant organizations |
| `@track_subscription()` | Track subscription-based billing |
| `@track_product()` | Track product-specific usage |
| `@track_subscriber()` | Identify end users |
| `@track_quality()` | Track response quality scores |
| `@track_job()` | Inject agentic job fields for cost/ROI correlation, e.g. `@track_job(job_id="loan-app-12345", type="loan_processing")` |

The tracking decorators above support static values, extraction from function arguments (`name_from_arg`), or extraction from object attributes (`name_from_attr`); `@track_job` supports static values and argument extraction (`job_id_from_arg`, `type_from_arg`) but has no attribute variant.

#### CrewAI Integration

```bash
pip install "revenium-python-sdk[litellm]" crewai
```

Pre-built wrapper for tracking CrewAI agent executions. **Note:** CrewAI requires Python 3.12 or earlier.

**Job outcome tracking:** pass the `agentic_job_*` kwargs to tie every LLM call in the crew to one agentic job, then report (or later amend) the job's business outcome. Requires a write-scope key (`rev_sk_`) — see [Agentic Outcomes](#agentic-outcomes-outcome-based-metering).

```python
from revenium_middleware.litellm.client.integrations.crewai import ReveniumCrewWrapper

crew = ReveniumCrewWrapper(
    agents=[support_agent],
    tasks=[triage_task],
    organization_id="AcmeCorp",
    subscription_id="82764738",
    product_id="Platinum",
    agentic_job_id="support-ticket-456",
    agentic_job_name="Support Ticket Triage",
    agentic_job_type="customer_support",
    agentic_job_version="2.0",
)
result = crew.kickoff()

crew.report_job_outcome(
    execution_status="SUCCESS",
    outcome_type="DEFLECTED",
    outcome_value=25.0,
)

# Later, if the business result changes:
# crew.amend_job_outcome(reason="Ticket reopened and escalated to a human agent",
#                        outcome_type="ESCALATED", outcome_value=0.0)
```

**LiteLLM environment variables:**
- `LITELLM_PROXY_URL` - Your LiteLLM proxy URL
- `LITELLM_API_KEY` - Your LiteLLM proxy API key

**Examples:** `examples/litellm/` - `getting_started.py`, `litellm_proxy_example.py`, `crewai_decorator_example.py`

---

### Perplexity

Supports both the OpenAI SDK (with Perplexity base URL) and the native Perplexity SDK, with streaming support.

#### Using OpenAI SDK

```python
from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI
import revenium_middleware.perplexity  # Auto-patches OpenAI

client = OpenAI(
    api_key=os.getenv("PERPLEXITY_API_KEY"),
    base_url="https://api.perplexity.ai"
)

response = client.chat.completions.create(
    model="sonar",
    messages=[{"role": "user", "content": "What is the capital of France?"}],
    usage_metadata={"organizationName": "AcmeCorp"}
)
```

#### Using Native Perplexity SDK

```python
from perplexity import Perplexity
import revenium_middleware.perplexity  # Auto-patches Perplexity

client = Perplexity(api_key=os.getenv("PERPLEXITY_API_KEY"))

response = client.chat.completions.create(
    model="sonar",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

Both approaches work identically - the middleware automatically detects which SDK you're using.

**Streaming:**

```python
stream = client.chat.completions.create(
    model="sonar-pro",
    messages=[{"role": "user", "content": "Write a poem"}],
    stream=True,
    usage_metadata={"task_type": "creative_writing"}
)
for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

**Examples:** `examples/perplexity/` - `getting_started.py`, `basic.py`, `streaming.py`, `example_decorator.py`

---

### fal.ai

Supports image, video, and audio generation through fal.ai with automatic media type detection.

```python
import revenium_middleware.fal  # Auto-activates
import fal_client

result = fal_client.subscribe(
    "fal-ai/flux/dev",
    arguments={
        "prompt": "A beautiful sunset over mountains",
        "image_size": "landscape_16_9"
    },
    usage_metadata={
        "organizationName": "AcmeCorp",
        "task_type": "image-generation"
    }
)

for image in result.get("images", []):
    print(f"Image URL: {image['url']}")
```

**Supported methods:** `fal_client.run`, `fal_client.subscribe`, `fal_client.stream` (and their async variants: `run_async`, `subscribe_async`, `stream_async`)

**Media type detection:** The middleware automatically detects the type of media being generated (image, video, audio) based on the application name for accurate cost tracking.

**Environment variables:**
- `FAL_KEY` - Your fal.ai API key

---

### LangChain

Callback handler that automatically tracks LLM calls, chains, tools, and agent actions.

```bash
pip install "revenium-python-sdk[langchain]"
```

Wrap any LangChain LLM (or embeddings model) with `wrap()` — the Revenium callback handler is attached for you:

```python
from langchain_openai import ChatOpenAI
from revenium_middleware.openai.langchain import wrap

llm = wrap(
    ChatOpenAI(model="gpt-4o-mini"),
    usage_metadata={
        "trace_id": "session-123",
        "agent": "support_agent",
    },
)
response = llm.invoke("Hello!")
```

**With chains:**

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

prompt = ChatPromptTemplate.from_template("Tell me a joke about {topic}")
chain = prompt | llm | StrOutputParser()
result = chain.invoke({"topic": "programming"})
```

**With agents:**

```python
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

@tool
def get_weather(city: str) -> str:
    """Get the weather for a city."""
    return f"Sunny, 72F in {city}"

agent = create_react_agent(llm, [get_weather])
result = agent.invoke({"messages": [HumanMessage(content="Weather in NYC?")]})
```

**Async support:** the handler is async-native — wrap once and use `ainvoke`/`astream` directly:

```python
llm = wrap(ChatOpenAI(model="gpt-4o-mini"))
response = await llm.ainvoke("Hello!")
```

**Supported providers:** OpenAI, Anthropic, Google, AWS Bedrock, Azure OpenAI, Cohere, HuggingFace, Ollama. Provider is auto-detected from LangChain class name or model name prefix.

**Attaching to an existing LLM:** use `attach_to()` to add tracking in-place, with any of the standard metadata fields (see [Metadata Fields](#metadata-fields)):

```python
from revenium_middleware.openai.langchain import attach_to

attach_to(llm, usage_metadata={
    "organizationName": "my_org",
    "productName": "my_product",
    "subscriber": {"id": "user_123", "email": "user@example.com"},
})
```

Credentials come from the standard environment variables (`REVENIUM_METERING_API_KEY`, `REVENIUM_METERING_BASE_URL`) or `revenium_middleware.configure()`.

---

### Griptape

Metered prompt and embedding drivers for [Griptape](https://github.com/griptape-ai/griptape) applications. Requires Python 3.10+.

```bash
pip install "revenium-python-sdk[griptape,openai]"     # OpenAI
pip install "revenium-python-sdk[griptape,anthropic]"  # Anthropic
pip install "revenium-python-sdk[griptape,ollama]"     # Ollama
pip install "revenium-python-sdk[griptape,litellm,litellm-proxy]"  # 100+ providers via LiteLLM
```

`ReveniumDriver` auto-detects the provider from the model name (`gpt-*` → OpenAI, `claude-*` → Anthropic, `llama`/`mistral`/... → Ollama, anything else → LiteLLM) and wraps the matching Griptape prompt driver with Revenium metering:

```python
import os
from griptape.structures import Agent
from revenium_middleware.griptape import ReveniumDriver

os.environ["REVENIUM_METERING_API_KEY"] = "your_revenium_key"

agent = Agent(prompt_driver=ReveniumDriver(
    model="gpt-4o-mini",
    usage_metadata={"task_type": "demo"},
))
agent.run("Hello!")
```

Force a provider with `force_provider="litellm"`, or wrap an existing driver with `ReveniumDriver(base_driver=...)`.

**Embeddings:**

```python
from revenium_middleware.griptape import ReveniumEmbeddingDriver

driver = ReveniumEmbeddingDriver(model="text-embedding-3-large")
```

**Provider-specific drivers:** for direct control, use `ReveniumOpenAiDriver`, `ReveniumAnthropicDriver`, `ReveniumOllamaDriver`, `ReveniumLiteLLMDriver` or `ReveniumOpenAiEmbeddingDriver` — each subclasses the corresponding Griptape driver and accepts a `usage_metadata` dict (see [Metadata Fields](#metadata-fields)).

**Migrating from `revenium-griptape`:** the standalone package is deprecated — install the `griptape` extra and change `from revenium_griptape import ReveniumDriver` to `from revenium_middleware.griptape import ReveniumDriver`. All driver class names are unchanged. One behaviour difference: the old package called `load_dotenv()` automatically at import time; the SDK never mutates your environment on import, so if you keep credentials in a `.env` file, call `load_dotenv()` yourself before creating a driver.

---

## Metadata Fields

Add business context to any API call by passing a `usage_metadata` dictionary. All fields are optional.

| Field | Description | Use Case |
|-------|-------------|----------|
| `trace_id` | Unique session or conversation identifier | Link multiple API calls together for debugging, session analytics, or distributed tracing |
| `task_type` | Type of AI task being performed | Categorize usage by workload (e.g., `"chat"`, `"code-generation"`, `"doc-summary"`) for cost analysis |
| `subscriber.id` | Unique user identifier | Track individual user consumption for billing, rate limiting, or analytics |
| `subscriber.email` | User email address | Identify users for support, compliance, or usage reports |
| `subscriber.credential.name` | Authentication credential name | Track which API key or service account made the request |
| `subscriber.credential.value` | Authentication credential value | Associate usage with specific credentials for security auditing |
| `organizationName` | Organization or company name | Multi-tenant cost allocation, usage quotas per organization. Auto-creates if not found |
| `subscription_id` | Subscription plan identifier | Track usage against subscription limits, identify plan upgrade opportunities |
| `productName` | Your product or feature name | Attribute AI costs to specific features (e.g., `"customer-chatbot"`, `"email-assistant"`). Auto-creates if not found |
| `agent` | AI agent or bot identifier | Distinguish between multiple AI agents or automation workflows |
| `response_quality_score` | Custom quality rating (0.0-1.0) | Track user satisfaction or automated quality metrics for model performance analysis |
| `effort` | Reasoning effort level requested of the model (free-form string, max 16 chars, `^[A-Za-z0-9_-]+$`) | Report what share of AI spend is high-effort reasoning, and compare cost per effort level |

**Example:**

```python
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Hello!"}],
    usage_metadata={
        "trace_id": "conv-28a7e9d4",
        "task_type": "customer-support",
        "subscriber": {
            "id": "user-1234",
            "email": "user@example.com",
            "credential": {
                "name": "engineering-api-key",
                "value": "sk-1234567890abcdef"
            }
        },
        "organizationName": "AcmeCorp",
        "subscription_id": "pro-plan-Q1",
        "productName": "customer-support-chatbot",
        "agent": "support-agent",
        "response_quality_score": 0.92,
        "effort": "high"
    }
)
```

**Reasoning effort:** `effort` records how hard the model was asked to think, so
high-effort reasoning spend can be separated from the rest. It is a free-form
string, not an enum -- vendor vocabularies differ and drift, so `low`, `medium`,
`high`, `xhigh` and `ultra` are all real values, and a level this SDK release has
never heard of is passed through untouched rather than rejected or rewritten. The
Revenium backend owns validation (at most 16 characters, matching
`^[A-Za-z0-9_-]+$`); a value it rejects fails the metering call visibly instead
of being silently dropped, so it surfaces in the log and in
`get_metering_status()`. It is distinct from the reported reasoning token count,
which measures the tokens actually spent rather than the level requested.

Set it per call in `usage_metadata` on any provider integration. In LiteLLM proxy
mode, where per-call attribution travels as request headers rather than
`usage_metadata`, send `x-revenium-effort` instead.

**Deprecation notice:** The legacy field aliases `organizationId`, `organization_id`, `productId`, and `product_id` are accepted by this SDK only as an input-layer convenience and emit a `DeprecationWarning`. The Revenium backend no longer accepts them — they are translated to `organizationName` / `productName` before the wire call. Migrate to `organization_name` / `organizationName` and `product_name` / `productName` now; the input-layer aliases will be removed in the next major release.

**API Reference:** [Complete metadata field documentation](https://revenium.readme.io/reference/meter_ai_completion)

---

## Trace Visualization & Distributed Tracing

Enhanced observability fields for tracking AI operations across environments, regions, and workflows. Fields can be set via environment variables (static/deployment-level defaults) or passed directly in `usage_metadata` (dynamic/per-request values). Direct values always take precedence.

### Available Fields

| Field | Environment Variable (Fallback) | Description | Use Case |
|-------|----------------------------------|-------------|----------|
| `environment` | `REVENIUM_ENVIRONMENT` (auto-detects: `ENVIRONMENT`, `DEPLOYMENT_ENV`) | Deployment environment | Track usage across `production`, `staging`, `dev` |
| `region` | `REVENIUM_REGION` (auto-detects: `AWS_REGION`, `AZURE_REGION`, `GCP_REGION`) | Cloud region identifier | Multi-region deployment tracking and latency analysis |
| `credential_alias` | `REVENIUM_CREDENTIAL_ALIAS` | Human-readable API key name | Track which credential was used for rotation and auditing |
| `trace_type` | `REVENIUM_TRACE_TYPE` | Workflow category (max 128 chars, alphanumeric/hyphens/underscores) | Group similar workflows (e.g., `"customer-support"`, `"data-analysis"`) |
| `trace_name` | `REVENIUM_TRACE_NAME` | Human-readable trace label (max 256 chars) | Label trace instances (e.g., `"Customer Support Chat"`) |
| `parent_transaction_id` | `REVENIUM_PARENT_TRANSACTION_ID` | Parent transaction ID | Link child operations to parents across microservices |
| `transaction_name` | `REVENIUM_TRANSACTION_NAME` | Human-friendly operation name | Label operations (e.g., `"Generate Response"`, `"Analyze Sentiment"`) |
| `retry_number` | `REVENIUM_RETRY_NUMBER` | Retry attempt number (0 = first attempt) | Track retry attempts for failed operations |
| `ticket_id` | `REVENIUM_TICKET_ID` | External ticket or issue ID (e.g., Jira, Linear) (max 256 chars) | Attribute AI costs to individual tickets or issues |
| `agent_version` | _(none — per call only)_ | Version of the AI agent that produced the call (max 64 chars) | Compare cost across agent releases; not `agentic_job_version`, which versions the job definition |
| `skill_name` | `REVENIUM_SKILL_NAME` | Name of the agent skill that produced the call (max 256 chars) | Attribute AI costs to the skill that generated them |
| `skill_source` | `REVENIUM_SKILL_SOURCE` | Where the skill was loaded from — accepted values: `bundled`, `projectSettings`, `userSettings`, `plugin` (case-sensitive) | Classify skill origin in the shared skill catalog |
| `skill_kind` | `REVENIUM_SKILL_KIND` | Kind of skill invoked — accepted value: `workflow` (omit otherwise) | Distinguish workflow skills in reporting |
| `skill_plugin_name` | `REVENIUM_SKILL_PLUGIN_NAME` | Plugin providing the skill, when `skill_source` is `plugin` (max 256 chars) | Attribute costs to a specific plugin |
| `skill_marketplace_name` | `REVENIUM_SKILL_MARKETPLACE_NAME` | Marketplace the skill or plugin was installed from (max 256 chars) | Track marketplace-sourced skill usage |
| `skill_invocation_trigger` | `REVENIUM_SKILL_INVOCATION_TRIGGER` | What triggered the skill (max 32 chars; common values: `user-slash`, `claude-proactive`, `nested-skill`) | Separate user-invoked from proactive skill usage |

**Note:** `operation_type` (e.g., `CHAT`, `EMBED`, `TOOL_CALL`) and `operation_subtype` (e.g., `function_call`, `streaming`) are automatically detected by the middleware and cannot be overridden.

### Usage

**Static fields via environment variables** (deployment-level defaults):

```bash
# .env file
REVENIUM_ENVIRONMENT=production
REVENIUM_REGION=us-east-1
REVENIUM_CREDENTIAL_ALIAS=prod-openai-key
REVENIUM_TRACE_TYPE=customer-support
```

**Dynamic fields via `usage_metadata`** (per-request values):

```python
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Hello!"}],
    usage_metadata={
        "environment": "production",
        "region": "us-east-1",
        "trace_type": "customer-support",
        "trace_name": "Support Chat Session",
        "transaction_name": "Generate Response",
        "parent_transaction_id": "parent-txn-123",
        "ticket_id": "JIRA-123",
        "agent_version": "1.4.2"
    }
)
```

**Best practice:** Use environment variables for static deployment configuration (`environment`, `region`, `credential_alias`) and pass dynamic values (`trace_name`, `transaction_name`, `organizationName`) directly in `usage_metadata` or via decorators.

### Distributed Tracing Example

```python
import uuid

workflow_id = str(uuid.uuid4())

# Step 1: Parent operation
parent_response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Analyze this document"}],
    usage_metadata={
        "trace_id": "analysis-session-456",
        "transaction_name": "Document Analysis",
        "task_type": "analysis"
    }
)

# Step 2: Child operation linked to parent
child_response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Summarize findings"}],
    usage_metadata={
        "trace_id": "analysis-session-456",
        "parent_transaction_id": parent_response.id,
        "transaction_name": "Summarize Results",
        "task_type": "summarization"
    }
)
```

---

## Decorator Support

### `@revenium_metadata` - Automatic Metadata Injection

Automatically injects metadata into all API calls within a function's scope. Eliminates the need to pass `usage_metadata` to every API call.

```python
from revenium_middleware import revenium_metadata

@revenium_metadata(
    trace_id="session-12345",
    task_type="customer-support",
    organizationName="AcmeCorp",
    environment="production"
)
def handle_customer_query(question: str) -> str:
    # All API calls automatically include the decorator metadata
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": question}]
    )
    return response.choices[0].message.content

answer = handle_customer_query("How do I reset my password?")
```

**Features:**
- **DRY Principle**: Define metadata once, apply to all API calls in the function
- **Composable**: Decorators can be nested - inner decorators inherit and override outer ones
- **API-level override**: `usage_metadata` passed directly to API calls always takes precedence over decorator metadata
- **Async support**: Works with both sync and async functions
- **Thread-safe**: Uses `contextvars` for proper isolation

**Nested decorators (metadata merging):**

```python
@revenium_metadata(organizationName="AcmeCorp", environment="production")
def outer_function():
    # Gets: organizationName, environment
    response1 = client.chat.completions.create(...)

    @revenium_metadata(trace_id="inner-trace", task_type="analysis")
    def inner_function():
        # Gets: organizationName, environment (inherited) + trace_id, task_type (added)
        response2 = client.chat.completions.create(...)
        return response2

    return inner_function()
```

**API-level override:**

```python
@revenium_metadata(organizationName="AcmeCorp", task_type="default")
def mixed_metadata():
    # Uses decorator metadata
    response1 = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Hello"}]
    )

    # API-level metadata overrides decorator's task_type
    response2 = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Hello"}],
        usage_metadata={
            "task_type": "special-override",  # Overrides decorator
            "trace_id": "api-level-trace"     # Adds new field
            # organizationName still inherited from decorator
        }
    )
```

### `@revenium_meter` - Selective Metering

Control which functions are metered when selective metering mode is enabled. This is useful for metering only specific high-value operations while ignoring others.

**Note:** This decorator only has an effect when `REVENIUM_SELECTIVE_METERING=true` is set. By default, all API calls are metered automatically.

```bash
# Enable selective metering
export REVENIUM_SELECTIVE_METERING=true
```

```python
from revenium_middleware import revenium_meter, revenium_metadata

@revenium_meter()
@revenium_metadata(task_type="premium-feature", organizationName="PremiumTier")
def premium_feature(prompt: str) -> str:
    # This WILL be metered (decorated with @revenium_meter)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

def free_feature(prompt: str) -> str:
    # This will NOT be metered (no @revenium_meter decorator)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content
```

**Accepted values for `REVENIUM_SELECTIVE_METERING`:**
- `"true"`, `"1"`, `"yes"`, `"on"` (case-insensitive) - Selective metering enabled
- `"false"`, `"0"`, `"no"`, `"off"`, or unset - All calls metered (default)

**Decorator order matters:** Place `@revenium_meter` before `@revenium_metadata` (outer to inner).

---

## Tool Metering

The `@meter_tool` decorator lets you meter arbitrary tool/function calls (web scrapers, database lookups, API fetchers, image generators, etc.) alongside your automatic LLM API metering.

```python
import os
from revenium_middleware import meter_tool, configure

# Configure the metering client for tool calls
configure(
    metering_url=os.getenv("REVENIUM_METERING_BASE_URL", "https://api.revenium.ai"),
    api_key=os.environ["REVENIUM_METERING_API_KEY"],
)

# Decorate any tool function to automatically meter it
@meter_tool("customer-database", operation="lookup", agent="support-bot")
def lookup_customer(customer_id: str) -> dict:
    """Timing and success/failure are automatically tracked."""
    return {"name": "Jane Smith", "plan": "Enterprise"}

# The decorator reports the tool call to Revenium automatically
result = lookup_customer("CUST-42")
```

**Manual reporting:**

```python
from revenium_middleware import report_tool_call

report_tool_call(
    tool_id="my-tool",
    operation="fetch",
    duration_ms=1234,
    success=True,
    usage_metadata={"records": 42},
)
```

---

## Prompt Capture

Optional capture of prompts and responses for analytics and debugging. **Disabled by default** to protect sensitive data.

### Enable

```bash
export REVENIUM_CAPTURE_PROMPTS=true
```

### What Gets Captured

| Field | Description | Source |
|-------|-------------|--------|
| `system_prompt` | System prompt content | From `system` parameter / system message |
| `input_messages` | User/assistant messages as JSON | From `messages` parameter |
| `output_response` | Assistant's response content | From response content blocks |
| `prompts_truncated` | Truncation flag | Set to `true` if any field exceeded 50,000 characters |

Each field has a maximum length of **50,000 characters**. If exceeded, it's truncated with a `...[TRUNCATED]` marker.

### Example

```python
import os
os.environ["REVENIUM_CAPTURE_PROMPTS"] = "true"

import revenium_middleware.openai
from openai import OpenAI

client = OpenAI()
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"}
    ],
    usage_metadata={"organizationName": "DemoOrg"}
)
# System prompt, input messages, and output response are now captured
```

Prompt capture works with both streaming and non-streaming requests, and with multimodal content (text, images, etc.).

### Security Considerations

- Prompts may contain sensitive user data
- Responses may include confidential information
- Only enable in environments where data capture is appropriate
- Ensure compliance with your data privacy policies
- Use selective metering with `@revenium_meter` to control which calls are captured

---

## Cost Controls / Enforcement

Block outbound provider requests client-side when a Revenium cost control trips. When the circuit breaker is enabled, the middleware polls compiled enforcement rules from the Revenium API in a background daemon thread and raises `BudgetExceededError` **before** the upstream call, preventing spend beyond the configured limit.

> **Terminology note:** The customer-facing entity is called a **cost control**, served by the backend at `/v2/api/ai/cost-controls`. This SDK polls a separate compiled-rules feed at `/v2/api/ai/enforcement-rules/{teamId}` and is unaffected by changes to the CRUD path — no SDK upgrade is required.

Currently wired for the OpenAI provider (other providers land via per-provider follow-on tickets).

### Enable

```bash
pip install 'revenium-python-sdk[openai]'
```

```env
REVENIUM_CIRCUIT_BREAKER_ENABLED=true
REVENIUM_METERING_API_KEY=hak_your_key_here
REVENIUM_TEAM_ID=your_hashed_team_id
REVENIUM_ENFORCEMENT_BASE_URL=https://api.revenium.ai/profitstream  # optional
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `REVENIUM_CIRCUIT_BREAKER_ENABLED` | `false` | Master switch. `true` / `1` / `yes` / `on` to enable. |
| `REVENIUM_BYPASS` | `false` | When `true`, every `check_enforcement` call short-circuits to a no-op. Useful for incident response. |
| `REVENIUM_TEAM_ID` | — | Hashed team ID. Path component on rule fetches; required when the breaker is enabled. |
| `REVENIUM_ENFORCEMENT_BASE_URL` | origin of `REVENIUM_METERING_BASE_URL` | Base URL for the enforcement API. Set when the enforcement API lives behind a context-path. |
| `REVENIUM_CB_POLL_INTERVAL_SECONDS` | `60` | Background poll interval for rule refreshes. |
| `REVENIUM_CB_FAIL_MODE` | `open` | `open` (default) lets calls through when no cache exists; `closed` raises `BudgetExceededError` until rules are loaded. |
| `REVENIUM_CACHE_DIR` | — | When set, the rule cache is mirrored to `<dir>/revenium_enforcement_rules.json` so a restarted process doesn't fail-closed on the very first call. |

### Public API

Enforcement auto-initializes when the OpenAI middleware loads:

```python
import revenium_middleware.openai  # auto-instruments openai
import openai

client = openai.OpenAI()
```

The pre-call check fires before every chat / embeddings / responses call. When the circuit breaker is disabled, it is a no-op. When enabled:

1. A daemon thread (`revenium-enforcement-poll`) starts on first use.
2. It polls `GET {REVENIUM_ENFORCEMENT_BASE_URL}/v2/api/ai/enforcement-rules/{REVENIUM_TEAM_ID}` every `REVENIUM_CB_POLL_INTERVAL_SECONDS` with the `x-api-key` header.
3. Rules are cached in-process (120 s TTL, refresh-on-stale with thundering-herd guard).
4. `204 No Content` is treated as "no rules configured" — the cache is cleared.

### Exception Contract

```python
from revenium_middleware.openai import BudgetExceededError
```

When a tripped rule matches the current request, the middleware raises before the OpenAI call is made. All structured fields are populated when the server provides them:

| Attribute | Type | Description |
|-----------|------|-------------|
| `message` | `str` | Human-readable reason, e.g. `"Request blocked by Revenium enforcement rule: monthly-gpt4-cap"` |
| `rule_name` | `str \| None` | Server-side rule name |
| `current_value` | `float \| None` | Current metric value at the time of the block |
| `threshold` | `float \| None` | Configured limit |
| `resets_at` | `str \| None` | ISO-8601 timestamp the rule next resets |
| `rule_id` | `str \| int \| None` | Server-side rule identifier |

`BudgetExceededError` does **not** inherit from `ReveniumMiddlewareError`, so the OpenAI middleware's `handle_exception_safely` decorator never swallows it — it always reaches your `except` block.

```python
from revenium_middleware.openai import BudgetExceededError
import openai

client = openai.OpenAI()

try:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Summarize the meeting notes"}],
    )
except BudgetExceededError as exc:
    print(f"Cost limit reached: {exc.message}")
    print(f"Rule {exc.rule_name}: {exc.current_value} / {exc.threshold}; resets {exc.resets_at}")
```

### Fail-Open vs Fail-Closed

By default (`REVENIUM_CB_FAIL_MODE=open`) enforcement failures never propagate to user code. If the rule fetch errors (network, 5xx, auth), the previous in-memory cache is preserved and a debug log line is emitted. If there is no cache yet, enforcement behaves as if no rules are configured and the request continues.

Set `REVENIUM_CB_FAIL_MODE=closed` to refuse calls until at least one rule fetch (or `REVENIUM_CACHE_DIR` snapshot) succeeds. Pair with `REVENIUM_CACHE_DIR` so a process restart loads the last-known rules rather than blocking every call until the first poll completes.

### Shadow Mode

Rules with `shadowMode: true` are observe-and-log: they are skipped by `check_enforcement`. Use shadow mode on the server side to audit a rule before flipping it to enforce.

### Inspecting a Rule and Its Roster

Two read-only calls answer "why was this caller blocked, and who else does this rule cover?" without going anywhere near the pre-call path. Both talk to the server directly, neither is cached, and neither reads or writes the cache `check_enforcement` evaluates — so what they report is what the server holds right now.

```python
from revenium_middleware._core import (
    fetch_enforcement_rule,
    fetch_enforcement_rule_roster,
)

rule = fetch_enforcement_rule("mN3xpQz")          # one rule, or None
roster = fetch_enforcement_rule_roster("mN3xpQz")  # who it measures, or None

if roster:
    print(f"{roster['blockedCount']} over the cap, {roster['warnedCount']} warned")
    for row in roster["rows"]:
        print(f"{row['label']}: ${row['spend']} / ${row['limit']} ({row['band']})")
```

`fetch_enforcement_rule_roster` takes `page`, `size`, `search` and `band` (`BLOCKED`, `WARNED`, `UNDER`, `ALL`); the server does the filtering, sorting, banding and paging, and the three band counts always describe the whole roster rather than the page you asked for. Both calls return `None` rather than raising when the team has no such compiled rule, the rule has no reading yet, or the enforcement API cannot be reached — the same fail-open posture as the rest of the circuit breaker. An empty `rule_id` raises `ValueError`.

The background poller is unaffected: it keeps reading the **whole team's** rules every `REVENIUM_CB_POLL_INTERVAL_SECONDS`. That is deliberate. The server computes the department-budget maps team-wide and attaches them to the team-wide read, so a poll narrowed to a single rule would stop receiving them and department budgets would quietly stop blocking anyone. Narrowing is an explicit, opt-in inspection call and never the refresh.

### End-to-End Example

See [`examples/openai/openai_blocking_demo.py`](examples/openai/openai_blocking_demo.py) for a runnable end-to-end demo using a seeded budget rule.

---

## Configuration Reference

### Required Environment Variables

| Variable | Description |
|----------|-------------|
| `REVENIUM_METERING_API_KEY` | Your Revenium API key (starts with `hak_` or `rev_`) |

### Configuring After Import

The metering client is normally built from the environment when
`revenium_middleware` is first imported. If your credentials only become
available later (a secrets-vault bootstrap, framework settings hooks, import
ordering), you don't need to restart: as soon as `REVENIUM_METERING_API_KEY`
appears in the environment, the next metered call picks it up automatically.
You can also configure programmatically at any time:

```python
import revenium_middleware

revenium_middleware.initialize_metering(
    api_key="hak_your_key",                    # defaults to REVENIUM_METERING_API_KEY
    base_url="https://api.revenium.ai",        # defaults to REVENIUM_METERING_BASE_URL
)
```

`initialize_metering()` returns `True` when metering is enabled after the
call; invoke it with no arguments to re-read the environment.

### Delivery Resilience (Store-and-Forward)

Metering events that still fail after the client's own retries (network
outages, 5xx, rate limiting) are not lost: they are held in a bounded
in-memory buffer and replayed automatically in the background every 30
seconds, reusing each event's original `Idempotency-Key` so replays can
never double-bill. Permanent failures (401/403/404/422) are never buffered.
The buffer holds up to 1000 events (oldest evicted first) for at most 24
hours, and is drained on graceful shutdown. Inspect it programmatically:

```python
from revenium_middleware import get_buffer_stats

print(get_buffer_stats())
# {'size': 0, 'max_size': 1000, 'total_buffered': 3, 'total_replayed': 3, ...}
```

### Optional Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `REVENIUM_METERING_BASE_URL` | `https://api.revenium.ai` | Revenium API endpoint |
| `REVENIUM_LOG_LEVEL` | `INFO` | Log level: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL` |
| `REVENIUM_CAPTURE_PROMPTS` | `false` | Enable prompt capture |
| `REVENIUM_SELECTIVE_METERING` | `false` | Only meter `@revenium_meter` decorated functions |
| `REVENIUM_TEAM_ID` | - | Team ID for cost lookups and outcome reporting (JobContext team resolution) |
| `REVENIUM_ENVIRONMENT` | - | Deployment environment (auto-detects from `ENVIRONMENT`, `DEPLOYMENT_ENV`) |
| `REVENIUM_REGION` | - | Cloud region (auto-detects from `AWS_REGION`, `AZURE_REGION`, `GCP_REGION`) |
| `REVENIUM_CREDENTIAL_ALIAS` | - | Human-readable API key name |
| `REVENIUM_TRACE_TYPE` | - | Workflow category identifier |
| `REVENIUM_TRACE_NAME` | - | Human-readable trace label |
| `REVENIUM_PARENT_TRANSACTION_ID` | - | Parent transaction ID for distributed tracing |
| `REVENIUM_TRANSACTION_NAME` | - | Human-friendly operation name |
| `REVENIUM_RETRY_NUMBER` | - | Retry attempt number |
| `REVENIUM_AGENTIC_JOB_ID` | - | Agentic job instance ID attached to all completions in the process (triggers backend job auto-creation) |
| `REVENIUM_AGENTIC_JOB_NAME` | - | Human-readable agentic job name |
| `REVENIUM_AGENTIC_JOB_TYPE` | - | Agentic job type category |
| `REVENIUM_AGENTIC_JOB_VERSION` | - | Agentic job version |
| `REVENIUM_WRITE_API_KEY` | - | Primary write-scope key (`rev_sk_`) for the agentic outcomes API (report/amend/history); falls back to `REVENIUM_OUTCOME_API_KEY` (deprecated), then `REVENIUM_METERING_API_KEY` |
| `REVENIUM_OUTCOME_API_KEY` | - | Deprecated fallback name for the write-scope key; used only when `REVENIUM_WRITE_API_KEY` is unset |
| `REVENIUM_PROFITSTREAM_BASE_URL` | `https://api.revenium.io` | Agentic outcomes API base URL |
| `REVENIUM_BEDROCK_DISABLE` | - | Set to `1` to disable Bedrock auto-detection; Foundry detection is unaffected |
| `REVENIUM_BUFFER_MAX_SIZE` | `1000` | Store-and-forward buffer capacity (oldest events evicted when full) |
| `REVENIUM_BUFFER_FLUSH_INTERVAL` | `30` | Seconds between automatic replay attempts for buffered events |

Per-call `usage_metadata` values take precedence over the `REVENIUM_AGENTIC_JOB_*` environment variables, and the LiteLLM proxy path sources job fields from `x-revenium-*` headers only — these process-level env fallbacks do not apply to proxied traffic.

### Provider-Specific Environment Variables

| Variable | Provider | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | OpenAI | OpenAI API key |
| `AZURE_OPENAI_ENDPOINT` | Azure OpenAI | Azure endpoint URL |
| `AZURE_OPENAI_API_KEY` | Azure OpenAI | Azure API key |
| `AZURE_OPENAI_DEPLOYMENT` | Azure OpenAI | Default deployment name |
| `ANTHROPIC_API_KEY` | Anthropic | Anthropic API key |
| `AWS_REGION` | Bedrock | AWS region for Bedrock (default: `us-east-1`) |
| `GOOGLE_API_KEY` | Google AI | Google AI SDK API key |
| `GOOGLE_CLOUD_PROJECT` | Vertex AI | GCP project ID |
| `GOOGLE_CLOUD_LOCATION` | Vertex AI | GCP location (default: `us-central1`) |
| `PERPLEXITY_API_KEY` | Perplexity | Perplexity API key |
| `FAL_KEY` | fal.ai | fal.ai API key |
| `LITELLM_PROXY_URL` | LiteLLM | LiteLLM proxy URL |
| `LITELLM_API_KEY` | LiteLLM | LiteLLM proxy API key |

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| **Middleware not working** | Verify `REVENIUM_METERING_API_KEY` is set correctly (must start with `hak_` or `rev_`) |
| **No data in dashboard** | Enable debug logging with `REVENIUM_LOG_LEVEL=DEBUG` |
| **Import errors** | Ensure the correct extra is installed (e.g., `pip install revenium-python-sdk[openai]`) |
| **Azure: wrong model name** | Middleware auto-resolves deployment names; check with debug logging |
| **Bedrock: AccessDenied** | Ensure `bedrock:InvokeModel` and `bedrock:InvokeModelWithResponseStream` permissions |
| **Bedrock: requests go to Anthropic** | Verify AWS credentials: `aws sts get-caller-identity` |
| **Google: embeddings show 0 tokens** | Expected with Google AI SDK; use Vertex AI for full token counting |
| **Google: "No module named 'vertexai'"** | Install correct extra: `pip install "revenium-python-sdk[google-vertex]"` |
| **Vertex AI: authentication errors** | Run `gcloud auth application-default login` |
| **Ollama: connection errors** | Ensure Ollama is running: `ollama serve` |
| **LangChain: provider shows "unknown"** | Ensure you're using a supported LangChain LLM class |
| **Streaming errors** | Check provider credentials; middleware auto-falls back gracefully |

**Debug mode:** Set `REVENIUM_LOG_LEVEL=DEBUG` to see detailed provider detection, routing decisions, and metering payloads.

**Force direct Anthropic API (instead of Bedrock):** Set `REVENIUM_BEDROCK_DISABLE=1` to disable Bedrock auto-detection. Foundry detection is unaffected - a Foundry client is still labelled `Foundry`.

**Check initialization status (Anthropic):** Use `revenium_middleware.anthropic.is_initialized()` to verify setup.

---

## Logging

This module uses Python's standard logging system. Control the log level with the `REVENIUM_LOG_LEVEL` environment variable:

```bash
# Enable debug logging
export REVENIUM_LOG_LEVEL=DEBUG

# Or when running your script
REVENIUM_LOG_LEVEL=DEBUG python your_script.py
```

Available log levels:
- `DEBUG`: Detailed debugging information (provider detection, routing decisions, metering payloads)
- `INFO`: General information (default)
- `WARNING`: Warning messages only
- `ERROR`: Error messages only
- `CRITICAL`: Critical error messages only

## Compatibility

- Python 3.8+
- Works with all supported AI provider SDKs (latest versions recommended)
- Thread-safe and production-ready for concurrent applications

## Documentation

For detailed documentation, visit [docs.revenium.io](https://docs.revenium.io)

### Server-Side Cost Controls

Cost controls (spend limits, throttling, alerts) are managed server-side in Revenium, not in this SDK. The SDK reports usage; Revenium evaluates it against your configured cost controls.

The cost-controls API endpoint is `/v2/api/ai/cost-controls`. This Python SDK does not call the endpoint directly — no SDK changes are required to use cost controls. If you manage cost controls via the Revenium API, HTTP client, or `curl`, see [docs.revenium.io](https://docs.revenium.io) for the current API reference.

## Contributing

See [CONTRIBUTING.md](./CONTRIBUTING.md)

## Code of Conduct

See [CODE_OF_CONDUCT.md](./CODE_OF_CONDUCT.md)

## Security

See [SECURITY.md](./SECURITY.md)

## License

This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.

## Support

For issues, feature requests, or contributions:

- **Website**: [www.revenium.ai](https://www.revenium.ai)
- **GitHub Repository**: [revenium/revenium-python-sdk](https://github.com/revenium/revenium-python-sdk)
- **Issues**: [Report bugs or request features](https://github.com/revenium/revenium-python-sdk/issues)
- **Documentation**: [docs.revenium.io](https://docs.revenium.io)
- **Email**: support@revenium.io

---

**Built by Revenium**
